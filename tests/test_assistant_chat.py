import asyncio
import hashlib
import json
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import httpx
import pytest
from fastapi.testclient import TestClient

from app.agent_capabilities import (
    WRITE_CHAPTER,
    PROPOSE_SETTINGS_PATCH,
    PROPOSE_STORY_PLAN,
    agent_capabilities,
    agent_manifest,
    native_agent_role_prompt,
    resolve_agent_dispatch,
)
from app.agent_intent import (
    AssistantIntentDecision,
    AssistantIntentResponse,
)
from app.agent_orchestrator import (
    AssistantAgentOrchestrator,
    _explicit_intent_decision,
)
from app.agent_model import (
    AssistantModelTurn,
    AssistantToolCall,
    ProviderAgentModel,
    MockAgentModel,
    _exact_repetition_findings,
    _mock_settings_patch,
    _repair_likely_accidental_repetitions,
    _unresolved_repetition_findings,
    compose_native_agent_system_prompt,
)
from app.assistant_chat_service import (
    MAX_USER_MESSAGE_CHARS,
    AssistantChatService,
    _author_explicitly_requested_deletion,
)
from app.assistant_chat_schema import (
    AssistantDraftProposal,
    ChapterDraftAuditIssue,
    ChapterDraftAuditResponse,
    ChapterDraftAuditResult,
)
from app.chapter_splitter import split_chapters
from app.config import Settings
from app.db import Database
from app.main import create_app
from app.security import hash_password
from app.story_planning_service import StoryPlanningService


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        app_name="test",
        app_env="test",
        secret_key="test-secret-long-enough",
        data_dir=tmp_path,
        database_path=tmp_path / "test.db",
        cookie_secure=False,
        allow_registration=True,
        max_upload_bytes=1_000_000,
        max_text_chars=1_000_000,
        target_chapter_chars=10_000,
        max_chapter_chars=30_000,
        model_api_key=None,
        model_base_url="https://api.deepseek.com",
        model_name="deepseek-v4-flash",
        model_thinking=False,
        model_reasoning_effort="high",
        model_max_tokens=5_000,
        model_connect_timeout_seconds=1,
        model_read_timeout_seconds=1,
        model_max_retries=0,
        worker_poll_seconds=0.01,
    )


def csrf_from(html: str) -> str:
    match = re.search(r'name="csrf" value="([^"]+)"', html)
    assert match
    return match.group(1)


def test_native_planner_prompt_uses_current_workspace_actions():
    prompt = native_agent_role_prompt("planner")

    assert "settings/" in prompt
    assert "edit" in prompt
    assert "propose_settings_patch" not in prompt


class FixedIntentModel(MockAgentModel):
    def __init__(
        self,
        intent: str,
        *,
        confidence: float = 0.95,
        target_chapter_id: Optional[str] = None,
    ):
        self.intent = intent
        self.confidence = confidence
        self.target_chapter_id = target_chapter_id

    async def classify_intent(self, **kwargs):
        del kwargs
        decision = AssistantIntentDecision(
            intent=self.intent,
            confidence=self.confidence,
            target_chapter_id=self.target_chapter_id,
            reason="测试指定的结构化任务意图",
        )
        return AssistantIntentResponse(
            decision=decision,
            raw_response=decision.model_dump_json(),
            input_tokens=2,
            output_tokens=1,
        )


def run_agent(
    service: AssistantChatService,
    claimed: dict,
    *,
    model: MockAgentModel | None = None,
):
    return asyncio.run(
        AssistantAgentOrchestrator(service).run(
            model=model or MockAgentModel(),
            item=claimed,
            payload=service.build_job_payload(claimed),
            provider_user_id="u_test",
        )
    )


class ScriptedNativeEditModel(MockAgentModel):
    def __init__(self):
        self.turns = 0
        self.seen_messages = []

    async def native_turn(self, **kwargs):
        self.turns += 1
        messages = [dict(item) for item in kwargs["messages"]]
        self.seen_messages.append(messages)
        tool_names = {
            item["function"]["name"] for item in kwargs["tools"]
        }
        if self.turns == 1:
            assert {"glob", "read", "grep", "edit", "compose"} <= (
                tool_names
            )
            assert "write" in tool_names
            return AssistantModelTurn(
                content="",
                reasoning="",
                tool_calls=(
                    AssistantToolCall(
                        id="call-read",
                        name="read",
                        arguments={
                            "path": "book/manuscript/chapters/001.md"
                        },
                        raw_arguments=(
                            '{"path":"book/manuscript/chapters/001.md"}'
                        ),
                    ),
                ),
                finish_reason="tool_calls",
                input_tokens=10,
                output_tokens=2,
            )
        if self.turns == 2:
            tool_message = messages[-1]
            assert tool_message["role"] == "tool"
            result = json.loads(tool_message["content"])["result"]
            return AssistantModelTurn(
                content="",
                reasoning="",
                tool_calls=(
                    AssistantToolCall(
                        id="call-edit",
                        name="edit",
                        arguments={
                            "path": "book/manuscript/chapters/001.md",
                            "old_string": "她立刻明白了一切。",
                            "new_string": "她把信纸翻到背面。",
                            "expected_revision": result["revision"],
                            "rationale": "用动作替换直接总结",
                        },
                    ),
                ),
                finish_reason="tool_calls",
                input_tokens=12,
                output_tokens=3,
            )
        assert messages[-1]["role"] == "tool"
        assert json.loads(messages[-1]["content"])["ok"] is True
        return AssistantModelTurn(
            content="已按你的要求做了局部修改，其他正文保持不变。",
            reasoning="",
            tool_calls=(),
            finish_reason="stop",
            input_tokens=9,
            output_tokens=8,
        )


class ScriptedNativeDraftModel(MockAgentModel):
    def __init__(self):
        self.turns = 0

    async def native_turn(self, **kwargs):
        self.turns += 1
        messages = kwargs["messages"]
        if self.turns == 1:
            return AssistantModelTurn(
                content="",
                reasoning="",
                tool_calls=(
                    AssistantToolCall(
                        id="draft-read",
                        name="read",
                        arguments={
                            "path": "book/manuscript/chapters/001.md"
                        },
                    ),
                ),
                finish_reason="tool_calls",
            )
        if self.turns == 2:
            revision = json.loads(messages[-1]["content"])["result"][
                "revision"
            ]
            return AssistantModelTurn(
                content="",
                reasoning="",
                tool_calls=(
                    AssistantToolCall(
                        id="draft-write",
                        name="compose",
                        arguments={
                            "path": "book/manuscript/chapters/001.md",
                            "instruction": "承接旧信，让林岚决定去灯塔调查。",
                            "expected_revision": revision,
                            "mode": "replace",
                            "target_chars": 1500,
                        },
                    ),
                ),
                finish_reason="tool_calls",
            )
        return AssistantModelTurn(
            content="第一章正文已经写入可撤回工作稿。",
            reasoning="",
            tool_calls=(),
            finish_reason="stop",
        )


class ScriptedNativeCreateThenComposeModel(MockAgentModel):
    def __init__(self):
        self.turns = 0

    async def native_turn(self, **kwargs):
        self.turns += 1
        messages = kwargs["messages"]
        available = {
            item["function"]["name"] for item in kwargs["tools"]
        }
        assert {"create", "compose", "read"} <= available
        if self.turns == 1:
            return AssistantModelTurn(
                content="",
                reasoning="",
                tool_calls=(
                    AssistantToolCall(
                        id="create-next",
                        name="create",
                        arguments={
                            "resource": "chapter",
                            "title": "第二章 灯塔",
                            "outline": "林岚抵达灯塔并发现新的异常。",
                            "key_points": "潮汐表；错误时间亮灯",
                        },
                    ),
                ),
                finish_reason="tool_calls",
            )
        if self.turns == 2:
            created = json.loads(messages[-1]["content"])["result"]
            assert created["path"] == "book/manuscript/chapters/002.md"
            return AssistantModelTurn(
                content="",
                reasoning="",
                tool_calls=(
                    AssistantToolCall(
                        id="read-next",
                        name="read",
                        arguments={"path": created["path"]},
                    ),
                ),
                finish_reason="tool_calls",
            )
        if self.turns == 3:
            reading = json.loads(messages[-1]["content"])["result"]
            return AssistantModelTurn(
                content="",
                reasoning="",
                tool_calls=(
                    AssistantToolCall(
                        id="compose-next",
                        name="compose",
                        arguments={
                            "path": reading["path"],
                            "instruction": "承接上一章，让林岚抵达灯塔。",
                            "expected_revision": reading["revision"],
                            "mode": "replace",
                            "target_chars": 1500,
                        },
                    ),
                ),
                finish_reason="tool_calls",
            )
        return AssistantModelTurn(
            content="新章节已创建并写入工作稿。",
            reasoning="",
            tool_calls=(),
            finish_reason="stop",
        )


class ScriptedNativeTaskModel(MockAgentModel):
    def __init__(self):
        self.turns = 0

    async def native_turn(self, **kwargs):
        self.turns += 1
        messages = kwargs["messages"]
        available = {
            item["function"]["name"] for item in kwargs["tools"]
        }
        assert "task" in available
        if self.turns == 1:
            return AssistantModelTurn(
                content="",
                reasoning="",
                tool_calls=(
                    AssistantToolCall(
                        id="task-continuity",
                        name="task",
                        arguments={
                            "kind": "continuity",
                            "objective": "核对第一章与作品核心设定是否矛盾",
                            "paths": [
                                "book/manuscript/chapters/001.md",
                                "book/settings/core.json",
                            ],
                        },
                    ),
                ),
                finish_reason="tool_calls",
                input_tokens=8,
                output_tokens=3,
            )
        task_result = json.loads(messages[-1]["content"])["result"]
        assert task_result["resource_count"] == 2
        assert "未发现直接矛盾" in task_result["report"]
        return AssistantModelTurn(
            content="第一章与当前核心设定没有直接矛盾。",
            reasoning="",
            tool_calls=(),
            finish_reason="stop",
            input_tokens=7,
            output_tokens=5,
        )


class NativeSpecialistModel(MockAgentModel):
    provider = "specialist-provider"
    model = "specialist-model"

    def __init__(self):
        self.calls = []

    async def native_turn(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs["tools"] == []
        prompt = kwargs["messages"][1]["content"]
        assert "book/manuscript/chapters/001.md" in prompt
        assert "book/settings/core.json" in prompt
        assert "conversation-history" not in prompt
        return AssistantModelTurn(
            content=(
                "未发现直接矛盾。book/manuscript/chapters/001.md 中的旧信"
                "符合 book/settings/core.json 的悬疑前提。"
            ),
            reasoning="",
            tool_calls=(),
            finish_reason="stop",
            input_tokens=20,
            output_tokens=12,
        )


class NativeProseModel(MockAgentModel):
    provider = "prose-provider"
    model = "prose-model"

    def __init__(self):
        self.requests = []

    async def native_turn(self, **kwargs):
        self.requests.append(kwargs)
        content = (
            "海雾贴着窗玻璃缓慢下沉。林岚把旧信折好，"
            "在灯塔的位置画了一个圈。\n\n天亮前，她出了门。"
        )
        callback = kwargs.get("on_text_delta")
        if callback is not None:
            midpoint = len(content) // 2
            for delta in (content[:midpoint], content[midpoint:]):
                callback_result = callback(delta)
                if asyncio.iscoroutine(callback_result):
                    await callback_result
        return AssistantModelTurn(
            content=content,
            reasoning="先规划连续性",
            tool_calls=(),
            finish_reason="stop",
            input_tokens=40,
            output_tokens=30,
        )


class RevisingNativeAuditModel(MockAgentModel):
    def __init__(self):
        self.audit_calls = 0
        self.last_context = None

    async def audit_chapter_draft(self, **kwargs):
        self.audit_calls += 1
        self.last_context = kwargs["context"]
        candidate = kwargs["draft"].content
        revised = candidate.replace(
            "天亮前，她出了门。",
            "潮声盖过钟响时，她收紧围巾，朝灯塔走去。",
        )
        result = ChapterDraftAuditResult(
            verdict="revised",
            issues=[
                ChapterDraftAuditIssue(
                    category="specificity",
                    description="结尾使用泛化时间和动作",
                    evidence="天亮前，她出了门。",
                )
            ],
            revised_content=revised,
            summary="只替换了缺少场景锚点的结尾。",
        )
        return ChapterDraftAuditResponse(
            result=result,
            raw_response=result.model_dump_json(),
            input_tokens=5,
            output_tokens=4,
        )


def test_native_agent_prompt_combines_workspace_and_book_instructions():
    prompt = compose_native_agent_system_prompt(
        book_prompt="本书避免解释主题。",
        agent_role="writer",
    )
    assert "不要暴露虚拟路径、revision" in prompt
    assert "book/ 是服务端映射出的作品资源" in prompt
    assert "本书避免解释主题" in prompt


def test_deepseek_native_turn_sends_real_tools_and_normalizes_call(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_read_1",
                                    "type": "function",
                                    "function": {
                                        "name": "read",
                                        "arguments": (
                                            '{"path":"book/settings/core.json"}'
                                        ),
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 21, "completion_tokens": 8},
            },
        )

    async def scenario():
        settings = replace(make_settings(tmp_path), model_api_key="key")
        model = ProviderAgentModel(
            settings, transport=httpx.MockTransport(handler)
        )
        try:
            return await model.native_turn(
                messages=[
                    {"role": "system", "content": "按需使用工具"},
                    {"role": "user", "content": "读取作品概览"},
                ],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "read",
                            "description": "读取资源",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"}
                                },
                                "required": ["path"],
                            },
                        },
                    }
                ],
                provider_user_id="user-1",
                max_tokens=1000,
            )
        finally:
            await model.close()

    turn = asyncio.run(scenario())

    assert seen["payload"]["tools"][0]["function"]["name"] == "read"
    assert seen["payload"]["tool_choice"] == "auto"
    assert turn.finish_reason == "tool_calls"
    assert turn.tool_calls[0].name == "read"
    assert turn.tool_calls[0].arguments == {
        "path": "book/settings/core.json"
    }
    assert turn.input_tokens == 21




def test_deepseek_draft_audit_returns_full_repaired_candidate(
    tmp_path: Path,
):
    seen = {}
    audit = {
        "verdict": "revised",
        "issues": [
            {
                "description": "磁带已经交给周启，随后又回到沈砚口袋",
            }
        ],
        "revised_content": "周启把磁带按在内袋里，沈砚手中只剩纸页。",
        "summary": "修正物品归属矛盾。",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                audit,
                                ensure_ascii=False,
                            )
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 21,
                    "completion_tokens": 13,
                },
            },
        )

    settings = replace(
        make_settings(tmp_path),
        model_api_key="sk-audit-test",  # pragma: allowlist secret
    )
    model = ProviderAgentModel(
        settings,
        transport=httpx.MockTransport(handler),
    )

    async def scenario():
        try:
            return await model.audit_chapter_draft(
                context={
                    "chapter": {
                        "id": "chapter-2",
                        "position": 2,
                        "title": "封锁",
                    }
                },
                question="磁带交给周启后不能再出现在沈砚手中。",
                draft=AssistantDraftProposal(
                    content=(
                        "沈砚把磁带交给周启。"
                        "片刻后，沈砚从内袋掏出磁带。"
                    ),
                    rationale="承接上一章。",
                ),
                observations=[
                    {
                        "tool_name": "read_chapter",
                        "status": "completed",
                        "result": {
                            "chapter_context": {
                                "previous_chapter_excerpt": (
                                    "周启伸手接过磁带。"
                                )
                            }
                        },
                    }
                ],
                provider_user_id="u_test",
            )
        finally:
            await model.close()

    response = asyncio.run(scenario())
    assert response.result.verdict == "revised"
    assert response.result.revised_content == audit["revised_content"]
    assert response.result.issues[0].category == "instruction"
    assert (
        response.result.issues[0].evidence
        == "磁带已经交给周启，随后又回到沈砚口袋"
    )
    assert response.input_tokens == 21
    assert response.output_tokens == 13
    request_text = json.dumps(
        seen["payload"]["messages"],
        ensure_ascii=False,
    )
    assert "落稿前小说编辑" in request_text
    assert "周启伸手接过磁带" in request_text
    assert "沈砚从内袋掏出磁带" in request_text


def test_draft_audit_repairs_likely_accidental_exact_dialogue_repetition(
    tmp_path: Path,
):
    repeated = "总比两个人都困在这里好。"
    candidate = (
        f"“{repeated}”沈砚把磁带递给周启。\n\n"
        f"“{repeated}”她侧身钻进通风井。"
    )
    revised = (
        f"“{repeated}”沈砚把磁带递给周启。\n\n"
        "她侧身钻进通风井。"
    )
    findings = _exact_repetition_findings(candidate)
    assert findings == [
        {
            "kind": "dialogue",
            "text": repeated,
            "occurrences": 2,
            "lines": [1, 3],
            "assessment": "likely_accidental",
            "echo_cues": [],
        }
    ]

    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        requests.append(payload)
        result = {
            "verdict": "pass",
            "issues": [],
            "revised_content": None,
            "summary": "没有发现需要修正的问题。",
        }
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                result,
                                ensure_ascii=False,
                            )
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                },
            },
        )

    settings = replace(
        make_settings(tmp_path),
        model_api_key="sk-audit-repeat",  # pragma: allowlist secret
    )
    model = ProviderAgentModel(
        settings,
        transport=httpx.MockTransport(handler),
    )

    async def scenario():
        try:
            return await model.audit_chapter_draft(
                context={"chapter": {"id": "chapter-3"}},
                question="做最小整章修订。",
                draft=AssistantDraftProposal(
                    content=candidate,
                    rationale="保留原有情节。",
                ),
                observations=[],
                provider_user_id="u_test",
            )
        finally:
            await model.close()

    response = asyncio.run(scenario())
    assert len(requests) == 1
    assert "deterministic_repetition_findings" in json.dumps(
        requests[0]["messages"],
        ensure_ascii=False,
    )
    assert response.result.verdict == "revised"
    assert response.result.revised_content == revised
    assert response.input_tokens == 10
    assert response.output_tokens == 5


def test_exact_repeat_with_explicit_echo_cue_is_preserved():
    repeated = "塔顶不用上去，守住所有出口。"
    content = (
        f"磁带里响起命令：“{repeated}”\n\n"
        "沈砚倒回去再听一遍，确认两段录音一字不差。\n\n"
        f"楼下的对讲机随即响起：“{repeated}”"
    )
    findings = _exact_repetition_findings(content)
    assert len(findings) == 1
    assert findings[0]["assessment"] == "needs_context_review"
    assert "一字不差" in findings[0]["echo_cues"]
    revised, repaired = _repair_likely_accidental_repetitions(
        content,
        findings,
    )
    assert revised == content
    assert repaired == []
    assert (
        _unresolved_repetition_findings(
            content,
            findings,
            summary="",
        )
        == []
    )


def test_near_duplicate_rewrite_is_removed_after_exact_repeat_audit():
    repeated = "总比两个人都困在这里好。"
    content = (
        f"“{repeated}”沈砚把磁带交给周启。\n\n"
        "“总比都困在这里强。”沈砚侧身钻进通风井。"
    )
    finding = {
        "kind": "dialogue",
        "text": repeated,
        "occurrences": 2,
        "lines": [1, 3],
        "assessment": "likely_accidental",
        "echo_cues": [],
    }
    revised, repaired = _repair_likely_accidental_repetitions(
        content,
        [finding],
    )
    assert revised == (
        f"“{repeated}”沈砚把磁带交给周启。\n\n"
        "沈砚侧身钻进通风井。"
    )
    assert repaired == [finding]




def test_structured_intent_dispatch_enforces_scope_and_permissions():
    assert PROPOSE_SETTINGS_PATCH in agent_capabilities("planner")
    assert PROPOSE_STORY_PLAN in agent_capabilities("story_planner")
    assert PROPOSE_SETTINGS_PATCH not in agent_capabilities("advisor")
    assert PROPOSE_SETTINGS_PATCH not in agent_capabilities("writer")
    assert PROPOSE_SETTINGS_PATCH not in agent_capabilities("editor")
    assert WRITE_CHAPTER in agent_capabilities("writer")
    assert WRITE_CHAPTER in agent_capabilities("editor")
    reference = resolve_agent_dispatch(
        requested_role="auto",
        scope_type="reference_chapter",
        intent="draft_prose",
        has_quote=False,
    )
    assert (reference.role, reference.intent, reference.reason) == (
        "researcher",
        "analyze_work",
        "reference_scope",
    )
    revision = resolve_agent_dispatch(
        requested_role="auto",
        scope_type="chapter",
        intent="revise_prose",
        has_quote=True,
    )
    assert (revision.role, revision.goal) == (
        "editor",
        "replace_selected_text",
    )
    whole_chapter_revision = resolve_agent_dispatch(
        requested_role="auto",
        scope_type="chapter",
        intent="revise_prose",
        has_quote=False,
    )
    assert (
        whole_chapter_revision.role,
        whole_chapter_revision.intent,
        whole_chapter_revision.reason,
        whole_chapter_revision.goal,
    ) == (
        "editor",
        "revise_prose",
        "whole_chapter_revision",
        "create_chapter_draft",
    )
    draft = resolve_agent_dispatch(
        requested_role="auto",
        scope_type="chapter",
        intent="draft_prose",
        has_quote=False,
    )
    assert (draft.role, draft.goal) == (
        "writer",
        "create_chapter_draft",
    )
    new_chapter = resolve_agent_dispatch(
        requested_role="auto",
        scope_type="chapter",
        intent="draft_new_chapter",
        has_quote=False,
    )
    assert (
        new_chapter.role,
        new_chapter.intent,
        new_chapter.goal,
    ) == (
        "writer",
        "draft_new_chapter",
        "create_chapter_draft",
    )
    planning = resolve_agent_dispatch(
        requested_role="auto",
        scope_type="project",
        intent="plan_story",
        has_quote=False,
    )
    assert (planning.role, planning.goal) == (
        "story_planner",
        "propose_story_plan",
    )
    analysis = resolve_agent_dispatch(
        requested_role="auto",
        scope_type="chapter",
        intent="analyze_work",
        has_quote=False,
    )
    assert (analysis.role, analysis.goal) == ("analyst", "answer")
    override = resolve_agent_dispatch(
        requested_role="advisor",
        scope_type="chapter",
        intent="draft_prose",
        has_quote=False,
    )
    assert (override.role, override.intent, override.reason) == (
        "advisor",
        "discuss",
        "author_override",
    )
    low_confidence = resolve_agent_dispatch(
        requested_role="auto",
        scope_type="chapter",
        intent="draft_prose",
        has_quote=False,
        confidence=0.4,
    )
    assert (
        low_confidence.role,
        low_confidence.intent,
        low_confidence.reason,
    ) == ("advisor", "discuss", "low_confidence_fallback")
    empty_project_draft = resolve_agent_dispatch(
        requested_role="auto",
        scope_type="project",
        intent="draft_prose",
        has_quote=False,
        settings_ready=False,
    )
    assert (
        empty_project_draft.role,
        empty_project_draft.intent,
        empty_project_draft.goal,
        empty_project_draft.settings_prerequisite,
    ) == (
        "writer",
        "draft_prose",
        "create_chapter_draft",
        False,
    )


def test_explicit_commands_route_locally_but_discussion_stays_ambiguous():
    next_chapter = _explicit_intent_decision(
        question="请直接写下一章，承接现在的封锁现场。",
        scope="novel_chapter",
        has_selected_quote=False,
    )
    assert next_chapter and next_chapter.intent == "draft_new_chapter"

    revision = _explicit_intent_decision(
        question="请只修改本章结尾，其他段落保持不变。",
        scope="novel_chapter",
        has_selected_quote=False,
    )
    assert revision and revision.intent == "revise_prose"

    appended_revision = _explicit_intent_decision(
        question=(
            "请在当前第一章末尾追加一段约100字的环境描写；"
            "保持现有人物和事实不变，直接写入。"
        ),
        scope="novel_chapter",
        has_selected_quote=False,
    )
    assert appended_revision and appended_revision.intent == "revise_prose"

    settings = _explicit_intent_decision(
        question="把林岚的人物设定改为遇事先核对物证。",
        scope="novel_project",
        has_selected_quote=False,
    )
    assert settings and settings.intent == "update_settings"

    read_only = _explicit_intent_decision(
        question="你觉得本章应该怎么修改？先讨论，不要动正文。",
        scope="novel_chapter",
        has_selected_quote=False,
    )
    assert read_only and read_only.intent == "discuss"

    scoped_edit = _explicit_intent_decision(
        question="不要改世界，只把林岚的人物动机写清楚。",
        scope="novel_project",
        has_selected_quote=False,
    )
    assert scoped_edit and scoped_edit.intent == "update_settings"


def test_settings_deletion_requires_a_positive_author_command():
    assert _author_explicitly_requested_deletion("删除林岚的人物卡")
    assert _author_explicitly_requested_deletion(
        "保留林岚，去掉重复的临时人物卡"
    )
    assert not _author_explicitly_requested_deletion(
        "不要删除林岚，只修改她的内在动机"
    )




def test_deepseek_intent_router_returns_structured_intent_only(
    tmp_path: Path,
):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        seen["payload"] = payload
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "intent": "draft_prose",
                                    "workflow": ["draft_prose"],
                                    "confidence": 0.93,
                                    "target_chapter_id": "c1",
                                    "reason": "作者明确要求续写当前章节",
                                },
                                ensure_ascii=False,
                            )
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 18,
                    "completion_tokens": 9,
                },
            },
        )

    settings = replace(
        make_settings(tmp_path),
        model_api_key="sk-intent-router-test",
    )
    model = ProviderAgentModel(
        settings, transport=httpx.MockTransport(handler)
    )

    async def scenario():
        try:
            return await model.classify_intent(
                context={
                    "scope": "novel_chapter",
                    "project": {"id": "p1", "title": "测试小说"},
                    "chapter": {
                        "id": "c1",
                        "position": 1,
                        "title": "第一章",
                    },
                    "current_chapter_excerpt": "不能进入意图分类请求的正文",
                    "dispatch": {
                        "requested_role": "auto",
                        "resolved_role": "pending",
                        "settings_ready": True,
                    },
                },
                history=[],
                question="请继续写这一章。",
                has_selected_quote=False,
                provider_user_id="u_test",
            )
        finally:
            await model.close()

    response = asyncio.run(scenario())
    assert response.decision.intent == "draft_prose"
    assert response.decision.workflow == ["draft_prose"]
    assert response.decision.target_chapter_id == "c1"
    assert response.input_tokens == 18
    body = seen["payload"]
    assert body["response_format"] == {"type": "json_object"}
    serialized_messages = json.dumps(
        body["messages"], ensure_ascii=False
    )
    assert "available_chapters" in serialized_messages
    assert "不能进入意图分类请求的正文" not in serialized_messages


def seed_novel(
    database: Database, tmp_path: Path
) -> tuple[int, str, str, str, str]:
    user_id = database.create_user(
        "chat-writer", hash_password("password-123")
    )
    project_id = "p" * 32
    chapter_id = "c" * 32
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="雾港来信",
        genre="悬疑",
        premise="记者收到一封不可能出现的旧日来信。",
        world_setting="冬季海港。",
        style_guide="克制，以动作承担情绪。",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
    )
    chapter_dir = (
        tmp_path
        / "novels"
        / str(user_id)
        / project_id
        / "chapters"
        / chapter_id
    )
    chapter_dir.mkdir(parents=True)
    content = "她立刻明白了一切。" + "海雾落在窗外，邮戳仍是湿的。" * 220
    content_path = chapter_dir / "content.txt"
    content_path.write_text(content, encoding="utf-8")
    database.add_novel_chapter(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        title="第一章 迟到的信",
        outline="林岚拆信并核对邮戳。",
        key_points="邮戳日期异常",
        content_path=content_path,
    )
    version_path = chapter_dir / "versions" / "manual-v1.txt"
    version_path.parent.mkdir()
    version_path.write_text(content, encoding="utf-8")
    version_id = database.record_manual_chapter_version(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        version_path=version_path,
        char_count=len(content),
        effective_char_count=len(content),
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        change_summary="初稿",
    )
    assert version_id
    with database.connection() as connection:
        connection.execute(
            """
            UPDATE novel_chapters SET head_version_id=?
            WHERE id=?
            """,
            (version_id, chapter_id),
        )
        connection.commit()
    return user_id, project_id, chapter_id, version_id, content


def test_native_agent_loop_round_trips_tool_results_and_edits_workspace(
    tmp_path: Path,
):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id, project_id, chapter_id, _version_id, _content = seed_novel(
        database, tmp_path
    )
    service = AssistantChatService(
        database, tmp_path / "novels", tmp_path / "documents"
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="chapter",
        title="局部修订",
        project_id=project_id,
        novel_chapter_id=chapter_id,
    )
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="把第一句的直接总结改成动作，其他内容别动。",
        provider="mock",
        model="native-test-model",
        credential_source="default",
        agent_role="editor",
        auto_commit=True,
    )
    claimed = service.claim_next_message()
    assert claimed and claimed["id"] == message_id
    payload = service.build_job_payload(claimed)
    model = ScriptedNativeEditModel()

    response = asyncio.run(
        AssistantAgentOrchestrator(service).run(
            model=model,
            item=claimed,
            payload=payload,
            provider_user_id="u_test",
        )
    )

    assert model.turns == 3
    assert response.result.draft is not None
    assert response.result.draft.content.startswith("她把信纸翻到背面。")
    assert [item["tool_name"] for item in response.agent_trace] == [
        "read",
        "edit",
    ]
    assert service.complete_message(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        response=response,
    )
    chapter = database.get_novel_chapter(user_id, project_id, chapter_id)
    assert Path(chapter["content_path"]).read_text(
        encoding="utf-8"
    ).startswith("她把信纸翻到背面。")


def test_native_agent_task_is_bounded_read_only_and_non_recursive(
    tmp_path: Path,
):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id, project_id, chapter_id, _version_id, content = seed_novel(
        database, tmp_path
    )
    service = AssistantChatService(
        database, tmp_path / "novels", tmp_path / "documents"
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="chapter",
        title="连续性核对",
        project_id=project_id,
        novel_chapter_id=chapter_id,
    )
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="只分析第一章和核心设定是否存在矛盾，不要修改。",
        provider="mock",
        model="native-agent-model",
        credential_source="default",
        agent_role="analyst",
    )
    claimed = service.claim_next_message()
    assert claimed and claimed["id"] == message_id
    agent_model = ScriptedNativeTaskModel()
    specialist_model = NativeSpecialistModel()

    response = asyncio.run(
        AssistantAgentOrchestrator(service).run(
            model=agent_model,
            routing_model=specialist_model,
            item=claimed,
            payload=service.build_job_payload(claimed),
            provider_user_id="u_test",
        )
    )

    assert agent_model.turns == 2
    assert len(specialist_model.calls) == 1
    assert response.result.draft is None
    assert response.result.settings_patch is None
    assert response.agent_trace == [
        {
            "sequence": 1,
            "tool_name": "task",
            "label": "委托专项分析",
            "status": "completed",
            "read_only": True,
            "category": "agent_action",
        }
    ]
    assert response.input_tokens >= 35
    chapter = database.get_novel_chapter(user_id, project_id, chapter_id)
    assert Path(chapter["content_path"]).read_text(encoding="utf-8") == content


def test_native_agent_delegates_long_chapter_text_to_plain_prose_pipeline(
    tmp_path: Path,
):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id, project_id, chapter_id, _version_id, _content = seed_novel(
        database, tmp_path
    )
    service = AssistantChatService(
        database, tmp_path / "novels", tmp_path / "documents"
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="chapter",
        title="重写第一章",
        project_id=project_id,
        novel_chapter_id=chapter_id,
    )
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="重写这一章，让林岚最后决定去灯塔。",
        provider="mock",
        model="native-agent-model",
        credential_source="default",
        agent_role="writer",
        auto_commit=True,
    )
    claimed = service.claim_next_message()
    assert claimed and claimed["id"] == message_id
    payload = service.build_job_payload(claimed)
    agent_model = ScriptedNativeDraftModel()
    prose_model = NativeProseModel()
    audit_model = RevisingNativeAuditModel()
    answer_updates = []

    response = asyncio.run(
        AssistantAgentOrchestrator(service).run(
            model=agent_model,
            routing_model=audit_model,
            prose_model=prose_model,
            item=claimed,
            payload=payload,
            provider_user_id="u_test",
            on_answer_update=answer_updates.append,
        )
    )

    assert agent_model.turns == 3
    assert len(prose_model.requests) == 1
    assert prose_model.requests[0]["tools"] == []
    assert "只返回正文" in prose_model.requests[0]["messages"][1][
        "content"
    ]
    assert response.result.draft is not None
    assert response.result.draft.content.startswith("海雾贴着窗玻璃")
    assert response.result.draft.content.endswith("朝灯塔走去。")
    assert len(answer_updates) == 3
    assert answer_updates[-1] == response.result.draft.content
    assert [item["tool_name"] for item in response.agent_trace] == [
        "read",
        "compose",
        "draft_quality_audit",
    ]
    assert audit_model.audit_calls == 1
    assert audit_model.last_context["quality_mode"] == "standard"
    assert audit_model.last_context["writing_packet"]["scene_contract"]
    assert response.input_tokens >= 45
    assert response.output_tokens >= 34
    assert service.complete_message(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        response=response,
    )
    with database.connection() as connection:
        memory_job = connection.execute(
            """
            SELECT operation, project_id, chapter_id, status
            FROM generation_jobs
            WHERE operation='extract_story_delta'
            """
        ).fetchone()
    assert memory_job is not None
    assert memory_job["project_id"] == project_id
    assert memory_job["chapter_id"] == chapter_id
    assert memory_job["status"] == "queued"


def test_low_quality_native_prose_skips_second_pass_audit(tmp_path: Path):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id, project_id, chapter_id, _version_id, _content = seed_novel(
        database, tmp_path
    )
    service = AssistantChatService(
        database, tmp_path / "novels", tmp_path / "documents"
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="chapter",
        title="快速续写",
        project_id=project_id,
        novel_chapter_id=chapter_id,
    )
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="快速重写这一章。",
        provider="mock",
        model="native-agent-model",
        credential_source="default",
        agent_role="writer",
        auto_commit=True,
        quality_mode="low",
    )
    claimed = service.claim_next_message()
    assert claimed and claimed["id"] == message_id
    audit_model = RevisingNativeAuditModel()

    response = asyncio.run(
        AssistantAgentOrchestrator(service).run(
            model=ScriptedNativeDraftModel(),
            routing_model=audit_model,
            prose_model=NativeProseModel(),
            item=claimed,
            payload=service.build_job_payload(claimed),
            provider_user_id="u_test",
        )
    )

    assert audit_model.audit_calls == 0
    assert response.result.draft is not None
    assert response.result.draft.content.endswith("天亮前，她出了门。")
    assert [item["tool_name"] for item in response.agent_trace] == [
        "read",
        "compose",
    ]


def test_chat_message_length_uses_only_a_high_server_safety_limit(
    tmp_path: Path,
):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id, project_id, _chapter_id, _version_id, _content = seed_novel(
        database, tmp_path
    )
    service = AssistantChatService(
        database, tmp_path / "novels", tmp_path / "documents"
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="project",
        title="长篇资料讨论",
        project_id=project_id,
    )

    long_question = "海" * 8_001
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question=long_question,
        provider="mock",
        model="mock-creative-chat",
        credential_source="default",
        agent_role="advisor",
    )
    queued = service.get_message(user_id=user_id, message_id=message_id)
    conversation = service.get_conversation(
        user_id=user_id, conversation_id=conversation_id
    )
    assert queued
    assert conversation
    assert conversation["messages"][0]["content"] == long_question

    with pytest.raises(ValueError, match="100,000"):
        service.queue_message(
            user_id=user_id,
            conversation_id=conversation_id,
            question="海" * (MAX_USER_MESSAGE_CHARS + 1),
            provider="mock",
            model="mock-creative-chat",
            credential_source="default",
            agent_role="advisor",
        )


def test_conversation_remembers_quality_mode_and_new_chat_inherits_it(
    tmp_path: Path,
):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id = database.create_user(
        "quality-mode-user", hash_password("password-123")
    )
    project_id = "q" * 32
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="模式测试",
        genre="悬疑",
        premise="测试模型模式能否可靠记忆。",
        world_setting="",
        style_guide="",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
    )
    database.upsert_model_routing_preferences(
        user_id=user_id,
        fast_provider="deepseek",
        fast_model="deepseek-v4-flash",
        quality_provider="deepseek",
        quality_model="deepseek-v4-pro",
        default_quality_mode="max",
    )
    service = AssistantChatService(
        database, tmp_path / "novels", tmp_path / "documents"
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="project",
        title="模式测试",
        project_id=project_id,
    )
    conversation = service.get_conversation(
        user_id=user_id, conversation_id=conversation_id
    )
    assert conversation and conversation["quality_mode"] == "max"

    assert (
        service.set_conversation_quality_mode(
            user_id=user_id,
            conversation_id=conversation_id,
            quality_mode="standard",
        )
        == "standard"
    )
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="讨论开场。",
        provider="mock",
        model="mock-creative-chat",
        credential_source="default",
        quality_mode="low",
    )
    claimed = service.claim_next_message()
    assert claimed and claimed["id"] == message_id
    assert claimed["quality_mode"] == "low"
    conversation = service.get_conversation(
        user_id=user_id, conversation_id=conversation_id
    )
    assert conversation and conversation["quality_mode"] == "low"
    assert (
        database.get_model_routing_preferences(user_id)[
            "default_quality_mode"
        ]
        == "low"
    )

    inherited_id = service.create_conversation(
        user_id=user_id,
        scope_type="project",
        title="新对话",
        project_id=project_id,
    )
    inherited = service.get_conversation(
        user_id=user_id, conversation_id=inherited_id
    )
    assert inherited and inherited["quality_mode"] == "low"


def test_long_conversation_keeps_memory_and_searches_complete_history(
    tmp_path: Path,
):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id, project_id, _chapter_id, _version_id, _content = seed_novel(
        database, tmp_path
    )
    service = AssistantChatService(
        database, tmp_path / "novels", tmp_path / "documents"
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="project",
        title="长期构思",
        project_id=project_id,
    )
    model = MockAgentModel()

    for index in range(10):
        question = (
            "请记住：女主把青铜钥匙藏在旧收音机里。"
            if index == 0
            else f"继续讨论第 {index + 1} 个构思问题。"
        )
        message_id = service.queue_message(
            user_id=user_id,
            conversation_id=conversation_id,
            question=question,
            provider="mock",
            model="mock-creative-chat",
            credential_source="default",
            agent_role="advisor",
        )
        claimed = service.claim_next_message()
        assert claimed and claimed["id"] == message_id
        response = run_agent(service, claimed, model=model)
        assert service.complete_message(
            message_id=message_id,
            claim_token=claimed["claim_token"],
            response=response,
        )

    current_message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="我之前说过青铜钥匙藏在哪里？",
        provider="mock",
        model="mock-creative-chat",
        credential_source="default",
        agent_role="auto",
    )
    claimed = service.claim_next_message()
    assert claimed and claimed["id"] == current_message_id
    payload = service.build_job_payload(claimed)

    assert len(payload["history"]) == 16
    assert "青铜钥匙藏在旧收音机里" in (
        payload["context"]["conversation_memory"]
    )
    assert payload["context"]["conversation_history_search_available"] is True
    conversation = service.get_conversation(
        user_id=user_id,
        conversation_id=conversation_id,
    )
    assert conversation["memory_message_count"] == 20
    assert "青铜钥匙" in conversation["memory_summary"]

    orchestrated = run_agent(
        service,
        claimed,
        model=FixedIntentModel("discuss"),
    )
    assert "旧收音机" in orchestrated.result.answer
    assert orchestrated.agent_trace == []


def test_project_writing_request_creates_blank_chapter_and_draft(
    tmp_path: Path,
):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id = database.create_user(
        "blank-chapter-writer", hash_password("password-123")
    )
    project_id = "b" * 32
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="",
        genre="",
        premise="一封迟到七年的信打破海港的平静。",
        world_setting="冬季海港。",
        style_guide="克制，以动作承担情绪。",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
    )
    service = AssistantChatService(
        database, tmp_path / "novels", tmp_path / "documents"
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="project",
        title="开始创作",
        project_id=project_id,
    )

    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="现在能开始写第一章正文了吗？",
        provider="mock",
        model="mock-creative-chat",
        credential_source="default",
        agent_role="auto",
        auto_commit=True,
    )

    chapters = database.list_novel_chapters(user_id, project_id)
    assert chapters == []
    conversation = service.get_conversation(
        user_id=user_id, conversation_id=conversation_id
    )
    assert conversation["scope_type"] == "project"

    claimed = service.claim_next_message()
    assert claimed and claimed["id"] == message_id
    payload = service.build_job_payload(claimed)
    assert payload["context"]["scope"] == "novel_project"
    assert payload["context"]["dispatch"] == {
        "requested_role": "auto",
        "resolved_role": "pending",
        "intent": "pending",
        "reason": "model_intent_pending",
        "goal": "classify_intent",
            "settings_ready": True,
            "ui_surface": "project",
            "scope_preflight": {},
    }
    response = asyncio.run(
        AssistantAgentOrchestrator(service).run(
            model=FixedIntentModel("draft_prose"),
            item=claimed,
            payload=payload,
            provider_user_id="u_test",
        )
    )
    assert response.result.draft is not None
    chapters = database.list_novel_chapters(user_id, project_id)
    assert len(chapters) == 1
    assert chapters[0]["title"] == ""
    chapter_id = str(chapters[0]["id"])
    conversation = service.get_conversation(
        user_id=user_id, conversation_id=conversation_id
    )
    assert conversation["scope_type"] == "chapter"
    assert conversation["novel_chapter_id"] == chapter_id
    assert service.complete_message(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        response=response,
    )
    chapter = database.get_novel_chapter(
        user_id, project_id, chapter_id
    )
    assert chapter["head_version_id"]
    assert Path(chapter["content_path"]).read_text(encoding="utf-8")


def test_new_chapter_request_creates_next_chapter_without_touching_previous(
    tmp_path: Path,
):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id, project_id, chapter_id, version_id, content = seed_novel(
        database, tmp_path
    )
    service = AssistantChatService(
        database, tmp_path / "novels", tmp_path / "documents"
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="chapter",
        title="继续写下一章",
        project_id=project_id,
        novel_chapter_id=chapter_id,
    )
    selected = content[-30:]
    selected_start = content.rfind(selected)
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="请承接现在的结尾，写一个新的下一章。",
        provider="mock",
        model="mock-creative-chat",
        credential_source="default",
        agent_role="auto",
        quote={
            "source_type": "novel_version",
            "project_id": project_id,
            "novel_chapter_id": chapter_id,
            "version_id": version_id,
            "start_offset": selected_start,
            "end_offset": selected_start + len(selected),
            "quote_text": selected,
            "content_hash": hashlib.sha256(
                content.encode("utf-8")
            ).hexdigest(),
        },
        auto_commit=True,
    )
    claimed = service.claim_next_message()
    assert claimed and claimed["id"] == message_id
    response = run_agent(service, claimed)

    chapters = database.list_novel_chapters(user_id, project_id)
    assert len(chapters) == 2
    assert [int(item["position"]) for item in chapters] == [1, 2]
    next_chapter_id = str(chapters[1]["id"])
    conversation = service.get_conversation(
        user_id=user_id, conversation_id=conversation_id
    )
    assert conversation["scope_type"] == "chapter"
    assert conversation["novel_chapter_id"] == next_chapter_id
    assert response.result.draft
    assert service.complete_message(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        response=response,
    )
    previous = database.get_novel_chapter(
        user_id, project_id, chapter_id
    )
    assert previous["head_version_id"] == version_id
    assert Path(previous["content_path"]).read_text(encoding="utf-8") == content
    generated = database.get_novel_chapter(
        user_id, project_id, next_chapter_id
    )
    assert generated["head_version_id"]
    assert Path(generated["content_path"]).read_text(
        encoding="utf-8"
    ).strip()
    stream_state = service.get_message_stream_state(
        user_id=user_id, message_id=message_id
    )
    assert stream_state
    assert stream_state["project_id"] == project_id
    assert stream_state["novel_chapter_id"] == next_chapter_id
    assert stream_state["terminal"] is True
    with database.connection() as connection:
        memory_job = connection.execute(
            """
            SELECT status, version_id
            FROM generation_jobs
            WHERE chapter_id=? AND operation='extract_story_delta'
            """,
            (next_chapter_id,),
        ).fetchone()
    assert memory_job
    assert memory_job["status"] == "queued"
    assert memory_job["version_id"] == generated["head_version_id"]


def test_writer_can_create_then_compose_next_chapter_in_one_agent_run(
    tmp_path: Path,
):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id, project_id, chapter_id, _version_id, _content = seed_novel(
        database, tmp_path
    )
    service = AssistantChatService(
        database, tmp_path / "novels", tmp_path / "documents"
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="chapter",
        title="自主创建下一章",
        project_id=project_id,
        novel_chapter_id=chapter_id,
    )
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="另开一章写林岚抵达灯塔。",
        provider="mock",
        model="native-agent-model",
        credential_source="default",
        agent_role="writer",
        auto_commit=True,
        quality_mode="low",
    )
    claimed = service.claim_next_message()
    assert claimed and claimed["id"] == message_id
    agent_model = ScriptedNativeCreateThenComposeModel()

    response = asyncio.run(
        AssistantAgentOrchestrator(service).run(
            model=agent_model,
            prose_model=NativeProseModel(),
            item=claimed,
            payload=service.build_job_payload(claimed),
            provider_user_id="u_test",
        )
    )

    assert agent_model.turns == 4
    assert response.result.draft is not None
    assert [item["tool_name"] for item in response.agent_trace] == [
        "create",
        "read",
        "compose",
    ]
    chapters = database.list_novel_chapters(user_id, project_id)
    assert len(chapters) == 2
    created = chapters[1]
    assert created["title"] == "第二章 灯塔"
    assert created["outline"].startswith("林岚抵达灯塔")
    conversation = service.get_conversation(
        user_id=user_id,
        conversation_id=conversation_id,
    )
    assert conversation["novel_chapter_id"] == created["id"]
    assert service.complete_message(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        response=response,
    )
    saved = database.get_novel_chapter(
        user_id,
        project_id,
        str(created["id"]),
    )
    assert saved["head_version_id"]
    assert Path(saved["content_path"]).read_text(encoding="utf-8")


def test_unwritten_project_script_can_create_first_chapter_directly(
    tmp_path: Path,
):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id = database.create_user(
        "script-onboarding-writer", hash_password("password-123")
    )
    project_id = "o" * 32
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="",
        genre="",
        premise="",
        world_setting="",
        style_guide="",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
    )
    service = AssistantChatService(
        database, tmp_path / "novels", tmp_path / "documents"
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="project",
        title="导入剧本素材",
        project_id=project_id,
    )
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="这是我提供的剧本素材，请直接写第一章。",
        provider="mock",
        model="mock-creative-chat",
        credential_source="default",
        agent_role="auto",
    )

    assert database.list_novel_chapters(user_id, project_id) == []
    conversation = service.get_conversation(
        user_id=user_id, conversation_id=conversation_id
    )
    assert conversation["scope_type"] == "project"

    claimed = service.claim_next_message()
    assert claimed and claimed["id"] == message_id
    payload = service.build_job_payload(claimed)
    assert payload["context"]["dispatch"] == {
        "requested_role": "auto",
        "resolved_role": "pending",
        "intent": "pending",
        "reason": "model_intent_pending",
        "goal": "classify_intent",
            "settings_ready": False,
            "ui_surface": "project",
            "scope_preflight": {},
    }
    response = asyncio.run(
        AssistantAgentOrchestrator(service).run(
            model=FixedIntentModel("draft_prose"),
            item=claimed,
            payload=payload,
            provider_user_id="u_test",
        )
    )
    assert response.result.draft is not None
    assert response.result.settings_patch is None
    assert service.complete_message(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        response=response,
    )
    message = service.get_message(
        user_id=user_id, message_id=message_id
    )
    assert message["response"]["agent"]["role"] == "writer"
    assert message["response"]["draft"]
    chapters = database.list_novel_chapters(user_id, project_id)
    assert len(chapters) == 1




def test_quote_bound_edit_saves_full_draft_without_changing_canon(
    tmp_path: Path,
):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id, project_id, chapter_id, version_id, content = seed_novel(
        database, tmp_path
    )
    service = AssistantChatService(
        database, tmp_path / "novels", tmp_path / "documents"
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="chapter",
        title="检查这一句",
        project_id=project_id,
        novel_chapter_id=chapter_id,
    )
    selected = "她立刻明白了一切。"
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="请改写这句，减少直接总结和 AI 味。",
        provider="mock",
        model="mock-creative-chat",
        credential_source="default",
        agent_role="editor",
        quote={
            "source_type": "novel_version",
            "project_id": project_id,
            "novel_chapter_id": chapter_id,
            "version_id": version_id,
            "start_offset": 0,
            "end_offset": len(selected),
            "quote_text": selected,
            "content_hash": hashlib.sha256(
                content.encode("utf-8")
            ).hexdigest(),
        },
    )
    claimed = service.claim_next_message()
    assert claimed and claimed["id"] == message_id
    response = run_agent(service, claimed)
    assert service.complete_message(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        response=response,
    )

    conversation = service.get_conversation(
        user_id=user_id, conversation_id=conversation_id
    )
    assistant_message = conversation["messages"][-1]
    assert assistant_message["status"] == "completed"
    assert "动作停在答案之前" in assistant_message["response"]["draft"][
        "content"
    ]

    saved = service.commit_draft_to_head(
        user_id=user_id, assistant_message_id=message_id
    )
    chapter = database.get_novel_chapter(
        user_id, project_id, chapter_id
    )
    assert chapter["head_version_id"] == saved["version_id"]
    candidate = database.get_chapter_version(
        user_id, project_id, chapter_id, saved["version_id"]
    )
    assert candidate["parent_version_id"] == version_id
    assert candidate["source"] == "assistant_chat"
    assert candidate["created_by"] == "assistant"
    candidate_text = Path(candidate["content_path"]).read_text(
        encoding="utf-8"
    )
    assert candidate_text != content
    assert "动作停在答案之前" in candidate_text

    second = service.commit_draft_to_head(
        user_id=user_id, assistant_message_id=message_id
    )
    assert second["version_id"] == saved["version_id"]


def test_quote_offsets_are_normalized_against_verified_source(
    tmp_path: Path,
):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id, project_id, chapter_id, version_id, content = seed_novel(
        database, tmp_path
    )
    service = AssistantChatService(
        database, tmp_path / "novels", tmp_path / "documents"
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="chapter",
        title="浏览器选区兼容",
        project_id=project_id,
        novel_chapter_id=chapter_id,
    )
    selected = "她立刻明白了一切。"
    service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="请检查这个选区。",
        provider="mock",
        model="mock-creative-chat",
        credential_source="default",
        agent_role="editor",
        quote={
            "source_type": "novel_version",
            "project_id": project_id,
            "novel_chapter_id": chapter_id,
            "version_id": version_id,
            "start_offset": 1,
            "end_offset": len(selected) + 1,
            "quote_text": selected,
            "content_hash": hashlib.sha256(
                content.encode("utf-8")
            ).hexdigest(),
        },
    )
    conversation = service.get_conversation(
        user_id=user_id, conversation_id=conversation_id
    )
    user_message = conversation["messages"][0]
    assert user_message["quote"]["start_offset"] == 0
    assert user_message["quote"]["end_offset"] == len(selected)


def test_quote_line_endings_are_normalized_from_browser_form(
    tmp_path: Path,
):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id, project_id, chapter_id, version_id, content = seed_novel(
        database, tmp_path
    )
    service = AssistantChatService(
        database, tmp_path / "novels", tmp_path / "documents"
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="chapter",
        title="表单换行兼容",
        project_id=project_id,
        novel_chapter_id=chapter_id,
    )
    selected = content[:40]
    content_with_lines = selected[:12] + "\n\n" + selected[12:]
    version_path = Path(
        database.get_chapter_version(
            user_id, project_id, chapter_id, version_id
        )["content_path"]
    )
    original = version_path.read_text(encoding="utf-8")
    version_path.write_text(
        content_with_lines + original[40:], encoding="utf-8"
    )
    normalized_content = version_path.read_text(encoding="utf-8")
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="检查带换行的选区。",
        provider="mock",
        model="mock-creative-chat",
        credential_source="default",
        agent_role="editor",
        quote={
            "source_type": "novel_version",
            "project_id": project_id,
            "novel_chapter_id": chapter_id,
            "version_id": version_id,
            "start_offset": 0,
            "end_offset": len(content_with_lines),
            "quote_text": content_with_lines.replace("\n", "\r\n"),
            "content_hash": hashlib.sha256(
                normalized_content.encode("utf-8")
            ).hexdigest(),
        },
    )
    message = service.get_conversation(
        user_id=user_id, conversation_id=conversation_id
    )["messages"][0]
    assert message["id"] != message_id
    assert message["quote"]["quote_text"] == content_with_lines


def test_chat_rejects_stale_or_cross_scope_quote(tmp_path: Path):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id, project_id, chapter_id, version_id, content = seed_novel(
        database, tmp_path
    )
    service = AssistantChatService(
        database, tmp_path / "novels", tmp_path / "documents"
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="chapter",
        title="检查引用",
        project_id=project_id,
        novel_chapter_id=chapter_id,
    )
    selected = "她立刻明白了一切。"
    with pytest.raises(ValueError, match="正文已发生变化"):
        service.queue_message(
            user_id=user_id,
            conversation_id=conversation_id,
            question="这句怎么改？",
            provider="mock",
            model="mock-creative-chat",
            credential_source="default",
            quote={
                "source_type": "novel_version",
                "project_id": project_id,
                "novel_chapter_id": chapter_id,
                "version_id": version_id,
                "start_offset": 0,
                "end_offset": len(selected),
                "quote_text": selected,
                "content_hash": "0" * 64,
            },
        )
    other_user = database.create_user(
        "other-chat-user", hash_password("password-123")
    )
    assert (
        service.get_conversation(
            user_id=other_user, conversation_id=conversation_id
        )
        is None
    )
    assert hashlib.sha256(content.encode("utf-8")).hexdigest()


def test_advisor_is_read_only_for_text_edit(tmp_path: Path):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id, project_id, chapter_id, version_id, content = seed_novel(
        database, tmp_path
    )
    service = AssistantChatService(
        database, tmp_path / "novels", tmp_path / "documents"
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="chapter",
        title="只讨论这句话",
        project_id=project_id,
        novel_chapter_id=chapter_id,
    )
    selected = "她立刻明白了一切。"
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="请改写这句。",
        provider="mock",
        model="mock-creative-chat",
        credential_source="default",
        agent_role="advisor",
        quote={
            "source_type": "novel_version",
            "project_id": project_id,
            "novel_chapter_id": chapter_id,
            "version_id": version_id,
            "start_offset": 0,
            "end_offset": len(selected),
            "quote_text": selected,
            "content_hash": hashlib.sha256(
                content.encode("utf-8")
            ).hexdigest(),
        },
    )
    claimed = service.claim_next_message()
    response = run_agent(service, claimed)
    assert service.complete_message(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        response=response,
    )
    message = service.get_message(
        user_id=user_id, message_id=message_id
    )
    assert message["response"]["agent"]["role"] == "advisor"
    assert message["response"]["draft"] is None


def test_project_chat_proposes_and_applies_settings_candidate(
    tmp_path: Path,
):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id = database.create_user(
        "settings-chat-writer", hash_password("password-123")
    )
    project_id = "s" * 32
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="",
        genre="",
        premise="",
        world_setting="",
        style_guide="",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
    )
    service = AssistantChatService(
        database, tmp_path / "novels", tmp_path / "documents"
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="project",
        title="讨论新书",
        project_id=project_id,
    )
    question = "我想写一个记者在雨夜收到七年前来信的悬疑故事。"
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question=question,
        provider="mock",
        model="mock-creative-chat",
        credential_source="default",
        agent_role="auto",
    )
    claimed = service.claim_next_message()
    response = asyncio.run(
        AssistantAgentOrchestrator(service).run(
            model=FixedIntentModel("update_settings"),
            item=claimed,
            payload=service.build_job_payload(claimed),
            provider_user_id="u_test",
        )
    )
    assert service.complete_message(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        response=response,
    )
    project = database.get_novel_project(user_id, project_id)
    assert project["premise"] == ""
    message = service.get_message(
        user_id=user_id, message_id=message_id
    )
    assert message["response"]["agent"]["role"] == "planner"
    assert message["response"]["settings_patch"]["premise"] == question
    assert message["response"]["settings_patch_status"] == "candidate"

    applied = service.apply_settings_candidate(
        user_id=user_id, assistant_message_id=message_id
    )
    assert set(applied["changed_fields"]) == {"premise", "genre"}
    project = database.get_novel_project(user_id, project_id)
    assert project["premise"] == question
    assert project["genre"] == "悬疑"
    second = service.apply_settings_candidate(
        user_id=user_id, assistant_message_id=message_id
    )
    assert second["already_applied"] is True


def test_project_chat_directly_applies_settings_by_default(tmp_path: Path):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id = database.create_user(
        "direct-settings-writer", hash_password("password-123")
    )
    project_id = "d" * 32
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="",
        genre="",
        premise="",
        world_setting="",
        style_guide="",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
    )
    service = AssistantChatService(
        database, tmp_path / "novels", tmp_path / "documents"
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="project",
        title="直接整理新书",
        project_id=project_id,
    )
    question = "我想写一个记者在雨夜收到七年前来信的悬疑故事。"
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question=question,
        provider="mock",
        model="mock-creative-chat",
        credential_source="default",
        agent_role="auto",
        ui_surface="settings",
        auto_commit=True,
    )
    claimed = service.claim_next_message()
    response = asyncio.run(
        AssistantAgentOrchestrator(service).run(
            model=FixedIntentModel("update_settings"),
            item=claimed,
            payload=service.build_job_payload(claimed),
            provider_user_id="u_test",
        )
    )

    assert service.complete_message(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        response=response,
    )
    project = database.get_novel_project(user_id, project_id)
    assert project["premise"] == question
    assert project["genre"] == "悬疑"
    message = service.get_message(user_id=user_id, message_id=message_id)
    assert message["response"]["settings_patch_status"] == "applied"
    assert (
        message["response"]["boundary"]["project_settings_unchanged"]
        is False
    )
    stream = service.get_message_stream_state(
        user_id=user_id, message_id=message_id
    )
    assert stream and stream["terminal"] is True


def test_explicit_discussion_only_never_writes_settings(tmp_path: Path):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id = database.create_user(
        "readonly-settings-writer", hash_password("password-123")
    )
    project_id = "r" * 32
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="只读讨论",
        genre="",
        premise="",
        world_setting="",
        style_guide="",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
    )
    service = AssistantChatService(
        database, tmp_path / "novels", tmp_path / "documents"
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="project",
        title="只讨论",
        project_id=project_id,
    )
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="只讨论一下这个构思，不要写入作品资料。",
        provider="mock",
        model="mock-creative-chat",
        credential_source="default",
        agent_role="auto",
        ui_surface="settings",
        auto_commit=True,
    )
    claimed = service.claim_next_message()
    response = asyncio.run(
        AssistantAgentOrchestrator(service).run(
            model=MockAgentModel(),
            item=claimed,
            payload=service.build_job_payload(claimed),
            provider_user_id="u_test",
        )
    )
    assert service.complete_message(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        response=response,
    )

    project = database.get_novel_project(user_id, project_id)
    assert project["premise"] == ""
    message = service.get_message(user_id=user_id, message_id=message_id)
    assert message["response"]["agent"]["role"] == "advisor"
    assert message["response"]["settings_patch"] is None


def test_settings_candidate_creates_structured_character_card(
    tmp_path: Path,
):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id = database.create_user(
        "classified-settings-writer", hash_password("password-123")
    )
    project_id = "m" * 32
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="分类资料测试",
        genre="悬疑",
        premise="记者追查一封来历不明的信。",
        world_setting="当代海港。",
        style_guide="",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
    )
    service = AssistantChatService(
        database, tmp_path / "novels", tmp_path / "documents"
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="project",
        title="整理人物",
        project_id=project_id,
    )
    question = "请记录人物资料：林悦是调查记者，遇事先核对物证。"
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question=question,
        provider="mock",
        model="mock-creative-chat",
        credential_source="default",
        agent_role="auto",
    )
    claimed = service.claim_next_message()
    class ExplicitSettingsModel(MockAgentModel):
        def __init__(self):
            self.native_turn_calls = 0

        async def classify_intent(self, **kwargs):
            raise AssertionError("明确设定命令不应调用模型分类")

        async def native_turn(self, **kwargs):
            self.native_turn_calls += 1
            return await super().native_turn(**kwargs)

    model = ExplicitSettingsModel()
    response = asyncio.run(
        AssistantAgentOrchestrator(service).run(
            model=model,
            item=claimed,
            payload=service.build_job_payload(claimed),
            provider_user_id="u_test",
        )
    )
    assert model.native_turn_calls >= 1
    assert response.result.settings_patch is not None
    proposed = response.result.settings_patch.model_dump(exclude_none=True)
    assert "world_setting" not in proposed
    assert "archive_rules" not in proposed
    edit = proposed["structured_edits"][0]
    assert edit["entity_type"] == "character"
    assert edit["action"] == "create"
    assert edit["changes"]["name"] == "林悦"
    assert edit["changes"]["role"] == "调查记者"
    assert service.complete_message(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        response=response,
    )

    applied = service.apply_settings_candidate(
        user_id=user_id,
        assistant_message_id=message_id,
    )
    assert applied["changed_fields"] == ["structured_edits"]
    project = database.get_novel_project(user_id, project_id)
    assert project["world_setting"] == "当代海港。"
    structured = service.structured_settings_editor.snapshot(
        user_id=user_id,
        project_id=project_id,
    )
    character = structured["characters"][0]
    assert character["name"] == "林悦"
    assert character["role"] == "调查记者"
    assert character["traits"] == "遇事先核对物证"


def test_mock_character_micro_edit_uses_only_requested_value():
    question = (
        "请只修改林澄的人物卡：把内在需求改为"
        "‘承认自己害怕再次误判，并学会让同伴复核’，"
        "其他字段不要动。"
    )
    intent = asyncio.run(
        MockAgentModel().classify_intent(
            context={
                "scope": "novel_project",
                "dispatch": {
                    "ui_surface": "settings",
                    "settings_ready": True,
                },
            },
            history=[],
            question=question,
            has_selected_quote=False,
            provider_user_id="u_test",
        )
    )
    patch = _mock_settings_patch(
        {
            "project": {},
            "structured_settings": {
                "characters": [{"id": "character-1", "name": "林澄"}]
            },
        },
        question,
    )

    assert intent.decision.intent == "update_settings"
    assert patch["structured_edits"][0]["changes"] == {
        "internal_need": "承认自己害怕再次误判，并学会让同伴复核"
    }


def test_project_chat_proposes_and_applies_versioned_story_plan(
    tmp_path: Path,
):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id = database.create_user(
        "story-plan-chat-writer", hash_password("password-123")
    )
    project_id = "g" * 32
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="雾港来信",
        genre="悬疑",
        premise="记者收到一封不可能出现的七年前来信。",
        world_setting="冬季海港。",
        style_guide="克制，以行动承担情绪。",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
    )
    service = AssistantChatService(
        database, tmp_path / "novels", tmp_path / "documents"
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="project",
        title="规划全书",
        project_id=project_id,
    )
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="请先规划全书的核心悬问、关键转折和最后兑现。",
        provider="mock",
        model="mock-creative-chat",
        credential_source="default",
        agent_role="auto",
        ui_surface="settings",
    )
    claimed = service.claim_next_message()
    response = asyncio.run(
        AssistantAgentOrchestrator(service).run(
            model=FixedIntentModel("plan_story"),
            item=claimed,
            payload=service.build_job_payload(claimed),
            provider_user_id="u_test",
        )
    )
    assert service.complete_message(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        response=response,
    )
    message = service.get_message(
        user_id=user_id, message_id=message_id
    )
    assert message["response"]["agent"]["role"] == "story_planner"
    assert message["response"]["story_plan_status"] == "candidate"
    assert message["response"]["story_plan"]["blueprint"]["major_turns"]

    applied = service.apply_story_plan_candidate(
        user_id=user_id, assistant_message_id=message_id
    )
    assert applied["already_applied"] is False
    plan = StoryPlanningService(database).get_blueprint(
        user_id=user_id, project_id=project_id
    )
    assert plan["confirmed_version_id"] == applied["version_id"]
    assert plan["confirmed"]["central_question"]
    second = service.apply_story_plan_candidate(
        user_id=user_id, assistant_message_id=message_id
    )
    assert second["already_applied"] is True


def test_project_chat_directly_applies_story_plan_by_default(
    tmp_path: Path,
):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id = database.create_user(
        "direct-plan-writer", hash_password("password-123")
    )
    project_id = "q" * 32
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="雾港来信",
        genre="悬疑",
        premise="记者收到一封不可能出现的七年前来信。",
        world_setting="冬季海港。",
        style_guide="克制，以行动承担情绪。",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
    )
    service = AssistantChatService(
        database, tmp_path / "novels", tmp_path / "documents"
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="project",
        title="直接规划全书",
        project_id=project_id,
    )
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="请规划全书的核心悬问、关键转折和最后兑现。",
        provider="mock",
        model="mock-creative-chat",
        credential_source="default",
        agent_role="auto",
        ui_surface="settings",
        auto_commit=True,
    )
    claimed = service.claim_next_message()
    response = asyncio.run(
        AssistantAgentOrchestrator(service).run(
            model=FixedIntentModel("plan_story"),
            item=claimed,
            payload=service.build_job_payload(claimed),
            provider_user_id="u_test",
        )
    )
    assert service.complete_message(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        response=response,
    )

    message = service.get_message(user_id=user_id, message_id=message_id)
    assert message["response"]["story_plan_status"] == "applied"
    plan = StoryPlanningService(database).get_blueprint(
        user_id=user_id, project_id=project_id
    )
    assert plan["confirmed_version_id"] == (
        message["response"]["story_plan_version_id"]
    )


def test_writer_creates_candidate_without_overwriting_canon(
    tmp_path: Path,
):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id, project_id, chapter_id, version_id, content = seed_novel(
        database, tmp_path
    )
    service = AssistantChatService(
        database, tmp_path / "novels", tmp_path / "documents"
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="chapter",
        title="续写这一章",
        project_id=project_id,
        novel_chapter_id=chapter_id,
    )
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="请紧接当前正文续写一段。",
        provider="mock",
        model="mock-creative-chat",
        credential_source="default",
        agent_role="writer",
    )
    claimed = service.claim_next_message()
    response = run_agent(service, claimed)
    assert service.complete_message(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        response=response,
    )
    chapter = database.get_novel_chapter(
        user_id, project_id, chapter_id
    )
    assert chapter["head_version_id"] == version_id
    assert Path(chapter["content_path"]).read_text(
        encoding="utf-8"
    ) == content
    message = service.get_message(
        user_id=user_id, message_id=message_id
    )
    assert message["response"]["agent"]["role"] == "writer"
    assert message["response"]["draft"]["content"].startswith(content)

    saved = service.commit_draft_to_head(
        user_id=user_id, assistant_message_id=message_id
    )
    chapter = database.get_novel_chapter(
        user_id, project_id, chapter_id
    )
    assert chapter["head_version_id"] == saved["version_id"]
    candidate = database.get_chapter_version(
        user_id, project_id, chapter_id, saved["version_id"]
    )
    assert candidate["kind"] == "assistant_draft"
    assert candidate["head_version_id"] == candidate["id"]
    candidate_text = Path(candidate["content_path"]).read_text(
        encoding="utf-8"
    )
    assert candidate_text.startswith(content)
    assert "雨声先落在窗外" in candidate_text


def test_writer_auto_commits_working_copy_and_can_revert(
    tmp_path: Path,
):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id, project_id, chapter_id, version_id, content = seed_novel(
        database, tmp_path
    )
    service = AssistantChatService(
        database, tmp_path / "novels", tmp_path / "documents"
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="chapter",
        title="自动续写",
        project_id=project_id,
        novel_chapter_id=chapter_id,
    )
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="紧接正文续写一个动作段落。",
        provider="mock",
        model="mock-creative-chat",
        credential_source="default",
        agent_role="writer",
        auto_commit=True,
    )
    claimed = service.claim_next_message()
    response = run_agent(service, claimed)
    assert service.complete_message(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        response=response,
    )
    chapter = database.get_novel_chapter(
        user_id, project_id, chapter_id
    )
    auto_version_id = str(chapter["head_version_id"])
    assert auto_version_id != version_id
    assert chapter["head_version_id"] == auto_version_id
    auto_content = Path(chapter["content_path"]).read_text(
        encoding="utf-8"
    )
    assert auto_content.startswith(content)
    assert len(auto_content) > len(content)
    message = service.get_message(
        user_id=user_id, message_id=message_id
    )
    assert message["applied_version_id"] == auto_version_id
    assert message["response"]["auto_commit"]["status"] == "applied"

    reverted = service.revert_auto_commit(
        user_id=user_id, assistant_message_id=message_id
    )
    chapter = database.get_novel_chapter(
        user_id, project_id, chapter_id
    )
    assert chapter["head_version_id"] == reverted["version_id"]
    assert chapter["head_version_id"] == reverted["version_id"]
    assert Path(chapter["content_path"]).read_text(
        encoding="utf-8"
    ) == content
    revert_version = database.get_chapter_version(
        user_id,
        project_id,
        chapter_id,
        str(reverted["version_id"]),
    )
    assert revert_version["kind"] == "assistant_revert"
    assert revert_version["parent_version_id"] == auto_version_id
    second = service.revert_auto_commit(
        user_id=user_id, assistant_message_id=message_id
    )
    assert second["already_reverted"] is True
    assert second["version_id"] == reverted["version_id"]


def test_editor_auto_commits_validated_selection(tmp_path: Path):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id, project_id, chapter_id, version_id, content = seed_novel(
        database, tmp_path
    )
    service = AssistantChatService(
        database, tmp_path / "novels", tmp_path / "documents"
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="chapter",
        title="自动局部修改",
        project_id=project_id,
        novel_chapter_id=chapter_id,
    )
    selected = "她立刻明白了一切。"
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="请改写成可观察的动作。",
        provider="mock",
        model="mock-creative-chat",
        credential_source="default",
        agent_role="editor",
        auto_commit=True,
        quote={
            "source_type": "novel_version",
            "project_id": project_id,
            "novel_chapter_id": chapter_id,
            "version_id": version_id,
            "start_offset": 0,
            "end_offset": len(selected),
            "quote_text": selected,
            "content_hash": hashlib.sha256(
                content.encode("utf-8")
            ).hexdigest(),
        },
    )
    claimed = service.claim_next_message()
    response = run_agent(service, claimed)
    assert service.complete_message(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        response=response,
    )
    message = service.get_message(
        user_id=user_id, message_id=message_id
    )
    assert message["response"]["auto_commit"]["status"] == "applied"
    assert message["applied_version_id"]
    chapter = database.get_novel_chapter(
        user_id, project_id, chapter_id
    )
    assert chapter["head_version_id"] == message["applied_version_id"]
    changed = Path(chapter["content_path"]).read_text(encoding="utf-8")
    assert changed != content
    assert "动作停在答案之前" in changed


def test_reference_chapter_chat_uses_source_but_cannot_create_draft(
    tmp_path: Path,
):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id = database.create_user(
        "reading-chat-user", hash_password("password-123")
    )
    text = (
        "第一章 雾中的门\n"
        "门没有立刻打开。脚步声先停在门后，随后灯光才从缝里漏出来。\n"
        "第二章 回声\n"
        "她把问题留在桌上，没有追问。"
    )
    chunks = split_chapters(text)
    document_dir = tmp_path / "documents" / str(user_id) / ("d" * 32)
    chapter_dir = document_dir / "chapters"
    chapter_dir.mkdir(parents=True)
    source_path = document_dir / "source.txt"
    source_path.write_text(text, encoding="utf-8")
    chapter_paths = []
    for index, chunk in enumerate(chunks, 1):
        path = chapter_dir / f"{index:05d}.txt"
        path.write_text(chunk.text, encoding="utf-8")
        chapter_paths.append(path)
    document_id = database.create_document(
        user_id=user_id,
        title="参考小说",
        original_filename="reference.txt",
        source_path=source_path,
        source_encoding="utf-8",
        text_length=len(text),
        chunks=chunks,
        chapter_paths=chapter_paths,
    )
    chapter = database.list_chapters(user_id, document_id)[0]
    chapter_text = Path(chapter["content_path"]).read_text(
        encoding="utf-8"
    )
    selected = "脚步声先停在门后"
    start = chapter_text.index(selected)
    service = AssistantChatService(
        database, tmp_path / "novels", tmp_path / "documents"
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="reference_chapter",
        title="拆解信息释放",
        document_id=document_id,
        reference_chapter_id=str(chapter["id"]),
    )
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="请分析这里的信息释放，并改写成原创段落。",
        provider="mock",
        model="mock-creative-chat",
        credential_source="default",
        agent_role="researcher",
        quote={
            "source_type": "reference_chapter",
            "document_id": document_id,
            "reference_chapter_id": chapter["id"],
            "start_offset": start,
            "end_offset": start + len(selected),
            "quote_text": selected,
            "content_hash": hashlib.sha256(
                chapter_text.encode("utf-8")
            ).hexdigest(),
        },
    )
    claimed = service.claim_next_message()
    payload = service.build_job_payload(claimed)
    assert payload["sources"][0]["kind"] == "reference_chapter"
    response = run_agent(service, claimed)
    assert service.complete_message(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        response=response,
    )
    message = service.get_message(
        user_id=user_id, message_id=message_id
    )
    assert message["response"]["agent"]["role"] == "researcher"
    assert message["response"]["draft"] is None
    with pytest.raises(ValueError, match="没有可提交"):
        service.commit_draft_to_head(
            user_id=user_id, assistant_message_id=message_id
        )


def test_novel_chat_web_flow_and_draft_action(tmp_path: Path):
    application = create_app(make_settings(tmp_path))
    with TestClient(application) as client:
        register = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "网页对话作者",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": csrf_from(register.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        novel_form = client.get("/novels/new")
        response = client.post(
            "/novels/new",
            data={
                "title": "雾港来信",
                "genre": "悬疑",
                "premise": "记者收到一封不可能出现的旧日来信。",
                "csrf": csrf_from(novel_form.text),
            },
            follow_redirects=False,
        )
        workbench_url = response.headers["location"]
        project_id = workbench_url.split("/novels/", 1)[1].split("/", 1)[0]
        project_url = f"/novels/{project_id}"
        workspace = client.get(workbench_url)
        assert "共创对话" in workspace.text
        response = client.post(
            f"{project_url}/chapters",
            data={
                "title": "第一章 迟到的信",
                "outline": "林岚拆信并核对邮戳。",
                "key_points": "邮戳日期异常",
                "csrf": csrf_from(workspace.text),
            },
            follow_redirects=False,
        )
        chapter_location = response.headers["location"]
        chapter_id = chapter_location.split("chapter_id=", 1)[1]
        chapter_url = f"{project_url}/chapters/{chapter_id}"
        assert client.get(chapter_url).status_code == 404
        chapter_page = client.get(chapter_location)
        content = (
            "她立刻明白了一切。"
            + "海雾落在窗外，邮戳仍是湿的。" * 220
        )
        response = client.post(
            f"{chapter_url}/save",
            data={
                "content": content,
                "change_summary": "初稿",
                "csrf": csrf_from(chapter_page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        saved_page = client.get(response.headers["location"])
        assert "共创对话" in saved_page.text
        assert 'class="studio-manuscript-view"' in saved_page.text
        assert "引用正文问 AI" not in saved_page.text

        assistant_url = (
            f"{project_url}/assistant?new=true&chapter_id={chapter_id}"
        )
        assistant_redirect = client.get(
            assistant_url, follow_redirects=False
        )
        assert assistant_redirect.status_code == 404
        assistant_page = client.get(
            f"{project_url}/workbench?chapter_id={chapter_id}"
        )
        assert assistant_page.status_code == 200
        assert "共创对话" in assistant_page.text
        assert 'aria-label="新建对话"' in assistant_page.text
        new_response = client.post(
            f"{project_url}/assistant/new",
            data={
                "csrf": csrf_from(assistant_page.text),
                "chapter_id": chapter_id,
            },
            follow_redirects=False,
        )
        assert new_response.status_code == 303
        new_conversation_url = new_response.headers["location"]
        new_conversation_page = client.get(new_conversation_url)
        assert new_conversation_page.status_code == 200
        conversation_match = re.search(
            r"conversation_id=([a-f0-9]+)",
            new_conversation_url,
        )
        assert conversation_match
        conversation_id = conversation_match.group(1)
        assert (
            f'name="conversation_id" value="{conversation_id}"'
            in new_conversation_page.text
        )
        source_hash = re.search(
            r'name="source_hash" value="([^"]+)"',
            assistant_page.text,
        )
        version_id = re.search(
            r'name="source_version_id" value="([^"]+)"',
            assistant_page.text,
        )
        assert source_hash and version_id
        selected = "她立刻明白了一切。"
        response = client.post(
            f"{project_url}/assistant/messages",
            data={
                "csrf": csrf_from(assistant_page.text),
                "question": "请改写这句，减少直接总结和 AI 味。",
                "conversation_id": conversation_id,
                "novel_chapter_id": chapter_id,
                "source_type": "novel_version",
                "source_version_id": version_id.group(1),
                "source_hash": source_hash.group(1),
                "quote_start": "0",
                "quote_end": str(len(selected)),
                "quote_text": selected,
                "return_view": "body",
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        conversation_url = response.headers["location"]
        deadline = time.monotonic() + 3
        rendered = ""
        while time.monotonic() < deadline:
            rendered = client.get(conversation_url).text
            if (
                "章节提交" in rendered
                and "已自动保存为当前版本" in rendered
            ):
                break
            time.sleep(0.03)
        assert "章节提交" in rendered
        assert "已自动保存为当前版本" in rendered
        assert 'aria-label="历史对话"' in rendered
        assert (
            f'action="/novels/{project_id}/assistant/conversations/'
            f'{conversation_id}/delete"'
            in rendered
        )
        assert re.search(
            r'action="/assistant/messages/[a-f0-9]+/branch"',
            rendered,
        )
        assert re.search(
            r'action="/assistant/messages/[a-f0-9]+/regenerate"',
            rendered,
        )
        action = re.search(
            r'action="/assistant/messages/([a-f0-9]+)/revert-auto-commit"',
            rendered,
        )
        assert action
        workbench = client.get(
            f"{project_url}/workbench?chapter_id={chapter_id}"
            f"&conversation_id={conversation_id}"
        )
        assert workbench.status_code == 200
        assert "已自动保存为当前版本" in workbench.text
        user = application.state.database.get_user_by_username(
            "网页对话作者"
        )
        chapter = application.state.database.get_novel_chapter(
            int(user["id"]), project_id, chapter_id
        )
        candidate = application.state.database.get_chapter_version(
            int(user["id"]),
            project_id,
            chapter_id,
            str(chapter["head_version_id"]),
        )
        assert candidate["source"] == "assistant_chat"
        assert candidate["head_version_id"] == candidate["id"]

        spare_conversation_id = (
            application.state.assistant_chat_service.create_conversation(
                user_id=int(user["id"]),
                scope_type="chapter",
                title="待删除对话",
                project_id=project_id,
                novel_chapter_id=chapter_id,
            )
        )
        delete_page = client.get(
            f"{project_url}/workbench?chapter_id={chapter_id}"
            f"&conversation_id={conversation_id}"
        )
        response = client.post(
            f"{project_url}/assistant/conversations/"
            f"{spare_conversation_id}/delete",
            data={
                "csrf": csrf_from(delete_page.text),
                "current_conversation_id": conversation_id,
                "chapter_id": chapter_id,
                "return_view": "body",
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert f"conversation_id={conversation_id}" in response.headers[
            "location"
        ]
        assert (
            application.state.assistant_chat_service.get_conversation(
                user_id=int(user["id"]),
                conversation_id=spare_conversation_id,
            )
            is None
        )
        still_present = application.state.database.get_chapter_version(
            int(user["id"]),
            project_id,
            chapter_id,
            str(candidate["id"]),
        )
        assert still_present["head_version_id"] == still_present["id"]


def test_edit_and_regenerate_create_branches_without_mutating_history(
    tmp_path: Path,
):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id, project_id, chapter_id, _version_id, _content = seed_novel(
        database, tmp_path
    )
    service = AssistantChatService(
        database, tmp_path / "novels", tmp_path / "documents"
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="chapter",
        title="讨论第二章",
        project_id=project_id,
        novel_chapter_id=chapter_id,
    )

    def complete_turn(question: str) -> None:
        message_id = service.queue_message(
            user_id=user_id,
            conversation_id=conversation_id,
            question=question,
            provider="mock",
            model="mock-creative-chat",
            credential_source="default",
            agent_role="advisor",
        )
        claimed = service.claim_next_message()
        assert claimed and claimed["id"] == message_id
        response = run_agent(service, claimed)
        assert service.complete_message(
            message_id=message_id,
            claim_token=claimed["claim_token"],
            response=response,
        )

    complete_turn("先讨论这一章的悬念入口。")
    complete_turn("把第二个方案说得更具体。")
    original = service.get_conversation(
        user_id=user_id, conversation_id=conversation_id
    )
    assert original
    assert [item["role"] for item in original["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    original_ids = [item["id"] for item in original["messages"]]
    version_count = len(
        database.list_chapter_versions(
            user_id, project_id, chapter_id
        )
    )

    edited = service.branch_conversation_from_message(
        user_id=user_id,
        message_id=str(original["messages"][2]["id"]),
        replacement_question="把第三个方案说得更具体。",
    )
    edited_conversation = service.get_conversation(
        user_id=user_id,
        conversation_id=str(edited["conversation_id"]),
    )
    assert edited["question"] == "把第三个方案说得更具体。"
    assert edited_conversation
    assert [item["content"] for item in edited_conversation["messages"]] == [
        original["messages"][0]["content"],
        original["messages"][1]["content"],
    ]
    assert all(
        item["id"] not in original_ids
        for item in edited_conversation["messages"]
    )

    regenerated = service.branch_conversation_from_message(
        user_id=user_id,
        message_id=str(original["messages"][3]["id"]),
    )
    assert regenerated["question"] == original["messages"][2]["content"]
    assert regenerated["conversation_id"] != edited["conversation_id"]

    unchanged = service.get_conversation(
        user_id=user_id, conversation_id=conversation_id
    )
    assert unchanged
    assert [item["id"] for item in unchanged["messages"]] == original_ids
    assert (
        len(
            database.list_chapter_versions(
                user_id, project_id, chapter_id
            )
        )
        == version_count
    )
    assert service.delete_conversation(
        user_id=user_id,
        conversation_id=str(edited["conversation_id"]),
    )
    assert (
        service.get_conversation(
            user_id=user_id,
            conversation_id=str(edited["conversation_id"]),
        )
        is None
    )
    assert service.get_conversation(
        user_id=user_id, conversation_id=conversation_id
    )
    assert (
        len(
            database.list_chapter_versions(
                user_id, project_id, chapter_id
            )
        )
        == version_count
    )


def test_queued_assistant_message_can_be_cancelled(tmp_path: Path):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id, project_id, _chapter_id, _version_id, _content = seed_novel(
        database, tmp_path
    )
    service = AssistantChatService(
        database, tmp_path / "novels", tmp_path / "documents"
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="project",
        title="取消测试",
        project_id=project_id,
    )
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="先不要开始。",
        provider="mock",
        model="mock-creative-chat",
        credential_source="default",
    )

    result = service.request_message_cancellation(
        user_id=user_id,
        message_id=message_id,
    )

    assert result["cancelled"] is True
    state = service.get_message_stream_state(
        user_id=user_id,
        message_id=message_id,
    )
    assert state
    assert state["status"] == "failed"
    assert state["run_state"] == "cancelled"
    assert state["terminal"] is True
    assert [event["event_type"] for event in state["events"]] == [
        "run.cancelled"
    ]


def test_running_assistant_message_cancellation_is_a_terminal_event(
    tmp_path: Path,
):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id, project_id, _chapter_id, _version_id, _content = seed_novel(
        database, tmp_path
    )
    service = AssistantChatService(
        database, tmp_path / "novels", tmp_path / "documents"
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="project",
        title="运行中取消",
        project_id=project_id,
    )
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="请分析人物动机。",
        provider="mock",
        model="mock-creative-chat",
        credential_source="default",
    )
    claimed = service.claim_next_message()
    assert claimed and claimed["id"] == message_id

    requested = service.request_message_cancellation(
        user_id=user_id,
        message_id=message_id,
    )
    assert requested["run_state"] == "cancelling"
    assert service.is_message_cancel_requested(
        message_id=message_id,
        claim_token=str(claimed["claim_token"]),
    )
    assert service.cancel_running_message(
        message_id=message_id,
        claim_token=str(claimed["claim_token"]),
    )

    state = service.get_message_stream_state(
        user_id=user_id,
        message_id=message_id,
    )
    assert state and state["cancelled"] is True
    assert [event["event_type"] for event in state["events"]][-2:] == [
        "run.cancelling",
        "run.cancelled",
    ]


def test_expired_agent_lease_recovers_and_closes_running_tools(
    tmp_path: Path,
):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id, project_id, _chapter_id, _version_id, _content = seed_novel(
        database, tmp_path
    )
    service = AssistantChatService(
        database, tmp_path / "novels", tmp_path / "documents"
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="project",
        title="租约恢复",
        project_id=project_id,
    )
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="读取作品资料。",
        provider="mock",
        model="mock-creative-chat",
        credential_source="default",
    )
    claimed = service.claim_next_message()
    assert claimed and claimed["id"] == message_id
    call_id = service.start_tool_call(
        message_id=message_id,
        claim_token=str(claimed["claim_token"]),
        sequence=1,
        agent_role="advisor",
        tool_name="read_book_settings",
        tool_label="读取作品设定",
        capability="read_project",
        read_only=True,
        arguments={},
    )
    with database.connection() as connection:
        connection.execute(
            """
            UPDATE assistant_messages
            SET lease_expires_at='2000-01-01T00:00:00+00:00'
            WHERE id=?
            """,
            (message_id,),
        )
        connection.commit()

    recovered = service.claim_next_message()
    assert recovered and recovered["id"] == message_id
    assert recovered["claim_token"] != claimed["claim_token"]
    with database.connection() as connection:
        tool = connection.execute(
            """
            SELECT status, error FROM assistant_tool_calls WHERE id=?
            """,
            (call_id,),
        ).fetchone()
    assert tool["status"] == "failed"
    assert "租约过期" in tool["error"]
    state = service.get_message_stream_state(
        user_id=user_id,
        message_id=message_id,
    )
    assert state
    event_types = [event["event_type"] for event in state["events"]]
    assert "run.recovered" in event_types
    assert event_types[-1] == "run.claimed"
