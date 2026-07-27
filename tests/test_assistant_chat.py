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
    CREATE_CANDIDATE_DRAFT,
    PROPOSE_SETTINGS_PATCH,
    PROPOSE_STORY_PLAN,
    PROPOSE_TEXT_PATCH,
    agent_capabilities,
    agent_manifest,
    resolve_agent_dispatch,
)
from app.agent_loop_schema import (
    AgentDecisionResponse,
    AgentLoopDecision,
    AssistantIntentDecision,
    AssistantIntentResponse,
)
from app.agent_orchestrator import AssistantAgentOrchestrator
from app.assistant_chat import (
    DeepSeekAssistantChatModel,
    JsonAnswerStream,
    MockAssistantChatModel,
    compose_agent_loop_system_prompt,
    compose_assistant_system_prompt,
)
from app.assistant_chat_service import AssistantChatService
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
        deepseek_api_key=None,
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-v4-flash",
        deepseek_thinking=False,
        deepseek_reasoning_effort="high",
        deepseek_max_tokens=5_000,
        deepseek_connect_timeout_seconds=1,
        deepseek_read_timeout_seconds=1,
        deepseek_max_retries=0,
        worker_poll_seconds=0.01,
    )


def csrf_from(html: str) -> str:
    match = re.search(r'name="csrf" value="([^"]+)"', html)
    assert match
    return match.group(1)


class FixedIntentModel(MockAssistantChatModel):
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


def test_assistant_system_prompt_combines_agent_and_book_instructions():
    prompt = compose_assistant_system_prompt(
        book_prompt="本书的冲突必须通过行动显现。",
    )
    assert "只输出一个合法 JSON object" in prompt
    assert "本书的冲突必须通过行动显现" in prompt
    agent_prompt = compose_agent_loop_system_prompt(
        book_prompt="本书先行动后解释。",
        agent_role="writer",
    )
    assert "create_chapter_draft" in agent_prompt
    assert "本书先行动后解释" in agent_prompt
    assert "不能覆盖服务端工具权限" in agent_prompt
    assert "作者明确要求搜索、核实、最新信息或来源" in agent_prompt
    assert "纯构思、创作或改写" in agent_prompt
    assert "查询作品内部设定、章节和故事记忆" in agent_prompt


def test_json_answer_stream_exposes_only_finished_answer_text():
    stream = JsonAnswerStream()
    assert stream.feed('{"action":"call_tool","tool_call":{') is None
    assert (
        stream.feed(
            '"name":"search_web","arguments":{"query":"天气"}},'
            '"answer":null,"citations":[]}'
        )
        is None
    )

    stream = JsonAnswerStream()
    assert stream.feed(
        '{"action":"finish","tool_call":null,"answer":"第一段'
    ) == "第一段"
    assert stream.feed('\\n第二段和一个字：\\u4e2d') == "第一段\n第二段和一个字：中"
    assert stream.feed('","citations":[]}') is None


def test_deepseek_agent_step_streams_sse_answer(tmp_path: Path):
    seen = {}
    decision = json.dumps(
        {
            "action": "finish",
            "tool_call": None,
            "answer": "先核对人物动机。\n再决定是否改写。",
            "citations": [],
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )
    chunks = [
        decision[:35],
        decision[35:68],
        decision[68:],
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content.decode("utf-8"))
        events = [
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {"content": chunk},
                            "finish_reason": (
                                "stop" if index == len(chunks) - 1 else None
                            ),
                        }
                    ]
                }
            )
            for index, chunk in enumerate(chunks)
        ]
        events.append(
            "data: "
            + json.dumps(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 17,
                        "completion_tokens": 11,
                    },
                }
            )
        )
        events.append("data: [DONE]")
        return httpx.Response(
            200,
            content=("\n\n".join(events) + "\n\n").encode(),
            headers={"content-type": "text/event-stream"},
        )

    settings = replace(
        make_settings(tmp_path),
        deepseek_api_key="sk-stream-test",  # pragma: allowlist secret
    )
    model = DeepSeekAssistantChatModel(
        settings,
        transport=httpx.MockTransport(handler),
    )
    updates: list[str] = []

    async def scenario():
        try:
            return await model.next_action(
                context={
                    "scope": "novel_project",
                    "project": {"id": "p1", "title": "测试小说"},
                    "agent": agent_manifest("advisor"),
                },
                history=[],
                question="讨论人物动机",
                selected_quote="",
                available_tools=[],
                observations=[],
                step=1,
                provider_user_id="u_test",
                on_answer_update=updates.append,
            )
        finally:
            await model.close()

    response = asyncio.run(scenario())
    assert response.decision.answer == "先核对人物动机。\n再决定是否改写。"
    assert response.input_tokens == 17
    assert response.output_tokens == 11
    assert updates
    assert updates[-1] == response.decision.answer
    assert seen["payload"]["stream"] is True
    assert seen["payload"]["stream_options"] == {"include_usage": True}


def test_streaming_chat_accepts_provider_json_fallback(tmp_path: Path):
    decision = {
        "action": "finish",
        "tool_call": None,
        "answer": "兼容接口没有返回 SSE，但回复仍可用。",
        "citations": [],
    }

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                decision,
                                ensure_ascii=False,
                            )
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 7,
                },
            },
        )

    settings = replace(
        make_settings(tmp_path),
        deepseek_api_key="sk-json-fallback",  # pragma: allowlist secret
    )
    model = DeepSeekAssistantChatModel(
        settings,
        transport=httpx.MockTransport(handler),
    )
    updates: list[str] = []

    async def scenario():
        try:
            return await model.next_action(
                context={
                    "scope": "novel_project",
                    "project": {"id": "p1", "title": "测试小说"},
                    "agent": agent_manifest("advisor"),
                },
                history=[],
                question="测试兼容接口",
                selected_quote="",
                available_tools=[],
                observations=[],
                step=1,
                provider_user_id="u_test",
                on_answer_update=updates.append,
            )
        finally:
            await model.close()

    response = asyncio.run(scenario())
    assert response.decision.answer == decision["answer"]
    assert updates == [decision["answer"]]
    assert response.input_tokens == 9
    assert response.output_tokens == 7


def test_structured_intent_dispatch_enforces_scope_and_permissions():
    assert PROPOSE_SETTINGS_PATCH in agent_capabilities("planner")
    assert PROPOSE_STORY_PLAN in agent_capabilities("story_planner")
    assert PROPOSE_SETTINGS_PATCH not in agent_capabilities("advisor")
    assert PROPOSE_SETTINGS_PATCH not in agent_capabilities("writer")
    assert PROPOSE_SETTINGS_PATCH not in agent_capabilities("editor")
    assert CREATE_CANDIDATE_DRAFT in agent_capabilities("writer")
    assert PROPOSE_TEXT_PATCH in agent_capabilities("editor")
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
    prerequisite = resolve_agent_dispatch(
        requested_role="auto",
        scope_type="project",
        intent="draft_prose",
        has_quote=False,
        settings_ready=False,
    )
    assert (
        prerequisite.role,
        prerequisite.intent,
        prerequisite.goal,
        prerequisite.settings_prerequisite,
    ) == (
        "planner",
        "update_settings",
        "propose_settings_patch",
        True,
    )


def test_deepseek_agent_step_receives_tool_catalog_not_full_manuscript(
    tmp_path: Path,
):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        seen["payload"] = payload
        decision = {
            "action": "call_tool",
            "tool_call": {
                "name": "read_chapter",
                "arguments": {},
            },
            "answer": None,
            "citations": [],
        }
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                decision, ensure_ascii=False
                            )
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 21,
                    "completion_tokens": 8,
                },
            },
        )

    settings = replace(
        make_settings(tmp_path),
        deepseek_api_key="sk-agent-loop-test",
    )
    model = DeepSeekAssistantChatModel(
        settings, transport=httpx.MockTransport(handler)
    )

    async def scenario():
        try:
            return await model.next_action(
                context={
                    "scope": "novel_chapter",
                    "project": {"id": "p1", "title": "测试小说"},
                    "chapter": {"id": "c1", "title": "第一章"},
                    "current_chapter_excerpt": "绝不能直接进入调度请求的正文",
                    "agent": agent_manifest("writer"),
                    "assistant_boundaries": {
                        "may_modify_canon": False,
                    },
                },
                history=[],
                question="请续写本章",
                selected_quote="",
                available_tools=[
                    {
                        "name": "read_chapter",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                        },
                    }
                ],
                observations=[],
                step=1,
                provider_user_id="u_test",
            )
        finally:
            await model.close()

    response = asyncio.run(scenario())
    assert response.decision.tool_call.name == "read_chapter"
    assert response.input_tokens == 21
    body = seen["payload"]
    assert body["response_format"] == {"type": "json_object"}
    serialized_messages = json.dumps(
        body["messages"], ensure_ascii=False
    )
    assert "read_chapter" in serialized_messages
    assert "绝不能直接进入调度请求的正文" not in serialized_messages


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
        deepseek_api_key="sk-intent-router-test",
    )
    model = DeepSeekAssistantChatModel(
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
            UPDATE novel_chapters SET canonical_version_id=?
            WHERE id=?
            """,
            (version_id, chapter_id),
        )
        connection.execute(
            """
            UPDATE novel_chapter_versions
            SET status='canonical', quality_status='pass'
            WHERE id=?
            """,
            (version_id,),
        )
        connection.commit()
    return user_id, project_id, chapter_id, version_id, content


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
    assert chapter["working_version_id"]
    assert Path(chapter["content_path"]).read_text(encoding="utf-8")


def test_unwritten_project_script_must_propose_settings_before_chapter(
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
    assert response.result.draft is None
    assert response.result.settings_patch is not None
    assert service.complete_message(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        response=response,
    )
    message = service.get_message(
        user_id=user_id, message_id=message_id
    )
    assert message["response"]["agent"]["role"] == "planner"
    assert message["response"]["settings_patch_status"] == "candidate"


def test_writer_cannot_finish_before_creating_requested_draft(
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
        title="必须真正创作",
        project_id=project_id,
        novel_chapter_id=chapter_id,
    )
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="请写出这一章的下一段正文。",
        provider="mock",
        model="mock-creative-chat",
        credential_source="default",
        agent_role="writer",
    )
    claimed = service.claim_next_message()
    assert claimed and claimed["id"] == message_id
    payload = service.build_job_payload(claimed)

    class PrematureFinishModel(MockAssistantChatModel):
        async def next_action(self, **kwargs):
            if not kwargs["available_tools"]:
                raw = {
                    "action": "finish",
                    "tool_call": None,
                    "answer": "候选正文已经创建。",
                    "citations": [],
                }
            elif not kwargs["observations"]:
                raw = {
                    "action": "finish",
                    "tool_call": None,
                    "answer": "我已经写好了。",
                    "citations": [],
                }
            else:
                raw = {
                    "action": "call_tool",
                    "tool_call": {
                        "name": "create_chapter_draft",
                        "arguments": {
                            "mode": "append",
                            "content": "门外的脚步停在第三块地砖上。",
                            "rationale": "把威胁转成可听见的动作。",
                        },
                    },
                    "answer": None,
                    "citations": [],
                }
            decision = AgentLoopDecision.model_validate(raw)
            return AgentDecisionResponse(
                decision=decision,
                raw_response=decision.model_dump_json(),
                input_tokens=1,
                output_tokens=1,
            )

    response = asyncio.run(
        AssistantAgentOrchestrator(service).run(
            model=PrematureFinishModel(),
            item=claimed,
            payload=payload,
            provider_user_id="u_test",
        )
    )

    assert response.result.draft is not None
    assert response.result.draft.content == "门外的脚步停在第三块地砖上。"
    running_message = service.get_message(
        user_id=user_id, message_id=message_id
    )
    assert [
        (item["action"], item["outcome_status"])
        for item in running_message["agent_steps"]
    ] == [
        ("finish", "denied"),
        ("call_tool", "completed"),
        ("finish", "completed"),
    ]
    assert "任务目标尚未完成" in (
        running_message["agent_steps"][0]["error"]
    )


def test_agent_loop_calls_domain_tools_and_auto_commits_working_copy(
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
        question="请续写这一章，让林岚先核对录音来源。",
        provider="mock",
        model="mock-creative-chat",
        credential_source="default",
        agent_role="auto",
        auto_commit=True,
    )
    claimed = service.claim_next_message()
    assert claimed and claimed["id"] == message_id
    payload = service.build_job_payload(claimed)
    assert payload["context"]["agent"]["role"] == "advisor"
    assert payload["context"]["dispatch"] == {
        "requested_role": "auto",
        "resolved_role": "pending",
        "intent": "pending",
        "reason": "model_intent_pending",
        "goal": "classify_intent",
            "settings_ready": True,
            "ui_surface": "chapter",
            "scope_preflight": {},
    }
    orchestrator = AssistantAgentOrchestrator(service)
    response = asyncio.run(
        orchestrator.run(
            model=FixedIntentModel("draft_prose"),
            item=claimed,
            payload=payload,
            provider_user_id="u_test",
        )
    )
    assert response.result.draft is not None
    assert [item["tool_name"] for item in response.agent_trace] == [
        "read_book_settings",
        "read_chapter",
        "create_chapter_draft",
    ]
    assert service.complete_message(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        response=response,
    )
    message = service.get_message(
        user_id=user_id, message_id=message_id
    )
    assert message["response"]["agent"]["role"] == "writer"
    assert message["response"]["auto_commit"]["status"] == "applied"
    assert [item["tool_name"] for item in message["tool_calls"]] == [
        "read_book_settings",
        "read_chapter",
        "create_chapter_draft",
    ]
    assert all(
        item["status"] == "completed" for item in message["tool_calls"]
    )
    assert [
        (item["action"], item["outcome_status"])
        for item in message["agent_steps"]
    ] == [
        ("call_tool", "completed"),
        ("call_tool", "completed"),
        ("call_tool", "completed"),
        ("finish", "completed"),
    ]
    chapter = database.get_novel_chapter(
        user_id, project_id, chapter_id
    )
    assert chapter["canonical_version_id"] == chapter["working_version_id"]
    assert chapter["working_version_id"] != version_id
    working = database.get_chapter_version(
        user_id,
        project_id,
        chapter_id,
        str(chapter["working_version_id"]),
    )
    working_text = Path(working["content_path"]).read_text(
        encoding="utf-8"
    )
    assert working_text.startswith(content)
    assert "雨声先落在窗外" in working_text


def test_agent_loop_records_and_blocks_forbidden_tool_call(
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
        title="只读讨论",
        project_id=project_id,
    )
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="讨论一下开场，不要写正文。",
        provider="mock",
        model="mock-creative-chat",
        credential_source="default",
        agent_role="advisor",
    )
    claimed = service.claim_next_message()
    assert claimed and claimed["id"] == message_id
    payload = service.build_job_payload(claimed)

    class ForbiddenToolModel(MockAssistantChatModel):
        async def next_action(self, **kwargs):
            observations = kwargs["observations"]
            if not observations:
                decision = AgentLoopDecision.model_validate(
                    {
                        "action": "call_tool",
                        "tool_call": {
                            "name": "create_chapter_draft",
                            "arguments": {
                                "mode": "replace",
                                "content": "不应被接受",
                                "rationale": "越权测试",
                            },
                        },
                        "answer": None,
                        "citations": [],
                    }
                )
            else:
                decision = AgentLoopDecision.model_validate(
                    {
                        "action": "finish",
                        "tool_call": None,
                        "answer": "越权工具已被服务端拦截。",
                        "citations": [],
                    }
                )
            return AgentDecisionResponse(
                decision=decision,
                raw_response=decision.model_dump_json(),
                input_tokens=0,
                output_tokens=0,
            )

    response = asyncio.run(
        AssistantAgentOrchestrator(service).run(
            model=ForbiddenToolModel(),
            item=claimed,
            payload=payload,
            provider_user_id="u_test",
        )
    )
    assert response.result.draft is None
    assert service.complete_message(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        response=response,
    )
    message = service.get_message(
        user_id=user_id, message_id=message_id
    )
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0]["status"] == "denied"
    assert message["tool_calls"][0]["tool_name"] == "create_chapter_draft"
    assert message["applied_version_id"] is None


def test_agent_loop_limits_web_search_to_two_attempts(tmp_path: Path):
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
        title="联网预算",
        project_id=project_id,
    )
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="请联网核实三项外部资料。",
        provider="mock",
        model="mock-creative-chat",
        credential_source="default",
        agent_role="advisor",
    )
    claimed = service.claim_next_message()
    assert claimed and claimed["id"] == message_id
    payload = service.build_job_payload(claimed)
    assert payload["context"]["web_search_available"] is True
    searches = []

    class OverSearchingModel(MockAssistantChatModel):
        def __init__(self):
            self.turn = 0

        async def next_action(self, **kwargs):
            self.turn += 1
            if self.turn <= 3 and kwargs["available_tools"]:
                raw = {
                    "action": "call_tool",
                    "tool_call": {
                        "name": "search_web",
                        "arguments": {
                            "query": f"外部资料 {self.turn}",
                            "max_results": 2,
                        },
                    },
                    "answer": None,
                    "citations": [],
                }
            else:
                raw = {
                    "action": "finish",
                    "tool_call": None,
                    "answer": "已根据两次搜索取得的资料完成回答。",
                    "citations": [],
                }
            decision = AgentLoopDecision.model_validate(raw)
            return AgentDecisionResponse(
                decision=decision,
                raw_response=decision.model_dump_json(),
                input_tokens=1,
                output_tokens=1,
            )

    response = asyncio.run(
        AssistantAgentOrchestrator(
            service,
            web_search=lambda _user_id, query, _limit: searches.append(
                query
            )
            or [
                {
                    "title": query,
                    "url": "https://example.com/" + str(len(searches)),
                    "snippet": "外部资料摘要。",
                }
            ],
        ).run(
            model=OverSearchingModel(),
            item=claimed,
            payload=payload,
            provider_user_id="u_test",
        )
    )
    assert searches == ["外部资料 1", "外部资料 2"]
    assert response.result.answer == "已根据两次搜索取得的资料完成回答。"
    running_message = service.get_message(
        user_id=user_id, message_id=message_id
    )
    assert [
        (item["tool_name"], item["status"])
        for item in running_message["tool_calls"]
    ] == [
        ("search_web", "completed"),
        ("search_web", "completed"),
        ("search_web", "denied"),
    ]
    assert "联网搜索次数已达到上限" in (
        running_message["tool_calls"][-1]["error"]
    )


def test_agent_loop_stops_repeated_call_and_forces_final_answer(
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
        title="循环收束",
        project_id=project_id,
    )
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="梳理现在的作品设定。",
        provider="mock",
        model="mock-creative-chat",
        credential_source="default",
        agent_role="advisor",
    )
    claimed = service.claim_next_message()
    assert claimed and claimed["id"] == message_id
    payload = service.build_job_payload(claimed)

    class SlowFinishingModel(MockAssistantChatModel):
        def __init__(self):
            self.seen_tools = []

        async def next_action(self, **kwargs):
            available_tools = kwargs["available_tools"]
            self.seen_tools.append(
                [item["name"] for item in available_tools]
            )
            if available_tools:
                decision = AgentLoopDecision.model_validate(
                    {
                        "action": "call_tool",
                        "tool_call": {
                            "name": "read_book_settings",
                            "arguments": {},
                        },
                        "answer": None,
                        "citations": [],
                    }
                )
            else:
                assert kwargs["step"] == 3
                assert (
                    kwargs["observations"][-1]["status"]
                    == "finalization_required"
                )
                decision = AgentLoopDecision.model_validate(
                    {
                        "action": "finish",
                        "tool_call": None,
                        "answer": "已根据现有资料完成收束。",
                        "citations": [],
                    }
                )
            return AgentDecisionResponse(
                decision=decision,
                raw_response=decision.model_dump_json(),
                input_tokens=1,
                output_tokens=1,
            )

    model = SlowFinishingModel()
    response = asyncio.run(
        AssistantAgentOrchestrator(service).run(
            model=model,
            item=claimed,
            payload=payload,
            provider_user_id="u_test",
        )
    )

    assert response.result.answer == "已根据现有资料完成收束。"
    assert len(model.seen_tools) == 3
    assert model.seen_tools[-1] == []
    assert response.input_tokens == 3
    assert response.output_tokens == 3
    assert len(response.agent_trace) == 2
    assert response.agent_trace[0]["status"] == "completed"
    assert response.agent_trace[1]["status"] == "denied"
    running_message = service.get_message(
        user_id=user_id, message_id=message_id
    )
    assert [
        (item["action"], item["outcome_status"])
        for item in running_message["agent_steps"]
    ] == [
        ("call_tool", "completed"),
        ("call_tool", "denied"),
        ("finish", "completed"),
    ]


def test_agent_loop_returns_fallback_if_model_ignores_forced_finish(
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
        title="异常循环收束",
        project_id=project_id,
    )
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="读取作品设定。",
        provider="mock",
        model="mock-creative-chat",
        credential_source="default",
        agent_role="advisor",
    )
    claimed = service.claim_next_message()
    assert claimed and claimed["id"] == message_id
    payload = service.build_job_payload(claimed)

    class NeverFinishingModel(MockAssistantChatModel):
        async def next_action(self, **kwargs):
            decision = AgentLoopDecision.model_validate(
                {
                    "action": "call_tool",
                    "tool_call": {
                        "name": "read_book_settings",
                        "arguments": {},
                    },
                    "answer": None,
                    "citations": [],
                }
            )
            return AgentDecisionResponse(
                decision=decision,
                raw_response=decision.model_dump_json(),
                input_tokens=0,
                output_tokens=0,
            )

    response = asyncio.run(
        AssistantAgentOrchestrator(service, max_model_turns=2).run(
            model=NeverFinishingModel(),
            item=claimed,
            payload=payload,
            provider_user_id="u_test",
        )
    )
    assert "自动收束" in response.result.answer
    assert "Agent 在" not in response.result.answer
    assert response.result.draft is None


def test_quote_bound_chat_saves_candidate_without_changing_canon(
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
    payload = service.build_job_payload(claimed)
    response = asyncio.run(
        MockAssistantChatModel().reply(
            **payload, provider_user_id="u_test"
        )
    )
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
    assert assistant_message["response"]["citations"][0]["quote"] == selected
    assert "/source?start=0" in assistant_message["response"][
        "citations"
    ][0]["url"]
    assert assistant_message["response"]["rewrite"]

    saved = service.save_rewrite_candidate(
        user_id=user_id, assistant_message_id=message_id
    )
    chapter = database.get_novel_chapter(
        user_id, project_id, chapter_id
    )
    assert chapter["canonical_version_id"] == version_id
    assert chapter["working_version_id"] == saved["version_id"]
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

    second = service.save_rewrite_candidate(
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


def test_advisor_is_read_only_for_text_rewrite(tmp_path: Path):
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
    response = asyncio.run(
        MockAssistantChatModel().reply(
            **service.build_job_payload(claimed),
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
    assert message["response"]["agent"]["role"] == "advisor"
    assert message["response"]["rewrite"] is None


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
    response = asyncio.run(
        MockAssistantChatModel().reply(
            **service.build_job_payload(claimed),
            provider_user_id="u_test",
        )
    )
    assert service.complete_message(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        response=response,
    )
    chapter = database.get_novel_chapter(
        user_id, project_id, chapter_id
    )
    assert chapter["canonical_version_id"] == version_id
    assert Path(chapter["content_path"]).read_text(
        encoding="utf-8"
    ) == content
    message = service.get_message(
        user_id=user_id, message_id=message_id
    )
    assert message["response"]["agent"]["role"] == "writer"
    assert message["response"]["draft"]["mode"] == "append"

    saved = service.save_draft_candidate(
        user_id=user_id, assistant_message_id=message_id
    )
    chapter = database.get_novel_chapter(
        user_id, project_id, chapter_id
    )
    assert chapter["canonical_version_id"] == version_id
    assert chapter["working_version_id"] == saved["version_id"]
    candidate = database.get_chapter_version(
        user_id, project_id, chapter_id, saved["version_id"]
    )
    assert candidate["kind"] == "assistant_draft"
    assert candidate["status"] == "candidate"
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
    response = asyncio.run(
        MockAssistantChatModel().reply(
            **service.build_job_payload(claimed),
            provider_user_id="u_test",
        )
    )
    assert service.complete_message(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        response=response,
    )
    chapter = database.get_novel_chapter(
        user_id, project_id, chapter_id
    )
    auto_version_id = str(chapter["working_version_id"])
    assert auto_version_id != version_id
    assert chapter["canonical_version_id"] == auto_version_id
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
    assert chapter["canonical_version_id"] == reverted["version_id"]
    assert chapter["working_version_id"] == reverted["version_id"]
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
    response = asyncio.run(
        MockAssistantChatModel().reply(
            **service.build_job_payload(claimed),
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
    assert message["response"]["auto_commit"]["status"] == "applied"
    assert message["applied_version_id"]
    chapter = database.get_novel_chapter(
        user_id, project_id, chapter_id
    )
    assert chapter["canonical_version_id"] == message["applied_version_id"]
    changed = Path(chapter["content_path"]).read_text(encoding="utf-8")
    assert changed != content
    assert "动作停在答案之前" in changed


def test_reference_chapter_chat_uses_source_but_cannot_create_rewrite(
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
    response = asyncio.run(
        MockAssistantChatModel().reply(
            **payload, provider_user_id="u_test"
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
    assert message["response"]["citations"][0]["quote"] == selected
    assert message["response"]["agent"]["role"] == "researcher"
    assert message["response"]["rewrite"] is None
    with pytest.raises(ValueError, match="没有可保存"):
        service.save_rewrite_candidate(
            user_id=user_id, assistant_message_id=message_id
        )


def test_novel_chat_web_flow_and_rewrite_action(tmp_path: Path):
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
        chapter_page = client.get(chapter_url)
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
        saved_page = client.get(chapter_url)
        assert "共创对话" in saved_page.text
        assert 'class="studio-manuscript-view"' in saved_page.text
        assert "引用正文问 AI" not in saved_page.text

        assistant_url = (
            f"{project_url}/assistant?new=true&chapter_id={chapter_id}"
        )
        assistant_redirect = client.get(
            assistant_url, follow_redirects=False
        )
        assert assistant_redirect.status_code == 303
        assert (
            assistant_redirect.headers["location"]
            == f"{project_url}/workbench?chapter_id={chapter_id}"
        )
        assistant_page = client.get(
            assistant_redirect.headers["location"]
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
                "局部修订提交" in rendered
                and "已自动保存为当前版本" in rendered
            ):
                break
            time.sleep(0.03)
        assert "回答依据" in rendered
        assert "局部修订提交" in rendered
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
        source_link = re.search(
            r'href="([^"]+/source\?start=0&amp;end=\d+)"',
            rendered,
        )
        assert source_link
        source_page = client.get(source_link.group(1).replace("&amp;", "&"))
        assert source_page.status_code == 200
        assert "cited-selection" in source_page.text

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
            str(chapter["working_version_id"]),
        )
        assert candidate["source"] == "assistant_chat"
        assert candidate["status"] == "canonical"

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
        assert still_present["status"] == "canonical"


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
        response = asyncio.run(
            MockAssistantChatModel().reply(
                **service.build_job_payload(claimed),
                provider_user_id="u_test",
            )
        )
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
