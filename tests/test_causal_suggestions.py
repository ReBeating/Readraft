from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.causal_suggestion_planner import (
    DEFAULT_CAUSAL_REVIEW_CONTEXT_BUDGET,
    DeepSeekCausalSuggestionPlanner,
    MockCausalSuggestionPlanner,
    compile_causal_review_context,
)
from app.causal_suggestion_schema import CausalSuggestionSet
from app.causal_suggestion_service import CausalSuggestionService
from app.config import Settings
from app.context_compiler import compile_planned_causal_links
from app.db import SCHEMA, Database, utc_now
from app.main import create_app
from app.migrations import MIGRATIONS
from app.planning_schema import (
    ChapterTaskCard,
    allocate_scene_requirement_refs,
)
from app.planning_service import PlanningService
from app.security import hash_password
from app.story_planning_schema import PlannedStoryArc, StoryBlueprint
from app.story_planning_service import StoryPlanningService
from app.story_plan_suggestion_service import StoryPlanSuggestionService
from app.story_structure_service import StoryStructureSuggestionService
from app.structure_link_service import StructureLinkService


def _settings(
    tmp_path: Path,
    *,
    api_key: str | None = None,
) -> Settings:
    return Settings(
        app_name="因果建议测试",
        app_env="test",
        secret_key="test-secret-long-enough",
        data_dir=tmp_path,
        database_path=tmp_path / "web.db",
        cookie_secure=False,
        allow_registration=True,
        max_upload_bytes=1_000_000,
        max_text_chars=1_000_000,
        target_chapter_chars=10_000,
        max_chapter_chars=30_000,
        deepseek_api_key=api_key,
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-chat",
        deepseek_thinking=False,
        deepseek_reasoning_effort="high",
        deepseek_max_tokens=5_000,
        deepseek_connect_timeout_seconds=1,
        deepseek_read_timeout_seconds=1,
        deepseek_max_retries=0,
        worker_poll_seconds=0.01,
        max_jobs_per_day=50,
    )


def _csrf(html: str) -> str:
    match = re.search(r'name="csrf" value="([^"]+)"', html)
    assert match
    return match.group(1)


def _blueprint() -> StoryBlueprint:
    return StoryBlueprint.model_validate(
        {
            "central_question": "父亲为何在失踪后继续寄信？",
            "protagonist_goal": "找到父亲并确认来信投递链。",
            "core_conflict": "林岚追查证据，档案部门持续抹除记录。",
            "stakes": "每次取得证据都会让知情者承担公开风险。",
            "opening_state": "林岚相信父亲已经死亡。",
            "ending_state": "林岚公开证据并理解父亲的选择。",
            "major_turns": [
                "新信迫使林岚回港",
                "档案记录暴露第二名经手人",
                "父亲从受害者变成主动参与者",
            ],
            "must_payoffs": ["解释来信的时间与投递方式"],
            "forbidden_shortcuts": ["不能让新角色口述全部真相"],
            "author_notes": "因果必须由人物行动推动。",
        }
    )


def _arc(title: str, arc_type: str, priority: int) -> PlannedStoryArc:
    return PlannedStoryArc.model_validate(
        {
            "arc_type": arc_type,
            "title": title,
            "dramatic_question": f"{title}最终如何改变林岚的选择？",
            "promise": f"持续给出与{title}有关的可验证变化。",
            "start_state": "只有一项来源不明的线索。",
            "target_payoff": f"用人物行动兑现{title}。",
            "involved_characters": ["林岚", "档案员"],
            "planned_turns": ["确认异常", "旧解释被行动后果推翻"],
            "lifecycle_status": "planned",
            "priority": priority,
        }
    )


def _card(thread: str) -> ChapterTaskCard:
    card = ChapterTaskCard.model_validate(
        {
            "purpose": "用一项可核查行动改变调查局面。",
            "start_state": "人物只有不完整线索。",
            "end_state": "人物的选择产生后续代价。",
            "central_conflict": "证据保全与公开调查冲突。",
            "emotional_value": "行动带来不可撤销的代价。",
            "plot_threads": [thread],
            "must_happen": ["人物采取一项具体行动"],
            "must_preserve": ["不能提前知道完整真相"],
            "forbidden": ["不能直接揭晓全书谜底"],
            "foreshadow_setup": [],
            "foreshadow_payoff": [],
            "ending_hook": "行动后果指向下一项证据。",
            "target_chars": 3000,
            "scenes": [
                {
                    "goal": "取得记录",
                    "obstacle": "记录即将被销毁",
                    "action": "要求封存设备日志",
                    "end_state": "日志被保留下来",
                },
                {
                    "goal": "决定如何公开",
                    "obstacle": "公开会惊动对手",
                    "action": "只公开部分编号",
                    "end_state": "对手被迫改变行动",
                },
            ],
        }
    )
    return allocate_scene_requirement_refs(card)


def _causal_proposal_payload(
    source_id: str,
    *,
    label: str,
) -> dict:
    return {
        "source_chapter_id": source_id,
        "target_chapter_id": "chapter-3",
        "relation_type": "enables",
        "cause_text": "较早章节的行动改变了目标人物可采取的选择。",
        "effect_text": "第三章出现同一个可观察的关系线反转结果。",
        "bridge_purpose": "比较不同人物行动如何触发同一个结果。",
        "source_evidence": ["较早章节记录了一项具体行动。"],
        "target_evidence": ["第三章骨架要求人物关系发生反转。"],
        "risk_if_omitted": "结果可能没有可追溯的行动来源。",
        "confidence": "medium",
        "alternative_label": label,
        "challenge_points": ["证据缺口：尚未证明人物直接感知这项行动。"],
        "missing_intermediate_steps": [],
        "disconfirmation_test": "删除该行动后结果仍成立，就能推翻这项解释。",
        "bridge_readiness": "direct",
        "semantic_checks": [
            {
                "category": category,
                "status": "uncertain",
                "finding": "现有冻结证据尚不足以排除这一项语义风险。",
                "evidence_refs": ["evidence-1"],
                "required_resolution": "",
            }
            for category in (
                "canon_consistency",
                "character_knowledge",
                "timeline",
                "world_rules",
                "continuity",
            )
        ],
        "arc_impacts": [],
    }


def _project(
    tmp_path: Path,
    *,
    database: Database | None = None,
    username: str = "causal-review-author",
) -> tuple[Database, int, str, str, list[str]]:
    database = database or Database(tmp_path / f"{username}.db")
    database.initialize()
    existing = database.get_user_by_username(username)
    user_id = (
        int(existing["id"])
        if existing
        else database.create_user(
            username,
            hash_password("password-123"),
        )
    )
    project_id = f"{username}-project"
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="雾港来信",
        genre="悬疑",
        premise="林岚追查失踪父亲署名的新信。",
        world_setting="当代海港。",
        style_guide="克制具体。",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
        planning_horizon=20,
        story_promise="每个关键揭示都能追溯到此前行动。",
        ending_constraint="必须解释全部来信的投递链。",
    )
    story = StoryPlanningService(database)
    story.save_blueprint(
        user_id=user_id,
        project_id=project_id,
        blueprint=_blueprint(),
        confirm=True,
    )
    story.create_arc(
        user_id=user_id,
        project_id=project_id,
        arc=_arc("父亲失踪主线", "main", 5),
        confirm=True,
    )
    story.create_arc(
        user_id=user_id,
        project_id=project_id,
        arc=_arc("档案员关系线", "relationship", 3),
        confirm=True,
    )
    planning = PlanningService(database)
    volume_id = planning.create_volume(
        user_id=user_id,
        project_id=project_id,
        title="第一卷 投递链",
        goal="锁定维护投递链的人。",
        start_state="只有一封新信。",
        end_state="确认投递链仍在运行。",
        major_conflict="档案与证词互相否定。",
        payoff="解释第一层投递方式。",
    )
    chapter_ids: list[str] = []
    roles = ["setup", "escalation", "reversal", "payoff"]
    arcs = [
        "父亲失踪主线",
        "父亲失踪主线",
        "档案员关系线",
        "档案员关系线",
    ]
    for position, (role, arc) in enumerate(zip(roles, arcs), start=1):
        chapter_id = f"{project_id}-chapter-{position}"
        chapter_dir = tmp_path / project_id / chapter_id
        chapter_dir.mkdir(parents=True, exist_ok=True)
        content_path = chapter_dir / "content.txt"
        content_path.write_text("", encoding="utf-8")
        database.add_novel_chapter(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            title=f"第{position}章 线索",
            outline="人物核查记录并让调查状态发生变化。",
            key_points="核查设备日志\n采取不可撤销的下一步行动",
            content_path=content_path,
            volume_id=volume_id,
        )
        with database.connection() as connection:
            connection.execute(
                """
                UPDATE novel_chapters
                SET skeleton_role=?, skeleton_arc_titles_json=?,
                    skeleton_ending_hook=?,
                    skeleton_application_id=?
                WHERE id=?
                """,
                (
                    role,
                    json.dumps([arc], ensure_ascii=False),
                    "新的行动后果迫使人物继续追查。",
                    "window-a" if position <= 2 else "window-b",
                    chapter_id,
                ),
            )
            connection.commit()
        chapter_ids.append(chapter_id)
    return database, user_id, project_id, volume_id, chapter_ids


def _complete(
    service: CausalSuggestionService,
    suggestion_id: str,
) -> CausalSuggestionSet:
    claimed = service.claim_next_suggestion()
    assert claimed and claimed["id"] == suggestion_id
    response = asyncio.run(
        MockCausalSuggestionPlanner().propose(
            context=claimed["context_snapshot"],
            instruction=str(claimed["instruction"]),
            provider_user_id="test-user",
        )
    )
    assert service.complete_suggestion(
        suggestion_id=suggestion_id,
        claim_token=str(claimed["claim_token"]),
        result=response.result,
        raw_response=response.raw_response,
        provider=response.provider,
        model=response.model,
        input_tokens=0,
        output_tokens=0,
    )
    return response.result


def test_causal_suggestion_schema_allows_zero_but_rejects_unsafe_links():
    empty = CausalSuggestionSet.model_validate(
        {
            "analysis_summary": "现有证据不足以建立直接因果。",
            "proposals": [],
            "unresolved_gaps": ["两章只在主题上相似。"],
            "no_proposal_reason": "只有相关性，没有可观察的行动后果。",
        }
    )
    empty.ensure_context_compatible(
        {
            "current_canonical_position": 0,
            "allowed_source_chapters": [],
            "future_chapters": [],
            "active_causal_links": [],
        }
    )
    with pytest.raises(ValueError, match="为什么不应强行"):
        CausalSuggestionSet.model_validate(
            {
                "analysis_summary": "现有证据不足以建立直接因果。",
                "proposals": [],
                "unresolved_gaps": [],
                "no_proposal_reason": "",
            }
        )


def test_causal_suggestion_schema_validates_comparable_explanations():
    payload = {
        "analysis_summary": "同一结果存在两项有证据但尚未证实的可能前因。",
        "proposals": [
            _causal_proposal_payload("chapter-1", label="行动压力解释"),
            _causal_proposal_payload("chapter-2", label="信息暴露解释"),
        ],
        "comparison_groups": [
            {
                "group_id": "target-3",
                "target_chapter_id": "chapter-3",
                "target_outcome": "第三章出现同一个可观察的关系线反转结果。",
                "proposal_indices": [0, 1],
                "compatibility": "uncertain",
                "comparison_summary": "两项解释可能互相替代，也可能共同作用。",
                "decision_question": "哪项行动真正改变了第三章人物的选择？",
            }
        ],
        "unresolved_gaps": [],
        "no_proposal_reason": "",
    }
    result = CausalSuggestionSet.model_validate(payload)
    result.ensure_context_compatible(
        {
            "current_canonical_position": 0,
            "allowed_source_chapters": [
                {"id": "chapter-1", "position": 1},
                {"id": "chapter-2", "position": 2},
                {"id": "chapter-3", "position": 3},
            ],
            "future_chapters": [
                {"id": "chapter-1", "position": 1},
                {"id": "chapter-2", "position": 2},
                {"id": "chapter-3", "position": 3},
            ],
            "active_causal_links": [],
            "evidence_catalog": [
                {
                    "id": "evidence-1",
                    "label": "章节骨架",
                    "value": "冻结章节记录了一项可核对行动。",
                }
            ],
        }
    )
    broken_target = json.loads(json.dumps(payload, ensure_ascii=False))
    broken_target["proposals"][1]["target_chapter_id"] = "chapter-4"
    with pytest.raises(ValueError, match="同一结果章"):
        CausalSuggestionSet.model_validate(broken_target)
    missing_challenge = json.loads(json.dumps(payload, ensure_ascii=False))
    missing_challenge["proposals"][1]["challenge_points"] = []
    with pytest.raises(ValueError, match="反证或证据缺口"):
        CausalSuggestionSet.model_validate(missing_challenge)
    inconsistent_bridge = json.loads(json.dumps(payload, ensure_ascii=False))
    inconsistent_bridge["proposals"][0]["missing_intermediate_steps"] = [
        "人物尚未得知前一章行动的后果。"
    ]
    with pytest.raises(ValueError, match="不能标记为直接可用"):
        CausalSuggestionSet.model_validate(inconsistent_bridge)


def test_causal_review_model_context_is_bounded_without_dropping_endpoints():
    future = []
    for position in range(1, 81):
        future.append(
            {
                "id": f"chapter-{position:03d}-" + "x" * 72,
                "position": position,
                "title": f"第{position}章 " + "标题" * 80,
                "outline": "很长的章节目的。" * 500,
                "key_points": ["很长的关键行动。" * 100] * 6,
                "skeleton_role": (
                    "payoff" if position % 5 == 0 else "escalation"
                ),
                "skeleton_arc_titles": ["主线", "关系线"],
                "skeleton_ending_hook": "很长的章末推动。" * 200,
                "skeleton_application_id": f"window-{position // 20}",
                "confirmed_task_card": {
                    "purpose": "很长的任务卡目的。" * 300,
                    "end_state": "很长的结束状态。" * 300,
                    "must_happen": ["必须发生的行动。" * 100] * 6,
                    "ending_hook": "很长的任务卡钩子。" * 200,
                },
            }
        )
    context = {
        "project": {"title": "长篇测试", "premise": "梗概" * 2000},
        "current_canonical_position": 0,
        "chapter_limit": 80,
        "future_chapters": future,
        "allowed_source_chapters": future,
        "canonical_source_chapters": [],
        "characters": [
            {
                "name": f"人物{index}",
                "role": "角色定位" * 100,
                "traits": "性格" * 500,
                "character_arc": "人物弧光" * 500,
            }
            for index in range(30)
        ],
        "confirmed_story_blueprint": {
            "central_question": "核心悬问" * 1000,
            "major_turns": ["重大转折" * 300] * 12,
        },
        "confirmed_planned_plot_arcs": [],
        "evidence_catalog": [
            {
                "id": f"evidence-{index:03d}",
                "kind": "test",
                "label": f"证据 {index}" + "标签" * 100,
                "value": "需要保留 ID 但可以压缩的长证据。" * 200,
            }
            for index in range(260)
        ],
        "active_causal_links": [],
        "deterministic_observations": [],
    }
    compiled = compile_causal_review_context(context)
    serialized = json.dumps(compiled, ensure_ascii=False)
    assert len(serialized) <= DEFAULT_CAUSAL_REVIEW_CONTEXT_BUDGET
    assert compiled["included_future_chapter_count"] == 80
    assert {
        item["id"] for item in compiled["future_chapters"]
    } == {item["id"] for item in future}
    assert compiled["included_evidence_count"] == 260
    assert {
        item["id"] for item in compiled["evidence_catalog"]
    } == {f"evidence-{index:03d}" for index in range(260)}
    assert compiled["truncated"] is True


def test_mock_causal_reviewer_caps_comparison_groups_in_long_window():
    future = [
        {
            "id": f"long-window-chapter-{position}",
            "position": position,
            "title": f"第{position}章",
            "outline": "人物用可核查行动改变下一步选择。",
            "key_points": ["核查日志并承担行动后果"],
            "skeleton_role": (
                "reversal" if position in {5, 9} else
                "payoff" if position in {6, 10} else
                "escalation"
            ),
            "skeleton_arc_titles": ["父亲失踪主线"],
            "skeleton_ending_hook": "行动后果迫使人物继续追查。",
            "is_canonical": False,
        }
        for position in range(1, 11)
    ]
    evidence_catalog = [
        {
            "id": "project:premise",
            "kind": "project",
            "label": "项目梗概",
            "value": "林岚追查父亲留下的投递链。",
        },
        {
            "id": "project:world_setting",
            "kind": "project",
            "label": "世界规则",
            "value": "当代海港，档案调阅受权限制度约束。",
        },
        {
            "id": "blueprint:core_conflict",
            "kind": "blueprint",
            "label": "全书冲突引擎",
            "value": "调查行动与档案抹除持续冲突。",
        },
        {
            "id": "arc:main-arc:summary",
            "kind": "planned_arc",
            "label": "规划剧情线《父亲失踪主线》",
            "value": "通过人物行动查清投递链。",
        },
        {
            "id": "continuity:active-issue-count",
            "kind": "continuity",
            "label": "当前连续性问题数量",
            "value": '{"active_issue_count":0}',
        },
        *[
            {
                "id": f"chapter:{chapter['id']}:summary",
                "kind": "future_chapter",
                "label": chapter["title"],
                "value": chapter["outline"],
            }
            for chapter in future
        ],
    ]
    context = {
        "current_canonical_position": 0,
        "future_chapters": future,
        "allowed_source_chapters": future,
        "canonical_source_chapters": [],
        "confirmed_planned_plot_arcs": [
            {
                "id": "main-arc",
                "title": "父亲失踪主线",
                "arc_type": "main",
                "lifecycle_status": "planned",
                "start_state": "只有来源不明的新信。",
            }
        ],
        "canonical_memory": {"continuity_issues": []},
        "evidence_catalog": evidence_catalog,
        "active_causal_links": [],
        "deterministic_observations": [],
    }
    response = asyncio.run(
        MockCausalSuggestionPlanner().propose(
            context=context,
            instruction="",
            provider_user_id="long-window-test",
        )
    )
    assert len(response.result.comparison_groups) == 3
    assert len(response.result.proposals) == 6


def test_deepseek_causal_reviewer_retries_and_sends_frozen_context(
    tmp_path: Path,
):
    database, user_id, project_id, _volume_id, _chapters = _project(
        tmp_path,
        username="deepseek-causal-review",
    )
    service = CausalSuggestionService(database)
    with database.connection() as connection:
        context = service._build_context(
            connection,
            user_id=user_id,
            project_id=project_id,
            chapter_limit=20,
        )
    mock_response = asyncio.run(
        MockCausalSuggestionPlanner().propose(
            context=context,
            instruction="",
            provider_user_id="mock",
        )
    )

    async def run():
        planner = DeepSeekCausalSuggestionPlanner(
            _settings(tmp_path, api_key="sk-test-secret")
        )
        payloads = []

        async def fake_post(payload):
            payloads.append(payload)
            content = (
                "{}"
                if len(payloads) == 1
                else mock_response.result.model_dump_json()
            )
            return {
                "choices": [
                    {
                        "message": {"content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 19,
                },
            }

        planner._analyzer._post = fake_post
        try:
            response = await planner.propose(
                context=context,
                instruction="重点核对关系线反转。",
                provider_user_id="stable-user",
            )
        finally:
            await planner.close()
        return response, payloads

    response, payloads = asyncio.run(run())
    assert response.result.proposals
    assert response.result.comparison_groups
    assert len(payloads) == 2
    first_prompt = payloads[0]["messages"][1]["content"]
    assert "重点核对关系线反转" in first_prompt
    assert "第3章 线索" in first_prompt
    assert "只读候选" in first_prompt
    assert "上一次 JSON 未通过" in payloads[1]["messages"][-1]["content"]
    assert response.input_tokens == 22
    assert response.output_tokens == 38


def test_evidence_review_rejects_unknown_evidence_and_arcs(
    tmp_path: Path,
):
    database, user_id, project_id, _volume_id, _chapters = _project(
        tmp_path,
        username="semantic-schema-author",
    )
    service = CausalSuggestionService(database)
    with database.connection() as connection:
        context = service._build_context(
            connection,
            user_id=user_id,
            project_id=project_id,
            chapter_limit=20,
        )
    response = asyncio.run(
        MockCausalSuggestionPlanner().propose(
            context=context,
            instruction="",
            provider_user_id="semantic-schema-test",
        )
    )
    base = response.result.model_dump(mode="json")

    unknown_evidence = json.loads(
        json.dumps(base, ensure_ascii=False)
    )
    unknown_evidence["proposals"][0]["semantic_checks"][0][
        "evidence_refs"
    ] = ["evidence:not-in-frozen-catalog"]
    with pytest.raises(ValueError, match="冻结目录之外"):
        CausalSuggestionSet.model_validate(
            unknown_evidence
        ).ensure_context_compatible(context)

    mismatched_arc = json.loads(json.dumps(base, ensure_ascii=False))
    mismatched_arc["proposals"][0]["arc_impacts"][0][
        "arc_title"
    ] = "并不存在的剧情线标题"
    with pytest.raises(ValueError, match="标题不匹配"):
        CausalSuggestionSet.model_validate(
            mismatched_arc
        ).ensure_context_compatible(context)

    invalid_support = json.loads(json.dumps(base, ensure_ascii=False))
    invalid_support["proposals"][0]["arc_impacts"][0][
        "required_support_chapter_ids"
    ] = [invalid_support["proposals"][0]["target_chapter_id"]]
    with pytest.raises(ValueError, match="严格位于"):
        CausalSuggestionSet.model_validate(
            invalid_support
        ).ensure_context_compatible(context)


def test_semantic_conflict_requires_explicit_author_override(
    tmp_path: Path,
):
    database, user_id, project_id, _volume_id, _chapters = _project(
        tmp_path,
        username="semantic-conflict-author",
    )
    service = CausalSuggestionService(database)
    suggestion_id = service.create_suggestion(
        user_id=user_id,
        project_id=project_id,
        chapter_limit=20,
        instruction="检查人物知情冲突。",
        provider="mock",
        model="mock-causality-reviewer",
        credential_source="default",
    )
    result = _complete(service, suggestion_id)
    proposal_index = result.comparison_groups[0].proposal_indices[0]
    proposal = result.proposals[proposal_index]
    assert proposal.bridge_readiness == "direct"

    changed = result.model_dump(mode="json")
    changed_check = changed["proposals"][proposal_index][
        "semantic_checks"
    ][1]
    changed_check["status"] = "conflict"
    changed_check["finding"] = (
        "现有知情记录明确显示目标人物此时尚未获得起因行动的信息。"
    )
    changed_check["required_resolution"] = (
        "补写可信的信息传递，或调整目标人物在结果章的行动依据。"
    )
    conflicted = CausalSuggestionSet.model_validate(changed)
    item = service.get_suggestion(
        user_id=user_id,
        suggestion_id=suggestion_id,
    )
    conflicted.ensure_context_compatible(item["context_snapshot"])
    with database.connection() as connection:
        connection.execute(
            """
            UPDATE novel_causal_link_suggestions
            SET result_json=?
            WHERE id=?
            """,
            (conflicted.model_dump_json(), suggestion_id),
        )
        connection.commit()

    with pytest.raises(ValueError, match="语义冲突"):
        service.accept_proposal(
            user_id=user_id,
            suggestion_id=suggestion_id,
            proposal_index=proposal_index,
            cause_text=proposal.cause_text,
            effect_text=proposal.effect_text,
            author_note="",
            comparison_confirmed=True,
            semantic_review_confirmed=True,
        )
    assert StructureLinkService(database).list_links(
        user_id=user_id,
        project_id=project_id,
    ) == []

    override_reason = (
        "作者会在第二章补出档案员转交日志编号的信息传递动作。"
    )
    accepted = service.accept_proposal(
        user_id=user_id,
        suggestion_id=suggestion_id,
        proposal_index=proposal_index,
        cause_text=proposal.cause_text,
        effect_text=proposal.effect_text,
        author_note="保留这项跨线解释。",
        comparison_confirmed=True,
        semantic_review_confirmed=True,
        semantic_override_reason=override_reason,
    )
    assert accepted["semantic_conflict_count"] == 1
    links = StructureLinkService(database).list_links(
        user_id=user_id,
        project_id=project_id,
    )
    assert len(links) == 1
    assert "保留这项跨线解释" in links[0]["author_note"]
    assert f"语义冲突覆盖：{override_reason}" in links[0]["author_note"]


def test_author_can_accept_and_dismiss_frozen_causal_candidates(
    tmp_path: Path,
):
    database, user_id, project_id, volume_id, chapters = _project(
        tmp_path
    )
    planning = PlanningService(database)
    for chapter_id, thread in (
        (chapters[1], "父亲失踪主线"),
        (chapters[2], "档案员关系线"),
    ):
        planning.upsert_task_card(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            volume_id=volume_id,
            card=_card(thread),
            confirm=True,
        )
    service = CausalSuggestionService(database)
    suggestion_id = service.create_suggestion(
        user_id=user_id,
        project_id=project_id,
        chapter_limit=20,
        instruction="重点核对跨窗口关系线反转。",
        provider="mock",
        model="mock-causality-reviewer",
        credential_source="default",
        max_jobs_per_day=50,
    )
    result = _complete(service, suggestion_id)
    assert len(result.proposals) >= 2
    assert result.comparison_groups
    assert {
        check.category for check in result.proposals[0].semantic_checks
    } == {
        "canon_consistency",
        "character_knowledge",
        "timeline",
        "world_rules",
        "continuity",
    }
    item = service.get_suggestion(
        user_id=user_id,
        suggestion_id=suggestion_id,
    )
    assert item and item["baseline_changed"] is False
    assert item["result"]["proposals"][0]["cross_line"] is True
    assert item["result"]["comparison_group_count"] >= 1
    assert item["result"]["proposals"][0]["comparison_group"]["size"] == 2
    assert {
        impact["arc_title"]
        for impact in item["result"]["proposals"][0]["arc_impacts"]
    } == {"父亲失踪主线", "档案员关系线"}
    frozen_evidence_ids = {
        evidence["id"]
        for evidence in item["context_snapshot"]["evidence_catalog"]
    }
    assert all(
        reference in frozen_evidence_ids
        for check in result.proposals[0].semantic_checks
        for reference in check.evidence_refs
    )

    proposal = result.proposals[0]
    accepted = service.accept_proposal(
        user_id=user_id,
        suggestion_id=suggestion_id,
        proposal_index=0,
        cause_text="第二章公开部分编号，迫使档案员改变证据处理方式。",
        effect_text="第三章设备日志因此暴露档案员参与了投递链。",
        author_note="作者确认这条跨线桥接。",
        comparison_confirmed=True,
        semantic_review_confirmed=True,
    )
    assert accepted["reset_task_card_count"] == 2
    links = StructureLinkService(database).list_links(
        user_id=user_id,
        project_id=project_id,
    )
    assert links[0]["source_chapter_id"] == proposal.source_chapter_id
    assert links[0]["target_chapter_id"] == proposal.target_chapter_id
    assert planning.get_task_card(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[1],
    )["status"] == "draft"

    after_accept = service.get_suggestion(
        user_id=user_id,
        suggestion_id=suggestion_id,
    )
    assert after_accept["baseline_changed"] is False
    assert (
        after_accept["result"]["proposals"][0]["review"]["decision"]
        == "accepted"
    )
    changed_card_data = _card("父亲失踪主线").model_dump(mode="json")
    changed_card_data["purpose"] = (
        "作者重新确认了另一项行动目标，因此旧因果证据需要重审。"
    )
    planning.upsert_task_card(
        user_id=user_id,
        project_id=project_id,
        chapter_id=proposal.source_chapter_id,
        volume_id=volume_id,
        card=ChapterTaskCard.model_validate(changed_card_data),
        confirm=True,
    )
    assert service.get_suggestion(
        user_id=user_id,
        suggestion_id=suggestion_id,
    )["baseline_changed"] is True
    service.dismiss_proposal(
        user_id=user_id,
        suggestion_id=suggestion_id,
        proposal_index=1,
    )
    reviewed = service.get_suggestion(
        user_id=user_id,
        suggestion_id=suggestion_id,
    )
    assert (
        reviewed["result"]["proposals"][1]["review"]["decision"]
        == "dismissed"
    )
    with pytest.raises(ValueError, match="已经采纳"):
        service.accept_proposal(
            user_id=user_id,
            suggestion_id=suggestion_id,
            proposal_index=0,
            cause_text="重复起因不能再次采纳。",
            effect_text="重复结果不能再次采纳。",
            author_note="",
            comparison_confirmed=True,
            semantic_review_confirmed=True,
        )


def test_comparison_and_intermediate_step_gates_are_authoritative(
    tmp_path: Path,
):
    database, user_id, project_id, _volume_id, _chapters = _project(
        tmp_path,
        username="causal-comparison-gates",
    )
    service = CausalSuggestionService(database)
    suggestion_id = service.create_suggestion(
        user_id=user_id,
        project_id=project_id,
        chapter_limit=20,
        instruction="比较关系线反转的不同前因。",
        provider="mock",
        model="mock-causality-reviewer",
        credential_source="default",
    )
    result = _complete(service, suggestion_id)
    group = result.comparison_groups[0]
    direct_index = group.proposal_indices[0]
    direct = result.proposals[direct_index]
    assert direct.bridge_readiness == "direct"
    with pytest.raises(ValueError, match="比较同一结果"):
        service.accept_proposal(
            user_id=user_id,
            suggestion_id=suggestion_id,
            proposal_index=direct_index,
            cause_text=direct.cause_text,
            effect_text=direct.effect_text,
            author_note="",
        )
    with pytest.raises(ValueError, match="五类语义复核"):
        service.accept_proposal(
            user_id=user_id,
            suggestion_id=suggestion_id,
            proposal_index=direct_index,
            cause_text=direct.cause_text,
            effect_text=direct.effect_text,
            author_note="",
            comparison_confirmed=True,
        )
    assert StructureLinkService(database).list_links(
        user_id=user_id,
        project_id=project_id,
    ) == []

    bridge_index = next(
        index
        for index in group.proposal_indices
        if result.proposals[index].missing_intermediate_steps
    )
    bridge = result.proposals[bridge_index]
    with pytest.raises(ValueError, match="缺少中间步骤"):
        service.accept_proposal(
            user_id=user_id,
            suggestion_id=suggestion_id,
            proposal_index=bridge_index,
            cause_text=bridge.cause_text,
            effect_text=bridge.effect_text,
            author_note="",
            comparison_confirmed=True,
            semantic_review_confirmed=True,
        )
    accepted = service.accept_proposal(
        user_id=user_id,
        suggestion_id=suggestion_id,
        proposal_index=bridge_index,
        cause_text=bridge.cause_text,
        effect_text=bridge.effect_text,
        author_note="将在第二章补出档案员收到压力并销毁记录的传递动作。",
        comparison_confirmed=True,
        semantic_review_confirmed=True,
    )
    assert accepted["changed"] is True
    links = StructureLinkService(database).list_links(
        user_id=user_id,
        project_id=project_id,
    )
    assert len(links) == 1
    assert "补出档案员收到压力" in links[0]["author_note"]
    target_context = database.get_writing_context(
        user_id,
        bridge.target_chapter_id,
    )
    compiled = compile_planned_causal_links(
        target_context,
        usage="write",
    )
    assert "补出档案员收到压力" in (
        compiled["incoming"][0]["author_note"]
    )


def test_external_baseline_change_and_owner_scope_block_acceptance(
    tmp_path: Path,
):
    database, user_id, project_id, _volume_id, chapters = _project(
        tmp_path,
        username="baseline-review-author",
    )
    other_id = database.create_user(
        "other-causal-reviewer",
        hash_password("password-123"),
    )
    service = CausalSuggestionService(database)
    suggestion_id = service.create_suggestion(
        user_id=user_id,
        project_id=project_id,
        chapter_limit=20,
        instruction="",
        provider="mock",
        model="mock-causality-reviewer",
        credential_source="default",
    )
    result = _complete(service, suggestion_id)
    assert result.proposals
    assert service.get_suggestion(
        user_id=other_id,
        suggestion_id=suggestion_id,
    ) is None
    with database.connection() as connection:
        connection.execute(
            """
            UPDATE novel_chapters
            SET skeleton_ending_hook='作者改成了另一项后果'
            WHERE id=?
            """,
            (chapters[1],),
        )
        connection.commit()
    item = service.get_suggestion(
        user_id=user_id,
        suggestion_id=suggestion_id,
    )
    assert item["baseline_changed"] is True
    with pytest.raises(ValueError, match="已经变化"):
        service.accept_proposal(
            user_id=user_id,
            suggestion_id=suggestion_id,
            proposal_index=0,
            cause_text=result.proposals[0].cause_text,
            effect_text=result.proposals[0].effect_text,
            author_note="",
            comparison_confirmed=True,
            semantic_review_confirmed=True,
        )
    with pytest.raises(ValueError, match="不存在"):
        service.dismiss_proposal(
            user_id=other_id,
            suggestion_id=suggestion_id,
            proposal_index=0,
        )


def test_active_causal_review_blocks_other_long_range_planners(
    tmp_path: Path,
):
    database, user_id, project_id, _volume_id, _chapters = _project(
        tmp_path,
        username="causal-task-lock-author",
    )
    service = CausalSuggestionService(database)
    service.create_suggestion(
        user_id=user_id,
        project_id=project_id,
        chapter_limit=20,
        instruction="",
        provider="mock",
        model="mock-causality-reviewer",
        credential_source="default",
    )
    with pytest.raises(ValueError, match="已有一个 AI 任务"):
        StoryPlanSuggestionService(database).create_suggestion(
            user_id=user_id,
            project_id=project_id,
            planning_mode="refine",
            instruction="",
            provider="mock",
            model="mock-story-planner",
            credential_source="default",
        )
    with pytest.raises(ValueError, match="已有一个 AI 任务"):
        StoryStructureSuggestionService(
            database,
            tmp_path / "novels",
        ).create_suggestion(
            user_id=user_id,
            project_id=project_id,
            chapter_count=10,
            instruction="",
            provider="mock",
            model="mock-story-structure-planner",
            credential_source="default",
        )


def test_v20_migrates_v17_links_without_creating_ai_or_adoption_records(
    tmp_path: Path,
):
    path = tmp_path / "legacy-v17.db"
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(SCHEMA)
    connection.execute(
        """
        CREATE TABLE schema_migrations(
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
        """
    )
    applied_at = utc_now()
    for migration in MIGRATIONS[:17]:
        connection.execute("BEGIN IMMEDIATE")
        migration.apply(connection, applied_at)
        connection.execute(
            """
            INSERT INTO schema_migrations(version, name, applied_at)
            VALUES (?, ?, ?)
            """,
            (migration.version, migration.name, applied_at),
        )
        connection.commit()
    connection.execute(
        """
        INSERT INTO users(username, password_hash, created_at)
        VALUES ('legacy-causal', 'hash', ?)
        """,
        (applied_at,),
    )
    user_id = int(
        connection.execute("SELECT last_insert_rowid()").fetchone()[0]
    )
    connection.execute(
        """
        INSERT INTO novel_projects(
            id, user_id, title, genre, premise, created_at, updated_at
        ) VALUES ('legacy-causal-project', ?, '旧因果项目', '悬疑',
                  '已有链接必须保留。', ?, ?)
        """,
        (user_id, applied_at, applied_at),
    )
    for position in (1, 2):
        connection.execute(
            """
            INSERT INTO novel_chapters(
                id, project_id, position, title, content_path,
                created_at, updated_at
            ) VALUES (?, 'legacy-causal-project', ?, ?, ?, ?, ?)
            """,
            (
                f"legacy-chapter-{position}",
                position,
                f"第{position}章",
                f"/tmp/legacy-{position}.txt",
                applied_at,
                applied_at,
            ),
        )
    connection.execute(
        """
        INSERT INTO novel_chapter_causal_links(
            id, project_id, source_chapter_id, target_chapter_id,
            relation_type, cause_text, effect_text, author_note,
            status, created_at, updated_at
        ) VALUES (
            'legacy-link', 'legacy-causal-project',
            'legacy-chapter-1', 'legacy-chapter-2', 'causes',
            '第一章行动改变局面。', '第二章出现可观察后果。', '',
            'active', ?, ?
        )
        """,
        (applied_at, applied_at),
    )
    connection.commit()
    connection.close()

    database = Database(path)
    database.initialize()
    with database.connection() as migrated:
        assert migrated.execute(
            "SELECT MAX(version) FROM schema_migrations"
                ).fetchone()[0] == 28
        assert migrated.execute(
            "SELECT COUNT(*) FROM novel_chapter_causal_links"
        ).fetchone()[0] == 1
        assert migrated.execute(
            "SELECT COUNT(*) FROM novel_causal_link_suggestions"
        ).fetchone()[0] == 0
        assert migrated.execute(
            "SELECT COUNT(*) FROM novel_causal_link_suggestion_reviews"
        ).fetchone()[0] == 0
        assert migrated.execute(
            "SELECT COUNT(*) FROM novel_causal_branch_simulations"
        ).fetchone()[0] == 0
        assert migrated.execute(
            "SELECT COUNT(*) FROM novel_causal_branch_adoptions"
        ).fetchone()[0] == 0
        assert migrated.execute(
            "SELECT COUNT(*) FROM novel_causal_branch_adoption_items"
        ).fetchone()[0] == 0
        assert migrated.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert migrated.execute("PRAGMA foreign_key_check").fetchall() == []


def test_causal_suggestion_web_flow_is_read_only_until_acceptance(
    tmp_path: Path,
):
    application = create_app(_settings(tmp_path))
    with TestClient(application) as client:
        register = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "网页因果审查作者",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": _csrf(register.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        (
            database,
            user_id,
            project_id,
            _volume_id,
            _chapters,
        ) = _project(
            tmp_path,
            database=application.state.database,
            username="网页因果审查作者",
        )
        workbench = client.get(
            f"/novels/{project_id}/workbench"
            "?view=settings&settings_tab=structure"
        )
        assert workbench.status_code == 200
        response = client.post(
            f"/novels/{project_id}/causal-link-suggestions",
            data={
                "chapter_limit": "20",
                "instruction": "重点检查跨线反转。",
                "csrf": _csrf(workbench.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        suggestion_url = response.headers["location"]
        suggestion_id = suggestion_url.rsplit("/", 1)[-1]
        for _ in range(100):
            status_response = client.get(
                f"/api/causal-link-suggestions/{suggestion_id}"
            )
            assert status_response.status_code == 200
            if status_response.json()["terminal"]:
                break
            time.sleep(0.02)
        page = client.get(suggestion_url)
        assert page.status_code == 200
        assert "建议不是正史，也不是已经生效的计划" in page.text
        assert "同一结果的 2 种可能前因" in page.text
        assert "反证 / 证据缺口" in page.text
        assert "五类语义预检" in page.text
        assert "跨剧情线联合影响" in page.text
        assert "模型预检，不是自动事实判决" in page.text
        assert "确认并建立因果链接" in page.text
        assert StructureLinkService(database).list_links(
            user_id=user_id,
            project_id=project_id,
        ) == []

        suggestion = CausalSuggestionService(database).get_suggestion(
            user_id=user_id,
            suggestion_id=suggestion_id,
        )
        proposal = suggestion["result"]["proposals"][0]
        response = client.post(
            f"/causal-link-suggestions/{suggestion_id}/proposals/0/accept",
            data={
                "cause_text": proposal["cause_text"],
                "effect_text": proposal["effect_text"],
                "author_note": "网页确认。",
                "confirm_comparison": "yes",
                "confirm_semantic_review": "yes",
                "confirm_changes": "yes",
                "csrf": _csrf(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        accepted_page = client.get(response.headers["location"])
        assert "候选已由你确认并建立为未来因果链接" in accepted_page.text
        assert len(
            StructureLinkService(database).list_links(
                user_id=user_id,
                project_id=project_id,
            )
        ) == 1
        task_page = client.get(
            f"/novels/{project_id}/chapters/"
            f"{proposal['target_chapter_id']}/task-card"
        )
        assert task_page.status_code == 200
        assert "作者判断：</b>网页确认。" in task_page.text
