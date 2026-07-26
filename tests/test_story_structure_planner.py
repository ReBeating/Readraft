from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings
from app.db import SCHEMA, Database, utc_now
from app.main import create_app
from app.migrations import MIGRATIONS
from app.planning_schema import (
    ChapterTaskCard,
    allocate_scene_requirement_refs,
)
from app.planning_service import PlanningService
from app.scene_service import SceneService
from app.security import hash_password
from app.story_planning_schema import PlannedStoryArc, StoryBlueprint
from app.story_planning_service import StoryPlanningService
from app.story_structure_planner import (
    DeepSeekStoryStructurePlanner,
    MockStoryStructurePlanner,
)
from app.story_structure_schema import (
    AuthorChapterSkeleton,
    StoryStructureProposalSet,
)
from app.story_structure_service import StoryStructureSuggestionService


def _settings(tmp_path: Path, *, api_key: str | None = None) -> Settings:
    return Settings(
        app_name="novelAI 测试",
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
        deepseek_api_key=api_key,
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-v4-flash",
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
            "central_question": "父亲为何仍能寄出新信？",
            "protagonist_goal": "林岚要找到父亲并确认来信来源。",
            "core_conflict": "档案部门持续抹除她取得的证据。",
            "stakes": "每次推进都会让一名证人失去公开作证的机会。",
            "opening_state": "林岚相信父亲已经死亡并拒绝回港。",
            "ending_state": "林岚公开完整证据并承担父亲选择的后果。",
            "major_turns": ["新信迫使她回港", "中段证词整体失效"],
            "must_payoffs": ["解释全部来信的投递方式"],
            "forbidden_shortcuts": ["不能让新角色口述全部真相"],
            "author_notes": "",
        }
    )


def _main_arc() -> PlannedStoryArc:
    return PlannedStoryArc.model_validate(
        {
            "arc_type": "main",
            "title": "父亲失踪主线",
            "dramatic_question": "林岚能否查明父亲失踪的完整因果？",
            "promise": "持续给出可核查证据和行动后果。",
            "start_state": "只有一封来源不明的新信。",
            "target_payoff": "完整解释父亲失踪与新信投递方式。",
            "involved_characters": ["林岚"],
            "planned_turns": ["确认邮戳异常", "旧证词被推翻"],
            "lifecycle_status": "active",
            "priority": 5,
            "author_notes": "",
        }
    )


def _confirmed_card() -> ChapterTaskCard:
    card = ChapterTaskCard.model_validate(
        {
            "purpose": "林岚核对旧档案并确定下一位证人。",
            "start_state": "林岚只有来信和邮戳。",
            "end_state": "林岚找到被删改的登记编号。",
            "central_conflict": "档案员拒绝提供原始记录。",
            "emotional_value": "怀疑转为主动追查。",
            "plot_threads": ["父亲失踪主线"],
            "must_happen": ["找到被删改的登记编号"],
            "must_preserve": ["林岚尚不知道父亲主动失踪"],
            "forbidden": ["不能直接揭晓全部投递方式"],
            "foreshadow_setup": ["登记编号末位重复"],
            "foreshadow_payoff": [],
            "ending_hook": "登记编号指向已经封闭的旧码头。",
            "target_chars": 3000,
            "scenes": [
                {
                    "pov_character": "林岚",
                    "goal": "查到原始登记",
                    "obstacle": "档案员拒绝",
                    "action": "比对纸本与系统记录",
                    "reveal": "编号被修改",
                    "conceal": "修改者身份未知",
                    "subtext": "",
                    "location": "档案馆",
                    "key_items": ["登记册"],
                    "end_state": "确认记录异常",
                    "transition": "转向旧码头",
                },
                {
                    "pov_character": "林岚",
                    "goal": "确认编号对应地点",
                    "obstacle": "地图已更新",
                    "action": "对照旧地图",
                    "reveal": "编号指向旧码头",
                    "conceal": "投递人未知",
                    "subtext": "",
                    "location": "资料室",
                    "key_items": ["旧地图"],
                    "end_state": "确定下一目标",
                    "transition": "前往旧码头",
                },
            ],
        }
    )
    return allocate_scene_requirement_refs(card)


def _make_canonical(
    database: Database,
    *,
    user_id: int,
    project_id: str,
    chapter_id: str,
) -> str:
    chapter = database.get_novel_chapter(
        user_id, project_id, chapter_id
    )
    content = "正史" * 1050
    version_path = Path(chapter["content_path"]).parent / "canon.txt"
    version_path.write_text(content, encoding="utf-8")
    version_id = database.record_manual_chapter_version(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        version_path=version_path,
        char_count=len(content),
        effective_char_count=len(content),
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )
    accepted = database.accept_chapter_version(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        version_id=str(version_id),
        override_reason="测试中由作者确认正史",
    )
    assert accepted
    return str(version_id)


def _project_with_confirmed_plan(
    tmp_path: Path,
) -> tuple[Database, int, str, str, str]:
    database = Database(tmp_path / "structure.db")
    database.initialize()
    user_id = database.create_user(
        "structure-owner", hash_password("password-123")
    )
    project_id = "structure-project"
    project_root = (
        tmp_path / "novels" / str(user_id) / project_id
    )
    (project_root / "chapters").mkdir(parents=True)
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="雾港来信",
        genre="悬疑",
        premise="林岚收到失踪父亲署名的新信。",
        world_setting="当代海港，档案仍以纸本与系统双轨保存。",
        style_guide="克制、具体。",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
        story_promise="逐层揭开一条可核对的现实谜团。",
        target_audience="偏好现实悬疑的成年读者。",
        core_appeal="证据链与人物选择同步推进。",
        ending_constraint="父亲失踪必须得到完整因果解释。",
        planning_horizon=20,
    )
    planning = PlanningService(database)
    volume_id = planning.create_volume(
        user_id=user_id,
        project_id=project_id,
        title="第一卷 回到雾港",
        goal="让林岚确认新信不是恶作剧并主动进入调查。",
        start_state="林岚拒绝相信父亲仍留下行动痕迹。",
        end_state="林岚确认有人持续维护父亲留下的投递链。",
        major_conflict="追查证据会破坏林岚当前稳定生活。",
        payoff="确认新信真实，并锁定第一个可调查地点。",
    )

    def add_chapter(chapter_id: str, title: str, outline: str) -> None:
        chapter_dir = project_root / "chapters" / chapter_id
        (chapter_dir / "versions").mkdir(parents=True)
        content_path = chapter_dir / "content.txt"
        content_path.write_text("", encoding="utf-8")
        database.add_novel_chapter(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            title=title,
            outline=outline,
            key_points="原关键点",
            content_path=content_path,
            volume_id=volume_id,
        )

    first_id = "structure-chapter-1"
    second_id = "structure-chapter-2"
    add_chapter(first_id, "第一章 迟到的信", "林岚确认邮戳异常。")
    add_chapter(second_id, "第二章 旧档案", "林岚查询旧档案。")
    _make_canonical(
        database,
        user_id=user_id,
        project_id=project_id,
        chapter_id=first_id,
    )
    planning.upsert_task_card(
        user_id=user_id,
        project_id=project_id,
        chapter_id=second_id,
        volume_id=volume_id,
        card=_confirmed_card(),
        confirm=True,
    )
    story_planning = StoryPlanningService(database)
    story_planning.save_blueprint(
        user_id=user_id,
        project_id=project_id,
        blueprint=_blueprint(),
        confirm=True,
    )
    story_planning.create_arc(
        user_id=user_id,
        project_id=project_id,
        arc=_main_arc(),
        confirm=True,
    )
    return database, user_id, project_id, first_id, second_id


def _complete(
    service: StoryStructureSuggestionService, suggestion_id: str
) -> StoryStructureProposalSet:
    item = service.claim_next_suggestion()
    assert item and item["id"] == suggestion_id
    response = asyncio.run(
        MockStoryStructurePlanner().propose(
            context=item["context_snapshot"],
            instruction=item["instruction"],
            provider_user_id="test-user",
        )
    )
    assert service.complete_suggestion(
        suggestion_id=suggestion_id,
        claim_token=item["claim_token"],
        result=response.result,
        raw_response=response.raw_response,
        provider=response.provider,
        model=response.model,
        input_tokens=0,
        output_tokens=0,
    )
    return response.result


def test_structure_schema_requires_three_context_compatible_options():
    context = {
        "project": {"title": "雾港来信"},
        "characters": [{"name": "林岚"}],
        "current_canonical_position": 4,
        "requested_chapter_count": 10,
        "allowed_plot_arcs": [
            {"title": "父亲失踪主线", "arc_type": "main"},
            {"title": "搭档关系线", "arc_type": "relationship"},
        ],
        "allowed_volume_start_positions": [2],
        "locked_volume": {},
    }
    response = asyncio.run(
        MockStoryStructurePlanner().propose(
            context=context,
            instruction="",
            provider_user_id="schema-test",
        )
    )
    result = response.result
    assert len(result.options) == 3
    assert all(len(option.chapters) == 10 for option in result.options)
    assert all(
        [chapter.position for chapter in option.chapters]
        == list(range(5, 15))
        for option in result.options
    )

    too_few = result.model_dump(mode="json")
    too_few["options"] = too_few["options"][:2]
    with pytest.raises(ValidationError):
        StoryStructureProposalSet.model_validate(too_few)

    bad_arc = result.model_dump(mode="json")
    bad_arc["options"][0]["volumes"][0]["arc_titles"].append("不存在的线")
    invalid = StoryStructureProposalSet.model_validate(bad_arc)
    with pytest.raises(ValueError, match="精确剧情线名称"):
        invalid.ensure_context_compatible(context)

    wrong_count = result.model_dump(mode="json")
    wrong_count["target_chapter_count"] = 11
    with pytest.raises(ValidationError, match="目标数量"):
        StoryStructureProposalSet.model_validate(wrong_count)


def test_deepseek_structure_planner_retries_and_sends_frozen_context(
    tmp_path: Path,
):
    context = {
        "project": {"title": "雾港来信"},
        "characters": [{"name": "林岚"}],
        "current_canonical_position": 1,
        "requested_chapter_count": 10,
        "allowed_plot_arcs": [
            {"title": "父亲失踪主线", "arc_type": "main"}
        ],
        "allowed_volume_start_positions": [1],
        "locked_volume": {},
        "confirmed_story_blueprint": {
            "central_question": "父亲为何寄出新信？"
        },
        "canonical_memory": {
            "story_facts": [{"predicate": "林岚已经回港"}]
        },
    }

    async def run():
        mock_response = await MockStoryStructurePlanner().propose(
            context=context,
            instruction="",
            provider_user_id="mock",
        )
        planner = DeepSeekStoryStructurePlanner(
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
                    "prompt_tokens": 12,
                    "completion_tokens": 24,
                },
            }

        planner._analyzer._post = fake_post
        try:
            response = await planner.propose(
                context=context,
                instruction="前五章尽快建立追查目标",
                provider_user_id="stable-user",
            )
        finally:
            await planner.close()
        return response, payloads

    response, payloads = asyncio.run(run())
    assert len(response.result.options) == 3
    assert len(payloads) == 2
    first_prompt = payloads[0]["messages"][1]["content"]
    assert "父亲为何寄出新信" in first_prompt
    assert "林岚已经回港" in first_prompt
    assert "前五章尽快建立追查目标" in first_prompt
    assert "上一次 JSON 未通过" in payloads[1]["messages"][-1]["content"]
    assert response.input_tokens == 24
    assert response.output_tokens == 48


def test_structure_apply_preserves_canon_resets_task_card_and_reverts(
    tmp_path: Path,
):
    database, user_id, project_id, first_id, second_id = (
        _project_with_confirmed_plan(tmp_path)
    )
    other_id = database.create_user(
        "structure-other", hash_password("password-123")
    )
    service = StoryStructureSuggestionService(
        database, tmp_path / "novels"
    )
    suggestion_id = service.create_suggestion(
        user_id=user_id,
        project_id=project_id,
        chapter_count=10,
        instruction="每卷必须有局部回报",
        provider="mock",
        model="mock-story-structure-planner",
        credential_source="default",
        max_jobs_per_day=50,
    )
    result = _complete(service, suggestion_id)
    assert result.target_chapter_count == 10
    assert service.get_suggestion(
        user_id=other_id, suggestion_id=suggestion_id
    ) is None
    suggestion = service.get_suggestion(
        user_id=user_id, suggestion_id=suggestion_id
    )
    assert len(suggestion["previews"]) == 3
    preview = suggestion["previews"][0]
    assert len(preview["chapters"]["create"]) == 9
    assert len(preview["chapters"]["update"]) == 1
    assert preview["task_cards"]["reset_to_draft"] == [2]
    assert not preview["conflicts"]

    canonical_before = database.get_novel_chapter(
        user_id, project_id, first_id
    )
    future_before = database.get_novel_chapter(
        user_id, project_id, second_id
    )
    task_before = PlanningService(database).get_task_card(
        user_id=user_id,
        project_id=project_id,
        chapter_id=second_id,
    )
    with pytest.raises(ValueError):
        service.apply_suggestion(
            user_id=other_id,
            suggestion_id=suggestion_id,
            option_index=0,
            preview_fingerprint=preview["fingerprint"],
        )
    with pytest.raises(ValueError, match="刷新"):
        service.apply_suggestion(
            user_id=user_id,
            suggestion_id=suggestion_id,
            option_index=0,
            preview_fingerprint="stale-preview",
        )

    applied = service.apply_suggestion(
        user_id=user_id,
        suggestion_id=suggestion_id,
        option_index=0,
        preview_fingerprint=preview["fingerprint"],
    )
    chapters = database.list_novel_chapters(user_id, project_id)
    assert [chapter["position"] for chapter in chapters] == list(
        range(1, 12)
    )
    canonical_after = database.get_novel_chapter(
        user_id, project_id, first_id
    )
    assert (
        canonical_after["canonical_version_id"]
        == canonical_before["canonical_version_id"]
    )
    assert canonical_after["title"] == canonical_before["title"]
    future_after = database.get_novel_chapter(
        user_id, project_id, second_id
    )
    assert future_after["canonical_version_id"] is None
    assert future_after["needs_recheck"] == 1
    assert future_after["plan_status"] == "draft"
    task_after = PlanningService(database).get_task_card(
        user_id=user_id,
        project_id=project_id,
        chapter_id=second_id,
    )
    assert task_after["source"] == "structure_planner"
    assert task_after["plot_threads"] == ["父亲失踪主线"]
    assert database.get_writing_context(
        user_id, second_id
    )["task_card"] is None

    reverted = service.revert_application(
        user_id=user_id,
        application_id=applied["application_id"],
    )
    assert reverted["recovery_path"]
    chapters_after_revert = database.list_novel_chapters(
        user_id, project_id
    )
    assert [chapter["position"] for chapter in chapters_after_revert] == [
        1,
        2,
    ]
    restored_future = database.get_novel_chapter(
        user_id, project_id, second_id
    )
    assert restored_future["title"] == future_before["title"]
    assert restored_future["outline"] == future_before["outline"]
    assert restored_future["needs_recheck"] == future_before["needs_recheck"]
    restored_task = PlanningService(database).get_task_card(
        user_id=user_id,
        project_id=project_id,
        chapter_id=second_id,
    )
    assert restored_task["status"] == "confirmed"
    assert restored_task["source"] == task_before["source"]
    assert restored_task["purpose"] == task_before["purpose"]
    assert len(PlanningService(database).list_volumes(
        user_id=user_id, project_id=project_id
    )) == 1
    assert Path(reverted["recovery_path"]).exists()
    with pytest.raises(ValueError, match="已经撤销"):
        service.revert_application(
            user_id=user_id,
            application_id=applied["application_id"],
        )


def test_structure_application_blocks_changed_confirmed_baseline(
    tmp_path: Path,
):
    database, user_id, project_id, _, _ = (
        _project_with_confirmed_plan(tmp_path)
    )
    service = StoryStructureSuggestionService(
        database, tmp_path / "novels"
    )
    suggestion_id = service.create_suggestion(
        user_id=user_id,
        project_id=project_id,
        chapter_count=10,
        instruction="",
        provider="mock",
        model="mock-story-structure-planner",
        credential_source="default",
    )
    _complete(service, suggestion_id)
    planning = StoryPlanningService(database)
    changed = _blueprint().model_copy(
        update={"core_conflict": "新的确认冲突引擎会改变未来结构。"}
    )
    planning.save_blueprint(
        user_id=user_id,
        project_id=project_id,
        blueprint=changed,
        confirm=True,
    )
    suggestion = service.get_suggestion(
        user_id=user_id, suggestion_id=suggestion_id
    )
    assert suggestion["application_blocker"]
    assert "确认蓝图" in suggestion["application_blocker"]
    with pytest.raises(ValueError, match="重新生成滚动结构"):
        service.apply_suggestion(
            user_id=user_id,
            suggestion_id=suggestion_id,
            option_index=0,
            preview_fingerprint="no-longer-valid",
        )


def test_author_can_edit_future_skeleton_without_touching_canon(
    tmp_path: Path,
):
    database, user_id, project_id, first_id, second_id = (
        _project_with_confirmed_plan(tmp_path)
    )
    planning = PlanningService(database)
    before_canon = database.get_novel_chapter(
        user_id, project_id, first_id
    )
    before_task = planning.get_task_card(
        user_id=user_id,
        project_id=project_id,
        chapter_id=second_id,
    )
    scene = before_task["scenes"][0]
    scene_path = (
        Path(
            database.get_novel_chapter(
                user_id, project_id, second_id
            )["content_path"]
        ).parent
        / "scenes"
        / str(scene["id"])
        / "versions"
        / "author-draft.txt"
    )
    scene_path.parent.mkdir(parents=True)
    scene_content = "林岚在档案馆逐页核对旧登记。"
    scene_path.write_text(scene_content, encoding="utf-8")
    SceneService(database).record_manual_version(
        user_id=user_id,
        project_id=project_id,
        chapter_id=second_id,
        scene_beat_id=str(scene["id"]),
        version_path=scene_path,
        content=scene_content,
    )
    edited = AuthorChapterSkeleton.model_validate(
        {
            "title": "第二章 被改写的登记",
            "structural_role": "reversal",
            "purpose": "林岚发现登记编号经过二次覆盖，原有推断因此失效。",
            "key_points": ["确认覆盖痕迹", "锁定第二名经手人"],
            "arc_titles": ["父亲失踪主线"],
            "ending_hook": "经手人的签名出现在父亲失踪后的新文件上。",
        }
    )
    result = planning.update_future_chapter_skeleton(
        user_id=user_id,
        project_id=project_id,
        chapter_id=second_id,
        volume_id=str(before_task["volume_id"]),
        skeleton=edited,
    )
    assert result == {
        "changed": True,
        "task_card_reset": True,
        "stale_scene_count": 1,
    }
    canonical_after = database.get_novel_chapter(
        user_id, project_id, first_id
    )
    assert (
        canonical_after["canonical_version_id"]
        == before_canon["canonical_version_id"]
    )
    assert canonical_after["title"] == before_canon["title"]
    future = database.get_novel_chapter(
        user_id, project_id, second_id
    )
    assert future["title"] == edited.title
    assert future["outline"] == edited.purpose
    assert future["skeleton_role"] == "reversal"
    assert json.loads(future["skeleton_arc_titles_json"]) == [
        "父亲失踪主线"
    ]
    assert future["needs_recheck"] == 1
    task = planning.get_task_card(
        user_id=user_id,
        project_id=project_id,
        chapter_id=second_id,
    )
    assert task["status"] == "draft"
    assert task["source"] == "manual"
    assert task["purpose"] == edited.purpose
    assert task["must_happen"] == edited.key_points
    assert task["ending_hook"] == edited.ending_hook
    assert task["scenes"][0]["draft_status"] == "stale"
    assert database.get_writing_context(
        user_id, second_id
    )["task_card"] is None

    other_id = database.create_user(
        "structure-edit-other", hash_password("password-123")
    )
    with pytest.raises(ValueError, match="章节不存在"):
        planning.update_future_chapter_skeleton(
            user_id=other_id,
            project_id=project_id,
            chapter_id=second_id,
            volume_id=str(before_task["volume_id"]),
            skeleton=edited,
        )
    with pytest.raises(ValueError, match="正史范围"):
        planning.update_future_chapter_skeleton(
            user_id=user_id,
            project_id=project_id,
            chapter_id=first_id,
            volume_id=str(before_task["volume_id"]),
            skeleton=edited,
        )
    invalid_arc = edited.model_copy(
        update={"arc_titles": ["不存在的确认线"]}
    )
    with pytest.raises(ValueError, match="只能引用"):
        planning.update_future_chapter_skeleton(
            user_id=user_id,
            project_id=project_id,
            chapter_id=second_id,
            volume_id=str(before_task["volume_id"]),
            skeleton=invalid_arc,
        )
    assert database.get_novel_chapter(
        user_id, project_id, second_id
    )["title"] == edited.title


def test_author_volume_edit_resets_only_future_task_cards(
    tmp_path: Path,
):
    database, user_id, project_id, first_id, second_id = (
        _project_with_confirmed_plan(tmp_path)
    )
    planning = PlanningService(database)
    volume = planning.list_volumes(
        user_id=user_id, project_id=project_id
    )[0]
    canonical_before = database.get_novel_chapter(
        user_id, project_id, first_id
    )
    result = planning.update_volume(
        user_id=user_id,
        project_id=project_id,
        volume_id=str(volume["id"]),
        title="第一卷 证据回潮",
        goal="让林岚确认档案删改与父亲失踪属于同一条行动链。",
        start_state="林岚只确认新信真实。",
        end_state="林岚锁定第一名仍在维护投递链的人。",
        major_conflict="公开证据和私人证词持续互相否定。",
        payoff="解释第一封新信如何绕开常规邮路。",
    )
    assert result["changed"]
    assert result["affected_chapter_count"] == 1
    assert result["reset_task_card_count"] == 1
    stored_volume = planning.list_volumes(
        user_id=user_id, project_id=project_id
    )[0]
    assert stored_volume["title"] == "第一卷 证据回潮"
    assert stored_volume["canonical_chapter_count"] == 1
    assert stored_volume["future_chapter_count"] == 1
    task = planning.get_task_card(
        user_id=user_id,
        project_id=project_id,
        chapter_id=second_id,
    )
    assert task["status"] == "draft"
    assert task["source"] == "manual"
    canonical_after = database.get_novel_chapter(
        user_id, project_id, first_id
    )
    assert (
        canonical_after["canonical_version_id"]
        == canonical_before["canonical_version_id"]
    )
    assert canonical_after["title"] == canonical_before["title"]
    no_change = planning.update_volume(
        user_id=user_id,
        project_id=project_id,
        volume_id=str(volume["id"]),
        title="第一卷 证据回潮",
        goal="让林岚确认档案删改与父亲失踪属于同一条行动链。",
        start_state="林岚只确认新信真实。",
        end_state="林岚锁定第一名仍在维护投递链的人。",
        major_conflict="公开证据和私人证词持续互相否定。",
        payoff="解释第一封新信如何绕开常规邮路。",
    )
    assert no_change == {
        "changed": False,
        "affected_chapter_count": 0,
        "reset_task_card_count": 0,
        "stale_scene_count": 0,
    }


def test_author_skeleton_edit_prevents_application_undo(
    tmp_path: Path,
):
    database, user_id, project_id, _, second_id = (
        _project_with_confirmed_plan(tmp_path)
    )
    service = StoryStructureSuggestionService(
        database, tmp_path / "novels"
    )
    suggestion_id = service.create_suggestion(
        user_id=user_id,
        project_id=project_id,
        chapter_count=10,
        instruction="",
        provider="mock",
        model="mock-story-structure-planner",
        credential_source="default",
    )
    _complete(service, suggestion_id)
    suggestion = service.get_suggestion(
        user_id=user_id, suggestion_id=suggestion_id
    )
    applied = service.apply_suggestion(
        user_id=user_id,
        suggestion_id=suggestion_id,
        option_index=0,
        preview_fingerprint=suggestion["previews"][0]["fingerprint"],
    )
    planning = PlanningService(database)
    task = planning.get_task_card(
        user_id=user_id,
        project_id=project_id,
        chapter_id=second_id,
    )
    planning.update_future_chapter_skeleton(
        user_id=user_id,
        project_id=project_id,
        chapter_id=second_id,
        volume_id=str(task["volume_id"]),
        skeleton=AuthorChapterSkeleton.model_validate(
            {
                "title": "第二章 作者接管后的骨架",
                "structural_role": "reversal",
                "purpose": "作者改变证据出现顺序，让原有推断在本章失效。",
                "key_points": ["发现覆盖痕迹", "更换下一调查目标"],
                "arc_titles": ["父亲失踪主线"],
                "ending_hook": "新目标与父亲最后一次公开露面重合。",
            }
        ),
    )
    with pytest.raises(ValueError, match="已经继续修改"):
        service.revert_application(
            user_id=user_id,
            application_id=applied["application_id"],
        )
    assert database.get_novel_chapter(
        user_id, project_id, second_id
    )["title"] == "第二章 作者接管后的骨架"


def test_v16_migrates_v15_database_and_preserves_existing_data(
    tmp_path: Path,
):
    path = tmp_path / "legacy-v15.db"
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
    for migration in MIGRATIONS[:15]:
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
        VALUES ('legacy-structure', 'hash', ?)
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
        ) VALUES ('legacy-structure-project', ?, '旧项目', '悬疑',
                  '旧项目资料必须保留。', ?, ?)
        """,
        (user_id, applied_at, applied_at),
    )
    connection.commit()
    connection.close()

    database = Database(path)
    database.initialize()
    assert database.get_novel_project(
        user_id, "legacy-structure-project"
    )["premise"] == "旧项目资料必须保留。"
    with database.connection() as migrated:
        assert migrated.execute(
            "SELECT MAX(version) FROM schema_migrations"
                ).fetchone()[0] == 29
        assert migrated.execute(
            "SELECT COUNT(*) FROM story_structure_suggestions"
        ).fetchone()[0] == 0
        assert migrated.execute(
            "SELECT COUNT(*) FROM story_structure_applications"
        ).fetchone()[0] == 0
        assert migrated.execute(
            "SELECT COUNT(*) FROM novel_chapter_causal_links"
        ).fetchone()[0] == 0
        assert migrated.execute(
            "SELECT COUNT(*) FROM novel_causal_link_suggestions"
        ).fetchone()[0] == 0
        chapter_columns = {
            str(row["name"])
            for row in migrated.execute(
                "PRAGMA table_info(novel_chapters)"
            ).fetchall()
        }
        assert "skeleton_arc_titles_json" in chapter_columns
        assert migrated.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert migrated.execute("PRAGMA foreign_key_check").fetchall() == []


def test_story_structure_web_flow_uses_preview_and_never_creates_canon(
    tmp_path: Path,
):
    application = create_app(_settings(tmp_path))
    with TestClient(application) as client:
        register = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "滚动结构作者",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": _csrf(register.text),
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
                "premise": "林岚收到失踪父亲署名的新信。",
                "story_promise": "逐层揭开可核对的谜团。",
                "ending_constraint": "必须解释全部来信。",
                "planning_horizon": "20",
                "csrf": _csrf(novel_form.text),
            },
            follow_redirects=False,
        )
        workbench_url = response.headers["location"]
        project_id = workbench_url.split("/novels/", 1)[1].split("/", 1)[0]
        project_url = f"/novels/{project_id}"
        workspace = client.get(workbench_url)

        blueprint_data = {
            **_blueprint().model_dump(mode="json"),
            "major_turns": "\n".join(_blueprint().major_turns),
            "must_payoffs": "\n".join(_blueprint().must_payoffs),
            "forbidden_shortcuts": "\n".join(
                _blueprint().forbidden_shortcuts
            ),
            "action": "confirm",
            "csrf": _csrf(workspace.text),
        }
        response = client.post(
            f"{project_url}/story-blueprint",
            data=blueprint_data,
            follow_redirects=False,
        )
        assert response.status_code == 303
        workspace = client.get(workbench_url)
        arc = _main_arc()
        response = client.post(
            f"{project_url}/plot-arcs",
            data={
                **arc.model_dump(mode="json"),
                "involved_characters": "\n".join(
                    arc.involved_characters
                ),
                "planned_turns": "\n".join(arc.planned_turns),
                "action": "confirm",
                "csrf": _csrf(workspace.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        workspace = client.get(workbench_url)
        response = client.post(
            f"{project_url}/story-structure-suggestions",
            data={
                "chapter_count": "10",
                "instruction": "每卷都有局部回报",
                "csrf": _csrf(workspace.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        suggestion_url = response.headers["location"]
        suggestion_id = suggestion_url.rsplit("/", 1)[-1]
        deadline = time.monotonic() + 4
        payload = {}
        while time.monotonic() < deadline:
            payload = client.get(
                f"/api/story-structure-suggestions/{suggestion_id}"
            ).json()
            if payload.get("terminal"):
                break
            time.sleep(0.03)
        assert payload["status"] == "completed"

        comparison = client.get(suggestion_url)
        assert "均衡阶梯加压" in comparison.text
        assert "反转前置重排" in comparison.text
        assert "双线交替汇流" in comparison.text
        assert "EXACT CHANGE PREVIEW" in comparison.text
        match = re.search(
            r'name="preview_fingerprint" value="([^"]+)"',
            comparison.text,
        )
        assert match
        response = client.post(
            f"{suggestion_url}/apply",
            data={
                "option_index": "0",
                "preview_fingerprint": match.group(1),
                "confirm_changes": "yes",
                "csrf": _csrf(comparison.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert (
            "view=settings&settings_tab=structure&saved=true"
            in response.headers["location"]
        )
        database = application.state.database
        user = database.get_user_by_username("滚动结构作者")
        chapters = database.list_novel_chapters(
            int(user["id"]), project_id
        )
        assert len(chapters) == 10
        assert all(
            chapter["canonical_version_id"] is None
            for chapter in chapters
        )
        assert all(chapter["plan_status"] == "draft" for chapter in chapters)
        chapter_page = client.get(
            f"/novels/{project_id}/chapters/{chapters[0]['id']}"
        )
        assert "任务卡草稿" in chapter_page.text
        assert "检查任务卡草稿" in chapter_page.text
        assert "父亲失踪主线" in chapter_page.text
        task_page = client.get(
            f"/novels/{project_id}/chapters/{chapters[0]['id']}/task-card"
        )
        assert "本章在长篇结构中的职责" in task_page.text
        assert "按作者判断修订这章骨架" in task_page.text
        response = client.post(
            f"/novels/{project_id}/chapters/{chapters[0]['id']}/skeleton",
            data={
                "title": "第一章 作者修订的潮线",
                "volume_id": chapters[0]["volume_id"],
                "structural_role": "reversal",
                "purpose": "林岚核对新信后发现第一条可被复查的投递矛盾。",
                "key_points": "确认邮戳来源\n锁定第一位经手人",
                "arc_titles": "父亲失踪主线",
                "ending_hook": "经手人的记录显示父亲失踪后仍签收过文件。",
                "csrf": _csrf(task_page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert "skeleton_saved=true" in response.headers["location"]
        updated_chapter = database.get_novel_chapter(
            int(user["id"]), project_id, str(chapters[0]["id"])
        )
        assert updated_chapter["title"] == "第一章 作者修订的潮线"
        assert updated_chapter["skeleton_role"] == "reversal"
        updated_task = PlanningService(database).get_task_card(
            user_id=int(user["id"]),
            project_id=project_id,
            chapter_id=str(chapters[0]["id"]),
        )
        assert updated_task["status"] == "draft"
        assert updated_task["source"] == "manual"

        workspace = client.get(workbench_url)
        volumes = PlanningService(database).list_volumes(
            user_id=int(user["id"]), project_id=project_id
        )
        response = client.post(
            f"{project_url}/volumes/{volumes[0]['id']}/edit",
            data={
                "title": "第一卷 作者修订卷名",
                "goal": "作者重新定义本卷目标。",
                "start_state": "只有一封新信。",
                "end_state": "锁定投递链维护者。",
                "major_conflict": "证词与档案互相冲突。",
                "payoff": "解释第一层投递方式。",
                "csrf": _csrf(workspace.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert (
            "view=settings&settings_tab=structure&saved=true"
            in response.headers["location"]
        )
        assert PlanningService(database).list_volumes(
            user_id=int(user["id"]), project_id=project_id
        )[0]["title"] == "第一卷 作者修订卷名"
