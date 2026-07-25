import asyncio
from pathlib import Path

import pytest

from app.db import Database
from app.planning_ai import MockChapterPlanner
from app.planning_schema import (
    ChapterTaskCard,
    chapter_task_card_fingerprint,
)
from app.planning_service import PlanningService
from app.security import hash_password
from app.writing import build_writing_messages


def test_confirmed_task_card_and_scene_beats_enter_writer_context(
    tmp_path: Path,
):
    database = Database(tmp_path / "planning.db")
    database.initialize()
    user_id = database.create_user(
        "planner", hash_password("password-123")
    )
    project_id = "project-planning"
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="雾港来信",
        genre="悬疑",
        premise="记者返回故乡调查父亲失踪案。",
        world_setting="当代海港小城。",
        style_guide="克制、具体。",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
        story_promise="每卷推进一层可核对的现实谜团。",
        target_audience="偏好慢热现实悬疑的读者",
        core_appeal="证据推理与压抑的亲情",
        ending_constraint="必须解释父亲失踪与来信来源。",
        planning_horizon=20,
    )
    database.add_novel_character(
        user_id=user_id,
        project_id=project_id,
        name="林岚",
        role="调查记者",
        traits="冷静、执拗",
        background="父亲十年前失踪",
        character_arc="重新面对故乡",
    )
    planning = PlanningService(database)
    volume_id = planning.create_volume(
        user_id=user_id,
        project_id=project_id,
        title="第一卷 回到雾港",
        goal="确认来信并非恶作剧。",
        start_state="林岚拒绝回忆故乡。",
        end_state="林岚主动进入父亲失踪案。",
        major_conflict="现实证据与十年前结论冲突。",
        payoff="确认父亲失踪案仍在被人操控。",
    )
    chapter_id = "chapter-planning"
    chapter_dir = tmp_path / "chapter"
    chapter_dir.mkdir()
    content_path = chapter_dir / "content.txt"
    content_path.write_text("", encoding="utf-8")
    database.add_novel_chapter(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        title="第一章 迟到的信",
        outline="林岚收到来信。",
        key_points="核对邮戳",
        content_path=content_path,
        volume_id=volume_id,
    )

    card = ChapterTaskCard.model_validate(
        {
            "purpose": "迫使林岚返回雾港。",
            "start_state": "林岚认定父亲已经死亡。",
            "end_state": "林岚踏上返回雾港的列车。",
            "central_conflict": "近期邮戳与父亲失踪十年的事实冲突。",
            "emotional_value": "被压下的希望重新出现。",
            "plot_threads": ["父亲失踪之谜"],
            "must_happen": ["确认邮戳", "购买车票"],
            "must_preserve": ["林岚不知道寄信人身份"],
            "forbidden": ["揭晓父亲下落"],
            "foreshadow_setup": ["信纸有海水气味"],
            "foreshadow_payoff": [],
            "ending_hook": "信纸带着只有雾港旧码头才有的气味。",
            "target_chars": 3000,
            "scenes": [
                {
                    "pov_character": "林岚",
                    "goal": "确认来信真假",
                    "obstacle": "邮戳日期不可能",
                    "action": "对照父亲旧信并检查邮戳",
                    "reveal": "笔迹一致",
                    "conceal": "寄信人身份",
                    "subtext": "",
                    "location": "林岚住处",
                    "key_items": ["来信", "父亲旧信"],
                    "end_state": "林岚承认信件值得调查",
                    "transition": "查询当夜车票",
                    "requirement_refs": [
                        {
                            "kind": "plot_thread",
                            "text": "父亲失踪之谜",
                        },
                        {
                            "kind": "must_happen",
                            "text": "确认邮戳",
                        },
                        {
                            "kind": "foreshadow_setup",
                            "text": "信纸有海水气味",
                        },
                    ],
                },
                {
                    "pov_character": "林岚",
                    "goal": "决定是否返回雾港",
                    "obstacle": "她抗拒回到故乡",
                    "action": "购买车票并收起旧信",
                    "reveal": "信纸带有海水气味",
                    "conceal": "气味来源",
                    "subtext": "",
                    "location": "车站",
                    "key_items": ["车票", "来信"],
                    "end_state": "林岚踏上列车",
                    "transition": "列车驶入雾区",
                    "requirement_refs": [
                        {
                            "kind": "must_happen",
                            "text": "购买车票",
                        },
                        {
                            "kind": "ending_hook",
                            "text": "信纸带着只有雾港旧码头才有的气味。",
                        },
                    ],
                },
            ],
        }
    )
    plan_id = planning.upsert_task_card(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        volume_id=volume_id,
        card=card,
        confirm=True,
    )
    assert plan_id
    stored = planning.get_task_card(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
    )
    assert stored["status"] == "confirmed"
    assert len(stored["scenes"]) == 2
    assert stored["scenes"][0]["requirement_refs"][1] == {
        "kind": "must_happen",
        "text": "确认邮戳",
    }
    assert stored["forbidden"] == ["揭晓父亲下落"]
    rolling = planning.get_rolling_plan(
        user_id=user_id, project_id=project_id
    )
    assert rolling["horizon"] == 20
    assert rolling["planned_count"] == 1
    assert rolling["confirmed_count"] == 1
    assert rolling["missing_count"] == 19

    context = database.get_writing_context(user_id, chapter_id)
    assert context["task_card"]["status"] == "confirmed"
    assert context["chapter"]["volume_title"] == "第一卷 回到雾港"
    messages = build_writing_messages(
        context=context,
        operation="draft",
        instruction="",
        current_content="",
        previous_content="",
    )
    prompt = str(messages[1]["content"])
    assert "每卷推进一层可核对的现实谜团" in prompt
    assert "确认来信真假" in prompt
    assert "揭晓父亲下落" in prompt
    assert "requirement_refs" in prompt
    assert "<confirmed_chapter_task_card>" in prompt


def test_task_card_cannot_be_confirmed_without_two_scenes(tmp_path: Path):
    card = ChapterTaskCard.model_validate(
        {
            "purpose": "推进主线",
            "start_state": "开始",
            "end_state": "结束",
            "central_conflict": "冲突",
            "ending_hook": "新问题",
            "scenes": [
                {
                    "goal": "目标",
                    "obstacle": "阻力",
                    "action": "行动",
                    "end_state": "变化",
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="至少 2 个场景节拍"):
        card.ensure_confirmable()


def test_explicit_scene_mapping_must_cover_every_task_requirement():
    card = ChapterTaskCard.model_validate(
        {
            "purpose": "推进主线",
            "start_state": "尚未取得证据",
            "end_state": "决定继续调查",
            "central_conflict": "证据与旧结论冲突",
            "plot_threads": ["失踪之谜"],
            "must_happen": ["核对邮戳"],
            "foreshadow_setup": ["信纸有盐味"],
            "ending_hook": "旧码头出现同样信纸。",
            "scenes": [
                {
                    "goal": "核对邮戳",
                    "obstacle": "记录缺失",
                    "action": "查询档案",
                    "end_state": "确认日期异常",
                    "requirement_refs": [
                        {
                            "kind": "must_happen",
                            "text": "核对邮戳",
                        }
                    ],
                },
                {
                    "goal": "追查信纸来源",
                    "obstacle": "商店拒绝提供记录",
                    "action": "比对纸张",
                    "end_state": "线索指向旧码头",
                },
            ],
        }
    )
    with pytest.raises(ValueError, match="没有分配到任何场景"):
        card.ensure_confirmable()


def test_scene_only_mock_planner_preserves_locked_card_and_maps_all_requirements():
    card = ChapterTaskCard.model_validate(
        {
            "purpose": "迫使林岚回到雾港。",
            "start_state": "林岚拒绝相信来信。",
            "end_state": "林岚买下车票。",
            "central_conflict": "新邮戳与旧结论冲突。",
            "emotional_value": "希望与戒备同时抬升。",
            "plot_threads": ["父亲失踪之谜"],
            "must_happen": ["确认邮戳", "购买车票"],
            "must_preserve": ["不知道寄信人"],
            "forbidden": ["揭晓父亲下落"],
            "foreshadow_setup": ["信纸有海水气味"],
            "foreshadow_payoff": [],
            "ending_hook": "气味来自雾港旧码头。",
            "target_chars": 3000,
            "scenes": [],
        }
    )

    response = asyncio.run(
        MockChapterPlanner().propose_scene_beats(
            context={"chapter": {}},
            task_card=card,
            instruction="",
            provider_user_id="test-user",
        )
    )

    response.result.ensure_covers(card)
    assert len(response.result.scenes) == 2
    expected = {
        (item.kind, item.text) for item in card.requirement_items()
    }
    actual = {
        (item.kind, item.text)
        for scene in response.result.scenes
        for item in scene.requirement_refs
    }
    assert actual == expected
    assert card.scenes == []
    assert card.must_happen == ["确认邮戳", "购买车票"]


def test_scene_breakdown_baseline_never_overwrites_newer_task_card(
    tmp_path: Path,
):
    database = Database(tmp_path / "scene-baseline.db")
    database.initialize()
    user_id = database.create_user(
        "scene-baseline-author", hash_password("password-123")
    )
    project_id = "scene-baseline-project"
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="雾港",
        genre="悬疑",
        premise="调查异常来信。",
        world_setting="当代海港小城。",
        style_guide="克制具体。",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
    )
    content_path = tmp_path / "scene-baseline.txt"
    content_path.write_text("", encoding="utf-8")
    chapter_id = "scene-baseline-chapter"
    database.add_novel_chapter(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        title="第一章",
        outline="核对来信。",
        key_points="确认邮戳",
        content_path=content_path,
    )
    planning = PlanningService(database)
    base = ChapterTaskCard.model_validate(
        {
            "purpose": "核对来信。",
            "start_state": "尚未相信。",
            "end_state": "决定调查。",
            "central_conflict": "邮戳异常。",
            "ending_hook": "线索指向码头。",
            "scenes": [
                {
                    "goal": "核对邮戳",
                    "obstacle": "记录缺失",
                    "action": "查询档案",
                    "end_state": "确认异常",
                },
                {
                    "goal": "决定调查",
                    "obstacle": "时间紧迫",
                    "action": "前往码头",
                    "end_state": "开始行动",
                },
            ],
        }
    )
    planning.upsert_task_card(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        volume_id=None,
        card=base,
        confirm=False,
    )
    stored = planning.get_task_card(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
    )
    old_fingerprint = chapter_task_card_fingerprint(stored)
    newer = base.model_copy(
        update={"purpose": "作者改为先保护证人，再核对来信。"}
    )
    planning.upsert_task_card(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        volume_id=None,
        card=newer,
        confirm=False,
    )

    with pytest.raises(ValueError, match="未覆盖较新的作者修改"):
        planning.upsert_task_card(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            volume_id=None,
            card=base,
            confirm=False,
            source="ai",
            expected_card_fingerprint=old_fingerprint,
        )
    current = planning.get_task_card(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
    )
    assert current["purpose"] == "作者改为先保护证人，再核对来信。"
