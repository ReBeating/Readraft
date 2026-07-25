import asyncio
import hashlib
from pathlib import Path

import pytest

from app.canon_impact_service import CanonImpactService
from app.db import Database
from app.planning_schema import (
    ChapterTaskCard,
    allocate_scene_requirement_refs,
)
from app.planning_service import PlanningService
from app.reader_planner import MockReaderPlanner
from app.reader_service import ReaderDecisionService
from app.security import hash_password


def _project(tmp_path: Path):
    database = Database(tmp_path / "reader.db")
    database.initialize()
    user_id = database.create_user(
        "reader-planner", hash_password("password-123")
    )
    project_id = "reader-project"
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
        core_appeal="证据推理与压抑亲情",
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
    return database, user_id, project_id


def _chapter(
    database: Database,
    tmp_path: Path,
    *,
    user_id: int,
    project_id: str,
    chapter_id: str,
    title: str,
) -> str:
    chapter_dir = tmp_path / chapter_id
    chapter_dir.mkdir()
    content_path = chapter_dir / "content.txt"
    content_path.write_text("", encoding="utf-8")
    database.add_novel_chapter(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        title=title,
        outline=f"{title}原计划",
        key_points="原关键点",
        content_path=content_path,
    )
    return chapter_id


def _make_canonical(
    database: Database,
    *,
    user_id: int,
    project_id: str,
    chapter_id: str,
    marker: str,
) -> str:
    chapter = database.get_novel_chapter(
        user_id, project_id, chapter_id
    )
    content = marker * 2100
    version_path = (
        Path(chapter["content_path"]).parent / f"{marker}-version.txt"
    )
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
        version_id=version_id,
        override_reason="测试环境人工确认这个版本",
    )
    assert accepted
    return str(version_id)


def _confirmed_card() -> ChapterTaskCard:
    card = ChapterTaskCard.model_validate(
        {
            "purpose": "推进调查线",
            "start_state": "林岚只有来信",
            "end_state": "林岚获得新证据",
            "central_conflict": "证据来源受到阻拦",
            "ending_hook": "新证据指向旧码头",
            "scenes": [
                {
                    "goal": "核对证据",
                    "obstacle": "记录缺失",
                    "action": "查询档案",
                    "end_state": "找到线索",
                },
                {
                    "goal": "验证线索",
                    "obstacle": "证人回避",
                    "action": "当面交涉",
                    "end_state": "确认下一目标",
                },
            ],
        }
    )
    return allocate_scene_requirement_refs(card)


def test_reader_request_requires_author_choice_before_updating_plan(
    tmp_path: Path,
):
    database, user_id, project_id = _project(tmp_path)
    first_id = _chapter(
        database,
        tmp_path,
        user_id=user_id,
        project_id=project_id,
        chapter_id="chapter-1",
        title="第一章 来信",
    )
    second_id = _chapter(
        database,
        tmp_path,
        user_id=user_id,
        project_id=project_id,
        chapter_id="chapter-2",
        title="第二章 旧档案",
    )
    _make_canonical(
        database,
        user_id=user_id,
        project_id=project_id,
        chapter_id=first_id,
        marker="甲",
    )
    planning = PlanningService(database)
    planning.upsert_task_card(
        user_id=user_id,
        project_id=project_id,
        chapter_id=second_id,
        volume_id=None,
        card=_confirmed_card(),
        confirm=True,
    )
    service = ReaderDecisionService(database, tmp_path / "novels")
    request_id = service.create_request(
        user_id=user_id,
        project_id=project_id,
        raw_text="希望女主不要这么快相信搭档。",
        request_type="relationship",
        impact_scope="next_three",
        priority="soft",
        constraints=["不得改写父亲死亡的正史"],
        author_note="可以增加试探，但不要拖慢调查。",
    )
    context = service.build_planning_context(
        user_id=user_id, request_id=request_id
    )
    response = asyncio.run(
        MockReaderPlanner().propose(
            context=context,
            request_data=context["request"],
            provider_user_id="test-user",
        )
    )
    assert len(response.result.alternatives) == 3
    assert sum(
        not item.affects_published_canon
        for item in response.result.alternatives
    ) >= 2
    original_second = database.get_novel_chapter(
        user_id, project_id, second_id
    )

    job_id = database.create_reader_planning_job(
        user_id=user_id,
        project_id=project_id,
        request_id=request_id,
        provider="mock",
        model="mock-reader-planner",
        credential_source="default",
    )
    claimed = database.claim_next_generation()
    assert claimed["id"] == job_id
    proposal_ids = service.save_proposals(
        user_id=user_id,
        request_id=request_id,
        job_id=job_id,
        result=response.result,
        provider=response.provider,
        model=response.model,
    )
    assert database.complete_reader_planning(
        job_id=job_id,
        claim_token=claimed["claim_token"],
        result={"request_id": request_id, "proposal_ids": proposal_ids},
        input_tokens=0,
        output_tokens=0,
    )
    unchanged_second = database.get_novel_chapter(
        user_id, project_id, second_id
    )
    assert unchanged_second["outline"] == original_second["outline"]

    with pytest.raises(ValueError, match="回改已发表正史"):
        service.accept_proposal(
            user_id=user_id, proposal_id=proposal_ids[2]
        )
    applied = service.accept_proposal(
        user_id=user_id, proposal_id=proposal_ids[0]
    )
    assert len(applied["applied"]) == 3
    chapters = database.list_novel_chapters(user_id, project_id)
    assert [chapter["position"] for chapter in chapters] == [1, 2, 3, 4]
    assert chapters[0]["canonical_version_id"]
    assert chapters[1]["outline"] != original_second["outline"]
    assert chapters[1]["plan_status"] == "draft"
    assert (tmp_path / "novels" / str(user_id) / project_id).exists()
    stored = service.get_request(user_id=user_id, request_id=request_id)
    assert stored["status"] == "adopted"
    assert len(stored["applications"]) == 3
    assert [item["status"] for item in stored["proposals"]].count(
        "accepted"
    ) == 1


def test_old_canon_change_requires_impact_report_and_keeps_selected_rechecks(
    tmp_path: Path,
):
    database, user_id, project_id = _project(tmp_path)
    chapter_ids = []
    for position in range(1, 4):
        chapter_id = _chapter(
            database,
            tmp_path,
            user_id=user_id,
            project_id=project_id,
            chapter_id=f"impact-{position}",
            title=f"第{position}章",
        )
        chapter_ids.append(chapter_id)
        _make_canonical(
            database,
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            marker=str(position),
        )
    first = database.get_novel_chapter(
        user_id, project_id, chapter_ids[0]
    )
    old_version_id = str(first["canonical_version_id"])
    replacement = "新" * 2100
    replacement_path = (
        Path(first["content_path"]).parent / "replacement.txt"
    )
    replacement_path.write_text(replacement, encoding="utf-8")
    replacement_id = database.record_manual_chapter_version(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_ids[0],
        version_path=replacement_path,
        char_count=len(replacement),
        effective_char_count=len(replacement),
        content_hash=hashlib.sha256(
            replacement.encode("utf-8")
        ).hexdigest(),
    )
    impact = CanonImpactService(database)
    report_id = impact.prepare_report(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_ids[0],
        proposed_version_id=str(replacement_id),
        override_reason="作者确认改动不会破坏核心设定",
    )
    report = impact.get_report(user_id=user_id, report_id=report_id)
    assert report["status"] == "pending"
    assert report["downstream_count"] == 2
    chapter_items = [
        item for item in report["items"] if item["item_type"] == "chapter"
    ]
    decisions = {
        item["id"]: {
            "decision": (
                "keep"
                if item["downstream_chapter_id"] == chapter_ids[1]
                else "recheck"
            ),
            "note": "逐章人工判断",
        }
        for item in report["items"]
    }
    assert impact.update_decisions(
        user_id=user_id,
        report_id=report_id,
        decisions=decisions,
    )
    accepted = database.accept_chapter_version(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_ids[0],
        version_id=str(replacement_id),
        override_reason="作者确认改动不会破坏核心设定",
        expected_old_canonical_version_id=old_version_id,
    )
    assert accepted["downstream_count"] == 2
    assert impact.mark_applied(user_id=user_id, report_id=report_id)
    assert database.get_novel_chapter(
        user_id, project_id, chapter_ids[1]
    )["needs_recheck"] == 0
    assert database.get_novel_chapter(
        user_id, project_id, chapter_ids[2]
    )["needs_recheck"] == 1
    applied_report = impact.get_report(
        user_id=user_id, report_id=report_id
    )
    assert applied_report["status"] == "applied"
    assert len(chapter_items) == 2
