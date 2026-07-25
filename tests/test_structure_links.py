from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.context_compiler import compile_planned_causal_links
from app.db import Database
from app.main import create_app
from app.planning_ai import MockChapterPlanner
from app.planning_schema import (
    ChapterTaskCard,
    allocate_scene_requirement_refs,
)
from app.planning_service import PlanningService
from app.security import hash_password
from app.structure_health import StructureHealthService
from app.structure_link_service import StructureLinkService
from app.writing import build_writing_messages


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        app_name="章节因果测试",
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
        deepseek_api_key=None,
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


def _card(thread: str) -> ChapterTaskCard:
    card = ChapterTaskCard.model_validate(
        {
            "purpose": "用一项可核查行动推动调查。",
            "start_state": "人物只有不完整线索。",
            "end_state": "人物的选择改变后续局面。",
            "central_conflict": "证据保全与公开调查发生冲突。",
            "emotional_value": "行动带来不可撤销的代价。",
            "plot_threads": [thread],
            "must_happen": ["人物采取一项具体行动"],
            "must_preserve": ["不能提前知道幕后人的完整身份"],
            "forbidden": ["不能直接揭晓全书真相"],
            "foreshadow_setup": [],
            "foreshadow_payoff": [],
            "ending_hook": "行动后果指向下一项证据。",
            "target_chars": 3000,
            "scenes": [
                {
                    "goal": "取得并核查一项记录",
                    "obstacle": "记录即将被销毁",
                    "action": "人物要求封存设备日志",
                    "end_state": "日志被保留下来",
                },
                {
                    "goal": "根据日志选择下一步",
                    "obstacle": "公开会惊动对手",
                    "action": "人物只公开部分编号",
                    "end_state": "对手被迫改变行动",
                },
            ],
        }
    )
    return allocate_scene_requirement_refs(card)


def _project(
    tmp_path: Path,
    *,
    username: str = "causal-author",
    database: Database | None = None,
) -> tuple[Database, int, str, str, list[str]]:
    database = database or Database(tmp_path / f"{username}.db")
    database.initialize()
    existing = database.get_user_by_username(username)
    user_id = (
        int(existing["id"])
        if existing
        else database.create_user(
            username, hash_password("password-123")
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
    chapter_ids = []
    roles = ["setup", "payoff", "setup", "payoff"]
    arcs = ["父亲失踪主线", "父亲失踪主线", "档案员关系线", "档案员关系线"]
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
            key_points="核查记录\n采取下一步行动",
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


def test_causal_link_invalidates_related_future_plans_and_enters_models(
    tmp_path: Path,
):
    database, user_id, project_id, volume_id, chapters = _project(
        tmp_path
    )
    planning = PlanningService(database)
    for chapter_id, thread in (
        (chapters[0], "父亲失踪主线"),
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

    links = StructureLinkService(database)
    result = links.create_link(
        user_id=user_id,
        project_id=project_id,
        source_chapter_id=chapters[0],
        target_chapter_id=chapters[2],
        relation_type="complicates",
        cause_text="林岚只公开部分编号，迫使档案员提前销毁复印记录。",
        effect_text="销毁行为留下设备日志，使林岚锁定档案员的参与。",
        author_note="让主线行动直接推动关系线反转。",
    )

    assert result["affected_chapter_count"] == 2
    assert result["reset_task_card_count"] == 2
    assert planning.get_task_card(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[0],
    )["status"] == "draft"
    assert planning.get_task_card(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[2],
    )["status"] == "draft"
    assert database.get_novel_chapter(
        user_id, project_id, chapters[0]
    )["needs_recheck"] == 1

    stored = links.list_links(
        user_id=user_id, project_id=project_id
    )
    assert len(stored) == 1
    assert stored[0]["cross_line"] is True
    assert stored[0]["shared_arc_titles"] == []
    assert stored[0]["relation_label"] == "后果升级"

    source_context = database.get_writing_context(
        user_id, chapters[0]
    )
    target_context = database.get_writing_context(
        user_id, chapters[2]
    )
    source_compiled = compile_planned_causal_links(
        source_context, usage="write"
    )
    target_compiled = compile_planned_causal_links(
        target_context, usage="plan"
    )
    assert source_compiled["included_count"] == 1
    assert source_compiled["outgoing"][0]["effect"].startswith(
        "销毁行为"
    )
    assert target_compiled["incoming"][0]["cause"].startswith(
        "林岚只公开"
    )
    assert (
        database.get_writing_context(user_id, chapters[1])[
            "planned_causal_links"
        ]
        == []
    )

    prompt = str(
        build_writing_messages(
            context=source_context,
            operation="draft",
            instruction="",
            current_content="",
            previous_content="",
        )[1]["content"]
    )
    assert "<planned_future_causality>" in prompt
    assert "迫使档案员提前销毁复印记录" in prompt
    assert "不是已经发生的正史" in prompt

    proposal = asyncio.run(
        MockChapterPlanner().propose(
            context=target_context,
            instruction="",
            provider_user_id="test-user",
        )
    )
    assert any(
        "销毁行为留下设备日志" in item
        for item in proposal.result.must_happen
    )


def test_causal_link_direction_duplicate_archive_and_owner_boundaries(
    tmp_path: Path,
):
    database, user_id, project_id, _volume_id, chapters = _project(
        tmp_path, username="boundary-author"
    )
    service = StructureLinkService(database)
    kwargs = {
        "user_id": user_id,
        "project_id": project_id,
        "source_chapter_id": chapters[0],
        "target_chapter_id": chapters[2],
        "relation_type": "causes",
        "cause_text": "公开编号使对手改变证据处理方式。",
        "effect_text": "设备日志因此暴露新的经手人。",
        "author_note": "",
    }
    created = service.create_link(**kwargs)

    with pytest.raises(ValueError, match="已经存在"):
        service.create_link(**kwargs)
    with pytest.raises(ValueError, match="不能倒流"):
        service.create_link(
            **{
                **kwargs,
                "source_chapter_id": chapters[3],
                "target_chapter_id": chapters[1],
            }
        )
    with pytest.raises(ValueError, match="小说项目不存在"):
        service.create_link(**{**kwargs, "user_id": user_id + 1000})
    assert service.list_links(
        user_id=user_id + 1000, project_id=project_id
    ) == []

    archived = service.archive_link(
        user_id=user_id,
        project_id=project_id,
        link_id=created["id"],
    )
    assert archived["changed"] is True
    assert service.list_links(
        user_id=user_id, project_id=project_id
    ) == []
    history = service.list_links(
        user_id=user_id,
        project_id=project_id,
        include_archived=True,
    )
    assert history[0]["status"] == "archived"
    assert database.get_writing_context(
        user_id, chapters[0]
    )["planned_causal_links"] == []

    database.create_generation_job(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapters[0],
        operation="draft",
        instruction="",
        provider="mock",
        model="mock-novel-writer",
        credential_source="default",
        max_jobs_per_day=50,
    )
    with pytest.raises(ValueError, match="AI 正在处理相关章节"):
        service.create_link(
            **{
                **kwargs,
                "target_chapter_id": chapters[3],
                "relation_type": "enables",
                "effect_text": "更晚章节承接新的调查后果。",
            }
        )


def test_canonical_boundary_hands_realized_link_back_to_memory(
    tmp_path: Path,
):
    database, user_id, project_id, _volume_id, chapters = _project(
        tmp_path, username="canon-link-author"
    )

    def make_canonical(chapter_id: str, label: str) -> None:
        version_path = (
            tmp_path / project_id / chapter_id / f"{label}.txt"
        )
        version_path.write_text(
            "林岚完成了一项可核查的调查行动。",
            encoding="utf-8",
        )
        version_id = database.record_manual_chapter_version(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            version_path=version_path,
            char_count=18,
        )
        assert version_id
        database.accept_chapter_version(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            version_id=version_id,
            override_reason="作者确认该短文本仅用于正史边界测试",
        )

    make_canonical(chapters[1], "canon-2")
    service = StructureLinkService(database)
    with pytest.raises(ValueError, match="结果章必须位于当前正史边界之后"):
        service.create_link(
            user_id=user_id,
            project_id=project_id,
            source_chapter_id=chapters[0],
            target_chapter_id=chapters[1],
            relation_type="causes",
            cause_text="更早行动改变调查方向。",
            effect_text="已确认章节出现新的调查结果。",
        )

    service.create_link(
        user_id=user_id,
        project_id=project_id,
        source_chapter_id=chapters[1],
        target_chapter_id=chapters[2],
        relation_type="pays_off",
        cause_text="第二章保留下来的编号成为后续核查入口。",
        effect_text="第三章设备日志兑现编号所指向的经手人。",
    )
    assert len(
        database.get_writing_context(
            user_id, chapters[1]
        )["planned_causal_links"]
    ) == 1

    make_canonical(chapters[2], "canon-3")
    assert database.get_writing_context(
        user_id, chapters[1]
    )["planned_causal_links"] == []
    stored = service.list_links(
        user_id=user_id, project_id=project_id
    )
    assert stored[0]["planning_status"] == "realized"
    report = StructureHealthService(database).get_report(
        user_id=user_id, project_id=project_id
    )
    assert report["counts"]["causal_links"] == 0


def test_explicit_causal_bridge_suppresses_only_the_window_hard_cut(
    tmp_path: Path,
):
    database, user_id, project_id, _volume_id, chapters = _project(
        tmp_path, username="health-link-author"
    )
    before = StructureHealthService(database).get_report(
        user_id=user_id, project_id=project_id
    )
    assert "WINDOW_ARC_HARD_CUT" in {
        item["code"] for item in before["findings"]
    }

    StructureLinkService(database).create_link(
        user_id=user_id,
        project_id=project_id,
        source_chapter_id=chapters[1],
        target_chapter_id=chapters[2],
        relation_type="enables",
        cause_text="主线调查公开一项编号，迫使档案员改变自保策略。",
        effect_text="档案员主动接近林岚，关系线因此进入反转。",
    )
    after = StructureHealthService(database).get_report(
        user_id=user_id, project_id=project_id
    )
    codes = {item["code"] for item in after["findings"]}
    assert "WINDOW_ARC_HARD_CUT" not in codes
    assert "WINDOW_RESTARTS_SAME_VOLUME" in codes
    boundary = after["boundaries"][0]
    assert len(boundary["causal_bridges"]) == 1
    assert after["counts"]["causal_links"] == 1
    assert after["counts"]["cross_line_causal_links"] == 1


def test_structure_link_web_flow_and_task_card_summary(tmp_path: Path):
    application = create_app(_settings(tmp_path))
    with TestClient(application) as client:
        register = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "网页因果作者",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": _csrf(register.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        database = application.state.database
        (
            _database,
            user_id,
            project_id,
            _volume_id,
            chapters,
        ) = _project(
            tmp_path,
            username="网页因果作者",
            database=database,
        )

        page = client.get(
            f"/novels/{project_id}/workbench"
            "?view=settings&settings_tab=structure"
        )
        assert page.status_code == 200
        response = client.post(
            f"/novels/{project_id}/structure-links",
            data={
                "source_chapter_id": chapters[0],
                "target_chapter_id": chapters[2],
                "relation_type": "causes",
                "cause_text": "林岚公开编号，迫使档案员销毁复印记录。",
                "effect_text": "设备日志暴露档案员参与了投递链。",
                "author_note": "跨线推进。",
                "csrf": _csrf(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert (
            "view=settings&settings_tab=structure&saved=true"
            in response.headers["location"]
        )
        target_page = client.get(
            f"/novels/{project_id}/chapters/{chapters[2]}/task-card"
        )
        assert target_page.status_code == 200
        assert "本章承接的原因与留下的后果" in target_page.text
        assert "设备日志暴露档案员参与了投递链" in target_page.text
        assert "本章必须承接" in target_page.text

        link = StructureLinkService(database).list_links(
            user_id=user_id, project_id=project_id
        )[0]
        response = client.post(
            f"/novels/{project_id}/structure-links/{link['id']}/archive",
            data={"csrf": _csrf(target_page.text)},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert (
            "view=settings&settings_tab=structure&saved=true"
            in response.headers["location"]
        )
        assert not StructureLinkService(database).list_links(
            user_id=user_id, project_id=project_id
        )
        archived = StructureLinkService(database).list_links(
            user_id=user_id,
            project_id=project_id,
            include_archived=True,
        )
        assert archived[0]["status"] == "archived"
