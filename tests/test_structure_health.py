from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.db import Database
from app.main import create_app
from app.planning_service import PlanningService
from app.security import hash_password
from app.story_planning_schema import PlannedStoryArc, StoryBlueprint
from app.story_planning_service import StoryPlanningService
from app.structure_health import StructureHealthService


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        app_name="结构体检测试",
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
    return html.split('name="csrf" value="', 1)[1].split('"', 1)[0]


def _blueprint() -> StoryBlueprint:
    return StoryBlueprint.model_validate(
        {
            "central_question": "父亲失踪后为什么仍有新信寄出？",
            "protagonist_goal": "林岚要找出信件来源与父亲下落。",
            "core_conflict": "档案、证词和新证据不断互相否定。",
            "stakes": "继续追查会让林岚失去现有生活。",
            "opening_state": "林岚认定父亲已经失踪十年。",
            "ending_state": "林岚公开完整真相并承担后果。",
            "major_turns": ["新信并非伪造", "父亲主动隐藏最后一次行踪"],
            "must_payoffs": ["解释全部新信如何寄出"],
            "forbidden_shortcuts": ["不能依靠突然出现的万能证人"],
        }
    )


def _arc(
    title: str,
    *,
    arc_type: str = "main",
    lifecycle: str = "active",
    priority: int = 5,
) -> PlannedStoryArc:
    return PlannedStoryArc.model_validate(
        {
            "arc_type": arc_type,
            "title": title,
            "dramatic_question": f"{title}最终会揭示什么？",
            "promise": f"持续推进并兑现{title}。",
            "start_state": "只有一条互相矛盾的线索。",
            "target_payoff": f"给出{title}的可核对答案。",
            "involved_characters": ["林岚"],
            "planned_turns": ["取得证据", "推翻最初解释"],
            "lifecycle_status": lifecycle,
            "priority": priority,
        }
    )


def _project(
    tmp_path: Path,
    *,
    username: str,
) -> tuple[Database, int, str, str]:
    database = Database(tmp_path / f"{username}.db")
    database.initialize()
    user_id = database.create_user(
        username, hash_password("password-123")
    )
    project_id = f"{username}-project"
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="雾港来信",
        genre="悬疑",
        premise="林岚收到失踪父亲署名的新信。",
        world_setting="当代海港。",
        style_guide="克制具体。",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
        planning_horizon=20,
    )
    StoryPlanningService(database).save_blueprint(
        user_id=user_id,
        project_id=project_id,
        blueprint=_blueprint(),
        confirm=True,
    )
    volume_id = PlanningService(database).create_volume(
        user_id=user_id,
        project_id=project_id,
        title="第一卷 迟到的信",
        goal="锁定新信投递链。",
        start_state="林岚认为新信是恶作剧。",
        end_state="林岚确认投递链被人为维护。",
        major_conflict="现实记录与旧案结论互相冲突。",
        payoff="解释第一层投递方式。",
    )
    return database, user_id, project_id, volume_id


def _add_arc(
    database: Database,
    *,
    user_id: int,
    project_id: str,
    arc: PlannedStoryArc,
) -> str:
    return StoryPlanningService(database).create_arc(
        user_id=user_id,
        project_id=project_id,
        arc=arc,
        confirm=True,
    )


def _add_skeleton_chapter(
    database: Database,
    tmp_path: Path,
    *,
    user_id: int,
    project_id: str,
    volume_id: str,
    position_hint: int,
    role: str,
    arc_titles: list[str],
    ending_hook: str = "新证据迫使林岚改变下一步行动。",
    application_id: str = "",
) -> str:
    chapter_id = f"{project_id}-chapter-{position_hint}"
    chapter_dir = tmp_path / project_id / chapter_id
    chapter_dir.mkdir(parents=True)
    content_path = chapter_dir / "content.txt"
    content_path.write_text("", encoding="utf-8")
    database.add_novel_chapter(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        title=f"第{position_hint}章 线索",
        outline="林岚核对证据并迫使调查方向发生变化。",
        key_points="核对一项记录\n据此采取下一步行动",
        content_path=content_path,
        volume_id=volume_id,
    )
    with database.connection() as connection:
        connection.execute(
            """
            UPDATE novel_chapters
            SET skeleton_role=?, skeleton_arc_titles_json=?,
                skeleton_ending_hook=?, skeleton_application_id=?
            WHERE id=?
            """,
            (
                role,
                json.dumps(arc_titles, ensure_ascii=False),
                ending_hook,
                application_id or None,
                chapter_id,
            ),
        )
        connection.commit()
    return chapter_id


def test_structure_health_reports_evidence_without_mutating_project(
    tmp_path: Path,
):
    database, user_id, project_id, volume_id = _project(
        tmp_path, username="health-author"
    )
    _add_arc(
        database,
        user_id=user_id,
        project_id=project_id,
        arc=_arc("父亲失踪主线"),
    )
    _add_arc(
        database,
        user_id=user_id,
        project_id=project_id,
        arc=_arc(
            "林岚与周时关系线",
            arc_type="relationship",
            lifecycle="planned",
            priority=2,
        ),
    )
    for position in range(1, 9):
        _add_skeleton_chapter(
            database,
            tmp_path,
            user_id=user_id,
            project_id=project_id,
            volume_id=volume_id,
            position_hint=position,
            role="escalation",
            arc_titles=[] if position == 1 else ["父亲失踪主线"],
            ending_hook="" if position == 1 else "证词指向下一位经手人。",
        )

    before = database.get_novel_project(user_id, project_id)
    report = StructureHealthService(database).get_report(
        user_id=user_id, project_id=project_id
    )
    after = database.get_novel_project(user_id, project_id)

    assert report
    assert report["status"] == "blocked"
    codes = {item["code"] for item in report["findings"]}
    assert {
        "CHAPTER_ARC_MISSING",
        "CHAPTER_HOOK_MISSING",
        "ROLE_RUN_TOO_LONG",
        "TURNING_POINT_GAP",
        "VOLUME_PAYOFF_MISSING",
        "VOLUME_REVERSAL_MISSING",
        "ARC_NOT_ADVANCED",
    }.issubset(codes)
    assert report["counts"]["future_chapters"] == 8
    assert report["chapters"][0]["health_severity"] == "blocking"
    main_arc = next(
        item
        for item in report["arcs"]
        if item["title"] == "父亲失踪主线"
    )
    relationship_arc = next(
        item
        for item in report["arcs"]
        if item["title"] == "林岚与周时关系线"
    )
    assert main_arc["touch_count"] == 7
    assert relationship_arc["coverage_status"] == "missing"
    assert before["updated_at"] == after["updated_at"]
    assert (
        StructureHealthService(database).get_report(
            user_id=user_id + 1000, project_id=project_id
        )
        is None
    )


def test_structure_health_marks_window_cut_but_not_source_change_itself(
    tmp_path: Path,
):
    database, user_id, project_id, volume_id = _project(
        tmp_path, username="boundary-author"
    )
    _add_arc(
        database,
        user_id=user_id,
        project_id=project_id,
        arc=_arc("父亲失踪主线"),
    )
    _add_arc(
        database,
        user_id=user_id,
        project_id=project_id,
        arc=_arc(
            "林岚与周时关系线",
            arc_type="relationship",
            priority=4,
        ),
    )
    roles = [
        "setup",
        "escalation",
        "payoff",
        "setup",
        "reversal",
        "payoff",
    ]
    for position, role in enumerate(roles, start=1):
        first_window = position <= 3
        _add_skeleton_chapter(
            database,
            tmp_path,
            user_id=user_id,
            project_id=project_id,
            volume_id=volume_id,
            position_hint=position,
            role=role,
            arc_titles=[
                (
                    "父亲失踪主线"
                    if first_window
                    else "林岚与周时关系线"
                )
            ],
            application_id=(
                "application-window-a"
                if first_window
                else "application-window-b"
            ),
        )

    report = StructureHealthService(database).get_report(
        user_id=user_id, project_id=project_id
    )

    assert report
    assert len(report["boundaries"]) == 1
    boundary = report["boundaries"][0]
    assert boundary["label"] == "结构窗口切换"
    assert boundary["left"]["position"] == 3
    assert boundary["right"]["position"] == 4
    assert boundary["shared_arcs"] == []
    assert boundary["health_severity"] == "warning"
    codes = {item["code"] for item in report["findings"]}
    assert "WINDOW_ARC_HARD_CUT" in codes
    assert "WINDOW_RESTARTS_SAME_VOLUME" in codes
    assert "WINDOW_SOURCE_CHANGED" not in codes


def test_structure_health_can_be_clean_for_a_complete_manual_window(
    tmp_path: Path,
):
    database, user_id, project_id, volume_id = _project(
        tmp_path, username="clean-author"
    )
    _add_arc(
        database,
        user_id=user_id,
        project_id=project_id,
        arc=_arc("父亲失踪主线"),
    )
    for position, role in enumerate(
        [
            "setup",
            "escalation",
            "reversal",
            "escalation",
            "payoff",
        ],
        start=1,
    ):
        _add_skeleton_chapter(
            database,
            tmp_path,
            user_id=user_id,
            project_id=project_id,
            volume_id=volume_id,
            position_hint=position,
            role=role,
            arc_titles=["父亲失踪主线"],
        )

    report = StructureHealthService(database).get_report(
        user_id=user_id, project_id=project_id
    )

    assert report
    assert report["status"] == "healthy"
    assert {
        item["code"] for item in report["findings"]
    } == {"CAUSAL_BRIDGE_MISSING"}
    assert all(
        item["severity"] == "info" for item in report["findings"]
    )
    assert report["counts"]["blocking"] == 0
    assert report["counts"]["warning"] == 0


def test_structure_health_rejects_a_future_window_that_skips_first_volume(
    tmp_path: Path,
):
    database, user_id, project_id, _ = _project(
        tmp_path, username="volume-gap-author"
    )
    _add_arc(
        database,
        user_id=user_id,
        project_id=project_id,
        arc=_arc("父亲失踪主线"),
    )
    second_volume_id = PlanningService(database).create_volume(
        user_id=user_id,
        project_id=project_id,
        title="第二卷 投递链",
        goal="找到维护投递链的人。",
        start_state="只有一位经手人的证词。",
        end_state="锁定投递链维护者。",
        major_conflict="经手人互相推翻证词。",
        payoff="解释第二层投递方式。",
    )
    for position, role in enumerate(
        ["setup", "reversal", "payoff"], start=1
    ):
        _add_skeleton_chapter(
            database,
            tmp_path,
            user_id=user_id,
            project_id=project_id,
            volume_id=second_volume_id,
            position_hint=position,
            role=role,
            arc_titles=["父亲失踪主线"],
        )

    report = StructureHealthService(database).get_report(
        user_id=user_id, project_id=project_id
    )

    assert report
    assert report["status"] == "blocked"
    assert "VOLUME_SEQUENCE_START" in {
        item["code"] for item in report["findings"]
    }


def test_removed_structure_health_page_is_not_routed(
    tmp_path: Path,
):
    application = create_app(_settings(tmp_path))
    with TestClient(application) as client:
        register = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "结构页面作者",
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
                "csrf": _csrf(novel_form.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        workbench_url = response.headers["location"]
        project_id = workbench_url.split("/novels/", 1)[1].split("/", 1)[0]

        workspace = client.get(workbench_url)
        assert workspace.status_code == 200
        assert f"/novels/{project_id}/structure-health" not in workspace.text

        page = client.get(f"/novels/{project_id}/structure-health")
        assert page.status_code == 404

        response = client.post(
            "/logout",
            data={"csrf": _csrf(workspace.text)},
            follow_redirects=False,
        )
        assert response.status_code == 303
        register = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "另一位结构作者",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": _csrf(register.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        forbidden = client.get(
            f"/novels/{project_id}/structure-health"
        )
        assert forbidden.status_code == 404
