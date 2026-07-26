import asyncio
import json
import re
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.context_compiler import (
    build_writing_context_snapshot,
    compile_story_plan_context,
)
from app.db import SCHEMA, Database, utc_now
from app.main import create_app
from app.migrations import MIGRATIONS
from app.planning_ai import DeepSeekChapterPlanner, MockChapterPlanner
from app.planning_schema import (
    ChapterTaskCard,
    allocate_scene_requirement_refs,
)
from app.planning_service import PlanningService
from app.security import hash_password
from app.story_planning_schema import PlannedStoryArc, StoryBlueprint
from app.story_planning_service import StoryPlanningService
from app.writing import build_writing_messages


def _settings(tmp_path: Path) -> Settings:
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
        max_jobs_per_day=50,
    )


def _csrf(html: str) -> str:
    match = re.search(r'name="csrf" value="([^"]+)"', html)
    assert match
    return match.group(1)


def _project_with_chapter(
    database: Database,
    tmp_path: Path,
    *,
    user_id: int,
    project_id: str,
    chapter_id: str = "chapter-one",
) -> str:
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="雾港来信",
        genre="悬疑",
        premise="林岚收到失踪父亲署名的新信。",
        world_setting="当代海港。",
        style_guide="克制、具体。",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
        story_promise="逐层揭开一条可核对的现实谜团。",
        ending_constraint="父亲失踪必须得到完整因果解释。",
    )
    chapter_dir = tmp_path / project_id / chapter_id
    chapter_dir.mkdir(parents=True)
    content_path = chapter_dir / "content.txt"
    content_path.write_text("", encoding="utf-8")
    database.add_novel_chapter(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        title="第一章 迟到的信",
        outline="林岚核对来信并决定回港。",
        key_points="确认邮戳异常",
        content_path=content_path,
    )
    return chapter_id


def _blueprint(*, question: str = "父亲为何在失踪后继续寄信？"):
    return StoryBlueprint.model_validate(
        {
            "central_question": question,
            "protagonist_goal": "找到父亲并确认自己记忆中的空白。",
            "core_conflict": "林岚追查证据，档案部门持续抹除异常记录。",
            "stakes": "每次取得证据都会让一名知情者失去公开作证的机会。",
            "opening_state": "林岚相信父亲已经死亡，也拒绝回到雾港。",
            "ending_state": "林岚公开证据，同时接受父亲主动失踪的选择。",
            "major_turns": [
                "新信迫使林岚回港",
                "她发现自己的证词也被篡改",
                "父亲从受害者变成主动参与者",
            ],
            "must_payoffs": ["解释所有来信的时间与投递方式"],
            "forbidden_shortcuts": ["不能让新角色口述全部真相"],
            "author_notes": "终局仍保留公开真相的道德代价。",
        }
    )


def _arc(
    *,
    title: str,
    payoff: str,
    priority: int,
    arc_type: str = "mystery",
    lifecycle_status: str = "planned",
) -> PlannedStoryArc:
    return PlannedStoryArc.model_validate(
        {
            "arc_type": arc_type,
            "title": title,
            "dramatic_question": f"{title}最终如何解释？",
            "promise": f"持续给出与{title}有关的可验证线索。",
            "start_state": "只有一封来源不明的信。",
            "target_payoff": payoff,
            "involved_characters": ["林岚"],
            "planned_turns": ["确认第一项异常", "旧解释被新证据推翻"],
            "lifecycle_status": lifecycle_status,
            "priority": priority,
        }
    )


def _confirmed_card(plot_threads: list[str]) -> ChapterTaskCard:
    card = ChapterTaskCard.model_validate(
        {
            "purpose": "迫使林岚返回雾港。",
            "start_state": "林岚拒绝相信来信。",
            "end_state": "林岚买下回港车票。",
            "central_conflict": "邮戳证据与既有结论冲突。",
            "plot_threads": plot_threads,
            "must_happen": ["核对邮戳"],
            "forbidden": ["揭晓父亲最终去向"],
            "ending_hook": "信纸带着旧码头才有的盐味。",
            "scenes": [
                {
                    "goal": "核对来信",
                    "obstacle": "邮戳日期不可能",
                    "action": "比对旧信和登记记录",
                    "end_state": "承认来信值得调查",
                },
                {
                    "goal": "决定是否回港",
                    "obstacle": "她抗拒故乡",
                    "action": "购买车票",
                    "end_state": "踏上回港列车",
                },
            ],
        }
    )
    return allocate_scene_requirement_refs(card)


def test_blueprint_draft_confirmation_and_restore_are_isolated(
    tmp_path: Path,
):
    database = Database(tmp_path / "blueprint.db")
    database.initialize()
    owner_id = database.create_user(
        "blueprint-owner", hash_password("password-123")
    )
    other_id = database.create_user(
        "blueprint-other", hash_password("password-123")
    )
    project_id = "blueprint-project"
    chapter_id = _project_with_chapter(
        database,
        tmp_path,
        user_id=owner_id,
        project_id=project_id,
    )
    service = StoryPlanningService(database)

    draft_id = service.save_blueprint(
        user_id=owner_id,
        project_id=project_id,
        blueprint=_blueprint(),
        confirm=False,
    )
    head = service.get_blueprint(
        user_id=owner_id, project_id=project_id
    )
    assert head["current"]["version_status"] == "draft"
    assert head["confirmed"] is None
    assert database.get_writing_context(owner_id, chapter_id)[
        "story_blueprint"
    ] is None

    confirmed_id = service.save_blueprint(
        user_id=owner_id,
        project_id=project_id,
        blueprint=_blueprint(),
        confirm=True,
    )
    context = database.get_writing_context(owner_id, chapter_id)
    assert context["story_blueprint"]["central_question"].startswith(
        "父亲为何"
    )

    service.save_blueprint(
        user_id=owner_id,
        project_id=project_id,
        blueprint=_blueprint(question="林岚是否愿意公开全部真相？"),
        confirm=False,
    )
    head = service.get_blueprint(
        user_id=owner_id, project_id=project_id
    )
    assert head["has_unconfirmed_changes"] is True
    assert head["current"]["central_question"].startswith("林岚是否")
    assert head["confirmed_version_id"] == confirmed_id
    assert database.get_writing_context(owner_id, chapter_id)[
        "story_blueprint"
    ]["central_question"].startswith("父亲为何")

    restored_id = service.restore_blueprint_version(
        user_id=owner_id,
        project_id=project_id,
        version_id=draft_id,
    )
    versions = service.list_blueprint_versions(
        user_id=owner_id, project_id=project_id
    )
    assert versions[0]["id"] == restored_id
    assert versions[0]["source"] == "restore"
    assert versions[0]["revision"] == 4
    assert service.get_blueprint(
        user_id=other_id, project_id=project_id
    ) is None
    assert service.list_blueprint_versions(
        user_id=other_id, project_id=project_id
    ) == []
    with pytest.raises(ValueError, match="不存在"):
        service.restore_blueprint_version(
            user_id=other_id,
            project_id=project_id,
            version_id=confirmed_id,
        )


def test_confirmed_arcs_enter_planner_but_writer_only_gets_selected_arc(
    tmp_path: Path,
):
    database = Database(tmp_path / "arcs.db")
    database.initialize()
    user_id = database.create_user(
        "arc-owner", hash_password("password-123")
    )
    project_id = "arc-project"
    chapter_id = _project_with_chapter(
        database,
        tmp_path,
        user_id=user_id,
        project_id=project_id,
    )
    story = StoryPlanningService(database)
    story.save_blueprint(
        user_id=user_id,
        project_id=project_id,
        blueprint=_blueprint(),
        confirm=True,
    )
    mystery_id = story.create_arc(
        user_id=user_id,
        project_id=project_id,
        arc=_arc(
            title="父亲失踪之谜",
            payoff="确认父亲主动留下了一条可公开的证据链。",
            priority=5,
        ),
        confirm=True,
    )
    story.create_arc(
        user_id=user_id,
        project_id=project_id,
        arc=_arc(
            title="林岚与母亲的关系",
            payoff="两人承认彼此隐瞒都源于保护欲。",
            priority=4,
            arc_type="relationship",
        ),
        confirm=True,
    )
    story.create_arc(
        user_id=user_id,
        project_id=project_id,
        arc=_arc(
            title="未确认草稿线",
            payoff="这句话绝不能进入模型上下文。",
            priority=3,
        ),
        confirm=False,
    )
    PlanningService(database).upsert_task_card(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        volume_id=None,
        card=_confirmed_card(["父亲失踪之谜"]),
        confirm=True,
    )

    context = database.get_writing_context(user_id, chapter_id)
    raw_titles = {item["title"] for item in context["planned_plot_arcs"]}
    assert raw_titles == {"父亲失踪之谜", "林岚与母亲的关系"}
    plan_context = compile_story_plan_context(context, usage="plan")
    assert {item["title"] for item in plan_context["plot_arcs"]} == raw_titles
    write_context = compile_story_plan_context(context, usage="write")
    assert [item["title"] for item in write_context["plot_arcs"]] == [
        "父亲失踪之谜"
    ]

    messages = build_writing_messages(
        context=context,
        operation="draft",
        instruction="",
        current_content="",
        previous_content="",
    )
    prompt = messages[1]["content"]
    assert "确认父亲主动留下了一条可公开的证据链" in prompt
    assert "两人承认彼此隐瞒都源于保护欲" not in prompt
    assert "这句话绝不能进入模型上下文" not in prompt
    snapshot = build_writing_context_snapshot(
        context=context,
        operation="draft",
        instruction="",
        current_content="",
        previous_content="",
    )
    assert snapshot["confirmed_story_plan"]["plot_arcs"][0][
        "title"
    ] == "父亲失踪之谜"

    story.update_arc(
        user_id=user_id,
        project_id=project_id,
        arc_id=mystery_id,
        arc=_arc(
            title="父亲失踪之谜",
            payoff="未确认的新终局不能进入上下文。",
            priority=5,
        ),
        confirm=False,
    )
    context = database.get_writing_context(user_id, chapter_id)
    assert context["planned_plot_arcs"][0]["target_payoff"].startswith(
        "确认父亲主动"
    )

    story.archive_arc(
        user_id=user_id,
        project_id=project_id,
        arc_id=mystery_id,
    )
    context = database.get_writing_context(user_id, chapter_id)
    assert context["task_card"] is None
    write_context = compile_story_plan_context(context, usage="write")
    assert write_context["plot_arcs"] == []
    stored_card = PlanningService(database).get_task_card(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
    )
    assert stored_card["status"] == "draft"
    chapter = database.get_novel_chapter(
        user_id, project_id, chapter_id
    )
    assert chapter["needs_recheck"] == 1


def test_mock_and_deepseek_chapter_planners_receive_confirmed_story_plan(
    tmp_path: Path,
):
    context = {
        "chapter": {
            "project_title": "雾港来信",
            "genre": "悬疑",
            "premise": "追查来信。",
            "story_promise": "现实谜团。",
            "target_audience": "悬疑读者",
            "core_appeal": "证据",
            "ending_constraint": "解释真相",
            "world_setting": "海港",
            "style_guide": "克制",
            "point_of_view": "第三人称限知",
            "position": 1,
            "title": "迟到的信",
            "outline": "核对来信",
            "key_points": "",
            "target_chapter_chars": 3000,
        },
        "characters": [],
        "canonical_memory": {},
        "story_blueprint": _blueprint().model_dump(mode="json"),
        "planned_plot_arcs": [
            {
                "id": "arc-main",
                "position": 1,
                **_arc(
                    title="父亲失踪之谜",
                    payoff="确认证据链。",
                    priority=5,
                ).model_dump(mode="json"),
            }
        ],
    }

    async def scenario():
        mock = await MockChapterPlanner().propose(
            context=context,
            instruction="",
            provider_user_id="mock-user",
        )
        assert mock.result.plot_threads == ["父亲失踪之谜"]

        planner = DeepSeekChapterPlanner(
            replace(_settings(tmp_path), deepseek_api_key="sk-test-story")
        )
        captured = {}
        expected = _confirmed_card(["父亲失踪之谜"])

        async def fake_post(payload):
            captured["payload"] = payload
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": expected.model_dump_json()
                        },
                    }
                ],
                "usage": {},
            }

        planner._analyzer._post = fake_post
        try:
            await planner.propose(
                context=context,
                instruction="",
                provider_user_id="deepseek-user",
            )
        finally:
            await planner.close()
        serialized = json.dumps(captured["payload"], ensure_ascii=False)
        assert "父亲为何在失踪后继续寄信" in serialized
        assert "父亲失踪之谜" in serialized
        assert "author_confirmed_story_plan_only" in serialized

    asyncio.run(scenario())


def test_new_confirmed_blueprint_invalidates_future_confirmed_task_cards(
    tmp_path: Path,
):
    database = Database(tmp_path / "blueprint-invalidation.db")
    database.initialize()
    user_id = database.create_user(
        "blueprint-invalidation-owner", hash_password("password-123")
    )
    project_id = "blueprint-invalidation-project"
    chapter_id = _project_with_chapter(
        database,
        tmp_path,
        user_id=user_id,
        project_id=project_id,
    )
    story = StoryPlanningService(database)
    story.save_blueprint(
        user_id=user_id,
        project_id=project_id,
        blueprint=_blueprint(),
        confirm=True,
    )
    planning = PlanningService(database)
    planning.upsert_task_card(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        volume_id=None,
        card=_confirmed_card([]),
        confirm=True,
    )
    story.save_blueprint(
        user_id=user_id,
        project_id=project_id,
        blueprint=_blueprint(question="公开真相是否值得全部代价？"),
        confirm=False,
    )
    assert planning.get_task_card(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
    )["status"] == "confirmed"

    story.save_blueprint(
        user_id=user_id,
        project_id=project_id,
        blueprint=_blueprint(question="公开真相是否值得全部代价？"),
        confirm=True,
    )
    assert planning.get_task_card(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
    )["status"] == "draft"
    assert database.get_writing_context(user_id, chapter_id)[
        "task_card"
    ] is None


def test_story_blueprint_web_flow_and_task_card_picker(tmp_path: Path):
    application = create_app(_settings(tmp_path))
    with TestClient(application) as client:
        register = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "全书蓝图作者",
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
                "premise": "林岚收到失踪父亲署名的新信，决定返回雾港调查。",
                "csrf": _csrf(novel_form.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        workbench_url = response.headers["location"]
        project_id = workbench_url.split("/novels/", 1)[1].split("/", 1)[0]
        project_url = f"/novels/{project_id}"
        workspace = client.get(
            f"{workbench_url}?view=settings&settings_tab=structure"
        )
        assert "全书蓝图" in workspace.text

        blueprint_data = {
            "central_question": "父亲为何仍在寄信？",
            "protagonist_goal": "找到父亲。",
            "core_conflict": "档案部门持续抹除异常记录。",
            "stakes": "证人会失去作证机会。",
            "opening_state": "林岚拒绝回港。",
            "ending_state": "林岚公开证据。",
            "major_turns": "被迫回港\n证词被篡改",
            "must_payoffs": "解释所有来信",
            "forbidden_shortcuts": "不能突然口述全部真相",
            "author_notes": "",
            "action": "confirm",
            "csrf": _csrf(workspace.text),
        }
        response = client.post(
            f"{project_url}/story-blueprint",
            data=blueprint_data,
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert (
            "view=settings&settings_tab=structure&saved=true"
            in response.headers["location"]
        )

        workspace = client.get(workbench_url)
        arc_data = {
            "arc_type": "mystery",
            "title": "父亲失踪之谜",
            "dramatic_question": "父亲为何失踪？",
            "promise": "持续给出可核对线索。",
            "start_state": "只有一封信。",
            "target_payoff": "形成完整证据链。",
            "involved_characters": "林岚",
            "planned_turns": "确认邮戳\n推翻旧结论",
            "lifecycle_status": "planned",
            "priority": "5",
            "author_notes": "",
            "action": "confirm",
            "csrf": _csrf(workspace.text),
        }
        response = client.post(
            f"{project_url}/plot-arcs",
            data=arc_data,
            follow_redirects=False,
        )
        assert response.status_code == 303

        workspace = client.get(workbench_url)
        response = client.post(
            f"{project_url}/chapters",
            data={
                "title": "第一章 迟到的信",
                "outline": "林岚核对来信。",
                "key_points": "确认邮戳",
                "csrf": _csrf(workspace.text),
            },
            follow_redirects=False,
        )
        chapter_location = response.headers["location"]
        chapter_id = chapter_location.split("chapter_id=", 1)[1]
        chapter_url = f"{project_url}/chapters/{chapter_id}"
        task_url = f"{chapter_url}/task-card"
        task_page = client.get(task_url)
        assert 'value="父亲失踪之谜"' in task_page.text
        assert "本次将读取确认版全书规划" in task_page.text

        response = client.post(
            task_url,
            data={
                "selected_plot_threads": "父亲失踪之谜",
                "purpose": "迫使林岚回港。",
                "start_state": "拒绝相信。",
                "end_state": "买下车票。",
                "central_conflict": "邮戳与结论冲突。",
                "ending_hook": "盐味来自旧码头。",
                "target_chars": "3000",
                "scene_goal_1": "核对来信",
                "scene_obstacle_1": "邮戳异常",
                    "scene_action_1": "比对旧信",
                    "scene_end_state_1": "承认值得调查",
                    "scene_requirement_1": "plot_thread:0",
                    "scene_goal_2": "决定回港",
                "scene_obstacle_2": "抗拒故乡",
                    "scene_action_2": "购买车票",
                    "scene_end_state_2": "踏上列车",
                    "scene_requirement_2": "ending_hook:0",
                "action": "confirm",
                "csrf": _csrf(task_page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        database = application.state.database
        user = database.get_user_by_username("全书蓝图作者")
        context = database.get_writing_context(int(user["id"]), chapter_id)
        assert context["task_card"]["plot_threads"] == ["父亲失踪之谜"]
        assert context["story_blueprint"]["central_question"] == (
            "父亲为何仍在寄信？"
        )


def test_project_delete_cascades_story_plan_without_fk_errors(tmp_path: Path):
    database = Database(tmp_path / "delete.db")
    database.initialize()
    user_id = database.create_user(
        "delete-owner", hash_password("password-123")
    )
    project_id = "delete-project"
    _project_with_chapter(
        database,
        tmp_path,
        user_id=user_id,
        project_id=project_id,
    )
    service = StoryPlanningService(database)
    service.save_blueprint(
        user_id=user_id,
        project_id=project_id,
        blueprint=_blueprint(),
        confirm=True,
    )
    service.create_arc(
        user_id=user_id,
        project_id=project_id,
        arc=_arc(
            title="父亲失踪之谜",
            payoff="确认真相。",
            priority=5,
        ),
        confirm=True,
    )
    with database.connection() as connection:
        connection.execute("DELETE FROM users WHERE id=?", (user_id,))
        connection.commit()
        assert connection.execute(
            "SELECT COUNT(*) FROM novel_story_blueprint_versions"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM novel_plot_arc_versions"
        ).fetchone()[0] == 0
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_latest_schema_migrates_v13_without_mutating_existing_project(
    tmp_path: Path,
):
    path = tmp_path / "legacy-v13.db"
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
    connection.commit()
    applied_at = utc_now()
    for migration in MIGRATIONS[:13]:
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
        VALUES ('legacy-blueprint-user', 'hash', ?)
        """,
        (applied_at,),
    )
    user_id = int(connection.execute("SELECT last_insert_rowid()").fetchone()[0])
    connection.execute(
        """
        INSERT INTO novel_projects(
            id, user_id, title, genre, premise, created_at, updated_at
        ) VALUES ('legacy-blueprint-project', ?, '旧项目', '悬疑',
                  '已有长篇项目不能被迁移修改。', ?, ?)
        """,
        (user_id, applied_at, applied_at),
    )
    connection.commit()
    connection.close()

    database = Database(path)
    database.initialize()
    project = database.get_novel_project(
        user_id, "legacy-blueprint-project"
    )
    assert project["title"] == "旧项目"
    with database.connection() as migrated:
        assert migrated.execute(
            "SELECT MAX(version) FROM schema_migrations"
                ).fetchone()[0] == 30
        assert migrated.execute(
            "SELECT COUNT(*) FROM novel_story_blueprint_versions"
        ).fetchone()[0] == 0
        assert migrated.execute(
            "SELECT COUNT(*) FROM novel_plot_arcs"
        ).fetchone()[0] == 0
        assert migrated.execute(
            "SELECT COUNT(*) FROM novel_chapter_causal_links"
        ).fetchone()[0] == 0
        assert migrated.execute(
            "SELECT COUNT(*) FROM novel_causal_link_suggestions"
        ).fetchone()[0] == 0
        assert migrated.execute(
            "SELECT COUNT(*) FROM novel_causal_link_suggestion_reviews"
        ).fetchone()[0] == 0
        assert migrated.execute(
            "SELECT COUNT(*) FROM novel_causal_branch_adoptions"
        ).fetchone()[0] == 0
        assert migrated.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert migrated.execute("PRAGMA foreign_key_check").fetchall() == []
