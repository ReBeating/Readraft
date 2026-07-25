import asyncio
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.credentials import CredentialCipher
from app.db import Database
from app.deepseek import MockAnalyzer
from app.main import create_app
from app.planning_schema import (
    ChapterTaskCard,
    allocate_scene_requirement_refs,
)
from app.planning_service import PlanningService
from app.scene_service import SceneService
from app.security import hash_password
from app.worker import AnalysisWorker
from app.writing import MockWriter


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        app_name="场景工作台测试",
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


def task_card(*, first_goal: str = "确认来信真假") -> ChapterTaskCard:
    card = ChapterTaskCard.model_validate(
        {
            "purpose": "迫使林岚返回雾港。",
            "start_state": "林岚认定父亲已经死亡。",
            "end_state": "林岚踏上返回雾港的列车。",
            "central_conflict": "近期邮戳与失踪十年的事实冲突。",
            "emotional_value": "被压下的希望重新出现。",
            "plot_threads": ["父亲失踪之谜"],
            "must_happen": ["确认邮戳", "购买车票"],
            "must_preserve": ["林岚不知道寄信人身份"],
            "forbidden": ["揭晓父亲下落"],
            "ending_hook": "信纸带着旧码头的气味。",
            "target_chars": 3000,
            "scenes": [
                {
                    "pov_character": "林岚",
                    "goal": first_goal,
                    "obstacle": "邮戳日期不可能",
                    "action": "对照旧信并致电邮局",
                    "reveal": "投递记录确实存在",
                    "conceal": "寄信人身份",
                    "subtext": "她不愿承认自己仍抱有希望",
                    "location": "林岚住处",
                    "key_items": ["来信", "父亲旧信"],
                    "end_state": "林岚承认信件值得调查",
                    "transition": "查询当夜车票",
                },
                {
                    "pov_character": "林岚",
                    "goal": "决定是否返回雾港",
                    "obstacle": "她抗拒回到故乡",
                    "action": "购买车票并收起旧信",
                    "reveal": "信纸带有海水气味",
                    "conceal": "气味来源",
                    "subtext": "买票等于承认父亲的案子尚未结束",
                    "location": "车站",
                    "key_items": ["车票", "来信"],
                    "end_state": "林岚踏上列车",
                    "transition": "列车驶入雾区",
                },
            ],
        }
    )
    return allocate_scene_requirement_refs(card)


def create_story(
    database: Database, tmp_path: Path, *, username: str = "scene-author"
) -> tuple[int, str, str, Path]:
    user_id = database.create_user(
        username, hash_password("password-123")
    )
    project_id = "scene-project"
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="雾港来信",
        genre="悬疑",
        premise="记者收到失踪父亲寄出的新信件。",
        world_setting="当代海港小城。",
        style_guide="克制具体，以行动和证据推进。",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
        story_promise="每章推进一项可验证的新证据。",
        ending_constraint="必须解释来信来源。",
    )
    chapter_id = "scene-chapter"
    chapter_dir = (
        tmp_path / "novels" / str(user_id) / project_id / "chapters"
        / chapter_id
    )
    chapter_dir.mkdir(parents=True)
    content_path = chapter_dir / "content.txt"
    content_path.write_text("", encoding="utf-8")
    database.add_novel_chapter(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        title="第一章 迟到的信",
        outline="林岚收到来信并决定返回雾港。",
        key_points="确认邮戳\n购买车票",
        content_path=content_path,
    )
    PlanningService(database).upsert_task_card(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        volume_id=None,
        card=task_card(),
        confirm=True,
    )
    return user_id, project_id, chapter_id, content_path


def test_scene_identity_survives_plan_edit_and_draft_becomes_stale(
    tmp_path: Path,
):
    database = Database(tmp_path / "scene.db")
    database.initialize()
    user_id, project_id, chapter_id, content_path = create_story(
        database, tmp_path
    )
    scenes = SceneService(database)
    before = scenes.get_workbench(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
    )
    first = before["scenes"][0]
    scene_path = (
        content_path.parent
        / "scenes"
        / str(first["id"])
        / "versions"
        / "manual.txt"
    )
    scene_path.parent.mkdir(parents=True)
    scene_text = "林岚把两封信并排放在桌上。" * 80
    scene_path.write_text(scene_text, encoding="utf-8")
    scenes.record_manual_version(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        scene_beat_id=str(first["id"]),
        version_path=scene_path,
        content=scene_text,
    )

    PlanningService(database).upsert_task_card(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        volume_id=None,
        card=task_card(first_goal="确认来信并查明邮戳记录"),
        confirm=True,
    )
    after = scenes.get_workbench(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
    )
    assert after["scenes"][0]["id"] == first["id"]
    assert after["scenes"][0]["is_stale"]
    assert after["scenes"][0]["content"] == scene_text
    assert not after["all_ready"]


def test_scene_worker_generates_audits_and_assembles_candidate(
    tmp_path: Path,
):
    settings = make_settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    user_id, project_id, chapter_id, content_path = create_story(
        database, tmp_path
    )
    scene_service = SceneService(database)
    workbench = scene_service.get_workbench(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
    )
    async def process(claimed):
        worker = AnalysisWorker(
            database,
            MockAnalyzer(),
            MockWriter(),
            settings.secret_key,
            settings,
            CredentialCipher(settings.credential_secret),
            poll_seconds=0.01,
        )
        await worker._process_generation(claimed)

    for scene in workbench["scenes"]:
        job_id = database.create_generation_job(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            operation="generate_scene",
            instruction="让线索先改变行动，不提前解释全部来源。",
            provider="mock",
            model="mock-novel-writer",
            credential_source="default",
            subject_id=str(scene["id"]),
            max_jobs_per_day=50,
        )
        claimed = database.claim_next_generation()
        assert claimed and claimed["id"] == job_id
        asyncio.run(process(claimed))
        job = database.get_generation_job(user_id, job_id)
        assert job["status"] == "completed"
        snapshot = json.loads(job["context_snapshot_json"])
        assert snapshot["operation"] == "generate_scene"
        assert snapshot["focused_scene"]["id"] == scene["id"]
        assert snapshot["focused_scene"]["plan_fingerprint"]
        assert len(snapshot["scene_sequence"]) == 2
        assert snapshot["canonical_memory"]["retrieval"]["engine"].startswith(
            "sqlite_fts5"
        )
        assert snapshot["canonical_memory"]["retrieval"]["query_terms"]
        assert snapshot["canonical_memory"]["retrieval"]["query_concepts"]

    workbench = scene_service.get_workbench(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
    )
    assert workbench["ready_count"] == 2
    assert workbench["all_ready"]
    assert all(
        scene["quality_status"] == "pass"
        for scene in workbench["scenes"]
    )
    assert all(
        scene["audit"]["report"]["scene_coverage"]
        for scene in workbench["scenes"]
    )

    assembly = scene_service.build_assembly(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
    )
    assert assembly["content"].count("场景结束时") == 2
    version_path = content_path.parent / "versions" / "assembled.txt"
    version_path.parent.mkdir(exist_ok=True)
    version_path.write_text(assembly["content"], encoding="utf-8")
    content_path.write_text(assembly["content"], encoding="utf-8")
    version_id = scene_service.record_assembly(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        version_path=version_path,
        content=assembly["content"],
        scene_versions=assembly["scene_versions"],
    )
    chapter = database.get_novel_chapter(
        user_id, project_id, chapter_id
    )
    assert chapter["working_version_id"] == version_id
    assert chapter["canonical_version_id"] is None
    version = database.get_chapter_version(
        user_id, project_id, chapter_id, version_id
    )
    assert version["source"] == "scene_assembly"
    assert version["quality_status"] == "pending"
    with database.connection() as connection:
        items = connection.execute(
            """
            SELECT position, scene_version_id
            FROM novel_scene_assembly_items
            WHERE chapter_version_id=?
            ORDER BY position
            """,
            (version_id,),
        ).fetchall()
    assert [int(row["position"]) for row in items] == [1, 2]
    with database.connection() as connection:
        connection.execute("DELETE FROM users WHERE id=?", (user_id,))
        connection.commit()
        assert not connection.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()


def test_manual_scene_requires_audit_and_author_can_record_override(
    tmp_path: Path,
):
    settings = make_settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    user_id, project_id, chapter_id, content_path = create_story(
        database, tmp_path
    )
    scene_service = SceneService(database)
    scene = scene_service.get_workbench(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
    )["scenes"][0]
    scene_id = str(scene["id"])
    manual_path = (
        content_path.parent
        / "scenes"
        / scene_id
        / "versions"
        / "manual-block.txt"
    )
    manual_path.parent.mkdir(parents=True)
    content = ("林岚反复核对邮戳与旧信。她没有提前下结论。" * 45) + (
        "[[AUDIT_BLOCK]]"
    )
    manual_path.write_text(content, encoding="utf-8")
    version_id = scene_service.record_manual_version(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        scene_beat_id=scene_id,
        version_path=manual_path,
        content=content,
    )
    assert scene_service.get_workbench(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
    )["scenes"][0]["quality_status"] == "pending"

    job_id = database.create_generation_job(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        operation="audit_scene",
        instruction="",
        provider="mock",
        model="mock-hard-auditor",
        credential_source="default",
        subject_id=version_id,
        max_jobs_per_day=50,
    )
    claimed = database.claim_next_generation()

    async def run_audit():
        worker = AnalysisWorker(
            database,
            MockAnalyzer(),
            MockWriter(),
            settings.secret_key,
            settings,
            CredentialCipher(settings.credential_secret),
            poll_seconds=0.01,
        )
        await worker._process_generation(claimed)

    asyncio.run(run_audit())
    checked = scene_service.get_workbench(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
    )["scenes"][0]
    assert checked["quality_status"] == "block"
    assert checked["version_hard_issue_count"] == 1
    assert not checked["ready"]
    assert scene_service.override_audit(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        scene_beat_id=scene_id,
        scene_version_id=version_id,
        reason="该标记用于作者主动测试门禁，正式组装前会在整章编辑中删除。",
    )
    overridden = scene_service.get_workbench(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
    )["scenes"][0]
    assert overridden["quality_status"] == "overridden"
    assert overridden["ready"]


def test_scene_workbench_web_flow(tmp_path: Path):
    settings = make_settings(tmp_path)
    application = create_app(settings)
    with TestClient(application) as client:
        register = client.get("/register")
        csrf = register.text.split('name="csrf" value="', 1)[1].split(
            '"', 1
        )[0]
        response = client.post(
            "/register",
            data={
                "username": "场景网页作者",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": csrf,
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        user = application.state.database.get_user_by_username(
            "场景网页作者"
        )
        user_id = int(user["id"])
        project_id = "web-scene-project"
        chapter_id = "web-scene-chapter"
        database = application.state.database
        database.create_novel_project(
            user_id=user_id,
            project_id=project_id,
            title="场景网页验收",
            genre="悬疑",
            premise="调查一封日期异常的来信。",
            world_setting="当代旧港。",
            style_guide="克制具体。",
            point_of_view="第三人称限知",
            target_chapter_chars=3000,
        )
        chapter_dir = (
            tmp_path / "novels" / str(user_id) / project_id / "chapters"
            / chapter_id
        )
        chapter_dir.mkdir(parents=True)
        content_path = chapter_dir / "content.txt"
        content_path.write_text("", encoding="utf-8")
        database.add_novel_chapter(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            title="第一章 来信",
            outline="林岚收到来信并改道调查。",
            key_points="核对邮戳",
            content_path=content_path,
        )
        PlanningService(database).upsert_task_card(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            volume_id=None,
            card=task_card(),
            confirm=True,
        )

        scene_url = (
            f"/novels/{project_id}/chapters/{chapter_id}/scenes"
        )
        page = client.get(scene_url)
        assert page.status_code == 200
        assert "场景工作台" in page.text
        assert "0 / 2" in page.text
        csrf = page.text.split('name="csrf" value="', 1)[1].split('"', 1)[0]
        workbench = SceneService(database).get_workbench(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
        )
        for scene in workbench["scenes"]:
            response = client.post(
                f"{scene_url}/{scene['id']}/generate",
                data={
                    "operation": "generate_scene",
                    "instruction": "",
                    "csrf": csrf,
                },
                follow_redirects=False,
            )
            assert response.status_code == 303
            job_id = response.headers["location"].rsplit("/", 1)[-1]
            deadline = time.monotonic() + 3
            payload = {}
            while time.monotonic() < deadline:
                payload = client.get(
                    f"/api/writing-jobs/{job_id}"
                ).json()
                if payload.get("terminal"):
                    break
                time.sleep(0.03)
            assert payload["status"] == "completed"
            assert payload["redirect_url"] == scene_url
            page = client.get(scene_url)
            csrf = page.text.split(
                'name="csrf" value="', 1
            )[1].split('"', 1)[0]

        page = client.get(scene_url)
        assert "2 / 2" in page.text
        assert "按顺序组装 2 个场景" in page.text
        response = client.post(
            f"{scene_url}/assemble",
            data={
                "csrf": page.text.split(
                    'name="csrf" value="', 1
                )[1].split('"', 1)[0]
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert "scene_assembled=true" in response.headers["location"]
        chapter_page = client.get(response.headers["location"])
        assert "形成新的章节候选稿" in chapter_page.text
        assert "场景组装" in chapter_page.text
