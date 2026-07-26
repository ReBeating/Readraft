import hashlib
import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.db import Database
from app.main import create_app
from app.memory_service import MemoryService
from app.planning_schema import (
    ChapterTaskCard,
    allocate_scene_requirement_refs,
)
from app.planning_service import PlanningService
from app.scene_service import SceneService
from app.security import hash_password
from app.style_service import StyleService
from app.workflow import ChapterWorkflowService


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        app_name="章节执行流程测试",
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


def csrf_from(html: str) -> str:
    return html.split('name="csrf" value="', 1)[1].split('"', 1)[0]


def task_card() -> ChapterTaskCard:
    card = ChapterTaskCard.model_validate(
        {
            "purpose": "林岚确认异常来信值得追查，并决定返回雾港。",
            "start_state": "林岚认为父亲已经失踪十年。",
            "end_state": "林岚带着新线索踏上返程列车。",
            "central_conflict": "近期邮戳与既有事实相互冲突。",
            "emotional_value": "被压下的希望重新出现。",
            "plot_threads": ["父亲失踪主线"],
            "must_happen": ["核对邮戳", "购买车票"],
            "must_preserve": ["林岚不知道寄信人身份"],
            "forbidden": ["揭晓父亲下落"],
            "ending_hook": "信纸带着旧码头的盐味。",
            "target_chars": 3000,
            "scenes": [
                {
                    "pov_character": "林岚",
                    "goal": "确认来信真假",
                    "obstacle": "邮戳日期不可能",
                    "action": "对照旧信并致电邮局",
                    "reveal": "投递记录确实存在",
                    "conceal": "寄信人身份",
                    "subtext": "她不愿承认自己仍抱有希望",
                    "location": "林岚住处",
                    "key_items": ["来信", "旧信"],
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
                    "subtext": "买票等于承认旧案尚未结束",
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
    database: Database, tmp_path: Path, *, username: str
) -> tuple[int, str, str, Path]:
    user_id = database.create_user(
        username, hash_password("password-123")
    )
    project_id = f"{username}-project"
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="雾港来信",
        genre="现实悬疑",
        premise="林岚收到失踪父亲寄来的新信件。",
        world_setting="当代海港小城。",
        style_guide="第三人称限知，克制具体。",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
        story_promise="每章推进一项可验证证据。",
    )
    chapter_id = f"{username}-chapter"
    chapter_dir = (
        tmp_path
        / "novels"
        / str(user_id)
        / project_id
        / "chapters"
        / chapter_id
    )
    (chapter_dir / "versions").mkdir(parents=True)
    content_path = chapter_dir / "content.txt"
    content_path.write_text("", encoding="utf-8")
    database.add_novel_chapter(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        title="第一章 迟到的信",
        outline="林岚确认来信并返回雾港。",
        key_points="核对邮戳\n购买车票",
        content_path=content_path,
    )
    return user_id, project_id, chapter_id, content_path


def confirm_voice(
    style_service: StyleService, *, user_id: int, project_id: str
) -> None:
    assert style_service.update_voice_profile(
        user_id=user_id,
        project_id=project_id,
        narration_rules="第三人称紧贴林岚，只写她可观察或推断的内容。",
        sentence_rhythm="调查段落短句推进，转折前允许一个长句。",
        dialogue_voice="林岚用追问回避情绪。",
        sensory_palette="盐雾、旧金属和潮湿纸张。",
        metaphor_policy="低密度，只用人物经验范围内的意象。",
        allowed_omissions="不解释恐惧，让动作与停顿承担。",
        preferred_patterns=["动作先于情绪判断"],
        banned_expressions=["内心深处"],
        author_notes="",
        confirm=True,
    )


def make_ready_scene_versions(
    database: Database,
    scene_service: SceneService,
    *,
    user_id: int,
    project_id: str,
    chapter_id: str,
    content_path: Path,
) -> None:
    workbench = scene_service.get_workbench(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
    )
    for scene in workbench["scenes"]:
        scene_id = str(scene["id"])
        version_path = (
            content_path.parent
            / "scenes"
            / scene_id
            / "versions"
            / "manual.txt"
        )
        version_path.parent.mkdir(parents=True)
        content = (
            f"林岚执行第 {scene['position']} 个场景的具体行动，"
            "她核对证据，遇到阻力，并让局面发生可观察变化。"
        ) * 60
        version_path.write_text(content, encoding="utf-8")
        version_id = scene_service.record_manual_version(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            scene_beat_id=scene_id,
            version_path=version_path,
            content=content,
        )
        with database.connection() as connection:
            connection.execute(
                """
                UPDATE novel_scene_versions
                SET quality_status='pass', hard_issue_count=0
                WHERE id=?
                """,
                (version_id,),
            )
            connection.commit()


def assemble_candidate(
    scene_service: SceneService,
    *,
    user_id: int,
    project_id: str,
    chapter_id: str,
    content_path: Path,
) -> str:
    assembly = scene_service.build_assembly(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
    )
    version_path = content_path.parent / "versions" / "assembly.txt"
    version_path.write_text(str(assembly["content"]), encoding="utf-8")
    content_path.write_text(str(assembly["content"]), encoding="utf-8")
    return scene_service.record_assembly(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        version_path=version_path,
        content=str(assembly["content"]),
        scene_versions=assembly["scene_versions"],
    )


def test_chapter_workflow_derives_every_author_gate(tmp_path: Path):
    database = Database(tmp_path / "workflow.db")
    database.initialize()
    user_id, project_id, chapter_id, content_path = create_story(
        database, tmp_path, username="workflow-author"
    )
    planning_service = PlanningService(database)
    scene_service = SceneService(database)
    memory_service = MemoryService(database)
    style_service = StyleService(database)
    workflow = ChapterWorkflowService(
        database,
        planning_service=planning_service,
        scene_service=scene_service,
        memory_service=memory_service,
        style_service=style_service,
    )

    state = workflow.get_state(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
    )
    assert state["stage"] == "task_card"
    assert state["completed_count"] == 0
    assert workflow.get_state(
        user_id=user_id + 999,
        project_id=project_id,
        chapter_id=chapter_id,
    ) is None

    planning_service.upsert_task_card(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        volume_id=None,
        card=task_card(),
        confirm=True,
    )
    state = workflow.get_state(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
    )
    assert state["stage"] == "scenes"
    assert state["steps"][0]["status"] == "complete"
    assert "/scenes#scene-" in state["primary_action"]["url"]

    first_scene = scene_service.get_workbench(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
    )["scenes"][0]
    job_id = database.create_generation_job(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        operation="generate_scene",
        instruction="",
        provider="mock",
        model="mock-scene-writer",
        credential_source="default",
        subject_id=str(first_scene["id"]),
        max_jobs_per_day=50,
    )
    state = workflow.get_state(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
    )
    assert state["stage"] == "running"
    assert state["primary_action"]["url"] == f"/writing-jobs/{job_id}"
    with database.connection() as connection:
        connection.execute(
            """
            UPDATE generation_jobs
            SET status='failed', error='测试结束排队状态'
            WHERE id=?
            """,
            (job_id,),
        )
        connection.commit()

    make_ready_scene_versions(
        database,
        scene_service,
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        content_path=content_path,
    )
    state = workflow.get_state(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
    )
    assert state["stage"] == "assembly"
    assert "组装 2 个已通过场景" == state["primary_action"]["label"]
    assert state["primary_action"]["method"] == "post"
    assert state["primary_action"]["url"].endswith("/scenes/assemble")

    version_id = assemble_candidate(
        scene_service,
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        content_path=content_path,
    )
    state = workflow.get_state(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
    )
    assert state["stage"] == "hard_audit"
    assert state["working_version_id"] == version_id

    with database.connection() as connection:
        connection.execute(
            """
            UPDATE novel_chapter_versions
            SET quality_status='pass', hard_issue_count=0
            WHERE id=?
            """,
            (version_id,),
        )
        connection.commit()
    state = workflow.get_state(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
    )
    assert state["stage"] == "style_profile"
    assert "确认作品声纹" in state["primary_action"]["label"]
    confirm_voice(
        style_service, user_id=user_id, project_id=project_id
    )
    state = workflow.get_state(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
    )
    assert state["stage"] == "style_audit"

    with database.connection() as connection:
        connection.execute(
            "UPDATE novel_chapter_versions SET style_status='pass' WHERE id=?",
            (version_id,),
        )
        connection.commit()
    state = workflow.get_state(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
    )
    assert state["stage"] == "canon"

    accepted = database.accept_chapter_version(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        version_id=version_id,
    )
    assert accepted
    state = workflow.get_state(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
    )
    assert state["stage"] == "memory_extract"
    assert state["primary_action"]["method"] == "post"

    delta_id = memory_service.create_proposal(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        version_id=version_id,
        payload={"chapter_summary": "林岚确认来信异常并踏上返程列车。"},
    )
    state = workflow.get_state(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
    )
    assert state["stage"] == "memory_review"
    assert state["canonical_delta_id"] == delta_id

    projected = memory_service.accept_delta(
        user_id=user_id, delta_id=delta_id
    )
    assert projected and projected["projected"]
    state = workflow.get_state(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
    )
    assert state["stage"] == "complete"
    assert state["completed_count"] == state["step_count"] == 6


def test_empty_project_opens_unified_settings_workbench(tmp_path: Path):
    application = create_app(make_settings(tmp_path))
    with TestClient(application) as client:
        register_page = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "新书作者",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": csrf_from(register_page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        novel_page = client.get("/novels/new")
        response = client.post(
            "/novels/new",
            data={
                "title": "还没有第一章",
                "genre": "悬疑",
                "premise": "林岚收到一封日期异常、来源不明的来信。",
                "csrf": csrf_from(novel_page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        workbench_url = response.headers["location"]

        workspace = client.get(workbench_url)

        assert workspace.status_code == 200
        assert workbench_url.endswith("/workbench")
        assert "还没有章节。" in workspace.text
        assert "＋ 新建章节" in workspace.text
        assert 'class="studio-archive-view"' in workspace.text
        assert "创作设定" in workspace.text
        assert "分析与笔记" in workspace.text
        assert "版本" in workspace.text


def test_workflow_ui_and_memory_retry_route(tmp_path: Path):
    application = create_app(make_settings(tmp_path))
    with TestClient(application) as client:
        register_page = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "流程网页作者",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": csrf_from(register_page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        database = application.state.database
        user = database.get_user_by_username("流程网页作者")
        user_id = int(user["id"])
        project_id = "workflow-web-project"
        chapter_id = "workflow-web-chapter"
        database.create_novel_project(
            user_id=user_id,
            project_id=project_id,
            title="流程网页验收",
            genre="悬疑",
            premise="林岚收到一封日期异常的来信。",
            world_setting="当代旧港。",
            style_guide="克制具体。",
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
        (chapter_dir / "versions").mkdir(parents=True)
        content_path = chapter_dir / "content.txt"
        content = "林岚逐项核对来信与旧档案，并在阻力中作出返程决定。" * 120
        content_path.write_text(content, encoding="utf-8")
        database.add_novel_chapter(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            title="第一章 来信",
            outline="林岚确认来信异常并决定回港。",
            key_points="核对来信\n购买车票",
            content_path=content_path,
        )
        version_path = chapter_dir / "versions" / "manual.txt"
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
        with database.connection() as connection:
            connection.execute(
                """
                UPDATE novel_chapter_versions
                SET quality_status='pass', hard_issue_count=0,
                    style_status='pass'
                WHERE id=?
                """,
                (version_id,),
            )
            connection.commit()
        assert database.accept_chapter_version(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            version_id=version_id,
        )

        chapter_url = f"/novels/{project_id}/chapters/{chapter_id}"
        page = client.get(chapter_url)
        assert page.status_code == 200
        assert "CHAPTER EXECUTION" in page.text
        assert "正史已确认，等待建立可检索记忆" in page.text
        assert "提取 Story Delta" in page.text
        extract_url = (
            f"{chapter_url}/versions/{version_id}/extract-memory"
        )
        assert f'action="{extract_url}"' in page.text

        workspace = client.get(
            f"/novels/{project_id}/workbench?chapter_id={chapter_id}"
        )
        assert workspace.status_code == 200
        assert "data-chapter-workflow" in workspace.text
        assert "正史已确认，等待建立可检索记忆" in workspace.text

        response = client.post(
            extract_url,
            data={"csrf": csrf_from(page.text)},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert response.headers["location"].startswith("/writing-jobs/")
        job_id = response.headers["location"].rsplit("/", 1)[-1]
        deadline = time.monotonic() + 3
        payload = {}
        while time.monotonic() < deadline:
            payload = client.get(f"/api/writing-jobs/{job_id}").json()
            if payload.get("terminal"):
                break
            time.sleep(0.03)
        assert payload["status"] == "completed"
        assert payload["redirect_url"].startswith("/story-deltas/")

        page = client.get(chapter_url)
        assert "最后核对本章造成的故事变化" in page.text
        assert "审核 Story Delta" in page.text
