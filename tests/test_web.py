import hashlib
import json
import re
import time
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.planning_service import PlanningService


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        app_name="测试拆文台",
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


def test_full_mock_workflow(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as client:
        page = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "测试者",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": csrf_from(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        page = client.get("/upload")
        response = client.post(
            "/upload",
            data={"title": "测试小说", "csrf": csrf_from(page.text)},
            files={
                "source_file": (
                    "novel.txt",
                    "第一章 开始\n这是第一章的正文内容。\n第二章 继续\n这是第二章的正文内容。".encode(),
                    "text/plain",
                )
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        document_url = response.headers["location"]

        page = client.get(document_url)
        response = client.post(
            f"{document_url}/analyze",
            data={"csrf": csrf_from(page.text)},
            follow_redirects=False,
        )
        assert response.status_code == 303
        job_url = response.headers["location"]
        job_id = job_url.rsplit("/", 1)[-1]

        deadline = time.monotonic() + 3
        payload = {}
        while time.monotonic() < deadline:
            payload = client.get(f"/api/jobs/{job_id}").json()
            if payload.get("terminal"):
                break
            time.sleep(0.03)

        assert payload["status"] == "completed"
        assert payload["completed"] == 2
        job_page = client.get(job_url)
        assert job_page.status_code == 200
        assert "已完成" in job_page.text
        analysis_match = re.search(r'href="/analyses/([^"]+)"', job_page.text)
        assert analysis_match
        analysis_page = client.get(f"/analyses/{analysis_match.group(1)}")
        assert analysis_page.status_code == 200
        assert "章节摘要" in analysis_page.text
        export = client.get(f"/jobs/{job_id}/export.json")
        assert export.status_code == 200
        assert len(export.json()["chapters"]) == 2


def test_scene_only_planner_locks_task_card_and_maps_requirements(tmp_path):
    application = create_app(make_settings(tmp_path))
    with TestClient(application) as client:
        register = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "场景拆解作者",
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
                "premise": "林岚核对异常来信并返回雾港。",
                "csrf": csrf_from(novel_form.text),
            },
            follow_redirects=False,
        )
        project_url = response.headers["location"]
        project_id = project_url.rsplit("/", 1)[-1]
        workspace = client.get(project_url)
        response = client.post(
            f"{project_url}/chapters",
            data={
                "title": "第一章 迟到的信",
                "outline": "林岚核对来信。",
                "key_points": "确认邮戳",
                "csrf": csrf_from(workspace.text),
            },
            follow_redirects=False,
        )
        chapter_url = response.headers["location"]
        chapter_id = chapter_url.rsplit("/", 1)[-1]
        task_url = f"{chapter_url}/task-card"
        task_page = client.get(task_url)
        response = client.post(
            task_url,
            data={
                "custom_plot_threads": "父亲失踪之谜",
                "purpose": "迫使林岚回到雾港。",
                "start_state": "林岚拒绝相信来信。",
                "end_state": "林岚买下返回雾港的车票。",
                "central_conflict": "新邮戳与十年前结论冲突。",
                "emotional_value": "希望与戒备同时抬升。",
                "must_happen": "确认邮戳\n购买车票",
                "must_preserve": "林岚不知道寄信人",
                "forbidden": "揭晓父亲下落",
                "foreshadow_setup": "信纸有海水气味",
                "foreshadow_payoff": "",
                "ending_hook": "气味来自雾港旧码头。",
                "target_chars": "3000",
                "action": "save_draft",
                "csrf": csrf_from(task_page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        database = application.state.database
        user = database.get_user_by_username("场景拆解作者")
        planning = PlanningService(database)
        before = planning.get_task_card(
            user_id=int(user["id"]),
            project_id=project_id,
            chapter_id=chapter_id,
        )
        task_page = client.get(task_url)
        assert "只拆场景，不改章节要求" in task_page.text
        assert "锁定 2 项必做事件" in task_page.text
        response = client.post(
            f"{task_url}/generate-scenes",
            data={
                "instruction": "让决定返乡发生在第二场。",
                "csrf": csrf_from(task_page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        job_id = response.headers["location"].rsplit("/", 1)[-1]
        deadline = time.monotonic() + 3
        payload = {}
        while time.monotonic() < deadline:
            payload = client.get(f"/api/writing-jobs/{job_id}").json()
            if payload.get("terminal"):
                break
            time.sleep(0.03)
        assert payload["status"] == "completed"
        assert payload["redirect_url"] == f"{task_url}#scene-beats"

        after = planning.get_task_card(
            user_id=int(user["id"]),
            project_id=project_id,
            chapter_id=chapter_id,
        )
        for field in (
            "purpose",
            "start_state",
            "end_state",
            "central_conflict",
            "emotional_value",
            "plot_threads",
            "must_happen",
            "must_preserve",
            "forbidden",
            "foreshadow_setup",
            "foreshadow_payoff",
            "ending_hook",
            "target_chars",
        ):
            assert after[field] == before[field]
        assert after["status"] == "draft"
        assert len(after["scenes"]) == 2
        expected = {
            ("plot_thread", "父亲失踪之谜"),
            ("must_happen", "确认邮戳"),
            ("must_happen", "购买车票"),
            ("foreshadow_setup", "信纸有海水气味"),
            ("ending_hook", "气味来自雾港旧码头。"),
        }
        actual = {
            (item["kind"], item["text"])
            for scene in after["scenes"]
            for item in scene["requirement_refs"]
        }
        assert actual == expected
        job = database.get_generation_job(int(user["id"]), job_id)
        snapshot = json.loads(str(job["context_snapshot_json"]))
        assert snapshot["operation"] == "plan_scene_beats"
        assert snapshot["locked_task_card"]["must_happen"] == [
            "确认邮戳",
            "购买车票",
        ]
        assert snapshot["task_card_fingerprint"]

        task_page = client.get(payload["redirect_url"])
        assert "这个场景负责落实哪些任务要求" in task_page.text
        assert 'value="must_happen:0"' in task_page.text
        requirement_sources = {
            "plot_thread": after["plot_threads"],
            "must_happen": after["must_happen"],
            "foreshadow_setup": after["foreshadow_setup"],
            "foreshadow_payoff": after["foreshadow_payoff"],
            "ending_hook": [after["ending_hook"]],
        }
        confirm_data = {
            "custom_plot_threads": "\n".join(after["plot_threads"]),
            "purpose": after["purpose"],
            "start_state": after["start_state"],
            "end_state": after["end_state"],
            "central_conflict": after["central_conflict"],
            "emotional_value": after["emotional_value"],
            "must_happen": "\n".join(after["must_happen"]),
            "must_preserve": "\n".join(after["must_preserve"]),
            "forbidden": "\n".join(after["forbidden"]),
            "foreshadow_setup": "\n".join(
                after["foreshadow_setup"]
            ),
            "foreshadow_payoff": "\n".join(
                after["foreshadow_payoff"]
            ),
            "ending_hook": after["ending_hook"],
            "target_chars": str(after["target_chars"]),
            "action": "confirm",
            "csrf": csrf_from(task_page.text),
        }
        for position, scene in enumerate(after["scenes"], start=1):
            for field in (
                "pov_character",
                "goal",
                "obstacle",
                "action",
                "reveal",
                "conceal",
                "subtext",
                "location",
                "end_state",
                "transition",
            ):
                confirm_data[f"scene_{field}_{position}"] = scene[field]
            confirm_data[f"scene_key_items_{position}"] = "\n".join(
                scene["key_items"]
            )
            confirm_data[f"scene_requirement_{position}"] = [
                (
                    f"{item['kind']}:"
                    f"{requirement_sources[item['kind']].index(item['text'])}"
                )
                for item in scene["requirement_refs"]
            ]
        response = client.post(
            task_url,
            data=confirm_data,
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert "confirmed=true" in response.headers["location"]
        confirmed = planning.get_task_card(
            user_id=int(user["id"]),
            project_id=project_id,
            chapter_id=chapter_id,
        )
        assert confirmed["status"] == "confirmed"
        context = database.get_writing_context(
            int(user["id"]), chapter_id
        )
        assert context["task_card"]["scenes"][1][
            "requirement_refs"
        ]


def test_analysis_technique_card_returns_to_planner_context(tmp_path):
    application = create_app(make_settings(tmp_path))
    with TestClient(application) as client:
        page = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "技法回流作者",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": csrf_from(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        page = client.get("/upload")
        response = client.post(
            "/upload",
            data={"title": "参考小说", "csrf": csrf_from(page.text)},
            files={
                "source_file": (
                    "reference.txt",
                    (
                        "第一章 线索\n"
                        "来信先让调查者改变行程，下一场景才解释邮戳来源。"
                    ).encode(),
                    "text/plain",
                )
            },
            follow_redirects=False,
        )
        document_url = response.headers["location"]
        document_page = client.get(document_url)
        response = client.post(
            f"{document_url}/analyze",
            data={"csrf": csrf_from(document_page.text)},
            follow_redirects=False,
        )
        job_url = response.headers["location"]
        job_id = job_url.rsplit("/", 1)[-1]
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if client.get(f"/api/jobs/{job_id}").json().get("terminal"):
                break
            time.sleep(0.03)
        job_page = client.get(job_url)
        analysis_match = re.search(r'href="/analyses/([^"]+)"', job_page.text)
        assert analysis_match
        analysis_url = f"/analyses/{analysis_match.group(1)}"
        analysis_page = client.get(analysis_url)
        assert "可迁移技法" in analysis_page.text
        assert "保存为技法卡" in analysis_page.text

        response = client.post(
            f"{analysis_url}/techniques/0",
            data={"csrf": csrf_from(analysis_page.text)},
            follow_redirects=False,
        )
        assert response.status_code == 303
        technique_url = response.headers["location"]
        technique_page = client.get(technique_url)
        assert "方法与原创性边界" in technique_page.text
        assert "让信息先影响行动、再补充解释" in technique_page.text

        new_page = client.get("/novels/new")
        response = client.post(
            "/novels/new",
            data={
                "title": "技法回流测试",
                "genre": "悬疑",
                "premise": "调查者追查异常邮戳。",
                "story_promise": "每章提供一项可核对的新证据。",
                "target_audience": "悬疑读者",
                "core_appeal": "证据推理",
                "ending_constraint": "解释邮戳来源",
                "world_setting": "当代小城",
                "style_guide": "克制具体",
                "point_of_view": "第三人称限知",
                "target_chapter_chars": "3000",
                "planning_horizon": "20",
                "csrf": csrf_from(new_page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        project_url = response.headers["location"]
        project_id = project_url.rsplit("/", 1)[-1]

        technique_page = client.get(technique_url)
        response = client.post(
            technique_url.split("?", 1)[0] + "/bindings",
            data={
                "target": f"project:{project_id}",
                "usage_modes": ["plan", "write", "audit"],
                "author_adaptation": "在本书中改成邮戳先改变调查行程。",
                "priority": "80",
                "csrf": csrf_from(technique_page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        workspace = client.get(project_url)
        assert "本书启用的写作技法" in workspace.text
        assert "让信息先影响行动、再补充解释" in workspace.text

        response = client.post(
            f"{project_url}/chapters",
            data={
                "title": "第一章 来信",
                "outline": "调查者收到异常来信。",
                "key_points": "核对邮戳",
                "volume_id": "",
                "csrf": csrf_from(workspace.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        chapter_url = response.headers["location"]
        chapter_page = client.get(chapter_url)
        assert "本章实际使用的技法" in chapter_page.text
        assert "先让一项信息改变人物的目标或代价" in chapter_page.text

        task_page = client.get(f"{chapter_url}/task-card")
        assert "本次将使用 1 张技法卡" in task_page.text
        response = client.post(
            f"{chapter_url}/task-card/generate",
            data={"instruction": "", "csrf": csrf_from(task_page.text)},
            follow_redirects=False,
        )
        assert response.status_code == 303
        planning_job_id = response.headers["location"].rsplit("/", 1)[-1]
        user = application.state.database.get_user_by_username("技法回流作者")
        deadline = time.monotonic() + 3
        job = None
        while time.monotonic() < deadline:
            job = application.state.database.get_generation_job(
                int(user["id"]), planning_job_id
            )
            if job and job["status"] in {"completed", "failed"}:
                break
            time.sleep(0.03)
        assert job and job["status"] == "completed"
        snapshot = json.loads(str(job["context_snapshot_json"]))
        assert snapshot["active_techniques"]["included_count"] == 1
        assert (
            snapshot["active_techniques"]["items"][0]["name"]
            == "让信息先影响行动、再补充解释"
        )


def test_registration_can_be_closed_even_for_empty_database(tmp_path):
    settings = replace(make_settings(tmp_path), allow_registration=False)
    with TestClient(create_app(settings)) as client:
        response = client.get("/register", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/login"
        login = client.get("/login")
        assert "创建一个" not in login.text


def test_production_rejects_default_session_secret(tmp_path):
    settings = replace(
        make_settings(tmp_path),
        app_env="production",
        secret_key="dev-only-change-this-secret-before-deploy",
        cookie_secure=True,
    )
    with pytest.raises(ValueError, match="APP_SECRET_KEY"):
        create_app(settings)


def test_user_can_save_update_and_delete_personal_api_key(tmp_path):
    application = create_app(make_settings(tmp_path))
    with TestClient(application) as client:
        page = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "个人API用户",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": csrf_from(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        page = client.get("/settings/api")
        assert page.status_code == 200
        raw_key = "sk-personal-secret-5678"
        response = client.post(
            "/settings/api",
            data={
                "api_key": raw_key,
                "model": "deepseek-v4-flash",
                "thinking": "enabled",
                "reasoning_effort": "max",
                "csrf": csrf_from(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert response.headers["location"] == "/settings/api?saved=true"

        saved_page = client.get(response.headers["location"])
        assert "个人 Key 已配置" in saved_page.text
        assert "sk-••••5678" in saved_page.text
        assert raw_key not in saved_page.text

        user = application.state.database.get_user_by_username("个人API用户")
        credential = application.state.database.get_api_credential(user["id"])
        assert raw_key not in credential["encrypted_key"]
        assert (
            application.state.credential_cipher.decrypt(
                credential["encrypted_key"]
            )
            == raw_key
        )

        response = client.post(
            "/settings/api/delete",
            data={"csrf": csrf_from(saved_page.text)},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert application.state.database.get_api_credential(user["id"]) is None


def test_unified_workbench_creates_edits_and_saves_book_prompt(tmp_path):
    application = create_app(make_settings(tmp_path))
    with TestClient(application) as client:
        page = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "工作台作者",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": csrf_from(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        page = client.get("/novels/new")
        response = client.post(
            "/novels/new",
            data={
                "title": "潮痕",
                "genre": "悬疑",
                "premise": "一名记者从旧录音里听见了尚未发生的海难警报。",
                "point_of_view": "第三人称限知",
                "target_chapter_chars": "3000",
                "return_to_workbench": "1",
                "csrf": csrf_from(page.text),
            },
            follow_redirects=False,
        )
        workbench_url = response.headers["location"]
        assert workbench_url.endswith("/workbench")
        project_id = workbench_url.split("/novels/", 1)[1].split("/", 1)[0]

        workbench = client.get(workbench_url)
        assert workbench.status_code == 200
        assert 'data-workbench' in workbench.text
        assert "默认单章篇幅" in workbench.text
        assert "不必先想好书名" in workbench.text
        for settings_tab in (
            "作品核心",
            "世界规则",
            "人物关系",
            "故事结构",
            "文风约束",
            "创作参数",
        ):
            assert settings_tab in workbench.text
        assert 'data-settings-panel="core"' in workbench.text
        assert 'data-settings-panel="parameters"' in workbench.text
        assert "高级规划" not in workbench.text
        assert "模型识别任务 · 服务端裁决权限" not in workbench.text
        assert "data-agent-note" not in workbench.text
        assert 'class="studio-chat-input"' in workbench.text
        assert "studio-agent-switcher" not in workbench.text
        assert 'name="agent_role"' not in workbench.text
        assert (
            'name="title" maxlength="120" required'
            not in workbench.text
        )

        response = client.post(
            f"/novels/{project_id}/chapters",
            data={
                "return_to_workbench": "1",
                "csrf": csrf_from(workbench.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert "/workbench?chapter_id=" in response.headers["location"]
        chapter_id = response.headers["location"].split("chapter_id=", 1)[1]

        chapter_page = client.get(response.headers["location"])
        assert "第1章 · 未命名章节" in chapter_page.text
        chapter = application.state.database.get_novel_chapter(
            application.state.database.get_user_by_username("工作台作者")[
                "id"
            ],
            project_id,
            chapter_id,
        )
        assert chapter["title"] == ""
        response = client.post(
            f"/novels/{project_id}/chapters/{chapter_id}/save",
            data={
                "content": "录音带转到第三圈时，海面仍旧平静。",
                "change_summary": "工作台自动保存",
                "return_to_workbench": "1",
                "csrf": csrf_from(chapter_page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        saved_page = client.get(response.headers["location"])
        assert "录音带转到第三圈时" in saved_page.text
        assert "/ 约 3000 字" in saved_page.text

        project = application.state.database.get_novel_project(
            application.state.database.get_user_by_username("工作台作者")[
                "id"
            ],
            project_id,
        )
        settings_page = client.get(
            f"/novels/{project_id}/workbench?view=settings"
        )
        response = client.post(
            f"/novels/{project_id}/settings",
            data={
                "title": project["title"],
                "genre": project["genre"],
                "premise": project["premise"],
                "world_setting": project["world_setting"],
                "style_guide": project["style_guide"],
                "point_of_view": project["point_of_view"],
                "target_chapter_chars": project["target_chapter_chars"],
                "planning_horizon": project["planning_horizon"],
                "ai_instructions": "本书让人物先行动，再解释。",
                "settings_tab": "parameters",
                "return_to_workbench": "1",
                "csrf": csrf_from(settings_page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert (
            "view=settings&settings_tab=parameters&saved=true"
            in response.headers["location"]
        )
        project = application.state.database.get_novel_project(
            application.state.database.get_user_by_username("工作台作者")[
                "id"
            ],
            project_id,
        )
        assert project["ai_instructions"] == "本书让人物先行动，再解释。"


def test_zero_input_project_enters_settings_and_applies_ai_candidate(
    tmp_path,
):
    application = create_app(make_settings(tmp_path))
    with TestClient(application) as client:
        page = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "空白项目作者",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": csrf_from(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        dashboard = client.get("/dashboard")
        response = client.post(
            "/novels/new/blank",
            data={"csrf": csrf_from(dashboard.text)},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert "view=settings" in response.headers["location"]
        workbench_url = response.headers["location"]
        project_id = workbench_url.split("/novels/", 1)[1].split("/", 1)[0]

        workbench = client.get(workbench_url)
        assert "未命名作品" in workbench.text
        assert 'name="agent_role"' not in workbench.text
        assert "studio-agent-switcher" not in workbench.text
        assert 'aria-label="新建对话"' in workbench.text
        assert (
            f'action="/novels/{project_id}/assistant/new"'
            in workbench.text
        )
        assert "默认单章篇幅" in workbench.text

        question = "我想写一个记者在雨夜收到七年前来信的悬疑故事。"
        response = client.post(
            f"/novels/{project_id}/assistant/messages",
            data={
                "csrf": csrf_from(workbench.text),
                "question": question,
                "conversation_id": "",
                "return_view": "settings",
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        conversation_url = response.headers["location"]
        deadline = time.monotonic() + 3
        rendered = ""
        while time.monotonic() < deadline:
            rendered = client.get(conversation_url).text
            if "应用到设定" in rendered:
                break
            time.sleep(0.03)
        assert "候选设定" in rendered
        assert "尚未写入" in rendered
        assert re.search(r"<small>\s*AI\s*</small>", rendered)
        action = re.search(
            r'action="/assistant/messages/([a-f0-9]+)/apply-settings"',
            rendered,
        )
        assert action

        response = client.post(
            f"/assistant/messages/{action.group(1)}/apply-settings",
            data={
                "csrf": csrf_from(rendered),
                "return_to_workbench": "1",
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        applied_page = client.get(response.headers["location"])
        assert "已应用" in applied_page.text
        user = application.state.database.get_user_by_username(
            "空白项目作者"
        )
        project = application.state.database.get_novel_project(
            int(user["id"]), project_id
        )
        assert project["premise"] == question
        assert project["genre"] == "悬疑"


def test_dashboard_can_delete_owned_novel_and_files(tmp_path):
    application = create_app(make_settings(tmp_path))
    with TestClient(application) as client:
        page = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "删除作品作者",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": csrf_from(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        dashboard = client.get("/dashboard")
        response = client.post(
            "/novels/new/blank",
            data={"csrf": csrf_from(dashboard.text)},
            follow_redirects=False,
        )
        assert response.status_code == 303
        project_id = response.headers["location"].split("/novels/", 1)[1].split(
            "/", 1
        )[0]
        database = application.state.database
        user_id = int(
            database.get_user_by_username("删除作品作者")["id"]
        )
        project_dir = tmp_path / "novels" / str(user_id) / project_id
        assert project_dir.is_dir()
        conversation_id = (
            application.state.assistant_chat_service.create_conversation(
                user_id=user_id,
                scope_type="project",
                title="删除作品前的讨论",
                project_id=project_id,
            )
        )
        assert application.state.assistant_chat_service.get_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
        )

        dashboard = client.get("/dashboard")
        assert 'data-delete-project-form' in dashboard.text
        assert f'action="/novels/{project_id}/delete"' in dashboard.text
        response = client.post(
            f"/novels/{project_id}/delete",
            data={"csrf": csrf_from(dashboard.text)},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert response.headers["location"] == "/dashboard?deleted=true"
        deleted_dashboard = client.get(response.headers["location"])
        assert deleted_dashboard.status_code == 200
        assert "作品已删除" in deleted_dashboard.text
        assert database.get_novel_project(user_id, project_id) is None
        assert (
            application.state.assistant_chat_service.get_conversation(
                user_id=user_id,
                conversation_id=conversation_id,
            )
            is None
        )
        assert not project_dir.exists()


def test_full_mock_novel_writing_workflow(tmp_path):
    application = create_app(make_settings(tmp_path))
    with TestClient(application) as client:
        page = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "小说作者",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": csrf_from(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        page = client.get("/novels/new")
        response = client.post(
            "/novels/new",
            data={
                "title": "雾港来信",
                "genre": "悬疑",
                "premise": "记者收到失踪父亲寄出的新信件，并返回雾港追查真相。",
                "world_setting": "当代海港小城。",
                "style_guide": "克制冷峻。",
                "point_of_view": "第三人称限知",
                "target_chapter_chars": "3000",
                "csrf": csrf_from(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        project_url = response.headers["location"]

        page = client.get(project_url)
        response = client.post(
            f"{project_url}/voice",
            data={
                "narration_rules": "第三人称紧贴林岚，只写她能观察或推断的内容。",
                "sentence_rhythm": "调查段落用短句推进，转折前允许一个长句。",
                "dialogue_voice": "林岚少解释，习惯用追问回避情绪。",
                "sensory_palette": "盐雾、旧金属、潮湿纸张。",
                "metaphor_policy": "低密度，只用人物经验范围内的意象。",
                "allowed_omissions": "不直接解释恐惧，让动作和停顿承担。",
                "preferred_patterns": "动作先于情绪判断",
                "banned_expressions": "不禁\n内心深处",
                "author_notes": "",
                "action": "confirm",
                "csrf": csrf_from(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert "#voice" in response.headers["location"]

        page = client.get(project_url)
        response = client.post(
            f"{project_url}/characters",
            data={
                "name": "林岚",
                "role": "调查记者",
                "traits": "冷静、执拗",
                "background": "父亲十年前失踪",
                "character_arc": "直面家庭秘密",
                "csrf": csrf_from(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        page = client.get(project_url)
        response = client.post(
            f"{project_url}/chapters",
            data={
                "title": "第一章 迟到的信",
                "outline": "林岚收到父亲署名的信，决定返回雾港。",
                "key_points": "邮戳是三天前",
                "csrf": csrf_from(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        chapter_url = response.headers["location"]

        task_page = client.get(f"{chapter_url}/task-card")
        assert "章节职责与边界" in task_page.text
        response = client.post(
            f"{chapter_url}/task-card/generate",
            data={
                "instruction": "保持现实悬疑，不提前揭晓寄信人。",
                "csrf": csrf_from(task_page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        planning_job_id = response.headers["location"].rsplit("/", 1)[-1]
        deadline = time.monotonic() + 3
        planning_payload = {}
        while time.monotonic() < deadline:
            planning_payload = client.get(
                f"/api/writing-jobs/{planning_job_id}"
            ).json()
            if planning_payload.get("terminal"):
                break
            time.sleep(0.03)
        assert planning_payload["status"] == "completed"
        assert planning_payload["redirect_url"] == f"{chapter_url}/task-card"
        task_page = client.get(planning_payload["redirect_url"])
        assert "Planner 提出任务卡草稿" in task_page.text
        assert "人物做出不可轻易撤回的选择" in task_page.text
        assert f'action="{chapter_url}/task-card"' in task_page.text

        response = client.post(
            f"{chapter_url}/task-card",
            data={
                "purpose": "林岚确认来信不可能出现，并决定返回雾港。",
                "start_state": "林岚仍在外地，认为父亲早已失踪。",
                "end_state": "林岚买下回雾港的车票。",
                "central_conflict": "理性判断与父亲笔迹证据相互冲突。",
                "emotional_value": "压抑多年的希望重新出现。",
                "must_happen": "核对邮戳\n确认笔迹\n购买车票",
                "must_preserve": "父亲已经失踪十年",
                "forbidden": "本章揭晓寄信人",
                "ending_hook": "信纸散发出雾港海水的气味。",
                "target_chars": "3000",
                "scene_pov_character_1": "林岚",
                "scene_location_1": "林岚住处",
                "scene_goal_1": "确认来信真假",
                "scene_obstacle_1": "邮戳与常识矛盾",
                "scene_action_1": "对照旧信笔迹并检查邮戳",
                "scene_end_state_1": "确认笔迹属于父亲",
                "scene_transition_1": "她开始查询返程车票",
                "scene_requirement_1": [
                    "must_happen:0",
                    "must_happen:1",
                ],
                "scene_pov_character_2": "林岚",
                "scene_location_2": "车站",
                "scene_goal_2": "决定是否返回雾港",
                "scene_obstacle_2": "她抗拒回到故乡",
                "scene_action_2": "买下当夜车票并再次检查信纸",
                "scene_end_state_2": "她踏上返程列车",
                "scene_transition_2": "发现信纸带有海水气味",
                "scene_requirement_2": [
                    "must_happen:2",
                    "ending_hook:0",
                ],
                "action": "confirm",
                "csrf": csrf_from(task_page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert "confirmed=true" in response.headers["location"]

        page = client.get(chapter_url)
        assert "AI 创作助手" in page.text
        assert "任务卡已确认" in page.text
        response = client.post(
            f"{chapter_url}/generate",
            data={
                "operation": "draft",
                "instruction": "从拆信开始",
                "csrf": csrf_from(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        job_url = response.headers["location"]
        job_id = job_url.rsplit("/", 1)[-1]

        deadline = time.monotonic() + 3
        payload = {}
        while time.monotonic() < deadline:
            payload = client.get(f"/api/writing-jobs/{job_id}").json()
            if payload.get("terminal"):
                break
            time.sleep(0.03)
        assert payload["status"] == "completed"
        user = application.state.database.get_user_by_username("小说作者")
        stored_job = application.state.database.get_generation_job(
            int(user["id"]), job_id
        )
        context_snapshot = json.loads(stored_job["context_snapshot_json"])
        assert context_snapshot["schema_version"] == 1
        assert context_snapshot["chapter"]["title"] == "第一章 迟到的信"
        assert context_snapshot["canonical_memory"]["source"] == (
            "author_confirmed_canon_only"
        )
        assert context_snapshot["canonical_memory"]["retrieval"][
            "engine"
        ].startswith("sqlite_fts5")
        assert context_snapshot["canonical_memory"]["retrieval"][
            "query_terms"
        ]
        generation_result = json.loads(stored_job["result_json"])
        assert generation_result["quality"]["verdict"] == "pass"
        assert generation_result["quality"]["effective_char_count"] >= 2000
        assert generation_result["quality"]["expansion_attempted"] is True
        job_page = client.get(job_url)
        assert "本次相关故事记忆" in job_page.text
        assert "仅检索本作品、当前正史分支" in job_page.text

        chapter_page = client.get(chapter_url)
        assert "本地演示草稿" in chapter_page.text
        assert "最近版本" in chapter_page.text
        assert "候选稿" in chapter_page.text
        assert "硬审计通过" in chapter_page.text
        version_match = re.search(
            r'action="([^"]+/versions/([a-f0-9]+)/accept)"',
            chapter_page.text,
        )
        assert version_match
        generated_version_id = version_match.group(2)
        generated_version = application.state.database.get_chapter_version(
            int(user["id"]),
            project_url.rsplit("/", 1)[-1],
            chapter_url.rsplit("/", 1)[-1],
            generated_version_id,
        )
        generated_path = Path(generated_version["content_path"])
        generated_crlf = generated_path.read_text(encoding="utf-8").replace(
            "\n", "\r\n"
        )
        generated_path.write_bytes(generated_crlf.encode("utf-8"))
        with application.state.database.connection() as connection:
            connection.execute(
                """
                UPDATE novel_chapter_versions
                SET content_hash=?
                WHERE id=?
                """,
                (
                    hashlib.sha256(
                        generated_crlf.encode("utf-8")
                    ).hexdigest(),
                    generated_version_id,
                ),
            )
            connection.commit()
        quality_url = (
            f"{chapter_url}/versions/{generated_version_id}/quality"
        )
        quality_page = client.get(quality_url)
        assert quality_page.status_code == 200
        assert "硬审计通过" in quality_page.text
        assert "已使用一次" in quality_page.text
        response = client.post(
            quality_url,
            data={"csrf": csrf_from(quality_page.text)},
            follow_redirects=False,
        )
        assert response.status_code == 303
        audit_job_id = response.headers["location"].rsplit("/", 1)[-1]
        deadline = time.monotonic() + 3
        audit_payload = {}
        while time.monotonic() < deadline:
            audit_payload = client.get(
                f"/api/writing-jobs/{audit_job_id}"
            ).json()
            if audit_payload.get("terminal"):
                break
            time.sleep(0.03)
        assert audit_payload["status"] == "completed"
        assert audit_payload["redirect_url"] == quality_url
        rerun_quality_page = client.get(quality_url)
        assert "硬审计通过" in rerun_quality_page.text

        style_url = (
            f"{chapter_url}/versions/{generated_version_id}/style"
        )
        style_page = client.get(style_url)
        assert "先定位，再决定是否修改" in style_page.text
        response = client.post(
            style_url,
            data={"csrf": csrf_from(style_page.text)},
            follow_redirects=False,
        )
        assert response.status_code == 303
        style_job_id = response.headers["location"].rsplit("/", 1)[-1]
        deadline = time.monotonic() + 3
        style_payload = {}
        while time.monotonic() < deadline:
            style_payload = client.get(
                f"/api/writing-jobs/{style_job_id}"
            ).json()
            if style_payload.get("terminal"):
                break
            time.sleep(0.03)
        assert style_payload["status"] == "completed"
        assert style_payload["redirect_url"] == style_url

        style_page = client.get(style_url)
        assert "具体问题" in style_page.text
        assert "工具说明" in style_page.text
        issue_match = re.search(r'href="/style-issues/([a-f0-9]+)"', style_page.text)
        assert issue_match
        issue_url = f"/style-issues/{issue_match.group(1)}"
        issue_page = client.get(issue_url)
        response = client.post(
            f"{issue_url}/rewrite",
            data={
                "instruction": "保留核对信件的动作，不新增事实。",
                "csrf": csrf_from(issue_page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        rewrite_job_id = response.headers["location"].rsplit("/", 1)[-1]
        deadline = time.monotonic() + 3
        rewrite_payload = {}
        while time.monotonic() < deadline:
            rewrite_payload = client.get(
                f"/api/writing-jobs/{rewrite_job_id}"
            ).json()
            if rewrite_payload.get("terminal"):
                break
            time.sleep(0.03)
        assert rewrite_payload["status"] == "completed"
        assert rewrite_payload["redirect_url"] == issue_url

        issue_page = client.get(issue_url)
        assert "候选 1" in issue_page.text
        assert "原文" in issue_page.text
        assert "改写后" in issue_page.text
        candidate_match = re.search(
            r'action="/style-rewrite-candidates/([a-f0-9]+)/accept"',
            issue_page.text,
        )
        assert candidate_match
        response = client.post(
            f"/style-rewrite-candidates/{candidate_match.group(1)}/accept",
            data={"csrf": csrf_from(issue_page.text)},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert response.headers["location"].startswith(chapter_url)

        revised_page = client.get(response.headers["location"])
        assert "定点改写候选" in revised_page.text
        assert "待硬审计" in revised_page.text
        chapter_record = application.state.database.get_novel_chapter(
            int(user["id"]),
            project_url.rsplit("/", 1)[-1],
            chapter_url.rsplit("/", 1)[-1],
        )
        revised_version_id = chapter_record["working_version_id"]
        revised_version = application.state.database.get_chapter_version(
            int(user["id"]),
            project_url.rsplit("/", 1)[-1],
            chapter_url.rsplit("/", 1)[-1],
            revised_version_id,
        )
        revised_path = Path(revised_version["content_path"])
        revised_crlf = revised_path.read_text(encoding="utf-8").replace(
            "\n", "\r\n"
        )
        revised_path.write_bytes(revised_crlf.encode("utf-8"))
        with application.state.database.connection() as connection:
            connection.execute(
                """
                UPDATE novel_chapter_versions
                SET content_hash=?
                WHERE id=?
                """,
                (
                    hashlib.sha256(
                        revised_crlf.encode("utf-8")
                    ).hexdigest(),
                    revised_version_id,
                ),
            )
            connection.commit()
        revised_quality_url = (
            f"{chapter_url}/versions/{revised_version_id}/quality"
        )
        revised_quality_page = client.get(revised_quality_url)
        response = client.post(
            revised_quality_url,
            data={"csrf": csrf_from(revised_quality_page.text)},
            follow_redirects=False,
        )
        assert response.status_code == 303
        revised_audit_job_id = response.headers["location"].rsplit("/", 1)[-1]
        deadline = time.monotonic() + 3
        revised_audit_payload = {}
        while time.monotonic() < deadline:
            revised_audit_payload = client.get(
                f"/api/writing-jobs/{revised_audit_job_id}"
            ).json()
            if revised_audit_payload.get("terminal"):
                break
            time.sleep(0.03)
        assert revised_audit_payload["status"] == "completed"
        revised_page = client.get(chapter_url)
        assert "硬审计通过" in revised_page.text
        accept_revised_url = (
            f"{chapter_url}/versions/{revised_version_id}/accept"
        )
        response = client.post(
            accept_revised_url,
            data={"csrf": csrf_from(revised_page.text)},
            follow_redirects=False,
        )
        assert response.status_code == 303
        memory_job_url = response.headers["location"]
        assert memory_job_url.startswith("/writing-jobs/")
        memory_job_id = memory_job_url.rsplit("/", 1)[-1]

        deadline = time.monotonic() + 3
        memory_payload = {}
        while time.monotonic() < deadline:
            memory_payload = client.get(
                f"/api/writing-jobs/{memory_job_id}"
            ).json()
            if memory_payload.get("terminal"):
                break
            time.sleep(0.03)
        assert memory_payload["status"] == "completed"
        assert memory_payload["redirect_url"].startswith("/story-deltas/")

        delta_page = client.get(memory_payload["redirect_url"])
        assert delta_page.status_code == 200
        assert "审核本章造成的变化" in delta_page.text
        assert "不是正史" in delta_page.text
        response = client.post(
            f"{memory_payload['redirect_url']}/accept",
            data={"csrf": csrf_from(delta_page.text)},
            follow_redirects=False,
        )
        assert response.status_code == 303
        canonical_page = client.get(response.headers["location"])
        assert "正史故事记忆已更新" in canonical_page.text

        project_id = project_url.rsplit("/", 1)[-1]
        export = client.get(f"/novels/{project_id}/export.txt")
        assert export.status_code == 200
        assert "第一章 迟到的信" in export.text


def test_reader_branches_and_old_canon_impact_web_workflow(tmp_path):
    application = create_app(make_settings(tmp_path))
    with TestClient(application) as client:
        page = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "剧情决策作者",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": csrf_from(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        page = client.get("/novels/new")
        response = client.post(
            "/novels/new",
            data={
                "title": "潮痕",
                "genre": "现实悬疑",
                "premise": "记者追查一封不可能出现的来信和旧码头事故。",
                "story_promise": "每三章给出一项可核对的新证据。",
                "target_audience": "喜欢慢热推理的读者",
                "core_appeal": "证据推理与关系试探",
                "ending_constraint": "必须解释来信来源。",
                "world_setting": "当代海港小城。",
                "style_guide": "克制具体。",
                "point_of_view": "第三人称限知",
                "target_chapter_chars": "3000",
                "planning_horizon": "20",
                "csrf": csrf_from(page.text),
            },
            follow_redirects=False,
        )
        project_url = response.headers["location"]
        project_id = project_url.rsplit("/", 1)[-1]

        page = client.get(project_url)
        response = client.post(
            f"{project_url}/chapters",
            data={
                "title": "第一章 来信",
                "outline": "林岚收到来信。",
                "key_points": "核对邮戳",
                "csrf": csrf_from(page.text),
            },
            follow_redirects=False,
        )
        first_url = response.headers["location"]
        first_id = first_url.rsplit("/", 1)[-1]
        page = client.get(project_url)
        response = client.post(
            f"{project_url}/chapters",
            data={
                "title": "第二章 旧档案",
                "outline": "林岚查询旧档案。",
                "key_points": "发现缺页",
                "csrf": csrf_from(page.text),
            },
            follow_redirects=False,
        )
        second_id = response.headers["location"].rsplit("/", 1)[-1]

        user = application.state.database.get_user_by_username(
            "剧情决策作者"
        )
        user_id = int(user["id"])

        def candidate(chapter_id: str, marker: str) -> str:
            chapter = application.state.database.get_novel_chapter(
                user_id, project_id, chapter_id
            )
            content = marker * 2100
            path = (
                Path(chapter["content_path"]).parent
                / f"{marker}-candidate.txt"
            )
            path.write_text(content, encoding="utf-8")
            version_id = (
                application.state.database.record_manual_chapter_version(
                    user_id=user_id,
                    project_id=project_id,
                    chapter_id=chapter_id,
                    version_path=path,
                    char_count=len(content),
                    effective_char_count=len(content),
                    content_hash=hashlib.sha256(
                        content.encode("utf-8")
                    ).hexdigest(),
                )
            )
            assert version_id
            return version_id

        first_version = candidate(first_id, "甲")
        application.state.database.accept_chapter_version(
            user_id=user_id,
            project_id=project_id,
            chapter_id=first_id,
            version_id=first_version,
            override_reason="测试环境作者确认第一版正史",
        )

        page = client.get(project_url)
        assert "读者意见与剧情调整" in page.text
        response = client.post(
            f"{project_url}/reader-requests",
            data={
                "raw_text": "希望女主不要这么快相信搭档。",
                "request_type": "relationship",
                "impact_scope": "next_three",
                "priority": "soft",
                "constraints": "不得改写来信已经存在的正史",
                "author_note": "可以增加试探，但不能拖慢调查。",
                "csrf": csrf_from(page.text),
            },
            follow_redirects=False,
        )
        request_url = response.headers["location"]
        request_id = request_url.rsplit("/", 1)[-1]
        request_page = client.get(request_url)
        assert "读者原话只交给 Planner 评估" in request_page.text
        response = client.post(
            f"{request_url}/propose",
            data={"csrf": csrf_from(request_page.text)},
            follow_redirects=False,
        )
        job_id = response.headers["location"].rsplit("/", 1)[-1]
        deadline = time.monotonic() + 3
        payload = {}
        while time.monotonic() < deadline:
            payload = client.get(f"/api/writing-jobs/{job_id}").json()
            if payload.get("terminal"):
                break
            time.sleep(0.03)
        assert payload["status"] == "completed"
        assert payload["redirect_url"] == request_url

        request_page = client.get(request_url)
        assert "三个走向与代价" in request_page.text
        assert "顺势强化" in request_page.text
        assert "延迟兑现" in request_page.text
        assert "涉及已发表正史，禁止直接采纳" in request_page.text
        proposal = re.search(
            r'action="/reader-proposals/([a-f0-9]+)/accept"',
            request_page.text,
        )
        assert proposal
        response = client.post(
            f"/reader-proposals/{proposal.group(1)}/accept",
            data={
                "request_id": request_id,
                "csrf": csrf_from(request_page.text),
            },
            follow_redirects=False,
        )
        assert response.headers["location"] == (
            f"{request_url}?adopted=true"
        )
        adopted_page = client.get(response.headers["location"])
        assert "未来滚动细纲已经更新" in adopted_page.text
        chapters = application.state.database.list_novel_chapters(
            user_id, project_id
        )
        assert [item["position"] for item in chapters] == [1, 2, 3, 4]

        second_version = candidate(second_id, "乙")
        application.state.database.accept_chapter_version(
            user_id=user_id,
            project_id=project_id,
            chapter_id=second_id,
            version_id=second_version,
            override_reason="测试环境作者确认第二版正史",
        )
        replacement_id = candidate(first_id, "新")
        first_page = client.get(first_url)
        response = client.post(
            f"{first_url}/versions/{replacement_id}/accept",
            data={
                "override_reason": "作者确认改动不会破坏核心设定",
                "csrf": csrf_from(first_page.text),
            },
            follow_redirects=False,
        )
        impact_url = response.headers["location"]
        assert impact_url.startswith("/canon-impact-reports/")
        assert (
            application.state.database.get_novel_chapter(
                user_id, project_id, first_id
            )["canonical_version_id"]
            == first_version
        )
        impact_page = client.get(impact_url)
        assert "尚未切换正史" in impact_page.text
        assert "第二章" in impact_page.text
        response = client.post(
            impact_url,
            data={
                "action": "confirm",
                "csrf": csrf_from(impact_page.text),
            },
            follow_redirects=False,
        )
        assert response.headers["location"].startswith("/writing-jobs/")
        assert (
            application.state.database.get_novel_chapter(
                user_id, project_id, first_id
            )["canonical_version_id"]
            == replacement_id
        )
        assert application.state.database.get_novel_chapter(
            user_id, project_id, second_id
        )["needs_recheck"] == 1
        report_id = impact_url.rsplit("/", 1)[-1]
        with application.state.database.connection() as connection:
            report = connection.execute(
                "SELECT status FROM canon_impact_reports WHERE id=?",
                (report_id,),
            ).fetchone()
        assert report["status"] == "applied"


def test_chapter_version_comparison_renders_full_diff(tmp_path):
    application = create_app(make_settings(tmp_path))
    with TestClient(application) as client:
        register_page = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "版本比较作者",
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
                "title": "雾港改稿",
                "genre": "悬疑",
                "premise": "林岚核对一封日期异常的来信。",
                "csrf": csrf_from(novel_page.text),
            },
            follow_redirects=False,
        )
        project_url = response.headers["location"]
        project_id = project_url.rsplit("/", 1)[-1]

        workspace = client.get(project_url)
        response = client.post(
            f"{project_url}/chapters",
            data={
                "title": "第一章 来信",
                "outline": "林岚核对邮戳。",
                "key_points": "邮戳日期异常",
                "csrf": csrf_from(workspace.text),
            },
            follow_redirects=False,
        )
        chapter_url = response.headers["location"]
        chapter_id = chapter_url.rsplit("/", 1)[-1]

        first_page = client.get(chapter_url)
        response = client.post(
            f"{chapter_url}/save",
            data={
                "content": "林岚看着邮戳。她解释自己很害怕。\n门外很安静。",
                "change_summary": "建立第一版。",
                "csrf": csrf_from(first_page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        second_page = client.get(chapter_url)
        response = client.post(
            f"{chapter_url}/save",
            data={
                "content": (
                    "林岚看着邮戳。她把杯子推远，手指仍压着杯沿。\n"
                    "门外传来三下脚步声。"
                ),
                "change_summary": "删去情绪解释，让动作承担反应。",
                "csrf": csrf_from(second_page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        user = application.state.database.get_user_by_username(
            "版本比较作者"
        )
        versions = application.state.database.list_chapter_versions(
            int(user["id"]), project_id, chapter_id
        )
        assert len(versions) == 2
        target_id = str(versions[0]["id"])
        base_id = str(versions[1]["id"])
        compare_url = f"{chapter_url}/versions/{target_id}/compare"

        chapter_page = client.get(chapter_url)
        assert f'href="{compare_url}"' in chapter_page.text

        comparison = client.get(compare_url)
        assert comparison.status_code == 200
        assert "CHAPTER VERSION DIFF" in comparison.text
        assert "删去情绪解释，让动作承担反应。" in comparison.text
        assert "解释自己很害怕" in comparison.text
        assert "把杯子推远，手指仍压着杯沿" in comparison.text
        assert 'class="removed"' in comparison.text
        assert 'class="added"' in comparison.text

        explicit_comparison = client.get(
            compare_url, params={"base_id": base_id}
        )
        assert explicit_comparison.status_code == 200
        assert f'<option value="{base_id}" selected>' in (
            explicit_comparison.text
        )

        invalid_comparison = client.get(
            compare_url, params={"base_id": target_id}
        )
        assert invalid_comparison.status_code == 404
