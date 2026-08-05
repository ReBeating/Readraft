import hashlib
import html
import json
import re
import time
from dataclasses import replace
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.memory_service import MemoryService
from app.technique_service import TechniqueService


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
        model_api_key=None,
        model_base_url="https://api.deepseek.com",
        model_name="deepseek-v4-flash",
        model_thinking=False,
        model_reasoning_effort="high",
        model_max_tokens=5_000,
        model_connect_timeout_seconds=1,
        model_read_timeout_seconds=1,
        model_max_retries=0,
        worker_poll_seconds=0.01,
    )


def csrf_from(html: str) -> str:
    match = re.search(r'name="csrf" value="([^"]+)"', html)
    assert match
    return match.group(1)


def commit_text_import(client: TestClient, response):
    assert response.status_code == 303
    preview_url = response.headers["location"]
    assert preview_url.startswith("/import/previews/")
    preview = client.get(preview_url)
    assert preview.status_code == 200
    assert "检查分章" in preview.text
    return client.post(
        preview_url + "/commit",
        data={"csrf": csrf_from(preview.text)},
        follow_redirects=False,
    )


def project_id_from_workbench(path: str) -> str:
    assert path.startswith("/novels/")
    assert "/workbench" in path
    return path.split("/novels/", 1)[1].split("/", 1)[0]


def chapter_id_from_workbench(path: str) -> str:
    match = re.search(r"[?&]chapter_id=([^&]+)", path)
    assert match
    return match.group(1)


def test_development_without_shared_key_never_uses_test_models(tmp_path):
    application = create_app(
        replace(make_settings(tmp_path), app_env="development")
    )
    with TestClient(application) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json()["model_provider"] == "personal-key-only"
        assert application.state.analyzer is None

        register = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "真实模型作者",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": csrf_from(register.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        dashboard = client.get("/dashboard")
        assert "尚未配置模型服务" in dashboard.text
        assert "本地模拟回复" not in dashboard.text
        assert "本地演示模式" not in dashboard.text
        assert dashboard.text.count('href="/import"') == 1
        assert dashboard.text.count('action="/novels/new/blank"') == 1
        assert "studio-empty-actions" not in dashboard.text

        upload = client.get("/import")
        imported = client.post(
            "/import",
            data={
                "title": "无 Key 参考文本",
                "csrf": csrf_from(upload.text),
            },
            files={
                "work_file": (
                    "reference.txt",
                    "第一章\n这段内容不能由测试模型分析。".encode(),
                    "text/plain",
                )
            },
            follow_redirects=False,
        )
        imported = commit_text_import(client, imported)
        assert imported.status_code == 303
        document_url = imported.headers["location"]
        document = client.get(document_url)
        analyze = client.post(
            f"{document_url}/analyze",
            data={"csrf": csrf_from(document.text)},
            follow_redirects=False,
        )
        assert analyze.status_code == 303
        assert analyze.headers["location"].startswith("/settings/api?error=")
        with application.state.database.connection() as connection:
            assert (
                connection.execute(
                    "SELECT COUNT(*) FROM analysis_jobs"
                ).fetchone()[0]
                == 0
            )


def test_chapter_buffer_does_not_create_history_and_rejects_stale_head(tmp_path):
    application = create_app(make_settings(tmp_path))
    with TestClient(application) as client:
        register = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "暂存并发作者",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": csrf_from(register.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        create_page = client.get("/novels/new")
        response = client.post(
            "/novels/new",
            data={
                "title": "暂存测试",
                "premise": "验证编辑缓冲与 main HEAD 分离。",
                "csrf": csrf_from(create_page.text),
            },
            follow_redirects=False,
        )
        project_id = project_id_from_workbench(response.headers["location"])
        workbench = client.get(response.headers["location"])
        response = client.post(
            f"/novels/{project_id}/chapters",
            data={
                "title": "第一章",
                "csrf": csrf_from(workbench.text),
            },
            follow_redirects=False,
        )
        chapter_id = chapter_id_from_workbench(response.headers["location"])
        chapter_url = f"/novels/{project_id}/chapters/{chapter_id}"
        chapter_page = client.get(response.headers["location"])
        csrf = csrf_from(chapter_page.text)

        buffered = client.post(
            f"{chapter_url}/buffer",
            data={
                "content": "这只是尚未正式保存的编辑。",
                "expected_head_version_id": "",
                "csrf": csrf,
            },
        )
        assert buffered.status_code == 200
        assert buffered.json()["buffered"] is True
        user_id = int(
            application.state.database.get_user_by_username("暂存并发作者")[
                "id"
            ]
        )
        assert application.state.database.count_chapter_versions(
            user_id,
            project_id,
            chapter_id,
        ) == 0
        recovered = client.get(response.headers["location"])
        assert "这只是尚未正式保存的编辑。" in recovered.text
        assert "已恢复上次未正式提交的暂存内容" in recovered.text

        saved = client.post(
            f"{chapter_url}/save",
            data={
                "content": "这是第一个正式版本。",
                "expected_head_version_id": "",
                "csrf": csrf_from(recovered.text),
            },
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert saved.status_code == 200
        first_head = saved.json()["version_id"]
        assert saved.json()["created_new_version"] is True
        assert application.state.database.count_chapter_versions(
            user_id,
            project_id,
            chapter_id,
        ) == 1

        stale = client.post(
            f"{chapter_url}/buffer",
            data={
                "content": "基于空白 HEAD 的过期编辑。",
                "expected_head_version_id": "",
                "csrf": csrf_from(client.get(response.headers["location"]).text),
            },
        )
        assert stale.status_code == 409
        assert stale.json()["conflict"] is True
        chapter = application.state.database.get_novel_chapter(
            user_id,
            project_id,
            chapter_id,
        )
        assert chapter["head_version_id"] == first_head


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

        page = client.get("/import")
        response = client.post(
            "/import",
            data={"title": "测试小说", "csrf": csrf_from(page.text)},
            files={
                "work_file": (
                    "novel.txt",
                    "第一章 开始\n这是第一章的正文内容。\n第二章 继续\n这是第二章的正文内容。".encode(),
                    "text/plain",
                )
            },
            follow_redirects=False,
        )
        response = commit_text_import(client, response)
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
        assert "全书文风画像" in job_page.text
        analysis_match = re.search(r'href="/analyses/([^"]+)"', job_page.text)
        assert analysis_match
        analysis_page = client.get(f"/analyses/{analysis_match.group(1)}")
        assert analysis_page.status_code == 200
        assert "章节摘要" in analysis_page.text
        assert "文风证据" in analysis_page.text
        export = client.get(f"/jobs/{job_id}/export.json")
        assert export.status_code == 200
        assert len(export.json()["chapters"]) == 2

        document_id = document_url.rsplit("/", 1)[-1]
        user = client.app.state.database.get_user_by_username("测试者")
        work = client.app.state.database.get_work_for_document(
            int(user["id"]), document_id
        )
        archive = client.get(
            f"/documents/{document_id}"
            "?view=archive&archive_tab=analysis"
        )
        assert "深度分析" in archive.text
        analysis_id = analysis_match.group(1)
        assert (
            f"/works/{work['id']}/archive/analyses/{analysis_id}/adopt"
            not in archive.text
        )
        assert "创建 main 后可采纳" in archive.text
        adopted = client.post(
            f"/works/{work['id']}/archive/analyses/{analysis_id}/adopt",
            data={
                "category": "structure",
                "title": "开篇推进规则",
                "content": "每章先建立一个可核对的新问题，再延迟解释。",
                "return_to": (
                    f"/documents/{document_id}"
                    "?view=archive&archive_tab=analysis"
                ),
                "csrf": csrf_from(archive.text),
            },
            follow_redirects=False,
        )
        assert adopted.status_code == 303
        assert "error=" in adopted.headers["location"]

        branch = client.post(
            f"/works/{work['id']}/main",
            data={
                "base_version_id": work["source_version"]["id"],
                "intent": "rewrite",
                "csrf": csrf_from(archive.text),
            },
            follow_redirects=False,
        )
        assert branch.status_code == 303
        project_id = branch.headers["location"].split("/novels/", 1)[1].split(
            "/", 1
        )[0]
        source_archive = client.get(
            f"/documents/{document_id}"
            "?view=archive&archive_tab=analysis"
        )
        adopted = client.post(
            f"/works/{work['id']}/archive/analyses/{analysis_id}/adopt",
            data={
                "category": "structure",
                "title": "开篇推进规则",
                "content": "每章先建立一个可核对的新问题，再延迟解释。",
                "return_to": (
                    f"/documents/{document_id}"
                    "?view=archive&archive_tab=analysis"
                ),
                "csrf": csrf_from(source_archive.text),
            },
            follow_redirects=False,
        )
        assert adopted.status_code == 303
        assert "adopted=true" in adopted.headers["location"]
        chapters = client.app.state.database.list_novel_chapters(
            int(user["id"]), project_id
        )
        writing_context = client.app.state.database.get_writing_context(
            int(user["id"]), str(chapters[0]["id"])
        )
        assert writing_context["confirmed_archive_rules"][0]["content"] == (
            "每章先建立一个可核对的新问题，再延迟解释。"
        )




def test_analysis_technique_card_can_bind_to_project(tmp_path):
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

        page = client.get("/import")
        response = client.post(
            "/import",
            data={"title": "参考小说", "csrf": csrf_from(page.text)},
            files={
                "work_file": (
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
        response = commit_text_import(client, response)
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
        workbench_url = response.headers["location"]
        project_id = project_id_from_workbench(workbench_url)
        project_url = f"/novels/{project_id}"

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
        user = application.state.database.get_user_by_username("技法回流作者")
        bindings = TechniqueService(
            application.state.database
        ).list_project_bindings(
            user_id=int(user["id"]),
            project_id=project_id,
        )
        assert len(bindings) == 1
        assert bindings[0]["name"] == "让信息先影响行动、再补充解释"
        assert bindings[0]["usage_modes"] == ["plan", "write", "audit"]


def test_registration_can_be_closed_even_for_empty_database(tmp_path):
    settings = replace(make_settings(tmp_path), allow_registration=False)
    with TestClient(create_app(settings)) as client:
        response = client.get("/register", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/login"
        login = client.get("/login")
        assert "创建一个" not in login.text


def test_auxiliary_pages_use_formal_header(tmp_path):
    application = create_app(make_settings(tmp_path))
    with TestClient(application) as client:
        login = client.get("/login")
        assert 'class="brand-symbol"' in login.text
        assert "brand-mark" not in login.text
        assert ">写</span>" not in login.text

        register = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "统一页头作者",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": csrf_from(register.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        techniques = client.get("/techniques")
        header = re.search(
            r'<header class="site-header">.*?</header>',
            techniques.text,
            re.DOTALL,
        )
        assert header
        rendered = header.group(0)
        assert 'class="brand-symbol"' in rendered
        assert "brand-mark" not in rendered
        assert ">写</span>" not in rendered
        assert 'href="/dashboard">作品库</a>' in rendered
        assert 'href="/techniques">技法库</a>' in rendered
        assert 'aria-label="模型配置"' in rendered
        assert 'aria-label="退出登录"' in rendered
        assert 'href="/import"' not in rendered


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
        assert "开启思考模式" not in page.text
        assert "思考强度" not in page.text
        assert "系统会按任务自动选择快速、推理或深度推理策略" not in page.text
        assert "提示词层级" not in page.text
        assert "隐私" not in page.text
        assert "退出登录" not in page.text
        assert "模型供应商" in page.text
        assert "提示词" in page.text
        assert "通用提示词" not in page.text
        assert "全局系统提示词" not in page.text
        prompt_page = client.get("/settings/api?tab=prompts")
        assert prompt_page.status_code == 200
        assert "系统提示词" in prompt_page.text
        assert "模型受限时如何继续" not in prompt_page.text
        assert "留空时不注入额外策略" not in prompt_page.text
        assert "模型服务商" not in prompt_page.text
        adapter_prompt = "受限时保留事件因果，改用非露骨叙述。"
        response = client.post(
            "/settings/model-adapter",
            data={
                "provider": "deepseek",
                "model_adapter_prompt": adapter_prompt,
                "csrf": csrf_from(prompt_page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert response.headers["location"] == (
            "/settings/api?provider=deepseek&tab=prompts"
            "&adapter_saved=true"
        )
        page = client.get(response.headers["location"])
        assert "提示词已保存" in page.text
        assert adapter_prompt in page.text
        raw_key = "sk-personal-secret-5678"
        response = client.post(
            "/settings/api",
            data={
                "api_key": raw_key,
                "model": "deepseek-v4-flash",
                "csrf": csrf_from(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert response.headers["location"] == "/settings/api?saved=true"

        saved_page = client.get(response.headers["location"])
        assert "个人模型已配置" in saved_page.text
        assert "sk-••••5678" in saved_page.text
        assert raw_key not in saved_page.text
        assert "data-api-key-toggle" in saved_page.text
        assert 'aria-label="删除 DeepSeek 配置"' in saved_page.text

        revealed = client.post(
            "/api/settings/api-key",
            data={
                "provider": "deepseek",
                "csrf": csrf_from(saved_page.text),
            },
        )
        assert revealed.status_code == 200
        assert revealed.json() == {"api_key": raw_key}
        assert revealed.headers["cache-control"] == "no-store, private"
        assert revealed.headers["pragma"] == "no-cache"

        user = application.state.database.get_user_by_username("个人API用户")
        credential = application.state.database.get_api_credential(user["id"])
        assert "system_prompt" not in credential
        assert raw_key not in credential["encrypted_key"]
        assert (
            application.state.credential_cipher.decrypt(
                credential["encrypted_key"]
            )
            == raw_key
        )
        assert (
            application.state.database.get_model_adapter_prompt(
                user["id"]
            )
            == adapter_prompt
        )

        response = client.post(
            "/settings/api/delete",
            data={
                "provider": "deepseek",
                "csrf": csrf_from(saved_page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert application.state.database.get_api_credential(user["id"]) is None
        assert (
            application.state.database.get_model_adapter_prompt(
                user["id"]
            )
            == adapter_prompt
        )


def test_api_key_reveal_is_scoped_to_current_user_and_provider(tmp_path):
    application = create_app(make_settings(tmp_path))
    with TestClient(application) as client:
        page = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "凭据所有者",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": csrf_from(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        page = client.get("/settings/api")
        owner_key = "sk-owner-secret-1234"
        response = client.post(
            "/settings/api",
            data={
                "provider": "deepseek",
                "api_key": owner_key,
                "model": "deepseek-chat",
                "models": ["deepseek-chat"],
                "csrf": csrf_from(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        settings_page = client.get("/settings/api")
        missing_provider = client.post(
            "/api/settings/api-key",
            data={
                "provider": "openai_compatible",
                "csrf": csrf_from(settings_page.text),
            },
        )
        assert missing_provider.status_code == 404
        assert owner_key not in missing_provider.text
        assert (
            missing_provider.headers["cache-control"]
            == "no-store, private"
        )

        logout = client.post(
            "/logout",
            data={"csrf": csrf_from(settings_page.text)},
            follow_redirects=False,
        )
        assert logout.status_code == 303

        page = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "其他凭据用户",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": csrf_from(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        other_settings = client.get("/settings/api")
        other_reveal = client.post(
            "/api/settings/api-key",
            data={
                "provider": "deepseek",
                "csrf": csrf_from(other_settings.text),
            },
        )
        assert other_reveal.status_code == 404
        assert owner_key not in other_reveal.text


def test_user_can_select_ollama_without_api_key(tmp_path):
    application = create_app(make_settings(tmp_path))
    with TestClient(application) as client:
        page = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "本机模型用户",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": csrf_from(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        page = client.get("/settings/api")
        assert "Google Gemini" in page.text
        assert "Ollama" in page.text
        assert "自定义 OpenAI" in page.text
        assert "OpenCode Go" in page.text
        response = client.post(
            "/settings/api",
            data={
                "provider": "ollama",
                "base_url": "http://192.168.50.20:11434/v1/",
                "api_key": "",
                "model": "qwen3:8b",
                "csrf": csrf_from(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        user = application.state.database.get_user_by_username(
            "本机模型用户"
        )
        credential = application.state.database.get_api_credential(
            user["id"]
        )
        assert credential["provider"] == "ollama"
        assert credential["base_url"] == (
            "http://192.168.50.20:11434/v1"
        )
        assert credential["model"] == "qwen3:8b"
        assert credential["key_hint"] == "无需 Key"
        assert (
            application.state.credential_cipher.decrypt(
                credential["encrypted_key"]
            )
            == ""
        )


def test_user_can_save_keyless_openai_compatible_endpoint(tmp_path):
    application = create_app(make_settings(tmp_path))
    with TestClient(application) as client:
        page = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "兼容接口用户",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": csrf_from(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        page = client.get("/settings/api")
        response = client.post(
            "/settings/api",
            data={
                "provider": "openai_compatible",
                "base_url": "https://gateway.example.com/openai/v1/",
                "api_key": "",
                "model": "my-chat-model",
                "csrf": csrf_from(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        user = application.state.database.get_user_by_username(
            "兼容接口用户"
        )
        credential = application.state.database.get_api_credential(
            user["id"]
        )
        assert credential["provider"] == "openai_compatible"
        assert credential["base_url"] == (
            "https://gateway.example.com/openai/v1"
        )
        assert credential["model"] == "my-chat-model"
        assert credential["key_hint"] == "未设置 Key"
        assert (
            application.state.credential_cipher.decrypt(
                credential["encrypted_key"]
            )
            == ""
        )


def test_model_catalog_uses_submitted_openai_compatible_base_url(
    tmp_path, monkeypatch
):
    seen = {}

    async def fake_fetch_models(**kwargs):
        seen.update(kwargs)
        return ["catalog-model"]

    monkeypatch.setattr("app.main.fetch_models", fake_fetch_models)
    application = create_app(make_settings(tmp_path))
    with TestClient(application) as client:
        page = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "兼容目录用户",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": csrf_from(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        page = client.get("/settings/api")
        response = client.post(
            "/api/settings/models",
            data={
                "provider": "openai_compatible",
                "base_url": "https://gateway.example.com/openai/v1/",
                "api_key": "",
                "csrf": csrf_from(page.text),
            },
        )
        assert response.status_code == 200
        assert response.json() == {"models": ["catalog-model"]}
        assert seen["provider_id"] == "openai_compatible"
        assert seen["base_url"] == (
            "https://gateway.example.com/openai/v1"
        )
        assert seen["api_key"] is None


def test_provider_settings_keep_separate_keys_and_model_lists(tmp_path):
    application = create_app(make_settings(tmp_path))
    with TestClient(application) as client:
        page = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "多服务商用户",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": csrf_from(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        page = client.get("/settings/api")
        adapter_prompt = "所有模型受限时均保留事件结果和人物后果。"
        response = client.post(
            "/settings/model-adapter",
            data={
                "provider": "deepseek",
                "model_adapter_prompt": adapter_prompt,
                "csrf": csrf_from(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        prompt_page = client.get(response.headers["location"])
        assert adapter_prompt in prompt_page.text
        deepseek_key = "sk-deepseek-provider-5678"
        response = client.post(
            "/settings/api",
            data={
                "provider": "deepseek",
                "api_key": deepseek_key,
                "model": "deepseek-reasoner",
                "models": ["deepseek-chat", "deepseek-reasoner"],
                "csrf": csrf_from(prompt_page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        compatible_page = client.get(
            "/settings/api?provider=openai_compatible"
        )
        assert "还没有模型" in compatible_page.text
        assert adapter_prompt not in compatible_page.text
        compatible_prompt_page = client.get(
            "/settings/api?provider=openai_compatible&tab=prompts"
        )
        assert adapter_prompt in compatible_prompt_page.text
        compatible_key = "compatible-provider-key-9012"
        response = client.post(
            "/settings/api",
            data={
                "provider": "openai_compatible",
                "base_url": "https://gateway.example.com/v1",
                "api_key": compatible_key,
                "model": "writer-fast",
                "models": ["writer-fast", "writer-pro"],
                "csrf": csrf_from(compatible_page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        deepseek_page = client.get("/settings/api?provider=deepseek")
        assert adapter_prompt not in deepseek_page.text
        assert "已保存：sk-••••5678（留空保留）" in deepseek_page.text
        assert 'name="models" value="deepseek-chat"' in deepseek_page.text
        assert 'name="models" value="deepseek-reasoner"' in deepseek_page.text
        assert deepseek_key not in deepseek_page.text

        compatible_page = client.get(
            "/settings/api?provider=openai_compatible"
        )
        assert "已保存：••••9012（留空保留）" in compatible_page.text
        assert "https://gateway.example.com/v1" in compatible_page.text
        assert 'name="models" value="writer-fast"' in compatible_page.text
        assert 'name="models" value="writer-pro"' in compatible_page.text
        assert compatible_key not in compatible_page.text
        assert "接口返回的模型" in compatible_page.text
        assert "加入我的模型" in compatible_page.text
        assert 'aria-label="删除 DeepSeek 配置"' in compatible_page.text
        assert (
            'aria-label="删除 自定义 OpenAI 配置"'
            in compatible_page.text
        )

        user = application.state.database.get_user_by_username(
            "多服务商用户"
        )
        assert application.state.database.has_api_credential(
            user["id"], "deepseek"
        )
        assert application.state.database.has_api_credential(
            user["id"], "openai_compatible"
        )


def test_workbench_quality_mode_uses_saved_two_model_strategy(
    tmp_path, monkeypatch
):
    queued = {}

    def fake_queue_message(_service, **kwargs):
        queued.update(kwargs)
        return "queued-message"

    monkeypatch.setattr(
        "app.assistant_chat_service.AssistantChatService.queue_message",
        fake_queue_message,
    )
    application = create_app(make_settings(tmp_path))
    with TestClient(application) as client:
        page = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "聊天选模型用户",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": csrf_from(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        settings_page = client.get("/settings/api")
        response = client.post(
            "/settings/api",
            data={
                "provider": "deepseek",
                "api_key": "sk-chat-model-picker-1234",
                "model": "deepseek-chat",
                "models": ["deepseek-chat", "deepseek-reasoner"],
                "csrf": csrf_from(settings_page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        routing_page = client.get("/settings/api?tab=routing")
        response = client.post(
            "/settings/model-routing",
            data={
                "fast_model_choice": "deepseek|deepseek-chat",
                "quality_model_choice": "deepseek|deepseek-reasoner",
                "default_quality_mode": "standard",
                "provider": "deepseek",
                "csrf": csrf_from(routing_page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        dashboard = client.get("/dashboard")
        created = client.post(
            "/novels/new/blank",
            data={"csrf": csrf_from(dashboard.text)},
            follow_redirects=False,
        )
        workbench = client.get(created.headers["location"])
        project_id = project_id_from_workbench(created.headers["location"])
        assert 'name="quality_mode"' in workbench.text
        assert "data-quality-mode" in workbench.text
        assert 'value="standard"' in workbench.text
        assert "Standard · 自动" in workbench.text
        model_choices = client.get("/api/settings/chat-models")
        assert model_choices.status_code == 200
        assert model_choices.json()["default"] == (
            "deepseek|deepseek-chat"
        )
        assert [
            model["value"]
            for model in model_choices.json()["groups"][0]["models"]
        ] == [
            "deepseek|deepseek-chat",
            "deepseek|deepseek-reasoner",
        ]
        assert model_choices.json()["default_quality_mode"] == "standard"
        assert [
            mode["value"]
            for mode in model_choices.json()["quality_modes"]
        ] == ["low", "standard", "max"]

        remembered = client.post(
            "/api/settings/quality-mode",
            data={
                "csrf": csrf_from(workbench.text),
                "quality_mode": "low",
            },
        )
        assert remembered.status_code == 200
        assert remembered.json()["quality_mode"] == "low"
        refreshed_workbench = client.get(created.headers["location"])
        assert re.search(
            r'<option(?=[^>]*value="low")(?=[^>]*selected)[^>]*>',
            refreshed_workbench.text,
            re.DOTALL,
        )

        response = client.post(
            f"/novels/{project_id}/assistant/messages",
            data={
                "csrf": csrf_from(refreshed_workbench.text),
                "question": "先讨论这一章的悬念推进。",
                "quality_mode": "max",
                "return_view": "settings",
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert queued["provider"] == "deepseek"
        assert queued["model"] == "deepseek-reasoner"
        assert queued["credential_source"] == "personal"
        assert queued["quality_mode"] == "max"


def test_remote_provider_rejects_missing_key_and_ignores_legacy_thinking(
    tmp_path,
):
    application = create_app(make_settings(tmp_path))
    with TestClient(application) as client:
        page = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "远程模型用户",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": csrf_from(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        page = client.get("/settings/api")
        response = client.post(
            "/settings/api",
            data={
                "provider": "openai",
                "api_key": "",
                "model": "gpt-4.1-mini",
                "csrf": csrf_from(page.text),
            },
        )
        assert response.status_code == 400
        assert "请填写 OpenAI API Key" in response.text

        response = client.post(
            "/settings/api",
            data={
                "provider": "gemini",
                "api_key": "gemini-test-key",
                "model": "gemini-2.5-flash",
                "thinking": "enabled",
                "reasoning_effort": "high",
                "csrf": csrf_from(response.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        user = application.state.database.get_user_by_username(
            "远程模型用户"
        )
        credential = application.state.database.get_api_credential(
            user["id"], "gemini"
        )
        assert credential["provider"] == "gemini"
        assert "thinking" not in credential
        assert "reasoning_effort" not in credential


def test_unified_workbench_uses_five_material_sections(tmp_path):
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
        assert "默认单章篇幅" not in workbench.text
        assert "作品资料" in workbench.text
        for settings_tab in (
            "作品概览",
            "世界",
            "人物",
            "剧情与结构",
            "叙事与文风",
        ):
            assert settings_tab in workbench.text
        assert 'data-settings-panel="core"' in workbench.text
        assert 'data-settings-panel="parameters"' not in workbench.text
        assert "创作参数" not in workbench.text
        assert "补充设定" not in workbench.text
        assert "高级规划" not in workbench.text
        assert "模型识别任务 · 服务端裁决权限" not in workbench.text
        assert "data-agent-note" not in workbench.text
        assert "data-chapter-workflow" not in workbench.text
        assert 'class="studio-chat-input"' in workbench.text
        assert "studio-agent-switcher" not in workbench.text
        assert 'name="agent_role"' not in workbench.text
        assert '<p class="studio-section-label">文件</p>' not in workbench.text
        assert "导出正文" not in workbench.text
        assert "导出作品归档" not in workbench.text
        model_settings_panel_url = (
            "/settings/api?embedded=true&return_to="
            + quote(workbench_url, safe="")
        )
        assert 'data-model-settings-open aria-label="模型配置"' in (
            workbench.text
        )
        assert "管理模型" not in workbench.text
        assert (
            'data-settings-url="'
            + model_settings_panel_url.replace("&", "&amp;")
            + '"'
        ) in workbench.text
        theme_position = workbench.text.index("data-theme-toggle")
        model_position = workbench.text.index(
            "data-model-settings-open aria-label=\"模型配置\""
        )
        ai_position = workbench.text.index('data-panel-toggle="ai"')
        assert theme_position < model_position < ai_position
        settings_page = client.get(model_settings_panel_url)
        assert settings_page.status_code == 200
        assert (
            'class="studio-surface studio-settings-page '
            'studio-settings-embedded"'
            in settings_page.text
        )
        assert "返回上一页" not in settings_page.text
        assert "模型供应商" in settings_page.text
        assert "提示词" in settings_page.text
        assert 'name="embedded" value="true"' in settings_page.text
        embedded_prompt_url = (
            "/settings/api?provider=deepseek&tab=prompts"
            "&embedded=true&return_to="
            + quote(workbench_url, safe="")
        )
        assert (
            'href="'
            + embedded_prompt_url.replace("&", "&amp;")
            + '"'
        ) in settings_page.text
        embedded_prompt_page = client.get(embedded_prompt_url)
        assert "系统提示词" in embedded_prompt_page.text
        assert "模型受限时如何继续" not in embedded_prompt_page.text
        assert "留空时不注入额外策略" not in embedded_prompt_page.text
        assert "模型服务商" not in embedded_prompt_page.text
        assert "提示词层级" not in settings_page.text
        assert "运行策略" not in settings_page.text
        assert "隐私" not in settings_page.text
        assert "退出登录" not in settings_page.text
        unsafe_settings = client.get(
            "/settings/api?return_to=https://example.com"
        )
        assert 'href="/dashboard" aria-label="返回上一页"' in (
            unsafe_settings.text
        )
        assert "https://example.com" not in unsafe_settings.text
        for legacy_path in (
            f"/novels/{project_id}#story-planner",
            f"/novels/{project_id}/structure-health",
            f"/novels/{project_id}/continuity",
            f"/novels/{project_id}/editing-memory",
        ):
            assert f'href="{legacy_path}"' not in workbench.text
        assert '<p class="studio-section-label">创作系统</p>' not in (
            workbench.text
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
        assert 'class="studio-manuscript-view"' in chapter_page.text
        assert "data-chapter-workflow" not in chapter_page.text
        assert "章节创作流程" not in chapter_page.text
        assert "规划本章" not in chapter_page.text
        assert (
            f'href="/novels/{project_id}/chapters/{chapter_id}/task-card"'
            not in chapter_page.text
        )
        assert (
            f'href="/novels/{project_id}/chapters/{chapter_id}/scenes"'
            not in chapter_page.text
        )
        assert "data-save-before-navigation" in chapter_page.text
        assert "data-save-before-submit" in chapter_page.text
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
        assert "/ 约 3000 字" not in saved_page.text
        assert "data-char-count" not in saved_page.text
        assert (
            f'data-actual-char-count>{len("录音带转到第三圈时，海面仍旧平静。")} 字'
            in saved_page.text
        )
        assert (
            '<span data-save-status aria-live="polite" hidden></span>'
            in saved_page.text
        )
        assert (
            f'href="/novels/{project_id}/workbench?view=archive'
            f'&archive_tab=analysis#story-memory-{chapter_id}"'
            in saved_page.text
        )

        user_id = application.state.database.get_user_by_username(
            "工作台作者"
        )["id"]
        deadline = time.monotonic() + 3
        chapter_memory = None
        while time.monotonic() < deadline:
            chapter_memory = MemoryService(
                application.state.database
            ).get_chapter_memory(
                user_id=user_id,
                project_id=project_id,
                chapter_id=chapter_id,
            )
            if chapter_memory:
                break
            time.sleep(0.03)
        assert chapter_memory
        analysis_page = client.get(
            f"/novels/{project_id}/workbench"
            "?view=archive&archive_tab=analysis"
        )
        assert "故事记忆" in analysis_page.text
        assert "正文保存后自动更新。" in analysis_page.text
        assert f'id="story-memory-{chapter_id}"' in analysis_page.text
        assert "录音带转到第三圈时" in analysis_page.text
        assert "事件" in analysis_page.text
        assert "依据：" in analysis_page.text

        project = application.state.database.get_novel_project(
            application.state.database.get_user_by_username("工作台作者")[
                "id"
            ],
            project_id,
        )
        settings_page = client.get(
            f"/novels/{project_id}/workbench?view=archive&archive_tab=creative"
        )
        response = client.post(
            f"/novels/{project_id}/settings",
            data={
                "title": project["title"],
                "genre": project["genre"],
                "premise": project["premise"],
                "theme": "记忆如何改变责任",
                "story_promise": "克制而持续的悬念",
                "target_audience": "偏爱人物驱动悬疑的读者",
                "core_appeal": "未来警报与旧案互相印证",
                "settings_tab": "core",
                "csrf": csrf_from(settings_page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert (
            "view=archive&archive_tab=creative&settings_tab=core&saved=true"
            in response.headers["location"]
        )
        project = application.state.database.get_novel_project(
            application.state.database.get_user_by_username("工作台作者")[
                "id"
            ],
            project_id,
        )
        assert project["theme"] == "记忆如何改变责任"
        assert project["ai_instructions"] == ""

        logout = client.post(
            "/logout",
            data={"csrf": csrf_from(settings_page.text)},
            follow_redirects=False,
        )
        assert logout.status_code == 303
        register = client.get("/register")
        other_user = client.post(
            "/register",
            data={
                "username": "其他工作台作者",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": csrf_from(register.text),
            },
            follow_redirects=False,
        )
        assert other_user.status_code == 303
        for owner_scoped_path in (
            f"/novels/{project_id}/workbench",
            f"/novels/{project_id}/structure-health",
            f"/novels/{project_id}/continuity",
            f"/novels/{project_id}/editing-memory",
            f"/novels/{project_id}/export.txt",
            f"/novels/{project_id}/chapters/{chapter_id}/task-card",
            f"/novels/{project_id}/chapters/{chapter_id}/scenes",
        ):
            assert client.get(owner_scoped_path).status_code == 404


def test_five_material_sections_are_editable_and_fixed_in_tags(tmp_path):
    application = create_app(make_settings(tmp_path))
    with TestClient(application) as client:
        register = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "五项资料作者",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": csrf_from(register.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        dashboard = client.get("/dashboard")
        created = client.post(
            "/novels/new/blank",
            data={"csrf": csrf_from(dashboard.text)},
            follow_redirects=False,
        )
        assert created.status_code == 303
        project_id = project_id_from_workbench(created.headers["location"])
        page = client.get(created.headers["location"])
        csrf = csrf_from(page.text)

        overview = client.post(
            f"/novels/{project_id}/settings",
            data={
                "title": "无潮之城",
                "genre": "幻想悬疑",
                "premise": "潮汐消失后，两名守塔人必须找回城市的时间。",
                "theme": "共同记忆是否构成真实",
                "story_promise": "逐层揭开城市失去潮汐的代价",
                "target_audience": "偏爱关系驱动谜团的读者",
                "core_appeal": "时间规则与搭档信任互相牵制",
                "settings_tab": "core",
                "csrf": csrf,
            },
            follow_redirects=False,
        )
        assert overview.status_code == 303
        world = client.post(
            f"/novels/{project_id}/settings",
            data={
                "world_setting": "城市以潮汐计时，退潮后所有钟表停摆。",
                "settings_tab": "world",
                "csrf": csrf,
            },
            follow_redirects=False,
        )
        assert world.status_code == 303
        world_entry = client.post(
            f"/novels/{project_id}/world-entries",
            data={
                "entry_type": "rule",
                "name": "潮汐钟",
                "description": "全城唯一仍能记录时间的装置。",
                "constraints": "每次启动都会抹去使用者一天的记忆。",
                "csrf": csrf,
            },
            follow_redirects=False,
        )
        assert world_entry.status_code == 303

        for name, role, goal in (
            ("岑野", "守塔人", "恢复潮汐钟"),
            ("苏弥", "档案员", "找回被抹去的城市史"),
        ):
            character = client.post(
                f"/novels/{project_id}/characters",
                data={
                    "name": name,
                    "role": role,
                    "external_goal": goal,
                    "internal_need": "学会把判断交给他人验证",
                    "central_conflict": "必须用自己的记忆换取线索",
                    "secret": "曾经主动启动过潮汐钟",
                    "traits": "克制，观察敏锐",
                    "speech_style": "短句，很少直接回答感受",
                    "background": "在无潮之夜后留在城中。",
                    "initial_state": "不信任搭档",
                    "character_arc": "从独自承担到共享真相",
                    "csrf": csrf,
                },
                follow_redirects=False,
            )
            assert character.status_code == 303

        user = application.state.database.get_user_by_username(
            "五项资料作者"
        )
        user_id = int(user["id"])
        characters = application.state.database.list_novel_characters(
            user_id, project_id
        )
        relationship = client.post(
            f"/novels/{project_id}/relationships",
            data={
                "character_a_id": characters[0]["id"],
                "character_b_id": characters[1]["id"],
                "relationship": "互相需要、互不完全信任的搭档",
                "tension": "双方都隐瞒了一次启动记录",
                "change_direction": "从交换线索到共同承担记忆损失",
                "csrf": csrf,
            },
            follow_redirects=False,
        )
        assert relationship.status_code == 303

        blueprint = client.post(
            f"/novels/{project_id}/story-blueprint",
            data={
                "central_question": "潮汐为何消失？",
                "protagonist_goal": "让城市重新开始计时",
                "core_conflict": "每条线索都要以个人记忆为代价",
                "stakes": "城市将永远停在同一天",
                "opening_state": "全城钟表停摆",
                "ending_state": "两人共同决定如何恢复时间",
                "major_turns": "发现潮汐钟的代价\n确认两人都改写过记录",
                "must_payoffs": "解释无潮之夜\n兑现两人的信任选择",
                "forbidden_shortcuts": "不能用无代价的魔法修复",
                "author_notes": "",
                "action": "confirm",
                "csrf": csrf,
            },
            follow_redirects=False,
        )
        assert blueprint.status_code == 303
        volume = client.post(
            f"/novels/{project_id}/volumes",
            data={
                "title": "停摆之城",
                "goal": "确认潮汐钟仍能启动",
                "start_state": "两人互不信任",
                "end_state": "两人决定共同调查",
                "major_conflict": "谁来支付第一次记忆代价",
                "payoff": "找到第一条未被改写的记录",
                "csrf": csrf,
            },
            follow_redirects=False,
        )
        assert volume.status_code == 303
        voice = client.post(
            f"/novels/{project_id}/voice",
            data={
                "point_of_view": "第三人称限知",
                "style_guide": "克制，不替人物总结全部情绪。",
                "narrative_tense": "过去时",
                "narrative_distance": "紧贴当前视角人物的感官",
                "tone": "冷静、潮湿、持续不安",
                "narration_rules": "只写当前人物可观察或推断的内容。",
                "sentence_rhythm": "调查段落使用短句。",
                "dialogue_voice": "重要问题经常以反问回避。",
                "sensory_palette": "盐、锈、停摆机械。",
                "metaphor_policy": "只使用城市生活经验中的意象。",
                "allowed_omissions": "用动作承担恐惧。",
                "preferred_patterns": "动作先于判断",
                "banned_expressions": "内心深处",
                "style_examples": (
                    "他听见齿轮停下，才想起城里已经没有潮声。"
                ),
                "author_notes": "",
                "action": "confirm",
                "csrf": csrf,
            },
            follow_redirects=False,
        )
        assert voice.status_code == 303

        chapter = client.post(
            f"/novels/{project_id}/chapters",
            data={"title": "无潮之夜", "csrf": csrf},
            follow_redirects=False,
        )
        assert chapter.status_code == 303
        chapter_id = chapter.headers["location"].split("chapter_id=", 1)[1]
        chapter_page = client.get(chapter.headers["location"])
        saved_chapter = client.post(
            f"/novels/{project_id}/chapters/{chapter_id}/save",
            data={
                "content": "潮汐停下的那一刻，塔顶最后一枚齿轮也失去了声音。",
                "csrf": csrf_from(chapter_page.text),
            },
            follow_redirects=False,
        )
        assert saved_chapter.status_code == 303
        work = application.state.database.get_work_for_project(
            user_id, project_id
        )
        assert work
        tagged = client.post(
            f"/works/{work['id']}/tags",
            data={"label": "资料快照", "csrf": csrf},
            follow_redirects=False,
        )
        assert tagged.status_code == 303
        assert "error=" not in tagged.headers["location"], (
            tagged.headers["location"]
        )
        fixed_work = application.state.database.get_work(
            user_id, str(work["id"])
        )
        fixed = next(
            item
            for item in fixed_work["tag_versions"]
            if item["label"] == "资料快照"
        )
        snapshot = fixed["creative_snapshot"]
        assert snapshot["schema"] == "readraft-creative-snapshot-v2"
        assert snapshot["project"]["theme"] == "共同记忆是否构成真实"
        assert snapshot["world_entries"][0]["name"] == "潮汐钟"
        assert snapshot["characters"][0]["external_goal"]
        assert snapshot["character_relationships"][0]["relationship"]
        assert snapshot["story_blueprint"]["central_question"] == (
            "潮汐为何消失？"
        )
        assert snapshot["volumes"][0]["title"] == "停摆之城"
        assert snapshot["voice"]["narrative_tense"] == "过去时"

        document_id = fixed["document_id"]
        readonly = client.get(
            f"/documents/{document_id}?view=archive"
            "&archive_tab=creative&settings_tab=world"
        )
        assert readonly.status_code == 200
        assert "潮汐钟" in readonly.text
        assert "每次启动都会抹去" in readonly.text
        assert "创作参数" not in readonly.text
        assert 'action="/novels/' not in readonly.text


def test_zero_input_project_enters_settings_and_ai_writes_directly(
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
        assert "view=archive" in response.headers["location"]
        assert "archive_tab=creative" in response.headers["location"]
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
        assert "作品概览" in workbench.text

        question = "我想写一个记者在雨夜收到七年前来信的悬疑故事。"
        response = client.post(
            f"/novels/{project_id}/assistant/messages",
            data={
                "csrf": csrf_from(workbench.text),
                "question": question,
                "conversation_id": "",
                "return_view": "archive",
                "return_archive_tab": "creative",
                "return_settings_tab": "core",
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        conversation_url = response.headers["location"]
        deadline = time.monotonic() + 3
        rendered = ""
        while time.monotonic() < deadline:
            rendered = client.get(conversation_url).text
            if "已直接写入" in rendered:
                break
            time.sleep(0.03)
        assert "已更新作品资料" in rendered
        assert "已直接写入" in rendered
        assert re.search(r"<small>\s*AI\s*</small>", rendered)
        assert "/apply-settings" not in rendered
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
        work = database.get_work_for_project(user_id, project_id)
        assert work
        project_dir = tmp_path / "novels" / str(user_id) / project_id
        assert project_dir.is_dir()
        conversation_id = (
            application.state.assistant_chat_service.conversations.create(
                user_id=user_id,
                scope_type="project",
                title="删除作品前的讨论",
                project_id=project_id,
            )
        )
        assert application.state.assistant_chat_service.conversations.get(
            user_id=user_id,
            conversation_id=conversation_id,
        )

        dashboard = client.get("/dashboard")
        assert 'data-delete-work-form' in dashboard.text
        assert f'action="/works/{work["id"]}/delete"' in dashboard.text
        assert "退出登录" in dashboard.text
        dashboard_theme_position = dashboard.text.index(
            "data-theme-toggle"
        )
        dashboard_model_position = dashboard.text.index(
            "data-model-settings-open aria-label=\"模型配置\""
        )
        dashboard_logout_position = dashboard.text.index(
            'aria-label="退出登录"'
        )
        assert (
            dashboard_theme_position
            < dashboard_model_position
            < dashboard_logout_position
        )
        for export_path in (
            f"/novels/{project_id}/export.txt",
            f"/works/{work['id']}/export.readraft.zip",
        ):
            assert f'href="{export_path}"' in dashboard.text
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
            application.state.assistant_chat_service.conversations.get(
                user_id=user_id,
                conversation_id=conversation_id,
            )
            is None
        )
        assert not project_dir.exists()


def test_complete_work_archive_can_be_exported_and_imported_from_ui(
    tmp_path,
):
    application = create_app(make_settings(tmp_path))
    with TestClient(application) as client:
        page = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "作品迁移用户",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": csrf_from(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        dashboard = client.get("/dashboard")
        assert "导入作品" in dashboard.text
        assert "导入参考书" in dashboard.text
        created = client.post(
            "/novels/new/blank",
            data={"csrf": csrf_from(dashboard.text)},
            follow_redirects=False,
        )
        project_id = created.headers["location"].split("/")[2]
        user = application.state.database.get_user_by_username(
            "作品迁移用户"
        )
        work = application.state.database.get_work_for_project(
            int(user["id"]), project_id
        )
        assert work
        workbench = client.get(created.headers["location"])
        assert "导出作品归档" not in workbench.text
        dashboard = client.get("/dashboard")
        assert "导出 main 正文（TXT）" in dashboard.text
        assert "导出作品版本库（ZIP）" in dashboard.text
        assert 'class="studio-book-controls"' in dashboard.text
        assert 'class="studio-book-menu"' in dashboard.text
        assert 'class="studio-mode-actions"' not in dashboard.text
        assert 'data-tooltip="阅读"' not in dashboard.text
        assert 'data-tooltip="创作"' not in dashboard.text
        assert 'data-tooltip="作品档案"' not in dashboard.text
        assert '<span aria-hidden="true">›</span>' not in dashboard.text
        assert (
            f'href="/works/{work["id"]}/export.readraft.zip"'
            in dashboard.text
        )
        assert client.get("/projects/import").status_code == 404
        assert client.get("/upload").status_code == 404
        assert client.post("/upload").status_code == 404
        assert (
            client.post("/documents/removed/writing-branches").status_code
            == 404
        )
        assert (
            client.post("/novels/removed/reading-snapshots").status_code
            == 404
        )
        assert (
            client.get(
                f"/novels/{project_id}/export.readraft.zip"
            ).status_code
            == 404
        )
        with application.state.database.connection() as connection:
            tables = {
                str(row["name"])
                for row in connection.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type='table'
                      AND name IN ('work_editions', 'work_versions')
                    """
                ).fetchall()
            }
        assert tables == {"work_versions"}

        exported = client.get(
            f"/works/{work['id']}/export.readraft.zip"
        )
        assert exported.status_code == 200
        assert exported.headers["content-type"] == "application/zip"
        assert ".readraft.zip" in exported.headers["content-disposition"]

        import_page = client.get("/import")
        assert import_page.status_code == 200
        imported = client.post(
            "/import",
            data={
                "csrf": csrf_from(import_page.text),
            },
            files={
                "work_file": (
                    "book.readraft.zip",
                    exported.content,
                    "application/zip",
                )
            },
            follow_redirects=False,
        )
        assert imported.status_code == 303
        assert imported.headers["location"].startswith(
            "/dashboard?imported=true"
        )
        projects = application.state.database.list_novel_projects(
            user["id"]
        )
        assert len(projects) == 2
        works = application.state.database.list_works(int(user["id"]))
        assert len(works) == 2


def test_import_creates_readonly_source_then_one_editable_main(
    tmp_path,
):
    application = create_app(make_settings(tmp_path))
    with TestClient(application) as client:
        register = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "统一作品用户",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": csrf_from(register.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        import_page = client.get("/import")
        assert "原始版本" in import_page.text
        assert "导入后先做什么" not in import_page.text
        imported = client.post(
            "/import",
            data={
                "title": "雨夜来信",
                "csrf": csrf_from(import_page.text),
            },
            files={
                "work_file": (
                    "rain.txt",
                    (
                        "第一章 来信\n记者在雨夜收到一封旧信。\n"
                        "第二章 回声\n寄信人早已失踪。"
                    ).encode(),
                    "text/plain",
                )
            },
            follow_redirects=False,
        )
        imported = commit_text_import(client, imported)
        assert imported.status_code == 303
        assert imported.headers["location"].startswith("/documents/")
        reader = client.get(imported.headers["location"])
        assert "创建 main 并改写" in reader.text
        assert "创建 main 并续写" in reader.text
        assert "作品资料" in reader.text
        assert 'data-workbench' in reader.text
        assert 'data-panel="directory"' in reader.text
        assert 'data-panel="ai"' in reader.text
        assert 'aria-label="章节原文"' in reader.text
        assert "阅读对话" in reader.text

        document_id = imported.headers["location"].split("/documents/", 1)[1]
        reference_match = re.search(
            r'name="reference_chapter_id" value="([a-f0-9]+)"',
            reader.text,
        )
        assert reference_match
        reference_chapter_id = reference_match.group(1)
        new_chat = client.post(
            f"/documents/{document_id}/assistant/new",
            data={
                "reference_chapter_id": reference_chapter_id,
                "csrf": csrf_from(reader.text),
            },
            follow_redirects=False,
        )
        assert new_chat.status_code == 303
        embedded_chat = client.get(new_chat.headers["location"])
        assert "阅读对话" in embedded_chat.text
        assert 'data-panel="directory"' in embedded_chat.text

        user = application.state.database.get_user_by_username(
            "统一作品用户"
        )
        works = application.state.database.list_works(int(user["id"]))
        assert len(works) == 1
        assert not works[0]["has_main"]
        assert works[0]["source_version"]["document_id"] == document_id
        rewritten = client.post(
            f"/works/{works[0]['id']}/main",
            data={
                "base_version_id": works[0]["source_version"]["id"],
                "intent": "rewrite",
                "csrf": csrf_from(reader.text),
            },
            follow_redirects=False,
        )
        assert rewritten.status_code == 303
        assert rewritten.headers["location"].startswith("/novels/")

        works = application.state.database.list_works(int(user["id"]))
        assert len(works) == 1
        assert works[0]["has_main"]
        assert len(works[0]["versions"]) == 2
        assert len(works[0]["tag_versions"]) == 1
        assert works[0]["source_version"]["label"] == "原始版本"
        project_id = rewritten.headers["location"].split("/novels/", 1)[1].split(
            "/", 1
        )[0]
        chapters = application.state.database.list_novel_chapters(
            int(user["id"]), project_id
        )
        assert len(chapters) == 2
        assert "记者在雨夜" in Path(chapters[0]["content_path"]).read_text(
            encoding="utf-8"
        )

        dashboard = client.get("/dashboard")
        assert dashboard.text.count("studio-work-row") == 1
        assert "main" in dashboard.text
        assert "2 个 Tag" not in dashboard.text
        assert "1 个 Tag" in dashboard.text
        assert 'class="studio-mode-actions"' not in dashboard.text
        archive = client.get(f"/works/{works[0]['id']}/archive")
        assert archive.url.path == f"/novels/{project_id}/workbench"
        assert archive.url.params["view"] == "archive"
        assert archive.url.params["archive_tab"] == "creative"
        assert "作品资料" in archive.text
        assert 'data-panel="directory"' in archive.text
        assert 'data-panel="ai"' in archive.text

        versions = client.get(
            f"/novels/{project_id}/workbench"
            "?view=archive&archive_tab=versions"
        )
        assert "原始版本" in versions.text
        assert "main" in versions.text
        assert "创建 Tag" in versions.text
        analysis_archive = client.get(
            f"/novels/{project_id}/workbench"
            "?view=archive&archive_tab=analysis"
        )
        saved = client.post(
            f"/works/{works[0]['id']}/archive",
            data={
                "entry_type": "analysis_note",
                "category": "structure",
                "title": "节奏观察",
                "content": "第一章先给结果，再延迟解释。",
                "evidence": "第 1 章",
                "content_version_id": works[0]["main_version"]["id"],
                "return_to": (
                    f"/novels/{project_id}/workbench"
                    "?view=archive&archive_tab=analysis"
                ),
                "csrf": csrf_from(analysis_archive.text),
            },
            follow_redirects=False,
        )
        assert saved.status_code == 303
        saved_archive = client.get(saved.headers["location"])
        assert "节奏观察" in saved_archive.text
        assert "第一章先给结果" in saved_archive.text
        assert saved_archive.url.path == f"/novels/{project_id}/workbench"
        assert saved_archive.url.params["view"] == "archive"
        assert saved_archive.url.params["archive_tab"] == "analysis"

        entries = application.state.database.list_work_archive_entries(
            int(user["id"]), str(works[0]["id"])
        )
        note = next(
            entry
            for entry in entries
            if entry["entry_type"] == "analysis_note"
        )
        adopted = client.post(
            f"/works/{works[0]['id']}/archive/entries/{note['id']}/adopt",
            data={
                "category": "structure",
                "content": "章节开头先展示结果，随后延迟解释原因。",
                "return_to": (
                    f"/novels/{project_id}/workbench"
                    "?view=archive&archive_tab=creative"
                    "&settings_tab=structure"
                ),
                "csrf": csrf_from(saved_archive.text),
            },
            follow_redirects=False,
        )
        assert adopted.status_code == 303
        adopted_page = client.get(adopted.headers["location"])
        assert "已采纳为创作设定" in adopted_page.text
        assert "章节开头先展示结果" in adopted_page.text

        entries = application.state.database.list_work_archive_entries(
            int(user["id"]), str(works[0]["id"])
        )
        confirmed = next(
            entry
            for entry in entries
            if entry["entry_type"] == "creative_rule"
        )
        delete_form_id = f"archive-rule-delete-{confirmed['id']}"
        assert f'form="{delete_form_id}"' in adopted_page.text
        assert re.search(
            rf"<form\b[^>]*\bid=\"{re.escape(delete_form_id)}\"",
            adopted_page.text,
        )
        assert (
            f'action="/works/{works[0]["id"]}/archive/entries/'
            f'{confirmed["id"]}/delete"'
            in adopted_page.text
        )
        assert confirmed["status"] == "confirmed"
        assert confirmed["category"] == "structure"
        writing_context = application.state.database.get_writing_context(
            int(user["id"]), str(chapters[0]["id"])
        )
        assert writing_context["confirmed_archive_rules"] == [
            {
                "id": confirmed["id"],
                "category": "structure",
                "title": "节奏观察",
                "content": "章节开头先展示结果，随后延迟解释原因。",
                "evidence": "第 1 章",
                "provenance": "adopted",
                "updated_at": confirmed["updated_at"],
            }
        ]


def test_imported_source_can_create_main_and_fixed_tag(tmp_path):
    application = create_app(make_settings(tmp_path))
    with TestClient(application) as client:
        register = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "阅读版本用户",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": csrf_from(register.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        import_page = client.get("/import")
        imported = client.post(
            "/import",
            data={
                "title": "海边录音",
                "csrf": csrf_from(import_page.text),
            },
            files={
                "work_file": (
                    "sea.md",
                    "第一章 录音\n海浪盖住了最后一句话。".encode(),
                    "text/markdown",
                )
            },
            follow_redirects=False,
        )
        imported = commit_text_import(client, imported)
        assert imported.status_code == 303
        assert imported.headers["location"].startswith("/documents/")
        source_reader = client.get(imported.headers["location"])
        user = application.state.database.get_user_by_username(
            "阅读版本用户"
        )
        work = application.state.database.list_works(int(user["id"]))[0]
        created_main = client.post(
            f"/works/{work['id']}/main",
            data={
                "base_version_id": work["source_version"]["id"],
                "intent": "rewrite",
                "csrf": csrf_from(source_reader.text),
            },
            follow_redirects=False,
        )
        assert created_main.status_code == 303
        assert created_main.headers["location"].startswith("/novels/")
        work = application.state.database.get_work(
            int(user["id"]), str(work["id"])
        )
        assert work and work["main_version"]
        workbench = client.get(created_main.headers["location"])
        saved_settings = client.post(
            f"/novels/{work['main_version']['project_id']}/settings",
            data={
                "title": "海边录音",
                "premise": "录音中的失踪者留下了第一版线索。",
                "point_of_view": "第三人称限知",
                "target_chapter_chars": "3000",
                "planning_horizon": "20",
                "csrf": csrf_from(workbench.text),
            },
            follow_redirects=False,
        )
        assert saved_settings.status_code == 303
        workbench = client.get(created_main.headers["location"])
        tagged = client.post(
            f"/works/{work['id']}/tags",
            data={
                "label": "一稿",
                "csrf": csrf_from(workbench.text),
            },
            follow_redirects=False,
        )
        assert tagged.status_code == 303
        assert tagged.headers["location"].startswith("/documents/")
        tag_page = client.get(tagged.headers["location"])
        assert "一稿" in tag_page.text
        assert "只读" in tag_page.text

        works = application.state.database.list_works(int(user["id"]))
        assert len(works) == 1
        assert works[0]["main_version"]
        assert len(works[0]["tag_versions"]) == 2
        fixed = next(
            version
            for version in works[0]["tag_versions"]
            if version["label"] == "一稿"
        )
        assert fixed["base_version"]["ref_name"] == "main"
        assert fixed["creative_snapshot"]["project"]["title"] == "海边录音"
        assert (
            fixed["creative_snapshot"]["project"]["premise"]
            == "录音中的失踪者留下了第一版线索。"
        )

        main_page = client.get(created_main.headers["location"])
        changed_settings = client.post(
            f"/novels/{work['main_version']['project_id']}/settings",
            data={
                "title": "海边录音",
                "premise": "main 已经改成完全不同的第二版线索。",
                "point_of_view": "第三人称限知",
                "target_chapter_chars": "3000",
                "planning_horizon": "20",
                "csrf": csrf_from(main_page.text),
            },
            follow_redirects=False,
        )
        assert changed_settings.status_code == 303
        refreshed = application.state.database.get_work(
            int(user["id"]), str(work["id"])
        )
        assert refreshed
        unchanged_tag = next(
            version
            for version in refreshed["tag_versions"]
            if version["label"] == "一稿"
        )
        assert (
            unchanged_tag["creative_snapshot"]["project"]["premise"]
            == "录音中的失踪者留下了第一版线索。"
        )
        versions_page = client.get(
            f"/novels/{work['main_version']['project_id']}/workbench"
            "?view=archive&archive_tab=versions"
        )
        source_version_id = str(refreshed["source_version"]["id"])
        assert (
            f"/works/{work['id']}/versions/{source_version_id}/delete"
            not in versions_page.text
        )
        protected_source = client.post(
            f"/works/{work['id']}/versions/{source_version_id}/delete",
            data={"csrf": csrf_from(versions_page.text)},
            follow_redirects=False,
        )
        assert protected_source.status_code == 303
        assert "error=" in protected_source.headers["location"]
        assert application.state.database.get_work_version(
            int(user["id"]), source_version_id
        )


def test_dashboard_resumes_each_version_at_its_last_chapter(tmp_path):
    application = create_app(make_settings(tmp_path))
    with TestClient(application) as client:
        register = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "续读位置用户",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": csrf_from(register.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        import_page = client.get("/import")
        imported = client.post(
            "/import",
            data={
                "title": "三章续读测试",
                "csrf": csrf_from(import_page.text),
            },
            files={
                "work_file": (
                    "resume.txt",
                    (
                        "第一章 起点\n第一段正文。\n\n"
                        "第二章 中段\n第二段正文。\n\n"
                        "第三章 结尾\n第三段正文。"
                    ).encode(),
                    "text/plain",
                )
            },
            follow_redirects=False,
        )
        imported = commit_text_import(client, imported)
        assert imported.status_code == 303
        document_id = imported.headers["location"].split(
            "/documents/", 1
        )[1]
        user = application.state.database.get_user_by_username(
            "续读位置用户"
        )
        user_id = int(user["id"])
        work = application.state.database.list_works(user_id)[0]
        work_id = str(work["id"])
        source_version_id = str(work["source_version"]["id"])
        source_chapters = application.state.database.list_chapters(
            user_id, document_id
        )
        assert len(source_chapters) == 3
        source_last_chapter_id = str(source_chapters[2]["id"])
        source_last_url = (
            f"/documents/{document_id}"
            f"?chapter_id={source_last_chapter_id}"
        )
        source_last_page = client.get(source_last_url)
        assert source_last_page.status_code == 200
        assert "第三段正文。" in source_last_page.text

        remembered_source = application.state.database.get_work(
            user_id, work_id
        )
        assert (
            remembered_source["source_version"]["last_chapter_id"]
            == source_last_chapter_id
        )
        assert (
            remembered_source["source_version"]["open_url"]
            == source_last_url
        )
        direct_source = client.get(f"/documents/{document_id}")
        assert "第三段正文。" in direct_source.text
        dashboard = client.get("/dashboard")
        assert f'href="/works/{work_id}"' in dashboard.text
        resume_source = client.get(
            f"/works/{work_id}", follow_redirects=False
        )
        assert resume_source.headers["location"] == source_last_url

        created_main = client.post(
            f"/works/{work_id}/main",
            data={
                "base_version_id": source_version_id,
                "intent": "rewrite",
                "csrf": csrf_from(source_last_page.text),
            },
            follow_redirects=False,
        )
        assert created_main.status_code == 303
        project_id = project_id_from_workbench(
            created_main.headers["location"]
        )
        main_chapters = (
            application.state.database.list_novel_chapters(
                user_id, project_id
            )
        )
        assert len(main_chapters) == 3
        main_middle_chapter_id = str(main_chapters[1]["id"])
        main_middle_url = (
            f"/novels/{project_id}/workbench"
            f"?chapter_id={main_middle_chapter_id}"
        )
        main_middle_page = client.get(main_middle_url)
        assert main_middle_page.status_code == 200
        assert "第二段正文。" in main_middle_page.text

        remembered_main = application.state.database.get_work(
            user_id, work_id
        )
        main_version_id = str(remembered_main["main_version"]["id"])
        assert (
            remembered_main["main_version"]["last_chapter_id"]
            == main_middle_chapter_id
        )
        assert (
            remembered_main["main_version"]["open_url"]
            == main_middle_url
        )
        direct_main = client.get(f"/novels/{project_id}/workbench")
        assert "第二段正文。" in direct_main.text
        dashboard = client.get("/dashboard")
        assert f'href="/works/{work_id}"' in dashboard.text
        resume_main = client.get(
            f"/works/{work_id}", follow_redirects=False
        )
        assert resume_main.headers["location"] == main_middle_url

        switch_source = client.get(
            f"/works/{work_id}/versions/{source_version_id}",
            follow_redirects=False,
        )
        assert switch_source.headers["location"] == source_last_url
        resume_source_again = client.get(
            f"/works/{work_id}", follow_redirects=False
        )
        assert resume_source_again.headers["location"] == source_last_url

        switch_main = client.get(
            f"/works/{work_id}/versions/{main_version_id}",
            follow_redirects=False,
        )
        assert switch_main.headers["location"] == main_middle_url
        resume_main_again = client.get(
            f"/works/{work_id}", follow_redirects=False
        )
        assert resume_main_again.headers["location"] == main_middle_url


def test_tag_reuses_exact_main_analysis_and_can_be_deleted(tmp_path):
    application = create_app(make_settings(tmp_path))
    with TestClient(application) as client:
        register = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "版本分析作者",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": csrf_from(register.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        dashboard = client.get("/dashboard")
        created = client.post(
            "/novels/new/blank",
            data={"csrf": csrf_from(dashboard.text)},
            follow_redirects=False,
        )
        project_id = project_id_from_workbench(
            created.headers["location"]
        )
        workbench = client.get(created.headers["location"])
        chapter = client.post(
            f"/novels/{project_id}/chapters",
            data={
                "title": "精确快照",
                "return_to_workbench": "1",
                "csrf": csrf_from(workbench.text),
            },
            follow_redirects=False,
        )
        chapter_id = chapter_id_from_workbench(
            chapter.headers["location"]
        )
        chapter_page = client.get(chapter.headers["location"])
        content = "潮声停下以后，林岚在灯塔门口找到了失踪者的录音。"
        saved = client.post(
            f"/novels/{project_id}/chapters/{chapter_id}/save",
            data={
                "content": content,
                "change_summary": "完成第一稿",
                "return_to_workbench": "1",
                "csrf": csrf_from(chapter_page.text),
            },
            follow_redirects=False,
        )
        assert saved.status_code == 303
        user = application.state.database.get_user_by_username(
            "版本分析作者"
        )
        user_id = int(user["id"])
        deadline = time.monotonic() + 3
        memory = None
        while time.monotonic() < deadline:
            memory = MemoryService(
                application.state.database
            ).get_chapter_memory(
                user_id=user_id,
                project_id=project_id,
                chapter_id=chapter_id,
            )
            if memory:
                break
            time.sleep(0.03)
        assert memory

        main_page = client.get(saved.headers["location"])
        assert "查看分析" in main_page.text
        assert (
            f"data-actual-char-count>{len(content)} 字"
            in main_page.text
        )
        work = application.state.database.get_work_for_project(
            user_id, project_id
        )
        tagged = client.post(
            f"/works/{work['id']}/tags",
            data={
                "label": "分析快照",
                "csrf": csrf_from(main_page.text),
            },
            follow_redirects=False,
        )
        assert tagged.status_code == 303
        document_id = tagged.headers["location"].split(
            "/documents/", 1
        )[1].split("?", 1)[0]
        tag_page = client.get(f"/documents/{document_id}")
        source_match = re.search(
            r'<textarea[^>]+aria-label="章节原文"[^>]*>(.*?)</textarea>',
            tag_page.text,
            re.DOTALL,
        )
        assert source_match
        assert html.unescape(source_match.group(1)) == content
        assert f"<span>{len(content)} 字</span>" in tag_page.text
        assert "补充深度分析" in tag_page.text
        assert "未分析" not in tag_page.text

        refreshed = application.state.database.get_work(
            user_id, str(work["id"])
        )
        tag_version = next(
            item
            for item in refreshed["tag_versions"]
            if item["label"] == "分析快照"
        )
        snapshot_records = (
            application.state.database
            .list_work_version_story_memory_records(
                user_id, str(tag_version["id"])
            )
        )
        assert len(snapshot_records) == 1
        assert snapshot_records[0]["memory_status"] == "ready"
        assert snapshot_records[0]["summary"] == memory["summary"]

        analysis_page = client.get(
            f"/documents/{document_id}"
            "?view=archive&archive_tab=analysis"
        )
        assert "从相同正文的 main 分析中冻结" in analysis_page.text
        assert memory["summary"] in analysis_page.text
        assert (
            f'id="story-memory-{snapshot_records[0]["chapter_id"]}"'
            in analysis_page.text
        )

        exported = client.get(
            f"/works/{work['id']}/export.readraft.zip"
        )
        assert exported.status_code == 200
        import_page = client.get("/import")
        imported = client.post(
            "/import",
            data={"csrf": csrf_from(import_page.text)},
            files={
                "work_file": (
                    "analysis-snapshot.readraft.zip",
                    exported.content,
                    "application/zip",
                )
            },
            follow_redirects=False,
        )
        assert imported.status_code == 303
        imported_work = next(
            item
            for item in application.state.database.list_works(user_id)
            if str(item["id"]) != str(work["id"])
        )
        imported_tag = next(
            item
            for item in imported_work["tag_versions"]
            if item["label"] == "分析快照"
        )
        imported_memories = (
            application.state.database
            .list_work_version_story_memory_records(
                user_id, str(imported_tag["id"])
            )
        )
        assert imported_memories[0]["memory_status"] == "ready"
        assert imported_memories[0]["summary"] == memory["summary"]

        document_chapter = application.state.database.list_chapters(
            user_id, document_id
        )[0]
        legacy_path = Path(str(document_chapter["content_path"]))
        legacy_path.write_text(
            f"精确快照\n{content}", encoding="utf-8"
        )
        with application.state.database.connection() as connection:
            connection.execute(
                "DELETE FROM work_version_story_memories "
                "WHERE work_version_id=?",
                (str(tag_version["id"]),),
            )
            connection.execute(
                "DELETE FROM schema_migrations WHERE version=36"
            )
            connection.commit()
        application.state.database.initialize()
        backfilled = (
            application.state.database
            .list_work_version_story_memory_records(
                user_id, str(tag_version["id"])
            )
        )
        assert backfilled[0]["memory_status"] == "ready"
        assert backfilled[0]["summary"] == memory["summary"]
        assert legacy_path.read_text(encoding="utf-8") == content
        normalized_chapter = (
            application.state.database.list_chapters(
                user_id, document_id
            )[0]
        )
        assert normalized_chapter["char_count"] == len(content)

        versions_page = client.get(
            f"/documents/{document_id}"
            "?view=archive&archive_tab=versions"
        )
        protected_main = client.post(
            f"/works/{work['id']}/versions/"
            f"{work['main_version']['id']}/delete",
            data={"csrf": csrf_from(versions_page.text)},
            follow_redirects=False,
        )
        assert protected_main.status_code == 303
        assert "error=" in protected_main.headers["location"]
        assert application.state.database.get_work_version(
            user_id, str(work["main_version"]["id"])
        )
        delete_action = (
            f"/works/{work['id']}/versions/{tag_version['id']}/delete"
        )
        assert delete_action in versions_page.text
        document_dir = Path(
            str(
                application.state.database.get_document(
                    user_id, document_id
                )["source_path"]
            )
        ).parent
        deleted = client.post(
            delete_action,
            data={"csrf": csrf_from(versions_page.text)},
            follow_redirects=False,
        )
        assert deleted.status_code == 303
        assert f"/novels/{project_id}/workbench" in (
            deleted.headers["location"]
        )
        assert "archive_tab=versions" in deleted.headers["location"]
        assert "removed=true" in deleted.headers["location"]
        assert (
            application.state.database.get_work_version(
                user_id, str(tag_version["id"])
            )
            is None
        )
        assert application.state.database.get_novel_project(
            user_id, project_id
        )
        assert not document_dir.exists()




def test_reader_branches_and_chapter_history_restore_web_workflow(tmp_path):
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
        workbench_url = response.headers["location"]
        project_id = project_id_from_workbench(workbench_url)
        project_url = f"/novels/{project_id}"

        page = client.get(workbench_url)
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
        first_id = chapter_id_from_workbench(
            response.headers["location"]
        )
        first_url = f"{project_url}/chapters/{first_id}"
        page = client.get(workbench_url)
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
        second_id = chapter_id_from_workbench(
            response.headers["location"]
        )

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
        page = client.get(workbench_url)
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
        assert application.state.database.get_novel_chapter(
            user_id, project_id, second_id
        )["head_version_id"] == second_version
        replacement_id = candidate(first_id, "新")
        first_page = client.get(
            f"{project_url}/workbench?chapter_id={first_id}"
        )
        response = client.post(
            f"{first_url}/versions/{first_version}/restore",
            data={
                "csrf": csrf_from(first_page.text),
            },
            follow_redirects=False,
        )
        assert response.headers["location"].startswith(
            f"{project_url}/workbench?chapter_id={first_id}"
        )
        assert "restored=true" in response.headers["location"]
        restored_head = str(
            application.state.database.get_novel_chapter(
                user_id, project_id, first_id
            )["head_version_id"]
        )
        assert restored_head not in {first_version, replacement_id}
        restored_version = application.state.database.get_chapter_version(
            user_id, project_id, first_id, restored_head
        )
        assert restored_version["kind"] == "history_restore"
        assert Path(restored_version["content_path"]).read_text(
            encoding="utf-8"
        ) == "甲" * 2100
        assert application.state.database.get_novel_chapter(
            user_id, project_id, second_id
        )["needs_recheck"] == 1


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
        workbench_url = response.headers["location"]
        project_id = project_id_from_workbench(workbench_url)
        project_url = f"/novels/{project_id}"

        workspace = client.get(workbench_url)
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
        chapter_id = chapter_id_from_workbench(
            response.headers["location"]
        )
        chapter_url = f"{project_url}/chapters/{chapter_id}"

        first_page = client.get(
            f"{project_url}/workbench?chapter_id={chapter_id}"
        )
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

        second_page = client.get(
            f"{project_url}/workbench?chapter_id={chapter_id}"
        )
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
        versions_archive = client.get(
            f"{project_url}/workbench?view=archive&archive_tab=versions"
        )
        assert versions_archive.status_code == 200
        assert "main HEAD 与章节历史" in versions_archive.text
        assert "1 个历史版本" in versions_archive.text
        assert "创建 Tag" in versions_archive.text
        versions = application.state.database.list_chapter_versions(
            int(user["id"]), project_id, chapter_id
        )
        assert len(versions) == 2
        target_id = str(versions[0]["id"])
        base_id = str(versions[1]["id"])
        compare_url = f"{chapter_url}/versions/{target_id}/compare"

        chapter_page = client.get(
            f"{project_url}/workbench?chapter_id={chapter_id}"
        )
        assert 'class="studio-manuscript-view"' in chapter_page.text
        assert f'href="{compare_url}"' not in chapter_page.text

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
