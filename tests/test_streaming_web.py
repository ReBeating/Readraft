import asyncio
import re
import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        app_name="test",
        app_env="test",
        secret_key="test-secret-long-enough",  # pragma: allowlist secret
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


def register(client: TestClient, username: str) -> None:
    page = client.get("/register")
    response = client.post(
        "/register",
        data={
            "username": username,
            "password": "password-123",  # pragma: allowlist secret
            "password_confirm": "password-123",  # pragma: allowlist secret
            "csrf": csrf_from(page.text),
        },
        follow_redirects=False,
    )
    assert response.status_code == 303


def test_exa_web_search_needs_no_key_and_can_be_disabled(tmp_path):
    application = create_app(make_settings(tmp_path))
    with TestClient(application) as client:
        register(client, "联网设置作者")
        settings_page = client.get("/settings/api?tab=search")
        assert settings_page.status_code == 200
        assert "使用 Exa；无需 API Key" in settings_page.text
        assert 'name="api_key"' not in settings_page.text

        response = client.post(
            "/settings/web-search",
            data={
                "csrf": csrf_from(settings_page.text),
                "enabled": "true",
                "provider": "deepseek",
                "return_to": "/dashboard",
                "embedded": "false",
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        saved_page = client.get(response.headers["location"])
        assert "联网搜索设置已保存" in saved_page.text

        user = application.state.database.get_user_by_username(
            "联网设置作者"
        )
        stored = application.state.database.get_web_search_settings(
            int(user["id"])
        )
        assert stored["enabled"] == 1
        assert "encrypted_key" not in stored
        assert "key_hint" not in stored

        response = client.post(
            "/settings/web-search",
            data={
                "csrf": csrf_from(saved_page.text),
                "provider": "deepseek",
                "return_to": "/dashboard",
                "embedded": "false",
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        disabled = application.state.database.get_web_search_settings(
            int(user["id"])
        )
        assert disabled["enabled"] == 0


def test_chat_submit_returns_json_and_sse_terminal_snapshot(tmp_path):
    application = create_app(make_settings(tmp_path))
    with TestClient(application) as client:
        register(client, "流式对话作者")
        dashboard = client.get("/dashboard")
        created = client.post(
            "/novels/new/blank",
            data={"csrf": csrf_from(dashboard.text)},
            follow_redirects=False,
        )
        assert created.status_code == 303
        workbench = client.get(created.headers["location"])
        assert workbench.status_code == 200
        assert workbench.text.index("novel-workbench-theme") < (
            workbench.text.index("app.css")
        )
        assert "?v=20260728-2" in workbench.text

        response = client.post(
            workbench.url.path.replace("/workbench", "")
            + "/assistant/messages",
            headers={"Accept": "application/json"},
            data={
                "csrf": csrf_from(workbench.text),
                "question": "先讨论这个故事的核心冲突。",
                "quality_mode": "standard",
                "return_view": "archive",
                "return_archive_tab": "creative",
                "return_settings_tab": "core",
            },
        )
        assert response.status_code == 202
        queued = response.json()
        assert queued["message_id"]
        assert queued["conversation_id"]
        assert queued["stream_url"].endswith(
            f"/{queued['message_id']}/stream"
        )

        deadline = time.monotonic() + 5
        state = {}
        while time.monotonic() < deadline:
            status = client.get(
                f"/api/assistant/messages/{queued['message_id']}"
            )
            assert status.status_code == 200
            state = status.json()
            if state["terminal"]:
                break
            time.sleep(0.02)
        assert state["terminal"] is True
        assert state["status"] == "completed"
        assert state["content"]
        assert state["sequence"] >= 1

        with client.stream("GET", queued["stream_url"]) as stream:
            assert stream.status_code == 200
            assert stream.headers["content-type"].startswith(
                "text/event-stream"
            )
            body = "".join(stream.iter_text())
        assert "event: snapshot" in body
        assert "event: agent" in body
        assert '"phase": "completed"' in body
        assert '"status": "completed"' in body
        assert "event: done" in body
        assert '"redirect_url": "/novels/' in body
        assert f"conversation_id={queued['conversation_id']}" in body


def test_running_chat_can_be_stopped_and_streams_cancelled_state(tmp_path):
    application = create_app(make_settings(tmp_path))
    with TestClient(application) as client:
        register(client, "停止生成作者")
        dashboard = client.get("/dashboard")
        created = client.post(
            "/novels/new/blank",
            data={"csrf": csrf_from(dashboard.text)},
            follow_redirects=False,
        )
        workbench = client.get(created.headers["location"])
        started = threading.Event()

        async def blocking_run(**_kwargs):
            started.set()
            await asyncio.sleep(60)

        application.state.worker.assistant_agent_orchestrator.run = blocking_run
        response = client.post(
            workbench.url.path.replace("/workbench", "")
            + "/assistant/messages",
            headers={"Accept": "application/json"},
            data={
                "csrf": csrf_from(workbench.text),
                "question": "请先分析人物关系。",
                "quality_mode": "standard",
            },
        )
        assert response.status_code == 202
        queued = response.json()
        assert started.wait(timeout=2)

        cancelled = client.post(
            f"/api/assistant/messages/{queued['message_id']}/cancel",
            headers={"Accept": "application/json"},
            data={"csrf": csrf_from(workbench.text)},
        )
        assert cancelled.status_code == 202
        assert cancelled.json()["interrupted"] is True

        deadline = time.monotonic() + 3
        state = {}
        while time.monotonic() < deadline:
            state = client.get(
                f"/api/assistant/messages/{queued['message_id']}"
            ).json()
            if state.get("terminal"):
                break
            time.sleep(0.02)
        assert state["terminal"] is True
        assert state["status"] == "failed"
        assert state["run_state"] == "cancelled"
        assert state["cancelled"] is True

        with client.stream("GET", queued["stream_url"]) as stream:
            body = "".join(stream.iter_text())
        assert "event: agent" in body
        assert '"phase": "cancelled"' in body
        assert '"cancelled": true' in body
