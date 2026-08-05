import re
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        app_name="统一作品工作台测试",
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
        model_name="deepseek-chat",
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


def register(client: TestClient) -> None:
    page = client.get("/register")
    response = client.post(
        "/register",
        data={
            "username": "统一工作台作者",
            "password": "password-123",
            "password_confirm": "password-123",
            "csrf": csrf_from(page.text),
        },
        follow_redirects=False,
    )
    assert response.status_code == 303


def create_project_and_chapter(
    client: TestClient,
) -> tuple[str, str, str]:
    page = client.get("/novels/new")
    response = client.post(
        "/novels/new",
        data={
            "title": "雾港来信",
            "genre": "悬疑",
            "premise": "林岚收到一封日期异常的来信。",
            "csrf": csrf_from(page.text),
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    workbench_url = response.headers["location"]
    project_id = workbench_url.split("/novels/", 1)[1].split("/", 1)[0]

    workbench = client.get(workbench_url)
    response = client.post(
        f"/novels/{project_id}/chapters",
        data={
            "title": "不存在的预报",
            "csrf": csrf_from(workbench.text),
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    chapter_url = response.headers["location"]
    chapter_id = chapter_url.split("chapter_id=", 1)[1].split("&", 1)[0]
    return project_id, chapter_id, chapter_url


def test_empty_project_opens_unified_workbench(tmp_path: Path):
    application = create_app(make_settings(tmp_path))
    with TestClient(application) as client:
        register(client)
        page = client.get("/novels/new")
        response = client.post(
            "/novels/new",
            data={
                "title": "还没有第一章",
                "genre": "悬疑",
                "premise": "林岚收到一封日期异常、来源不明的来信。",
                "csrf": csrf_from(page.text),
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


def test_legacy_chapter_surfaces_can_no_longer_render(tmp_path: Path):
    application = create_app(make_settings(tmp_path))
    with TestClient(application) as client:
        register(client)
        project_id, chapter_id, workbench_url = create_project_and_chapter(
            client
        )

        workspace = client.get(workbench_url)
        assert workspace.status_code == 200
        assert 'class="studio-manuscript-view"' in workspace.text
        assert "data-chapter-workflow" not in workspace.text
        assert "章节创作流程" not in workspace.text
        assert "任务卡" not in workspace.text
        assert "场景工作台" not in workspace.text
        assert (
            f"/novels/{project_id}/chapters/{chapter_id}/task-card"
            not in workspace.text
        )
        assert (
            f"/novels/{project_id}/chapters/{chapter_id}/scenes"
            not in workspace.text
        )

        legacy_urls = (
            f"/novels/{project_id}/chapters/{chapter_id}",
            f"/novels/{project_id}/chapters/{chapter_id}/task-card",
            f"/novels/{project_id}/chapters/{chapter_id}/scenes",
        )
        for legacy_url in legacy_urls:
            response = client.get(legacy_url, follow_redirects=False)
            assert response.status_code == 404
