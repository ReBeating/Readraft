from __future__ import annotations

import sqlite3
import zipfile
from pathlib import Path

import pytest

from app.backup import (
    BackupError,
    create_backup,
    restore_backup,
    verify_backup,
)
from app.config import Settings
from app.db import Database
from app.security import hash_password


def make_settings(root: Path) -> Settings:
    return Settings(
        app_name="Readraft 备份测试",
        app_env="test",
        secret_key="same-test-secret-for-backup",
        data_dir=root / "data",
        database_path=root / "data" / "readraft.db",
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
    )


def create_novel(
    settings: Settings,
    *,
    username: str,
    project_id: str,
    content: str,
) -> tuple[int, str, str]:
    settings.ensure_directories()
    database = Database(settings.database_path)
    database.initialize()
    user_id = database.create_user(
        username, hash_password("password-123")
    )
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="备份中的长篇",
        genre="悬疑",
        premise="一封晚到十年的信改变了返乡计划。",
        world_setting="当代沿海城市",
        style_guide="克制、具体",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
    )
    chapter_id = f"{project_id}-chapter"
    chapter_dir = settings.novels_dir / project_id / "chapters" / chapter_id
    version_dir = chapter_dir / "versions"
    version_dir.mkdir(parents=True)
    current_path = chapter_dir / "current.txt"
    version_path = version_dir / "manual.txt"
    current_path.write_text(content, encoding="utf-8")
    version_path.write_text(content, encoding="utf-8")
    database.add_novel_chapter(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        title="第一章 迟到的信",
        outline="收到信并决定返乡",
        key_points="邮戳、旧信、返程票",
        content_path=current_path,
    )
    version_id = database.record_manual_chapter_version(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        version_path=version_path,
        char_count=len(content),
        content_hash="test-hash",
        change_summary="备份测试版本",
    )
    assert version_id
    return user_id, project_id, chapter_id


def test_full_backup_is_portable_and_verifiable(tmp_path: Path):
    source_settings = make_settings(tmp_path / "source")
    user_id, project_id, chapter_id = create_novel(
        source_settings,
        username="backup-author",
        project_id="portable-project",
        content=(
            "她把两封旧信并排放在桌上，先核对邮戳，"
            "再订下返程票。"
        ),
    )
    archive = tmp_path / "exports" / "readraft-backup.zip"

    created = create_backup(source_settings, archive)
    checked = verify_backup(archive)

    assert created.file_count >= 3
    assert checked.file_count == created.file_count
    assert checked.total_bytes == created.total_bytes
    with zipfile.ZipFile(archive) as payload:
        names = set(payload.namelist())
        assert "manifest.json" in names
        assert "database.sqlite3" in names
        assert any(
            name.endswith("/current.txt") for name in names
        )
        extracted_database = tmp_path / "portable.sqlite3"
        extracted_database.write_bytes(payload.read("database.sqlite3"))
    with sqlite3.connect(extracted_database) as connection:
        stored_path = connection.execute(
            "SELECT content_path FROM novel_chapters WHERE id=?",
            (chapter_id,),
        ).fetchone()[0]
    assert stored_path.startswith("__READRAFT_DATA__/")
    assert str(source_settings.data_dir) not in stored_path

    target_settings = make_settings(tmp_path / "target")
    old_user_id, old_project_id, _old_chapter_id = create_novel(
        target_settings,
        username="old-author",
        project_id="old-project",
        content="这份数据应只出现在恢复前安全备份里。",
    )
    restored = restore_backup(
        target_settings, archive, replace=True
    )

    assert restored.safety_backup
    assert restored.safety_backup.is_file()
    assert verify_backup(restored.safety_backup).file_count >= 3
    database = Database(target_settings.database_path)
    project = database.get_novel_project(user_id, project_id)
    assert project and project["title"] == "备份中的长篇"
    assert database.get_novel_project(old_user_id, old_project_id) is None
    chapter = database.get_novel_chapter(
        user_id, project_id, chapter_id
    )
    assert chapter
    restored_path = Path(str(chapter["content_path"]))
    assert restored_path.is_relative_to(target_settings.data_dir)
    assert restored_path.read_text(encoding="utf-8").startswith(
        "她把两封旧信"
    )
    assert not (
        target_settings.novels_dir
        / old_project_id
        / "chapters"
    ).exists()


def test_backup_rejects_unsafe_destination_and_tampering(tmp_path: Path):
    settings = make_settings(tmp_path / "source")
    create_novel(
        settings,
        username="safe-author",
        project_id="safe-project",
        content="可靠的备份必须能发现归档被改动。",
    )
    with pytest.raises(BackupError, match="数据目录之外"):
        create_backup(settings, settings.data_dir / "inside.zip")

    archive = tmp_path / "safe.zip"
    create_backup(settings, archive)
    with zipfile.ZipFile(archive, "a") as payload:
        payload.writestr("unexpected.txt", b"tampered")
    with pytest.raises(BackupError, match="清单不一致"):
        verify_backup(archive)


def test_restore_requires_explicit_replace(tmp_path: Path):
    settings = make_settings(tmp_path / "source")
    create_novel(
        settings,
        username="restore-author",
        project_id="restore-project",
        content="恢复必须显式确认。",
    )
    archive = tmp_path / "restore.zip"
    create_backup(settings, archive)

    with pytest.raises(BackupError, match="replace=True"):
        restore_backup(settings, archive)
