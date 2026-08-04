from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from app.config import Settings
from app.db import Database
from app.doctor import inspect_integrity
from app.security import hash_password
from app.work_library import create_version_tag


def make_settings(root: Path) -> Settings:
    return Settings(
        app_name="Readraft Doctor Test",
        app_env="test",
        secret_key="doctor-test-secret-key",  # pragma: allowlist secret
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


def create_chapter(settings: Settings) -> tuple[Database, int, str, str, Path, Path]:
    settings.ensure_directories()
    database = Database(settings.database_path)
    database.initialize()
    user_id = database.create_user("doctor", hash_password("password-123"))
    project_id = "doctor-project"
    chapter_id = "doctor-chapter"
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="完整性测试",
        genre="悬疑",
        premise="检查版本链",
        world_setting="",
        style_guide="",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
    )
    chapter_dir = (
        settings.novels_dir
        / str(user_id)
        / project_id
        / "chapters"
        / chapter_id
    )
    versions_dir = chapter_dir / "versions"
    versions_dir.mkdir(parents=True)
    cache_path = chapter_dir / "content.txt"
    version_path = versions_dir / "manual.txt"
    content = "雨停以后，她在台阶下找到第二封信。"
    cache_path.write_text(content, encoding="utf-8")
    version_path.write_text(content, encoding="utf-8")
    database.add_novel_chapter(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        title="第一章",
        outline="找到第二封信",
        key_points="雨、台阶、信",
        content_path=cache_path,
    )
    version_id = database.record_manual_chapter_version(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        version_path=version_path,
        char_count=len(content),
        effective_char_count=len(content),
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        expected_old_head_version_id="",
    )
    assert version_id
    return database, user_id, project_id, chapter_id, cache_path, version_path


def test_doctor_accepts_consistent_repository(tmp_path: Path):
    settings = make_settings(tmp_path)
    create_chapter(settings)

    report = inspect_integrity(settings)

    assert report.ok
    assert not [item for item in report.issues if item.severity == "error"]


def test_doctor_repairs_only_non_authoritative_cache(tmp_path: Path):
    settings = make_settings(tmp_path)
    _, _, _, _, cache_path, version_path = create_chapter(settings)
    cache_path.write_text("浏览器缓存中的旧稿", encoding="utf-8")

    broken = inspect_integrity(settings)
    assert not broken.ok
    assert any(item.code == "chapter_cache_stale" for item in broken.issues)

    repaired = inspect_integrity(settings, repair=True)
    assert repaired.ok
    assert repaired.repaired_files == 1
    assert cache_path.read_text(encoding="utf-8") == version_path.read_text(
        encoding="utf-8"
    )


def test_doctor_detects_immutable_version_corruption(tmp_path: Path):
    settings = make_settings(tmp_path)
    _, _, _, _, _, version_path = create_chapter(settings)
    version_path.write_text("被篡改的历史", encoding="utf-8")

    report = inspect_integrity(settings, repair=True)

    assert not report.ok
    assert any(item.code == "hash_mismatch" for item in report.issues)
    assert version_path.read_text(encoding="utf-8") == "被篡改的历史"


def test_doctor_prunes_only_explicit_orphans(tmp_path: Path):
    settings = make_settings(tmp_path)
    _, _, _, _, _, version_path = create_chapter(settings)
    orphan = version_path.parent / "abandoned.txt"
    orphan.write_text("未提交", encoding="utf-8")

    inspected = inspect_integrity(settings)
    assert orphan.exists()
    assert any(item.code == "orphan_file" for item in inspected.issues)

    pruned = inspect_integrity(settings, prune_orphans=True)
    assert pruned.ok
    assert pruned.pruned_files == 1
    assert not orphan.exists()


def test_database_triggers_reject_cross_chapter_version_links(tmp_path: Path):
    settings = make_settings(tmp_path)
    database, user_id, project_id, _, _, first_path = create_chapter(settings)
    first = database.list_novel_chapters(user_id, project_id)[0]
    second_id = "second-chapter"
    second_dir = first_path.parents[2] / second_id
    second_dir.mkdir()
    second_cache = second_dir / "content.txt"
    second_cache.write_text("", encoding="utf-8")
    database.add_novel_chapter(
        user_id=user_id,
        project_id=project_id,
        chapter_id=second_id,
        title="第二章",
        outline="",
        key_points="",
        content_path=second_cache,
    )

    with database.connection() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="HEAD belongs"):
            connection.execute(
                "UPDATE novel_chapters SET head_version_id=? WHERE id=?",
                (first["head_version_id"], second_id),
            )


def test_migration_backfills_legacy_tag_manifest(tmp_path: Path):
    settings = make_settings(tmp_path)
    database, user_id, project_id, chapter_id, _, version_path = create_chapter(
        settings
    )
    document_id = create_version_tag(
        database=database,
        documents_dir=settings.documents_dir,
        user_id=user_id,
        project_id=project_id,
        label="旧版一稿",
    )
    work = database.get_work_for_document(user_id, document_id)
    assert work and work["current_version"]
    tag_id = str(work["current_version"]["id"])

    with database.connection() as connection:
        connection.execute(
            "DELETE FROM work_tag_chapter_heads WHERE work_version_id=?",
            (tag_id,),
        )
        connection.execute("DELETE FROM schema_migrations WHERE version=55")
        connection.commit()
    assert not inspect_integrity(settings).ok

    database.initialize()

    with database.connection() as connection:
        manifest = connection.execute(
            """
            SELECT source_chapter_id, source_version_id, content_hash
            FROM work_tag_chapter_heads
            WHERE work_version_id=?
            """,
            (tag_id,),
        ).fetchone()
    assert manifest
    assert manifest["source_chapter_id"] == chapter_id
    assert manifest["source_version_id"]
    assert manifest["content_hash"] == hashlib.sha256(
        version_path.read_bytes()
    ).hexdigest()
    assert inspect_integrity(settings).ok
