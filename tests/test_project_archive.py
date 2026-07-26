from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from app.config import Settings
from app.db import Database, utc_now
from app.project_archive import (
    ProjectArchiveError,
    create_project_archive,
    import_project_archive,
    verify_project_archive,
)
from app.security import hash_password


def make_settings(root: Path) -> Settings:
    return Settings(
        app_name="novelAI 作品归档测试",
        app_env="test",
        secret_key="project-archive-test-secret",
        data_dir=root / "data",
        database_path=root / "data" / "novelai.db",
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
    )


def create_project(settings: Settings) -> tuple[Database, int, str, str]:
    settings.ensure_directories()
    database = Database(settings.database_path)
    database.initialize()
    user_id = database.create_user(
        "archive-author", hash_password("password-123")
    )
    project_id = "portable-project"
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="纸灯塔",
        genre="悬疑",
        premise="修表师从一张旧船票中发现失约的真相。",
        world_setting="当代海边小城",
        style_guide="克制、具体",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
        story_promise="每个时间证据都能被核对",
    )
    database.add_novel_character(
        user_id=user_id,
        project_id=project_id,
        name="林岚",
        role="修表师",
        traits="谨慎、敏锐",
        background="经营一家老钟表店",
        character_arc="从回避旧约到主动核对真相",
    )
    chapter_id = "portable-chapter"
    chapter_root = (
        settings.novels_dir
        / str(user_id)
        / project_id
        / "chapters"
        / chapter_id
    )
    versions_root = chapter_root / "versions"
    versions_root.mkdir(parents=True)
    content = "林岚把船票压在灯下，发现日期比记忆早了一天。"
    content_path = chapter_root / "content.txt"
    version_path = versions_root / "manual.txt"
    content_path.write_text(content, encoding="utf-8")
    version_path.write_text(content, encoding="utf-8")
    database.add_novel_chapter(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        title="第一章 停走的秒针",
        outline="核对船票并前往旧码头",
        key_points="船票日期、修理簿、旧码头",
        content_path=content_path,
    )
    version_id = database.record_manual_chapter_version(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        version_path=version_path,
        char_count=len(content),
        content_hash="portable-content-hash",
        change_summary="归档测试版本",
    )
    assert version_id
    now = utc_now()
    with database.connection() as connection:
        connection.execute(
            """
            INSERT INTO reference_technique_cards(
                id, user_id, name, dimension, source_location,
                observation, effect, execution_rule,
                originality_boundary, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "portable-technique",
                user_id,
                "线索先改变行动",
                "information",
                "作者手工创建",
                "信息先产生后果，再解释来源。",
                "维持可核对的问题。",
                "先让线索改变目标，再延后解释。",
                "不复用参考文本的人名、物件或措辞。",
                now,
                now,
            ),
        )
        connection.execute(
            """
            INSERT INTO novel_technique_bindings(
                id, technique_id, project_id, scope_type,
                usage_modes_json, author_adaptation, priority,
                status, created_at, updated_at
            ) VALUES (?, ?, ?, 'project', '["write"]', ?, 70,
                      'enabled', ?, ?)
            """,
            (
                "portable-binding",
                "portable-technique",
                project_id,
                "改成船票先推动行动。",
                now,
                now,
            ),
        )
        connection.commit()
    database.upsert_api_credential(
        user_id=user_id,
        provider="deepseek",
        encrypted_key="CREDENTIAL-MUST-NOT-BE-EXPORTED",
        key_hint="sk-••••test",
        model="deepseek-v4-flash",
    )
    return database, user_id, project_id, chapter_id


def test_project_archive_round_trip_preserves_project_state(tmp_path: Path):
    settings = make_settings(tmp_path)
    database, user_id, project_id, chapter_id = create_project(settings)
    archive = tmp_path / "paper-lighthouse.novelai.zip"

    summary = create_project_archive(
        database=database,
        novels_dir=settings.novels_dir,
        user_id=user_id,
        project_id=project_id,
        destination=archive,
        max_uncompressed_bytes=20 * 1024 * 1024,
    )
    manifest = verify_project_archive(
        archive, max_uncompressed_bytes=20 * 1024 * 1024
    )

    assert summary.file_count == 2
    assert summary.row_count >= 5
    assert "users" not in manifest["tables"]
    assert "api_credentials" not in manifest["tables"]
    assert "api_models" not in manifest["tables"]
    assert b"CREDENTIAL-MUST-NOT-BE-EXPORTED" not in archive.read_bytes()
    assert (
        manifest["tables"]["novel_chapters"][0]["content_path"]
        == "project://chapters/portable-chapter/content.txt"
    )

    imported = import_project_archive(
        database=database,
        novels_dir=settings.novels_dir,
        user_id=user_id,
        archive_path=archive,
        max_uncompressed_bytes=20 * 1024 * 1024,
    )

    assert imported.project_id != project_id
    project = database.get_novel_project(user_id, imported.project_id)
    assert project and project["title"] == "纸灯塔"
    characters = database.list_novel_characters(
        user_id, imported.project_id
    )
    assert [character["name"] for character in characters] == ["林岚"]
    chapters = database.list_novel_chapters(user_id, imported.project_id)
    assert len(chapters) == 1
    assert chapters[0]["id"] != chapter_id
    imported_path = Path(str(chapters[0]["content_path"]))
    assert imported_path.is_relative_to(
        settings.novels_dir / str(user_id) / imported.project_id
    )
    assert imported_path.read_text(encoding="utf-8").startswith("林岚把船票")
    with database.connection() as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        imported_binding = connection.execute(
            """
            SELECT b.project_id, c.user_id, c.name
            FROM novel_technique_bindings b
            JOIN reference_technique_cards c ON c.id=b.technique_id
            WHERE b.project_id=?
            """,
            (imported.project_id,),
        ).fetchone()
        assert dict(imported_binding) == {
            "project_id": imported.project_id,
            "user_id": user_id,
            "name": "线索先改变行动",
        }
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM api_credentials WHERE user_id=?",
                (user_id,),
            ).fetchone()[0]
            == 1
        )


def test_project_archive_rejects_path_traversal(tmp_path: Path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as payload:
        payload.writestr("../escape.txt", "unsafe")
        payload.writestr(
            "manifest.json",
            json.dumps(
                {
                    "format": "novelai-project",
                    "version": 1,
                    "project": {"id": "p", "title": "unsafe"},
                    "tables": {
                        "novel_projects": [
                            {"id": "p", "user_id": 1}
                        ]
                    },
                    "files": [],
                }
            ),
        )

    with pytest.raises(ProjectArchiveError, match="不安全"):
        verify_project_archive(
            archive, max_uncompressed_bytes=1024 * 1024
        )


def test_project_archive_detects_file_tampering(tmp_path: Path):
    settings = make_settings(tmp_path / "source")
    database, user_id, project_id, _chapter_id = create_project(settings)
    archive = tmp_path / "original.zip"
    create_project_archive(
        database=database,
        novels_dir=settings.novels_dir,
        user_id=user_id,
        project_id=project_id,
        destination=archive,
        max_uncompressed_bytes=20 * 1024 * 1024,
    )
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(archive, "r") as source:
        entries = {
            info.filename: source.read(info.filename)
            for info in source.infolist()
        }
    file_name = next(name for name in entries if name.startswith("files/"))
    entries[file_name] += b"tampered"
    with zipfile.ZipFile(tampered, "w") as target:
        for name, value in entries.items():
            target.writestr(name, value)

    with pytest.raises(ProjectArchiveError, match="大小与清单不一致|校验失败"):
        verify_project_archive(
            tampered, max_uncompressed_bytes=20 * 1024 * 1024
        )


def test_project_archive_rejects_unowned_project(tmp_path: Path):
    settings = make_settings(tmp_path)
    database, _user_id, project_id, _chapter_id = create_project(settings)
    other_user = database.create_user(
        "other-author", hash_password("password-456")
    )

    with pytest.raises(ProjectArchiveError, match="不属于"):
        create_project_archive(
            database=database,
            novels_dir=settings.novels_dir,
            user_id=other_user,
            project_id=project_id,
            destination=tmp_path / "forbidden.zip",
            max_uncompressed_bytes=20 * 1024 * 1024,
        )


def test_project_archive_waits_for_active_ai_task(tmp_path: Path):
    settings = make_settings(tmp_path)
    database, user_id, project_id, chapter_id = create_project(settings)
    database.create_generation_job(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        operation="draft",
        instruction="仅用于验证归档并发保护",
        provider="deepseek",
        model="deepseek-v4-flash",
        credential_source="personal",
    )

    with pytest.raises(ProjectArchiveError, match="AI 任务正在"):
        create_project_archive(
            database=database,
            novels_dir=settings.novels_dir,
            user_id=user_id,
            project_id=project_id,
            destination=tmp_path / "busy-project.zip",
            max_uncompressed_bytes=20 * 1024 * 1024,
        )
