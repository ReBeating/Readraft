from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from app.config import Settings
from app.db import Database, utc_now
from app.security import hash_password
from app.work_archive import (
    WORK_ARCHIVE_FORMAT,
    WorkArchiveError,
    create_work_archive,
    detect_archive_format,
    import_work_archive,
    verify_work_archive,
)
from app.work_library import (
    create_reading_snapshot,
    create_writing_branch_from_document,
)


def make_settings(root: Path) -> Settings:
    return Settings(
        app_name="novelAI 作品归档测试",
        app_env="test",
        secret_key="work-archive-test-secret",  # pragma: allowlist secret
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


def test_complete_work_archive_round_trip_preserves_editions_and_archive(
    tmp_path: Path,
):
    settings = make_settings(tmp_path)
    database, user_id, project_id, _chapter_id = create_project(settings)
    work_id, _edition_id = database.ensure_project_work(
        user_id=user_id,
        project_id=project_id,
    )
    note_id = database.add_work_archive_entry(
        user_id=user_id,
        work_id=work_id,
        entry_type="analysis_note",
        title="钟表意象",
        content="停走的秒针总在人物回避旧约时出现。",
        evidence="第一章",
        category="structure",
    )
    setting_id = database.adopt_work_archive_entry(
        user_id=user_id,
        work_id=work_id,
        entry_id=note_id,
        category="structure",
        title="钟表意象规则",
        content="人物回避旧约时，让停走的秒针推动下一步行动。",
    )
    document_id = create_reading_snapshot(
        database=database,
        documents_dir=settings.documents_dir,
        user_id=user_id,
        project_id=project_id,
    )
    rewrite_project_id = create_writing_branch_from_document(
        database=database,
        novels_dir=settings.novels_dir,
        user_id=user_id,
        document_id=document_id,
        intent="rewrite",
    )
    archive = tmp_path / "complete-work.novelai.zip"

    summary = create_work_archive(
        database=database,
        novels_dir=settings.novels_dir,
        documents_dir=settings.documents_dir,
        user_id=user_id,
        work_id=work_id,
        destination=archive,
        max_uncompressed_bytes=20 * 1024 * 1024,
    )
    manifest = verify_work_archive(
        archive,
        max_uncompressed_bytes=20 * 1024 * 1024,
    )

    assert detect_archive_format(archive) == WORK_ARCHIVE_FORMAT
    assert summary.edition_count == 3
    assert manifest["counts"]["editions"] == 3
    assert len(manifest["roots"]["projects"]) == 2
    assert len(manifest["roots"]["documents"]) == 1
    assert "works" in manifest["tables"]
    assert "work_editions" in manifest["tables"]
    assert "work_archive_entries" in manifest["tables"]
    assert "documents" in manifest["tables"]
    assert "api_credentials" not in manifest["tables"]
    assert b"CREDENTIAL-MUST-NOT-BE-EXPORTED" not in archive.read_bytes()

    imported = import_work_archive(
        database=database,
        novels_dir=settings.novels_dir,
        documents_dir=settings.documents_dir,
        user_id=user_id,
        archive_path=archive,
        max_uncompressed_bytes=20 * 1024 * 1024,
        max_documents=50,
        max_stored_chars=20_000_000,
    )

    assert imported.work_id != work_id
    assert imported.edition_count == 3
    imported_work = database.get_work(user_id, imported.work_id)
    assert imported_work and imported_work["title"] == "纸灯塔"
    assert len(imported_work["writing_editions"]) == 2
    assert len(imported_work["reading_editions"]) == 1
    imported_rewrite = next(
        edition
        for edition in imported_work["writing_editions"]
        if edition["branch_intent"] == "rewrite"
    )
    assert imported_rewrite["project_id"] != rewrite_project_id
    assert imported_rewrite["source_edition"]["kind"] == "snapshot"
    imported_snapshot = imported_work["reading_editions"][0]
    assert imported_snapshot["source_edition"]["project_id"] != project_id

    entries = database.list_work_archive_entries(
        user_id,
        imported.work_id,
    )
    imported_note = next(
        entry for entry in entries if entry["entry_type"] == "analysis_note"
    )
    imported_setting = next(
        entry for entry in entries if entry["entry_type"] == "creative_rule"
    )
    assert imported_note["id"] != note_id
    assert imported_setting["id"] != setting_id
    assert imported_note["status"] == "adopted"
    assert imported_note["adopted_setting_id"] == imported_setting["id"]
    assert imported_setting["adopted_from_entry_id"] == imported_note["id"]
    assert imported_setting["status"] == "confirmed"

    imported_chapters = database.list_novel_chapters(
        user_id,
        str(imported_rewrite["project_id"]),
    )
    assert len(imported_chapters) == 1
    imported_content_path = Path(
        str(imported_chapters[0]["content_path"])
    )
    assert imported_content_path.is_relative_to(
        settings.novels_dir
        / str(user_id)
        / str(imported_rewrite["project_id"])
    )
    assert "林岚把船票" in imported_content_path.read_text(
        encoding="utf-8"
    )
    imported_document = database.get_document(
        user_id,
        str(imported_snapshot["document_id"]),
    )
    assert imported_document
    assert Path(str(imported_document["source_path"])).is_relative_to(
        settings.documents_dir
        / str(user_id)
        / str(imported_snapshot["document_id"])
    )
    with database.connection() as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM api_credentials WHERE user_id=?",
                (user_id,),
            ).fetchone()[0]
            == 1
        )


def test_work_archive_rejects_path_traversal(tmp_path: Path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as payload:
        payload.writestr("../escape.txt", "unsafe")
        payload.writestr(
            "manifest.json",
            json.dumps(
                {
                    "format": WORK_ARCHIVE_FORMAT,
                    "version": 1,
                    "work": {"id": "w", "title": "unsafe"},
                    "roots": {"projects": [], "documents": []},
                    "tables": {
                        "works": [{"id": "w", "user_id": 1}],
                        "work_editions": [
                            {
                                "id": "e",
                                "work_id": "w",
                                "project_id": "p",
                                "document_id": None,
                            }
                        ],
                    },
                    "files": [],
                }
            ),
        )

    with pytest.raises(WorkArchiveError, match="不安全"):
        verify_work_archive(
            archive,
            max_uncompressed_bytes=1024 * 1024,
        )


def test_work_archive_detects_file_tampering(tmp_path: Path):
    settings = make_settings(tmp_path / "source")
    database, user_id, project_id, _chapter_id = create_project(settings)
    work_id, _edition_id = database.ensure_project_work(
        user_id=user_id,
        project_id=project_id,
    )
    archive = tmp_path / "original.zip"
    create_work_archive(
        database=database,
        novels_dir=settings.novels_dir,
        documents_dir=settings.documents_dir,
        user_id=user_id,
        work_id=work_id,
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

    with pytest.raises(
        WorkArchiveError,
        match="大小与清单不一致|校验失败",
    ):
        verify_work_archive(
            tampered,
            max_uncompressed_bytes=20 * 1024 * 1024,
        )


def test_work_archive_rejects_unowned_work(tmp_path: Path):
    settings = make_settings(tmp_path)
    database, user_id, project_id, _chapter_id = create_project(settings)
    work_id, _edition_id = database.ensure_project_work(
        user_id=user_id,
        project_id=project_id,
    )
    other_user = database.create_user(
        "other-author",
        hash_password("password-456"),
    )

    with pytest.raises(WorkArchiveError, match="不属于"):
        create_work_archive(
            database=database,
            novels_dir=settings.novels_dir,
            documents_dir=settings.documents_dir,
            user_id=other_user,
            work_id=work_id,
            destination=tmp_path / "forbidden.zip",
            max_uncompressed_bytes=20 * 1024 * 1024,
        )


def test_work_archive_waits_for_active_ai_task(tmp_path: Path):
    settings = make_settings(tmp_path)
    database, user_id, project_id, chapter_id = create_project(settings)
    work_id, _edition_id = database.ensure_project_work(
        user_id=user_id,
        project_id=project_id,
    )
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

    with pytest.raises(WorkArchiveError, match="AI 任务正在"):
        create_work_archive(
            database=database,
            novels_dir=settings.novels_dir,
            documents_dir=settings.documents_dir,
            user_id=user_id,
            work_id=work_id,
            destination=tmp_path / "busy-work.zip",
            max_uncompressed_bytes=20 * 1024 * 1024,
        )
