from __future__ import annotations

import json
import hashlib
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
    create_version_tag,
)


def make_settings(root: Path) -> Settings:
    return Settings(
        app_name="Readraft 作品归档测试",
        app_env="test",
        secret_key="work-archive-test-secret",  # pragma: allowlist secret
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


def create_project(settings: Settings) -> tuple[Database, int, str, str]:
    settings.ensure_directories()
    database = Database(settings.database_path)
    database.initialize()
    user_id = database.create_user("archive-author", hash_password("password-123"))
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
        theme="记忆是否能成为证据",
        story_promise="每个时间证据都能被核对",
    )
    first_character_id = database.add_novel_character(
        user_id=user_id,
        project_id=project_id,
        name="林岚",
        role="修表师",
        traits="谨慎、敏锐",
        background="经营一家老钟表店",
        character_arc="从回避旧约到主动核对真相",
        external_goal="查明船票日期异常",
        secret="曾经改过修理簿上的日期",
    )
    second_character_id = database.add_novel_character(
        user_id=user_id,
        project_id=project_id,
        name="周屿",
        role="灯塔管理员",
        traits="沉默、固执",
        background="保管旧码头航行记录",
        character_arc="从隐瞒见证到公开记录",
    )
    database.add_character_relationship(
        user_id=user_id,
        project_id=project_id,
        character_a_id=first_character_id,
        character_b_id=second_character_id,
        relationship="互相怀疑的旧识",
        tension="两人掌握的日期彼此矛盾",
        change_direction="从互相验证到共同作证",
    )
    database.add_world_entry(
        user_id=user_id,
        project_id=project_id,
        entry_type="location",
        name="纸灯塔",
        description="旧码头唯一仍在运作的灯塔。",
        constraints="只在退潮后的半小时开放。",
    )
    chapter_id = "portable-chapter"
    chapter_root = (
        settings.novels_dir / str(user_id) / project_id / "chapters" / chapter_id
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
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
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


def test_repository_migration_removes_hidden_legacy_branches(tmp_path: Path):
    settings = make_settings(tmp_path)
    settings.ensure_directories()
    database = Database(settings.database_path)
    database.initialize()
    user_id = database.create_user("repository-author", hash_password("password-123"))
    for project_id in ("current-project", "hidden-old-project"):
        database.create_novel_project(
            user_id=user_id,
            project_id=project_id,
            title=project_id,
            genre="悬疑",
            premise="测试版本边界。",
            world_setting="测试世界。",
            style_guide="克制。",
            point_of_view="第三人称限知",
            target_chapter_chars=3000,
        )

    with database.connection() as connection:
        current = connection.execute(
            "SELECT work_id FROM work_versions WHERE project_id=?",
            ("current-project",),
        ).fetchone()
        hidden = connection.execute(
            "SELECT work_id FROM work_versions WHERE project_id=?",
            ("hidden-old-project",),
        ).fetchone()
        connection.execute(
            """
            UPDATE work_versions
            SET work_id=?, ref_type='legacy', ref_name='legacy-1',
                label='旧分支', is_editable=0
            WHERE project_id=?
            """,
            (str(current["work_id"]), "hidden-old-project"),
        )
        connection.execute(
            "DELETE FROM works WHERE id=?",
            (str(hidden["work_id"]),),
        )
        connection.execute(
            "UPDATE works SET last_ref_name='legacy-1' WHERE id=?",
            (str(current["work_id"]),),
        )
        connection.execute("DELETE FROM schema_migrations WHERE version=44")
        connection.commit()

    database.initialize()

    assert database.get_novel_project(user_id, "hidden-old-project") is None
    work = database.get_work(user_id, str(current["work_id"]))
    assert work is not None
    assert work["last_ref_name"] == "main"
    assert {version["ref_type"] for version in work["versions"]} == {"branch"}


def test_complete_work_archive_round_trip_preserves_versions_and_archive(
    tmp_path: Path,
):
    settings = make_settings(tmp_path)
    database, user_id, project_id, _chapter_id = create_project(settings)
    work_id, main_version_id = database.ensure_project_work(
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
        content_version_id=main_version_id,
    )
    setting_id = database.adopt_work_archive_entry(
        user_id=user_id,
        work_id=work_id,
        entry_id=note_id,
        category="structure",
        title="钟表意象规则",
        content="人物回避旧约时，让停走的秒针推动下一步行动。",
    )
    create_version_tag(
        database=database,
        documents_dir=settings.documents_dir,
        user_id=user_id,
        project_id=project_id,
        label="一稿",
    )
    archive = tmp_path / "complete-work.readraft.zip"

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
    assert summary.version_count == 2
    assert manifest["counts"]["versions"] == 2
    assert len(manifest["roots"]["projects"]) == 1
    assert len(manifest["roots"]["documents"]) == 1
    assert "works" in manifest["tables"]
    assert "work_versions" in manifest["tables"]
    assert "work_tag_chapter_heads" in manifest["tables"]
    assert "work_archive_entries" in manifest["tables"]
    assert "novel_world_entries" in manifest["tables"]
    assert "novel_character_relationships" in manifest["tables"]
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
    assert imported.version_count == 2
    imported_work = database.get_work(user_id, imported.work_id)
    assert imported_work and imported_work["title"] == "纸灯塔"
    assert imported_work["main_version"]
    assert imported_work["main_version"]["project_id"] != project_id
    assert len(imported_work["tag_versions"]) == 1
    imported_tag = imported_work["tag_versions"][0]
    assert imported_tag["label"] == "一稿"
    assert imported_tag["base_version"]["ref_name"] == "main"
    assert imported_tag["creative_snapshot"]["project"]["title"] == "纸灯塔"
    assert imported_tag["creative_snapshot"]["project"]["theme"] == (
        "记忆是否能成为证据"
    )
    assert imported_tag["creative_snapshot"]["world_entries"][0]["name"] == ("纸灯塔")

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
        str(imported_work["main_version"]["project_id"]),
    )
    assert len(imported_chapters) == 1
    imported_project_id = str(imported_work["main_version"]["project_id"])
    imported_head_id = str(imported_chapters[0]["head_version_id"] or "")
    assert imported_head_id
    imported_versions = database.list_chapter_versions(
        user_id,
        imported_project_id,
        str(imported_chapters[0]["id"]),
    )
    assert [
        version["id"]
        for version in imported_versions
        if version["is_head"]
    ] == [imported_head_id]
    assert (
        database.list_world_entries(user_id, imported_project_id)[0]["name"] == "纸灯塔"
    )
    assert (
        database.list_character_relationships(user_id, imported_project_id)[0][
            "relationship"
        ]
        == "互相怀疑的旧识"
    )
    imported_content_path = Path(str(imported_chapters[0]["content_path"]))
    assert imported_content_path.is_relative_to(
        settings.novels_dir
        / str(user_id)
        / str(imported_work["main_version"]["project_id"])
    )
    assert "林岚把船票" in imported_content_path.read_text(encoding="utf-8")
    imported_document = database.get_document(
        user_id,
        str(imported_tag["document_id"]),
    )
    assert imported_document
    assert Path(str(imported_document["source_path"])).is_relative_to(
        settings.documents_dir / str(user_id) / str(imported_tag["document_id"])
    )
    with database.connection() as connection:
        imported_manifest = connection.execute(
            """
            SELECT manifest.source_chapter_id,
                   manifest.source_version_id,
                   manifest.content_hash
            FROM work_tag_chapter_heads manifest
            WHERE manifest.work_version_id=?
            """,
            (imported_tag["id"],),
        ).fetchone()
        assert imported_manifest
        assert imported_manifest["source_chapter_id"] == str(
            imported_chapters[0]["id"]
        )
        assert imported_manifest["source_version_id"] == imported_head_id
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM api_credentials WHERE user_id=?",
                (user_id,),
            ).fetchone()[0]
            == 1
        )


def test_tag_freezes_current_main_head_after_main_advances(tmp_path: Path):
    settings = make_settings(tmp_path)
    database, user_id, project_id, chapter_id = create_project(settings)
    chapter = database.get_novel_chapter(user_id, project_id, chapter_id)
    assert chapter
    first_head_id = str(chapter["head_version_id"])
    first_head = database.get_chapter_version(
        user_id,
        project_id,
        chapter_id,
        first_head_id,
    )
    assert first_head
    first_content = Path(str(first_head["content_path"])).read_text(
        encoding="utf-8"
    )
    Path(str(chapter["content_path"])).write_text(
        "这是故意制造的过期 content.txt 缓存。",
        encoding="utf-8",
    )

    document_id = create_version_tag(
        database=database,
        documents_dir=settings.documents_dir,
        user_id=user_id,
        project_id=project_id,
        label="冻结的一稿",
    )

    next_content = "林岚重查船票，确认真正被改动的是灯塔值班簿。"
    chapter_path = Path(str(chapter["content_path"]))
    version_path = chapter_path.parent / "versions" / "manual-next.txt"
    chapter_path.write_text(next_content, encoding="utf-8")
    version_path.write_text(next_content, encoding="utf-8")
    next_head_id = database.record_manual_chapter_version(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        version_path=version_path,
        char_count=len(next_content),
        content_hash="next-content-hash",
        change_summary="推进 main HEAD",
    )
    assert next_head_id and next_head_id != first_head_id

    tag_chapters = database.list_chapters(user_id, document_id)
    assert len(tag_chapters) == 1
    assert Path(str(tag_chapters[0]["content_path"])).read_text(
        encoding="utf-8"
    ) == first_content
    with database.connection() as connection:
        manifest = connection.execute(
            """
            SELECT manifest.source_chapter_id,
                   manifest.source_version_id,
                   manifest.content_hash
            FROM work_tag_chapter_heads manifest
            JOIN work_versions version
              ON version.id=manifest.work_version_id
            WHERE version.document_id=?
            """,
            (document_id,),
        ).fetchone()
    assert manifest["source_chapter_id"] == chapter_id
    assert manifest["source_version_id"] == first_head_id
    assert manifest["content_hash"] == hashlib.sha256(
        first_content.encode("utf-8")
    ).hexdigest()
    current = database.get_novel_chapter(user_id, project_id, chapter_id)
    assert current and current["head_version_id"] == next_head_id

    assert database.delete_novel_project(user_id, project_id) is True
    assert database.get_document(user_id, document_id)
    remaining_work = database.get_work_for_document(user_id, document_id)
    assert remaining_work
    with database.connection() as connection:
        preserved_manifest = connection.execute(
            """
            SELECT source_chapter_id, source_version_id, content_hash
            FROM work_tag_chapter_heads
            WHERE document_chapter_id=?
            """,
            (tag_chapters[0]["id"],),
        ).fetchone()
    assert preserved_manifest["source_chapter_id"] == chapter_id
    assert preserved_manifest["source_version_id"] == first_head_id


def test_work_archive_rejects_path_traversal(tmp_path: Path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as payload:
        payload.writestr("../escape.txt", "unsafe")
        payload.writestr(
            "manifest.json",
            json.dumps(
                {
                    "format": WORK_ARCHIVE_FORMAT,
                    "version": 2,
                    "work": {"id": "w", "title": "unsafe"},
                    "roots": {"projects": [], "documents": []},
                    "tables": {
                        "works": [{"id": "w", "user_id": 1}],
                        "work_versions": [
                            {
                                "id": "e",
                                "work_id": "w",
                                "ref_type": "branch",
                                "ref_name": "main",
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
    work_id, _version_id = database.ensure_project_work(
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
            info.filename: source.read(info.filename) for info in source.infolist()
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
    work_id, _version_id = database.ensure_project_work(
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
    work_id, _version_id = database.ensure_project_work(
        user_id=user_id,
        project_id=project_id,
    )
    database.create_chapter_planning_job(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
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


def test_uncategorized_analysis_must_be_classified_before_adoption(
    tmp_path: Path,
):
    settings = make_settings(tmp_path)
    database, user_id, project_id, _chapter_id = create_project(settings)
    work_id, version_id = database.ensure_project_work(
        user_id=user_id,
        project_id=project_id,
    )
    note_id = database.add_work_archive_entry(
        user_id=user_id,
        work_id=work_id,
        entry_type="analysis_note",
        title="暂未归类",
        content="这条观察还不知道会约束哪一部分创作。",
        category="uncategorized",
        content_version_id=version_id,
    )

    with pytest.raises(ValueError, match="有效的创作设定分类"):
        database.adopt_work_archive_entry(
            user_id=user_id,
            work_id=work_id,
            entry_id=note_id,
            category="uncategorized",
        )

    note = next(
        item
        for item in database.list_work_archive_entries(user_id, work_id, version_id)
        if item["id"] == note_id
    )
    assert note["entry_type"] == "analysis_note"
    assert note["adopted_setting_id"] is None
