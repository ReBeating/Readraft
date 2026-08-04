import hashlib
from pathlib import Path

import pytest

from app.db import ChapterHeadConflict, Database
from app.security import hash_password


def test_novel_project_and_head_history_lifecycle(tmp_path: Path):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id = database.create_user("writer", hash_password("password-123"))
    project_id = "p" * 32
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="雾港来信",
        genre="悬疑",
        premise="记者收到失踪父亲寄出的新信件，并返回雾港追查真相。",
        world_setting="当代海港小城，冬季常年有雾。",
        style_guide="克制，重视场景细节。",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
    )
    database.add_novel_character(
        user_id=user_id,
        project_id=project_id,
        name="林岚",
        role="主角，调查记者",
        traits="冷静、执拗",
        background="父亲十年前在雾港失踪",
        character_arc="从拒绝故乡到直面家庭秘密",
    )

    chapter_id = "c" * 32
    chapter_dir = tmp_path / "chapters" / chapter_id
    chapter_dir.mkdir(parents=True)
    content_path = chapter_dir / "content.txt"
    content_path.write_text("", encoding="utf-8")
    database.add_novel_chapter(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        title="第一章 迟到的信",
        outline="林岚收到父亲署名的信，决定返回雾港。",
        key_points="信件邮戳是三天前\n信封带有海水气味",
        content_path=content_path,
    )

    context = database.get_writing_context(user_id, chapter_id)
    assert context["chapter"]["project_title"] == "雾港来信"
    assert context["characters"][0]["name"] == "林岚"

    version_path = chapter_dir / "versions" / "first.txt"
    version_path.parent.mkdir()
    first_content = "林岚拆开了那封迟到十年的信。"
    version_path.write_text(first_content, encoding="utf-8")
    first_version_id = database.record_manual_chapter_version(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        version_path=version_path,
        char_count=len(first_content),
        content_hash=hashlib.sha256(first_content.encode()).hexdigest(),
        expected_old_head_version_id="",
    )
    assert first_version_id

    chapter = database.get_novel_chapter(user_id, project_id, chapter_id)
    assert chapter["status"] == "written"
    assert chapter["char_count"] == len(first_content)
    assert len(database.list_chapter_versions(user_id, project_id, chapter_id)) == 1

    rewrite_content = "林岚收起信，买下了回雾港的车票。"
    rewrite_path = chapter_dir / "versions" / "rewrite.txt"
    rewrite_path.write_text(rewrite_content, encoding="utf-8")
    head_version_id = database.record_manual_chapter_version(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        version_path=rewrite_path,
        char_count=len(rewrite_content),
        content_hash=hashlib.sha256(rewrite_content.encode()).hexdigest(),
        change_summary="直接采用新正文",
        expected_old_head_version_id=first_version_id,
    )
    assert head_version_id
    current_chapter = database.get_novel_chapter(
        user_id, project_id, chapter_id
    )
    assert current_chapter["head_version_id"] == head_version_id
    head_version = database.get_chapter_version(
        user_id,
        project_id,
        chapter_id,
        head_version_id,
    )
    assert head_version["head_version_id"] == head_version["id"]

    with database.connection() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM generation_jobs WHERE user_id=?",
            (user_id,),
        ).fetchone()[0] == 0

    assert database.delete_novel_project(user_id, project_id) is True
    assert database.get_novel_project(user_id, project_id) is None
    assert database.list_novel_chapters(user_id, project_id) == []
    with database.connection() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM novel_characters WHERE project_id=?",
            (project_id,),
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM generation_jobs WHERE project_id=?",
            (project_id,),
        ).fetchone()[0] == 0


def test_novel_project_delete_waits_for_active_generation(tmp_path: Path):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id = database.create_user("busy-writer", hash_password("password-123"))
    project_id = "b" * 32
    chapter_id = "d" * 32
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="进行中",
        genre="悬疑",
        premise="",
        world_setting="",
        style_guide="",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
    )
    chapter_dir = tmp_path / "novels" / chapter_id
    chapter_dir.mkdir(parents=True)
    content_path = chapter_dir / "content.txt"
    content_path.write_text("", encoding="utf-8")
    database.add_novel_chapter(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        title="第一章",
        outline="",
        key_points="",
        content_path=content_path,
    )
    database.create_chapter_planning_job(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        instruction="",
        provider="mock",
        model="mock-novel-writer",
        credential_source="default",
    )

    with pytest.raises(ValueError, match="正在排队或运行"):
        database.delete_novel_project(user_id, project_id)
    assert database.get_novel_project(user_id, project_id) is not None


def test_edit_buffer_is_recoverable_but_not_a_version(tmp_path: Path):
    database = Database(tmp_path / "buffer.db")
    database.initialize()
    user_id = database.create_user("buffer-writer", hash_password("password-123"))
    project_id = "buffer-project"
    chapter_id = "buffer-chapter"
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="暂存测试",
        genre="",
        premise="",
        world_setting="",
        style_guide="",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
    )
    chapter_dir = tmp_path / "buffer-chapter"
    versions_dir = chapter_dir / "versions"
    versions_dir.mkdir(parents=True)
    cache_path = chapter_dir / "content.txt"
    cache_path.write_text("第一版", encoding="utf-8")
    database.add_novel_chapter(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        title="第一章",
        outline="",
        key_points="",
        content_path=cache_path,
    )
    first_path = versions_dir / "first.txt"
    first_path.write_text("第一版", encoding="utf-8")
    first_id = database.record_manual_chapter_version(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        version_path=first_path,
        char_count=3,
        content_hash=hashlib.sha256("第一版".encode()).hexdigest(),
        expected_old_head_version_id="",
    )
    assert first_id

    buffered = database.save_chapter_edit_buffer(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        base_version_id=str(first_id),
        content="尚未正式提交的第二版",
        content_hash=hashlib.sha256(
            "尚未正式提交的第二版".encode()
        ).hexdigest(),
    )
    assert buffered["buffered"] is True
    chapter = database.get_novel_chapter(user_id, project_id, chapter_id)
    assert chapter["head_version_id"] == first_id
    assert chapter["edit_buffer_content"] == "尚未正式提交的第二版"
    assert database.count_chapter_versions(user_id, project_id, chapter_id) == 1

    second_path = versions_dir / "second.txt"
    second_path.write_text("尚未正式提交的第二版", encoding="utf-8")
    buffered_hash = hashlib.sha256(
        "尚未正式提交的第二版".encode()
    ).hexdigest()
    second_id = database.record_manual_chapter_version(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        version_path=second_path,
        char_count=10,
        content_hash=buffered_hash,
        expected_old_head_version_id=str(first_id),
    )
    assert second_id and second_id != first_id
    chapter = database.get_novel_chapter(user_id, project_id, chapter_id)
    assert chapter["edit_buffer_content"] is None
    assert chapter["head_content_hash"] == buffered_hash

    with pytest.raises(ChapterHeadConflict):
        database.save_chapter_edit_buffer(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            base_version_id=str(first_id),
            content="旧标签页的内容",
            content_hash=hashlib.sha256("旧标签页的内容".encode()).hexdigest(),
        )
    with database.connection() as connection:
        columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(novel_chapter_versions)"
            ).fetchall()
        }
    assert "status" not in columns


def test_chapter_history_is_counted_and_paginated(tmp_path: Path):
    database = Database(tmp_path / "history.db")
    database.initialize()
    user_id = database.create_user("history-writer", hash_password("password-123"))
    project_id = "history-project"
    chapter_id = "history-chapter"
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="历史测试",
        genre="",
        premise="",
        world_setting="",
        style_guide="",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
    )
    chapter_dir = tmp_path / "history-chapter"
    versions_dir = chapter_dir / "versions"
    versions_dir.mkdir(parents=True)
    cache_path = chapter_dir / "content.txt"
    cache_path.write_text("", encoding="utf-8")
    database.add_novel_chapter(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        title="第一章",
        outline="",
        key_points="",
        content_path=cache_path,
    )
    head_id = ""
    for index in range(35):
        content = f"第 {index + 1} 个版本"
        version_path = versions_dir / f"{index + 1:03d}.txt"
        version_path.write_text(content, encoding="utf-8")
        next_id = database.record_manual_chapter_version(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            version_path=version_path,
            char_count=len(content),
            content_hash=hashlib.sha256(content.encode()).hexdigest(),
            expected_old_head_version_id=head_id,
        )
        assert next_id
        head_id = str(next_id)

    assert database.count_chapter_versions(user_id, project_id, chapter_id) == 35
    first_page = database.list_chapter_versions(
        user_id, project_id, chapter_id, limit=20, offset=0
    )
    second_page = database.list_chapter_versions(
        user_id, project_id, chapter_id, limit=20, offset=20
    )
    assert len(first_page) == 20
    assert len(second_page) == 15
    assert {item["id"] for item in first_page}.isdisjoint(
        {item["id"] for item in second_page}
    )
    assert sum(bool(item["is_head"]) for item in first_page + second_page) == 1
