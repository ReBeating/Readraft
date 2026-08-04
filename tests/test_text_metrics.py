from pathlib import Path

from app.db import Database
from app.security import hash_password
from app.text_metrics import effective_char_count


def test_effective_char_count_ignores_all_whitespace():
    assert effective_char_count("林 岚\n拆\t信\r\n") == 4


def test_manual_version_can_become_current_without_audit(tmp_path: Path):
    database = Database(tmp_path / "chapter-version.db")
    database.initialize()
    user_id = database.create_user(
        "chapter-version-author", hash_password("password-123")
    )
    project_id = "p" * 32
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="雾港来信",
        genre="悬疑",
        premise="记者返回故乡调查父亲失踪案。",
        world_setting="当代海港。",
        style_guide="克制。",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
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
        title="第一章",
        outline="林岚收到来信。",
        key_points="核对邮戳",
        content_path=content_path,
    )
    version_path = chapter_dir / "manual.txt"
    version_path.write_text("极短章节。", encoding="utf-8")
    version_id = database.record_manual_chapter_version(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        version_path=version_path,
        char_count=5,
        effective_char_count=5,
    )

    assert database.get_novel_chapter(
        user_id, project_id, chapter_id
    )["head_version_id"] == version_id
    version = database.get_chapter_version(
        user_id, project_id, chapter_id, str(version_id)
    )
    assert version["head_version_id"] == version["id"]
    assert version["quality_status"] == "pass"
