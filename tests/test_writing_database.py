import json
from pathlib import Path

import pytest

from app.db import Database
from app.security import hash_password


def test_novel_project_and_generation_lifecycle(tmp_path: Path):
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

    job_id = database.create_generation_job(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        operation="draft",
        instruction="开头从拆信开始",
        provider="mock",
        model="mock-novel-writer",
        credential_source="default",
    )
    claimed = database.claim_next_generation()
    assert claimed is not None
    assert claimed["id"] == job_id
    context = database.get_writing_context(user_id, chapter_id)
    assert context["chapter"]["project_title"] == "雾港来信"
    assert context["characters"][0]["name"] == "林岚"

    version_path = chapter_dir / "versions" / f"{job_id}.txt"
    version_path.parent.mkdir()
    version_path.write_text("林岚拆开了那封迟到十年的信。", encoding="utf-8")
    assert database.complete_generation(
        job_id=job_id,
        claim_token=claimed["claim_token"],
        version_path=version_path,
        result_char_count=16,
        input_tokens=100,
        output_tokens=80,
    )

    job = database.get_generation_job(user_id, job_id)
    assert job["status"] == "completed"
    chapter = database.get_novel_chapter(user_id, project_id, chapter_id)
    assert chapter["status"] == "draft"
    assert chapter["char_count"] == 16
    assert len(database.list_chapter_versions(user_id, project_id, chapter_id)) == 1

    canonical_job_id = database.create_generation_job(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        operation="rewrite",
        instruction="直接采用新正文",
        provider="mock",
        model="mock-novel-writer",
        credential_source="default",
    )
    canonical_claim = database.claim_next_generation()
    assert canonical_claim["id"] == canonical_job_id
    canonical_path = (
        chapter_dir / "versions" / f"{canonical_job_id}.txt"
    )
    canonical_path.write_text("林岚收起信，买下了回雾港的车票。", encoding="utf-8")
    canonical_version_id = database.complete_generation(
        job_id=canonical_job_id,
        claim_token=canonical_claim["claim_token"],
        version_path=canonical_path,
        result_char_count=18,
        input_tokens=120,
        output_tokens=90,
        accept_as_canonical=True,
    )
    assert canonical_version_id
    canonical_job = database.get_generation_job(
        user_id, canonical_job_id
    )
    assert canonical_job["status"] == "completed"
    assert json.loads(canonical_job["result_json"]) == {
        "version_id": canonical_version_id,
        "canonical": True,
    }
    canonical_chapter = database.get_novel_chapter(
        user_id, project_id, chapter_id
    )
    assert canonical_chapter["canonical_version_id"] == canonical_version_id
    canonical_version = database.get_chapter_version(
        user_id,
        project_id,
        chapter_id,
        canonical_version_id,
    )
    assert canonical_version["status"] == "canonical"

    for index in range(12):
        extra_job_id = database.create_generation_job(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            operation="rewrite",
            instruction=f"第 {index + 1} 次测试",
            provider="mock",
            model="mock-novel-writer",
            credential_source="default",
        )
        extra_claim = database.claim_next_generation()
        assert extra_claim["id"] == extra_job_id
        assert database.fail_generation(
            extra_job_id,
            extra_claim["claim_token"],
            "测试任务主动结束",
        )
    with database.connection() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM generation_jobs WHERE user_id=?",
            (user_id,),
        ).fetchone()[0] == 14

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
    database.create_generation_job(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        operation="draft",
        instruction="",
        provider="mock",
        model="mock-novel-writer",
        credential_source="default",
    )

    with pytest.raises(ValueError, match="正在排队或运行"):
        database.delete_novel_project(user_id, project_id)
    assert database.get_novel_project(user_id, project_id) is not None
