from __future__ import annotations

import hashlib
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Optional

from .chapter_splitter import ChapterChunk
from .db import Database


def _secure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    path.chmod(0o700)


def _write_private_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o600)


def _create_editable_chapters(
    *,
    database: Database,
    novels_dir: Path,
    user_id: int,
    project_id: str,
    chapters: Iterable[tuple[str, str]],
) -> None:
    for title, content in chapters:
        chapter_id = uuid.uuid4().hex
        chapter_dir = (
            novels_dir
            / str(user_id)
            / project_id
            / "chapters"
            / chapter_id
        )
        versions_dir = chapter_dir / "versions"
        _secure_directory(versions_dir)
        content_path = chapter_dir / "content.txt"
        _write_private_text(content_path, content)
        database.add_novel_chapter(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            title=title,
            outline="",
            key_points="",
            content_path=content_path,
        )
        if content:
            version_path = versions_dir / f"imported-{uuid.uuid4().hex}.txt"
            _write_private_text(version_path, content)
            database.record_manual_chapter_version(
                user_id=user_id,
                project_id=project_id,
                chapter_id=chapter_id,
                version_path=version_path,
                char_count=len(content),
                effective_char_count=len(content.strip()),
                content_hash=hashlib.sha256(
                    content.encode("utf-8")
                ).hexdigest(),
                change_summary="导入为可编辑创作稿",
            )


def create_writing_project_from_chunks(
    *,
    database: Database,
    novels_dir: Path,
    user_id: int,
    title: str,
    chunks: Iterable[ChapterChunk],
    work_id: Optional[str] = None,
    source_edition_id: Optional[str] = None,
    branch_intent: str = "original",
    edition_label: str = "创作稿",
) -> str:
    project_id = uuid.uuid4().hex
    user_root = novels_dir / str(user_id)
    user_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    user_root.chmod(0o700)
    project_dir = user_root / project_id
    try:
        _secure_directory(project_dir)
        _secure_directory(project_dir / "chapters")
        database.create_novel_project(
            user_id=user_id,
            project_id=project_id,
            title=title,
            genre="",
            premise="",
            world_setting="",
            style_guide="",
            point_of_view="第三人称限知",
            target_chapter_chars=3000,
            work_id=work_id,
            source_edition_id=source_edition_id,
            branch_intent=branch_intent,
            edition_label=edition_label,
        )
        _create_editable_chapters(
            database=database,
            novels_dir=novels_dir,
            user_id=user_id,
            project_id=project_id,
            chapters=(
                (str(chunk.title or ""), str(chunk.text))
                for chunk in chunks
            ),
        )
    except Exception:
        if database.get_novel_project(user_id, project_id):
            database.delete_novel_project(user_id, project_id)
        shutil.rmtree(project_dir, ignore_errors=True)
        raise
    return project_id


def create_reading_document_from_chunks(
    *,
    database: Database,
    documents_dir: Path,
    user_id: int,
    title: str,
    original_filename: str,
    source_encoding: str,
    source_text: str,
    chunks: Iterable[ChapterChunk],
    max_documents: Optional[int] = None,
    max_stored_chars: Optional[int] = None,
    work_id: Optional[str] = None,
    source_edition_id: Optional[str] = None,
    edition_kind: str = "source",
    branch_intent: str = "original",
    edition_label: str = "导入原文",
) -> str:
    chunk_list = list(chunks)
    document_id = uuid.uuid4().hex
    user_root = documents_dir / str(user_id)
    user_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    user_root.chmod(0o700)
    document_dir = user_root / document_id
    chapters_dir = document_dir / "chapters"
    try:
        _secure_directory(document_dir)
        _secure_directory(chapters_dir)
        source_path = document_dir / "source.txt"
        _write_private_text(source_path, source_text)
        chapter_paths: list[Path] = []
        for position, chunk in enumerate(chunk_list, start=1):
            chapter_path = chapters_dir / f"{position:05d}.txt"
            _write_private_text(chapter_path, chunk.text)
            chapter_paths.append(chapter_path)
        database.create_document(
            user_id=user_id,
            title=title,
            original_filename=original_filename,
            source_path=source_path,
            source_encoding=source_encoding,
            text_length=len(source_text),
            chunks=chunk_list,
            chapter_paths=chapter_paths,
            max_documents=max_documents,
            max_stored_chars=max_stored_chars,
            work_id=work_id,
            source_edition_id=source_edition_id,
            edition_kind=edition_kind,
            branch_intent=branch_intent,
            edition_label=edition_label,
        )
    except Exception:
        shutil.rmtree(document_dir, ignore_errors=True)
        raise
    return document_id


def create_writing_branch_from_document(
    *,
    database: Database,
    novels_dir: Path,
    user_id: int,
    document_id: str,
    intent: str,
) -> str:
    if intent not in {"rewrite", "sequel"}:
        raise ValueError("请选择改写或续写")
    document = database.get_document(user_id, document_id)
    work = database.get_work_for_document(user_id, document_id)
    if not document or not work:
        raise ValueError("导入原文不存在")
    source_edition = next(
        (
            edition
            for edition in work["editions"]
            if str(edition.get("document_id") or "") == document_id
        ),
        None,
    )
    if not source_edition:
        raise ValueError("找不到原文所属版本")
    base_title = str(document.get("title") or "未命名作品").strip()
    suffix = "改写稿" if intent == "rewrite" else "续写"
    title = f"{base_title} · {suffix}"
    if intent == "rewrite":
        source_chapters = database.list_chapters(user_id, document_id)
        chunks = [
            ChapterChunk(
                title=str(chapter.get("title") or ""),
                text=Path(str(chapter["content_path"])).read_text(
                    encoding="utf-8"
                ),
                kind=str(chapter.get("kind") or "chapter"),
                source_start=int(chapter.get("source_start") or 0),
                source_end=int(chapter.get("source_end") or 0),
                part_number=int(chapter.get("part_number") or 1),
                part_count=int(chapter.get("part_count") or 1),
            )
            for chapter in source_chapters
        ]
    else:
        chunks = []
    return create_writing_project_from_chunks(
        database=database,
        novels_dir=novels_dir,
        user_id=user_id,
        title=title,
        chunks=chunks,
        work_id=str(work["id"]),
        source_edition_id=str(source_edition["id"]),
        branch_intent=intent,
        edition_label=suffix,
    )


def _snapshot_chunks(
    chapters: Iterable[Mapping[str, object]],
) -> tuple[str, list[ChapterChunk]]:
    pieces: list[str] = []
    chunks: list[ChapterChunk] = []
    cursor = 0
    for chapter in chapters:
        content = Path(str(chapter["content_path"])).read_text(
            encoding="utf-8"
        )
        if not content.strip():
            continue
        position = int(chapter.get("position") or len(chunks) + 1)
        title = str(chapter.get("title") or "").strip()
        title = title or f"第{position}章"
        piece = f"{title}\n{content.strip()}"
        if pieces:
            cursor += 2
        start = cursor
        end = start + len(piece)
        pieces.append(piece)
        chunks.append(
            ChapterChunk(
                title=title,
                text=piece,
                kind="chapter",
                source_start=start,
                source_end=end,
            )
        )
        cursor = end
    return "\n\n".join(pieces), chunks


def create_reading_snapshot(
    *,
    database: Database,
    documents_dir: Path,
    user_id: int,
    project_id: str,
    max_documents: Optional[int] = None,
    max_stored_chars: Optional[int] = None,
) -> str:
    project = database.get_novel_project(user_id, project_id)
    work = database.get_work_for_project(user_id, project_id)
    if not project or not work:
        raise ValueError("创作稿不存在")
    source_edition = next(
        (
            edition
            for edition in work["editions"]
            if str(edition.get("project_id") or "") == project_id
        ),
        None,
    )
    if not source_edition:
        raise ValueError("找不到创作稿所属版本")
    source_text, chunks = _snapshot_chunks(
        database.list_novel_chapters(user_id, project_id)
    )
    if not source_text.strip() or not chunks:
        raise ValueError("作品还没有可生成阅读版的正文")

    title = str(project.get("title") or "未命名作品").strip()
    label = f"阅读版 · {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    return create_reading_document_from_chunks(
        database=database,
        documents_dir=documents_dir,
        user_id=user_id,
        title=title,
        original_filename=f"{title}.snapshot.txt",
        source_encoding="utf-8",
        source_text=source_text,
        chunks=chunks,
        max_documents=max_documents,
        max_stored_chars=max_stored_chars,
        work_id=str(work["id"]),
        source_edition_id=str(source_edition["id"]),
        edition_kind="snapshot",
        branch_intent="snapshot",
        edition_label=label,
    )
