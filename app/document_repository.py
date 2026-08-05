"""Persistence boundary for imported read-only documents and their chapters."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .chapter_splitter import ChapterChunk


class DocumentRepository:
    def __init__(
        self,
        database: Any,
        *,
        attach_work_version: Callable[..., tuple[str, str]],
        now: Callable[[], str],
    ) -> None:
        self.database = database
        self.attach_work_version = attach_work_version
        self.now = now

    def create(
        self,
        *,
        user_id: int,
        title: str,
        original_filename: str,
        source_path: Path,
        source_encoding: str,
        text_length: int,
        chunks: Iterable[ChapterChunk],
        chapter_paths: Iterable[Path],
        max_documents: int | None = None,
        max_stored_chars: int | None = None,
        work_id: str | None = None,
        base_version_id: str | None = None,
        ref_name: str = "source",
        version_label: str = "原始版本",
        intent: str = "original",
        content_hash: str = "",
        creative_snapshot: Mapping[str, Any] | None = None,
        story_memory_snapshots: Iterable[Mapping[str, Any]] | None = None,
        source_head_snapshots: Iterable[Mapping[str, Any]] | None = None,
    ) -> str:
        document_id = source_path.parent.name
        chunk_list = list(chunks)
        path_list = list(chapter_paths)
        memory_snapshot_list = list(story_memory_snapshots or [])
        source_head_snapshot_list = list(source_head_snapshots or [])
        if len(chunk_list) != len(path_list):
            raise ValueError("章节与文件数量不一致")

        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            usage = connection.execute(
                """
                SELECT COUNT(*) AS document_count,
                       COALESCE(SUM(char_count), 0) AS stored_chars
                FROM documents WHERE user_id=?
                """,
                (user_id,),
            ).fetchone()
            if (
                max_documents is not None
                and int(usage["document_count"] or 0) >= max_documents
            ):
                connection.rollback()
                raise ValueError(f"每个账号最多保存 {max_documents} 本文档")
            if (
                max_stored_chars is not None
                and int(usage["stored_chars"] or 0) + text_length > max_stored_chars
            ):
                connection.rollback()
                raise ValueError(f"账号累计正文不能超过 {max_stored_chars:,} 字")
            connection.execute(
                """
                INSERT INTO documents(
                    id, user_id, title, original_filename, source_path,
                    source_encoding, char_count, split_strategy, source_hash,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    user_id,
                    title,
                    original_filename,
                    str(source_path),
                    source_encoding,
                    text_length,
                    "heading+smart-fallback-v2",
                    content_hash,
                    self.now(),
                ),
            )
            chapter_ids: dict[int, str] = {}
            for position, (chunk, content_path) in enumerate(
                zip(chunk_list, path_list), start=1
            ):
                chapter_id = uuid.uuid4().hex
                chapter_ids[position] = chapter_id
                connection.execute(
                    """
                    INSERT INTO chapters(
                        id, document_id, position, title, kind, content_path,
                        char_count, source_start, source_end, part_number, part_count,
                        split_confidence, split_reason, title_source, content_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chapter_id,
                        document_id,
                        position,
                        chunk.title,
                        chunk.kind,
                        str(content_path),
                        len(chunk.text),
                        chunk.source_start,
                        chunk.source_end,
                        chunk.part_number,
                        chunk.part_count,
                        float(chunk.split_confidence),
                        str(chunk.split_reason),
                        str(chunk.title_source),
                        hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
                    ),
                )
            _, work_version_id = self.attach_work_version(
                connection,
                user_id=user_id,
                title=title,
                ref_type="tag",
                ref_name=ref_name,
                label=version_label,
                intent=intent,
                document_id=document_id,
                work_id=work_id,
                base_version_id=base_version_id,
                content_hash=content_hash,
                creative_snapshot=creative_snapshot,
                origin="imported",
            )
            self._store_source_heads(
                connection,
                work_version_id=work_version_id,
                chapter_ids=chapter_ids,
                chunks=chunk_list,
                snapshots=source_head_snapshot_list,
            )
            self._store_memory_snapshots(
                connection,
                work_version_id=work_version_id,
                chapter_ids=chapter_ids,
                snapshots=memory_snapshot_list,
            )
            connection.commit()
        return document_id

    def _store_source_heads(
        self,
        connection: Any,
        *,
        work_version_id: str,
        chapter_ids: Mapping[int, str],
        chunks: list[ChapterChunk],
        snapshots: list[Mapping[str, Any]],
    ) -> None:
        for snapshot in snapshots:
            try:
                chapter_position = int(snapshot["chapter_position"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Tag 章节清单缺少有效位置") from exc
            document_chapter_id = chapter_ids.get(chapter_position)
            source_chapter_id = str(snapshot.get("source_chapter_id") or "").strip()
            source_version_id = str(snapshot.get("source_version_id") or "").strip()
            snapshot_hash = str(snapshot.get("content_hash") or "").strip()
            if not document_chapter_id or not source_chapter_id:
                raise ValueError("Tag 章节清单指向不存在的章节")
            if not snapshot_hash:
                raise ValueError("Tag 章节清单缺少正文校验值")
            chunk_hash = hashlib.sha256(
                chunks[chapter_position - 1].text.encode("utf-8")
            ).hexdigest()
            if chunk_hash != snapshot_hash:
                raise ValueError("Tag 章节清单与正文快照不一致")
            if source_version_id:
                source = connection.execute(
                    """
                    SELECT 1 FROM novel_chapter_versions version
                    WHERE version.id=? AND version.chapter_id=?
                    """,
                    (source_version_id, source_chapter_id),
                ).fetchone()
                if not source:
                    raise ValueError("Tag 来源 HEAD 已不存在")
            connection.execute(
                """
                INSERT INTO work_tag_chapter_heads(
                    work_version_id, document_chapter_id,
                    source_chapter_id, source_version_id,
                    position, content_hash
                ) VALUES (?, ?, ?, NULLIF(?, ''), ?, ?)
                """,
                (
                    work_version_id,
                    document_chapter_id,
                    source_chapter_id,
                    source_version_id,
                    chapter_position,
                    snapshot_hash,
                ),
            )

    def _store_memory_snapshots(
        self,
        connection: Any,
        *,
        work_version_id: str,
        chapter_ids: Mapping[int, str],
        snapshots: list[Mapping[str, Any]],
    ) -> None:
        for snapshot in snapshots:
            try:
                chapter_position = int(snapshot["chapter_position"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("故事记忆快照缺少有效章节位置") from exc
            document_chapter_id = chapter_ids.get(chapter_position)
            payload = snapshot.get("payload")
            keywords = snapshot.get("keywords") or []
            summary = str(snapshot.get("summary") or "").strip()
            snapshot_hash = str(snapshot.get("content_hash") or "").strip()
            if not document_chapter_id:
                raise ValueError("故事记忆快照指向不存在的章节")
            if (
                not isinstance(payload, Mapping)
                or not isinstance(keywords, list)
                or not summary
                or not snapshot_hash
            ):
                raise ValueError("故事记忆快照内容不完整")
            connection.execute(
                """
                INSERT INTO work_version_story_memories(
                    id, work_version_id, document_chapter_id,
                    content_hash, summary, keywords_json,
                    payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    work_version_id,
                    document_chapter_id,
                    snapshot_hash,
                    summary,
                    json.dumps(keywords, ensure_ascii=False),
                    json.dumps(dict(payload), ensure_ascii=False),
                    self.now(),
                ),
            )

    def list(self, user_id: int) -> list[dict[str, Any]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT d.*,
                    (SELECT COUNT(*) FROM chapters c WHERE c.document_id=d.id)
                        AS chapter_count,
                    (SELECT j.id FROM analysis_jobs j WHERE j.document_id=d.id
                        ORDER BY j.created_at DESC LIMIT 1) AS latest_job_id,
                    (SELECT j.status FROM analysis_jobs j WHERE j.document_id=d.id
                        ORDER BY j.created_at DESC LIMIT 1) AS latest_job_status,
                    (SELECT j.completed_chapters FROM analysis_jobs j
                        WHERE j.document_id=d.id ORDER BY j.created_at DESC LIMIT 1)
                        AS completed_chapters
                FROM documents d
                WHERE d.user_id=?
                ORDER BY d.created_at DESC
                """,
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get(self, user_id: int, document_id: str) -> dict[str, Any] | None:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT d.*,
                    (SELECT COUNT(*) FROM chapters c WHERE c.document_id=d.id)
                        AS chapter_count,
                    (SELECT j.id FROM analysis_jobs j WHERE j.document_id=d.id
                        ORDER BY j.created_at DESC LIMIT 1) AS latest_job_id,
                    (SELECT j.status FROM analysis_jobs j WHERE j.document_id=d.id
                        ORDER BY j.created_at DESC LIMIT 1) AS latest_job_status
                FROM documents d
                WHERE d.id=? AND d.user_id=?
                """,
                (document_id, user_id),
            ).fetchone()
        return dict(row) if row else None

    def list_chapters(
        self, user_id: int, document_id: str, job_id: str | None = None
    ) -> list[dict[str, Any]]:
        parameters: list[Any] = [user_id, document_id]
        analysis_join = ""
        analysis_fields = (
            "NULL AS analysis_id, NULL AS analysis_status, NULL AS result_json"
        )
        if job_id:
            analysis_join = (
                "LEFT JOIN chapter_analyses a ON a.chapter_id=c.id AND a.job_id=?"
            )
            analysis_fields = (
                "a.id AS analysis_id, a.status AS analysis_status, a.result_json"
            )
            parameters = [job_id, user_id, document_id]
        with self.database.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT c.*, {analysis_fields}
                FROM chapters c
                JOIN documents d ON d.id=c.document_id
                {analysis_join}
                WHERE d.user_id=? AND d.id=?
                ORDER BY c.position
                """,
                parameters,
            ).fetchall()
        return [dict(row) for row in rows]
