from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any, Dict, List

from .db import Database, utc_now
from .structure_link_schema import (
    CAUSAL_RELATION_LABELS,
    CAUSAL_RELATION_TYPES,
)


def _load_list(raw: Any) -> List[str]:
    try:
        value = json.loads(str(raw or "[]"))
    except (TypeError, ValueError):
        return []
    if not isinstance(value, list):
        return []
    result: List[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result


def _clean(
    value: str,
    label: str,
    *,
    max_length: int,
    required: bool = False,
    min_length: int = 1,
) -> str:
    cleaned = str(value or "").strip()
    if required and len(cleaned) < min_length:
        raise ValueError(f"{label}至少需要 {min_length} 个字符")
    if len(cleaned) > max_length:
        raise ValueError(f"{label}不能超过 {max_length:,} 个字符")
    return cleaned


class StructureLinkService:
    """Manage author-authored causal commitments between planned chapters."""

    def __init__(self, database: Database):
        self.database = database

    def list_links(
        self,
        *,
        user_id: int,
        project_id: str,
        include_archived: bool = False,
    ) -> List[Dict[str, Any]]:
        status_clause = "" if include_archived else "AND link.status='active'"
        with self.database.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT link.*,
                       source.position AS source_position,
                       source.title AS source_title,
                       source.skeleton_role AS source_role,
                       source.skeleton_arc_titles_json AS source_arcs_json,
                       source.head_version_id
                           AS source_head_version_id,
                       target.position AS target_position,
                       target.title AS target_title,
                       target.skeleton_role AS target_role,
                       target.skeleton_arc_titles_json AS target_arcs_json,
                       target.head_version_id
                           AS target_head_version_id
                FROM novel_chapter_causal_links link
                JOIN novel_projects project ON project.id=link.project_id
                JOIN novel_chapters source
                  ON source.id=link.source_chapter_id
                JOIN novel_chapters target
                  ON target.id=link.target_chapter_id
                WHERE link.project_id=? AND project.user_id=?
                  {status_clause}
                ORDER BY target.position, source.position, link.created_at
                """,
                (project_id, user_id),
            ).fetchall()
        return [self._decode_link(row) for row in rows]

    def create_link(
        self,
        *,
        user_id: int,
        project_id: str,
        source_chapter_id: str,
        target_chapter_id: str,
        relation_type: str,
        cause_text: str,
        effect_text: str,
        author_note: str = "",
    ) -> Dict[str, Any]:
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                result = self._create_link_in_connection(
                    connection,
                    user_id=user_id,
                    project_id=project_id,
                    source_chapter_id=source_chapter_id,
                    target_chapter_id=target_chapter_id,
                    relation_type=relation_type,
                    cause_text=cause_text,
                    effect_text=effect_text,
                    author_note=author_note,
                )
            except Exception:
                connection.rollback()
                raise
            connection.commit()
        return result

    def _create_link_in_connection(
        self,
        connection,
        *,
        user_id: int,
        project_id: str,
        source_chapter_id: str,
        target_chapter_id: str,
        relation_type: str,
        cause_text: str,
        effect_text: str,
        author_note: str = "",
        link_id: str = "",
        now: str = "",
    ) -> Dict[str, Any]:
        """Create one link inside a caller-owned immediate transaction."""

        if relation_type not in CAUSAL_RELATION_TYPES:
            raise ValueError("请选择有效的因果关系类型")
        if source_chapter_id == target_chapter_id:
            raise ValueError("起因章和结果章不能是同一章")
        cause = _clean(
            cause_text,
            "起因",
            max_length=1200,
            required=True,
            min_length=4,
        )
        effect = _clean(
            effect_text,
            "结果",
            max_length=1200,
            required=True,
            min_length=4,
        )
        note = _clean(author_note, "作者备注", max_length=3000)
        link_id = link_id or uuid.uuid4().hex
        now = now or utc_now()
        project = connection.execute(
            """
            SELECT p.id,
                   COALESCE(MAX(
                       CASE
                         WHEN ch.head_version_id IS NOT NULL
                         THEN ch.position
                       END
                   ), 0) AS current_canonical_position
            FROM novel_projects p
            LEFT JOIN novel_chapters ch ON ch.project_id=p.id
            WHERE p.id=? AND p.user_id=?
            GROUP BY p.id
            """,
            (project_id, user_id),
        ).fetchone()
        if not project:
            raise ValueError("小说项目不存在")
        rows = connection.execute(
            """
            SELECT id, position, head_version_id
            FROM novel_chapters
            WHERE project_id=? AND id IN (?, ?)
            """,
            (
                project_id,
                source_chapter_id,
                target_chapter_id,
            ),
        ).fetchall()
        chapters = {str(row["id"]): row for row in rows}
        source = chapters.get(source_chapter_id)
        target = chapters.get(target_chapter_id)
        if not source or not target:
            raise ValueError("起因章或结果章不存在")
        source_position = int(source["position"])
        target_position = int(target["position"])
        current_position = int(project["current_canonical_position"] or 0)
        if source_position >= target_position:
            raise ValueError("结果章必须晚于起因章，因果链接不能倒流")
        if target["head_version_id"] or target_position <= current_position:
            raise ValueError("结果章必须位于当前正史边界之后")
        if (
            not source["head_version_id"]
            and source_position <= current_position
        ):
            raise ValueError(
                "起因章位于正史边界内但尚未确认，不能作为因果起点"
            )
        self._ensure_jobs_idle(
            connection,
            chapter_ids=[source_chapter_id, target_chapter_id],
        )
        try:
            connection.execute(
                """
                INSERT INTO novel_chapter_causal_links(
                    id, project_id, source_chapter_id,
                    target_chapter_id, relation_type, cause_text,
                    effect_text, author_note, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
                """,
                (
                    link_id,
                    project_id,
                    source_chapter_id,
                    target_chapter_id,
                    relation_type,
                    cause,
                    effect,
                    note,
                    now,
                    now,
                ),
            )
        except sqlite3.IntegrityError as exc:
            if "UNIQUE constraint failed" in str(exc):
                raise ValueError("相同的因果链接已经存在") from exc
            raise
        invalidation = self._invalidate_future_chapters(
            connection,
            project_id=project_id,
            chapter_ids=[source_chapter_id, target_chapter_id],
            now=now,
        )
        connection.execute(
            "UPDATE novel_projects SET updated_at=? WHERE id=?",
            (now, project_id),
        )
        return {
            "id": link_id,
            "project_id": project_id,
            "changed": True,
            **invalidation,
        }

    def archive_link(
        self,
        *,
        user_id: int,
        project_id: str,
        link_id: str,
    ) -> Dict[str, Any]:
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            link = connection.execute(
                """
                SELECT link.*
                FROM novel_chapter_causal_links link
                JOIN novel_projects project ON project.id=link.project_id
                WHERE link.id=? AND link.project_id=?
                  AND project.user_id=?
                """,
                (link_id, project_id, user_id),
            ).fetchone()
            if not link:
                connection.rollback()
                raise ValueError("因果链接不存在")
            if str(link["status"]) != "active":
                connection.rollback()
                raise ValueError("因果链接已经归档")
            chapter_ids = [
                str(link["source_chapter_id"]),
                str(link["target_chapter_id"]),
            ]
            self._ensure_jobs_idle(connection, chapter_ids=chapter_ids)
            connection.execute(
                """
                UPDATE novel_chapter_causal_links
                SET status='archived', archived_at=?, updated_at=?
                WHERE id=? AND status='active'
                """,
                (now, now, link_id),
            )
            invalidation = self._invalidate_future_chapters(
                connection,
                project_id=project_id,
                chapter_ids=chapter_ids,
                now=now,
            )
            connection.execute(
                "UPDATE novel_projects SET updated_at=? WHERE id=?",
                (now, project_id),
            )
            connection.commit()
        return {
            "id": link_id,
            "project_id": project_id,
            "changed": True,
            **invalidation,
        }

    @staticmethod
    def _ensure_jobs_idle(connection, *, chapter_ids: List[str]) -> None:
        placeholders = ",".join("?" for _ in chapter_ids)
        active = connection.execute(
            f"""
            SELECT 1
            FROM generation_jobs
            WHERE chapter_id IN ({placeholders})
              AND status IN ('queued', 'running')
            LIMIT 1
            """,
            tuple(chapter_ids),
        ).fetchone()
        if active:
            raise ValueError(
                "AI 正在处理相关章节，请等待任务完成后再修改因果链接"
            )

    @staticmethod
    def _invalidate_future_chapters(
        connection,
        *,
        project_id: str,
        chapter_ids: List[str],
        now: str,
    ) -> Dict[str, int]:
        unique_ids = list(dict.fromkeys(chapter_ids))
        placeholders = ",".join("?" for _ in unique_ids)
        rows = connection.execute(
            f"""
            SELECT ch.id AS chapter_id, cp.id AS plan_id,
                   cp.status AS plan_status
            FROM novel_chapters ch
            LEFT JOIN novel_chapter_plans cp ON cp.chapter_id=ch.id
            WHERE ch.project_id=?
              AND ch.head_version_id IS NULL
              AND ch.id IN ({placeholders})
            """,
            (project_id, *unique_ids),
        ).fetchall()
        reset_count = 0
        for row in rows:
            plan_id = str(row["plan_id"] or "")
            if plan_id:
                if str(row["plan_status"] or "") == "confirmed":
                    reset_count += 1
                connection.execute(
                    """
                    UPDATE novel_chapter_plans
                    SET status='draft', confirmed_at=NULL, updated_at=?
                    WHERE id=?
                    """,
                    (now, plan_id),
                )
            connection.execute(
                """
                UPDATE novel_chapters
                SET needs_recheck=1, updated_at=?
                WHERE id=?
                """,
                (now, str(row["chapter_id"])),
            )
        return {
            "affected_chapter_count": len(rows),
            "reset_task_card_count": reset_count,
        }

    @staticmethod
    def _decode_link(row) -> Dict[str, Any]:
        item = dict(row)
        source_arcs = _load_list(item.pop("source_arcs_json", "[]"))
        target_arcs = _load_list(item.pop("target_arcs_json", "[]"))
        shared_arcs = sorted(set(source_arcs) & set(target_arcs))
        item["source_arc_titles"] = source_arcs
        item["target_arc_titles"] = target_arcs
        item["shared_arc_titles"] = shared_arcs
        item["cross_line"] = bool(
            source_arcs and target_arcs and not shared_arcs
        )
        item["relation_label"] = CAUSAL_RELATION_LABELS.get(
            str(item.get("relation_type") or ""),
            str(item.get("relation_type") or ""),
        )
        item["source_is_canonical"] = bool(
            item.pop("source_head_version_id", None)
        )
        item["target_is_canonical"] = bool(
            item.pop("target_head_version_id", None)
        )
        item["planning_status"] = (
            "realized" if item["target_is_canonical"] else "active"
        )
        return item
