"""Conversation ownership, history queries, and user preferences."""

from __future__ import annotations

import re
import uuid
from typing import Any

from .assistant_result import decode_agent_step, decode_message, decode_tool_call
from .db import Database, utc_now
from .model_routing import normalize_quality_mode


CONVERSATION_SCOPES = frozenset(
    {"project", "chapter", "document", "reference_chapter"}
)
def clean_conversation_title(value: str, fallback: str) -> str:
    title = re.sub(r"\s+", " ", value).strip()
    return (title or fallback)[:120]


class AssistantConversationRepository:
    def __init__(self, database: Database):
        self.database = database

    def create(
        self,
        *,
        user_id: int,
        scope_type: str,
        title: str,
        project_id: str | None = None,
        document_id: str | None = None,
        novel_chapter_id: str | None = None,
        reference_chapter_id: str | None = None,
    ) -> str:
        if scope_type not in CONVERSATION_SCOPES:
            raise ValueError("不支持的对话范围")
        conversation_id = uuid.uuid4().hex
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            fallback = "新的创作讨论"
            if scope_type in {"project", "chapter"}:
                project = connection.execute(
                    "SELECT id, title FROM novel_projects WHERE id=? AND user_id=?",
                    (project_id, user_id),
                ).fetchone()
                if not project:
                    connection.rollback()
                    raise ValueError("小说项目不存在")
                document_id = None
                reference_chapter_id = None
                if scope_type == "chapter":
                    chapter = connection.execute(
                        "SELECT id, title FROM novel_chapters WHERE id=? AND project_id=?",
                        (novel_chapter_id, project_id),
                    ).fetchone()
                    if not chapter:
                        connection.rollback()
                        raise ValueError("章节不存在")
                    fallback = f"讨论《{chapter['title'] or '未命名章节'}》"
                else:
                    novel_chapter_id = None
                    fallback = f"讨论《{project['title']}》"
            else:
                document = connection.execute(
                    "SELECT id, title FROM documents WHERE id=? AND user_id=?",
                    (document_id, user_id),
                ).fetchone()
                if not document:
                    connection.rollback()
                    raise ValueError("拆书文档不存在")
                project_id = None
                novel_chapter_id = None
                if scope_type == "reference_chapter":
                    chapter = connection.execute(
                        "SELECT id, title FROM chapters WHERE id=? AND document_id=?",
                        (reference_chapter_id, document_id),
                    ).fetchone()
                    if not chapter:
                        connection.rollback()
                        raise ValueError("参考章节不存在")
                    fallback = f"拆解《{chapter['title']}》"
                else:
                    reference_chapter_id = None
                    fallback = f"拆解《{document['title']}》"
            preference = connection.execute(
                "SELECT default_quality_mode FROM user_model_preferences WHERE user_id=?",
                (user_id,),
            ).fetchone()
            quality_mode = normalize_quality_mode(
                preference["default_quality_mode"] if preference else "standard"
            )
            connection.execute(
                """
                INSERT INTO assistant_conversations(
                    id, user_id, scope_type, project_id, document_id,
                    novel_chapter_id, reference_chapter_id, title,
                    quality_mode, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    user_id,
                    scope_type,
                    project_id,
                    document_id,
                    novel_chapter_id,
                    reference_chapter_id,
                    clean_conversation_title(title, fallback),
                    quality_mode,
                    now,
                    now,
                ),
            )
            connection.commit()
        return conversation_id

    def list_project(self, *, user_id: int, project_id: str) -> list[dict[str, Any]]:
        return self._list(user_id=user_id, field="project_id", value=project_id)

    def list_document(
        self, *, user_id: int, document_id: str
    ) -> list[dict[str, Any]]:
        return self._list(user_id=user_id, field="document_id", value=document_id)

    def _list(
        self, *, user_id: int, field: str, value: str
    ) -> list[dict[str, Any]]:
        if field not in {"project_id", "document_id"}:
            raise ValueError("不支持的对话索引")
        with self.database.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT c.*, nch.title AS novel_chapter_title,
                       nch.position AS novel_chapter_position,
                       rch.title AS reference_chapter_title,
                       rch.position AS reference_chapter_position,
                       (SELECT m.status FROM assistant_messages m
                        WHERE m.conversation_id=c.id AND m.role='assistant'
                        ORDER BY m.created_at DESC, m.rowid DESC LIMIT 1)
                           AS latest_status
                FROM assistant_conversations c
                LEFT JOIN novel_chapters nch ON nch.id=c.novel_chapter_id
                LEFT JOIN chapters rch ON rch.id=c.reference_chapter_id
                WHERE c.user_id=? AND c.{field}=?
                ORDER BY c.updated_at DESC, c.rowid DESC LIMIT 80
                """,
                (user_id, value),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete(
        self, *, user_id: int, conversation_id: str
    ) -> dict[str, Any] | None:
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT id, scope_type, project_id, document_id,
                       novel_chapter_id, reference_chapter_id
                FROM assistant_conversations WHERE id=? AND user_id=?
                """,
                (conversation_id, user_id),
            ).fetchone()
            if not row:
                connection.rollback()
                return None
            active = connection.execute(
                """
                SELECT 1 FROM assistant_messages
                WHERE conversation_id=? AND role='assistant'
                  AND status IN ('queued', 'running') LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
            if active:
                connection.rollback()
                raise ValueError("AI 正在回复，完成后才能删除这段对话")
            cursor = connection.execute(
                "DELETE FROM assistant_conversations WHERE id=? AND user_id=?",
                (conversation_id, user_id),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return None
            connection.commit()
        return dict(row)

    def get(
        self, *, user_id: int, conversation_id: str
    ) -> dict[str, Any] | None:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT c.*, p.title AS project_title,
                       d.title AS document_title,
                       nch.title AS novel_chapter_title,
                       nch.position AS novel_chapter_position,
                       rch.title AS reference_chapter_title,
                       rch.position AS reference_chapter_position
                FROM assistant_conversations c
                LEFT JOIN novel_projects p ON p.id=c.project_id
                LEFT JOIN documents d ON d.id=c.document_id
                LEFT JOIN novel_chapters nch ON nch.id=c.novel_chapter_id
                LEFT JOIN chapters rch ON rch.id=c.reference_chapter_id
                WHERE c.id=? AND c.user_id=?
                """,
                (conversation_id, user_id),
            ).fetchone()
            if not row:
                return None
            messages = connection.execute(
                """
                SELECT m.*, q.source_type,
                       q.project_id AS quote_project_id,
                       q.document_id AS quote_document_id,
                       q.novel_chapter_id AS quote_novel_chapter_id,
                       q.version_id AS quote_version_id,
                       q.reference_chapter_id AS quote_reference_chapter_id,
                       q.start_offset AS quote_start_offset,
                       q.end_offset AS quote_end_offset,
                       q.quote_text, q.content_hash AS quote_content_hash,
                       q.source_label AS quote_source_label
                FROM assistant_messages m
                LEFT JOIN assistant_message_quotes q ON q.message_id=m.id
                WHERE m.conversation_id=?
                ORDER BY m.created_at, m.rowid
                """,
                (conversation_id,),
            ).fetchall()
            tool_rows = connection.execute(
                """
                SELECT tc.* FROM assistant_tool_calls tc
                JOIN assistant_messages m ON m.id=tc.assistant_message_id
                WHERE m.conversation_id=?
                ORDER BY tc.assistant_message_id, tc.sequence
                """,
                (conversation_id,),
            ).fetchall()
            step_rows = connection.execute(
                """
                SELECT step.* FROM assistant_agent_steps step
                JOIN assistant_messages m ON m.id=step.assistant_message_id
                WHERE m.conversation_id=?
                ORDER BY step.assistant_message_id, step.sequence
                """,
                (conversation_id,),
            ).fetchall()
        tool_calls: dict[str, list[dict[str, Any]]] = {}
        for row_item in tool_rows:
            tool_calls.setdefault(str(row_item["assistant_message_id"]), []).append(
                decode_tool_call(dict(row_item))
            )
        steps: dict[str, list[dict[str, Any]]] = {}
        for row_item in step_rows:
            steps.setdefault(str(row_item["assistant_message_id"]), []).append(
                decode_agent_step(dict(row_item))
            )
        result = dict(row)
        result["messages"] = []
        for message in messages:
            decoded = decode_message(dict(message))
            decoded["tool_calls"] = tool_calls.get(str(message["id"]), [])
            decoded["agent_steps"] = steps.get(str(message["id"]), [])
            result["messages"].append(decoded)
        result["pending_message_id"] = next(
            (
                str(message["id"])
                for message in reversed(result["messages"])
                if message["role"] == "assistant"
                and message["status"] in {"queued", "running"}
            ),
            None,
        )
        return result

    def set_quality_mode(
        self, *, user_id: int, conversation_id: str, quality_mode: str
    ) -> str:
        selected = normalize_quality_mode(quality_mode)
        if str(quality_mode or "").strip().lower() != selected:
            raise ValueError("不支持的模型强度")
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                """
                SELECT 1 FROM assistant_messages
                WHERE conversation_id=? AND role='assistant'
                  AND status IN ('queued', 'running') LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
            if active:
                connection.rollback()
                raise ValueError("AI 正在回复，请完成后再切换模型强度")
            cursor = connection.execute(
                """
                UPDATE assistant_conversations SET quality_mode=?, updated_at=?
                WHERE id=? AND user_id=?
                """,
                (selected, now, conversation_id, user_id),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise ValueError("对话不存在")
            connection.execute(
                """
                INSERT INTO user_model_preferences(
                    user_id, default_quality_mode, created_at, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    default_quality_mode=excluded.default_quality_mode,
                    updated_at=excluded.updated_at
                """,
                (user_id, selected, now, now),
            )
            connection.commit()
        return selected
