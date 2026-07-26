from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .agent_capabilities import (
    CREATE_CANDIDATE_DRAFT,
    PROPOSE_SETTINGS_PATCH,
    PROPOSE_STORY_PLAN,
    PROPOSE_TEXT_PATCH,
    agent_capabilities,
    agent_manifest,
    normalize_requested_agent_role,
    resolve_agent_dispatch,
)
from .agent_loop_schema import AssistantIntentDecision
from .assistant_chat_schema import (
    AssistantChatResponse,
    AssistantDraftProposal,
    AssistantSettingsPatch,
    AssistantStoryPlanProposal,
)
from .context_compiler import build_writing_context_snapshot
from .db import Database, utc_after, utc_now
from .quality_audit import effective_char_count
from .story_planning_service import StoryPlanningService


CONVERSATION_SCOPES = {
    "project",
    "chapter",
    "document",
    "reference_chapter",
}

SETTING_FIELD_LABELS = {
    "title": "书名",
    "genre": "题材",
    "premise": "故事核心",
    "story_promise": "作品承诺",
    "target_audience": "目标读者",
    "core_appeal": "核心吸引力",
    "ending_constraint": "结局与结构约束",
    "world_setting": "世界规则",
    "style_guide": "叙事风格规范",
    "ai_instructions": "本书 AI 协作补充指令",
    "point_of_view": "叙事视角",
    "target_chapter_chars": "默认单章篇幅",
    "planning_horizon": "规划章节范围",
}

def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _load_json(value: Any, fallback: Any) -> Any:
    try:
        return json.loads(str(value))
    except (TypeError, ValueError):
        return fallback


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_text(path: Any) -> str:
    try:
        return Path(str(path)).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ""


def _atomic_write(path: Path, content: str, token: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.parent / f".{path.name}.{token}.tmp"
    temporary.write_text(content, encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)
    path.chmod(0o600)


def _clean_title(value: str, fallback: str) -> str:
    title = re.sub(r"\s+", " ", value).strip()
    return (title or fallback)[:120]


def _bounded_source(
    text: str,
    *,
    selected_start: Optional[int] = None,
    selected_end: Optional[int] = None,
    max_chars: int = 30_000,
) -> tuple[str, int]:
    if len(text) <= max_chars:
        return text, 0
    if selected_start is None or selected_end is None:
        return text[-max_chars:], len(text) - max_chars
    quote_length = max(0, selected_end - selected_start)
    padding = max(0, max_chars - quote_length)
    left = max(0, selected_start - padding // 2)
    right = min(len(text), left + max_chars)
    left = max(0, right - max_chars)
    return text[left:right], left


def _resolve_quote_offsets(
    text: str,
    *,
    start: int,
    end: int,
    quote_text: str,
) -> tuple[int, int]:
    """Resolve a browser selection against the verified immutable source.

    Browser selection offsets are UTF-16 based while Python slices count
    Unicode code points. Once the complete source hash has been verified, the
    quoted text itself is the authoritative selection. Prefer an exact slice;
    otherwise choose the matching occurrence nearest the submitted offset.
    """
    if end <= len(text) and text[start:end] == quote_text:
        return start, end
    occurrences: list[int] = []
    position = text.find(quote_text)
    while position >= 0:
        occurrences.append(position)
        position = text.find(quote_text, position + 1)
    if not occurrences:
        raise ValueError("引用位置与正文版本不一致")
    resolved_start = min(occurrences, key=lambda item: abs(item - start))
    return resolved_start, resolved_start + len(quote_text)


class AssistantChatService:
    def __init__(
        self,
        database: Database,
        novels_dir: Path,
        documents_dir: Path,
    ):
        self.database = database
        self.novels_dir = novels_dir
        self.documents_dir = documents_dir

    def _create_empty_chapter(
        self, *, user_id: int, project_id: str
    ) -> str:
        chapter_id = uuid.uuid4().hex
        chapter_dir = (
            self.novels_dir
            / str(user_id)
            / project_id
            / "chapters"
            / chapter_id
        )
        try:
            (chapter_dir / "versions").mkdir(
                parents=True, exist_ok=False, mode=0o700
            )
            os.chmod(chapter_dir, 0o700)
            os.chmod(chapter_dir / "versions", 0o700)
            content_path = chapter_dir / "content.txt"
            content_path.write_text("", encoding="utf-8")
            content_path.chmod(0o600)
            self.database.add_novel_chapter(
                user_id=user_id,
                project_id=project_id,
                chapter_id=chapter_id,
                title="",
                outline="",
                key_points="",
                content_path=content_path,
            )
        except Exception:
            shutil.rmtree(chapter_dir, ignore_errors=True)
            raise
        return chapter_id

    def _bind_conversation_to_chapter(
        self,
        *,
        user_id: int,
        conversation_id: str,
        project_id: str,
        chapter_id: str,
    ) -> None:
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            chapter = connection.execute(
                """
                SELECT 1 FROM novel_chapters ch
                JOIN novel_projects p ON p.id=ch.project_id
                WHERE ch.id=? AND ch.project_id=? AND p.user_id=?
                """,
                (chapter_id, project_id, user_id),
            ).fetchone()
            if not chapter:
                connection.rollback()
                raise ValueError("准备写作的章节不存在")
            cursor = connection.execute(
                """
                UPDATE assistant_conversations
                SET scope_type='chapter', novel_chapter_id=?, updated_at=?
                WHERE id=? AND user_id=? AND project_id=?
                  AND scope_type='project'
                """,
                (
                    chapter_id,
                    utc_now(),
                    conversation_id,
                    user_id,
                    project_id,
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise ValueError("对话范围已经变化，请重新发送")
            connection.commit()

    def _prepare_project_writing_scope(
        self,
        *,
        user_id: int,
        conversation: Mapping[str, Any],
        target_chapter_id: str = "",
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        if str(conversation.get("scope_type") or "") != "project":
            return dict(conversation), {}
        project_id = str(conversation.get("project_id") or "")
        chapters = self.database.list_novel_chapters(
            user_id, project_id
        )
        created = False
        clean_target = str(target_chapter_id or "").strip()
        if clean_target:
            selected = next(
                (
                    item
                    for item in chapters
                    if str(item.get("id") or "") == clean_target
                ),
                None,
            )
            if not selected:
                raise ValueError("模型选择的目标章节不存在")
            chapter_id = clean_target
        elif not chapters:
            chapter_id = self._create_empty_chapter(
                user_id=user_id, project_id=project_id
            )
            created = True
        elif len(chapters) == 1:
            chapter_id = str(chapters[0]["id"])
        else:
            raise ValueError(
                "作品已有多个章节，请先从目录打开目标章节，再让 AI 创作；"
                "如果要新建下一章，请先在目录中创建空白章节。"
            )
        self._bind_conversation_to_chapter(
            user_id=user_id,
            conversation_id=str(conversation["id"]),
            project_id=project_id,
            chapter_id=chapter_id,
        )
        prepared = self.get_conversation(
            user_id=user_id,
            conversation_id=str(conversation["id"]),
        )
        if not prepared:
            raise ValueError("切换章节写作范围失败")
        return prepared, {
            "action": (
                "created_empty_chapter"
                if created
                else "selected_existing_chapter"
            ),
            "chapter_id": chapter_id,
        }

    def _project_settings_ready(
        self,
        *,
        user_id: int,
        project_id: str,
    ) -> bool:
        """Return whether a new project has enough foundation to draft prose.

        A project with existing written content is already onboarded, even if
        its newer setting fields are still sparse.  For an unwritten project,
        require a premise plus at least one supporting setting (or a genre or
        character) before allowing the writer role.  This keeps a pasted
        script in the setting-discussion phase instead of silently creating a
        first chapter.
        """
        project = self.database.get_novel_project(user_id, project_id)
        if not project:
            return True
        if int(project.get("total_chars") or 0) > 0:
            return True
        if not str(project.get("premise") or "").strip():
            return False
        supporting_fields = (
            "genre",
            "story_promise",
            "target_audience",
            "core_appeal",
            "ending_constraint",
            "world_setting",
            "style_guide",
            "ai_instructions",
        )
        if any(str(project.get(key) or "").strip() for key in supporting_fields):
            return True
        return bool(self.database.list_novel_characters(user_id, project_id))

    def create_conversation(
        self,
        *,
        user_id: int,
        scope_type: str,
        title: str,
        project_id: Optional[str] = None,
        document_id: Optional[str] = None,
        novel_chapter_id: Optional[str] = None,
        reference_chapter_id: Optional[str] = None,
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
                    """
                    SELECT id, title FROM novel_projects
                    WHERE id=? AND user_id=?
                    """,
                    (project_id, user_id),
                ).fetchone()
                if not project:
                    connection.rollback()
                    raise ValueError("小说项目不存在")
                document_id = None
                reference_chapter_id = None
                if scope_type == "chapter":
                    chapter = connection.execute(
                        """
                        SELECT id, title FROM novel_chapters
                        WHERE id=? AND project_id=?
                        """,
                        (novel_chapter_id, project_id),
                    ).fetchone()
                    if not chapter:
                        connection.rollback()
                        raise ValueError("章节不存在")
                    fallback = (
                        f"讨论《{chapter['title'] or '未命名章节'}》"
                    )
                else:
                    novel_chapter_id = None
                    fallback = f"讨论《{project['title']}》"
            else:
                document = connection.execute(
                    """
                    SELECT id, title FROM documents
                    WHERE id=? AND user_id=?
                    """,
                    (document_id, user_id),
                ).fetchone()
                if not document:
                    connection.rollback()
                    raise ValueError("拆书文档不存在")
                project_id = None
                novel_chapter_id = None
                if scope_type == "reference_chapter":
                    chapter = connection.execute(
                        """
                        SELECT id, title FROM chapters
                        WHERE id=? AND document_id=?
                        """,
                        (reference_chapter_id, document_id),
                    ).fetchone()
                    if not chapter:
                        connection.rollback()
                        raise ValueError("参考章节不存在")
                    fallback = f"拆解《{chapter['title']}》"
                else:
                    reference_chapter_id = None
                    fallback = f"拆解《{document['title']}》"
            connection.execute(
                """
                INSERT INTO assistant_conversations(
                    id, user_id, scope_type, project_id, document_id,
                    novel_chapter_id, reference_chapter_id, title,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    user_id,
                    scope_type,
                    project_id,
                    document_id,
                    novel_chapter_id,
                    reference_chapter_id,
                    _clean_title(title, fallback),
                    now,
                    now,
                ),
            )
            connection.commit()
        return conversation_id

    def list_project_conversations(
        self, *, user_id: int, project_id: str
    ) -> List[Dict[str, Any]]:
        return self._list_conversations(
            user_id=user_id,
            field="project_id",
            value=project_id,
        )

    def list_document_conversations(
        self, *, user_id: int, document_id: str
    ) -> List[Dict[str, Any]]:
        return self._list_conversations(
            user_id=user_id,
            field="document_id",
            value=document_id,
        )

    def _list_conversations(
        self, *, user_id: int, field: str, value: str
    ) -> List[Dict[str, Any]]:
        if field not in {"project_id", "document_id"}:
            raise ValueError("不支持的对话索引")
        with self.database.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT c.*,
                       nch.title AS novel_chapter_title,
                       nch.position AS novel_chapter_position,
                       rch.title AS reference_chapter_title,
                       rch.position AS reference_chapter_position,
                       (
                           SELECT m.status FROM assistant_messages m
                           WHERE m.conversation_id=c.id
                             AND m.role='assistant'
                           ORDER BY m.created_at DESC, m.rowid DESC
                           LIMIT 1
                       ) AS latest_status
                FROM assistant_conversations c
                LEFT JOIN novel_chapters nch
                  ON nch.id=c.novel_chapter_id
                LEFT JOIN chapters rch
                  ON rch.id=c.reference_chapter_id
                WHERE c.user_id=? AND c.{field}=?
                ORDER BY c.updated_at DESC, c.rowid DESC
                LIMIT 80
                """,
                (user_id, value),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_conversation(
        self, *, user_id: int, conversation_id: str
    ) -> Optional[Dict[str, Any]]:
        """Delete an owned conversation after all pending work has stopped.

        Generated manuscript versions are intentionally not deleted here.
        They are part of the novel's version history, not merely chat output.
        """
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT id, scope_type, project_id, document_id,
                       novel_chapter_id, reference_chapter_id
                FROM assistant_conversations
                WHERE id=? AND user_id=?
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
                  AND status IN ('queued', 'running')
                LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
            if active:
                connection.rollback()
                raise ValueError("AI 正在回复，完成后才能删除这段对话")
            cursor = connection.execute(
                """
                DELETE FROM assistant_conversations
                WHERE id=? AND user_id=?
                """,
                (conversation_id, user_id),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return None
            connection.commit()
        return dict(row)

    def branch_conversation_from_message(
        self,
        *,
        user_id: int,
        message_id: str,
        replacement_question: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create an append-only branch before one user turn.

        Editing a user message or regenerating its assistant response must not
        mutate the original audit trail. Completed plain-text history before
        that turn is copied into a new conversation, while generated drafts,
        applied-version links and tool traces remain attached only to the
        original messages.
        """
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            source = connection.execute(
                """
                SELECT m.id AS source_message_id,
                       m.role AS source_role,
                       m.status AS source_status,
                       m.parent_user_message_id,
                       c.id AS source_conversation_id,
                       c.scope_type, c.project_id, c.document_id,
                       c.novel_chapter_id, c.reference_chapter_id,
                       c.title
                FROM assistant_messages m
                JOIN assistant_conversations c
                  ON c.id=m.conversation_id
                WHERE m.id=? AND c.user_id=?
                """,
                (message_id, user_id),
            ).fetchone()
            if not source:
                connection.rollback()
                raise ValueError("对话消息不存在")
            if (
                str(source["source_role"]) == "assistant"
                and str(source["source_status"]) in {"queued", "running"}
            ):
                connection.rollback()
                raise ValueError("AI 正在回复，暂时不能重新生成")
            target_user_message_id = (
                str(source["parent_user_message_id"] or "")
                if str(source["source_role"]) == "assistant"
                else str(source["source_message_id"])
            )
            target = connection.execute(
                """
                SELECT m.rowid AS message_rowid, m.content,
                       q.source_type,
                       q.project_id AS quote_project_id,
                       q.document_id AS quote_document_id,
                       q.novel_chapter_id AS quote_novel_chapter_id,
                       q.version_id AS quote_version_id,
                       q.reference_chapter_id AS quote_reference_chapter_id,
                       q.start_offset AS quote_start_offset,
                       q.end_offset AS quote_end_offset,
                       q.quote_text, q.content_hash AS quote_content_hash
                FROM assistant_messages m
                LEFT JOIN assistant_message_quotes q
                  ON q.message_id=m.id
                WHERE m.id=? AND m.conversation_id=?
                  AND m.role='user'
                """,
                (
                    target_user_message_id,
                    source["source_conversation_id"],
                ),
            ).fetchone()
            if not target:
                connection.rollback()
                raise ValueError("原问题不存在，无法建立分支")
            question = (
                replacement_question
                if replacement_question is not None
                else str(target["content"])
            )
            clean_question = str(question).strip()
            if not clean_question:
                connection.rollback()
                raise ValueError("修改后的消息不能为空")
            if len(clean_question) > 8_000:
                connection.rollback()
                raise ValueError("单条消息不能超过 8,000 个字符")
            active = connection.execute(
                """
                SELECT 1
                FROM assistant_messages m
                JOIN assistant_conversations c
                  ON c.id=m.conversation_id
                WHERE c.user_id=? AND m.role='assistant'
                  AND m.status IN ('queued', 'running')
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
            if active:
                connection.rollback()
                raise ValueError("上一条对话回复仍在生成，请稍候")

            history = connection.execute(
                """
                SELECT m.*
                FROM assistant_messages m
                WHERE m.conversation_id=? AND m.rowid<?
                  AND m.status='completed'
                ORDER BY m.rowid
                """,
                (
                    source["source_conversation_id"],
                    target["message_rowid"],
                ),
            ).fetchall()
            conversation_id = uuid.uuid4().hex
            now = utc_now()
            branch_title = _clean_title(
                f"{source['title']} · 分支", "新的创作讨论"
            )
            connection.execute(
                """
                INSERT INTO assistant_conversations(
                    id, user_id, scope_type, project_id, document_id,
                    novel_chapter_id, reference_chapter_id, title,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    user_id,
                    source["scope_type"],
                    source["project_id"],
                    source["document_id"],
                    source["novel_chapter_id"],
                    source["reference_chapter_id"],
                    branch_title,
                    now,
                    now,
                ),
            )
            message_ids: Dict[str, str] = {}
            for original in history:
                original_id = str(original["id"])
                role = str(original["role"])
                parent_id = None
                if role == "assistant":
                    parent_id = message_ids.get(
                        str(original["parent_user_message_id"] or "")
                    )
                    if not parent_id:
                        continue
                copied_id = uuid.uuid4().hex
                message_ids[original_id] = copied_id
                response_json = (
                    _json(
                        {
                            "branch_copy": True,
                            "source_message_id": original_id,
                        }
                    )
                    if role == "assistant"
                    else None
                )
                connection.execute(
                    """
                    INSERT INTO assistant_messages(
                        id, conversation_id, role, content,
                        parent_user_message_id, status, provider, model,
                        credential_source, response_json, input_tokens,
                        output_tokens, created_at, started_at, finished_at
                    ) VALUES (?, ?, ?, ?, ?, 'completed', ?, ?, ?, ?, 0, 0,
                              ?, NULL, ?)
                    """,
                    (
                        copied_id,
                        conversation_id,
                        role,
                        original["content"],
                        parent_id,
                        original["provider"],
                        original["model"],
                        original["credential_source"],
                        response_json,
                        original["created_at"],
                        original["finished_at"] or original["created_at"],
                    ),
                )
                if role == "user":
                    quote_row = connection.execute(
                        """
                        SELECT * FROM assistant_message_quotes
                        WHERE message_id=?
                        """,
                        (original_id,),
                    ).fetchone()
                    if quote_row:
                        connection.execute(
                            """
                            INSERT INTO assistant_message_quotes(
                                id, message_id, source_type, project_id,
                                document_id, novel_chapter_id, version_id,
                                reference_chapter_id, start_offset,
                                end_offset, quote_text, content_hash,
                                source_label, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                uuid.uuid4().hex,
                                copied_id,
                                quote_row["source_type"],
                                quote_row["project_id"],
                                quote_row["document_id"],
                                quote_row["novel_chapter_id"],
                                quote_row["version_id"],
                                quote_row["reference_chapter_id"],
                                quote_row["start_offset"],
                                quote_row["end_offset"],
                                quote_row["quote_text"],
                                quote_row["content_hash"],
                                quote_row["source_label"],
                                quote_row["created_at"],
                            ),
                        )
            connection.commit()

        quote_payload = None
        if target["quote_text"]:
            quote_payload = {
                "source_type": target["source_type"],
                "project_id": target["quote_project_id"],
                "document_id": target["quote_document_id"],
                "novel_chapter_id": target[
                    "quote_novel_chapter_id"
                ],
                "version_id": target["quote_version_id"],
                "reference_chapter_id": target[
                    "quote_reference_chapter_id"
                ],
                "start_offset": target["quote_start_offset"],
                "end_offset": target["quote_end_offset"],
                "quote_text": target["quote_text"],
                "content_hash": target["quote_content_hash"],
            }
        return {
            "conversation_id": conversation_id,
            "scope_type": str(source["scope_type"]),
            "project_id": source["project_id"],
            "document_id": source["document_id"],
            "novel_chapter_id": source["novel_chapter_id"],
            "reference_chapter_id": source["reference_chapter_id"],
            "question": clean_question,
            "quote": quote_payload,
        }

    def get_conversation(
        self, *, user_id: int, conversation_id: str
    ) -> Optional[Dict[str, Any]]:
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
                LEFT JOIN novel_chapters nch
                  ON nch.id=c.novel_chapter_id
                LEFT JOIN chapters rch
                  ON rch.id=c.reference_chapter_id
                WHERE c.id=? AND c.user_id=?
                """,
                (conversation_id, user_id),
            ).fetchone()
            if not row:
                return None
            messages = connection.execute(
                """
                SELECT m.*, q.source_type, q.project_id AS quote_project_id,
                       q.document_id AS quote_document_id,
                       q.novel_chapter_id AS quote_novel_chapter_id,
                       q.version_id AS quote_version_id,
                       q.reference_chapter_id AS quote_reference_chapter_id,
                       q.start_offset AS quote_start_offset,
                       q.end_offset AS quote_end_offset,
                       q.quote_text, q.content_hash AS quote_content_hash,
                       q.source_label AS quote_source_label
                FROM assistant_messages m
                LEFT JOIN assistant_message_quotes q
                  ON q.message_id=m.id
                WHERE m.conversation_id=?
                ORDER BY m.created_at, m.rowid
                LIMIT 300
                """,
                (conversation_id,),
            ).fetchall()
            tool_rows = connection.execute(
                """
                SELECT tc.*
                FROM assistant_tool_calls tc
                JOIN assistant_messages m
                  ON m.id=tc.assistant_message_id
                WHERE m.conversation_id=?
                ORDER BY tc.assistant_message_id, tc.sequence
                """,
                (conversation_id,),
            ).fetchall()
            step_rows = connection.execute(
                """
                SELECT step.*
                FROM assistant_agent_steps step
                JOIN assistant_messages m
                  ON m.id=step.assistant_message_id
                WHERE m.conversation_id=?
                ORDER BY step.assistant_message_id, step.sequence
                """,
                (conversation_id,),
            ).fetchall()
        tool_calls_by_message: dict[str, list[dict[str, Any]]] = {}
        for tool_row in tool_rows:
            decoded = self._decode_tool_call(dict(tool_row))
            tool_calls_by_message.setdefault(
                str(tool_row["assistant_message_id"]), []
            ).append(decoded)
        steps_by_message: dict[str, list[dict[str, Any]]] = {}
        for step_row in step_rows:
            decoded = self._decode_agent_step(dict(step_row))
            steps_by_message.setdefault(
                str(step_row["assistant_message_id"]), []
            ).append(decoded)
        result = dict(row)
        result["messages"] = []
        for message in messages:
            decoded = self._decode_message(dict(message))
            decoded["tool_calls"] = tool_calls_by_message.get(
                str(message["id"]), []
            )
            decoded["agent_steps"] = steps_by_message.get(
                str(message["id"]), []
            )
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

    def queue_message(
        self,
        *,
        user_id: int,
        conversation_id: str,
        question: str,
        provider: str,
        model: str,
        credential_source: str,
        quote: Optional[Mapping[str, Any]] = None,
        agent_role: str = "auto",
        ui_surface: str = "",
        auto_commit: bool = False,
        max_jobs_per_day: Optional[int] = None,
    ) -> str:
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("请输入想和 AI 讨论的问题")
        if len(clean_question) > 8_000:
            raise ValueError("单条消息不能超过 8,000 个字符")
        if credential_source not in {"default", "personal"}:
            raise ValueError("不支持的 API 凭据来源")
        requested_agent_role = normalize_requested_agent_role(agent_role)
        conversation = self.get_conversation(
            user_id=user_id, conversation_id=conversation_id
        )
        if not conversation:
            raise ValueError("对话不存在")
        clean_ui_surface = str(ui_surface or "").strip().lower()
        if clean_ui_surface not in {
            "project",
            "settings",
            "chapter",
            "reference",
        }:
            clean_ui_surface = {
                "project": "project",
                "chapter": "chapter",
                "document": "reference",
                "reference_chapter": "reference",
            }.get(str(conversation.get("scope_type") or ""), "project")
        normalized_quote = (
            self._validate_quote(
                user_id=user_id,
                conversation=conversation,
                quote=quote,
            )
            if quote
            else None
        )
        settings_ready = True
        if str(conversation.get("scope_type") or "") in {
            "project",
            "chapter",
        }:
            settings_ready = self._project_settings_ready(
                user_id=user_id,
                project_id=str(conversation.get("project_id") or ""),
            )
        scope_preflight: Dict[str, Any] = {}
        if requested_agent_role == "auto":
            clean_agent_role = "advisor"
            dispatch_context: Dict[str, Any] = {
                "requested_role": "auto",
                "resolved_role": "pending",
                "intent": "pending",
                "reason": "model_intent_pending",
                "goal": "classify_intent",
                "settings_ready": settings_ready,
                "ui_surface": clean_ui_surface,
                "scope_preflight": {},
            }
            settings_prerequisite = False
        else:
            dispatch = resolve_agent_dispatch(
                requested_role=requested_agent_role,
                scope_type=str(conversation["scope_type"]),
                intent="discuss",
                has_quote=bool(normalized_quote),
                settings_ready=settings_ready,
            )
            clean_agent_role = dispatch.role
            if clean_agent_role == "writer":
                conversation, scope_preflight = (
                    self._prepare_project_writing_scope(
                        user_id=user_id,
                        conversation=conversation,
                    )
                )
            settings_prerequisite = dispatch.settings_prerequisite
            dispatch_context = {
                "requested_role": requested_agent_role,
                "resolved_role": clean_agent_role,
                "intent": dispatch.intent,
                "reason": (
                    str(scope_preflight.get("action"))
                    if scope_preflight
                    else dispatch.reason
                ),
                "goal": dispatch.goal,
                "settings_ready": settings_ready,
                "ui_surface": clean_ui_surface,
                "scope_preflight": scope_preflight,
            }
            if settings_prerequisite:
                dispatch_context["settings_prerequisite"] = True
        snapshot = self._build_context_snapshot(
            user_id=user_id,
            conversation=conversation,
            normalized_quote=normalized_quote,
            agent_role=clean_agent_role,
            ui_surface=clean_ui_surface,
            auto_commit=auto_commit,
            settings_prerequisite=settings_prerequisite,
        )
        snapshot["context"]["dispatch"] = dispatch_context
        user_message_id = uuid.uuid4().hex
        assistant_message_id = uuid.uuid4().hex
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            owner = connection.execute(
                """
                SELECT 1 FROM assistant_conversations
                WHERE id=? AND user_id=?
                """,
                (conversation_id, user_id),
            ).fetchone()
            if not owner:
                connection.rollback()
                raise ValueError("对话不存在")
            active = connection.execute(
                """
                SELECT 1
                FROM assistant_messages m
                JOIN assistant_conversations c
                  ON c.id=m.conversation_id
                WHERE c.user_id=? AND m.role='assistant'
                  AND m.status IN ('queued', 'running')
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
            if active:
                connection.rollback()
                raise ValueError("上一条对话回复仍在生成，请稍候")
            if credential_source == "personal":
                credential = connection.execute(
                    """
                    SELECT 1 FROM api_credentials
                    WHERE user_id=? AND provider=?
                    """,
                    (user_id, provider),
                ).fetchone()
                if not credential:
                    connection.rollback()
                    raise ValueError(
                        "所选模型服务 API Key 或凭据不存在，请重新配置"
                    )
            if max_jobs_per_day is not None:
                day_start = datetime.now(timezone.utc).replace(
                    hour=0, minute=0, second=0, microsecond=0
                ).isoformat(timespec="seconds")
                count = connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM assistant_messages m
                    JOIN assistant_conversations c
                      ON c.id=m.conversation_id
                    WHERE c.user_id=? AND m.role='assistant'
                      AND m.created_at>=?
                    """,
                    (user_id, day_start),
                ).fetchone()
                if int(count["count"] or 0) >= max_jobs_per_day:
                    connection.rollback()
                    raise ValueError(
                        f"今天已达到 {max_jobs_per_day} 个 AI 任务的上限"
                    )
            connection.execute(
                """
                INSERT INTO assistant_messages(
                    id, conversation_id, role, content, status,
                    provider, model, credential_source, created_at,
                    finished_at
                ) VALUES (?, ?, 'user', ?, 'completed', '', '', 'default',
                          ?, ?)
                """,
                (
                    user_message_id,
                    conversation_id,
                    clean_question,
                    now,
                    now,
                ),
            )
            if normalized_quote:
                connection.execute(
                    """
                    INSERT INTO assistant_message_quotes(
                        id, message_id, source_type, project_id, document_id,
                        novel_chapter_id, version_id, reference_chapter_id,
                        start_offset, end_offset, quote_text, content_hash,
                        source_label, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid.uuid4().hex,
                        user_message_id,
                        normalized_quote["source_type"],
                        normalized_quote.get("project_id"),
                        normalized_quote.get("document_id"),
                        normalized_quote.get("novel_chapter_id"),
                        normalized_quote.get("version_id"),
                        normalized_quote.get("reference_chapter_id"),
                        normalized_quote["start_offset"],
                        normalized_quote["end_offset"],
                        normalized_quote["quote_text"],
                        normalized_quote["content_hash"],
                        normalized_quote["source_label"],
                        now,
                    ),
                )
            connection.execute(
                """
                INSERT INTO assistant_messages(
                    id, conversation_id, role, parent_user_message_id,
                    status, provider, model, credential_source,
                    context_snapshot_json, created_at
                ) VALUES (?, ?, 'assistant', ?, 'queued', ?, ?, ?, ?, ?)
                """,
                (
                    assistant_message_id,
                    conversation_id,
                    user_message_id,
                    provider,
                    model,
                    credential_source,
                    _json(snapshot),
                    now,
                ),
            )
            message_count = connection.execute(
                """
                SELECT COUNT(*) AS count FROM assistant_messages
                WHERE conversation_id=? AND role='user'
                """,
                (conversation_id,),
            ).fetchone()
            if int(message_count["count"] or 0) == 1:
                generated_title = _clean_title(
                    clean_question.splitlines()[0], "新的创作讨论"
                )
                generated_title = generated_title[:42]
                connection.execute(
                    """
                    UPDATE assistant_conversations
                    SET title=?, updated_at=? WHERE id=?
                    """,
                    (generated_title, now, conversation_id),
                )
            else:
                connection.execute(
                    """
                    UPDATE assistant_conversations
                    SET updated_at=? WHERE id=?
                    """,
                    (now, conversation_id),
                )
            connection.commit()
        return assistant_message_id

    def resolve_pending_dispatch(
        self,
        *,
        message_id: str,
        claim_token: str,
        decision: AssistantIntentDecision,
    ) -> Dict[str, Any]:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT m.context_snapshot_json, m.parent_user_message_id,
                       m.conversation_id, c.user_id
                FROM assistant_messages m
                JOIN assistant_conversations c
                  ON c.id=m.conversation_id
                WHERE m.id=? AND m.role='assistant' AND m.status='running'
                  AND m.claim_token=?
                """,
                (message_id, claim_token),
            ).fetchone()
            if not row:
                raise ValueError("对话任务租约已经失效")
            quote_row = connection.execute(
                """
                SELECT * FROM assistant_message_quotes
                WHERE message_id=?
                """,
                (row["parent_user_message_id"],),
            ).fetchone()
        old_snapshot = _load_json(row["context_snapshot_json"], {})
        old_context = dict(old_snapshot.get("context") or {})
        old_dispatch = dict(old_context.get("dispatch") or {})
        ui_surface = str(
            old_dispatch.get("ui_surface")
            or old_context.get("ui_surface")
            or ""
        )
        if (
            str(old_dispatch.get("requested_role") or "") != "auto"
            or str(old_dispatch.get("resolved_role") or "") != "pending"
        ):
            return old_snapshot

        user_id = int(row["user_id"])
        conversation = self.get_conversation(
            user_id=user_id,
            conversation_id=str(row["conversation_id"]),
        )
        if not conversation:
            raise ValueError("对话不存在")
        scope_type = str(conversation.get("scope_type") or "")
        settings_ready = True
        if scope_type in {"project", "chapter"}:
            settings_ready = self._project_settings_ready(
                user_id=user_id,
                project_id=str(conversation.get("project_id") or ""),
            )
        dispatch = resolve_agent_dispatch(
            requested_role="auto",
            scope_type=scope_type,
            intent=decision.intent,
            has_quote=bool(quote_row),
            settings_ready=settings_ready,
            confidence=decision.confidence,
        )
        scope_preflight: Dict[str, Any] = {}
        routing_notice = ""
        if dispatch.role == "writer":
            try:
                conversation, scope_preflight = (
                    self._prepare_project_writing_scope(
                        user_id=user_id,
                        conversation=conversation,
                        target_chapter_id=(
                            decision.target_chapter_id or ""
                        ),
                    )
                )
            except ValueError as exc:
                dispatch = resolve_agent_dispatch(
                    requested_role="auto",
                    scope_type=scope_type,
                    intent="discuss",
                    has_quote=bool(quote_row),
                    settings_ready=settings_ready,
                    confidence=1,
                )
                routing_notice = str(exc)

        boundaries = dict(old_context.get("assistant_boundaries") or {})
        normalized_quote = (
            self._validate_quote(
                user_id=user_id,
                conversation=conversation,
                quote=dict(quote_row),
            )
            if quote_row
            else None
        )
        snapshot = self._build_context_snapshot(
            user_id=user_id,
            conversation=conversation,
            normalized_quote=normalized_quote,
            agent_role=dispatch.role,
            ui_surface=ui_surface,
            auto_commit=bool(
                boundaries.get("auto_commit_working_copy")
            ),
            settings_prerequisite=dispatch.settings_prerequisite,
        )
        dispatch_context: Dict[str, Any] = {
            "requested_role": "auto",
            "resolved_role": dispatch.role,
            "intent": dispatch.intent,
            "reason": (
                str(scope_preflight.get("action"))
                if scope_preflight
                else (
                    "target_chapter_required"
                    if routing_notice
                    else dispatch.reason
                )
            ),
            "goal": dispatch.goal,
            "settings_ready": settings_ready,
            "ui_surface": ui_surface,
            "confidence": decision.confidence,
            "classifier": decision.model_dump(mode="json"),
            "scope_preflight": scope_preflight,
        }
        if routing_notice:
            dispatch_context["routing_notice"] = routing_notice
        if dispatch.settings_prerequisite:
            dispatch_context["settings_prerequisite"] = True
        snapshot["context"]["dispatch"] = dispatch_context

        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE assistant_messages
                SET context_snapshot_json=?
                WHERE id=? AND role='assistant' AND status='running'
                  AND claim_token=?
                """,
                (_json(snapshot), message_id, claim_token),
            )
            connection.commit()
        if cursor.rowcount != 1:
            raise ValueError("对话任务租约已经失效")
        return snapshot

    def claim_next_message(self) -> Optional[Dict[str, Any]]:
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = utc_now()
            connection.execute(
                """
                UPDATE assistant_messages
                SET status='queued', started_at=NULL, claim_token=NULL,
                    lease_expires_at=NULL,
                    error='上一次处理租约已过期，已自动重新排队'
                WHERE role='assistant' AND status='running'
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at<=?
                """,
                (now,),
            )
            row = connection.execute(
                """
                SELECT m.*, c.user_id, c.scope_type, c.project_id,
                       c.document_id, c.novel_chapter_id,
                       c.reference_chapter_id
                FROM assistant_messages m
                JOIN assistant_conversations c
                  ON c.id=m.conversation_id
                WHERE m.role='assistant' AND m.status='queued'
                ORDER BY m.created_at, m.rowid
                LIMIT 1
                """
            ).fetchone()
            if not row:
                connection.commit()
                return None
            claim_token = uuid.uuid4().hex
            cursor = connection.execute(
                """
                UPDATE assistant_messages
                SET status='running', started_at=?, error=NULL,
                    claim_token=?, lease_expires_at=?
                WHERE id=? AND role='assistant' AND status='queued'
                """,
                (
                    now,
                    claim_token,
                    utc_after(30 * 60),
                    row["id"],
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return None
            connection.commit()
        item = dict(row)
        item["claim_token"] = claim_token
        return item

    def build_job_payload(self, item: Mapping[str, Any]) -> Dict[str, Any]:
        snapshot = _load_json(item.get("context_snapshot_json"), {})
        with self.database.connection() as connection:
            user_row = connection.execute(
                """
                SELECT m.rowid, m.content, q.quote_text
                FROM assistant_messages m
                LEFT JOIN assistant_message_quotes q
                  ON q.message_id=m.id
                WHERE m.id=? AND m.conversation_id=?
                """,
                (
                    item["parent_user_message_id"],
                    item["conversation_id"],
                ),
            ).fetchone()
            if not user_row:
                raise ValueError("对话问题不存在")
            history_rows = connection.execute(
                """
                SELECT role, content
                FROM assistant_messages
                WHERE conversation_id=? AND rowid<?
                  AND status='completed'
                  AND content!=''
                ORDER BY rowid DESC
                LIMIT 12
                """,
                (item["conversation_id"], user_row["rowid"]),
            ).fetchall()
        return {
            "context": dict(snapshot.get("context") or {}),
            "sources": list(snapshot.get("sources") or []),
            "history": [
                dict(row) for row in reversed(history_rows)
            ],
            "question": str(user_row["content"]),
            "selected_quote": str(user_row["quote_text"] or ""),
        }

    def start_tool_call(
        self,
        *,
        message_id: str,
        claim_token: str,
        sequence: int,
        agent_role: str,
        tool_name: str,
        tool_label: str,
        capability: str,
        read_only: bool,
        arguments: Mapping[str, Any],
        initial_status: str = "running",
        error: str = "",
    ) -> str:
        if initial_status not in {"running", "denied"}:
            raise ValueError("工具初始状态无效")
        call_id = uuid.uuid4().hex
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            owner = connection.execute(
                """
                SELECT 1 FROM assistant_messages
                WHERE id=? AND role='assistant' AND status='running'
                  AND claim_token=?
                """,
                (message_id, claim_token),
            ).fetchone()
            if not owner:
                connection.rollback()
                raise ValueError("对话任务租约已经失效")
            latest = connection.execute(
                """
                SELECT COALESCE(MAX(sequence), 0) AS latest
                FROM assistant_tool_calls
                WHERE assistant_message_id=?
                """,
                (message_id,),
            ).fetchone()
            actual_sequence = max(
                int(sequence), int(latest["latest"] or 0) + 1
            )
            connection.execute(
                """
                INSERT INTO assistant_tool_calls(
                    id, assistant_message_id, sequence, agent_role,
                    tool_name, tool_label, capability, read_only,
                    arguments_json, result_json, status, error,
                    started_at, finished_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '{}', ?, ?, ?, ?)
                """,
                (
                    call_id,
                    message_id,
                    actual_sequence,
                    agent_role,
                    tool_name[:100],
                    tool_label[:120],
                    capability[:100],
                    1 if read_only else 0,
                    _json(dict(arguments)),
                    initial_status,
                    error[:2000] or None,
                    now,
                    now if initial_status == "denied" else None,
                ),
            )
            connection.commit()
        return call_id

    def record_agent_step(
        self,
        *,
        message_id: str,
        claim_token: str,
        sequence: int,
        agent_role: str,
        action: str,
        tool_name: str,
        tool_label: str,
        available_tools: Sequence[str],
        decision: Mapping[str, Any],
        outcome_status: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int,
        error: str = "",
    ) -> str:
        if action not in {"call_tool", "finish", "fallback"}:
            raise ValueError("Agent 步骤动作无效")
        if outcome_status not in {
            "completed",
            "denied",
            "failed",
            "fallback",
        }:
            raise ValueError("Agent 步骤结果无效")
        step_id = uuid.uuid4().hex
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            owner = connection.execute(
                """
                SELECT 1 FROM assistant_messages
                WHERE id=? AND role='assistant' AND status='running'
                  AND claim_token=?
                """,
                (message_id, claim_token),
            ).fetchone()
            if not owner:
                connection.rollback()
                raise ValueError("对话任务租约已经失效")
            latest = connection.execute(
                """
                SELECT COALESCE(MAX(sequence), 0) AS latest
                FROM assistant_agent_steps
                WHERE assistant_message_id=?
                """,
                (message_id,),
            ).fetchone()
            actual_sequence = max(
                int(sequence), int(latest["latest"] or 0) + 1
            )
            connection.execute(
                """
                INSERT INTO assistant_agent_steps(
                    id, assistant_message_id, sequence, agent_role,
                    action, tool_name, tool_label, available_tools_json,
                    decision_json, outcome_status, error, input_tokens,
                    output_tokens, latency_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    step_id,
                    message_id,
                    actual_sequence,
                    agent_role,
                    action,
                    tool_name[:100],
                    tool_label[:120],
                    _json(list(available_tools)),
                    _json(dict(decision)),
                    outcome_status,
                    error[:2000] or None,
                    max(0, int(input_tokens)),
                    max(0, int(output_tokens)),
                    max(0, int(latency_ms)),
                    utc_now(),
                ),
            )
            connection.commit()
        return step_id

    def finish_tool_call(
        self,
        *,
        call_id: str,
        message_id: str,
        claim_token: str,
        status: str,
        result: Mapping[str, Any],
        error: str,
    ) -> bool:
        if status not in {"completed", "failed"}:
            raise ValueError("工具结束状态无效")
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE assistant_tool_calls
                SET status=?, result_json=?, error=?, finished_at=?
                WHERE id=? AND assistant_message_id=? AND status='running'
                  AND EXISTS(
                      SELECT 1 FROM assistant_messages m
                      WHERE m.id=assistant_tool_calls.assistant_message_id
                        AND m.status='running' AND m.claim_token=?
                  )
                """,
                (
                    status,
                    _json(dict(result)),
                    error[:2000] or None,
                    utc_now(),
                    call_id,
                    message_id,
                    claim_token,
                ),
            )
            connection.commit()
        return cursor.rowcount == 1

    def complete_message(
        self,
        *,
        message_id: str,
        claim_token: str,
        response: AssistantChatResponse,
    ) -> bool:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT m.context_snapshot_json, m.parent_user_message_id,
                       c.user_id
                FROM assistant_messages m
                JOIN assistant_conversations c
                  ON c.id=m.conversation_id
                WHERE m.id=? AND m.role='assistant'
                  AND m.status='running' AND m.claim_token=?
                """,
                (message_id, claim_token),
            ).fetchone()
            if not row:
                return False
            quote = connection.execute(
                """
                SELECT * FROM assistant_message_quotes
                WHERE message_id=?
                """,
                (row["parent_user_message_id"],),
            ).fetchone()
            snapshot = _load_json(row["context_snapshot_json"], {})
            normalized = self._normalize_response(
                response=response,
                sources=[
                    *list(snapshot.get("sources") or []),
                    *list(response.accessed_sources or []),
                ],
                quote=dict(quote) if quote else None,
                context=dict(snapshot.get("context") or {}),
            )
            cursor = connection.execute(
                """
                UPDATE assistant_messages
                SET status='completed', content=?, response_json=?,
                    raw_response=?, input_tokens=?, output_tokens=?,
                    finished_at=?, claim_token=NULL, lease_expires_at=NULL,
                    error=NULL
                WHERE id=? AND role='assistant' AND status='running'
                  AND claim_token=?
                """,
                (
                    normalized["answer"],
                    _json(normalized),
                    response.raw_response,
                    response.input_tokens,
                    response.output_tokens,
                    utc_now(),
                    message_id,
                    claim_token,
                ),
            )
            connection.commit()
        accepted = cursor.rowcount == 1
        if accepted:
            self._auto_commit_completed_message(
                user_id=int(row["user_id"]),
                assistant_message_id=message_id,
                response=normalized,
                context=dict(snapshot.get("context") or {}),
            )
        return accepted

    def _auto_commit_completed_message(
        self,
        *,
        user_id: int,
        assistant_message_id: str,
        response: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> None:
        boundaries = dict(context.get("assistant_boundaries") or {})
        if not boundaries.get("auto_commit_working_copy"):
            return
        kind = ""
        try:
            if isinstance(response.get("draft"), dict):
                kind = "draft"
                result = self.save_draft_candidate(
                    user_id=user_id,
                    assistant_message_id=assistant_message_id,
                )
            elif isinstance(response.get("rewrite"), dict):
                kind = "rewrite"
                result = self.save_rewrite_candidate(
                    user_id=user_id,
                    assistant_message_id=assistant_message_id,
                )
            else:
                return
        except (OSError, UnicodeError, ValueError) as exc:
            self._record_auto_commit(
                assistant_message_id=assistant_message_id,
                status="failed",
                kind=kind,
                error=str(exc),
            )
            return
        self._record_auto_commit(
            assistant_message_id=assistant_message_id,
            status="applied",
            kind=kind,
            version_id=str(result["version_id"]),
        )

    def _record_auto_commit(
        self,
        *,
        assistant_message_id: str,
        status: str,
        kind: str,
        version_id: str = "",
        error: str = "",
        reverted_version_id: str = "",
    ) -> None:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT response_json FROM assistant_messages
                WHERE id=? AND role='assistant' AND status='completed'
                """,
                (assistant_message_id,),
            ).fetchone()
            if not row:
                return
            response = _load_json(row["response_json"], {})
            response["auto_commit"] = {
                "status": status,
                "kind": kind,
                "version_id": version_id or None,
                "reverted_version_id": reverted_version_id or None,
                "error": error[:1000] or None,
                "updated_at": utc_now(),
            }
            connection.execute(
                """
                UPDATE assistant_messages SET response_json=?
                WHERE id=?
                """,
                (_json(response), assistant_message_id),
            )
            connection.commit()

    def fail_message(
        self,
        message_id: str,
        claim_token: str,
        error: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> bool:
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE assistant_messages
                SET status='failed', error=?, input_tokens=?,
                    output_tokens=?, finished_at=?, claim_token=NULL,
                    lease_expires_at=NULL
                WHERE id=? AND role='assistant' AND status='running'
                  AND claim_token=?
                """,
                (
                    error[:2000],
                    input_tokens,
                    output_tokens,
                    utc_now(),
                    message_id,
                    claim_token,
                ),
            )
            connection.commit()
        return cursor.rowcount == 1

    def release_claim(
        self, message_id: str, claim_token: str, error: str
    ) -> bool:
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE assistant_messages
                SET status='queued', started_at=NULL, claim_token=NULL,
                    lease_expires_at=NULL, error=?
                WHERE id=? AND role='assistant' AND status='running'
                  AND claim_token=?
                """,
                (error[:2000], message_id, claim_token),
            )
            connection.commit()
        return cursor.rowcount == 1

    def get_message(
        self, *, user_id: int, message_id: str
    ) -> Optional[Dict[str, Any]]:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT m.*, c.project_id, c.document_id
                FROM assistant_messages m
                JOIN assistant_conversations c
                  ON c.id=m.conversation_id
                WHERE m.id=? AND c.user_id=?
                """,
                (message_id, user_id),
            ).fetchone()
            tool_rows = (
                connection.execute(
                    """
                    SELECT * FROM assistant_tool_calls
                    WHERE assistant_message_id=?
                    ORDER BY sequence
                    """,
                    (message_id,),
                ).fetchall()
                if row
                else []
            )
            step_rows = (
                connection.execute(
                    """
                    SELECT * FROM assistant_agent_steps
                    WHERE assistant_message_id=?
                    ORDER BY sequence
                    """,
                    (message_id,),
                ).fetchall()
                if row
                else []
            )
        if not row:
            return None
        result = self._decode_message(dict(row))
        result["tool_calls"] = [
            self._decode_tool_call(dict(tool_row))
            for tool_row in tool_rows
        ]
        result["agent_steps"] = [
            self._decode_agent_step(dict(step_row))
            for step_row in step_rows
        ]
        return result

    def save_rewrite_candidate(
        self, *, user_id: int, assistant_message_id: str
    ) -> Dict[str, str]:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT m.*, c.project_id,
                       q.novel_chapter_id, q.version_id,
                       q.start_offset, q.end_offset, q.quote_text,
                       q.content_hash, ch.content_path AS chapter_content_path
                FROM assistant_messages m
                JOIN assistant_conversations c
                  ON c.id=m.conversation_id
                JOIN assistant_message_quotes q
                  ON q.message_id=m.parent_user_message_id
                JOIN novel_chapters ch
                  ON ch.id=q.novel_chapter_id
                JOIN novel_projects p ON p.id=ch.project_id
                WHERE m.id=? AND c.user_id=? AND p.user_id=?
                  AND m.role='assistant' AND m.status='completed'
                  AND q.source_type='novel_version'
                """,
                (assistant_message_id, user_id, user_id),
            ).fetchone()
            if not row:
                raise ValueError("这条回复没有可保存的选区改写")
            if row["applied_version_id"]:
                return {
                    "project_id": str(row["project_id"]),
                    "chapter_id": str(row["novel_chapter_id"]),
                    "conversation_id": str(row["conversation_id"]),
                    "version_id": str(row["applied_version_id"]),
                }
            response = _load_json(row["response_json"], {})
            rewrite = response.get("rewrite")
            if not isinstance(rewrite, dict):
                raise ValueError("这条回复没有可保存的选区改写")
            replacement = str(rewrite.get("replacement_text") or "")
            if not replacement or len(replacement) > 20_000:
                raise ValueError("改写候选内容无效")
            source = connection.execute(
                """
                SELECT v.content_path
                FROM novel_chapter_versions v
                JOIN novel_chapters ch ON ch.id=v.chapter_id
                JOIN novel_projects p ON p.id=ch.project_id
                WHERE v.id=? AND ch.id=? AND p.user_id=?
                """,
                (row["version_id"], row["novel_chapter_id"], user_id),
            ).fetchone()
            if not source:
                raise ValueError("引用的正文版本已不存在")
        source_text = _read_text(source["content_path"])
        start = int(row["start_offset"])
        end = int(row["end_offset"])
        if (
            _sha256(source_text) != str(row["content_hash"])
            or source_text[start:end] != str(row["quote_text"])
        ):
            raise ValueError("引用版本校验失败，未创建候选稿")
        new_content = source_text[:start] + replacement + source_text[end:]
        version_id = uuid.uuid4().hex
        token = secrets.token_hex(16)
        chapter_content_path = Path(str(row["chapter_content_path"]))
        version_path = (
            chapter_content_path.parent
            / "versions"
            / f"assistant-{token}.txt"
        )
        _atomic_write(version_path, new_content, token)
        count = len(new_content)
        effective_count = effective_char_count(new_content)
        quality_status = "block" if effective_count < 2000 else "pending"
        hard_issue_count = 1 if effective_count < 2000 else 0
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                """
                SELECT 1 FROM generation_jobs
                WHERE chapter_id=? AND status IN ('queued', 'running')
                LIMIT 1
                """,
                (row["novel_chapter_id"],),
            ).fetchone()
            if active:
                connection.rollback()
                raise ValueError("AI 正在生成本章，请完成后再保存改写")
            still_unapplied = connection.execute(
                """
                SELECT applied_version_id FROM assistant_messages
                WHERE id=? AND status='completed'
                """,
                (assistant_message_id,),
            ).fetchone()
            if not still_unapplied:
                connection.rollback()
                raise ValueError("对话回复不存在")
            if still_unapplied["applied_version_id"]:
                connection.rollback()
                return {
                    "project_id": str(row["project_id"]),
                    "chapter_id": str(row["novel_chapter_id"]),
                    "conversation_id": str(row["conversation_id"]),
                    "version_id": str(
                        still_unapplied["applied_version_id"]
                    ),
                }
            _atomic_write(chapter_content_path, new_content, token)
            connection.execute(
                """
                INSERT INTO novel_chapter_versions(
                    id, chapter_id, kind, content_path, char_count,
                    created_at, parent_version_id, status, source,
                    content_hash, change_summary, created_by,
                    quality_status, effective_char_count, hard_issue_count
                ) VALUES (
                    ?, ?, 'assistant_rewrite', ?, ?, ?, ?, 'candidate',
                    'assistant_chat', ?, ?, 'assistant', ?, ?, ?
                )
                """,
                (
                    version_id,
                    row["novel_chapter_id"],
                    str(version_path),
                    count,
                    now,
                    row["version_id"],
                    _sha256(new_content),
                    str(rewrite.get("rationale") or "")[:1000],
                    quality_status,
                    effective_count,
                    hard_issue_count,
                ),
            )
            connection.execute(
                """
                UPDATE novel_chapters
                SET char_count=?, status=?, working_version_id=?,
                    updated_at=?
                WHERE id=?
                """,
                (
                    count,
                    "draft" if count else "planned",
                    version_id,
                    now,
                    row["novel_chapter_id"],
                ),
            )
            connection.execute(
                """
                UPDATE novel_projects SET updated_at=? WHERE id=?
                """,
                (now, row["project_id"]),
            )
            auto_commit = dict(response.get("auto_commit") or {})
            if auto_commit:
                response["auto_commit"] = {
                    **auto_commit,
                    "status": "applied",
                    "version_id": version_id,
                    "error": None,
                    "updated_at": now,
                }
            connection.execute(
                """
                UPDATE assistant_messages
                SET applied_version_id=?, response_json=?
                WHERE id=? AND applied_version_id IS NULL
                """,
                (
                    version_id,
                    _json(response),
                    assistant_message_id,
                ),
            )
            connection.commit()
        return {
            "project_id": str(row["project_id"]),
            "chapter_id": str(row["novel_chapter_id"]),
            "conversation_id": str(row["conversation_id"]),
            "version_id": version_id,
        }

    def apply_settings_candidate(
        self, *, user_id: int, assistant_message_id: str
    ) -> Dict[str, Any]:
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT m.response_json, m.context_snapshot_json,
                       c.id AS conversation_id, c.project_id, c.scope_type
                FROM assistant_messages m
                JOIN assistant_conversations c
                  ON c.id=m.conversation_id
                JOIN novel_projects p ON p.id=c.project_id
                WHERE m.id=? AND c.user_id=? AND p.user_id=?
                  AND m.role='assistant' AND m.status='completed'
                  AND c.scope_type IN ('project', 'chapter')
                """,
                (assistant_message_id, user_id, user_id),
            ).fetchone()
            if not row:
                connection.rollback()
                raise ValueError("这条回复没有可应用的作品设定")
            response = _load_json(row["response_json"], {})
            if response.get("settings_patch_status") == "applied":
                connection.commit()
                return {
                    "project_id": str(row["project_id"]),
                    "conversation_id": str(row["conversation_id"]),
                    "changed_fields": list(
                        response.get("settings_patch_applied_values") or {}
                    ),
                    "already_applied": True,
                }
            raw_patch = response.get("settings_patch")
            if not isinstance(raw_patch, dict):
                connection.rollback()
                raise ValueError("这条回复没有可应用的作品设定")
            try:
                patch = AssistantSettingsPatch.model_validate(
                    raw_patch
                ).model_dump(exclude_none=True)
            except ValueError as exc:
                connection.rollback()
                raise ValueError("候选设定结构无效") from exc
            snapshot = _load_json(row["context_snapshot_json"], {})
            context = dict(snapshot.get("context") or {})
            capabilities = {
                str(item)
                for item in (
                    (context.get("agent") or {}).get("capabilities") or []
                )
            }
            if PROPOSE_SETTINGS_PATCH not in capabilities:
                connection.rollback()
                raise ValueError("生成这条回复的角色没有设定修改权限")
            project = connection.execute(
                """
                SELECT * FROM novel_projects
                WHERE id=? AND user_id=?
                """,
                (row["project_id"], user_id),
            ).fetchone()
            if not project:
                connection.rollback()
                raise ValueError("小说项目不存在")
            current = dict(project)
            baseline = dict(context.get("project") or {})
            stale_fields = [
                key
                for key, value in patch.items()
                if key in baseline
                and current.get(key) != baseline.get(key)
                and current.get(key) != value
            ]
            if stale_fields:
                connection.rollback()
                labels = "、".join(
                    SETTING_FIELD_LABELS.get(key, key)
                    for key in stale_fields
                )
                raise ValueError(
                    f"{labels}在讨论后已经变化，请让 AI 基于最新设定重新整理"
                )
            before = {key: current.get(key) for key in patch}
            assignments = ", ".join(f"{key}=?" for key in patch)
            values = [patch[key] for key in patch]
            now = utc_now()
            cursor = connection.execute(
                f"""
                UPDATE novel_projects
                SET {assignments}, updated_at=?
                WHERE id=? AND user_id=?
                """,
                (*values, now, row["project_id"], user_id),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise ValueError("保存候选设定失败")
            response["settings_patch_status"] = "applied"
            response["settings_patch_applied_at"] = now
            response["settings_patch_before"] = before
            response["settings_patch_applied_values"] = patch
            connection.execute(
                """
                UPDATE assistant_messages SET response_json=?
                WHERE id=?
                """,
                (_json(response), assistant_message_id),
            )
            connection.commit()
        return {
            "project_id": str(row["project_id"]),
            "conversation_id": str(row["conversation_id"]),
            "changed_fields": list(patch),
            "already_applied": False,
        }

    def apply_story_plan_candidate(
        self, *, user_id: int, assistant_message_id: str
    ) -> Dict[str, Any]:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT m.response_json, m.context_snapshot_json,
                       c.id AS conversation_id, c.project_id
                FROM assistant_messages m
                JOIN assistant_conversations c
                  ON c.id=m.conversation_id
                JOIN novel_projects p ON p.id=c.project_id
                WHERE m.id=? AND c.user_id=? AND p.user_id=?
                  AND m.role='assistant' AND m.status='completed'
                  AND c.scope_type IN ('project', 'chapter')
                """,
                (assistant_message_id, user_id, user_id),
            ).fetchone()
        if not row:
            raise ValueError("这条回复没有可采用的故事规划")

        response = _load_json(row["response_json"], {})
        if response.get("story_plan_status") == "applied":
            return {
                "project_id": str(row["project_id"]),
                "conversation_id": str(row["conversation_id"]),
                "version_id": str(
                    response.get("story_plan_version_id") or ""
                ),
                "already_applied": True,
            }
        raw_plan = response.get("story_plan")
        if not isinstance(raw_plan, dict):
            raise ValueError("这条回复没有可采用的故事规划")
        try:
            proposal = AssistantStoryPlanProposal.model_validate(raw_plan)
        except ValueError as exc:
            raise ValueError("故事规划候选结构无效") from exc

        snapshot = _load_json(row["context_snapshot_json"], {})
        context = dict(snapshot.get("context") or {})
        capabilities = {
            str(item)
            for item in (
                (context.get("agent") or {}).get("capabilities") or []
            )
        }
        if PROPOSE_STORY_PLAN not in capabilities:
            raise ValueError("生成这条回复的内部任务没有故事规划权限")

        planning_service = StoryPlanningService(self.database)
        current = planning_service.get_blueprint(
            user_id=user_id,
            project_id=str(row["project_id"]),
        )
        baseline_version_id = str(
            context.get("story_blueprint_version_id") or ""
        )
        current_version_id = str(
            (current or {}).get("confirmed_version_id") or ""
        )
        if current_version_id != baseline_version_id:
            raise ValueError(
                "故事规划在讨论后已经变化，请让 AI 基于最新版本重新规划"
            )

        version_id = planning_service.save_blueprint(
            user_id=user_id,
            project_id=str(row["project_id"]),
            blueprint=proposal.blueprint,
            confirm=True,
            source="story_planner",
        )
        now = utc_now()
        response["story_plan_status"] = "applied"
        response["story_plan_applied_at"] = now
        response["story_plan_version_id"] = version_id
        boundary = dict(response.get("boundary") or {})
        boundary["story_plan_unchanged"] = False
        response["boundary"] = boundary
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE assistant_messages SET response_json=?
                WHERE id=? AND role='assistant' AND status='completed'
                """,
                (_json(response), assistant_message_id),
            )
            connection.commit()
        if cursor.rowcount != 1:
            raise ValueError("保存故事规划候选状态失败")
        return {
            "project_id": str(row["project_id"]),
            "conversation_id": str(row["conversation_id"]),
            "version_id": version_id,
            "already_applied": False,
        }

    def save_draft_candidate(
        self, *, user_id: int, assistant_message_id: str
    ) -> Dict[str, Any]:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT m.response_json, m.context_snapshot_json,
                       m.applied_version_id, c.id AS conversation_id,
                       c.project_id, c.novel_chapter_id,
                       ch.content_path, ch.working_version_id
                FROM assistant_messages m
                JOIN assistant_conversations c
                  ON c.id=m.conversation_id
                JOIN novel_chapters ch ON ch.id=c.novel_chapter_id
                JOIN novel_projects p ON p.id=ch.project_id
                WHERE m.id=? AND c.user_id=? AND p.user_id=?
                  AND m.role='assistant' AND m.status='completed'
                  AND c.scope_type='chapter'
                """,
                (assistant_message_id, user_id, user_id),
            ).fetchone()
            if not row:
                raise ValueError("这条回复没有可保存的章节候选稿")
            if row["applied_version_id"]:
                return {
                    "project_id": str(row["project_id"]),
                    "chapter_id": str(row["novel_chapter_id"]),
                    "conversation_id": str(row["conversation_id"]),
                    "version_id": str(row["applied_version_id"]),
                    "already_applied": True,
                }
            response = _load_json(row["response_json"], {})
            raw_draft = response.get("draft")
            if not isinstance(raw_draft, dict):
                raise ValueError("这条回复没有可保存的章节候选稿")
            try:
                draft = AssistantDraftProposal.model_validate(raw_draft)
            except ValueError as exc:
                raise ValueError("章节候选稿结构无效") from exc
            snapshot = _load_json(row["context_snapshot_json"], {})
            context = dict(snapshot.get("context") or {})
            capabilities = {
                str(item)
                for item in (
                    (context.get("agent") or {}).get("capabilities") or []
                )
            }
            if CREATE_CANDIDATE_DRAFT not in capabilities:
                raise ValueError("生成这条回复的角色没有创作候选稿权限")

        chapter_content_path = Path(str(row["content_path"]))
        current_content = _read_text(chapter_content_path)
        expected_hash = str(
            context.get("current_chapter_hash") or _sha256("")
        )
        if _sha256(current_content) != expected_hash:
            raise ValueError("正文在生成候选稿后已经变化，请重新生成")
        if draft.mode == "append" and current_content.strip():
            new_content = (
                current_content.rstrip()
                + "\n\n"
                + draft.content.lstrip()
            )
        else:
            new_content = draft.content
        if len(new_content) > 200_000:
            raise ValueError("应用后单章正文会超过 200000 字")

        version_id = uuid.uuid4().hex
        token = secrets.token_hex(16)
        version_path = (
            chapter_content_path.parent
            / "versions"
            / f"assistant-draft-{token}.txt"
        )
        _atomic_write(version_path, new_content, token)
        count = len(new_content)
        effective_count = effective_char_count(new_content)
        quality_status = "block" if effective_count < 2000 else "pending"
        hard_issue_count = 1 if effective_count < 2000 else 0
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                """
                SELECT 1 FROM generation_jobs
                WHERE chapter_id=? AND status IN ('queued', 'running')
                LIMIT 1
                """,
                (row["novel_chapter_id"],),
            ).fetchone()
            if active:
                connection.rollback()
                raise ValueError("AI 正在生成本章，请完成后再保存候选稿")
            latest = connection.execute(
                """
                SELECT m.applied_version_id, ch.working_version_id
                FROM assistant_messages m
                JOIN assistant_conversations c
                  ON c.id=m.conversation_id
                JOIN novel_chapters ch ON ch.id=c.novel_chapter_id
                WHERE m.id=?
                """,
                (assistant_message_id,),
            ).fetchone()
            if not latest:
                connection.rollback()
                raise ValueError("对话回复不存在")
            if latest["applied_version_id"]:
                connection.rollback()
                return {
                    "project_id": str(row["project_id"]),
                    "chapter_id": str(row["novel_chapter_id"]),
                    "conversation_id": str(row["conversation_id"]),
                    "version_id": str(latest["applied_version_id"]),
                    "already_applied": True,
                }
            if str(latest["working_version_id"] or "") != str(
                row["working_version_id"] or ""
            ):
                connection.rollback()
                raise ValueError("正文版本已经变化，请重新生成候选稿")
            _atomic_write(chapter_content_path, new_content, token)
            connection.execute(
                """
                INSERT INTO novel_chapter_versions(
                    id, chapter_id, kind, content_path, char_count,
                    created_at, parent_version_id, status, source,
                    content_hash, change_summary, created_by,
                    quality_status, effective_char_count, hard_issue_count
                ) VALUES (
                    ?, ?, 'assistant_draft', ?, ?, ?, ?, 'candidate',
                    'assistant_chat', ?, ?, 'assistant', ?, ?, ?
                )
                """,
                (
                    version_id,
                    row["novel_chapter_id"],
                    str(version_path),
                    count,
                    now,
                    row["working_version_id"],
                    _sha256(new_content),
                    draft.rationale[:1000],
                    quality_status,
                    effective_count,
                    hard_issue_count,
                ),
            )
            connection.execute(
                """
                UPDATE novel_chapters
                SET char_count=?, status='draft', working_version_id=?,
                    updated_at=?
                WHERE id=?
                """,
                (
                    count,
                    version_id,
                    now,
                    row["novel_chapter_id"],
                ),
            )
            response["draft_status"] = "applied"
            response["draft_applied_at"] = now
            auto_commit = dict(response.get("auto_commit") or {})
            if auto_commit:
                response["auto_commit"] = {
                    **auto_commit,
                    "status": "applied",
                    "version_id": version_id,
                    "error": None,
                    "updated_at": now,
                }
            connection.execute(
                """
                UPDATE assistant_messages
                SET applied_version_id=?, response_json=?
                WHERE id=? AND applied_version_id IS NULL
                """,
                (
                    version_id,
                    _json(response),
                    assistant_message_id,
                ),
            )
            connection.execute(
                "UPDATE novel_projects SET updated_at=? WHERE id=?",
                (now, row["project_id"]),
            )
            connection.commit()
        return {
            "project_id": str(row["project_id"]),
            "chapter_id": str(row["novel_chapter_id"]),
            "conversation_id": str(row["conversation_id"]),
            "version_id": version_id,
            "already_applied": False,
        }

    def revert_auto_commit(
        self, *, user_id: int, assistant_message_id: str
    ) -> Dict[str, Any]:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT m.response_json, m.applied_version_id,
                       c.id AS conversation_id, c.project_id,
                       c.novel_chapter_id, ch.content_path,
                       ch.working_version_id, ch.canonical_version_id,
                       v.parent_version_id
                FROM assistant_messages m
                JOIN assistant_conversations c
                  ON c.id=m.conversation_id
                JOIN novel_chapters ch ON ch.id=c.novel_chapter_id
                JOIN novel_projects p ON p.id=ch.project_id
                JOIN novel_chapter_versions v
                  ON v.id=m.applied_version_id
                WHERE m.id=? AND c.user_id=? AND p.user_id=?
                  AND m.role='assistant' AND m.status='completed'
                """,
                (assistant_message_id, user_id, user_id),
            ).fetchone()
            if not row:
                raise ValueError("这条回复没有可撤回的自动提交")
            response = _load_json(row["response_json"], {})
            auto_commit = dict(response.get("auto_commit") or {})
            agent_role = str(
                (response.get("agent") or {}).get("role") or "advisor"
            )
            if auto_commit.get("status") == "reverted":
                return {
                    "project_id": str(row["project_id"]),
                    "chapter_id": str(row["novel_chapter_id"]),
                    "conversation_id": str(row["conversation_id"]),
                    "version_id": str(
                        auto_commit.get("reverted_version_id") or ""
                    ),
                    "agent_role": agent_role,
                    "already_reverted": True,
                }
            if auto_commit.get("status") != "applied":
                raise ValueError("这条回复不是自动提交，无法在这里撤回")
            applied_version_id = str(row["applied_version_id"])
            if str(row["working_version_id"] or "") != applied_version_id:
                raise ValueError("后面已有新的工作稿，请从版本历史恢复")
            if str(row["canonical_version_id"] or "") == applied_version_id:
                raise ValueError("该版本已经进入正史，请使用正史替换流程")
            parent_version_id = str(row["parent_version_id"] or "")
            parent = None
            if parent_version_id:
                parent = connection.execute(
                    """
                    SELECT v.content_path
                    FROM novel_chapter_versions v
                    JOIN novel_chapters ch ON ch.id=v.chapter_id
                    JOIN novel_projects p ON p.id=ch.project_id
                    WHERE v.id=? AND ch.id=? AND p.id=? AND p.user_id=?
                    """,
                    (
                        parent_version_id,
                        row["novel_chapter_id"],
                        row["project_id"],
                        user_id,
                    ),
                ).fetchone()
                if not parent:
                    raise ValueError("自动提交的父版本已经不存在")
        restored_content = (
            _read_text(parent["content_path"]) if parent else ""
        )
        version_id = uuid.uuid4().hex
        token = secrets.token_hex(16)
        chapter_content_path = Path(str(row["content_path"]))
        version_path = (
            chapter_content_path.parent
            / "versions"
            / f"assistant-revert-{token}.txt"
        )
        _atomic_write(version_path, restored_content, token)
        count = len(restored_content)
        effective_count = effective_char_count(restored_content)
        quality_status = "block" if effective_count < 2000 else "pending"
        hard_issue_count = 1 if effective_count < 2000 else 0
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            latest = connection.execute(
                """
                SELECT ch.working_version_id, ch.canonical_version_id,
                       m.response_json
                FROM assistant_messages m
                JOIN assistant_conversations c
                  ON c.id=m.conversation_id
                JOIN novel_chapters ch ON ch.id=c.novel_chapter_id
                WHERE m.id=? AND c.user_id=?
                """,
                (assistant_message_id, user_id),
            ).fetchone()
            if not latest:
                connection.rollback()
                raise ValueError("对话回复不存在")
            latest_response = _load_json(
                latest["response_json"], {}
            )
            latest_auto_commit = dict(
                latest_response.get("auto_commit") or {}
            )
            if latest_auto_commit.get("status") == "reverted":
                connection.rollback()
                return {
                    "project_id": str(row["project_id"]),
                    "chapter_id": str(row["novel_chapter_id"]),
                    "conversation_id": str(row["conversation_id"]),
                    "version_id": str(
                        latest_auto_commit.get("reverted_version_id")
                        or ""
                    ),
                    "agent_role": agent_role,
                    "already_reverted": True,
                }
            if str(latest["working_version_id"] or "") != applied_version_id:
                connection.rollback()
                raise ValueError("后面已有新的工作稿，请从版本历史恢复")
            if str(latest["canonical_version_id"] or "") == applied_version_id:
                connection.rollback()
                raise ValueError("该版本已经进入正史，请使用正史替换流程")
            _atomic_write(chapter_content_path, restored_content, token)
            connection.execute(
                """
                INSERT INTO novel_chapter_versions(
                    id, chapter_id, kind, content_path, char_count,
                    created_at, parent_version_id, status, source,
                    content_hash, change_summary, created_by,
                    quality_status, effective_char_count, hard_issue_count
                ) VALUES (
                    ?, ?, 'assistant_revert', ?, ?, ?, ?, 'candidate',
                    'assistant_chat', ?, ?, 'author', ?, ?, ?
                )
                """,
                (
                    version_id,
                    row["novel_chapter_id"],
                    str(version_path),
                    count,
                    now,
                    applied_version_id,
                    _sha256(restored_content),
                    "撤回 AI 自动提交",
                    quality_status,
                    effective_count,
                    hard_issue_count,
                ),
            )
            connection.execute(
                """
                UPDATE novel_chapters
                SET char_count=?, status=?, working_version_id=?,
                    updated_at=?
                WHERE id=?
                """,
                (
                    count,
                    "draft" if count else "planned",
                    version_id,
                    now,
                    row["novel_chapter_id"],
                ),
            )
            latest_response["auto_commit"] = {
                **latest_auto_commit,
                "status": "reverted",
                "reverted_version_id": version_id,
                "updated_at": now,
            }
            connection.execute(
                """
                UPDATE assistant_messages SET response_json=?
                WHERE id=?
                """,
                (_json(latest_response), assistant_message_id),
            )
            connection.execute(
                """
                UPDATE novel_projects SET updated_at=? WHERE id=?
                """,
                (now, row["project_id"]),
            )
            connection.commit()
        return {
            "project_id": str(row["project_id"]),
            "chapter_id": str(row["novel_chapter_id"]),
            "conversation_id": str(row["conversation_id"]),
            "version_id": version_id,
            "agent_role": agent_role,
            "already_reverted": False,
        }

    def get_novel_source(
        self,
        *,
        user_id: int,
        project_id: str,
        chapter_id: str,
        version_id: str,
    ) -> Optional[Dict[str, Any]]:
        version = self.database.get_chapter_version(
            user_id, project_id, chapter_id, version_id
        )
        if not version:
            return None
        result = dict(version)
        result["content"] = _read_text(version["content_path"])
        result["source_type"] = "novel_version"
        return result

    def get_reference_source(
        self,
        *,
        user_id: int,
        document_id: str,
        reference_chapter_id: str,
    ) -> Optional[Dict[str, Any]]:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT c.*, d.title AS document_title
                FROM chapters c
                JOIN documents d ON d.id=c.document_id
                WHERE c.id=? AND c.document_id=? AND d.user_id=?
                """,
                (reference_chapter_id, document_id, user_id),
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["content"] = _read_text(row["content_path"])
        result["source_type"] = "reference_chapter"
        return result

    def _validate_quote(
        self,
        *,
        user_id: int,
        conversation: Mapping[str, Any],
        quote: Mapping[str, Any],
    ) -> Dict[str, Any]:
        source_type = str(quote.get("source_type") or "")
        try:
            start = int(quote.get("start_offset"))
            end = int(quote.get("end_offset"))
        except (TypeError, ValueError) as exc:
            raise ValueError("引用位置无效") from exc
        quote_text = (
            str(quote.get("quote_text") or "")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )
        supplied_hash = str(quote.get("content_hash") or "")
        if not quote_text or len(quote_text) > 6_000:
            raise ValueError("引用文字需在 1–6,000 个字符之间")
        if start < 0 or end <= start:
            raise ValueError("引用位置无效")
        if source_type == "novel_version":
            project_id = str(quote.get("project_id") or "")
            chapter_id = str(quote.get("novel_chapter_id") or "")
            version_id = str(quote.get("version_id") or "")
            if project_id != str(conversation.get("project_id") or ""):
                raise ValueError("引用不属于当前小说项目")
            version = self.database.get_chapter_version(
                user_id, project_id, chapter_id, version_id
            )
            if not version:
                raise ValueError("引用的正文版本不存在")
            if (
                conversation["scope_type"] == "chapter"
                and chapter_id
                != str(conversation.get("novel_chapter_id") or "")
            ):
                raise ValueError("引用不属于当前章节对话")
            text = _read_text(version["content_path"])
            content_hash = _sha256(text)
            if supplied_hash != content_hash:
                raise ValueError("正文已发生变化，请保存后重新选择引用")
            start, end = _resolve_quote_offsets(
                text,
                start=start,
                end=end,
                quote_text=quote_text,
            )
            return {
                "source_type": source_type,
                "project_id": project_id,
                "document_id": None,
                "novel_chapter_id": chapter_id,
                "version_id": version_id,
                "reference_chapter_id": None,
                "start_offset": start,
                "end_offset": end,
                "quote_text": quote_text,
                "content_hash": content_hash,
                "source_label": (
                    f"第 {version['position']} 章"
                    f"《{version['chapter_title'] or '未命名章节'}》版本"
                ),
                "content": text,
            }
        if source_type == "reference_chapter":
            document_id = str(quote.get("document_id") or "")
            chapter_id = str(quote.get("reference_chapter_id") or "")
            if document_id != str(conversation.get("document_id") or ""):
                raise ValueError("引用不属于当前拆书文档")
            source = self.get_reference_source(
                user_id=user_id,
                document_id=document_id,
                reference_chapter_id=chapter_id,
            )
            if not source:
                raise ValueError("引用的参考章节不存在")
            if (
                conversation["scope_type"] == "reference_chapter"
                and chapter_id
                != str(
                    conversation.get("reference_chapter_id") or ""
                )
            ):
                raise ValueError("引用不属于当前参考章节对话")
            text = str(source["content"])
            content_hash = _sha256(text)
            if supplied_hash != content_hash:
                raise ValueError("参考章节已发生变化，请重新选择引用")
            try:
                start, end = _resolve_quote_offsets(
                    text,
                    start=start,
                    end=end,
                    quote_text=quote_text,
                )
            except ValueError as exc:
                raise ValueError("引用位置与参考章节不一致") from exc
            return {
                "source_type": source_type,
                "project_id": None,
                "document_id": document_id,
                "novel_chapter_id": None,
                "version_id": None,
                "reference_chapter_id": chapter_id,
                "start_offset": start,
                "end_offset": end,
                "quote_text": quote_text,
                "content_hash": content_hash,
                "source_label": (
                    f"参考书第 {source['position']} 章"
                    f"《{source['title']}》"
                ),
                "content": text,
            }
        raise ValueError("不支持的引用来源")

    def _build_context_snapshot(
        self,
        *,
        user_id: int,
        conversation: Mapping[str, Any],
        normalized_quote: Optional[Mapping[str, Any]],
        agent_role: str = "advisor",
        ui_surface: str = "",
        auto_commit: bool = False,
        settings_prerequisite: bool = False,
    ) -> Dict[str, Any]:
        scope = str(conversation["scope_type"])
        if scope in {"project", "chapter"}:
            context, sources = self._build_novel_context(
                user_id=user_id,
                conversation=conversation,
            )
            with self.database.connection() as connection:
                blueprint_head = connection.execute(
                    """
                    SELECT h.confirmed_version_id
                    FROM novel_story_blueprint_heads h
                    JOIN novel_projects p ON p.id=h.project_id
                    WHERE h.project_id=? AND p.user_id=?
                    """,
                    (conversation.get("project_id"), user_id),
                ).fetchone()
            context["story_blueprint_version_id"] = (
                str(blueprint_head["confirmed_version_id"])
                if blueprint_head
                and blueprint_head["confirmed_version_id"]
                else None
            )
        else:
            context, sources = self._build_document_context(
                user_id=user_id,
                conversation=conversation,
            )
        if normalized_quote:
            selected_source = self._source_from_quote(normalized_quote)
            sources = [
                selected_source,
                *[
                    source
                    for source in sources
                    if source["source_id"]
                    != selected_source["source_id"]
                ],
            ]
            context["selected_quote"] = {
                key: normalized_quote.get(key)
                for key in (
                    "source_type",
                    "source_label",
                    "start_offset",
                    "end_offset",
                    "quote_text",
                )
            }
        manifest = agent_manifest(agent_role)
        capabilities = agent_capabilities(agent_role)
        context["ui_surface"] = str(ui_surface or "")
        context["agent"] = manifest
        context["assistant_boundaries"] = {
            "may_modify_canon": False,
            "may_modify_story_memory": False,
            "may_modify_task_cards": False,
            "may_apply_settings": False,
            "may_apply_text_patch": False,
            "may_propose_settings_patch": (
                PROPOSE_SETTINGS_PATCH in capabilities
            ),
            "may_propose_text_patch": (
                PROPOSE_TEXT_PATCH in capabilities
            ),
            "may_propose_story_plan": (
                PROPOSE_STORY_PLAN in capabilities
            ),
            "may_create_candidate_draft": (
                CREATE_CANDIDATE_DRAFT in capabilities
            ),
            "auto_commit_working_copy": bool(auto_commit),
            "rewrite_requires_author_action": not auto_commit,
            "settings_patch_requires_author_action": True,
            "story_plan_requires_author_action": True,
            "settings_prerequisite": bool(settings_prerequisite),
            "must_propose_settings_before_writing": bool(
                settings_prerequisite
            ),
        }
        return {"context": context, "sources": sources[:12]}

    def _build_linked_work_context(
        self, *, user_id: int, project_id: str
    ) -> Dict[str, Any]:
        with self.database.connection() as connection:
            linked_rows = connection.execute(
                """
                SELECT target.intent, d.id AS document_id,
                       d.title AS document_title,
                       c.id AS chapter_id, c.position, c.title,
                       c.content_path, a.result_json
                FROM work_versions target
                JOIN work_versions source
                  ON source.id=target.base_version_id
                JOIN documents d ON d.id=source.document_id
                JOIN chapters c ON c.document_id=d.id
                LEFT JOIN chapter_analyses a ON a.id=(
                    SELECT latest.id
                    FROM chapter_analyses latest
                    JOIN analysis_jobs job ON job.id=latest.job_id
                    WHERE latest.chapter_id=c.id
                      AND latest.status='completed'
                      AND job.user_id=?
                    ORDER BY latest.finished_at DESC, latest.rowid DESC
                    LIMIT 1
                )
                WHERE target.project_id=?
                ORDER BY c.position
                LIMIT 120
                """,
                (user_id, project_id),
            ).fetchall()
            archive_rows = connection.execute(
                """
                SELECT entry.entry_type, entry.title, entry.content,
                       entry.evidence, entry.status, entry.category,
                       entry.provenance
                FROM work_versions version
                JOIN work_archive_entries entry
                  ON entry.work_id=version.work_id
                WHERE version.project_id=?
                ORDER BY entry.updated_at DESC
                LIMIT 80
                """,
                (project_id,),
            ).fetchall()
        linked_source = None
        if linked_rows:
            excerpt_ids = {
                str(row["chapter_id"]) for row in linked_rows[-4:]
            }
            linked_source = {
                "document_id": str(linked_rows[0]["document_id"]),
                "title": str(linked_rows[0]["document_title"]),
                "relationship": str(linked_rows[0]["intent"]),
                "semantics": (
                    "这是不可变的来源文本及其描述性分析。改写或续写时可"
                    "参考事实与结构，但不得把分析观察自动当成作者确认的"
                    "创作规则。"
                ),
                "chapters": [
                    {
                        "id": str(row["chapter_id"]),
                        "position": int(row["position"]),
                        "title": str(row["title"]),
                        "analysis": self._compact_analysis(
                            _load_json(row["result_json"], {})
                        ),
                        "excerpt": (
                            _read_text(row["content_path"])[:4_000]
                            if str(row["chapter_id"]) in excerpt_ids
                            else ""
                        ),
                    }
                    for row in linked_rows
                ],
            }
        archive_items = [dict(row) for row in archive_rows]
        return {
            "linked_source": linked_source,
            "work_archive": {
                "semantics": (
                    "分析与笔记是带依据的描述性材料，不能自动约束创作；"
                    "只有 status=confirmed 的 creative_rules 是作者已经"
                    "采纳的创作设定。"
                ),
                "descriptive_observations": [
                    item
                    for item in archive_items
                    if item["entry_type"]
                    in {"source_fact", "analysis_note"}
                ],
                "creative_rules": [
                    item
                    for item in archive_items
                    if item["entry_type"] == "creative_rule"
                    and item["status"] == "confirmed"
                ],
                "materials": [
                    item
                    for item in archive_items
                    if item["entry_type"] == "material"
                ],
            },
        }

    def _build_novel_context(
        self,
        *,
        user_id: int,
        conversation: Mapping[str, Any],
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        project_id = str(conversation["project_id"])
        sources: List[Dict[str, Any]] = []
        if conversation["scope_type"] == "chapter":
            chapter_id = str(conversation["novel_chapter_id"])
            raw_context = self.database.get_writing_context(
                user_id, chapter_id
            )
            if not raw_context:
                raise ValueError("章节写作上下文不存在")
            chapter = raw_context["chapter"]
            current_content = ""
            current_version_id = str(
                chapter.get("working_version_id")
                or chapter.get("canonical_version_id")
                or ""
            )
            if current_version_id:
                version = self.database.get_chapter_version(
                    user_id,
                    project_id,
                    chapter_id,
                    current_version_id,
                )
                if version:
                    current_content = _read_text(version["content_path"])
                    sources.append(
                        self._novel_source_item(
                            version=version,
                            project_id=project_id,
                            chapter_id=chapter_id,
                            text=current_content,
                        )
                    )
            previous_content = ""
            previous = raw_context.get("previous_chapter")
            if previous and previous.get("content_path"):
                previous_content = _read_text(previous["content_path"])
                previous_version = self._get_version_for_path(
                    user_id=user_id,
                    project_id=project_id,
                    chapter_id=str(previous["id"]),
                    content_path=str(previous["content_path"]),
                )
                if previous_version:
                    sources.append(
                        self._novel_source_item(
                            version=previous_version,
                            project_id=project_id,
                            chapter_id=str(previous["id"]),
                            text=previous_content,
                            max_chars=6_000,
                        )
                    )
            snapshot = build_writing_context_snapshot(
                context=raw_context,
                operation="creative_conversation",
                instruction="",
                current_content=current_content,
                previous_content=previous_content,
            )
            snapshot["scope"] = "novel_chapter"
            snapshot["current_chapter_hash"] = _sha256(
                current_content
            )
            snapshot["current_version_id"] = current_version_id
            snapshot.update(
                self._build_linked_work_context(
                    user_id=user_id, project_id=project_id
                )
            )
            return snapshot, sources

        with self.database.connection() as connection:
            project = connection.execute(
                """
                SELECT id, title, genre, premise, world_setting,
                       style_guide, ai_instructions, point_of_view,
                       target_chapter_chars,
                       story_promise, target_audience, core_appeal,
                       ending_constraint, planning_horizon
                FROM novel_projects WHERE id=? AND user_id=?
                """,
                (project_id, user_id),
            ).fetchone()
            if not project:
                raise ValueError("小说项目不存在")
            characters = connection.execute(
                """
                SELECT name, role, traits, background, character_arc
                FROM novel_characters
                WHERE project_id=? ORDER BY position LIMIT 40
                """,
                (project_id,),
            ).fetchall()
            chapters = connection.execute(
                """
                SELECT id, position, title, outline, key_points, status,
                       canonical_version_id, working_version_id
                FROM novel_chapters
                WHERE project_id=? ORDER BY position LIMIT 120
                """,
                (project_id,),
            ).fetchall()
            blueprint = connection.execute(
                """
                SELECT v.central_question, v.protagonist_goal,
                       v.core_conflict, v.stakes, v.opening_state,
                       v.ending_state, v.major_turns_json,
                       v.must_payoffs_json, v.forbidden_shortcuts_json
                FROM novel_story_blueprint_heads h
                JOIN novel_story_blueprint_versions v
                  ON v.id=h.confirmed_version_id
                WHERE h.project_id=? AND v.version_status='confirmed'
                """,
                (project_id,),
            ).fetchone()
            arcs = connection.execute(
                """
                SELECT v.arc_type, v.title, v.dramatic_question,
                       v.promise, v.start_state, v.target_payoff,
                       v.lifecycle_status, v.planned_turns_json
                FROM novel_plot_arcs a
                JOIN novel_plot_arc_versions v
                  ON v.id=a.confirmed_version_id
                WHERE a.project_id=? AND v.version_status='confirmed'
                ORDER BY v.priority DESC, a.position LIMIT 30
                """,
                (project_id,),
            ).fetchall()
            memories = connection.execute(
                """
                SELECT ch.position, ch.title, m.summary,
                       m.key_events_json, m.unresolved_questions_json
                FROM chapter_memory m
                JOIN novel_chapters ch ON ch.id=m.chapter_id
                WHERE m.project_id=? AND m.record_status='canon'
                ORDER BY ch.position DESC LIMIT 10
                """,
                (project_id,),
            ).fetchall()
            source_versions = connection.execute(
                """
                SELECT v.*, ch.position, ch.title AS chapter_title,
                       p.title AS project_title,
                       ch.canonical_version_id, ch.working_version_id
                FROM novel_chapters ch
                JOIN novel_projects p ON p.id=ch.project_id
                JOIN novel_chapter_versions v
                  ON v.id=COALESCE(
                    ch.canonical_version_id, ch.working_version_id
                  )
                WHERE ch.project_id=?
                ORDER BY ch.position DESC LIMIT 6
                """,
                (project_id,),
            ).fetchall()
        blueprint_item = dict(blueprint) if blueprint else {}
        for key in (
            "major_turns_json",
            "must_payoffs_json",
            "forbidden_shortcuts_json",
        ):
            if key in blueprint_item:
                blueprint_item[key.removesuffix("_json")] = _load_json(
                    blueprint_item.pop(key), []
                )
        arc_items = []
        for row in arcs:
            item = dict(row)
            item["planned_turns"] = _load_json(
                item.pop("planned_turns_json"), []
            )
            arc_items.append(item)
        memory_items = []
        for row in memories:
            item = dict(row)
            item["key_events"] = _load_json(
                item.pop("key_events_json"), []
            )
            item["unresolved_questions"] = _load_json(
                item.pop("unresolved_questions_json"), []
            )
            memory_items.append(item)
        for version in source_versions:
            version_item = dict(version)
            text = _read_text(version_item["content_path"])
            sources.append(
                self._novel_source_item(
                    version=version_item,
                    project_id=project_id,
                    chapter_id=str(version_item["chapter_id"]),
                    text=text,
                    max_chars=5_000,
                )
            )
        linked_work_context = self._build_linked_work_context(
            user_id=user_id, project_id=project_id
        )
        return (
            {
                "scope": "novel_project",
                "project": dict(project),
                "characters": [dict(row) for row in characters],
                "chapter_plan": [dict(row) for row in chapters],
                "confirmed_story_blueprint": blueprint_item,
                "confirmed_plot_arcs": arc_items,
                "canonical_recent_memory": memory_items,
                **linked_work_context,
            },
            sources,
        )

    def _build_document_context(
        self,
        *,
        user_id: int,
        conversation: Mapping[str, Any],
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        document_id = str(conversation["document_id"])
        with self.database.connection() as connection:
            document = connection.execute(
                """
                SELECT id, title, original_filename, char_count,
                       split_strategy
                FROM documents WHERE id=? AND user_id=?
                """,
                (document_id, user_id),
            ).fetchone()
            if not document:
                raise ValueError("拆书文档不存在")
            rows = connection.execute(
                """
                SELECT c.id, c.position, c.title, c.kind, c.content_path,
                       c.char_count, a.id AS analysis_id, a.result_json
                FROM chapters c
                LEFT JOIN chapter_analyses a ON a.id=(
                    SELECT ca.id
                    FROM chapter_analyses ca
                    JOIN analysis_jobs j ON j.id=ca.job_id
                    WHERE ca.chapter_id=c.id
                      AND ca.status='completed'
                      AND j.user_id=?
                    ORDER BY ca.finished_at DESC, ca.rowid DESC
                    LIMIT 1
                )
                WHERE c.document_id=?
                ORDER BY c.position
                LIMIT 300
                """,
                (user_id, document_id),
            ).fetchall()
            archive_rows = connection.execute(
                """
                SELECT entry.entry_type, entry.title, entry.content,
                       entry.evidence, entry.status, entry.category,
                       entry.provenance
                FROM work_versions version
                JOIN work_archive_entries entry
                  ON entry.work_id=version.work_id
                JOIN works work ON work.id=version.work_id
                WHERE version.document_id=? AND work.user_id=?
                ORDER BY entry.updated_at DESC
                LIMIT 120
                """,
                (document_id, user_id),
            ).fetchall()
        focus_id = str(
            conversation.get("reference_chapter_id") or ""
        )
        chapter_items: List[Dict[str, Any]] = []
        sources: List[Dict[str, Any]] = []
        for row in rows:
            result = _load_json(row["result_json"], {})
            compact = self._compact_analysis(result)
            chapter_items.append(
                {
                    "id": row["id"],
                    "position": row["position"],
                    "title": row["title"],
                    "kind": row["kind"],
                    "char_count": row["char_count"],
                    "analysis": compact,
                }
            )
            if row["analysis_id"] and compact:
                analysis_text = json.dumps(
                    compact, ensure_ascii=False, indent=2
                )
                sources.append(
                    {
                        "source_id": f"analysis:{row['analysis_id']}",
                        "kind": "reference_analysis",
                        "label": (
                            f"拆书分析 · 第 {row['position']} 章"
                            f"《{row['title']}》"
                        ),
                        "text": analysis_text[:8_000],
                        "base_offset": 0,
                        "url": f"/analyses/{row['analysis_id']}",
                        "document_id": document_id,
                        "reference_chapter_id": str(row["id"]),
                    }
                )
            if str(row["id"]) == focus_id:
                text = _read_text(row["content_path"])
                sources.insert(
                    0,
                    {
                        "source_id": (
                            f"reference-chapter:{row['id']}"
                        ),
                        "kind": "reference_chapter",
                        "label": (
                            f"参考书第 {row['position']} 章"
                            f"《{row['title']}》"
                        ),
                        "text": text[:30_000],
                        "base_offset": 0,
                        "url": (
                            f"/documents/{document_id}/chapters/"
                            f"{row['id']}/source"
                        ),
                        "document_id": document_id,
                        "reference_chapter_id": str(row["id"]),
                    },
                )
        archive_items = [dict(row) for row in archive_rows]
        return (
            {
                "scope": (
                    "reference_chapter"
                    if focus_id
                    else "reference_document"
                ),
                "document": dict(document),
                "chapters_and_analysis": chapter_items,
                "work_archive": {
                    "semantics": (
                        "分析与笔记是描述性记录；只有作者明确采纳且状态为"
                        " confirmed 的创作设定才可作为后续写作约束。"
                    ),
                    "descriptive_observations": [
                        item
                        for item in archive_items
                        if item["entry_type"]
                        in {"source_fact", "analysis_note"}
                    ],
                    "creative_rules": [
                        item
                        for item in archive_items
                        if item["entry_type"] == "creative_rule"
                        and item["status"] == "confirmed"
                    ],
                    "materials": [
                        item
                        for item in archive_items
                        if item["entry_type"] == "material"
                    ],
                },
                "originality_boundary": (
                    "只学习抽象方法；不续写、不复刻专有名词、独特措辞或"
                    "具体情节。"
                ),
            },
            sources,
        )

    def _source_from_quote(
        self, quote: Mapping[str, Any]
    ) -> Dict[str, Any]:
        text = str(quote["content"])
        start = int(quote["start_offset"])
        end = int(quote["end_offset"])
        bounded, base = _bounded_source(
            text, selected_start=start, selected_end=end
        )
        if quote["source_type"] == "novel_version":
            source_id = f"novel-version:{quote['version_id']}"
            url = (
                f"/novels/{quote['project_id']}/chapters/"
                f"{quote['novel_chapter_id']}/versions/"
                f"{quote['version_id']}/source"
            )
        else:
            source_id = (
                f"reference-chapter:{quote['reference_chapter_id']}"
            )
            url = (
                f"/documents/{quote['document_id']}/chapters/"
                f"{quote['reference_chapter_id']}/source"
            )
        return {
            "source_id": source_id,
            "kind": str(quote["source_type"]),
            "label": str(quote["source_label"]),
            "text": bounded,
            "base_offset": base,
            "url": url,
            "project_id": quote.get("project_id"),
            "document_id": quote.get("document_id"),
            "novel_chapter_id": quote.get("novel_chapter_id"),
            "version_id": quote.get("version_id"),
            "reference_chapter_id": quote.get(
                "reference_chapter_id"
            ),
        }

    def _novel_source_item(
        self,
        *,
        version: Mapping[str, Any],
        project_id: str,
        chapter_id: str,
        text: str,
        max_chars: int = 30_000,
    ) -> Dict[str, Any]:
        bounded, base = _bounded_source(text, max_chars=max_chars)
        state = (
            "正史"
            if str(version.get("canonical_version_id") or "")
            == str(version["id"])
            else "候选"
        )
        return {
            "source_id": f"novel-version:{version['id']}",
            "kind": "novel_version",
            "label": (
                f"第 {version['position']} 章"
                f"《{version['chapter_title'] or '未命名章节'}》· {state}版本"
            ),
            "text": bounded,
            "base_offset": base,
            "url": (
                f"/novels/{project_id}/chapters/{chapter_id}/versions/"
                f"{version['id']}/source"
            ),
            "project_id": project_id,
            "novel_chapter_id": chapter_id,
            "version_id": str(version["id"]),
        }

    def _get_version_for_path(
        self,
        *,
        user_id: int,
        project_id: str,
        chapter_id: str,
        content_path: str,
    ) -> Optional[Dict[str, Any]]:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT v.*, ch.position, ch.title AS chapter_title,
                       ch.canonical_version_id, ch.working_version_id,
                       p.title AS project_title
                FROM novel_chapter_versions v
                JOIN novel_chapters ch ON ch.id=v.chapter_id
                JOIN novel_projects p ON p.id=ch.project_id
                WHERE v.content_path=? AND ch.id=? AND p.id=?
                  AND p.user_id=?
                LIMIT 1
                """,
                (content_path, chapter_id, project_id, user_id),
            ).fetchone()
        return dict(row) if row else None

    @staticmethod
    def _compact_analysis(result: Any) -> Dict[str, Any]:
        if not isinstance(result, dict):
            return {}
        return {
            key: result.get(key)
            for key in (
                "chapter_title",
                "summary",
                "characters",
                "scenes",
                "key_events",
                "foreshadowing",
                "conflicts",
                "ending_hook",
                "techniques",
            )
            if result.get(key) not in (None, "", [])
        }

    def _normalize_response(
        self,
        *,
        response: AssistantChatResponse,
        sources: Sequence[Mapping[str, Any]],
        quote: Optional[Mapping[str, Any]],
        context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        source_map = {
            str(source.get("source_id") or ""): source
            for source in sources
            if source.get("source_id")
        }
        citations = []
        for proposal in response.result.citations:
            source = source_map.get(proposal.source_id)
            if not source:
                continue
            source_text = str(source.get("text") or "")
            local_start = source_text.find(proposal.quote)
            if local_start < 0:
                continue
            absolute_start = int(source.get("base_offset") or 0) + local_start
            absolute_end = absolute_start + len(proposal.quote)
            url = str(source.get("url") or "")
            separator = "&" if "?" in url else "?"
            if url:
                url = (
                    f"{url}{separator}start={absolute_start}"
                    f"&end={absolute_end}"
                )
            citations.append(
                {
                    "source_id": proposal.source_id,
                    "label": str(source.get("label") or "来源"),
                    "quote": proposal.quote,
                    "note": proposal.note,
                    "url": url,
                    "start_offset": absolute_start,
                    "end_offset": absolute_end,
                }
            )
        capabilities = {
            str(item)
            for item in (
                (context.get("agent") or {}).get("capabilities") or []
            )
        }
        rewrite = None
        if (
            PROPOSE_TEXT_PATCH in capabilities
            and
            response.result.rewrite is not None
            and quote
            and str(quote.get("source_type")) == "novel_version"
        ):
            rewrite = response.result.rewrite.model_dump(mode="json")
        settings_patch = None
        if (
            PROPOSE_SETTINGS_PATCH in capabilities
            and str(context.get("scope") or "")
            in {"novel_project", "novel_chapter"}
            and response.result.settings_patch is not None
        ):
            settings_patch = response.result.settings_patch.model_dump(
                mode="json", exclude_none=True
            )
        story_plan = None
        if (
            PROPOSE_STORY_PLAN in capabilities
            and str(context.get("scope") or "")
            in {"novel_project", "novel_chapter"}
            and response.result.story_plan is not None
        ):
            story_plan = response.result.story_plan.model_dump(
                mode="json"
            )
        draft = None
        if (
            CREATE_CANDIDATE_DRAFT in capabilities
            and str(context.get("scope") or "") == "novel_chapter"
            and response.result.draft is not None
        ):
            draft = response.result.draft.model_dump(mode="json")
        agent = dict(context.get("agent") or {})
        auto_commit_requested = bool(
            (context.get("assistant_boundaries") or {}).get(
                "auto_commit_working_copy"
            )
        )
        auto_commit: Dict[str, Any] = {}
        if auto_commit_requested and (draft or rewrite):
            auto_commit = {
                "status": "pending",
                "kind": "draft" if draft else "rewrite",
                "version_id": None,
                "reverted_version_id": None,
                "error": None,
                "updated_at": utc_now(),
            }
        return {
            "answer": response.result.answer,
            "citations": citations,
            "rewrite": rewrite,
            "draft": draft,
            "draft_status": "candidate" if draft else None,
            "settings_patch": settings_patch,
            "settings_patch_status": (
                "candidate" if settings_patch else None
            ),
            "story_plan": story_plan,
            "story_plan_status": (
                "candidate" if story_plan else None
            ),
            "auto_commit": auto_commit,
            "provider": response.provider,
            "model": response.model,
            "agent": {
                "role": str(agent.get("role") or "advisor"),
                "label": str(agent.get("label") or "讨论"),
                "capabilities": sorted(capabilities),
            },
            "agent_run": {
                "dispatch": dict(context.get("dispatch") or {}),
                "tool_calls": list(response.agent_trace or []),
            },
            "boundary": {
                "canon_unchanged": True,
                "story_memory_unchanged": True,
                "task_card_unchanged": True,
                "project_settings_unchanged": True,
                "story_plan_unchanged": True,
            },
        }

    @staticmethod
    def _decode_message(message: Dict[str, Any]) -> Dict[str, Any]:
        message["response"] = _load_json(
            message.get("response_json"), {}
        )
        agent = message["response"].get("agent")
        if isinstance(agent, dict):
            role = str(agent.get("role") or "")
            try:
                agent["label"] = agent_manifest(role)["label"]
            except ValueError:
                pass
        if message.get("quote_text"):
            message["quote"] = {
                "source_type": message.get("source_type"),
                "project_id": message.get("quote_project_id"),
                "document_id": message.get("quote_document_id"),
                "novel_chapter_id": message.get(
                    "quote_novel_chapter_id"
                ),
                "version_id": message.get("quote_version_id"),
                "reference_chapter_id": message.get(
                    "quote_reference_chapter_id"
                ),
                "start_offset": message.get("quote_start_offset"),
                "end_offset": message.get("quote_end_offset"),
                "quote_text": message.get("quote_text"),
                "content_hash": message.get("quote_content_hash"),
                "source_label": message.get("quote_source_label"),
            }
        else:
            message["quote"] = None
        return message

    @staticmethod
    def _decode_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
        tool_call["arguments"] = _load_json(
            tool_call.get("arguments_json"), {}
        )
        tool_call["result"] = _load_json(
            tool_call.get("result_json"), {}
        )
        tool_call["read_only"] = bool(tool_call.get("read_only"))
        return tool_call

    @staticmethod
    def _decode_agent_step(step: Dict[str, Any]) -> Dict[str, Any]:
        step["available_tools"] = _load_json(
            step.get("available_tools_json"), []
        )
        step["decision"] = _load_json(
            step.get("decision_json"), {}
        )
        return step
