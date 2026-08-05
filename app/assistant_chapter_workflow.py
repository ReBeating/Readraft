from __future__ import annotations

import logging
import os
import secrets
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .agent_capabilities import CREATE_CHAPTER, WRITE_CHAPTER
from .assistant_file_ops import (
    atomic_write as _atomic_write,
    clean_title as _clean_title,
    read_text as _read_text,
    sha256_text as _sha256,
)
from .db import utc_now
from .json_support import (
    dump_canonical_json as _json,
    load_json as _load_json,
)
from .text_metrics import effective_char_count


logger = logging.getLogger(__name__)


class AssistantChapterWorkflowMixin:
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

    def _create_and_bind_next_chapter(
        self,
        *,
        user_id: int,
        conversation: Mapping[str, Any],
        title: str = "",
        outline: str = "",
        key_points: str = "",
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        project_id = str(conversation.get("project_id") or "")
        conversation_id = str(conversation.get("id") or "")
        if not project_id or not conversation_id:
            raise ValueError("当前对话没有可写作的作品")
        clean_title = _clean_title(str(title or ""), "")
        clean_outline = str(outline or "").strip()
        clean_key_points = str(key_points or "").strip()

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
            now = utc_now()
            with self.database.connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    owner = connection.execute(
                        """
                        SELECT c.scope_type, c.novel_chapter_id
                        FROM assistant_conversations c
                        JOIN novel_projects p ON p.id=c.project_id
                        WHERE c.id=? AND c.user_id=? AND c.project_id=?
                          AND p.user_id=?
                          AND c.scope_type IN ('project', 'chapter')
                        """,
                        (
                            conversation_id,
                            user_id,
                            project_id,
                            user_id,
                        ),
                    ).fetchone()
                    if not owner:
                        raise ValueError("对话范围已经变化，请重新发送")
                    last_chapter = connection.execute(
                        """
                        SELECT position, volume_id
                        FROM novel_chapters
                        WHERE project_id=?
                        ORDER BY position DESC
                        LIMIT 1
                        """,
                        (project_id,),
                    ).fetchone()
                    next_position = (
                        int(last_chapter["position"]) + 1
                        if last_chapter
                        else 1
                    )
                    volume_id = (
                        str(last_chapter["volume_id"])
                        if last_chapter and last_chapter["volume_id"]
                        else None
                    )
                    connection.execute(
                        """
                        INSERT INTO novel_chapters(
                            id, project_id, position, title, outline,
                            key_points, content_path, volume_id,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            chapter_id,
                            project_id,
                            next_position,
                            clean_title,
                            clean_outline,
                            clean_key_points,
                            str(content_path),
                            volume_id,
                            now,
                            now,
                        ),
                    )
                    cursor = connection.execute(
                        """
                        UPDATE assistant_conversations
                        SET scope_type='chapter', novel_chapter_id=?,
                            updated_at=?
                        WHERE id=? AND user_id=? AND project_id=?
                          AND scope_type IN ('project', 'chapter')
                        """,
                        (
                            chapter_id,
                            now,
                            conversation_id,
                            user_id,
                            project_id,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise ValueError("对话范围已经变化，请重新发送")
                    connection.execute(
                        "UPDATE novel_projects SET updated_at=? WHERE id=?",
                        (now, project_id),
                    )
                    connection.commit()
                except Exception:
                    connection.rollback()
                    raise
        except Exception:
            shutil.rmtree(chapter_dir, ignore_errors=True)
            raise

        prepared = self.conversations.get(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if not prepared:
            raise ValueError("切换到新章节失败")
        return prepared, {
            "action": "created_next_chapter",
            "chapter_id": chapter_id,
            "chapter_position": next_position,
            "previous_chapter_id": str(
                conversation.get("novel_chapter_id") or ""
            ),
            "title": clean_title,
            "outline": clean_outline,
            "key_points": clean_key_points,
        }

    def create_next_chapter_for_agent(
        self,
        *,
        message_id: str,
        claim_token: str,
        user_id: int,
        title: str = "",
        outline: str = "",
        key_points: str = "",
    ) -> Dict[str, Any]:
        """Create and bind a next chapter during one active Agent run."""

        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT m.context_snapshot_json, m.conversation_id
                FROM assistant_messages m
                JOIN assistant_conversations c
                  ON c.id=m.conversation_id
                WHERE m.id=? AND m.role='assistant' AND m.status='running'
                  AND m.claim_token=? AND c.user_id=?
                """,
                (message_id, claim_token, user_id),
            ).fetchone()
        if not row:
            raise ValueError("对话任务租约已经失效")
        old_snapshot = _load_json(row["context_snapshot_json"], {})
        old_context = dict(old_snapshot.get("context") or {})
        capabilities = {
            str(value)
            for value in ((old_context.get("agent") or {}).get("capabilities") or [])
        }
        if CREATE_CHAPTER not in capabilities:
            raise PermissionError("当前 Agent 没有创建章节权限")
        project_id = str((old_context.get("project") or {}).get("id") or "")
        if not project_id or not self.database.is_main_project(user_id, project_id):
            raise PermissionError("只有 main 作品可以创建章节")
        conversation = self.conversations.get(
            user_id=user_id,
            conversation_id=str(row["conversation_id"]),
        )
        if not conversation:
            raise ValueError("对话不存在")
        prepared, creation = self._create_and_bind_next_chapter(
            user_id=user_id,
            conversation=conversation,
            title=title,
            outline=outline,
            key_points=key_points,
        )
        old_boundaries = dict(old_context.get("assistant_boundaries") or {})
        old_dispatch = dict(old_context.get("dispatch") or {})
        role = str((old_context.get("agent") or {}).get("role") or "writer")
        snapshot = self._build_context_snapshot(
            user_id=user_id,
            conversation=prepared,
            normalized_quote=None,
            agent_role=role,
            ui_surface=str(
                old_dispatch.get("ui_surface")
                or old_context.get("ui_surface")
                or ""
            ),
            auto_commit=bool(
                old_boundaries.get("auto_advance_main_head")
            ),
            settings_prerequisite=bool(
                old_boundaries.get("settings_prerequisite")
            ),
        )
        snapshot["context"]["dispatch"] = {
            **old_dispatch,
            "scope_preflight": creation,
            "reason": "agent_created_next_chapter",
        }
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
        return {
            "context": dict(snapshot.get("context") or {}),
            "sources": list(snapshot.get("sources") or []),
            "creation": creation,
        }

    def start_or_resume_chapter_workflow_for_agent(
        self,
        *,
        message_id: str,
        claim_token: str,
        user_id: int,
        chapters: Sequence[Mapping[str, Any]],
        resume_latest: bool,
    ) -> Dict[str, Any]:
        """Create or resume one durable sequential chapter workflow."""

        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                """
                SELECT message.context_snapshot_json,
                       message.conversation_id, conversation.project_id
                FROM assistant_messages message
                JOIN assistant_conversations conversation
                  ON conversation.id=message.conversation_id
                WHERE message.id=? AND message.role='assistant'
                  AND message.status='running' AND message.claim_token=?
                  AND conversation.user_id=?
                  AND conversation.scope_type IN ('project', 'chapter')
                """,
                (message_id, claim_token, user_id),
            ).fetchone()
            if not active:
                connection.rollback()
                raise ValueError("对话任务租约已经失效")
            snapshot = _load_json(active["context_snapshot_json"], {})
            context = dict(snapshot.get("context") or {})
            capabilities = {
                str(value)
                for value in (
                    (context.get("agent") or {}).get("capabilities") or []
                )
            }
            if not {CREATE_CHAPTER, WRITE_CHAPTER}.issubset(
                capabilities
            ):
                connection.rollback()
                raise PermissionError("当前 Agent 没有连续创作权限")
            project_id = str(active["project_id"] or "")
            if not project_id or not self.database.is_main_project(
                user_id, project_id
            ):
                connection.rollback()
                raise PermissionError("只有 main 作品可以连续创作")
            now = utc_now()
            if resume_latest:
                workflow = connection.execute(
                    """
                    SELECT * FROM assistant_chapter_workflows
                    WHERE user_id=? AND project_id=?
                      AND status IN ('pending', 'running', 'paused')
                    ORDER BY updated_at DESC, rowid DESC
                    LIMIT 1
                    """,
                    (user_id, project_id),
                ).fetchone()
                if not workflow:
                    connection.rollback()
                    raise ValueError("没有可恢复的连续章节工作流")
                workflow_id = str(workflow["id"])
                connection.execute(
                    """
                    UPDATE assistant_chapter_workflow_items
                    SET status='pending', error=''
                    WHERE workflow_id=? AND status IN ('running', 'failed')
                    """,
                    (workflow_id,),
                )
                connection.execute(
                    """
                    UPDATE assistant_chapter_workflows
                    SET conversation_id=?, status='running', error='',
                        updated_at=?, finished_at=NULL
                    WHERE id=?
                    """,
                    (
                        active["conversation_id"],
                        now,
                        workflow_id,
                    ),
                )
            else:
                existing = connection.execute(
                    """
                    SELECT id FROM assistant_chapter_workflows
                    WHERE source_message_id=?
                    """,
                    (message_id,),
                ).fetchone()
                if existing:
                    workflow_id = str(existing["id"])
                    connection.execute(
                        """
                        UPDATE assistant_chapter_workflow_items
                        SET status='pending', error=''
                        WHERE workflow_id=? AND status IN ('running', 'failed')
                        """,
                        (workflow_id,),
                    )
                    connection.execute(
                        """
                        UPDATE assistant_chapter_workflows
                        SET status='running', error='', updated_at=?
                        WHERE id=? AND status<>'completed'
                        """,
                        (now, workflow_id),
                    )
                else:
                    if not chapters:
                        connection.rollback()
                        raise ValueError("连续创作至少需要一个章节目标")
                    workflow_id = uuid.uuid4().hex
                    connection.execute(
                        """
                        INSERT INTO assistant_chapter_workflows(
                            id, user_id, project_id, conversation_id,
                            source_message_id, status, total_count,
                            completed_count, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, 'running', ?, 0, ?, ?)
                        """,
                        (
                            workflow_id,
                            user_id,
                            project_id,
                            active["conversation_id"],
                            message_id,
                            len(chapters),
                            now,
                            now,
                        ),
                    )
                    for sequence, chapter in enumerate(chapters, start=1):
                        connection.execute(
                            """
                            INSERT INTO assistant_chapter_workflow_items(
                                id, workflow_id, sequence, title, outline,
                                key_points, instruction, target_chars,
                                status, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                            """,
                            (
                                uuid.uuid4().hex,
                                workflow_id,
                                sequence,
                                str(chapter.get("title") or "")[:200],
                                str(chapter.get("outline") or ""),
                                str(chapter.get("key_points") or ""),
                                str(chapter.get("instruction") or ""),
                                chapter.get("target_chars"),
                                now,
                            ),
                        )
            connection.commit()
        return self.get_chapter_workflow(
            user_id=user_id,
            workflow_id=workflow_id,
        )

    def get_chapter_workflow(
        self,
        *,
        user_id: int,
        workflow_id: str,
    ) -> Dict[str, Any]:
        with self.database.connection() as connection:
            workflow = connection.execute(
                """
                SELECT * FROM assistant_chapter_workflows
                WHERE id=? AND user_id=?
                """,
                (workflow_id, user_id),
            ).fetchone()
            items = connection.execute(
                """
                SELECT * FROM assistant_chapter_workflow_items
                WHERE workflow_id=? ORDER BY sequence
                """,
                (workflow_id,),
            ).fetchall()
        if not workflow:
            raise ValueError("连续章节工作流不存在")
        return {
            **dict(workflow),
            "items": [dict(item) for item in items],
        }

    def prepare_next_chapter_workflow_item_for_agent(
        self,
        *,
        message_id: str,
        claim_token: str,
        user_id: int,
        workflow_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Bind or create exactly one pending chapter and refresh context."""

        while True:
            with self.database.connection() as connection:
                active = connection.execute(
                    """
                    SELECT message.context_snapshot_json,
                           message.conversation_id
                    FROM assistant_messages message
                    JOIN assistant_conversations conversation
                      ON conversation.id=message.conversation_id
                    JOIN assistant_chapter_workflows workflow
                      ON workflow.id=? AND workflow.user_id=?
                     AND workflow.project_id=conversation.project_id
                    WHERE message.id=? AND message.role='assistant'
                      AND message.status='running' AND message.claim_token=?
                      AND conversation.user_id=?
                    """,
                    (
                        workflow_id,
                        user_id,
                        message_id,
                        claim_token,
                        user_id,
                    ),
                ).fetchone()
                item = connection.execute(
                    """
                    SELECT item.*, workflow.project_id
                    FROM assistant_chapter_workflow_items item
                    JOIN assistant_chapter_workflows workflow
                      ON workflow.id=item.workflow_id
                    WHERE item.workflow_id=? AND item.status='pending'
                    ORDER BY item.sequence LIMIT 1
                    """,
                    (workflow_id,),
                ).fetchone()
                if not active:
                    raise ValueError("对话任务租约已经失效")
                if not item:
                    return None
                if item["version_id"]:
                    persisted = connection.execute(
                        """
                        SELECT 1 FROM novel_chapters
                        WHERE id=? AND head_version_id=?
                        """,
                        (item["chapter_id"], item["version_id"]),
                    ).fetchone()
                    if persisted:
                        now = utc_now()
                        connection.execute(
                            """
                            UPDATE assistant_chapter_workflow_items
                            SET status='completed', error='', finished_at=?
                            WHERE id=?
                            """,
                            (now, item["id"]),
                        )
                        connection.commit()
                        self._refresh_chapter_workflow_counts(workflow_id)
                        continue
                    connection.execute(
                        """
                        UPDATE assistant_chapter_workflow_items
                        SET version_id=NULL
                        WHERE id=? AND status='pending'
                        """,
                        (item["id"],),
                    )
                    connection.commit()
            conversation = self.conversations.get(
                user_id=user_id,
                conversation_id=str(active["conversation_id"]),
            )
            if not conversation:
                raise ValueError("对话不存在")
            creation: dict[str, Any] = {}
            if item["chapter_id"]:
                self._bind_conversation_to_chapter(
                    user_id=user_id,
                    conversation_id=str(active["conversation_id"]),
                    project_id=str(item["project_id"]),
                    chapter_id=str(item["chapter_id"]),
                )
                prepared = self.conversations.get(
                    user_id=user_id,
                    conversation_id=str(active["conversation_id"]),
                )
                if not prepared:
                    raise ValueError("恢复章节工作范围失败")
            else:
                prepared, creation = self._create_and_bind_next_chapter(
                    user_id=user_id,
                    conversation=conversation,
                    title=str(item["title"] or ""),
                    outline=str(item["outline"] or ""),
                    key_points=str(item["key_points"] or ""),
                )
            chapter_id = str(
                item["chapter_id"] or creation.get("chapter_id") or ""
            )
            now = utc_now()
            with self.database.connection() as connection:
                cursor = connection.execute(
                    """
                    UPDATE assistant_chapter_workflow_items
                    SET chapter_id=?, status='running', attempts=attempts+1,
                        error='', started_at=?
                    WHERE id=? AND workflow_id=? AND status='pending'
                    """,
                    (chapter_id, now, item["id"], workflow_id),
                )
                connection.commit()
            if cursor.rowcount != 1:
                raise ValueError("连续章节工作项已经被其他任务处理")
            old_snapshot = _load_json(active["context_snapshot_json"], {})
            old_context = dict(old_snapshot.get("context") or {})
            old_boundaries = dict(
                old_context.get("assistant_boundaries") or {}
            )
            old_dispatch = dict(old_context.get("dispatch") or {})
            role = str((old_context.get("agent") or {}).get("role") or "writer")
            refreshed = self._build_context_snapshot(
                user_id=user_id,
                conversation=prepared,
                normalized_quote=None,
                agent_role=role,
                ui_surface=str(
                    old_dispatch.get("ui_surface")
                    or old_context.get("ui_surface")
                    or ""
                ),
                auto_commit=bool(
                    old_boundaries.get("auto_advance_main_head")
                ),
                settings_prerequisite=False,
            )
            refreshed["context"]["dispatch"] = {
                **old_dispatch,
                "reason": "chapter_workflow",
                "workflow_id": workflow_id,
                "workflow_sequence": int(item["sequence"]),
            }
            with self.database.connection() as connection:
                cursor = connection.execute(
                    """
                    UPDATE assistant_messages SET context_snapshot_json=?
                    WHERE id=? AND role='assistant' AND status='running'
                      AND claim_token=?
                    """,
                    (_json(refreshed), message_id, claim_token),
                )
                connection.commit()
            if cursor.rowcount != 1:
                raise ValueError("对话任务租约已经失效")
            return {
                "item": {**dict(item), "chapter_id": chapter_id},
                "context": dict(refreshed.get("context") or {}),
                "sources": list(refreshed.get("sources") or []),
            }

    def complete_chapter_workflow_item_for_agent(
        self,
        *,
        message_id: str,
        claim_token: str,
        user_id: int,
        workflow_id: str,
        item_id: str,
        content: str,
        change_summary: str,
    ) -> Dict[str, Any]:
        clean_content = str(content or "")
        if not clean_content.strip():
            raise ValueError("连续章节正文不能为空")
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT item.chapter_id, item.sequence, item.status,
                       chapter.content_path AS cache_content_path,
                       chapter.head_version_id,
                       head.content_path AS head_content_path,
                       head.content_hash AS head_content_hash,
                       workflow.project_id, message.provider, message.model,
                       message.credential_source
                FROM assistant_chapter_workflow_items item
                JOIN assistant_chapter_workflows workflow
                  ON workflow.id=item.workflow_id
                JOIN novel_chapters chapter ON chapter.id=item.chapter_id
                LEFT JOIN novel_chapter_versions head
                  ON head.id=chapter.head_version_id
                JOIN assistant_messages message
                  ON message.id=?
                 AND message.conversation_id=workflow.conversation_id
                JOIN assistant_conversations conversation
                  ON conversation.id=message.conversation_id
                WHERE item.id=? AND item.workflow_id=?
                  AND item.status='running' AND workflow.user_id=?
                  AND message.status='running' AND message.claim_token=?
                  AND conversation.user_id=workflow.user_id
                  AND conversation.project_id=workflow.project_id
                """,
                (
                    message_id,
                    item_id,
                    workflow_id,
                    user_id,
                    claim_token,
                ),
            ).fetchone()
        if not row:
            raise ValueError("连续章节工作项或任务租约已经失效")
        chapter_path = Path(str(row["cache_content_path"]))
        previous_content = _read_text(row["head_content_path"])
        if row["head_version_id"] and _sha256(previous_content) != str(
            row["head_content_hash"] or ""
        ):
            raise ValueError("当前章节文件与工作版本不一致，请先恢复后再续写")
        token = secrets.token_hex(16)
        version_id = uuid.uuid4().hex
        version_path = (
            chapter_path.parent / "versions" / f"assistant-series-{token}.txt"
        )
        _atomic_write(version_path, clean_content, token)
        now = utc_now()
        count = len(clean_content)
        effective_count = effective_char_count(clean_content)
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                latest = connection.execute(
                    """
                    SELECT item.status, item.chapter_id,
                           chapter.head_version_id
                    FROM assistant_chapter_workflow_items item
                    JOIN novel_chapters chapter ON chapter.id=item.chapter_id
                    WHERE item.id=? AND item.workflow_id=?
                    """,
                    (item_id, workflow_id),
                ).fetchone()
                if not latest or str(latest["status"]) != "running":
                    raise ValueError("连续章节工作项状态已经变化")
                if str(latest["head_version_id"] or "") != str(
                    row["head_version_id"] or ""
                ):
                    raise ValueError("章节正文版本已经变化，请从断点重新执行")
                connection.execute(
                    """
                    INSERT INTO novel_chapter_versions(
                        id, chapter_id, kind, content_path, char_count,
                        created_at, parent_version_id, source,
                        content_hash, change_summary, created_by,
                        effective_char_count
                    ) VALUES (
                        ?, ?, 'assistant_series', ?, ?, ?, ?,
                        'assistant_chat', ?, ?, 'assistant', ?
                    )
                    """,
                    (
                        version_id,
                        row["chapter_id"],
                        str(version_path),
                        count,
                        now,
                        row["head_version_id"],
                        _sha256(clean_content),
                        str(change_summary or "连续章节创作")[:1000],
                        effective_count,
                    ),
                )
                advanced = self.database.set_chapter_head_in_transaction(
                    connection,
                    user_id=user_id,
                    project_id=str(row["project_id"]),
                    chapter_id=str(row["chapter_id"]),
                    version_id=version_id,
                    expected_old_head_version_id=str(
                        row["head_version_id"] or ""
                    ),
                    now=now,
                )
                if not advanced:
                    raise ValueError("连续章节版本没有成为 main HEAD")
                cursor = connection.execute(
                    """
                    UPDATE assistant_chapter_workflow_items
                    SET version_id=?, status='completed', error='',
                        finished_at=?
                    WHERE id=? AND workflow_id=? AND status='running'
                    """,
                    (version_id, now, item_id, workflow_id),
                )
                if cursor.rowcount != 1:
                    raise ValueError("连续章节工作项状态已经变化")
                connection.commit()
            except Exception:
                connection.rollback()
                version_path.unlink(missing_ok=True)
                raise
        try:
            _atomic_write(chapter_path, clean_content, secrets.token_hex(16))
        except Exception:
            logger.warning(
                "failed to refresh non-authoritative chapter cache",
                exc_info=True,
            )
        try:
            targets = self.database.list_memory_refresh_targets(
                user_id=user_id,
                project_id=str(row["project_id"]),
                chapter_id=str(row["chapter_id"]),
            )
            for target in targets:
                self.database.create_memory_extraction_job(
                    user_id=user_id,
                    project_id=str(row["project_id"]),
                    chapter_id=str(target["chapter_id"]),
                    version_id=str(target["head_version_id"]),
                    provider=str(row["provider"]),
                    model=str(row["model"]),
                    credential_source=str(row["credential_source"]),
                )
        except Exception:
            logger.exception(
                "failed to queue chapter workflow memory refresh workflow=%s item=%s",
                workflow_id,
                item_id,
            )
        self._refresh_chapter_workflow_counts(workflow_id)
        return {
            "workflow_id": workflow_id,
            "item_id": item_id,
            "sequence": int(row["sequence"]),
            "chapter_id": str(row["chapter_id"]),
            "version_id": version_id,
            "char_count": count,
        }

    def pause_chapter_workflow_for_agent(
        self,
        *,
        user_id: int,
        workflow_id: str,
        item_id: str,
        error: str,
    ) -> Dict[str, Any]:
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE assistant_chapter_workflow_items
                SET status='failed', error=?, finished_at=?
                WHERE id=? AND workflow_id=? AND status='running'
                """,
                (str(error)[:2000], now, item_id, workflow_id),
            )
            connection.execute(
                """
                UPDATE assistant_chapter_workflows
                SET status='paused', error=?, updated_at=?
                WHERE id=? AND user_id=?
                """,
                (str(error)[:2000], now, workflow_id, user_id),
            )
            connection.commit()
        return self.get_chapter_workflow(
            user_id=user_id,
            workflow_id=workflow_id,
        )

    def _refresh_chapter_workflow_counts(self, workflow_id: str) -> None:
        now = utc_now()
        with self.database.connection() as connection:
            counts = connection.execute(
                """
                SELECT COUNT(*) AS total,
                       SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END)
                           AS completed
                FROM assistant_chapter_workflow_items
                WHERE workflow_id=?
                """,
                (workflow_id,),
            ).fetchone()
            total = int(counts["total"] or 0)
            completed = int(counts["completed"] or 0)
            status = "completed" if total and completed == total else "running"
            connection.execute(
                """
                UPDATE assistant_chapter_workflows
                SET completed_count=?, status=?, error='', updated_at=?,
                    finished_at=?
                WHERE id=?
                """,
                (
                    completed,
                    status,
                    now,
                    now if status == "completed" else None,
                    workflow_id,
                ),
            )
            connection.commit()

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
                  AND scope_type IN ('project', 'chapter')
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
        prepared = self.conversations.get(
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
            "theme",
            "story_promise",
            "target_audience",
            "core_appeal",
            "ending_constraint",
            "world_setting",
            "style_guide",
        )
        if any(str(project.get(key) or "").strip() for key in supporting_fields):
            return True
        return bool(self.database.list_novel_characters(user_id, project_id))
