from __future__ import annotations

import logging
import secrets
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .agent_capabilities import (
    WRITE_CHAPTER,
    CREATE_TECHNIQUE_CARD,
    MANAGE_CHAPTERS,
    MANAGE_NOTES,
    PROPOSE_SETTINGS_PATCH,
    PROPOSE_STORY_PLAN,
    normalize_requested_agent_role,
    resolve_agent_dispatch,
)
from .assistant_application import AssistantApplicationMixin
from .assistant_chapter_workflow import AssistantChapterWorkflowMixin
from .assistant_context import AssistantContextMixin
from .assistant_file_ops import (
    atomic_write as _atomic_write,
    clean_title as _clean_title,
    read_text as _read_text,
    sha256_text as _sha256,
)
from .agent_intent import AssistantIntentDecision
from .assistant_chat_schema import (
    AssistantChapterPatch,
    AssistantChatResponse,
    AssistantDraftProposal,
    AssistantNotePatch,
    AssistantSettingsPatch,
    AssistantStoryPlanProposal,
    AssistantTechniquePatch,
    AssistantVersionRestoreProposal,
)
from .assistant_conversation_repository import AssistantConversationRepository
from .assistant_result import (
    decode_agent_step,
    decode_message,
    decode_tool_call,
    normalize_assistant_response,
)
from .conversation_memory import (
    compile_conversation_context,
)
from .db import Database, utc_after, utc_now
from .json_support import (
    dump_canonical_json as _json,
    load_json as _load_json,
)
from .model_routing import normalize_quality_mode
from .story_planning_service import StoryPlanningService
from .structured_settings import (
    StructuredSettingsEditor,
    filter_structured_edits,
)
from .text_metrics import effective_char_count


logger = logging.getLogger(__name__)


class AssistantChatService(
    AssistantApplicationMixin,
    AssistantChapterWorkflowMixin,
    AssistantContextMixin,
):
    def __init__(
        self,
        database: Database,
        novels_dir: Path,
        documents_dir: Path,
    ):
        self.database = database
        self.conversations = AssistantConversationRepository(database)
        self.novels_dir = novels_dir
        self.documents_dir = documents_dir
        self.structured_settings_editor = StructuredSettingsEditor(database)

    @staticmethod
    def _append_agent_event(
        connection: Any,
        *,
        message_id: str,
        event_type: str,
        phase: str,
        status: str,
        label: str,
        payload: Mapping[str, Any] | None = None,
        created_at: str | None = None,
    ) -> int:
        row = connection.execute(
            """
            SELECT run_sequence FROM assistant_messages WHERE id=?
            """,
            (message_id,),
        ).fetchone()
        if not row:
            raise ValueError("对话消息不存在")
        sequence = int(row["run_sequence"] or 0) + 1
        connection.execute(
            """
            UPDATE assistant_messages
            SET run_sequence=?, run_state=?, run_state_label=?,
                stream_sequence=stream_sequence+1
            WHERE id=?
            """,
            (sequence, phase, label[:160], message_id),
        )
        connection.execute(
            """
            INSERT INTO assistant_agent_events(
                id, assistant_message_id, sequence, event_type, phase,
                status, label, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uuid.uuid4().hex,
                message_id,
                sequence,
                event_type[:120],
                phase[:80],
                status[:40],
                label[:160],
                _json(dict(payload or {})),
                created_at or utc_now(),
            ),
        )
        return sequence

    def transition_agent_run(
        self,
        *,
        message_id: str,
        claim_token: str,
        phase: str,
        event_type: str,
        status: str,
        label: str,
        payload: Mapping[str, Any] | None = None,
    ) -> bool:
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
                return False
            self._append_agent_event(
                connection,
                message_id=message_id,
                event_type=event_type,
                phase=phase,
                status=status,
                label=label,
                payload=payload,
            )
            connection.commit()
        return True

    def is_message_cancel_requested(
        self, *, message_id: str, claim_token: str
    ) -> bool:
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT cancel_requested_at
                FROM assistant_messages
                WHERE id=? AND role='assistant' AND status='running'
                  AND claim_token=?
                """,
                (message_id, claim_token),
            ).fetchone()
        return bool(row and row["cancel_requested_at"])

    def request_message_cancellation(
        self,
        *,
        user_id: int,
        message_id: str,
        reason: str = "作者停止生成",
    ) -> Dict[str, Any]:
        now = utc_now()
        clean_reason = str(reason or "作者停止生成")[:500]
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT m.status, m.run_state, m.cancel_requested_at
                FROM assistant_messages m
                JOIN assistant_conversations c
                  ON c.id=m.conversation_id
                WHERE m.id=? AND m.role='assistant' AND c.user_id=?
                """,
                (message_id, user_id),
            ).fetchone()
            if not row:
                connection.rollback()
                raise ValueError("对话消息不存在")
            current_status = str(row["status"])
            if current_status in {"completed", "failed"}:
                connection.commit()
                return {
                    "message_id": message_id,
                    "status": current_status,
                    "run_state": str(row["run_state"] or current_status),
                    "cancelled": str(row["run_state"]) == "cancelled",
                }
            if current_status == "queued":
                connection.execute(
                    """
                    UPDATE assistant_messages
                    SET status='failed', cancel_requested_at=?,
                        cancel_reason=?, error='已停止生成', finished_at=?,
                        claim_token=NULL, lease_expires_at=NULL
                    WHERE id=? AND status='queued'
                    """,
                    (now, clean_reason, now, message_id),
                )
                self._append_agent_event(
                    connection,
                    message_id=message_id,
                    event_type="run.cancelled",
                    phase="cancelled",
                    status="cancelled",
                    label="已停止",
                    payload={"reason": clean_reason},
                    created_at=now,
                )
                connection.commit()
                return {
                    "message_id": message_id,
                    "status": "failed",
                    "run_state": "cancelled",
                    "cancelled": True,
                }
            connection.execute(
                """
                UPDATE assistant_messages
                SET cancel_requested_at=COALESCE(cancel_requested_at, ?),
                    cancel_reason=CASE WHEN cancel_reason='' THEN ?
                                       ELSE cancel_reason END
                WHERE id=? AND status='running'
                """,
                (now, clean_reason, message_id),
            )
            if not row["cancel_requested_at"]:
                self._append_agent_event(
                    connection,
                    message_id=message_id,
                    event_type="run.cancelling",
                    phase="cancelling",
                    status="running",
                    label="正在停止",
                    payload={"reason": clean_reason},
                    created_at=now,
                )
            connection.commit()
        return {
            "message_id": message_id,
            "status": "running",
            "run_state": "cancelling",
            "cancelled": False,
        }

    def cancel_running_message(
        self,
        *,
        message_id: str,
        claim_token: str,
        reason: str = "作者停止生成",
    ) -> bool:
        now = utc_now()
        clean_reason = str(reason or "作者停止生成")[:500]
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE assistant_messages
                SET status='failed', cancel_requested_at=COALESCE(
                        cancel_requested_at, ?
                    ), cancel_reason=CASE WHEN cancel_reason='' THEN ?
                                          ELSE cancel_reason END,
                    error='已停止生成', finished_at=?, claim_token=NULL,
                    lease_expires_at=NULL
                WHERE id=? AND role='assistant' AND status='running'
                  AND claim_token=?
                """,
                (now, clean_reason, now, message_id, claim_token),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return False
            connection.execute(
                """
                UPDATE assistant_tool_calls
                SET status='failed', error='任务已停止', finished_at=?
                WHERE assistant_message_id=? AND status='running'
                """,
                (now, message_id),
            )
            self._append_agent_event(
                connection,
                message_id=message_id,
                event_type="run.cancelled",
                phase="cancelled",
                status="cancelled",
                label="已停止",
                payload={"reason": clean_reason},
                created_at=now,
            )
            connection.commit()
        return True

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
                       c.title, c.quality_mode
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
                    quality_mode, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    normalize_quality_mode(source["quality_mode"]),
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
                        credential_source, quality_mode, response_json, input_tokens,
                        output_tokens, created_at, started_at, finished_at,
                        run_state, run_state_label
                    ) VALUES (?, ?, ?, ?, ?, 'completed', ?, ?, ?, ?, ?, 0, 0,
                              ?, NULL, ?, 'completed', '已完成')
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
                        normalize_quality_mode(original["quality_mode"]),
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
        quality_mode: str = "",
    ) -> str:
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("请输入想和 AI 讨论的问题")
        if credential_source not in {"default", "personal"}:
            raise ValueError("不支持的 API 凭据来源")
        requested_agent_role = normalize_requested_agent_role(agent_role)
        conversation = self.conversations.get(
            user_id=user_id, conversation_id=conversation_id
        )
        if not conversation:
            raise ValueError("对话不存在")
        clean_quality_mode = normalize_quality_mode(
            quality_mode or conversation.get("quality_mode")
        )
        if (
            quality_mode
            and str(quality_mode).strip().lower() != clean_quality_mode
        ):
            raise ValueError("不支持的模型强度")
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
            connection.execute(
                """
                INSERT INTO assistant_messages(
                    id, conversation_id, role, content, status,
                    provider, model, credential_source, quality_mode,
                    created_at, finished_at, run_state, run_state_label
                ) VALUES (?, ?, 'user', ?, 'completed', '', '', 'default',
                          ?, ?, ?, 'completed', '')
                """,
                (
                    user_message_id,
                    conversation_id,
                    clean_question,
                    clean_quality_mode,
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
                    quality_mode, context_snapshot_json, created_at,
                    run_state, run_state_label
                ) VALUES (?, ?, 'assistant', ?, 'queued', ?, ?, ?, ?, ?, ?,
                          'queued', '等待处理')
                """,
                (
                    assistant_message_id,
                    conversation_id,
                    user_message_id,
                    provider,
                    model,
                    credential_source,
                    clean_quality_mode,
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
                    SET title=?, quality_mode=?, updated_at=? WHERE id=?
                    """,
                    (
                        generated_title,
                        clean_quality_mode,
                        now,
                        conversation_id,
                    ),
                )
            else:
                connection.execute(
                    """
                    UPDATE assistant_conversations
                    SET quality_mode=?, updated_at=? WHERE id=?
                    """,
                    (clean_quality_mode, now, conversation_id),
                )
            connection.execute(
                """
                INSERT INTO user_model_preferences(
                    user_id, default_quality_mode, created_at, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    default_quality_mode=excluded.default_quality_mode,
                    updated_at=excluded.updated_at
                """,
                (user_id, clean_quality_mode, now, now),
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
        conversation = self.conversations.get(
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
        previous_chapter_quote = (
            self._validate_quote(
                user_id=user_id,
                conversation=conversation,
                quote=dict(quote_row),
            )
            if quote_row and dispatch.intent == "draft_new_chapter"
            else None
        )
        scope_preflight: Dict[str, Any] = {}
        routing_notice = ""
        if dispatch.role == "writer":
            try:
                if dispatch.intent == "draft_new_chapter":
                    conversation, scope_preflight = (
                        self._create_and_bind_next_chapter(
                            user_id=user_id,
                            conversation=conversation,
                        )
                    )
                else:
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
            previous_chapter_quote
            or self._validate_quote(
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
                boundaries.get("auto_advance_main_head")
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
            expired_rows = connection.execute(
                """
                SELECT id, cancel_requested_at, cancel_reason
                FROM assistant_messages
                WHERE role='assistant' AND status='running'
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at<=?
                """,
                (now,),
            ).fetchall()
            for expired in expired_rows:
                expired_id = str(expired["id"])
                connection.execute(
                    """
                    UPDATE assistant_tool_calls
                    SET status='failed',
                        error='任务租约过期，工具执行结果已丢弃',
                        finished_at=?
                    WHERE assistant_message_id=? AND status='running'
                    """,
                    (now, expired_id),
                )
                if expired["cancel_requested_at"]:
                    connection.execute(
                        """
                        UPDATE assistant_messages
                        SET status='failed', started_at=NULL,
                            claim_token=NULL, lease_expires_at=NULL,
                            error='已停止生成', finished_at=?
                        WHERE id=? AND status='running'
                        """,
                        (now, expired_id),
                    )
                    self._append_agent_event(
                        connection,
                        message_id=expired_id,
                        event_type="run.cancelled",
                        phase="cancelled",
                        status="cancelled",
                        label="已停止",
                        payload={
                            "reason": str(
                                expired["cancel_reason"]
                                or "停止请求在恢复任务时生效"
                            )
                        },
                        created_at=now,
                    )
                    continue
                connection.execute(
                    """
                    UPDATE assistant_messages
                    SET status='queued', started_at=NULL, claim_token=NULL,
                        lease_expires_at=NULL,
                        error='上一次处理租约已过期，已自动重新排队'
                    WHERE id=? AND status='running'
                    """,
                    (expired_id,),
                )
                self._append_agent_event(
                    connection,
                    message_id=expired_id,
                    event_type="run.recovered",
                    phase="queued",
                    status="queued",
                    label="已恢复，等待重新处理",
                    payload={"reason": "lease_expired"},
                    created_at=now,
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
                    claim_token=?, lease_expires_at=?,
                    cancel_requested_at=NULL, cancel_reason='',
                    stream_content=''
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
            self._append_agent_event(
                connection,
                message_id=str(row["id"]),
                event_type="run.claimed",
                phase="routing",
                status="running",
                label="正在理解请求",
                payload={"recovered": bool(expired_rows)},
                created_at=now,
            )
            connection.commit()
        item = dict(row)
        item["claim_token"] = claim_token
        item["run_state"] = "routing"
        item["run_state_label"] = "正在理解请求"
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
                SELECT id, rowid, role, content, created_at
                FROM assistant_messages
                WHERE conversation_id=? AND rowid<?
                  AND status='completed'
                  AND content!=''
                ORDER BY rowid
                """,
                (item["conversation_id"], user_row["rowid"]),
            ).fetchall()
            (
                history,
                conversation_memory,
                conversation_memory_state,
            ) = compile_conversation_context(history_rows)
            connection.execute(
                """
                UPDATE assistant_conversations
                SET memory_summary=?, memory_state_json=?,
                    memory_message_count=?
                WHERE id=?
                """,
                (
                    conversation_memory,
                    _json(conversation_memory_state),
                    len(history_rows),
                    item["conversation_id"],
                ),
            )
            connection.commit()
        context = dict(snapshot.get("context") or {})
        context["conversation_id"] = str(item["conversation_id"])
        context["current_user_message_id"] = str(
            item["parent_user_message_id"]
        )
        context["conversation_memory"] = conversation_memory
        context["conversation_memory_state"] = conversation_memory_state
        context["conversation_history_search_available"] = True
        return {
            "context": context,
            "sources": list(snapshot.get("sources") or []),
            "history": history,
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
        if action not in {"call_tool", "finish"}:
            raise ValueError("Agent 步骤动作无效")
        if outcome_status not in {
            "completed",
            "denied",
            "failed",
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
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT m.context_snapshot_json, c.user_id
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
            snapshot = _load_json(row["context_snapshot_json"], {})
            normalized = normalize_assistant_response(
                response=response,
                sources=[
                    *list(snapshot.get("sources") or []),
                    *list(response.accessed_sources or []),
                ],
                context=dict(snapshot.get("context") or {}),
            )
            cursor = connection.execute(
                """
                UPDATE assistant_messages
                SET status='completed', content=?, response_json=?,
                    stream_content=?, stream_sequence=stream_sequence+1,
                    raw_response=?, input_tokens=?, output_tokens=?,
                    provider=?, model=?,
                    finished_at=?, claim_token=NULL, lease_expires_at=NULL,
                    error=NULL
                WHERE id=? AND role='assistant' AND status='running'
                  AND claim_token=? AND cancel_requested_at IS NULL
                """,
                (
                    normalized["answer"],
                    _json(normalized),
                    normalized["answer"],
                    response.raw_response,
                    response.input_tokens,
                    response.output_tokens,
                    response.provider,
                    response.model,
                    utc_now(),
                    message_id,
                    claim_token,
                ),
            )
            if cursor.rowcount == 1:
                self._append_agent_event(
                    connection,
                    message_id=message_id,
                    event_type="run.completed",
                    phase="completed",
                    status="completed",
                    label="已完成",
                    payload={
                        "provider": response.provider,
                        "model": response.model,
                    },
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
            self._auto_apply_completed_message(
                user_id=int(row["user_id"]),
                assistant_message_id=message_id,
                response=normalized,
                context=dict(snapshot.get("context") or {}),
            )
        return accepted

    def set_message_stream(
        self,
        *,
        message_id: str,
        claim_token: str,
        content: str,
    ) -> bool:
        bounded = str(content or "")
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE assistant_messages
                SET stream_content=?, stream_sequence=stream_sequence+1
                WHERE id=? AND role='assistant' AND status='running'
                  AND claim_token=? AND stream_content<>?
                """,
                (
                    bounded,
                    message_id,
                    claim_token,
                    bounded,
                ),
            )
            connection.commit()
        return cursor.rowcount == 1

    def get_message_stream_state(
        self,
        *,
        user_id: int,
        message_id: str,
        after_event_sequence: int = 0,
    ) -> Optional[Dict[str, Any]]:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT m.id, m.conversation_id, m.status, m.content,
                       m.stream_content, m.stream_sequence, m.error,
                       m.response_json, m.provider, m.model,
                       m.run_state, m.run_state_label, m.run_sequence,
                       m.cancel_requested_at,
                       c.scope_type, c.project_id, c.document_id,
                       c.novel_chapter_id, c.reference_chapter_id
                FROM assistant_messages m
                JOIN assistant_conversations c
                  ON c.id=m.conversation_id
                WHERE m.id=? AND m.role='assistant' AND c.user_id=?
                """,
                (message_id, user_id),
            ).fetchone()
            event_rows = (
                connection.execute(
                    """
                    SELECT sequence, event_type, phase, status, label,
                           payload_json, created_at
                    FROM assistant_agent_events
                    WHERE assistant_message_id=? AND sequence>?
                    ORDER BY sequence
                    """,
                    (message_id, max(0, int(after_event_sequence))),
                ).fetchall()
                if row
                else []
            )
        if not row:
            return None
        result = dict(row)
        response = _load_json(result.pop("response_json"), {})
        auto_commit_status = str(
            (response.get("auto_commit") or {}).get("status") or ""
        )
        settings_patch_status = str(
            response.get("settings_patch_status") or ""
        )
        story_plan_status = str(response.get("story_plan_status") or "")
        chapter_patch_status = str(
            response.get("chapter_patch_status") or ""
        )
        note_patch_status = str(response.get("note_patch_status") or "")
        version_restore_status = str(
            response.get("version_restore_status") or ""
        )
        technique_patch_status = str(
            response.get("technique_patch_status") or ""
        )
        application_pending = any(
            status == "pending"
            for status in (
                auto_commit_status,
                settings_patch_status,
                story_plan_status,
                chapter_patch_status,
                note_patch_status,
                version_restore_status,
                technique_patch_status,
            )
        )
        message_status = str(result["status"])
        result["content"] = str(
            result.get("stream_content")
            or result.get("content")
            or ""
        )
        result["terminal"] = (
            message_status == "failed"
            or (
                message_status == "completed"
                and not application_pending
            )
        )
        result["cancel_requested"] = bool(
            result.pop("cancel_requested_at", None)
        )
        result["cancelled"] = str(result.get("run_state") or "") == (
            "cancelled"
        )
        result["events"] = [
            {
                **{
                    key: value
                    for key, value in dict(event).items()
                    if key != "payload_json"
                },
                "payload": _load_json(event["payload_json"], {}),
            }
            for event in event_rows
        ]
        return result

    def _auto_commit_completed_message(
        self,
        *,
        user_id: int,
        assistant_message_id: str,
        response: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> None:
        boundaries = dict(context.get("assistant_boundaries") or {})
        if not boundaries.get("auto_advance_main_head"):
            return
        kind = ""
        try:
            if isinstance(response.get("draft"), dict):
                kind = "draft"
                result = self.commit_draft_to_head(
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
        self._queue_head_memory_refresh(
            user_id=user_id,
            assistant_message_id=assistant_message_id,
            result=result,
        )

    def _queue_head_memory_refresh(
        self,
        *,
        user_id: int,
        assistant_message_id: str,
        result: Mapping[str, Any],
    ) -> None:
        try:
            with self.database.connection() as connection:
                runtime = connection.execute(
                    """
                    SELECT provider, model, credential_source
                    FROM assistant_messages
                    WHERE id=? AND role='assistant'
                    """,
                    (assistant_message_id,),
                ).fetchone()
            if not runtime:
                return
            targets = self.database.list_memory_refresh_targets(
                user_id=user_id,
                project_id=str(result["project_id"]),
                chapter_id=str(result["chapter_id"]),
            )
            for target in targets:
                self.database.create_memory_extraction_job(
                    user_id=user_id,
                    project_id=str(result["project_id"]),
                    chapter_id=str(target["chapter_id"]),
                    version_id=str(target["head_version_id"]),
                    provider=str(runtime["provider"]),
                    model=str(runtime["model"]),
                    credential_source=str(runtime["credential_source"]),
                )
        except Exception:
            logger.exception(
                "failed to queue assistant chapter memory extraction "
                "message=%s chapter=%s",
                assistant_message_id,
                result.get("chapter_id"),
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

    def _auto_apply_completed_message(
        self,
        *,
        user_id: int,
        assistant_message_id: str,
        response: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> None:
        """Apply reversible settings work before exposing a completed turn."""

        boundaries = dict(context.get("assistant_boundaries") or {})
        operations = (
            (
                "technique_patch",
                bool(boundaries.get("auto_apply_techniques")),
                lambda: self.apply_technique_patch_candidate(
                    user_id=user_id,
                    assistant_message_id=assistant_message_id,
                ),
            ),
            (
                "version_restore",
                bool(boundaries.get("auto_advance_main_head")),
                lambda: self.apply_version_restore_candidate(
                    user_id=user_id,
                    assistant_message_id=assistant_message_id,
                ),
            ),
            (
                "note_patch",
                bool(boundaries.get("auto_apply_notes")),
                lambda: self.apply_note_patch_candidate(
                    user_id=user_id,
                    assistant_message_id=assistant_message_id,
                ),
            ),
            (
                "chapter_patch",
                bool(boundaries.get("auto_apply_chapter_metadata")),
                lambda: self.apply_chapter_patch_candidate(
                    user_id=user_id,
                    assistant_message_id=assistant_message_id,
                ),
            ),
            (
                "settings_patch",
                bool(boundaries.get("auto_apply_settings")),
                lambda: self.apply_settings_candidate(
                    user_id=user_id,
                    assistant_message_id=assistant_message_id,
                ),
            ),
            (
                "story_plan",
                bool(boundaries.get("auto_apply_story_plan")),
                lambda: self.apply_story_plan_candidate(
                    user_id=user_id,
                    assistant_message_id=assistant_message_id,
                ),
            ),
        )
        for kind, enabled, apply_operation in operations:
            if not enabled or not isinstance(response.get(kind), dict):
                continue
            try:
                applied = apply_operation()
                if kind == "version_restore" and isinstance(applied, Mapping):
                    self._queue_head_memory_refresh(
                        user_id=user_id,
                        assistant_message_id=assistant_message_id,
                        result=applied,
                    )
            except Exception as exc:
                logger.exception(
                    "failed to auto-apply assistant %s message=%s",
                    kind,
                    assistant_message_id,
                )
                self._record_auto_apply_failure(
                    assistant_message_id=assistant_message_id,
                    kind=kind,
                    error=str(exc),
                )

    def _record_auto_apply_failure(
        self,
        *,
        assistant_message_id: str,
        kind: str,
        error: str,
    ) -> None:
        status_field = {
            "chapter_patch": "chapter_patch_status",
            "settings_patch": "settings_patch_status",
            "story_plan": "story_plan_status",
            "note_patch": "note_patch_status",
            "version_restore": "version_restore_status",
            "technique_patch": "technique_patch_status",
        }.get(kind)
        if not status_field:
            return
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
            stored = _load_json(row["response_json"], {})
            stored[status_field] = "failed"
            stored[f"{kind}_error"] = error[:1000] or "自动写入失败"
            stored[f"{kind}_updated_at"] = utc_now()
            connection.execute(
                """
                UPDATE assistant_messages SET response_json=?
                WHERE id=?
                """,
                (_json(stored), assistant_message_id),
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
            connection.execute("BEGIN IMMEDIATE")
            owner = connection.execute(
                """
                SELECT cancel_requested_at, cancel_reason
                FROM assistant_messages
                WHERE id=? AND role='assistant' AND status='running'
                  AND claim_token=?
                """,
                (message_id, claim_token),
            ).fetchone()
            if not owner:
                connection.rollback()
                return False
            if owner["cancel_requested_at"]:
                connection.rollback()
                return self.cancel_running_message(
                    message_id=message_id,
                    claim_token=claim_token,
                    reason=str(owner["cancel_reason"] or "作者停止生成"),
                )
            cursor = connection.execute(
                """
                UPDATE assistant_messages
                SET status='failed', error=?, input_tokens=?,
                    output_tokens=?, finished_at=?, claim_token=NULL,
                    lease_expires_at=NULL,
                    stream_sequence=stream_sequence+1
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
            if cursor.rowcount == 1:
                connection.execute(
                    """
                    UPDATE assistant_tool_calls
                    SET status='failed', error=?, finished_at=?
                    WHERE assistant_message_id=? AND status='running'
                    """,
                    (error[:2000], utc_now(), message_id),
                )
                self._append_agent_event(
                    connection,
                    message_id=message_id,
                    event_type="run.failed",
                    phase="failed",
                    status="failed",
                    label="生成失败",
                    payload={"error": error[:500]},
                )
            connection.commit()
        return cursor.rowcount == 1

    def release_claim(
        self, message_id: str, claim_token: str, error: str
    ) -> bool:
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            owner = connection.execute(
                """
                SELECT cancel_requested_at, cancel_reason
                FROM assistant_messages
                WHERE id=? AND role='assistant' AND status='running'
                  AND claim_token=?
                """,
                (message_id, claim_token),
            ).fetchone()
            if not owner:
                connection.rollback()
                return False
            if owner["cancel_requested_at"]:
                connection.rollback()
                return self.cancel_running_message(
                    message_id=message_id,
                    claim_token=claim_token,
                    reason=str(owner["cancel_reason"] or "作者停止生成"),
                )
            cursor = connection.execute(
                """
                UPDATE assistant_messages
                SET status='queued', started_at=NULL, claim_token=NULL,
                    lease_expires_at=NULL, error=?, stream_content=''
                WHERE id=? AND role='assistant' AND status='running'
                  AND claim_token=?
                """,
                (error[:2000], message_id, claim_token),
            )
            if cursor.rowcount == 1:
                connection.execute(
                    """
                    UPDATE assistant_tool_calls
                    SET status='failed',
                        error='任务已重新排队，当前工具结果已丢弃',
                        finished_at=?
                    WHERE assistant_message_id=? AND status='running'
                    """,
                    (utc_now(), message_id),
                )
                self._append_agent_event(
                    connection,
                    message_id=message_id,
                    event_type="run.requeued",
                    phase="queued",
                    status="queued",
                    label="等待重新处理",
                    payload={"reason": error[:500]},
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
        result = decode_message(dict(row))
        result["tool_calls"] = [
            decode_tool_call(dict(tool_row))
            for tool_row in tool_rows
        ]
        result["agent_steps"] = [
            decode_agent_step(dict(step_row))
            for step_row in step_rows
        ]
        return result
