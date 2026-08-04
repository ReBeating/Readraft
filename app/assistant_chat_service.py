from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .agent_capabilities import (
    WRITE_CHAPTER,
    CREATE_CHAPTER,
    CREATE_TECHNIQUE_CARD,
    MANAGE_CHAPTERS,
    MANAGE_NOTES,
    PROPOSE_SETTINGS_PATCH,
    PROPOSE_STORY_PLAN,
    agent_capabilities,
    agent_manifest,
    normalize_requested_agent_role,
    resolve_agent_dispatch,
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
from .assistant_result import (
    compact_analysis,
    decode_agent_step,
    decode_message,
    decode_tool_call,
    normalize_assistant_response,
)
from .context_compiler import build_writing_context_snapshot
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


CONVERSATION_SCOPES = {
    "project",
    "chapter",
    "document",
    "reference_chapter",
}

MAX_USER_MESSAGE_CHARS = 100_000
SETTING_FIELD_LABELS = {
    "title": "书名",
    "genre": "题材",
    "premise": "一句话故事",
    "theme": "主题",
    "story_promise": "读者体验",
    "target_audience": "目标读者",
    "core_appeal": "核心吸引力",
    "ending_constraint": "结局约束",
    "world_setting": "世界概述",
    "style_guide": "叙事风格规范",
    "point_of_view": "叙事视角",
    "archive_rules": "分类作品资料",
    "structured_edits": "结构化资料修改",
}


def _author_explicitly_requested_deletion(question: str) -> bool:
    clean = re.sub(r"\s+", " ", str(question or "")).strip()
    clean = re.sub(
        r"(?:不要|别|不能|不可|不许|无需|不用).{0,8}"
        r"(?:删掉|删除|移除|去掉|撤销)",
        "",
        clean,
    )
    return bool(re.search(r"(?:删掉|删除|移除|去掉|撤销)", clean))


def _author_explicitly_requested_restore(question: str) -> bool:
    clean = str(question or "").strip()
    clean = re.sub(
        r"(?:不要|别|不能|不可|不许|无需|不用).{0,10}"
        r"(?:恢复|还原|回退|退回|撤回)",
        "",
        clean,
    )
    return bool(re.search(r"(?:恢复|还原|回退|退回|撤回到)", clean))


def _author_explicitly_requested_technique_card(question: str) -> bool:
    clean = str(question or "").strip()
    clean = re.sub(
        r"(?:不要|别|不能|不可|不许|无需|不用).{0,12}"
        r"(?:技法卡|保存技法|提炼技法|沉淀技法)",
        "",
        clean,
    )
    return bool(
        re.search(
            r"(?:技法卡|保存.{0,8}技法|提炼.{0,8}技法|沉淀.{0,8}技法)",
            clean,
        )
    )

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chapter_metadata_revision(chapter: Mapping[str, Any]) -> str:
    content = json.dumps(
        {
            "chapter_id": str(chapter["id"]),
            "position": int(chapter["position"]),
            "title": str(chapter.get("title") or ""),
            "outline": str(chapter.get("outline") or ""),
            "key_points": str(chapter.get("key_points") or ""),
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return _sha256(content)


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
        clean_outline = str(outline or "").strip()[:6000]
        clean_key_points = str(key_points or "").strip()[:6000]

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

        prepared = self.get_conversation(
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
        conversation = self.get_conversation(
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
                                str(chapter.get("outline") or "")[:6000],
                                str(chapter.get("key_points") or "")[:6000],
                                str(chapter.get("instruction") or "")[:6000],
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
            conversation = self.get_conversation(
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
                prepared = self.get_conversation(
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
        if len(clean_content) > 200_000:
            raise ValueError("连续章节单章不能超过 200000 字")
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
                        quality_status, effective_char_count, hard_issue_count
                    ) VALUES (
                        ?, ?, 'assistant_series', ?, ?, ?, ?,
                        'assistant_chat', ?, ?, 'assistant', 'pass', ?, 0
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
            preference = connection.execute(
                """
                SELECT default_quality_mode
                FROM user_model_preferences
                WHERE user_id=?
                """,
                (user_id,),
            ).fetchone()
            quality_mode = normalize_quality_mode(
                preference["default_quality_mode"]
                if preference
                else "standard"
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
                    _clean_title(title, fallback),
                    quality_mode,
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
            if len(clean_question) > MAX_USER_MESSAGE_CHARS:
                connection.rollback()
                raise ValueError(
                    "单条消息不能超过 "
                    f"{MAX_USER_MESSAGE_CHARS:,} 个字符"
                )
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
            decoded = decode_tool_call(dict(tool_row))
            tool_calls_by_message.setdefault(
                str(tool_row["assistant_message_id"]), []
            ).append(decoded)
        steps_by_message: dict[str, list[dict[str, Any]]] = {}
        for step_row in step_rows:
            decoded = decode_agent_step(dict(step_row))
            steps_by_message.setdefault(
                str(step_row["assistant_message_id"]), []
            ).append(decoded)
        result = dict(row)
        result["messages"] = []
        for message in messages:
            decoded = decode_message(dict(message))
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

    def set_conversation_quality_mode(
        self,
        *,
        user_id: int,
        conversation_id: str,
        quality_mode: str,
    ) -> str:
        clean_quality_mode = normalize_quality_mode(quality_mode)
        if str(quality_mode or "").strip().lower() != clean_quality_mode:
            raise ValueError("不支持的模型强度")
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
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
                raise ValueError("AI 正在回复，请完成后再切换模型强度")
            cursor = connection.execute(
                """
                UPDATE assistant_conversations
                SET quality_mode=?, updated_at=?
                WHERE id=? AND user_id=?
                """,
                (
                    clean_quality_mode,
                    now,
                    conversation_id,
                    user_id,
                ),
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
                (user_id, clean_quality_mode, now, now),
            )
            connection.commit()
        return clean_quality_mode

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
        if len(clean_question) > MAX_USER_MESSAGE_CHARS:
            raise ValueError(
                "单条消息不能超过 "
                f"{MAX_USER_MESSAGE_CHARS:,} 个字符"
            )
        if credential_source not in {"default", "personal"}:
            raise ValueError("不支持的 API 凭据来源")
        requested_agent_role = normalize_requested_agent_role(agent_role)
        conversation = self.get_conversation(
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
        bounded = str(content or "")[:30_000]
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


    def apply_chapter_patch_candidate(
        self,
        *,
        user_id: int,
        assistant_message_id: str,
    ) -> Dict[str, Any]:
        """Atomically update chapter metadata and ordering on main."""

        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT m.response_json, m.context_snapshot_json,
                       parent.content AS author_request,
                       c.id AS conversation_id, c.project_id
                FROM assistant_messages m
                LEFT JOIN assistant_messages parent
                  ON parent.id=m.parent_user_message_id
                JOIN assistant_conversations c
                  ON c.id=m.conversation_id
                JOIN novel_projects p ON p.id=c.project_id
                JOIN work_versions version
                  ON version.project_id=c.project_id
                 AND version.ref_type='branch'
                 AND version.ref_name='main'
                 AND version.is_editable=1
                JOIN works work
                  ON work.id=version.work_id AND work.user_id=c.user_id
                WHERE m.id=? AND c.user_id=? AND p.user_id=?
                  AND m.role='assistant' AND m.status='completed'
                  AND c.scope_type IN ('project', 'chapter')
                """,
                (assistant_message_id, user_id, user_id),
            ).fetchone()
            if not row:
                connection.rollback()
                raise ValueError("这条回复没有可应用的章节修改")
            response = _load_json(row["response_json"], {})
            if response.get("chapter_patch_status") == "applied":
                connection.commit()
                return {
                    "project_id": str(row["project_id"]),
                    "conversation_id": str(row["conversation_id"]),
                    "changed_chapter_ids": list(
                        response.get("chapter_patch_changed_ids") or []
                    ),
                    "already_applied": True,
                }
            raw_patch = response.get("chapter_patch")
            if not isinstance(raw_patch, dict):
                connection.rollback()
                raise ValueError("这条回复没有可应用的章节修改")
            try:
                patch = AssistantChapterPatch.model_validate(raw_patch)
            except ValueError as exc:
                connection.rollback()
                raise ValueError("章节修改结构无效") from exc
            deleted_ids = {
                edit.chapter_id for edit in patch.edits if edit.delete
            }
            if deleted_ids and not _author_explicitly_requested_deletion(
                str(row["author_request"] or "")
            ):
                connection.rollback()
                raise ValueError("删除章节前必须由作者明确提出删除目标")
            snapshot = _load_json(row["context_snapshot_json"], {})
            context = dict(snapshot.get("context") or {})
            capabilities = {
                str(value)
                for value in (
                    (context.get("agent") or {}).get("capabilities") or []
                )
            }
            if MANAGE_CHAPTERS not in capabilities:
                connection.rollback()
                raise ValueError("生成这条回复的内部任务没有章节管理权限")
            chapter_rows = connection.execute(
                """
                SELECT id, position, title, outline, key_points,
                       content_path
                FROM novel_chapters
                WHERE project_id=?
                ORDER BY position, created_at, id
                """,
                (row["project_id"],),
            ).fetchall()
            chapters = [dict(chapter) for chapter in chapter_rows]
            by_id = {str(chapter["id"]): chapter for chapter in chapters}
            before: dict[str, dict[str, Any]] = {}
            for edit in patch.edits:
                current = by_id.get(edit.chapter_id)
                if current is None:
                    connection.rollback()
                    raise ValueError("要修改的章节已经不存在")
                if _chapter_metadata_revision(current) != (
                    edit.expected_revision
                ):
                    connection.rollback()
                    raise ValueError(
                        "章节资料在讨论后已经变化，请重新读取后再修改"
                    )
                before[edit.chapter_id] = {
                    "position": int(current["position"]),
                    "title": str(current.get("title") or ""),
                    "outline": str(current.get("outline") or ""),
                    "key_points": str(current.get("key_points") or ""),
                }
                if edit.delete:
                    active = connection.execute(
                        """
                        SELECT 1 FROM generation_jobs
                        WHERE chapter_id=? AND status IN ('queued', 'running')
                        LIMIT 1
                        """,
                        (edit.chapter_id,),
                    ).fetchone()
                    if active:
                        connection.rollback()
                        raise ValueError("AI 正在处理该章，不能删除")
                    continue
                for field in ("title", "outline", "key_points"):
                    value = getattr(edit, field)
                    if value is not None:
                        current[field] = value

            ordered_ids = [
                str(chapter["id"])
                for chapter in chapters
                if str(chapter["id"]) not in deleted_ids
            ]
            for edit in patch.edits:
                if edit.delete or edit.position is None:
                    continue
                target_index = int(edit.position) - 1
                if target_index >= len(ordered_ids):
                    connection.rollback()
                    raise ValueError("章节目标位置超出当前目录范围")
                ordered_ids.remove(edit.chapter_id)
                ordered_ids.insert(target_index, edit.chapter_id)

            now = utc_now()
            for edit in patch.edits:
                if edit.delete:
                    continue
                current = by_id[edit.chapter_id]
                connection.execute(
                    """
                    UPDATE novel_chapters
                    SET title=?, outline=?, key_points=?, updated_at=?
                    WHERE id=? AND project_id=?
                    """,
                    (
                        current["title"],
                        current["outline"],
                        current["key_points"],
                        now,
                        edit.chapter_id,
                        row["project_id"],
                    ),
                )
            recovery_paths: list[tuple[Path, Path]] = []
            for chapter_id in deleted_ids:
                current = by_id[chapter_id]
                source = Path(str(current["content_path"])).parent
                recovery = (
                    self.novels_dir
                    / str(user_id)
                    / str(row["project_id"])
                    / ".chapter-recovery"
                    / f"{assistant_message_id}-{chapter_id}"
                )
                if recovery.exists():
                    connection.rollback()
                    raise ValueError("章节恢复副本已经存在，请先人工核对")
                recovery.mkdir(parents=True, exist_ok=False, mode=0o700)
                if source.exists():
                    shutil.copytree(source, recovery / "files")
                (recovery / "metadata.json").write_text(
                    json.dumps(
                        {
                            "project_id": str(row["project_id"]),
                            "chapter": current,
                            "deleted_at": now,
                        },
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    ),
                    encoding="utf-8",
                )
                recovery_paths.append((source, recovery))
                connection.execute(
                    """
                    UPDATE assistant_conversations
                    SET scope_type='project', novel_chapter_id=NULL,
                        updated_at=?
                    WHERE novel_chapter_id=? AND project_id=?
                    """,
                    (now, chapter_id, row["project_id"]),
                )
                connection.execute(
                    "DELETE FROM novel_chapters WHERE id=? AND project_id=?",
                    (chapter_id, row["project_id"]),
                )

            current_order = [
                str(chapter["id"])
                for chapter in chapters
                if str(chapter["id"]) not in deleted_ids
            ]
            if ordered_ids != current_order or deleted_ids:
                for temporary_position, chapter_id in enumerate(
                    ordered_ids, start=1
                ):
                    connection.execute(
                        """
                        UPDATE novel_chapters SET position=?
                        WHERE id=? AND project_id=?
                        """,
                        (
                            -temporary_position,
                            chapter_id,
                            row["project_id"],
                        ),
                    )
                for position, chapter_id in enumerate(ordered_ids, start=1):
                    connection.execute(
                        """
                        UPDATE novel_chapters
                        SET position=?, updated_at=?
                        WHERE id=? AND project_id=?
                        """,
                        (position, now, chapter_id, row["project_id"]),
                    )
            changed_ids = [edit.chapter_id for edit in patch.edits]
            response["chapter_patch_status"] = "applied"
            response["chapter_patch_applied_at"] = now
            response["chapter_patch_before"] = before
            response["chapter_patch_changed_ids"] = changed_ids
            boundary = dict(response.get("boundary") or {})
            boundary["chapter_metadata_unchanged"] = False
            response["boundary"] = boundary
            connection.execute(
                """
                UPDATE assistant_messages SET response_json=?
                WHERE id=?
                """,
                (_json(response), assistant_message_id),
            )
            connection.execute(
                "UPDATE novel_projects SET updated_at=? WHERE id=?",
                (now, row["project_id"]),
            )
            connection.commit()
        for source, _recovery in recovery_paths:
            if source.exists():
                shutil.rmtree(source, ignore_errors=True)
        return {
            "project_id": str(row["project_id"]),
            "conversation_id": str(row["conversation_id"]),
            "changed_chapter_ids": changed_ids,
            "already_applied": False,
            "deleted_chapter_ids": sorted(deleted_ids),
        }

    def apply_technique_patch_candidate(
        self,
        *,
        user_id: int,
        assistant_message_id: str,
    ) -> Dict[str, Any]:
        """Persist evidence-linked technique cards from a reference turn."""

        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT m.response_json, m.context_snapshot_json,
                       parent.content AS author_request,
                       conversation.id AS conversation_id,
                       conversation.document_id
                FROM assistant_messages m
                LEFT JOIN assistant_messages parent
                  ON parent.id=m.parent_user_message_id
                JOIN assistant_conversations conversation
                  ON conversation.id=m.conversation_id
                JOIN documents document
                  ON document.id=conversation.document_id
                 AND document.user_id=conversation.user_id
                WHERE m.id=? AND conversation.user_id=?
                  AND m.role='assistant' AND m.status='completed'
                  AND conversation.scope_type IN ('document', 'reference_chapter')
                """,
                (assistant_message_id, user_id),
            ).fetchone()
            if not row:
                connection.rollback()
                raise ValueError("这条回复没有可保存的参考技法")
            response = _load_json(row["response_json"], {})
            if response.get("technique_patch_status") == "applied":
                connection.commit()
                return {
                    "conversation_id": str(row["conversation_id"]),
                    "technique_ids": list(
                        response.get("technique_patch_applied_ids") or []
                    ),
                    "already_applied": True,
                }
            raw_patch = response.get("technique_patch")
            if not isinstance(raw_patch, dict):
                connection.rollback()
                raise ValueError("这条回复没有可保存的参考技法")
            try:
                patch = AssistantTechniquePatch.model_validate(raw_patch)
            except ValueError as exc:
                connection.rollback()
                raise ValueError("参考技法卡结构无效") from exc
            snapshot = _load_json(row["context_snapshot_json"], {})
            context = dict(snapshot.get("context") or {})
            capabilities = {
                str(value)
                for value in (
                    (context.get("agent") or {}).get("capabilities") or []
                )
            }
            if CREATE_TECHNIQUE_CARD not in capabilities:
                connection.rollback()
                raise ValueError("生成这条回复的内部任务没有技法卡权限")
            if not _author_explicitly_requested_technique_card(
                str(row["author_request"] or "")
            ):
                connection.rollback()
                raise ValueError("保存技法卡前必须由作者明确提出提炼或保存要求")

            now = utc_now()
            technique_ids: list[str] = []
            created_ids: list[str] = []
            for card in patch.cards:
                if card.source_document_id != str(row["document_id"]):
                    connection.rollback()
                    raise ValueError("技法卡来源不属于当前参考书")
                source = connection.execute(
                    """
                    SELECT chapter.id, chapter.content_path
                    FROM chapters chapter
                    JOIN documents document
                      ON document.id=chapter.document_id
                    WHERE chapter.id=? AND chapter.document_id=?
                      AND document.user_id=?
                    """,
                    (
                        card.source_chapter_id,
                        card.source_document_id,
                        user_id,
                    ),
                ).fetchone()
                if not source:
                    connection.rollback()
                    raise ValueError("技法卡的参考章节已经不存在")
                if _sha256(_read_text(source["content_path"])) != (
                    card.source_expected_revision
                ):
                    connection.rollback()
                    raise ValueError("参考正文在分析后已经变化，请重新读取")
                observation = card.observation
                existing = connection.execute(
                    """
                    SELECT id FROM reference_technique_cards
                    WHERE user_id=? AND source_chapter_id=? AND name=?
                      AND execution_rule=? AND status='active'
                    LIMIT 1
                    """,
                    (
                        user_id,
                        card.source_chapter_id,
                        observation.name,
                        observation.execution_rule,
                    ),
                ).fetchone()
                if existing:
                    technique_ids.append(str(existing["id"]))
                    continue
                technique_id = uuid.uuid4().hex
                connection.execute(
                    """
                    INSERT INTO reference_technique_cards(
                        id, user_id, source_document_id, source_chapter_id,
                        name, dimension, source_location, observation, effect,
                        suitable_for_json, unsuitable_for_json,
                        execution_rule, originality_boundary, author_note,
                        status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                              'active', ?, ?)
                    """,
                    (
                        technique_id,
                        user_id,
                        card.source_document_id,
                        card.source_chapter_id,
                        observation.name,
                        observation.dimension,
                        observation.source_location,
                        observation.observation,
                        observation.effect,
                        _json(observation.suitable_for),
                        _json(observation.unsuitable_for),
                        observation.execution_rule,
                        observation.originality_boundary,
                        card.author_note,
                        now,
                        now,
                    ),
                )
                technique_ids.append(technique_id)
                created_ids.append(technique_id)

            response["technique_patch_status"] = "applied"
            response["technique_patch_applied_at"] = now
            response["technique_patch_applied_ids"] = technique_ids
            response["technique_patch_created_ids"] = created_ids
            boundary = dict(response.get("boundary") or {})
            boundary["technique_library_unchanged"] = False
            response["boundary"] = boundary
            connection.execute(
                "UPDATE assistant_messages SET response_json=? WHERE id=?",
                (_json(response), assistant_message_id),
            )
            connection.commit()
        return {
            "conversation_id": str(row["conversation_id"]),
            "technique_ids": technique_ids,
            "created_ids": created_ids,
            "already_applied": False,
        }

    def apply_version_restore_candidate(
        self,
        *,
        user_id: int,
        assistant_message_id: str,
    ) -> Dict[str, Any]:
        """Copy an immutable historical version into a new working version."""

        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT m.response_json, m.context_snapshot_json,
                       parent.content AS author_request,
                       c.id AS conversation_id, c.project_id
                FROM assistant_messages m
                LEFT JOIN assistant_messages parent
                  ON parent.id=m.parent_user_message_id
                JOIN assistant_conversations c
                  ON c.id=m.conversation_id
                JOIN novel_projects project ON project.id=c.project_id
                JOIN work_versions work_version
                  ON work_version.project_id=c.project_id
                 AND work_version.ref_type='branch'
                 AND work_version.ref_name='main'
                 AND work_version.is_editable=1
                JOIN works work
                  ON work.id=work_version.work_id AND work.user_id=c.user_id
                WHERE m.id=? AND c.user_id=? AND project.user_id=?
                  AND m.role='assistant' AND m.status='completed'
                  AND c.scope_type IN ('project', 'chapter')
                """,
                (assistant_message_id, user_id, user_id),
            ).fetchone()
            if not row:
                connection.rollback()
                raise ValueError("这条回复没有可恢复的历史版本")
            response = _load_json(row["response_json"], {})
            if response.get("version_restore_status") == "applied":
                connection.commit()
                return {
                    "project_id": str(row["project_id"]),
                    "conversation_id": str(row["conversation_id"]),
                    "chapter_id": str(
                        response.get("version_restore_chapter_id") or ""
                    ),
                    "version_id": str(
                        response.get("version_restore_applied_version_id")
                        or ""
                    ),
                    "already_applied": True,
                }
            raw_restore = response.get("version_restore")
            if not isinstance(raw_restore, dict):
                connection.rollback()
                raise ValueError("这条回复没有可恢复的历史版本")
            try:
                proposal = AssistantVersionRestoreProposal.model_validate(
                    raw_restore
                )
            except ValueError as exc:
                connection.rollback()
                raise ValueError("历史版本恢复结构无效") from exc
            snapshot = _load_json(row["context_snapshot_json"], {})
            context = dict(snapshot.get("context") or {})
            capabilities = {
                str(value)
                for value in (
                    (context.get("agent") or {}).get("capabilities") or []
                )
            }
            if WRITE_CHAPTER not in capabilities:
                connection.rollback()
                raise ValueError("生成这条回复的内部任务没有正文恢复权限")
            if not _author_explicitly_requested_restore(
                str(row["author_request"] or "")
            ):
                connection.rollback()
                raise ValueError("恢复历史版本前必须由作者明确提出恢复要求")
            chapter = connection.execute(
                """
                SELECT chapter.id, chapter.content_path,
                       chapter.head_version_id,
                       head.content_path AS active_content_path
                FROM novel_chapters chapter
                JOIN novel_projects project
                  ON project.id=chapter.project_id
                LEFT JOIN novel_chapter_versions head
                  ON head.id=chapter.head_version_id
                WHERE chapter.id=? AND chapter.project_id=?
                  AND project.user_id=?
                """,
                (proposal.chapter_id, row["project_id"], user_id),
            ).fetchone()
            source = connection.execute(
                """
                SELECT version.id, version.content_path
                FROM novel_chapter_versions version
                JOIN novel_chapters chapter
                  ON chapter.id=version.chapter_id
                WHERE version.id=? AND version.chapter_id=?
                  AND chapter.project_id=?
                """,
                (
                    proposal.version_id,
                    proposal.chapter_id,
                    row["project_id"],
                ),
            ).fetchone()
            if not chapter or not source:
                connection.rollback()
                raise ValueError("章节或历史版本已经不存在")
            current_head_id = str(chapter["head_version_id"] or "")
            if current_head_id != str(proposal.head_version_id or ""):
                connection.rollback()
                raise ValueError("main HEAD 在讨论后已经变化，请重新比较")
            source_content = _read_text(source["content_path"])
            current_content = _read_text(chapter["active_content_path"])
            if _sha256(source_content) != proposal.source_expected_revision:
                connection.rollback()
                raise ValueError("历史版本文件校验失败，不能恢复")
            if _sha256(current_content) != proposal.head_expected_revision:
                connection.rollback()
                raise ValueError("当前正文在讨论后已经变化，请重新读取")
            if source_content == current_content:
                connection.rollback()
                raise ValueError("所选历史版本与当前正文完全相同")

            token = secrets.token_hex(16)
            version_id = uuid.uuid4().hex
            chapter_content_path = Path(str(chapter["content_path"]))
            version_path = (
                chapter_content_path.parent
                / "versions"
                / f"assistant-restore-{token}.txt"
            )
            now = utc_now()
            count = len(source_content)
            effective_count = effective_char_count(source_content)
            version_written = False
            try:
                _atomic_write(version_path, source_content, token)
                version_written = True
                connection.execute(
                    """
                    INSERT INTO novel_chapter_versions(
                        id, chapter_id, kind, content_path, char_count,
                        created_at, parent_version_id, source,
                        content_hash, change_summary, created_by,
                        quality_status, effective_char_count, hard_issue_count
                    ) VALUES (
                        ?, ?, 'assistant_restore', ?, ?, ?, ?,
                        'assistant_chat', ?, ?, 'author', 'pass', ?, 0
                    )
                    """,
                    (
                        version_id,
                        proposal.chapter_id,
                        str(version_path),
                        count,
                        now,
                        current_head_id or None,
                        _sha256(source_content),
                        proposal.rationale[:1000],
                        effective_count,
                    ),
                )
                advanced = self.database.set_chapter_head_in_transaction(
                    connection,
                    user_id=user_id,
                    project_id=str(row["project_id"]),
                    chapter_id=proposal.chapter_id,
                    version_id=version_id,
                    expected_old_head_version_id=current_head_id,
                    now=now,
                )
                if not advanced:
                    raise ValueError("恢复版本没有成为 main HEAD")
                response["version_restore_status"] = "applied"
                response["version_restore_applied_at"] = now
                response["version_restore_chapter_id"] = proposal.chapter_id
                response["version_restore_source_version_id"] = proposal.version_id
                response["version_restore_applied_version_id"] = version_id
                boundary = dict(response.get("boundary") or {})
                boundary["main_head_unchanged"] = False
                response["boundary"] = boundary
                connection.execute(
                    """
                    UPDATE assistant_messages
                    SET response_json=?, applied_version_id=?
                    WHERE id=?
                    """,
                    (_json(response), version_id, assistant_message_id),
                )
                connection.execute(
                    "UPDATE novel_projects SET updated_at=? WHERE id=?",
                    (now, row["project_id"]),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                if version_written:
                    version_path.unlink(missing_ok=True)
                raise
        try:
            _atomic_write(
                chapter_content_path,
                source_content,
                secrets.token_hex(16),
            )
        except Exception:
            logger.warning(
                "failed to refresh non-authoritative chapter cache",
                exc_info=True,
            )
        return {
            "project_id": str(row["project_id"]),
            "conversation_id": str(row["conversation_id"]),
            "chapter_id": proposal.chapter_id,
            "version_id": version_id,
            "source_version_id": proposal.version_id,
            "already_applied": False,
        }

    def apply_note_patch_candidate(
        self,
        *,
        user_id: int,
        assistant_message_id: str,
    ) -> Dict[str, Any]:
        """Atomically persist author-requested project notes on main."""

        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT m.response_json, m.context_snapshot_json,
                       parent.content AS author_request,
                       c.id AS conversation_id, c.project_id
                FROM assistant_messages m
                LEFT JOIN assistant_messages parent
                  ON parent.id=m.parent_user_message_id
                JOIN assistant_conversations c
                  ON c.id=m.conversation_id
                JOIN novel_projects project ON project.id=c.project_id
                JOIN work_versions version
                  ON version.project_id=c.project_id
                 AND version.ref_type='branch'
                 AND version.ref_name='main'
                 AND version.is_editable=1
                JOIN works work
                  ON work.id=version.work_id AND work.user_id=c.user_id
                WHERE m.id=? AND c.user_id=? AND project.user_id=?
                  AND m.role='assistant' AND m.status='completed'
                  AND c.scope_type IN ('project', 'chapter')
                """,
                (assistant_message_id, user_id, user_id),
            ).fetchone()
            if not row:
                connection.rollback()
                raise ValueError("这条回复没有可应用的作者笔记")
            response = _load_json(row["response_json"], {})
            if response.get("note_patch_status") == "applied":
                connection.commit()
                return {
                    "project_id": str(row["project_id"]),
                    "conversation_id": str(row["conversation_id"]),
                    "changed_note_keys": list(
                        response.get("note_patch_changed_keys") or []
                    ),
                    "already_applied": True,
                }
            raw_patch = response.get("note_patch")
            if not isinstance(raw_patch, dict):
                connection.rollback()
                raise ValueError("这条回复没有可应用的作者笔记")
            try:
                patch = AssistantNotePatch.model_validate(raw_patch)
            except ValueError as exc:
                connection.rollback()
                raise ValueError("作者笔记修改结构无效") from exc
            snapshot = _load_json(row["context_snapshot_json"], {})
            context = dict(snapshot.get("context") or {})
            capabilities = {
                str(value)
                for value in (
                    (context.get("agent") or {}).get("capabilities") or []
                )
            }
            if MANAGE_NOTES not in capabilities:
                connection.rollback()
                raise ValueError("生成这条回复的内部任务没有笔记管理权限")
            if any(edit.action == "delete" for edit in patch.edits) and not (
                _author_explicitly_requested_deletion(
                    str(row["author_request"] or "")
                )
            ):
                connection.rollback()
                raise ValueError("删除作者笔记前必须由作者明确提出删除目标")

            existing_rows = connection.execute(
                """
                SELECT id, note_key, title, content, source_message_id,
                       created_at, updated_at
                FROM novel_author_notes
                WHERE project_id=?
                """,
                (row["project_id"],),
            ).fetchall()
            by_key = {str(item["note_key"]): dict(item) for item in existing_rows}
            by_id = {str(item["id"]): dict(item) for item in existing_rows}
            before: dict[str, Any] = {}
            now = utc_now()
            changed_keys: list[str] = []
            for edit in patch.edits:
                current = by_key.get(edit.note_key)
                if edit.action == "create":
                    if current is not None:
                        connection.rollback()
                        raise ValueError(
                            f"作者笔记 {edit.note_key} 已存在，请重新读取后修改"
                        )
                    note_id = uuid.uuid4().hex
                    connection.execute(
                        """
                        INSERT INTO novel_author_notes(
                            id, project_id, note_key, title, content,
                            source_message_id, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            note_id,
                            row["project_id"],
                            edit.note_key,
                            str(edit.title or edit.note_key),
                            str(edit.content or ""),
                            assistant_message_id,
                            now,
                            now,
                        ),
                    )
                    before[edit.note_key] = None
                else:
                    current_by_id = by_id.get(str(edit.note_id or ""))
                    if current is None or current_by_id is None or (
                        str(current["id"]) != str(current_by_id["id"])
                    ):
                        connection.rollback()
                        raise ValueError("要修改的作者笔记已经不存在或路径已变化")
                    if _sha256(str(current["content"] or "")) != (
                        edit.expected_revision
                    ):
                        connection.rollback()
                        raise ValueError(
                            "作者笔记在讨论后已经变化，请重新读取后再修改"
                        )
                    before[edit.note_key] = current
                    if edit.action == "delete":
                        connection.execute(
                            """
                            DELETE FROM novel_author_notes
                            WHERE id=? AND project_id=?
                            """,
                            (edit.note_id, row["project_id"]),
                        )
                    else:
                        connection.execute(
                            """
                            UPDATE novel_author_notes
                            SET title=?, content=?, source_message_id=?,
                                updated_at=?
                            WHERE id=? AND project_id=?
                            """,
                            (
                                str(edit.title or current["title"] or edit.note_key),
                                str(edit.content or ""),
                                assistant_message_id,
                                now,
                                edit.note_id,
                                row["project_id"],
                            ),
                        )
                changed_keys.append(edit.note_key)

            response["note_patch_status"] = "applied"
            response["note_patch_applied_at"] = now
            response["note_patch_before"] = before
            response["note_patch_changed_keys"] = changed_keys
            boundary = dict(response.get("boundary") or {})
            boundary["author_notes_unchanged"] = False
            response["boundary"] = boundary
            connection.execute(
                "UPDATE assistant_messages SET response_json=? WHERE id=?",
                (_json(response), assistant_message_id),
            )
            connection.execute(
                "UPDATE novel_projects SET updated_at=? WHERE id=?",
                (now, row["project_id"]),
            )
            connection.commit()
        return {
            "project_id": str(row["project_id"]),
            "conversation_id": str(row["conversation_id"]),
            "changed_note_keys": changed_keys,
            "already_applied": False,
        }

    def apply_settings_candidate(
        self,
        *,
        user_id: int,
        assistant_message_id: str,
        selected_paths: Optional[set[str]] = None,
    ) -> Dict[str, Any]:
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT m.response_json, m.context_snapshot_json,
                       parent.content AS author_request,
                       c.id AS conversation_id, c.project_id, c.scope_type
                FROM assistant_messages m
                LEFT JOIN assistant_messages parent
                  ON parent.id=m.parent_user_message_id
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
                validated_patch = AssistantSettingsPatch.model_validate(
                    raw_patch
                )
                patch = validated_patch.model_dump(exclude_none=True)
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
            archive_rules = list(patch.pop("archive_rules", []))
            patch.pop("structured_edits", None)
            structured_edits = list(
                validated_patch.structured_edits or []
            )
            if (
                any(edit.action == "delete" for edit in structured_edits)
                and not _author_explicitly_requested_deletion(
                    str(row["author_request"] or "")
                )
            ):
                connection.rollback()
                raise ValueError(
                    "删除作品资料前必须由作者明确提出删除目标"
                )
            if selected_paths is not None:
                patch = {
                    key: value
                    for key, value in patch.items()
                    if f"project:{key}" in selected_paths
                }
                archive_rules = [
                    rule
                    for index, rule in enumerate(archive_rules)
                    if f"archive-rule:{index}" in selected_paths
                ]
                structured_edits = filter_structured_edits(
                    structured_edits, selected_paths
                )
                if not patch and not archive_rules and not structured_edits:
                    connection.rollback()
                    raise ValueError("请至少选择一项要应用的修改")
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
            now = utc_now()
            if patch:
                assignments = ", ".join(f"{key}=?" for key in patch)
                values = [patch[key] for key in patch]
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
            if archive_rules:
                main = connection.execute(
                    """
                    SELECT version.id AS content_version_id,
                           version.work_id
                    FROM work_versions version
                    JOIN works work ON work.id=version.work_id
                    WHERE version.project_id=? AND work.user_id=?
                      AND version.ref_type='branch'
                      AND version.ref_name='main'
                      AND version.is_editable=1
                    """,
                    (row["project_id"], user_id),
                ).fetchone()
                if not main:
                    connection.rollback()
                    raise ValueError("作品缺少可写的 main 分支")
                for rule in archive_rules:
                    category = str(rule["category"])
                    title = str(rule.get("title") or "")
                    content = str(rule["content"])
                    duplicate = connection.execute(
                        """
                        SELECT id FROM work_archive_entries
                        WHERE work_id=? AND entry_type='creative_rule'
                          AND status='confirmed' AND category=?
                          AND title=? AND content=?
                        LIMIT 1
                        """,
                        (
                            main["work_id"],
                            category,
                            title,
                            content,
                        ),
                    ).fetchone()
                    if duplicate:
                        continue
                    connection.execute(
                        """
                        INSERT INTO work_archive_entries(
                            id, work_id, content_version_id,
                            entry_type, title, content,
                            provenance, status, evidence, source_ref,
                            category, adopted_at, created_at, updated_at
                        ) VALUES (
                            ?, ?, ?, 'creative_rule', ?, ?,
                            'assistant', 'confirmed', '', ?, ?, ?, ?, ?
                        )
                        """,
                        (
                            uuid.uuid4().hex,
                            main["work_id"],
                            main["content_version_id"],
                            title,
                            content,
                            f"assistant:{assistant_message_id}",
                            category,
                            now,
                            now,
                            now,
                        ),
                    )
                connection.execute(
                    "UPDATE works SET updated_at=? WHERE id=?",
                    (now, main["work_id"]),
                )
            applied_structured_edits: list[dict[str, Any]] = []
            if structured_edits:
                baseline_structured = context.get("structured_settings")
                if not isinstance(baseline_structured, Mapping):
                    baseline_structured = (
                        self.structured_settings_editor.snapshot_in_connection(
                            connection,
                            user_id=user_id,
                            project_id=str(row["project_id"]),
                        )
                    )
                applied_structured_edits = (
                    self.structured_settings_editor.apply_in_connection(
                        connection,
                        user_id=user_id,
                        project_id=str(row["project_id"]),
                        edits=structured_edits,
                        baseline_snapshot=baseline_structured,
                    )
                )
            applied_values = dict(patch)
            if archive_rules:
                applied_values["archive_rules"] = archive_rules
                before["archive_rules"] = []
            if applied_structured_edits:
                applied_values["structured_edits"] = (
                    applied_structured_edits
                )
                before["structured_edits"] = [
                    item.get("before") for item in applied_structured_edits
                ]
            response["settings_patch_status"] = "applied"
            response["settings_patch_applied_at"] = now
            response["settings_patch_before"] = before
            response["settings_patch_applied_values"] = applied_values
            boundary = dict(response.get("boundary") or {})
            boundary["project_settings_unchanged"] = False
            response["boundary"] = boundary
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
            "changed_fields": list(applied_values),
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

    def commit_draft_to_head(
        self, *, user_id: int, assistant_message_id: str
    ) -> Dict[str, Any]:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT m.response_json, m.context_snapshot_json,
                       m.applied_version_id, c.id AS conversation_id,
                       c.project_id, c.novel_chapter_id,
                       ch.content_path AS cache_content_path,
                       ch.head_version_id,
                       head.content_path AS head_content_path
                FROM assistant_messages m
                JOIN assistant_conversations c
                  ON c.id=m.conversation_id
                JOIN novel_chapters ch ON ch.id=c.novel_chapter_id
                LEFT JOIN novel_chapter_versions head
                  ON head.id=ch.head_version_id
                JOIN novel_projects p ON p.id=ch.project_id
                WHERE m.id=? AND c.user_id=? AND p.user_id=?
                  AND m.role='assistant' AND m.status='completed'
                  AND c.scope_type='chapter'
                """,
                (assistant_message_id, user_id, user_id),
            ).fetchone()
            if not row:
                raise ValueError("这条回复没有可提交的章节正文")
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
                raise ValueError("这条回复没有可提交的章节正文")
            try:
                draft = AssistantDraftProposal.model_validate(raw_draft)
            except ValueError as exc:
                raise ValueError("章节正文结构无效") from exc
            snapshot = _load_json(row["context_snapshot_json"], {})
            context = dict(snapshot.get("context") or {})
            capabilities = {
                str(item)
                for item in (
                    (context.get("agent") or {}).get("capabilities") or []
                )
            }
            if WRITE_CHAPTER not in capabilities:
                raise ValueError("生成这条回复的角色没有正文写入权限")

        chapter_content_path = Path(str(row["cache_content_path"]))
        current_content = _read_text(row["head_content_path"])
        expected_hash = str(
            context.get("current_chapter_hash") or _sha256("")
        )
        if _sha256(current_content) != expected_hash:
            raise ValueError("正文在生成结果后已经变化，请重新生成")
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
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                """
                SELECT 1 FROM generation_jobs
                WHERE chapter_id=? AND status IN ('queued', 'running')
                    AND operation<>'extract_story_delta'
                LIMIT 1
                """,
                (row["novel_chapter_id"],),
            ).fetchone()
            if active:
                connection.rollback()
                raise ValueError("AI 正在生成本章，请完成后再提交正文")
            latest = connection.execute(
                """
                SELECT m.applied_version_id, ch.head_version_id
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
            if str(latest["head_version_id"] or "") != str(
                row["head_version_id"] or ""
            ):
                connection.rollback()
                raise ValueError("main HEAD 已经变化，请重新生成正文")
            connection.execute(
                """
                INSERT INTO novel_chapter_versions(
                    id, chapter_id, kind, content_path, char_count,
                    created_at, parent_version_id, source,
                    content_hash, change_summary, created_by,
                    quality_status, effective_char_count, hard_issue_count
                ) VALUES (
                    ?, ?, 'assistant_draft', ?, ?, ?, ?,
                    'assistant_chat', ?, ?, 'assistant', 'pass', ?, 0
                )
                """,
                (
                    version_id,
                    row["novel_chapter_id"],
                    str(version_path),
                    count,
                    now,
                    row["head_version_id"],
                    _sha256(new_content),
                    draft.rationale[:1000],
                    effective_count,
                ),
            )
            advanced = self.database.set_chapter_head_in_transaction(
                connection,
                user_id=user_id,
                project_id=str(row["project_id"]),
                chapter_id=str(row["novel_chapter_id"]),
                version_id=version_id,
                expected_old_head_version_id=str(
                    row["head_version_id"] or ""
                ),
                now=now,
            )
            if not advanced:
                connection.rollback()
                raise ValueError("正文没有成为 main HEAD")
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
            connection.commit()
        try:
            _atomic_write(
                chapter_content_path,
                new_content,
                secrets.token_hex(16),
            )
        except Exception:
            logger.warning(
                "failed to refresh non-authoritative chapter cache",
                exc_info=True,
            )
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
                       ch.head_version_id,
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
            if str(row["head_version_id"] or "") != applied_version_id:
                raise ValueError("后面已有新的正文版本，请从版本历史恢复")
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
        quality_status = "pass"
        hard_issue_count = 0
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            latest = connection.execute(
                """
                SELECT ch.head_version_id,
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
            if str(latest["head_version_id"] or "") != applied_version_id:
                connection.rollback()
                raise ValueError("后面已有新的正文版本，请从版本历史恢复")
            connection.execute(
                """
                INSERT INTO novel_chapter_versions(
                    id, chapter_id, kind, content_path, char_count,
                    created_at, parent_version_id, source,
                    content_hash, change_summary, created_by,
                    quality_status, effective_char_count, hard_issue_count
                ) VALUES (
                    ?, ?, 'assistant_revert', ?, ?, ?, ?,
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
            advanced = self.database.set_chapter_head_in_transaction(
                connection,
                user_id=user_id,
                project_id=str(row["project_id"]),
                chapter_id=str(row["novel_chapter_id"]),
                version_id=version_id,
                expected_old_head_version_id=applied_version_id,
                now=now,
            )
            if not advanced:
                connection.rollback()
                raise ValueError("撤回版本没有成为 main HEAD")
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
        try:
            _atomic_write(
                chapter_content_path,
                restored_content,
                secrets.token_hex(16),
            )
        except Exception:
            logger.warning(
                "failed to refresh non-authoritative chapter cache",
                exc_info=True,
            )
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
        search_settings = self.database.get_web_search_summary(user_id)
        context["web_search_available"] = bool(
            search_settings is None or search_settings.get("enabled")
        )
        context["agent"] = manifest
        auto_apply_settings = bool(
            auto_commit and PROPOSE_SETTINGS_PATCH in capabilities
        )
        auto_apply_story_plan = bool(
            auto_commit and PROPOSE_STORY_PLAN in capabilities
        )
        auto_apply_chapter_metadata = bool(
            auto_commit and MANAGE_CHAPTERS in capabilities
        )
        auto_apply_notes = bool(
            auto_commit and MANAGE_NOTES in capabilities
        )
        auto_apply_techniques = bool(
            auto_commit and CREATE_TECHNIQUE_CARD in capabilities
        )
        context["assistant_boundaries"] = {
            "may_modify_canon": False,
            "may_modify_story_memory": False,
            "may_modify_task_cards": False,
            "may_apply_settings": auto_apply_settings,
            "may_propose_settings_patch": (
                PROPOSE_SETTINGS_PATCH in capabilities
            ),
            "may_propose_story_plan": (
                PROPOSE_STORY_PLAN in capabilities
            ),
            "may_write_chapter": (
                WRITE_CHAPTER in capabilities
            ),
            "auto_advance_main_head": bool(auto_commit),
            "auto_apply_settings": auto_apply_settings,
            "auto_apply_story_plan": auto_apply_story_plan,
            "auto_apply_chapter_metadata": auto_apply_chapter_metadata,
            "auto_apply_notes": auto_apply_notes,
            "auto_apply_techniques": auto_apply_techniques,
            "settings_patch_requires_author_action": (
                not auto_apply_settings
            ),
            "story_plan_requires_author_action": (
                not auto_apply_story_plan
            ),
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
                        "analysis": compact_analysis(
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
                chapter.get("head_version_id")
                or chapter.get("head_version_id")
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
            snapshot["structured_settings"] = (
                self.structured_settings_editor.snapshot(
                    user_id=user_id,
                    project_id=project_id,
                )
            )
            snapshot.update(
                self._build_linked_work_context(
                    user_id=user_id, project_id=project_id
                )
            )
            return snapshot, sources

        with self.database.connection() as connection:
            project = connection.execute(
                """
                SELECT id, title, genre, premise, theme, world_setting,
                       style_guide, point_of_view,
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
                       head_version_id
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
                       ch.head_version_id
                FROM novel_chapters ch
                JOIN novel_projects p ON p.id=ch.project_id
                JOIN novel_chapter_versions v ON v.id=ch.head_version_id
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
        structured_settings = self.structured_settings_editor.snapshot(
            user_id=user_id,
            project_id=project_id,
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
                "structured_settings": structured_settings,
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
            compact = compact_analysis(result)
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
            "main HEAD"
            if str(version.get("head_version_id") or "")
            == str(version["id"])
            else "历史"
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
                       ch.head_version_id,
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
