from __future__ import annotations

import json
import logging
import re
import secrets
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .agent_capabilities import (
    CREATE_TECHNIQUE_CARD,
    MANAGE_CHAPTERS,
    MANAGE_NOTES,
    PROPOSE_SETTINGS_PATCH,
    PROPOSE_STORY_PLAN,
    WRITE_CHAPTER,
)
from .assistant_chat_schema import (
    AssistantChapterPatch,
    AssistantDraftProposal,
    AssistantNotePatch,
    AssistantSettingsPatch,
    AssistantStoryPlanProposal,
    AssistantTechniquePatch,
    AssistantVersionRestoreProposal,
)
from .assistant_file_ops import (
    atomic_write as _atomic_write,
    read_text as _read_text,
    sha256_text as _sha256,
)
from .db import utc_now
from .json_support import (
    dump_canonical_json as _json,
    load_json as _load_json,
)
from .story_planning_service import StoryPlanningService
from .structured_settings import filter_structured_edits
from .text_metrics import effective_char_count


logger = logging.getLogger(__name__)


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


class AssistantApplicationMixin:
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
                        effective_char_count
                    ) VALUES (
                        ?, ?, 'assistant_restore', ?, ?, ?, ?,
                        'assistant_chat', ?, ?, 'author', ?
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
                    effective_char_count
                ) VALUES (
                    ?, ?, 'assistant_draft', ?, ?, ?, ?,
                    'assistant_chat', ?, ?, 'assistant', ?
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
                    effective_char_count
                ) VALUES (
                    ?, ?, 'assistant_revert', ?, ?, ?, ?,
                    'assistant_chat', ?, ?, 'author', ?
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
                    effective_count,
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
