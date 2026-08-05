from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .agent_capabilities import (
    CREATE_TECHNIQUE_CARD,
    MANAGE_CHAPTERS,
    MANAGE_NOTES,
    PROPOSE_SETTINGS_PATCH,
    PROPOSE_STORY_PLAN,
    WRITE_CHAPTER,
    agent_capabilities,
    agent_manifest,
)
from .assistant_result import compact_analysis
from .context_compiler import build_writing_context_snapshot
from .json_support import load_json as _load_json


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_text(path: Any) -> str:
    try:
        return Path(str(path)).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ""


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


class AssistantContextMixin:
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
        if not quote_text:
            raise ValueError("引用文字不能为空")
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
        return {"context": context, "sources": sources}

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
                """,
                (project_id,),
            ).fetchall()
            aggregate_row = None
            if linked_rows:
                aggregate_row = connection.execute(
                    """
                    SELECT aggregate_json
                    FROM analysis_jobs
                    WHERE document_id=? AND user_id=?
                      AND status IN ('completed', 'partial')
                      AND aggregate_json NOT IN ('', '{}')
                    ORDER BY finished_at DESC, created_at DESC
                    LIMIT 1
                    """,
                    (str(linked_rows[0]["document_id"]), user_id),
                ).fetchone()
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
                "style_profile": (
                    _load_json(aggregate_row["aggregate_json"], {}).get(
                        "style_profile"
                    )
                    if aggregate_row
                    else None
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
                WHERE project_id=? ORDER BY position
                """,
                (project_id,),
            ).fetchall()
            chapters = connection.execute(
                """
                SELECT id, position, title, outline, key_points, status,
                       head_version_id
                FROM novel_chapters
                WHERE project_id=? ORDER BY position
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
                        "text": analysis_text,
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
                        "text": text,
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
