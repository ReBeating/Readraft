from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional

from .db import Database
from .web_paths import (
    document_workbench_path,
    workbench_path,
)


POV_OPTIONS = (
    "第三人称限知",
    "第一人称",
    "第三人称全知",
    "多视角",
)
WORKBENCH_SETTING_TABS = (
    ("core", "作品概览"),
    ("world", "世界"),
    ("characters", "人物"),
    ("structure", "剧情与结构"),
    ("style", "叙事与文风"),
)
WORKBENCH_SETTING_TAB_KEYS = frozenset(key for key, _label in WORKBENCH_SETTING_TABS)
WORK_ARCHIVE_TAB_KEYS = frozenset({"creative", "analysis", "versions"})
WORK_ARCHIVE_CATEGORIES = (
    ("core", "作品概览"),
    ("world", "世界"),
    ("character", "人物"),
    ("structure", "剧情与结构"),
    ("style", "叙事与文风"),
)
WORK_ANALYSIS_CATEGORIES = (
    ("uncategorized", "未分类"),
    *WORK_ARCHIVE_CATEGORIES,
)
WORLD_ENTRY_TYPE_OPTIONS = (
    ("background", "背景"),
    ("rule", "规则与边界"),
    ("faction", "组织与势力"),
    ("location", "地点"),
    ("element", "物品、能力或术语"),
)
STORY_ARC_TYPE_OPTIONS = (
    "main",
    "subplot",
    "character",
    "relationship",
    "mystery",
    "world",
)


class WorkbenchNotFound(LookupError):
    pass


class WorkbenchUnavailable(RuntimeError):
    pass


def _read_optional_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


class WorkbenchViewBuilder:
    """Builds the data model for editable and immutable workbench pages.

    Route handlers remain responsible for authentication and HTTP responses. Keeping
    the database orchestration here makes the two workbench routes testable without
    growing the application factory further.
    """

    def __init__(
        self,
        *,
        database: Database,
        memory_service: Any,
        assistant_chat_service: Any,
        planning_service: Any,
        story_planning_service: Any,
        style_service: Any,
        continuity_service: Any,
    ) -> None:
        self.database = database
        self.memory_service = memory_service
        self.assistant_chat_service = assistant_chat_service
        self.planning_service = planning_service
        self.story_planning_service = story_planning_service
        self.style_service = style_service
        self.continuity_service = continuity_service

    def build_novel(
        self,
        *,
        user_id: int,
        project_id: str,
        chapter_id: Optional[str],
        conversation_id: Optional[str],
        view: str,
        archive_tab: str,
        settings_tab: str,
    ) -> dict[str, Any]:
        database = self.database
        project = database.get_novel_project(user_id, project_id)
        if not project:
            raise WorkbenchNotFound
        work = database.get_work_for_project(user_id, project_id)
        if not work:
            database.ensure_project_work(user_id=user_id, project_id=project_id)
            work = database.get_work_for_project(user_id, project_id)
        current_version = database.get_work_version_for_project(user_id, project_id)
        if (
            not work
            or not current_version
            or str(current_version.get("ref_name") or "") != "main"
            or not bool(current_version.get("is_editable"))
        ):
            raise WorkbenchUnavailable("只有 main 分支可以进入创作工作台")

        chapters = database.list_novel_chapters(user_id, project_id)
        effective_view = view if view in {"body", "archive"} else "body"
        if not chapters and effective_view == "body":
            effective_view = "archive"
        active_archive_tab = (
            archive_tab if archive_tab in WORK_ARCHIVE_TAB_KEYS else "creative"
        )
        active_settings_tab = (
            settings_tab if settings_tab in WORKBENCH_SETTING_TAB_KEYS else "core"
        )
        selected_chapter = self._select_chapter(
            chapters=chapters,
            current_version=current_version,
            chapter_id=chapter_id,
            effective_view=effective_view,
        )
        database.set_work_version(
            user_id=user_id,
            work_id=str(work["id"]),
            version_id=str(current_version["id"]),
            chapter_id=(str(selected_chapter["id"]) if selected_chapter else None),
        )

        chapter_view = self._editable_chapter_view(
            user_id=user_id,
            project_id=project_id,
            chapters=chapters,
            selected_chapter=selected_chapter,
        )
        project_memory_records: list[dict[str, Any]] = []
        if selected_chapter or (
            effective_view == "archive" and active_archive_tab == "analysis"
        ):
            project_memory_records = (
                self.memory_service.list_project_chapter_memory_records(
                    user_id=user_id,
                    project_id=project_id,
                )
            )
        chapter_story_memory = None
        if selected_chapter:
            chapter_story_memory = next(
                (
                    item
                    for item in project_memory_records
                    if str(item["chapter_id"]) == str(selected_chapter["id"])
                ),
                None,
            )

        conversations = self.assistant_chat_service.conversations.list_project(
            user_id=user_id,
            project_id=project_id,
        )
        active_conversation = self._active_novel_conversation(
            user_id=user_id,
            project_id=project_id,
            conversation_id=conversation_id,
            effective_view=effective_view,
            selected_chapter=selected_chapter,
            conversations=conversations,
        )

        archive_view = self._editable_archive_view(
            user_id=user_id,
            project_id=project_id,
            work=work,
            current_version=current_version,
            chapters=chapters,
            effective_view=effective_view,
            active_archive_tab=active_archive_tab,
            project_memory_records=project_memory_records,
        )
        return {
            "work": work,
            "current_version": current_version,
            "project": project,
            "display_title": str(project.get("title") or "").strip()
            or "未命名作品",
            "chapters": chapters,
            "chapter": selected_chapter,
            "chapter_story_memory": chapter_story_memory,
            "conversations": conversations,
            "active_conversation": active_conversation,
            "view": effective_view,
            "active_archive_tab": active_archive_tab,
            "active_settings_tab": active_settings_tab,
            "archive_base_url": f"/novels/{project_id}/workbench",
            "archive_return_to": workbench_path(
                project_id,
                archive_tab=active_archive_tab,
            ),
            "archive_readonly": False,
            "archive_story_memory_enabled": True,
            "creative_snapshot": {},
            **chapter_view,
            **archive_view,
        }

    def build_document(
        self,
        *,
        user_id: int,
        document_id: str,
        chapter_id: Optional[str],
        conversation_id: Optional[str],
        view: str,
        archive_tab: str,
        settings_tab: str,
    ) -> dict[str, Any]:
        database = self.database
        document = database.get_document(user_id, document_id)
        if not document:
            raise WorkbenchNotFound
        work = database.get_work_for_document(user_id, document_id)
        current_version = database.get_work_version_for_document(user_id, document_id)
        if not work or not current_version:
            raise WorkbenchUnavailable("固定版本不属于作品版本库")

        chapters = database.list_chapters(
            user_id,
            document_id,
            document.get("latest_job_id"),
        )
        version_memory_records = database.list_work_version_story_memory_records(
            user_id,
            str(current_version["id"]),
        )
        memory_by_chapter = {
            str(item["chapter_id"]): item for item in version_memory_records
        }
        has_memory_snapshots = any(
            item["memory_status"] == "ready" for item in version_memory_records
        )
        effective_view = "archive" if view == "archive" else "body"
        active_archive_tab = (
            archive_tab if archive_tab in WORK_ARCHIVE_TAB_KEYS else "analysis"
        )
        active_settings_tab = (
            settings_tab if settings_tab in WORKBENCH_SETTING_TAB_KEYS else "core"
        )
        selected_chapter = self._select_chapter(
            chapters=chapters,
            current_version=current_version,
            chapter_id=chapter_id,
            effective_view=effective_view,
        )
        database.set_work_version(
            user_id=user_id,
            work_id=str(work["id"]),
            version_id=str(current_version["id"]),
            chapter_id=(str(selected_chapter["id"]) if selected_chapter else None),
        )

        chapter_view = self._readonly_chapter_view(
            chapters=chapters,
            selected_chapter=selected_chapter,
        )
        conversations = self.assistant_chat_service.conversations.list_document(
            user_id=user_id,
            document_id=document_id,
        )
        active_conversation = self._active_document_conversation(
            user_id=user_id,
            document_id=document_id,
            conversation_id=conversation_id,
            effective_view=effective_view,
            selected_chapter=selected_chapter,
            conversations=conversations,
        )
        archive_entries: list[dict[str, Any]] = []
        archive_analyses: list[dict[str, Any]] = []
        if effective_view == "archive":
            archive_entries = database.list_work_archive_entries(
                user_id,
                str(work["id"]),
                str(current_version["id"]),
            )
            archive_analyses = database.list_work_analyses(
                user_id,
                str(work["id"]),
                str(current_version["id"]),
            )

        return {
            "work": work,
            "current_version": current_version,
            "document": document,
            "chapters": chapters,
            "chapter": selected_chapter,
            "chapter_story_memory": (
                memory_by_chapter.get(str(selected_chapter["id"]))
                if selected_chapter
                else None
            ),
            "conversations": conversations,
            "active_conversation": active_conversation,
            "view": effective_view,
            "active_archive_tab": active_archive_tab,
            "active_settings_tab": active_settings_tab,
            "archive_project": None,
            "archive_base_url": f"/documents/{document_id}",
            "archive_characters": [],
            "setting_characters": [],
            "setting_story_blueprint": None,
            "setting_story_arcs": [],
            "setting_voice_profile": None,
            "archive_entries": archive_entries,
            "archive_analyses": archive_analyses,
            "archive_story_memory_records": (
                version_memory_records if active_archive_tab == "analysis" else []
            ),
            "archive_story_memory_enabled": (
                has_memory_snapshots
                or str(current_version.get("intent") or "") == "snapshot"
            ),
            "has_story_memory_snapshots": has_memory_snapshots,
            "archive_return_to": document_workbench_path(
                document_id,
                view="archive",
                archive_tab=active_archive_tab,
            ),
            "archive_readonly": True,
            "creative_snapshot": current_version.get("creative_snapshot", {}),
            **chapter_view,
        }

    @staticmethod
    def _select_chapter(
        *,
        chapters: list[dict[str, Any]],
        current_version: dict[str, Any],
        chapter_id: Optional[str],
        effective_view: str,
    ) -> Optional[dict[str, Any]]:
        if effective_view != "body":
            return None
        if chapter_id:
            selected = next(
                (item for item in chapters if str(item["id"]) == chapter_id),
                None,
            )
            if not selected:
                raise WorkbenchNotFound
            return selected
        if not chapters:
            return None
        remembered_id = str(current_version.get("last_chapter_id") or "")
        return next(
            (item for item in chapters if str(item["id"]) == remembered_id),
            chapters[0],
        )

    def _editable_chapter_view(
        self,
        *,
        user_id: int,
        project_id: str,
        chapters: list[dict[str, Any]],
        selected_chapter: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        content = ""
        selected_index = -1
        previous_chapter = None
        next_chapter = None
        head_version = None
        head_version_hash = ""
        has_edit_buffer = False
        stale_edit_buffer = False
        if selected_chapter:
            selected_index, previous_chapter, next_chapter = self._chapter_neighbors(
                chapters,
                selected_chapter,
            )
            current_head_id = str(selected_chapter.get("head_version_id") or "")
            if current_head_id:
                head_version = self.database.get_chapter_version(
                    user_id,
                    project_id,
                    str(selected_chapter["id"]),
                    current_head_id,
                )
            committed_content = (
                _read_optional_text(Path(str(head_version["content_path"])))
                if head_version
                else ""
            )
            if head_version:
                head_version_hash = hashlib.sha256(
                    committed_content.encode("utf-8")
                ).hexdigest()
            buffer_content = selected_chapter.get("edit_buffer_content")
            buffer_base = str(
                selected_chapter.get("edit_buffer_base_version_id") or ""
            )
            if buffer_content is not None and buffer_base == current_head_id:
                content = str(buffer_content)
                has_edit_buffer = True
            else:
                content = committed_content
                stale_edit_buffer = buffer_content is not None
        return {
            "chapter_content": content,
            "chapter_index": selected_index,
            "previous_chapter": previous_chapter,
            "next_chapter": next_chapter,
            "head_version": head_version,
            "head_version_hash": head_version_hash,
            "has_edit_buffer": has_edit_buffer,
            "stale_edit_buffer": stale_edit_buffer,
        }

    def _readonly_chapter_view(
        self,
        *,
        chapters: list[dict[str, Any]],
        selected_chapter: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        content = ""
        content_hash = ""
        selected_index = -1
        previous_chapter = None
        next_chapter = None
        if selected_chapter:
            selected_index, previous_chapter, next_chapter = self._chapter_neighbors(
                chapters,
                selected_chapter,
            )
            content = _read_optional_text(Path(str(selected_chapter["content_path"])))
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return {
            "chapter_content": content,
            "chapter_content_hash": content_hash,
            "chapter_index": selected_index,
            "previous_chapter": previous_chapter,
            "next_chapter": next_chapter,
        }

    @staticmethod
    def _chapter_neighbors(
        chapters: list[dict[str, Any]],
        selected_chapter: dict[str, Any],
    ) -> tuple[int, Optional[dict[str, Any]], Optional[dict[str, Any]]]:
        selected_index = next(
            index
            for index, item in enumerate(chapters)
            if str(item["id"]) == str(selected_chapter["id"])
        )
        previous_chapter = chapters[selected_index - 1] if selected_index > 0 else None
        next_chapter = (
            chapters[selected_index + 1]
            if selected_index + 1 < len(chapters)
            else None
        )
        return selected_index, previous_chapter, next_chapter

    def _active_novel_conversation(
        self,
        *,
        user_id: int,
        project_id: str,
        conversation_id: Optional[str],
        effective_view: str,
        selected_chapter: Optional[dict[str, Any]],
        conversations: list[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        if conversation_id:
            conversation = self.assistant_chat_service.conversations.get(
                user_id=user_id,
                conversation_id=conversation_id,
            )
            if (
                not conversation
                or str(conversation.get("project_id") or "") != project_id
            ):
                raise WorkbenchNotFound
            return conversation
        latest = None
        if effective_view == "archive":
            latest = next(
                (
                    item
                    for item in conversations
                    if str(item.get("scope_type") or "") == "project"
                ),
                None,
            )
        elif selected_chapter:
            latest = next(
                (
                    item
                    for item in conversations
                    if str(item.get("novel_chapter_id") or "")
                    == str(selected_chapter["id"])
                ),
                None,
            )
        if not latest:
            return None
        return self.assistant_chat_service.conversations.get(
            user_id=user_id,
            conversation_id=str(latest["id"]),
        )

    def _active_document_conversation(
        self,
        *,
        user_id: int,
        document_id: str,
        conversation_id: Optional[str],
        effective_view: str,
        selected_chapter: Optional[dict[str, Any]],
        conversations: list[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        if conversation_id:
            conversation = self.assistant_chat_service.conversations.get(
                user_id=user_id,
                conversation_id=conversation_id,
            )
            if (
                not conversation
                or str(conversation.get("document_id") or "") != document_id
            ):
                raise WorkbenchNotFound
            return conversation
        latest = None
        if effective_view == "body" and selected_chapter:
            latest = next(
                (
                    item
                    for item in conversations
                    if str(item.get("reference_chapter_id") or "")
                    == str(selected_chapter["id"])
                ),
                None,
            )
        elif effective_view == "archive":
            latest = next(
                (
                    item
                    for item in conversations
                    if not item.get("reference_chapter_id")
                ),
                None,
            )
        if not latest:
            return None
        return self.assistant_chat_service.conversations.get(
            user_id=user_id,
            conversation_id=str(latest["id"]),
        )

    def _editable_archive_view(
        self,
        *,
        user_id: int,
        project_id: str,
        work: dict[str, Any],
        current_version: dict[str, Any],
        chapters: list[dict[str, Any]],
        effective_view: str,
        active_archive_tab: str,
        project_memory_records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        setting_characters: list[dict[str, Any]] = []
        setting_world_entries: list[dict[str, Any]] = []
        setting_relationships: list[dict[str, Any]] = []
        setting_volumes: list[dict[str, Any]] = []
        setting_story_blueprint = None
        setting_story_arcs: list[dict[str, Any]] = []
        setting_voice_profile = None
        current_story_state: dict[str, Any] = {}
        story_state_refreshing_count = 0
        archive_entries: list[dict[str, Any]] = []
        archive_analyses: list[dict[str, Any]] = []
        archive_story_memory_records: list[dict[str, Any]] = []
        chapter_histories: list[dict[str, Any]] = []
        if effective_view == "archive":
            setting_characters = self.database.list_novel_characters(
                user_id,
                project_id,
            )
        if effective_view == "archive" and active_archive_tab == "creative":
            setting_world_entries = self.database.list_world_entries(
                user_id,
                project_id,
            )
            setting_relationships = self.database.list_character_relationships(
                user_id,
                project_id,
            )
            setting_volumes = self.planning_service.list_volumes(
                user_id=user_id,
                project_id=project_id,
            )
            setting_story_blueprint = self.story_planning_service.get_blueprint(
                user_id=user_id,
                project_id=project_id,
            )
            setting_story_arcs = self.story_planning_service.list_arcs(
                user_id=user_id,
                project_id=project_id,
            )
            setting_voice_profile = self.style_service.get_voice_profile(
                user_id=user_id,
                project_id=project_id,
            )
            continuity_dashboard = self.continuity_service.get_dashboard(
                user_id=user_id,
                project_id=project_id,
            )
            current_story_state = dict((continuity_dashboard or {}).get("state") or {})
            story_state_refreshing_count = sum(
                1 for item in chapters if bool(item.get("needs_recheck"))
            )
        if effective_view == "archive":
            archive_entries = self.database.list_work_archive_entries(
                user_id,
                str(work["id"]),
                str(current_version["id"]),
            )
            archive_analyses = self.database.list_work_analyses(
                user_id,
                str(work["id"]),
                str(current_version["id"]),
            )
            if active_archive_tab == "analysis":
                archive_story_memory_records = project_memory_records
            elif active_archive_tab == "versions":
                for item in chapters:
                    versions = self.database.list_chapter_versions(
                        user_id,
                        project_id,
                        str(item["id"]),
                        limit=5,
                    )
                    version_count = self.database.count_chapter_versions(
                        user_id,
                        project_id,
                        str(item["id"]),
                    )
                    chapter_histories.append(
                        {
                            "chapter": item,
                            "versions": versions,
                            "version_count": version_count,
                            "history_count": max(0, version_count - 1),
                        }
                    )
        return {
            "setting_characters": setting_characters,
            "setting_world_entries": setting_world_entries,
            "setting_relationships": setting_relationships,
            "setting_volumes": setting_volumes,
            "setting_story_blueprint": setting_story_blueprint,
            "setting_story_arcs": setting_story_arcs,
            "setting_voice_profile": setting_voice_profile,
            "current_story_state": current_story_state,
            "story_state_refreshing_count": story_state_refreshing_count,
            "archive_project": self.database.get_novel_project(user_id, project_id),
            "archive_characters": setting_characters,
            "archive_entries": archive_entries,
            "archive_analyses": archive_analyses,
            "archive_story_memory_records": archive_story_memory_records,
            "chapter_histories": chapter_histories,
        }
