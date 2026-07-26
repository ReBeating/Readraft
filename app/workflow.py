from __future__ import annotations

from typing import Any, Dict, Optional

from .db import Database
from .memory_service import MemoryService
from .planning_service import PlanningService
from .scene_service import SceneService


WORKFLOW_STEP_LABELS = (
    ("task_card", "章节计划"),
    ("draft", "当前正文"),
    ("memory", "后台记忆"),
)


def _action(
    *,
    label: str,
    url: str,
    hint: str = "",
    method: str = "get",
) -> Dict[str, str]:
    return {
        "label": label,
        "url": url,
        "hint": hint,
        "method": method,
    }


def _step(
    *,
    key: str,
    label: str,
    status: str,
    detail: str,
    url: str,
) -> Dict[str, str]:
    return {
        "key": key,
        "label": label,
        "status": status,
        "detail": detail,
        "url": url,
    }


class ChapterWorkflowService:
    """Describe the chapter's current state without adding review gates."""

    def __init__(
        self,
        database: Database,
        *,
        planning_service: Optional[PlanningService] = None,
        scene_service: Optional[SceneService] = None,
        memory_service: Optional[MemoryService] = None,
    ):
        self.database = database
        self.planning_service = planning_service or PlanningService(database)
        self.scene_service = scene_service or SceneService(database)
        self.memory_service = memory_service or MemoryService(database)

    def get_state(
        self, *, user_id: int, project_id: str, chapter_id: str
    ) -> Optional[Dict[str, Any]]:
        chapter = self.database.get_novel_chapter(
            user_id, project_id, chapter_id
        )
        if not chapter:
            return None

        task_card = self.planning_service.get_task_card(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
        )
        workbench = self.scene_service.get_workbench(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
        )
        versions = self.database.list_chapter_versions(
            user_id, project_id, chapter_id
        )
        working_version_id = str(chapter.get("working_version_id") or "")
        canonical_version_id = str(
            chapter.get("canonical_version_id") or ""
        )
        current_version_id = canonical_version_id or working_version_id
        current_version = next(
            (
                item
                for item in versions
                if str(item.get("id") or "") == current_version_id
            ),
            None,
        )
        canon_current = bool(
            canonical_version_id
            and working_version_id == canonical_version_id
        )
        task_confirmed = bool(
            task_card and str(task_card.get("status") or "") == "confirmed"
        )
        scene_count = int((workbench or {}).get("scene_count") or 0)
        ready_count = int((workbench or {}).get("ready_count") or 0)
        all_scenes_ready = bool((workbench or {}).get("all_ready"))
        active_job = self._active_writing_job(user_id=user_id)

        deltas = self.memory_service.list_chapter_deltas(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
        )
        canonical_delta = next(
            (
                item
                for item in deltas
                if canonical_version_id
                and str(item.get("version_id") or "")
                == canonical_version_id
                and str(item.get("status") or "") == "projected"
            ),
            None,
        )
        memory_running = self._memory_is_running(
            user_id=user_id,
            chapter_id=chapter_id,
            version_id=canonical_version_id,
        )

        chapter_url = f"/novels/{project_id}/chapters/{chapter_id}"
        task_url = f"{chapter_url}/task-card"
        scenes_url = f"{chapter_url}/scenes"

        if task_confirmed:
            task_status = "complete"
            task_detail = f"已确认 {scene_count} 个场景节拍"
        elif canonical_version_id:
            task_status = "skipped"
            task_detail = "已有正文；可以需要时再补充章节计划"
        else:
            task_status = "current"
            task_detail = "确定本章目标和场景安排"

        if canon_current and current_version:
            draft_status = "complete"
            draft_detail = (
                f"当前版本 · {int(current_version.get('char_count') or 0)} 字"
            )
        elif current_version:
            draft_status = "current"
            draft_detail = "已有正文版本，尚未设为当前版本"
        elif task_confirmed:
            draft_status = "current"
            draft_detail = (
                f"{ready_count}/{scene_count} 个场景已有正文"
                if scene_count
                else "可以开始写作"
            )
        else:
            draft_status = "locked"
            draft_detail = "完成章节计划后开始写作"

        if canonical_delta:
            memory_status = "complete"
            memory_detail = "已根据当前正文自动更新"
        elif memory_running:
            memory_status = "current"
            memory_detail = "正在后台整理，不需要人工审核"
        elif canonical_version_id:
            memory_status = "skipped"
            memory_detail = "暂未更新，不影响正文；下次保存时会再次整理"
        else:
            memory_status = "locked"
            memory_detail = "正文保存后自动整理"

        steps = [
            _step(
                key="task_card",
                label="章节计划",
                status=task_status,
                detail=task_detail,
                url=task_url,
            ),
            _step(
                key="draft",
                label="当前正文",
                status=draft_status,
                detail=draft_detail,
                url=chapter_url,
            ),
            _step(
                key="memory",
                label="后台记忆",
                status=memory_status,
                detail=memory_detail,
                url=chapter_url,
            ),
        ]

        stage, headline, explanation, primary, secondary = self._next_action(
            project_id=project_id,
            chapter_id=chapter_id,
            chapter_url=chapter_url,
            task_url=task_url,
            scenes_url=scenes_url,
            task_confirmed=task_confirmed,
            scene_count=scene_count,
            ready_count=ready_count,
            all_scenes_ready=all_scenes_ready,
            workbench=workbench,
            current_version=current_version,
            working_version_id=working_version_id,
            canon_current=canon_current,
            memory_running=memory_running,
            active_job=active_job,
        )
        completed_count = sum(
            str(item["status"]) in {"complete", "skipped"} for item in steps
        )
        return {
            "project_id": project_id,
            "chapter_id": chapter_id,
            "chapter_title": str(chapter.get("title") or ""),
            "chapter_position": int(chapter.get("position") or 0),
            "stage": stage,
            "headline": headline,
            "explanation": explanation,
            "primary_action": primary,
            "secondary_actions": secondary,
            "steps": steps,
            "completed_count": completed_count,
            "step_count": len(steps),
            "progress_percent": round(completed_count / len(steps) * 100),
            "complete": canon_current,
            "active_job": active_job,
            "working_version_id": working_version_id,
            "canonical_version_id": canonical_version_id,
            "canonical_delta_id": str(
                (canonical_delta or {}).get("id") or ""
            ),
        }

    def _active_writing_job(
        self, *, user_id: int
    ) -> Optional[Dict[str, Any]]:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT id, project_id, chapter_id, operation, status,
                       subject_id, version_id
                FROM generation_jobs
                WHERE user_id=? AND status IN ('queued', 'running')
                    AND operation<>'extract_story_delta'
                ORDER BY created_at, rowid
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
        return dict(row) if row else None

    def _memory_is_running(
        self,
        *,
        user_id: int,
        chapter_id: str,
        version_id: str,
    ) -> bool:
        if not version_id:
            return False
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM generation_jobs
                WHERE user_id=? AND chapter_id=? AND version_id=?
                    AND operation='extract_story_delta'
                    AND status IN ('queued', 'running')
                LIMIT 1
                """,
                (user_id, chapter_id, version_id),
            ).fetchone()
        return row is not None

    @staticmethod
    def _next_action(
        *,
        project_id: str,
        chapter_id: str,
        chapter_url: str,
        task_url: str,
        scenes_url: str,
        task_confirmed: bool,
        scene_count: int,
        ready_count: int,
        all_scenes_ready: bool,
        workbench: Optional[Dict[str, Any]],
        current_version: Optional[Dict[str, Any]],
        working_version_id: str,
        canon_current: bool,
        memory_running: bool,
        active_job: Optional[Dict[str, Any]],
    ) -> tuple[
        str,
        str,
        str,
        Dict[str, str],
        list[Dict[str, str]],
    ]:
        if canon_current and (not active_job or memory_running):
            return (
                "memory_sync" if memory_running else "complete",
                "当前正文已经保存",
                (
                    "故事记忆正在后台更新；不需要审核，也不影响你继续阅读或创作。"
                    if memory_running
                    else "保存或生成会直接更新当前正文，旧版本仍保留在版本历史中。"
                ),
                _action(
                    label="返回作品工作台",
                    url=f"/novels/{project_id}/workbench",
                ),
                [],
            )

        if active_job:
            operation = str(active_job.get("operation") or "")
            operation_label = {
                "plan_chapter": "章节规划",
                "plan_scene_beats": "场景节拍拆解",
                "generate_scene": "场景生成",
                "rewrite_scene": "场景重写",
                "audit_ai_style": "文风分析",
                "rewrite_style_issue": "定点改写",
                "extract_story_delta": "后台记忆整理",
            }.get(operation, "AI 写作")
            same_chapter = (
                str(active_job.get("chapter_id") or "") == chapter_id
            )
            return (
                "running",
                (
                    f"{operation_label}正在进行"
                    if same_chapter
                    else f"另一个章节的{operation_label}正在进行"
                ),
                "任务会在后台继续，完成后直接更新对应内容。",
                _action(
                    label="查看正在运行的任务",
                    url=f"/writing-jobs/{active_job['id']}",
                ),
                [],
            )

        if current_version and working_version_id:
            return (
                "version",
                "已有正文版本可使用",
                "这是旧流程留下的未采用版本；选择后会成为当前正文，其他版本仍保留。",
                _action(
                    label="设为当前正文",
                    url=(
                        f"{chapter_url}/versions/"
                        f"{working_version_id}/accept"
                    ),
                    method="post",
                ),
                [_action(label="先查看正文", url=chapter_url)],
            )

        if not task_confirmed:
            return (
                "task_card",
                "先确定本章要写什么",
                "章节计划用来约束目标和场景；确认后即可直接生成或手写正文。",
                _action(label="规划本章", url=task_url),
                [],
            )

        if scene_count >= 2 and all_scenes_ready:
            return (
                "assembly",
                "所有场景都已有正文",
                "按任务卡顺序组装后，会直接成为当前章节，旧版本仍保留。",
                _action(
                    label=f"组装 {scene_count} 个场景",
                    url=f"{scenes_url}/assemble",
                    method="post",
                ),
                [
                    _action(
                        label="改用整章生成",
                        url=f"{chapter_url}#writer",
                    )
                ],
            )

        if scene_count >= 2:
            unfinished = next(
                (
                    item
                    for item in (workbench or {}).get("scenes", [])
                    if not item.get("ready")
                ),
                None,
            )
            anchor = f"#scene-{unfinished['id']}" if unfinished else ""
            return (
                "scenes",
                "开始写正文",
                f"已有 {ready_count}/{scene_count} 个场景具备正文；也可以直接使用整章生成。",
                _action(
                    label=(
                        "继续未完成场景"
                        if ready_count
                        else "开始逐场景创作"
                    ),
                    url=f"{scenes_url}{anchor}",
                ),
                [
                    _action(
                        label="使用整章生成",
                        url=f"{chapter_url}#writer",
                    )
                ],
            )

        return (
            "draft",
            "开始写正文",
            "可以直接手写或使用整章生成；保存后会成为当前版本。",
            _action(label="打开正文编辑器", url=f"{chapter_url}#writer"),
            [_action(label="补充场景计划", url=task_url)],
        )
