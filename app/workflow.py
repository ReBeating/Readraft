from __future__ import annotations

from typing import Any, Dict, Optional

from .db import Database
from .memory_service import MemoryService
from .planning_service import PlanningService
from .scene_service import SceneService
from .style_service import StyleService


WORKFLOW_STEP_LABELS = (
    ("task_card", "确认任务卡"),
    ("draft", "形成正文候选"),
    ("hard_audit", "通过硬审计"),
    ("style_audit", "去 AI 味审校"),
    ("canon", "作者确认正史"),
    ("memory", "审核故事记忆"),
)

QUALITY_READY_STATUSES = {"pass", "overridden"}
STYLE_READY_STATUSES = {"pass", "reviewed"}
DELTA_REVIEW_STATUSES = {"proposed", "author_edited"}


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
    """Derive the author's next safe action without mutating story state."""

    def __init__(
        self,
        database: Database,
        *,
        planning_service: Optional[PlanningService] = None,
        scene_service: Optional[SceneService] = None,
        memory_service: Optional[MemoryService] = None,
        style_service: Optional[StyleService] = None,
    ):
        self.database = database
        self.planning_service = planning_service or PlanningService(database)
        self.scene_service = scene_service or SceneService(database)
        self.memory_service = memory_service or MemoryService(database)
        self.style_service = style_service or StyleService(database)

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
        working_version_id = str(
            chapter.get("working_version_id") or ""
        )
        canonical_version_id = str(
            chapter.get("canonical_version_id") or ""
        )
        working_version = next(
            (
                item
                for item in versions
                if str(item.get("id") or "") == working_version_id
            ),
            None,
        )
        if not working_version and canonical_version_id:
            working_version = next(
                (
                    item
                    for item in versions
                    if str(item.get("id") or "")
                    == canonical_version_id
                ),
                None,
            )
            if working_version:
                working_version_id = canonical_version_id

        voice_profile = self.style_service.get_voice_profile(
            user_id=user_id, project_id=project_id
        )
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
                and str(item.get("status") or "")
                in {
                    "proposed",
                    "author_edited",
                    "accepted",
                    "projected",
                }
            ),
            None,
        )
        active_job = self._active_writing_job(user_id=user_id)

        chapter_url = f"/novels/{project_id}/chapters/{chapter_id}"
        task_url = f"{chapter_url}/task-card"
        scenes_url = f"{chapter_url}/scenes"
        task_confirmed = bool(
            task_card and str(task_card.get("status") or "") == "confirmed"
        )
        scene_count = int((workbench or {}).get("scene_count") or 0)
        ready_count = int((workbench or {}).get("ready_count") or 0)
        all_scenes_ready = bool((workbench or {}).get("all_ready"))
        quality_status = str(
            (working_version or {}).get("quality_status") or "pending"
        )
        hard_ready = bool(
            working_version and quality_status in QUALITY_READY_STATUSES
        )
        style_status = str(
            (working_version or {}).get("style_status") or "pending"
        )
        voice_confirmed = bool(
            voice_profile
            and str(voice_profile.get("status") or "") == "confirmed"
        )
        canon_current = bool(
            working_version_id
            and canonical_version_id
            and working_version_id == canonical_version_id
        )
        delta_status = str(
            (canonical_delta or {}).get("status") or ""
        )
        memory_projected = bool(
            canon_current and delta_status == "projected"
        )

        quality_url = (
            f"{chapter_url}/versions/{working_version_id}/quality"
            if working_version_id
            else chapter_url
        )
        style_url = (
            f"{chapter_url}/versions/{working_version_id}/style"
            if working_version_id
            else chapter_url
        )

        steps = self._build_steps(
            task_confirmed=task_confirmed,
            task_url=task_url,
            chapter_url=chapter_url,
            scenes_url=scenes_url,
            scene_count=scene_count,
            ready_count=ready_count,
            working_version=working_version,
            quality_status=quality_status,
            quality_url=quality_url,
            hard_ready=hard_ready,
            style_status=style_status,
            style_url=style_url,
            voice_confirmed=voice_confirmed,
            canon_current=canon_current,
            canonical_delta=canonical_delta,
            memory_projected=memory_projected,
        )

        stage, headline, explanation, primary, secondary = (
            self._next_action(
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
                working_version=working_version,
                working_version_id=working_version_id,
                quality_status=quality_status,
                quality_url=quality_url,
                hard_ready=hard_ready,
                style_status=style_status,
                style_url=style_url,
                voice_confirmed=voice_confirmed,
                canon_current=canon_current,
                canonical_version_id=canonical_version_id,
                canonical_delta=canonical_delta,
                memory_projected=memory_projected,
                active_job=active_job,
            )
        )
        finished_statuses = {"complete", "skipped"}
        completed_count = sum(
            str(item["status"]) in finished_statuses for item in steps
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
            "complete": stage == "complete",
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
                ORDER BY created_at, rowid
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
        return dict(row) if row else None

    @staticmethod
    def _build_steps(
        *,
        task_confirmed: bool,
        task_url: str,
        chapter_url: str,
        scenes_url: str,
        scene_count: int,
        ready_count: int,
        working_version: Optional[Dict[str, Any]],
        quality_status: str,
        quality_url: str,
        hard_ready: bool,
        style_status: str,
        style_url: str,
        voice_confirmed: bool,
        canon_current: bool,
        canonical_delta: Optional[Dict[str, Any]],
        memory_projected: bool,
    ) -> list[Dict[str, str]]:
        steps: list[Dict[str, str]] = []
        if task_confirmed:
            task_status = "complete"
            task_detail = f"已确认 {scene_count} 个场景节拍"
        elif canon_current:
            task_status = "skipped"
            task_detail = "本章已成为正史，未补录历史任务卡"
        else:
            task_status = "current"
            task_detail = "确认章节职责与至少两个场景节拍"
        steps.append(
            _step(
                key="task_card",
                label="确认任务卡",
                status=task_status,
                detail=task_detail,
                url=task_url,
            )
        )

        if working_version:
            source = str(working_version.get("source") or "")
            source_label = {
                "scene_assembly": "场景组装候选",
                "generated": "整章生成候选",
                "manual": "作者手工候选",
                "targeted_rewrite": "定点改写候选",
                "restored": "恢复候选",
            }.get(source, "正文候选")
            draft_status = "complete"
            draft_detail = (
                f"{source_label} · "
                f"{int(working_version.get('char_count') or 0)} 字"
            )
        elif task_confirmed:
            draft_status = "current"
            if scene_count:
                draft_detail = (
                    f"场景模式 {ready_count}/{scene_count} 可组装，"
                    "也可使用整章模式"
                )
            else:
                draft_detail = "先补足场景节拍，再生成正文候选"
        else:
            draft_status = "locked"
            draft_detail = "任务卡确认后开放"
        steps.append(
            _step(
                key="draft",
                label="形成正文候选",
                status=draft_status,
                detail=draft_detail,
                url=scenes_url if scene_count else f"{chapter_url}#writer",
            )
        )

        if not working_version:
            hard_status = "locked"
            hard_detail = "正文候选生成后开放"
        elif hard_ready:
            hard_status = "complete"
            hard_detail = (
                "作者已记录覆盖原因"
                if quality_status == "overridden"
                else "当前候选已通过硬门禁"
            )
        elif quality_status == "block":
            hard_status = "attention"
            hard_detail = "存在阻断问题，需要修改、重审或明确覆盖"
        else:
            hard_status = "current"
            hard_detail = "核对正史、知情、任务覆盖、视角与字数"
        steps.append(
            _step(
                key="hard_audit",
                label="通过硬审计",
                status=hard_status,
                detail=hard_detail,
                url=quality_url,
            )
        )

        if not working_version or not hard_ready:
            style_step_status = "locked"
            style_detail = "硬审计就绪后进行软质量检查"
        elif style_status in STYLE_READY_STATUSES:
            style_step_status = "complete"
            style_detail = (
                "没有定位到明显问题"
                if style_status == "pass"
                else "定位问题已由作者逐项处理"
            )
        elif canon_current:
            style_step_status = "skipped"
            style_detail = "软审校未完成；作者已继续确认正史"
        elif not voice_confirmed:
            style_step_status = "attention"
            style_detail = "先确认作品声纹，审校器才有作者化基准"
        elif style_status == "issues":
            style_step_status = "attention"
            style_detail = "仍有具体问题等待作者处理或保留"
        else:
            style_step_status = "current"
            style_detail = "定位具体原句，不做无差别全文洗稿"
        steps.append(
            _step(
                key="style_audit",
                label="去 AI 味审校",
                status=style_step_status,
                detail=style_detail,
                url=style_url,
            )
        )

        if canon_current:
            canon_status = "complete"
            canon_detail = "当前工作版本就是作者确认正史"
        elif working_version and hard_ready:
            canon_status = "current"
            canon_detail = "软审校是建议步骤；正史决定仍由作者作出"
        else:
            canon_status = "locked"
            canon_detail = "正文候选需先完成硬门禁"
        steps.append(
            _step(
                key="canon",
                label="作者确认正史",
                status=canon_status,
                detail=canon_detail,
                url=quality_url,
            )
        )

        delta_status = str((canonical_delta or {}).get("status") or "")
        if memory_projected:
            memory_status = "complete"
            memory_detail = "本章变化已进入后续章节使用的正史 Memory"
            memory_url = (
                f"/story-deltas/{canonical_delta['id']}"
                if canonical_delta
                else chapter_url
            )
        elif canon_current and delta_status in DELTA_REVIEW_STATUSES:
            memory_status = "current"
            memory_detail = "Observer 提案等待作者核对正文证据"
            memory_url = f"/story-deltas/{canonical_delta['id']}"
        elif canon_current:
            memory_status = "current"
            memory_detail = "提取本章事件、事实、知情、剧情线与伏笔变化"
            memory_url = f"{chapter_url}#chapter-workflow"
        else:
            memory_status = "locked"
            memory_detail = "正文成为正史后才允许提取"
            memory_url = chapter_url
        steps.append(
            _step(
                key="memory",
                label="审核故事记忆",
                status=memory_status,
                detail=memory_detail,
                url=memory_url,
            )
        )
        return steps

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
        working_version: Optional[Dict[str, Any]],
        working_version_id: str,
        quality_status: str,
        quality_url: str,
        hard_ready: bool,
        style_status: str,
        style_url: str,
        voice_confirmed: bool,
        canon_current: bool,
        canonical_version_id: str,
        canonical_delta: Optional[Dict[str, Any]],
        memory_projected: bool,
        active_job: Optional[Dict[str, Any]],
    ) -> tuple[
        str,
        str,
        str,
        Dict[str, str],
        list[Dict[str, str]],
    ]:
        if active_job:
            operation = str(active_job.get("operation") or "")
            operation_label = {
                "plan_chapter": "章节规划",
                "plan_scene_beats": "场景节拍拆解",
                "generate_scene": "场景生成",
                "rewrite_scene": "场景重写",
                "audit_scene": "场景检查",
                "audit_chapter": "整章硬审计",
                "audit_ai_style": "AI 味审校",
                "rewrite_style_issue": "定点改写",
                "extract_story_delta": "故事记忆提取",
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
                (
                    "任务在后台运行；结果仍会停在候选或待审核状态。"
                    if same_chapter
                    else "当前账号一次只运行一个写作任务；完成后即可继续本章。"
                ),
                _action(
                    label="查看正在运行的任务",
                    url=f"/writing-jobs/{active_job['id']}",
                ),
                [],
            )

        delta_status = str((canonical_delta or {}).get("status") or "")
        if canon_current:
            if (
                canonical_delta
                and delta_status in DELTA_REVIEW_STATUSES
            ):
                return (
                    "memory_review",
                    "最后核对本章造成的故事变化",
                    "Observer 只提交提案；确认后，事件、事实、知情、剧情线和伏笔才会进入后续 Writer。",
                    _action(
                        label="审核 Story Delta",
                        url=f"/story-deltas/{canonical_delta['id']}",
                    ),
                    [],
                )
            if not memory_projected:
                return (
                    "memory_extract",
                    "正史已确认，等待建立可检索记忆",
                    "如果自动提取曾失败或被拒绝，可以从当前正史安全重试；它仍只会生成待审核提案。",
                    _action(
                        label="提取 Story Delta",
                        url=(
                            f"/novels/{project_id}/chapters/{chapter_id}"
                            f"/versions/{canonical_version_id}"
                            "/extract-memory"
                        ),
                        method="post",
                    ),
                    [],
                )
            return (
                "complete",
                "本章创作闭环已经完成",
                "正文和 Story Delta 都已由作者确认；后续章节可以读取这份正史状态。",
                _action(
                    label="返回作品工作台",
                    url=f"/novels/{project_id}/workbench",
                ),
                (
                    [
                        _action(
                            label="查看已确认故事记忆",
                            url=f"/story-deltas/{canonical_delta['id']}",
                        )
                    ]
                    if canonical_delta
                    else []
                ),
            )

        if not task_confirmed:
            return (
                "task_card",
                "先确认本章要完成什么",
                "任务卡把滚动骨架拆成状态变化和 2–5 个可执行场景；未确认计划不会进入 Writer。",
                _action(label="检查并确认任务卡", url=task_url),
                [],
            )

        if not working_version:
            if scene_count >= 2:
                if all_scenes_ready:
                    label = f"组装 {scene_count} 个已通过场景"
                    explanation = (
                        "全部场景已经就绪；组装只会创建章节候选，"
                        "不会修改正史。"
                    )
                    target = f"{scenes_url}/assemble"
                    action_method = "post"
                    stage = "assembly"
                else:
                    unfinished = next(
                        (
                            item
                            for item in (workbench or {}).get("scenes", [])
                            if not item.get("ready")
                        ),
                        None,
                    )
                    anchor = (
                        f"#scene-{unfinished['id']}" if unfinished else ""
                    )
                    label = (
                        "继续第一个未就绪场景"
                        if ready_count
                        else "开始逐场景创作"
                    )
                    explanation = (
                        f"已有 {ready_count}/{scene_count} 个场景可组装。"
                        "逐场景模式会保留版本、检查结果和明确边界。"
                    )
                    target = f"{scenes_url}{anchor}"
                    action_method = "get"
                    stage = "scenes"
                return (
                    stage,
                    "把任务卡变成可审阅的正文候选",
                    explanation,
                    _action(
                        label=label,
                        url=target,
                        method=action_method,
                    ),
                    [
                        _action(
                            label="改用整章快速生成",
                            url=f"{chapter_url}#writer",
                            hint="仍只生成候选版本",
                        ),
                        _action(label="返回修改任务卡", url=task_url),
                    ],
                )
            return (
                "draft",
                "任务卡缺少可执行场景",
                "先补足至少两个场景节拍；也可以在章节编辑器使用整章快速生成。",
                _action(label="补足场景节拍", url=task_url),
                [
                    _action(
                        label="使用整章生成",
                        url=f"{chapter_url}#writer",
                    )
                ],
            )

        if not hard_ready:
            return (
                "hard_audit",
                (
                    "候选稿被硬门禁阻断"
                    if quality_status == "block"
                    else "候选稿等待整章硬审计"
                ),
                (
                    "先核对跨场景连续性、正史、人物知情、必做项、"
                    "禁止事项、视角和 2000 字下限。"
                ),
                _action(
                    label=(
                        "打开并处理硬审计问题"
                        if quality_status == "block"
                        else "打开整章硬审计"
                    ),
                    url=quality_url,
                ),
                [_action(label="返回正文编辑器", url=chapter_url)],
            )

        if not canon_current:
            if not voice_confirmed:
                return (
                    "style_profile",
                    "先给去 AI 味审校一个作者化基准",
                    "确认作品声纹后，审校器才能按你的叙述距离、句段节奏、对话和留白规则定位问题。",
                    _action(
                        label="填写并确认作品声纹",
                        url=(
                            f"/novels/{project_id}/workbench"
                            "?view=archive&archive_tab=creative&settings_tab=style"
                        ),
                    ),
                    [
                        _action(
                            label="暂时跳过软审校，查看正史决定",
                            url=quality_url,
                        )
                    ],
                )
            if style_status not in STYLE_READY_STATUSES:
                return (
                    "style_audit",
                    (
                        "逐项处理已定位的 AI 味问题"
                        if style_status == "issues"
                        else "进行一次定点式去 AI 味审校"
                    ),
                    "软审校不会自动洗稿；每条建议必须引用具体原句，作者可以改写、忽略或保留。",
                    _action(
                        label=(
                            "查看并处理具体问题"
                            if style_status == "issues"
                            else "打开 AI 味审校"
                        ),
                        url=style_url,
                    ),
                    [
                        _action(
                            label="暂时跳过软审校，查看正史决定",
                            url=quality_url,
                        )
                    ],
                )
            return (
                "canon",
                "候选稿已完成提交前检查",
                "硬审计已经通过，AI 味问题也已检查；请通读正文并由作者决定是否成为正史。",
                _action(label="通读并确认正史", url=quality_url),
                [_action(label="返回正文继续手工修改", url=chapter_url)],
            )

        raise RuntimeError("章节工作流状态不完整")
