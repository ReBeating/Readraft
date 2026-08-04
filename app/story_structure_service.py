from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .db import Database, utc_after, utc_now
from .json_support import (
    dump_canonical_json as _json,
    json_fingerprint as _fingerprint,
    load_json as _load_json,
)
from .story_plan_suggestion_service import StoryPlanSuggestionService
from .story_structure_schema import (
    StoryStructureOption,
    StoryStructureProposalSet,
)


VOLUME_FIELDS = (
    "title",
    "goal",
    "start_state",
    "end_state",
    "major_conflict",
    "payoff",
)
CHAPTER_RESTORE_FIELDS = (
    "title",
    "outline",
    "key_points",
    "status",
    "volume_id",
    "char_count",
    "head_version_id",
    "needs_recheck",
    "skeleton_role",
    "skeleton_arc_titles_json",
    "skeleton_ending_hook",
    "skeleton_application_id",
    "created_at",
    "updated_at",
)
PLAN_RESTORE_FIELDS = (
    "purpose",
    "start_state",
    "end_state",
    "central_conflict",
    "emotional_value",
    "plot_threads_json",
    "must_happen_json",
    "must_preserve_json",
    "forbidden_json",
    "foreshadow_setup_json",
    "foreshadow_payoff_json",
    "ending_hook",
    "target_chars",
    "status",
    "source",
    "created_at",
    "updated_at",
    "confirmed_at",
)


def _chapter_values(chapter) -> Dict[str, Any]:
    return {
        "title": chapter.title,
        "outline": chapter.purpose,
        "key_points": "\n".join(chapter.key_points),
        "volume_position": chapter.volume_position,
        "skeleton_role": chapter.structural_role,
        "skeleton_arc_titles_json": _json(chapter.arc_titles),
        "skeleton_ending_hook": chapter.ending_hook,
    }


class StoryStructureSuggestionService:
    def __init__(self, database: Database, novels_dir: Path):
        self.database = database
        self.novels_dir = novels_dir

    @staticmethod
    def _build_context(
        connection,
        *,
        user_id: int,
        project_id: str,
        chapter_count: int,
    ) -> Optional[Dict[str, Any]]:
        base = StoryPlanSuggestionService._build_context(
            connection,
            user_id=user_id,
            project_id=project_id,
        )
        if base is None:
            return None
        current_position = int(
            base.get("current_canonical_position") or 0
        )
        volume_rows = connection.execute(
            """
            SELECT v.id, v.position, v.title, v.goal, v.start_state,
                   v.end_state, v.major_conflict, v.payoff, v.status,
                   v.created_at, v.updated_at,
                   (
                       SELECT COUNT(*) FROM novel_chapters ch
                       WHERE ch.volume_id=v.id
                   ) AS chapter_count,
                   (
                       SELECT COUNT(*) FROM novel_chapters ch
                       WHERE ch.volume_id=v.id
                         AND ch.head_version_id IS NOT NULL
                   ) AS canonical_chapter_count
            FROM novel_volumes v
            WHERE v.project_id=?
            ORDER BY v.position
            """,
            (project_id,),
        ).fetchall()
        volumes = [dict(row) for row in volume_rows]
        protected = [
            item for item in volumes if int(item["canonical_chapter_count"])
        ]
        locked_volume = (
            max(protected, key=lambda item: int(item["position"]))
            if protected
            else None
        )
        if locked_volume:
            locked_position = int(locked_volume["position"])
            allowed_starts = [locked_position, locked_position + 1]
        else:
            allowed_starts = [1]

        future_rows = connection.execute(
            """
            SELECT ch.id, ch.position, ch.title, ch.outline,
                   ch.key_points, ch.status, ch.needs_recheck,
                   ch.skeleton_role, ch.skeleton_arc_titles_json,
                   ch.skeleton_ending_hook,
                   v.position AS volume_position,
                   v.title AS volume_title,
                   cp.status AS task_card_status
            FROM novel_chapters ch
            LEFT JOIN novel_volumes v ON v.id=ch.volume_id
            LEFT JOIN novel_chapter_plans cp ON cp.chapter_id=ch.id
            WHERE ch.project_id=? AND ch.position>?
            ORDER BY ch.position
            LIMIT ?
            """,
            (project_id, current_position, chapter_count),
        ).fetchall()
        future_chapters: List[Dict[str, Any]] = []
        for row in future_rows:
            item = dict(row)
            item["skeleton_arc_titles"] = _load_json(
                item.pop("skeleton_arc_titles_json"), []
            )
            future_chapters.append(item)

        confirmed_arcs = [
            dict(item)
            for item in (
                base.get("confirmed_planned_plot_arcs") or []
            )
        ]
        allowed_plot_arcs = [
            item
            for item in confirmed_arcs
            if str(item.get("lifecycle_status") or "")
            in {"planned", "active"}
        ]
        return {
            **base,
            "context_policy": {
                "plan_source": "author_confirmed_plan_only",
                "canon_source": "author_confirmed_canon_only",
                "drafts_excluded": True,
                "proposal_only_until_author_applies": True,
                "published_canon_is_immutable": True,
            },
            "requested_chapter_count": chapter_count,
            "volumes": volumes,
            "future_chapters": future_chapters,
            "locked_volume": (
                {
                    key: locked_volume[key]
                    for key in ("position", *VOLUME_FIELDS)
                }
                if locked_volume
                else {}
            ),
            "allowed_volume_start_positions": allowed_starts,
            "allowed_plot_arcs": allowed_plot_arcs,
        }

    @staticmethod
    def _ensure_ready(context: Mapping[str, Any]) -> None:
        if not context.get("confirmed_story_blueprint"):
            raise ValueError(
                "请先确认一版全书蓝图，再展开分卷和滚动章节骨架"
            )
        main_arcs = [
            item
            for item in (context.get("allowed_plot_arcs") or [])
            if str(item.get("arc_type") or "") == "main"
        ]
        if not main_arcs:
            raise ValueError(
                "请先确认至少一条处于“计划中”或“推进中”的主线"
            )

    @staticmethod
    def _hard_context(context: Mapping[str, Any]) -> Dict[str, Any]:
        project = dict(context.get("project") or {})
        return {
            "project": {
                key: project.get(key)
                for key in (
                    "project_id",
                    "title",
                    "genre",
                    "premise",
                    "story_promise",
                    "target_audience",
                    "core_appeal",
                    "ending_constraint",
                    "world_setting",
                    "point_of_view",
                )
            },
            "current_canonical_position": int(
                context.get("current_canonical_position") or 0
            ),
            "characters": list(context.get("characters") or []),
            "confirmed_story_blueprint": dict(
                context.get("confirmed_story_blueprint") or {}
            ),
            "allowed_plot_arcs": list(
                context.get("allowed_plot_arcs") or []
            ),
            "canonical_memory": dict(
                context.get("canonical_memory") or {}
            ),
            "locked_volume": dict(context.get("locked_volume") or {}),
            "active_techniques": dict(
                context.get("active_techniques") or {}
            ),
        }

    @classmethod
    def _ensure_application_context(
        cls,
        frozen_context: Mapping[str, Any],
        current_context: Mapping[str, Any],
    ) -> None:
        if _fingerprint(cls._hard_context(frozen_context)) != _fingerprint(
            cls._hard_context(current_context)
        ):
            raise ValueError(
                "生成方案后，确认蓝图、剧情线、正史、人物资料或规划技法"
                "已经变化；请重新生成滚动结构，避免把旧方案覆盖到新基线"
            )

    def create_suggestion(
        self,
        *,
        user_id: int,
        project_id: str,
        chapter_count: int,
        instruction: str,
        provider: str,
        model: str,
        credential_source: str,
    ) -> str:
        if not 10 <= chapter_count <= 30:
            raise ValueError("滚动结构必须规划未来 10–30 章")
        if credential_source not in {"default", "personal"}:
            raise ValueError("不支持的 API 凭据来源")
        clean_instruction = instruction.strip()
        if len(clean_instruction) > 4000:
            raise ValueError("本次结构重点不能超过 4,000 个字符")
        suggestion_id = uuid.uuid4().hex
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            context = self._build_context(
                connection,
                user_id=user_id,
                project_id=project_id,
                chapter_count=chapter_count,
            )
            if context is None:
                connection.rollback()
                raise ValueError("小说项目不存在")
            self._ensure_ready(context)
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
            active_structure = connection.execute(
                """
                SELECT id, project_id, chapter_count, instruction
                FROM story_structure_suggestions
                WHERE user_id=? AND status IN ('queued', 'running')
                ORDER BY created_at LIMIT 1
                """,
                (user_id,),
            ).fetchone()
            if active_structure:
                if (
                    str(active_structure["project_id"]) == project_id
                    and int(active_structure["chapter_count"])
                    == chapter_count
                    and str(active_structure["instruction"])
                    == clean_instruction
                ):
                    connection.rollback()
                    return str(active_structure["id"])
                connection.rollback()
                raise ValueError(
                    "你已有一个滚动结构任务正在排队或运行，请等待其完成"
                )
            active_other = connection.execute(
                """
                SELECT
                  (SELECT id FROM generation_jobs
                   WHERE user_id=? AND status IN ('queued', 'running')
                   ORDER BY created_at LIMIT 1) AS generation_id,
                  (SELECT id FROM analysis_jobs
                   WHERE user_id=? AND status IN ('queued', 'running')
                   ORDER BY created_at LIMIT 1) AS analysis_id,
                  (SELECT id FROM voice_profile_suggestions
                   WHERE user_id=? AND status IN ('queued', 'running')
                   ORDER BY created_at LIMIT 1) AS voice_id,
                  (SELECT id FROM editing_preference_suggestions
                   WHERE user_id=? AND status IN ('queued', 'running')
                   ORDER BY created_at LIMIT 1) AS preference_id,
                  (SELECT id FROM story_plan_suggestions
                   WHERE user_id=? AND status IN ('queued', 'running')
                   ORDER BY created_at LIMIT 1) AS story_plan_id,
                  (SELECT id FROM novel_causal_link_suggestions
                   WHERE user_id=? AND status IN ('queued', 'running')
                   ORDER BY created_at LIMIT 1) AS causal_id,
                  (SELECT id FROM novel_causal_branch_simulations
                   WHERE user_id=? AND status IN ('queued', 'running')
                   ORDER BY created_at LIMIT 1) AS causal_branch_id
                """,
                (
                    user_id,
                    user_id,
                    user_id,
                    user_id,
                    user_id,
                    user_id,
                    user_id,
                ),
            ).fetchone()
            if any(active_other[key] for key in active_other.keys()):
                connection.rollback()
                raise ValueError(
                    "你已有一个 AI 任务正在排队或运行，请等待其完成"
                )
            connection.execute(
                """
                INSERT INTO story_structure_suggestions(
                    id, project_id, user_id, instruction, chapter_count,
                    provider, model, credential_source, status,
                    baseline_fingerprint, context_snapshot_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?)
                """,
                (
                    suggestion_id,
                    project_id,
                    user_id,
                    clean_instruction,
                    chapter_count,
                    provider,
                    model,
                    credential_source,
                    _fingerprint(context),
                    _json(context),
                    now,
                ),
            )
            connection.commit()
        return suggestion_id

    def list_suggestions(
        self, *, user_id: int, project_id: str, limit: int = 8
    ) -> List[Dict[str, Any]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT suggestion.*,
                       (
                           SELECT COUNT(*)
                           FROM story_structure_applications application
                           WHERE application.suggestion_id=suggestion.id
                             AND application.status='applied'
                       ) AS active_application_count
                FROM story_structure_suggestions suggestion
                JOIN novel_projects p ON p.id=suggestion.project_id
                WHERE suggestion.project_id=? AND p.user_id=?
                ORDER BY suggestion.created_at DESC, suggestion.rowid DESC
                LIMIT ?
                """,
                (project_id, user_id, max(1, min(limit, 30))),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_suggestion(
        self, *, user_id: int, suggestion_id: str
    ) -> Optional[Dict[str, Any]]:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT suggestion.*, p.title AS project_title,
                       p.genre, p.premise
                FROM story_structure_suggestions suggestion
                JOIN novel_projects p ON p.id=suggestion.project_id
                WHERE suggestion.id=? AND p.user_id=?
                """,
                (suggestion_id, user_id),
            ).fetchone()
            if not row:
                return None
            applications = connection.execute(
                """
                SELECT * FROM story_structure_applications
                WHERE suggestion_id=?
                ORDER BY created_at DESC, rowid DESC
                """,
                (suggestion_id,),
            ).fetchall()
            current_context = self._build_context(
                connection,
                user_id=user_id,
                project_id=str(row["project_id"]),
                chapter_count=int(row["chapter_count"]),
            )
            item = dict(row)
            item["context_snapshot"] = _load_json(
                item.pop("context_snapshot_json"), {}
            )
            raw_result = item.pop("result_json")
            item["result"] = None
            item["previews"] = []
            item["application_blocker"] = ""
            proposal_set: Optional[StoryStructureProposalSet] = None
            if raw_result:
                try:
                    proposal_set = (
                        StoryStructureProposalSet.model_validate_json(
                            str(raw_result)
                        )
                    )
                    proposal_set.ensure_context_compatible(
                        item["context_snapshot"]
                    )
                    item["result"] = proposal_set.model_dump(mode="json")
                except ValueError:
                    item["result_error"] = (
                        "已保存滚动结构方案损坏或不再符合冻结上下文"
                    )
            if proposal_set and current_context is not None:
                try:
                    self._ensure_ready(current_context)
                    self._ensure_application_context(
                        item["context_snapshot"], current_context
                    )
                    proposal_set.ensure_context_compatible(current_context)
                except ValueError as exc:
                    item["application_blocker"] = str(exc)
                else:
                    for option_index, option in enumerate(
                        proposal_set.options
                    ):
                        preview = self._build_preview_in_connection(
                            connection,
                            suggestion_id=suggestion_id,
                            option_index=option_index,
                            project_id=str(row["project_id"]),
                            current_position=int(
                                current_context[
                                    "current_canonical_position"
                                ]
                            ),
                            option=option,
                        )
                        item["previews"].append(
                            {
                                key: value
                                for key, value in preview.items()
                                if key != "_before_state"
                            }
                        )
        item["applications"] = []
        for raw_application in applications:
            application = dict(raw_application)
            application["summary"] = _load_json(
                application.pop("summary_json"), {}
            )
            application["baseline_changed"] = bool(
                application["baseline_changed"]
            )
            application.pop("before_state_json", None)
            application.pop("after_state_json", None)
            item["applications"].append(application)
        item["baseline_changed"] = bool(
            current_context is not None
            and _fingerprint(current_context)
            != str(item["baseline_fingerprint"])
        )
        return item

    def get_status(
        self, *, user_id: int, suggestion_id: str
    ) -> Optional[str]:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT suggestion.status
                FROM story_structure_suggestions suggestion
                JOIN novel_projects p ON p.id=suggestion.project_id
                WHERE suggestion.id=? AND p.user_id=?
                """,
                (suggestion_id, user_id),
            ).fetchone()
        return str(row["status"]) if row else None

    def claim_next_suggestion(self) -> Optional[Dict[str, Any]]:
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = utc_now()
            connection.execute(
                """
                UPDATE story_structure_suggestions
                SET status='queued', started_at=NULL, claim_token=NULL,
                    lease_expires_at=NULL,
                    error='上一次滚动结构规划租约已过期，已自动重新排队'
                WHERE status='running' AND lease_expires_at IS NOT NULL
                    AND lease_expires_at<=?
                """,
                (now,),
            )
            row = connection.execute(
                """
                SELECT suggestion.*
                FROM story_structure_suggestions suggestion
                JOIN novel_projects p ON p.id=suggestion.project_id
                WHERE suggestion.status='queued'
                ORDER BY suggestion.created_at
                LIMIT 1
                """
            ).fetchone()
            if not row:
                connection.commit()
                return None
            claim_token = uuid.uuid4().hex
            cursor = connection.execute(
                """
                UPDATE story_structure_suggestions
                SET status='running', started_at=?, error=NULL,
                    claim_token=?, lease_expires_at=?
                WHERE id=? AND status='queued'
                """,
                (
                    now,
                    claim_token,
                    utc_after(2 * 60 * 60),
                    row["id"],
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return None
            connection.commit()
        result = dict(row)
        result["claim_token"] = claim_token
        result["context_snapshot"] = _load_json(
            result.pop("context_snapshot_json"), {}
        )
        return result

    def complete_suggestion(
        self,
        *,
        suggestion_id: str,
        claim_token: str,
        result: StoryStructureProposalSet,
        raw_response: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> bool:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT context_snapshot_json
                FROM story_structure_suggestions
                WHERE id=? AND status='running' AND claim_token=?
                """,
                (suggestion_id, claim_token),
            ).fetchone()
            if not row:
                return False
            context = _load_json(row["context_snapshot_json"], {})
            result.ensure_context_compatible(context)
            cursor = connection.execute(
                """
                UPDATE story_structure_suggestions
                SET status='completed', result_json=?, raw_response=?,
                    provider=?, model=?, input_tokens=?, output_tokens=?,
                    finished_at=?, claim_token=NULL, lease_expires_at=NULL,
                    error=NULL
                WHERE id=? AND status='running' AND claim_token=?
                """,
                (
                    result.model_dump_json(),
                    raw_response,
                    provider,
                    model,
                    input_tokens,
                    output_tokens,
                    utc_now(),
                    suggestion_id,
                    claim_token,
                ),
            )
            connection.commit()
        return cursor.rowcount == 1

    def fail_suggestion(
        self,
        *,
        suggestion_id: str,
        claim_token: str,
        error: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> bool:
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE story_structure_suggestions
                SET status='failed', error=?, input_tokens=?,
                    output_tokens=?, finished_at=?, claim_token=NULL,
                    lease_expires_at=NULL
                WHERE id=? AND status='running' AND claim_token=?
                """,
                (
                    error[:2000],
                    input_tokens,
                    output_tokens,
                    utc_now(),
                    suggestion_id,
                    claim_token,
                ),
            )
            connection.commit()
        return cursor.rowcount == 1

    def release_claim(
        self, suggestion_id: str, claim_token: str, error: str
    ) -> bool:
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE story_structure_suggestions
                SET status='queued', started_at=NULL, claim_token=NULL,
                    lease_expires_at=NULL, error=?
                WHERE id=? AND status='running' AND claim_token=?
                """,
                (error[:2000], suggestion_id, claim_token),
            )
            connection.commit()
        return cursor.rowcount == 1

    @staticmethod
    def _capture_state(
        connection,
        *,
        project_id: str,
        volume_positions: List[int],
        chapter_positions: List[int],
    ) -> Dict[str, Any]:
        volumes: Dict[str, Any] = {
            str(position): None for position in volume_positions
        }
        if volume_positions:
            placeholders = ",".join("?" for _ in volume_positions)
            rows = connection.execute(
                f"""
                SELECT v.*,
                       (
                           SELECT COUNT(*) FROM novel_chapters ch
                           WHERE ch.volume_id=v.id
                       ) AS chapter_count,
                       (
                           SELECT COUNT(*) FROM novel_chapters ch
                           WHERE ch.volume_id=v.id
                             AND ch.head_version_id IS NOT NULL
                       ) AS canonical_chapter_count
                FROM novel_volumes v
                WHERE v.project_id=? AND v.position IN ({placeholders})
                ORDER BY v.position
                """,
                (project_id, *volume_positions),
            ).fetchall()
            for row in rows:
                volumes[str(row["position"])] = dict(row)

        chapters: Dict[str, Any] = {
            str(position): None for position in chapter_positions
        }
        if chapter_positions:
            placeholders = ",".join("?" for _ in chapter_positions)
            rows = connection.execute(
                f"""
                SELECT ch.*, v.position AS volume_position,
                       (
                           SELECT COUNT(*) FROM novel_chapter_versions cv
                           WHERE cv.chapter_id=ch.id
                       ) AS version_count,
                       (
                           SELECT COUNT(*)
                           FROM novel_chapter_edit_buffers buffer
                           WHERE buffer.chapter_id=ch.id
                       ) AS edit_buffer_count,
                       (
                           SELECT COUNT(*) FROM generation_jobs job
                           WHERE job.chapter_id=ch.id
                       ) AS generation_count,
                       (
                           SELECT COUNT(*) FROM generation_jobs job
                           WHERE job.chapter_id=ch.id
                             AND job.status IN ('queued', 'running')
                       ) AS active_job_count,
                       (
                           SELECT COUNT(*)
                           FROM reader_plan_applications application
                           WHERE application.chapter_id=ch.id
                       ) AS reader_application_count,
                       (
                           SELECT COUNT(*)
                           FROM novel_technique_bindings binding
                           WHERE binding.chapter_id=ch.id
                       ) AS technique_binding_count
                FROM novel_chapters ch
                LEFT JOIN novel_volumes v ON v.id=ch.volume_id
                WHERE ch.project_id=? AND ch.position IN ({placeholders})
                ORDER BY ch.position
                """,
                (project_id, *chapter_positions),
            ).fetchall()
            for row in rows:
                chapter = dict(row)
                plan = connection.execute(
                    """
                    SELECT * FROM novel_chapter_plans
                    WHERE chapter_id=?
                    """,
                    (chapter["id"],),
                ).fetchone()
                plan_item = dict(plan) if plan else None
                scenes: List[Dict[str, Any]] = []
                if plan:
                    scenes = [
                        dict(item)
                        for item in connection.execute(
                            """
                            SELECT * FROM novel_scene_beats
                            WHERE plan_id=?
                            ORDER BY position
                            """,
                            (plan["id"],),
                        ).fetchall()
                    ]
                chapter["task_card"] = plan_item
                chapter["scenes"] = scenes
                chapters[str(chapter["position"])] = chapter
        return {
            "volume_positions": volume_positions,
            "chapter_positions": chapter_positions,
            "volumes": volumes,
            "chapters": chapters,
        }

    @classmethod
    def _build_preview_in_connection(
        cls,
        connection,
        *,
        suggestion_id: str,
        option_index: int,
        project_id: str,
        current_position: int,
        option: StoryStructureOption,
    ) -> Dict[str, Any]:
        volume_positions = [volume.position for volume in option.volumes]
        chapter_positions = [chapter.position for chapter in option.chapters]
        before_state = cls._capture_state(
            connection,
            project_id=project_id,
            volume_positions=volume_positions,
            chapter_positions=chapter_positions,
        )
        volume_summary: Dict[str, List[Dict[str, Any]]] = {
            "create": [],
            "update": [],
            "preserve": [],
        }
        conflicts: List[str] = []
        for proposed in option.volumes:
            existing = before_state["volumes"][str(proposed.position)]
            desired = {
                field: getattr(proposed, field) for field in VOLUME_FIELDS
            }
            if existing is None:
                volume_summary["create"].append(
                    {"position": proposed.position, "title": proposed.title}
                )
                continue
            changed_fields = [
                field
                for field in VOLUME_FIELDS
                if str(existing[field] or "") != str(desired[field] or "")
            ]
            if int(existing["canonical_chapter_count"] or 0):
                if changed_fields:
                    conflicts.append(
                        f"第 {proposed.position} 卷已经承载正史，不能改写卷资料"
                    )
                volume_summary["preserve"].append(
                    {
                        "position": proposed.position,
                        "title": str(existing["title"]),
                        "reason": "已承载正史，原样保留",
                    }
                )
            elif changed_fields:
                volume_summary["update"].append(
                    {
                        "position": proposed.position,
                        "before_title": str(existing["title"]),
                        "after_title": proposed.title,
                        "changed_fields": changed_fields,
                    }
                )
            else:
                volume_summary["preserve"].append(
                    {
                        "position": proposed.position,
                        "title": proposed.title,
                        "reason": "资料相同",
                    }
                )

        chapter_summary: Dict[str, List[Dict[str, Any]]] = {
            "create": [],
            "update": [],
            "preserve": [],
        }
        task_card_summary: Dict[str, List[int]] = {
            "create": [],
            "reset_to_draft": [],
            "preserve": [],
        }
        for proposed in option.chapters:
            existing = before_state["chapters"][str(proposed.position)]
            desired = _chapter_values(proposed)
            if existing is None:
                chapter_summary["create"].append(
                    {
                        "position": proposed.position,
                        "title": proposed.title,
                        "volume_position": proposed.volume_position,
                    }
                )
                task_card_summary["create"].append(proposed.position)
                continue
            if existing["head_version_id"]:
                conflicts.append(
                    f"第 {proposed.position} 章已经成为正史，不能应用骨架"
                )
                continue
            if int(existing["active_job_count"] or 0):
                conflicts.append(
                    f"第 {proposed.position} 章有 AI 任务正在运行"
                )
            changed_fields = []
            for field in (
                "title",
                "outline",
                "key_points",
                "skeleton_role",
                "skeleton_arc_titles_json",
                "skeleton_ending_hook",
            ):
                if str(existing[field] or "") != str(
                    desired[field] or ""
                ):
                    changed_fields.append(field)
            if int(existing["volume_position"] or 0) != int(
                desired["volume_position"]
            ):
                changed_fields.append("volume_position")
            if changed_fields:
                chapter_summary["update"].append(
                    {
                        "position": proposed.position,
                        "before_title": str(existing["title"]),
                        "after_title": proposed.title,
                        "before_volume_position": existing[
                            "volume_position"
                        ],
                        "after_volume_position": proposed.volume_position,
                        "changed_fields": changed_fields,
                    }
                )
                if existing["task_card"]:
                    task_card_summary["reset_to_draft"].append(
                        proposed.position
                    )
                else:
                    task_card_summary["create"].append(proposed.position)
            else:
                chapter_summary["preserve"].append(
                    {
                        "position": proposed.position,
                        "title": proposed.title,
                    }
                )
                task_card_summary["preserve"].append(proposed.position)

        extra_chapters = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM novel_chapters
            WHERE project_id=? AND position>? AND position>?
            """,
            (
                project_id,
                current_position,
                max(chapter_positions),
            ),
        ).fetchone()
        proposed_volume_set = set(volume_positions)
        existing_extra_volumes = connection.execute(
            """
            SELECT position FROM novel_volumes
            WHERE project_id=?
            ORDER BY position
            """,
            (project_id,),
        ).fetchall()
        extra_volume_count = sum(
            int(row["position"]) not in proposed_volume_set
            for row in existing_extra_volumes
        )
        fingerprint_payload = {
            "suggestion_id": suggestion_id,
            "option_index": option_index,
            "option": option.model_dump(mode="json"),
            "before_state": before_state,
        }
        return {
            "option_index": option_index,
            "fingerprint": _fingerprint(fingerprint_payload),
            "volumes": volume_summary,
            "chapters": chapter_summary,
            "task_cards": task_card_summary,
            "preserved_extra_volume_count": extra_volume_count,
            "preserved_extra_chapter_count": int(
                extra_chapters["count"] or 0
            ),
            "conflicts": conflicts,
            "_before_state": before_state,
        }

    def apply_suggestion(
        self,
        *,
        user_id: int,
        suggestion_id: str,
        option_index: int,
        preview_fingerprint: str,
    ) -> Dict[str, Any]:
        now = utc_now()
        application_id = uuid.uuid4().hex
        created_directories: List[Path] = []
        with self.database.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """
                    SELECT suggestion.*
                    FROM story_structure_suggestions suggestion
                    JOIN novel_projects p ON p.id=suggestion.project_id
                    WHERE suggestion.id=? AND p.user_id=?
                      AND suggestion.status='completed'
                    """,
                    (suggestion_id, user_id),
                ).fetchone()
                if not row:
                    connection.rollback()
                    raise ValueError("滚动结构方案不存在或尚未生成完成")
                try:
                    proposal_set = (
                        StoryStructureProposalSet.model_validate_json(
                            str(row["result_json"] or "")
                        )
                    )
                except ValueError as exc:
                    raise ValueError(
                        "已保存的滚动结构方案损坏，请重新生成"
                    ) from exc
                if option_index < 0 or option_index >= len(
                    proposal_set.options
                ):
                    raise ValueError("所选滚动结构方案不存在")
                current_context = self._build_context(
                    connection,
                    user_id=user_id,
                    project_id=str(row["project_id"]),
                    chapter_count=int(row["chapter_count"]),
                )
                if current_context is None:
                    raise ValueError("小说项目不存在")
                self._ensure_ready(current_context)
                self._ensure_application_context(
                    _load_json(row["context_snapshot_json"], {}),
                    current_context,
                )
                proposal_set.ensure_context_compatible(current_context)
                option = proposal_set.options[option_index]
                preview = self._build_preview_in_connection(
                    connection,
                    suggestion_id=suggestion_id,
                    option_index=option_index,
                    project_id=str(row["project_id"]),
                    current_position=int(
                        current_context["current_canonical_position"]
                    ),
                    option=option,
                )
                if preview["conflicts"]:
                    raise ValueError("；".join(preview["conflicts"]))
                if (
                    not preview_fingerprint
                    or preview_fingerprint != preview["fingerprint"]
                ):
                    raise ValueError(
                        "未来结构在你查看后发生了变化，请刷新页面重新核对差异"
                    )
                baseline_changed = (
                    _fingerprint(current_context)
                    != str(row["baseline_fingerprint"])
                )
                before_state = preview["_before_state"]
                volume_ids: Dict[int, str] = {}
                proposed_volume_by_position = {
                    volume.position: volume for volume in option.volumes
                }
                for position, proposed in proposed_volume_by_position.items():
                    existing = before_state["volumes"][str(position)]
                    if existing is None:
                        volume_id = uuid.uuid4().hex
                        connection.execute(
                            """
                            INSERT INTO novel_volumes(
                                id, project_id, position, title, goal,
                                start_state, end_state, major_conflict,
                                payoff, status, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,
                                      'planning', ?, ?)
                            """,
                            (
                                volume_id,
                                row["project_id"],
                                position,
                                proposed.title,
                                proposed.goal,
                                proposed.start_state,
                                proposed.end_state,
                                proposed.major_conflict,
                                proposed.payoff,
                                now,
                                now,
                            ),
                        )
                    else:
                        volume_id = str(existing["id"])
                        if not int(existing["canonical_chapter_count"] or 0):
                            connection.execute(
                                """
                                UPDATE novel_volumes
                                SET title=?, goal=?, start_state=?,
                                    end_state=?, major_conflict=?, payoff=?,
                                    updated_at=?
                                WHERE id=? AND project_id=?
                                """,
                                (
                                    proposed.title,
                                    proposed.goal,
                                    proposed.start_state,
                                    proposed.end_state,
                                    proposed.major_conflict,
                                    proposed.payoff,
                                    now,
                                    volume_id,
                                    row["project_id"],
                                ),
                            )
                    volume_ids[position] = volume_id

                project_target_chars = int(
                    current_context["project"]["target_chapter_chars"]
                )
                chapter_by_position = {
                    chapter.position: chapter for chapter in option.chapters
                }
                for position, proposed in chapter_by_position.items():
                    existing = before_state["chapters"][str(position)]
                    desired = _chapter_values(proposed)
                    volume_id = volume_ids[proposed.volume_position]
                    chapter_changed = existing is None
                    if existing is not None:
                        chapter_changed = any(
                            (
                                str(existing["title"] or "")
                                != desired["title"],
                                str(existing["outline"] or "")
                                != desired["outline"],
                                str(existing["key_points"] or "")
                                != desired["key_points"],
                                int(existing["volume_position"] or 0)
                                != proposed.volume_position,
                                str(existing["skeleton_role"] or "")
                                != desired["skeleton_role"],
                                str(
                                    existing[
                                        "skeleton_arc_titles_json"
                                    ]
                                    or ""
                                )
                                != desired["skeleton_arc_titles_json"],
                                str(
                                    existing["skeleton_ending_hook"] or ""
                                )
                                != desired["skeleton_ending_hook"],
                            )
                        )
                    if existing is None:
                        chapter_id = uuid.uuid4().hex
                        chapter_dir = (
                            self.novels_dir
                            / str(user_id)
                            / str(row["project_id"])
                            / "chapters"
                            / chapter_id
                        )
                        (chapter_dir / "versions").mkdir(
                            parents=True, exist_ok=False, mode=0o700
                        )
                        os.chmod(chapter_dir, 0o700)
                        os.chmod(chapter_dir / "versions", 0o700)
                        content_path = chapter_dir / "content.txt"
                        content_path.write_text("", encoding="utf-8")
                        content_path.chmod(0o600)
                        created_directories.append(chapter_dir)
                        connection.execute(
                            """
                            INSERT INTO novel_chapters(
                                id, project_id, position, title, outline,
                                key_points, status, content_path, char_count,
                                volume_id, needs_recheck, skeleton_role,
                                skeleton_arc_titles_json,
                                skeleton_ending_hook,
                                skeleton_application_id,
                                created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, 'planned', ?, 0, ?,
                                      0, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                chapter_id,
                                row["project_id"],
                                position,
                                proposed.title,
                                proposed.purpose,
                                "\n".join(proposed.key_points),
                                str(content_path),
                                volume_id,
                                proposed.structural_role,
                                _json(proposed.arc_titles),
                                proposed.ending_hook,
                                application_id,
                                now,
                                now,
                            ),
                        )
                    else:
                        chapter_id = str(existing["id"])
                        if chapter_changed:
                            connection.execute(
                                """
                                UPDATE novel_chapters
                                SET title=?, outline=?, key_points=?,
                                    volume_id=?, skeleton_role=?,
                                    skeleton_arc_titles_json=?,
                                    skeleton_ending_hook=?,
                                    skeleton_application_id=?,
                                    needs_recheck=CASE
                                        WHEN EXISTS(
                                            SELECT 1
                                            FROM novel_chapter_plans cp
                                            WHERE cp.chapter_id=novel_chapters.id
                                        ) THEN 1
                                        ELSE needs_recheck
                                    END,
                                    updated_at=?
                                WHERE id=? AND project_id=?
                                  AND head_version_id IS NULL
                                """,
                                (
                                    proposed.title,
                                    proposed.purpose,
                                    "\n".join(proposed.key_points),
                                    volume_id,
                                    proposed.structural_role,
                                    _json(proposed.arc_titles),
                                    proposed.ending_hook,
                                    application_id,
                                    now,
                                    chapter_id,
                                    row["project_id"],
                                ),
                            )
                    if not chapter_changed:
                        continue
                    existing_plan = (
                        existing["task_card"]
                        if existing is not None
                        else None
                    )
                    if existing_plan:
                        connection.execute(
                            """
                            UPDATE novel_chapter_plans
                            SET purpose=?, plot_threads_json=?,
                                must_happen_json=?, ending_hook=?,
                                target_chars=?, status='draft',
                                source='structure_planner',
                                confirmed_at=NULL, updated_at=?
                            WHERE id=? AND chapter_id=?
                            """,
                            (
                                proposed.purpose,
                                _json(proposed.arc_titles),
                                _json(proposed.key_points),
                                proposed.ending_hook,
                                project_target_chars,
                                now,
                                existing_plan["id"],
                                chapter_id,
                            ),
                        )
                        connection.execute(
                            """
                            UPDATE novel_scene_beats
                            SET draft_status=CASE
                                    WHEN current_version_id IS NOT NULL
                                    THEN 'stale'
                                    ELSE draft_status
                                END,
                                updated_at=?
                            WHERE plan_id=? AND beat_status='active'
                            """,
                            (now, existing_plan["id"]),
                        )
                    else:
                        connection.execute(
                            """
                            INSERT INTO novel_chapter_plans(
                                id, project_id, chapter_id, purpose,
                                plot_threads_json, must_happen_json,
                                ending_hook, target_chars, status, source,
                                created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'draft',
                                      'structure_planner', ?, ?)
                            """,
                            (
                                uuid.uuid4().hex,
                                row["project_id"],
                                chapter_id,
                                proposed.purpose,
                                _json(proposed.arc_titles),
                                _json(proposed.key_points),
                                proposed.ending_hook,
                                project_target_chars,
                                now,
                                now,
                            ),
                        )
                connection.execute(
                    "UPDATE novel_projects SET updated_at=? WHERE id=?",
                    (now, row["project_id"]),
                )
                after_state = self._capture_state(
                    connection,
                    project_id=str(row["project_id"]),
                    volume_positions=before_state["volume_positions"],
                    chapter_positions=before_state["chapter_positions"],
                )
                after_fingerprint = _fingerprint(after_state)
                summary = {
                    key: value
                    for key, value in preview.items()
                    if key not in {"_before_state", "fingerprint"}
                }
                connection.execute(
                    """
                    INSERT INTO story_structure_applications(
                        id, suggestion_id, project_id, option_index,
                        preview_fingerprint, before_state_json,
                        after_state_json, after_fingerprint, summary_json,
                        status, baseline_changed, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'applied', ?, ?)
                    """,
                    (
                        application_id,
                        suggestion_id,
                        row["project_id"],
                        option_index,
                        preview["fingerprint"],
                        _json(before_state),
                        _json(after_state),
                        after_fingerprint,
                        _json(summary),
                        int(baseline_changed),
                        now,
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                for directory in reversed(created_directories):
                    shutil.rmtree(directory, ignore_errors=True)
                raise
        return {
            "application_id": application_id,
            "project_id": str(row["project_id"]),
            "option_index": option_index,
            "option_label": option.label,
            "baseline_changed": baseline_changed,
            "summary": summary,
        }

    def revert_application(
        self, *, user_id: int, application_id: str
    ) -> Dict[str, Any]:
        now = utc_now()
        moved_directories: List[tuple[Path, Path]] = []
        recovery_root: Optional[Path] = None
        with self.database.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """
                    SELECT application.*, suggestion.user_id
                    FROM story_structure_applications application
                    JOIN story_structure_suggestions suggestion
                      ON suggestion.id=application.suggestion_id
                    JOIN novel_projects p
                      ON p.id=application.project_id
                    WHERE application.id=? AND p.user_id=?
                    """,
                    (application_id, user_id),
                ).fetchone()
                if not row:
                    raise ValueError("滚动结构采纳记录不存在")
                if str(row["status"]) != "applied":
                    raise ValueError("这次结构采纳已经撤销")
                before_state = _load_json(
                    row["before_state_json"], {}
                )
                after_state = _load_json(row["after_state_json"], {})
                current_state = self._capture_state(
                    connection,
                    project_id=str(row["project_id"]),
                    volume_positions=[
                        int(item)
                        for item in after_state.get(
                            "volume_positions", []
                        )
                    ],
                    chapter_positions=[
                        int(item)
                        for item in after_state.get(
                            "chapter_positions", []
                        )
                    ],
                )
                if (
                    _fingerprint(current_state)
                    != str(row["after_fingerprint"])
                ):
                    raise ValueError(
                        "采纳后这些分卷、章节或任务卡已经继续修改，"
                        "为避免覆盖新工作，系统不能自动撤销"
                    )

                created_chapter_ids: List[str] = []
                for position in after_state["chapter_positions"]:
                    key = str(position)
                    before_chapter = before_state["chapters"].get(key)
                    after_chapter = after_state["chapters"].get(key)
                    if before_chapter is not None or after_chapter is None:
                        continue
                    if (
                        int(after_chapter.get("version_count") or 0)
                        or int(after_chapter.get("edit_buffer_count") or 0)
                        or int(after_chapter.get("generation_count") or 0)
                        or int(
                            after_chapter.get("reader_application_count")
                            or 0
                        )
                        or int(
                            after_chapter.get("technique_binding_count")
                            or 0
                        )
                        or after_chapter.get("head_version_id")
                    ):
                        raise ValueError(
                            f"第 {position} 章已经产生正文、任务或绑定，不能撤销"
                        )
                    content_path = Path(
                        str(after_chapter["content_path"])
                    )
                    try:
                        content = content_path.read_text(encoding="utf-8")
                    except FileNotFoundError as exc:
                        raise ValueError(
                            f"第 {position} 章的本地文件已变化，不能安全撤销"
                        ) from exc
                    if content:
                        raise ValueError(
                            f"第 {position} 章已经写入正文，不能撤销"
                        )
                    created_chapter_ids.append(
                        str(after_chapter["id"])
                    )

                if created_chapter_ids:
                    recovery_root = (
                        self.novels_dir
                        / str(user_id)
                        / str(row["project_id"])
                        / ".structure-recovery"
                        / application_id
                    )
                    if recovery_root.exists():
                        raise ValueError(
                            "本次撤销的恢复目录已存在，请先人工核对"
                        )
                    recovery_root.mkdir(
                        parents=True, exist_ok=False, mode=0o700
                    )
                    for chapter_id in created_chapter_ids:
                        source = (
                            self.novels_dir
                            / str(user_id)
                            / str(row["project_id"])
                            / "chapters"
                            / chapter_id
                        )
                        destination = recovery_root / chapter_id
                        if not source.exists():
                            raise ValueError(
                                "待撤销章节目录不存在，不能安全继续"
                            )
                        shutil.move(str(source), str(destination))
                        moved_directories.append((source, destination))

                for position in before_state["chapter_positions"]:
                    key = str(position)
                    before_chapter = before_state["chapters"].get(key)
                    after_chapter = after_state["chapters"].get(key)
                    if after_chapter is None:
                        continue
                    if before_chapter is None:
                        connection.execute(
                            "DELETE FROM novel_chapters WHERE id=?",
                            (after_chapter["id"],),
                        )
                        continue
                    before_plan = before_chapter.get("task_card")
                    after_plan = after_chapter.get("task_card")
                    if before_plan is None and after_plan is not None:
                        connection.execute(
                            "DELETE FROM novel_chapter_plans WHERE id=?",
                            (after_plan["id"],),
                        )
                    elif before_plan is not None:
                        assignments = ", ".join(
                            f"{field}=?" for field in PLAN_RESTORE_FIELDS
                        )
                        connection.execute(
                            f"""
                            UPDATE novel_chapter_plans
                            SET {assignments}
                            WHERE id=? AND chapter_id=?
                            """,
                            (
                                *[
                                    before_plan[field]
                                    for field in PLAN_RESTORE_FIELDS
                                ],
                                before_plan["id"],
                                before_chapter["id"],
                            ),
                        )
                        before_scenes = {
                            str(scene["id"]): scene
                            for scene in before_chapter.get("scenes") or []
                        }
                        for scene_id, scene in before_scenes.items():
                            connection.execute(
                                """
                                UPDATE novel_scene_beats
                                SET draft_status=?, updated_at=?
                                WHERE id=? AND plan_id=?
                                """,
                                (
                                    scene["draft_status"],
                                    scene["updated_at"],
                                    scene_id,
                                    before_plan["id"],
                                ),
                            )
                    chapter_assignments = ", ".join(
                        f"{field}=?" for field in CHAPTER_RESTORE_FIELDS
                    )
                    connection.execute(
                        f"""
                        UPDATE novel_chapters
                        SET {chapter_assignments}
                        WHERE id=? AND project_id=?
                        """,
                        (
                            *[
                                before_chapter[field]
                                for field in CHAPTER_RESTORE_FIELDS
                            ],
                            before_chapter["id"],
                            row["project_id"],
                        ),
                    )

                for position in before_state["volume_positions"]:
                    key = str(position)
                    before_volume = before_state["volumes"].get(key)
                    after_volume = after_state["volumes"].get(key)
                    if after_volume is None:
                        continue
                    if before_volume is None:
                        remaining = connection.execute(
                            """
                            SELECT COUNT(*) AS count
                            FROM novel_chapters WHERE volume_id=?
                            """,
                            (after_volume["id"],),
                        ).fetchone()
                        if int(remaining["count"] or 0):
                            raise ValueError(
                                f"第 {position} 卷已有其他章节，不能撤销"
                            )
                        connection.execute(
                            "DELETE FROM novel_volumes WHERE id=?",
                            (after_volume["id"],),
                        )
                    else:
                        connection.execute(
                            """
                            UPDATE novel_volumes
                            SET title=?, goal=?, start_state=?, end_state=?,
                                major_conflict=?, payoff=?, status=?,
                                created_at=?, updated_at=?
                            WHERE id=? AND project_id=?
                            """,
                            (
                                before_volume["title"],
                                before_volume["goal"],
                                before_volume["start_state"],
                                before_volume["end_state"],
                                before_volume["major_conflict"],
                                before_volume["payoff"],
                                before_volume["status"],
                                before_volume["created_at"],
                                before_volume["updated_at"],
                                before_volume["id"],
                                row["project_id"],
                            ),
                        )
                connection.execute(
                    """
                    UPDATE story_structure_applications
                    SET status='reverted', reverted_at=?, recovery_path=?
                    WHERE id=? AND status='applied'
                    """,
                    (
                        now,
                        str(recovery_root) if recovery_root else "",
                        application_id,
                    ),
                )
                connection.execute(
                    "UPDATE novel_projects SET updated_at=? WHERE id=?",
                    (now, row["project_id"]),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                for source, destination in reversed(moved_directories):
                    if destination.exists() and not source.exists():
                        source.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(destination), str(source))
                if (
                    recovery_root
                    and recovery_root.exists()
                    and not any(recovery_root.iterdir())
                ):
                    recovery_root.rmdir()
                raise
        return {
            "application_id": application_id,
            "suggestion_id": str(row["suggestion_id"]),
            "project_id": str(row["project_id"]),
            "recovery_path": str(recovery_root) if recovery_root else "",
        }
