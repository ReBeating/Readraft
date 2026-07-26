from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Dict, List, Mapping, Optional

from .context_compiler import compile_active_techniques
from .db import Database, utc_after, utc_now
from .story_planner_schema import (
    StoryPlanProposalSet,
    StoryPlanningMode,
)
from .story_planning_service import StoryPlanningService


ALLOWED_PLANNING_MODES = {"create", "refine", "rethink"}


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _load_json(value: Any, fallback: Any) -> Any:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return fallback
    return parsed


def _fingerprint(context: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json(context).encode("utf-8")).hexdigest()


class StoryPlanSuggestionService:
    def __init__(self, database: Database):
        self.database = database
        self.story_planning_service = StoryPlanningService(database)

    @staticmethod
    def _build_context(
        connection,
        *,
        user_id: int,
        project_id: str,
    ) -> Optional[Dict[str, Any]]:
        project = connection.execute(
            """
            SELECT id AS project_id, title, genre, premise, story_promise,
                   target_audience, core_appeal, ending_constraint,
                   world_setting, style_guide, point_of_view,
                   planning_horizon, target_chapter_chars,
                   canonical_branch_id
            FROM novel_projects
            WHERE id=? AND user_id=?
            """,
            (project_id, user_id),
        ).fetchone()
        if not project:
            return None
        branch_id = str(project["canonical_branch_id"] or "main")
        current_row = connection.execute(
            """
            SELECT COALESCE(MAX(position), 0) AS position
            FROM novel_chapters
            WHERE project_id=? AND canonical_version_id IS NOT NULL
            """,
            (project_id,),
        ).fetchone()
        current_position = int(current_row["position"])
        horizon = int(project["planning_horizon"] or 20)
        characters = connection.execute(
            """
            SELECT name, role, traits, background, character_arc
            FROM novel_characters
            WHERE project_id=? ORDER BY position
            """,
            (project_id,),
        ).fetchall()
        volumes = connection.execute(
            """
            SELECT position, title, goal, start_state, end_state,
                   major_conflict, payoff, status
            FROM novel_volumes
            WHERE project_id=? ORDER BY position
            """,
            (project_id,),
        ).fetchall()
        future_chapters = connection.execute(
            """
            SELECT ch.position, ch.title, ch.outline, ch.key_points,
                   v.title AS volume_title,
                   cp.status AS task_card_status
            FROM novel_chapters ch
            LEFT JOIN novel_volumes v ON v.id=ch.volume_id
            LEFT JOIN novel_chapter_plans cp ON cp.chapter_id=ch.id
            WHERE ch.project_id=? AND ch.position>?
            ORDER BY ch.position
            LIMIT ?
            """,
            (project_id, current_position, horizon),
        ).fetchall()
        blueprint = connection.execute(
            """
            SELECT v.id AS version_id, v.revision, v.central_question,
                   v.protagonist_goal, v.core_conflict, v.stakes,
                   v.opening_state, v.ending_state, v.major_turns_json,
                   v.must_payoffs_json, v.forbidden_shortcuts_json,
                   v.author_notes
            FROM novel_story_blueprint_heads h
            JOIN novel_story_blueprint_versions v
                ON v.id=h.confirmed_version_id
            WHERE h.project_id=? AND v.version_status='confirmed'
            """,
            (project_id,),
        ).fetchone()
        arc_rows = connection.execute(
            """
            SELECT a.id, a.position, v.id AS version_id, v.arc_type,
                   v.title, v.dramatic_question, v.promise,
                   v.start_state, v.target_payoff,
                   v.involved_characters_json, v.planned_turns_json,
                   v.lifecycle_status, v.priority, v.author_notes
            FROM novel_plot_arcs a
            JOIN novel_plot_arc_versions v
                ON v.id=a.confirmed_version_id
            WHERE a.project_id=? AND v.version_status='confirmed'
                AND v.lifecycle_status!='abandoned'
            ORDER BY v.priority DESC, a.position
            """,
            (project_id,),
        ).fetchall()
        recent_memory = connection.execute(
            """
            SELECT ch.position, ch.title, m.summary,
                   m.key_events_json, m.unresolved_questions_json,
                   m.keywords_json
            FROM chapter_memory m
            JOIN novel_chapters ch ON ch.id=m.chapter_id
            WHERE m.project_id=? AND m.branch_id=?
                AND m.record_status='canon'
            ORDER BY ch.position DESC LIMIT 8
            """,
            (project_id, branch_id),
        ).fetchall()
        facts = connection.execute(
            """
            SELECT ch.position AS source_chapter_position,
                   f.fact_type, f.subject_name, f.predicate,
                   f.object_json
            FROM story_facts f
            JOIN novel_chapters ch ON ch.id=f.chapter_id
            WHERE f.project_id=? AND f.branch_id=?
                AND f.fact_status='canon'
            ORDER BY ch.position DESC, f.created_at DESC LIMIT 100
            """,
            (project_id, branch_id),
        ).fetchall()
        knowledge = connection.execute(
            """
            SELECT ch.position AS source_chapter_position,
                   k.character_name, k.fact_text, k.knowledge_state,
                   k.learned_via
            FROM character_knowledge k
            JOIN novel_chapters ch ON ch.id=k.chapter_id
            WHERE k.project_id=? AND k.branch_id=?
                AND k.record_status='canon'
            ORDER BY ch.position DESC, k.created_at DESC LIMIT 100
            """,
            (project_id, branch_id),
        ).fetchall()
        threads = connection.execute(
            """
            SELECT ch.position AS source_chapter_position,
                   t.thread_name, t.thread_type, t.action,
                   t.update_text, t.promise, t.target_payoff
            FROM plot_threads t
            JOIN novel_chapters ch ON ch.id=t.chapter_id
            WHERE t.project_id=? AND t.branch_id=?
                AND t.record_status='canon'
            ORDER BY ch.position DESC, t.created_at DESC LIMIT 60
            """,
            (project_id, branch_id),
        ).fetchall()
        hooks = connection.execute(
            """
            SELECT ch.position AS source_chapter_position,
                   f.hook_name, f.action, f.description,
                   f.intended_payoff
            FROM foreshadowing f
            JOIN novel_chapters ch ON ch.id=f.chapter_id
            WHERE f.project_id=? AND f.branch_id=?
                AND f.record_status='canon'
            ORDER BY ch.position DESC, f.created_at DESC LIMIT 60
            """,
            (project_id, branch_id),
        ).fetchall()
        technique_rows = connection.execute(
            """
            SELECT b.technique_id AS id, tc.name, tc.dimension,
                   tc.execution_rule, tc.effect,
                   tc.originality_boundary, b.author_adaptation,
                   b.scope_type, b.priority, b.usage_modes_json
            FROM novel_technique_bindings b
            JOIN reference_technique_cards tc ON tc.id=b.technique_id
            WHERE b.project_id=? AND b.status='enabled'
                AND tc.status='active' AND b.scope_type='project'
            ORDER BY b.priority DESC, b.created_at
            """,
            (project_id,),
        ).fetchall()

        blueprint_item: Dict[str, Any] = {}
        if blueprint:
            blueprint_item = dict(blueprint)
            blueprint_item["major_turns"] = _load_json(
                blueprint_item.pop("major_turns_json"), []
            )
            blueprint_item["must_payoffs"] = _load_json(
                blueprint_item.pop("must_payoffs_json"), []
            )
            blueprint_item["forbidden_shortcuts"] = _load_json(
                blueprint_item.pop("forbidden_shortcuts_json"), []
            )
        arcs: List[Dict[str, Any]] = []
        for row in arc_rows:
            item = dict(row)
            item["involved_characters"] = _load_json(
                item.pop("involved_characters_json"), []
            )
            item["planned_turns"] = _load_json(
                item.pop("planned_turns_json"), []
            )
            arcs.append(item)
        memory_items: List[Dict[str, Any]] = []
        for row in recent_memory:
            item = dict(row)
            item["key_events"] = _load_json(
                item.pop("key_events_json"), []
            )
            item["unresolved_questions"] = _load_json(
                item.pop("unresolved_questions_json"), []
            )
            item["keywords"] = _load_json(item.pop("keywords_json"), [])
            memory_items.append(item)
        fact_items: List[Dict[str, Any]] = []
        for row in facts:
            item = dict(row)
            item["object"] = _load_json(item.pop("object_json"), {})
            fact_items.append(item)
        technique_items: List[Dict[str, Any]] = []
        for row in technique_rows:
            item = dict(row)
            item["usage_modes"] = _load_json(
                item.pop("usage_modes_json"), []
            )
            item["scope_label"] = str(project["title"])
            technique_items.append(item)

        return {
            "context_policy": {
                "plan_source": "author_confirmed_plan_only",
                "canon_source": "author_confirmed_canon_only",
                "drafts_excluded": True,
                "volume_sketches_are_display_only": True,
            },
            "project": {
                key: project[key]
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
                    "style_guide",
                    "point_of_view",
                    "planning_horizon",
                    "target_chapter_chars",
                )
            },
            "current_canonical_position": current_position,
            "characters": [dict(row) for row in characters],
            "volumes": [dict(row) for row in volumes],
            "future_chapters": [dict(row) for row in future_chapters],
            "confirmed_story_blueprint": blueprint_item,
            "confirmed_planned_plot_arcs": arcs,
            "canonical_memory": {
                "recent_chapters": memory_items,
                "story_facts": fact_items,
                "character_knowledge": [
                    dict(row) for row in knowledge
                ],
                "plot_threads": [dict(row) for row in threads],
                "foreshadowing": [dict(row) for row in hooks],
            },
            "active_techniques": compile_active_techniques(
                technique_items, usage="plan"
            ),
        }

    def create_suggestion(
        self,
        *,
        user_id: int,
        project_id: str,
        planning_mode: StoryPlanningMode,
        instruction: str,
        provider: str,
        model: str,
        credential_source: str,
    ) -> str:
        if planning_mode not in ALLOWED_PLANNING_MODES:
            raise ValueError("不支持的全书规划模式")
        if credential_source not in {"default", "personal"}:
            raise ValueError("不支持的 API 凭据来源")
        clean_instruction = instruction.strip()
        if len(clean_instruction) > 4000:
            raise ValueError("本次规划重点不能超过 4,000 个字符")
        suggestion_id = uuid.uuid4().hex
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            context = self._build_context(
                connection,
                user_id=user_id,
                project_id=project_id,
            )
            if context is None:
                connection.rollback()
                raise ValueError("小说项目不存在")
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
            active_story_plan = connection.execute(
                """
                SELECT id, project_id, planning_mode, instruction
                FROM story_plan_suggestions
                WHERE user_id=? AND status IN ('queued', 'running')
                ORDER BY created_at LIMIT 1
                """,
                (user_id,),
            ).fetchone()
            if active_story_plan:
                if (
                    str(active_story_plan["project_id"]) == project_id
                    and str(active_story_plan["planning_mode"])
                    == planning_mode
                    and str(active_story_plan["instruction"])
                    == clean_instruction
                ):
                    connection.rollback()
                    return str(active_story_plan["id"])
                connection.rollback()
                raise ValueError(
                    "你已有一个全书规划任务正在排队或运行，请等待其完成"
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
                  (SELECT id FROM story_structure_suggestions
                   WHERE user_id=? AND status IN ('queued', 'running')
                   ORDER BY created_at LIMIT 1) AS structure_id,
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
            baseline_fingerprint = _fingerprint(context)
            connection.execute(
                """
                INSERT INTO story_plan_suggestions(
                    id, project_id, user_id, planning_mode, instruction,
                    provider, model, credential_source, status,
                    baseline_fingerprint, context_snapshot_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?)
                """,
                (
                    suggestion_id,
                    project_id,
                    user_id,
                    planning_mode,
                    clean_instruction,
                    provider,
                    model,
                    credential_source,
                    baseline_fingerprint,
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
                       (SELECT COUNT(*)
                        FROM story_plan_suggestion_applications application
                        WHERE application.suggestion_id=suggestion.id)
                           AS application_count
                FROM story_plan_suggestions suggestion
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
                FROM story_plan_suggestions suggestion
                JOIN novel_projects p ON p.id=suggestion.project_id
                WHERE suggestion.id=? AND p.user_id=?
                """,
                (suggestion_id, user_id),
            ).fetchone()
            if not row:
                return None
            applications = connection.execute(
                """
                SELECT * FROM story_plan_suggestion_applications
                WHERE suggestion_id=?
                ORDER BY created_at, rowid
                """,
                (suggestion_id,),
            ).fetchall()
            current_context = self._build_context(
                connection,
                user_id=user_id,
                project_id=str(row["project_id"]),
            )
        item = dict(row)
        item["context_snapshot"] = _load_json(
            item.pop("context_snapshot_json"), {}
        )
        raw_result = item.pop("result_json")
        item["result"] = None
        if raw_result:
            try:
                item["result"] = StoryPlanProposalSet.model_validate_json(
                    str(raw_result)
                ).model_dump(mode="json")
            except ValueError:
                item["result_error"] = "已保存方案结构损坏，请重新生成"
        item["applications"] = []
        for row_item in applications:
            application = dict(row_item)
            application["selected_arc_indices"] = _load_json(
                application.pop("selected_arc_indices_json"), []
            )
            application["applied_arcs"] = _load_json(
                application.pop("applied_arcs_json"), []
            )
            application["baseline_changed"] = bool(
                application["baseline_changed"]
            )
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
                FROM story_plan_suggestions suggestion
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
                UPDATE story_plan_suggestions
                SET status='queued', started_at=NULL, claim_token=NULL,
                    lease_expires_at=NULL,
                    error='上一次全书规划租约已过期，已自动重新排队'
                WHERE status='running' AND lease_expires_at IS NOT NULL
                    AND lease_expires_at<=?
                """,
                (now,),
            )
            row = connection.execute(
                """
                SELECT suggestion.*
                FROM story_plan_suggestions suggestion
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
                UPDATE story_plan_suggestions
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
        result: StoryPlanProposalSet,
        raw_response: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> bool:
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE story_plan_suggestions
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
                UPDATE story_plan_suggestions
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
                UPDATE story_plan_suggestions
                SET status='queued', started_at=NULL, claim_token=NULL,
                    lease_expires_at=NULL, error=?
                WHERE id=? AND status='running' AND claim_token=?
                """,
                (error[:2000], suggestion_id, claim_token),
            )
            connection.commit()
        return cursor.rowcount == 1

    def apply_suggestion(
        self,
        *,
        user_id: int,
        suggestion_id: str,
        option_index: int,
        apply_blueprint: bool,
        selected_arc_indices: List[int],
    ) -> Dict[str, Any]:
        normalized_arc_indices = sorted(set(selected_arc_indices))
        if any(index < 0 for index in normalized_arc_indices):
            raise ValueError("所选剧情线不存在")
        if not apply_blueprint and not normalized_arc_indices:
            raise ValueError("至少选择全书蓝图或一条剧情线")
        now = utc_now()
        application_id = uuid.uuid4().hex
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT suggestion.*
                FROM story_plan_suggestions suggestion
                JOIN novel_projects p ON p.id=suggestion.project_id
                WHERE suggestion.id=? AND p.user_id=?
                    AND suggestion.status='completed'
                """,
                (suggestion_id, user_id),
            ).fetchone()
            if not row:
                connection.rollback()
                raise ValueError("全书方案不存在或尚未生成完成")
            try:
                proposal_set = StoryPlanProposalSet.model_validate_json(
                    str(row["result_json"] or "")
                )
            except ValueError as exc:
                connection.rollback()
                raise ValueError("已保存的全书方案结构损坏，请重新生成") from exc
            if option_index < 0 or option_index >= len(
                proposal_set.options
            ):
                connection.rollback()
                raise ValueError("所选全书方案不存在")
            option = proposal_set.options[option_index]
            if any(
                index >= len(option.plot_arcs)
                for index in normalized_arc_indices
            ):
                connection.rollback()
                raise ValueError("所选剧情线不存在")
            current_context = self._build_context(
                connection,
                user_id=user_id,
                project_id=str(row["project_id"]),
            )
            if current_context is None:
                connection.rollback()
                raise ValueError("小说项目不存在")
            baseline_changed = (
                _fingerprint(current_context)
                != str(row["baseline_fingerprint"])
            )
            selected_arcs = [
                option.plot_arcs[index]
                for index in normalized_arc_indices
            ]
            applied = (
                StoryPlanningService._apply_draft_bundle_in_connection(
                    connection,
                    project_id=str(row["project_id"]),
                    blueprint=option.blueprint if apply_blueprint else None,
                    arcs=selected_arcs,
                    source="story_planner",
                    now=now,
                )
            )
            connection.execute(
                """
                INSERT INTO story_plan_suggestion_applications(
                    id, suggestion_id, project_id, option_index,
                    apply_blueprint, selected_arc_indices_json,
                    created_blueprint_version_id, applied_arcs_json,
                    baseline_changed, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    application_id,
                    suggestion_id,
                    row["project_id"],
                    option_index,
                    int(apply_blueprint),
                    _json(normalized_arc_indices),
                    applied["blueprint_version_id"],
                    _json(applied["arcs"]),
                    int(baseline_changed),
                    now,
                ),
            )
            connection.execute(
                "UPDATE novel_projects SET updated_at=? WHERE id=?",
                (now, row["project_id"]),
            )
            connection.commit()
        return {
            "application_id": application_id,
            "project_id": str(row["project_id"]),
            "option_index": option_index,
            "option_label": option.label,
            "baseline_changed": baseline_changed,
            **applied,
        }
