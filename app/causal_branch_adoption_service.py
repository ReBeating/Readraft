from __future__ import annotations

import uuid
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .causal_branch_adoption_schema import (
    CausalBranchAdoptionItemSnapshot,
    CausalBranchTaskPatch,
)
from .causal_branch_schema import (
    CausalBranchOption,
    CausalBranchSimulationSet,
)
from .causal_branch_service import (
    BRANCH_CHAPTER_IMPACT_LABELS,
    BRANCH_EVIDENCE_STATUS_LABELS,
    BRANCH_KEY_LABELS,
    CausalBranchSimulationService,
)
from .db import Database, utc_now
from .json_support import (
    dump_canonical_json as _json,
    json_fingerprint as _fingerprint,
    load_json as _load_json,
)


ADOPTION_STATUS_LABELS = {
    "draft": "逐项审核中",
    "applied": "已写入未来任务卡",
    "abandoned": "已放弃",
    "reverted": "已安全撤销",
}
ADOPTION_DECISION_LABELS = {
    "pending": "待决定",
    "accepted": "准备采用",
    "rejected": "不采用",
}

CHAPTER_STATE_FIELDS = (
    "id",
    "project_id",
    "position",
    "title",
    "outline",
    "key_points",
    "volume_id",
    "skeleton_role",
    "skeleton_arc_titles_json",
    "skeleton_ending_hook",
    "skeleton_application_id",
    "head_version_id",
    "char_count",
    "needs_recheck",
    "target_chapter_chars",
    "updated_at",
)
PLAN_STATE_FIELDS = (
    "id",
    "project_id",
    "chapter_id",
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
PLAN_RESTORE_FIELDS = tuple(
    field
    for field in PLAN_STATE_FIELDS
    if field not in {"id", "project_id", "chapter_id"}
)
SCENE_STATE_FIELDS = (
    "id",
    "plan_id",
    "position",
    "beat_status",
    "draft_status",
    "current_version_id",
    "updated_at",
)


def _unique(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw).strip()
        normalized = value.casefold()
        if value and normalized not in seen:
            seen.add(normalized)
            result.append(value)
    return result


def _list(value: Any) -> List[str]:
    loaded = _load_json(value, [])
    if not isinstance(loaded, list):
        return []
    return _unique(str(item) for item in loaded)


def _text_lines(value: Any) -> List[str]:
    return _unique(str(value or "").splitlines())


def _subset_state(
    state: Mapping[str, Any],
    chapter_ids: Iterable[str],
) -> Dict[str, Any]:
    chapters = dict(state.get("chapters") or {})
    return {
        "chapters": {
            chapter_id: chapters[chapter_id]
            for chapter_id in sorted(set(chapter_ids))
            if chapter_id in chapters
        }
    }


class CausalBranchAdoptionService:
    """Turns one simulated branch into author-reviewed future plan patches."""

    def __init__(self, database: Database):
        self.database = database
        self.branch_service = CausalBranchSimulationService(database)

    @staticmethod
    def _capture_state(
        connection,
        *,
        project_id: str,
        chapter_ids: Iterable[str],
    ) -> Dict[str, Any]:
        unique_ids = sorted(set(str(item) for item in chapter_ids if item))
        if not unique_ids:
            return {"chapters": {}}
        placeholders = ",".join("?" for _ in unique_ids)
        rows = connection.execute(
            f"""
            SELECT ch.*, project.target_chapter_chars
            FROM novel_chapters ch
            JOIN novel_projects project ON project.id=ch.project_id
            WHERE ch.project_id=? AND ch.id IN ({placeholders})
            ORDER BY ch.position
            """,
            (project_id, *unique_ids),
        ).fetchall()
        chapters: Dict[str, Any] = {}
        for row in rows:
            chapter_id = str(row["id"])
            plan = connection.execute(
                """
                SELECT * FROM novel_chapter_plans
                WHERE chapter_id=?
                """,
                (chapter_id,),
            ).fetchone()
            scenes = []
            if plan:
                scenes = [
                    {
                        field: scene[field]
                        for field in SCENE_STATE_FIELDS
                    }
                    for scene in connection.execute(
                        """
                        SELECT * FROM novel_scene_beats
                        WHERE plan_id=?
                        ORDER BY position, id
                        """,
                        (plan["id"],),
                    ).fetchall()
                ]
            chapters[chapter_id] = {
                "chapter": {
                    field: row[field] for field in CHAPTER_STATE_FIELDS
                },
                "plan": (
                    {
                        field: plan[field]
                        for field in PLAN_STATE_FIELDS
                    }
                    if plan
                    else None
                ),
                "scenes": scenes,
            }
        if set(chapters) != set(unique_ids):
            raise ValueError("落地清单引用的未来章节已经不存在")
        return {"chapters": chapters}

    @staticmethod
    def _ensure_future_chapters(
        connection,
        *,
        project_id: str,
        chapter_ids: Iterable[str],
    ) -> None:
        ids = sorted(set(chapter_ids))
        placeholders = ",".join("?" for _ in ids)
        boundary = connection.execute(
            """
            SELECT COALESCE(MAX(position), 0) AS position
            FROM novel_chapters
            WHERE project_id=? AND head_version_id IS NOT NULL
            """,
            (project_id,),
        ).fetchone()
        rows = connection.execute(
            f"""
            SELECT id, position, head_version_id
            FROM novel_chapters
            WHERE project_id=? AND id IN ({placeholders})
            """,
            (project_id, *ids),
        ).fetchall()
        current_position = int(boundary["position"] or 0)
        if len(rows) != len(ids):
            raise ValueError("落地清单引用的未来章节已经不存在")
        invalid = [
            int(row["position"])
            for row in rows
            if row["head_version_id"]
            or int(row["position"]) <= current_position
        ]
        if invalid:
            raise ValueError(
                "第 "
                + "、".join(str(item) for item in sorted(invalid))
                + " 章已经进入正史范围，不能应用旧分支"
            )

    @staticmethod
    def _ensure_jobs_idle(
        connection,
        *,
        chapter_ids: Iterable[str],
    ) -> None:
        ids = sorted(set(chapter_ids))
        placeholders = ",".join("?" for _ in ids)
        active = connection.execute(
            f"""
            SELECT ch.position, job.operation
            FROM generation_jobs job
            JOIN novel_chapters ch ON ch.id=job.chapter_id
            WHERE job.chapter_id IN ({placeholders})
              AND job.status IN ('queued', 'running')
            ORDER BY ch.position
            LIMIT 1
            """,
            tuple(ids),
        ).fetchone()
        if active:
            raise ValueError(
                f"第 {int(active['position'])} 章有 AI 任务正在处理，"
                "请等待完成后再修改任务卡"
            )

    @staticmethod
    def _load_completed_simulation(
        connection,
        *,
        user_id: int,
        simulation_id: str,
    ):
        row = connection.execute(
            """
            SELECT simulation.*
            FROM novel_causal_branch_simulations simulation
            JOIN novel_projects project
              ON project.id=simulation.project_id
            WHERE simulation.id=? AND project.user_id=?
            """,
            (simulation_id, user_id),
        ).fetchone()
        if not row:
            raise ValueError("长期因果分支不存在")
        if str(row["status"]) != "completed" or not row["result_json"]:
            raise ValueError("长期因果分支尚未完成，不能建立落地清单")
        context = _load_json(row["context_snapshot_json"], {})
        try:
            result = CausalBranchSimulationSet.model_validate_json(
                str(row["result_json"])
            )
            result.ensure_context_compatible(context)
        except ValueError as exc:
            raise ValueError("长期因果分支结果已经损坏") from exc
        return row, context, result

    def _load_accepted_source(
        self,
        connection,
        *,
        user_id: int,
        simulation,
    ):
        source, proposal = self.branch_service._load_source_proposal(
            connection,
            user_id=user_id,
            suggestion_id=str(simulation["source_suggestion_id"]),
            proposal_index=int(simulation["proposal_index"]),
            require_pending=False,
        )
        review = connection.execute(
            """
            SELECT review.*, link.status AS link_status,
                   link.source_chapter_id AS link_source_chapter_id,
                   link.target_chapter_id AS link_target_chapter_id,
                   link.relation_type AS link_relation_type
            FROM novel_causal_link_suggestion_reviews review
            LEFT JOIN novel_chapter_causal_links link
              ON link.id=review.causal_link_id
            WHERE review.suggestion_id=? AND review.proposal_index=?
            """,
            (
                simulation["source_suggestion_id"],
                simulation["proposal_index"],
            ),
        ).fetchone()
        if not review or str(review["decision"]) != "accepted":
            raise ValueError(
                "请先返回原因果候选，完成语义核对并由作者采纳这条链接"
            )
        if (
            not review["causal_link_id"]
            or str(review["link_status"] or "") != "active"
        ):
            raise ValueError("原始因果链接已经归档，不能继续落地旧分支")
        if (
            str(review["link_source_chapter_id"])
            != proposal.source_chapter_id
            or str(review["link_target_chapter_id"])
            != proposal.target_chapter_id
            or str(review["link_relation_type"]) != proposal.relation_type
        ):
            raise ValueError("已采纳因果链接与推演候选不再一致")
        if not self.branch_service._source_baseline_is_current(
            connection,
            user_id=user_id,
            source=source,
        ):
            raise ValueError(
                "项目资料、正史、未来骨架或链接在推演后发生了变化；"
                "请重新生成因果建议与长期分支"
            )
        return review

    @staticmethod
    def _build_item(
        *,
        branch: CausalBranchOption,
        impact_index: int,
    ) -> CausalBranchAdoptionItemSnapshot:
        impact = branch.chapter_impacts[impact_index]
        chapter_id = impact.chapter_id
        must_happen = [
            impact.planned_change,
            f"因果承接要求：{impact.causal_role}",
        ]
        for transfer in branch.knowledge_transfers:
            if transfer.chapter_id != chapter_id:
                continue
            must_happen.append(
                f"{transfer.character_name}须通过“{transfer.channel}”"
                f"从“{transfer.before_state}”转为“{transfer.after_state}”；"
                f"避免：{transfer.risk}"
            )
        plot_threads = [
            trajectory.arc_title
            for trajectory in branch.arc_trajectories
            if chapter_id in trajectory.affected_chapter_ids
        ]
        setup = (
            [impact.planned_change]
            if impact.impact_type in {"setup", "information_transfer"}
            else []
        )
        payoff = [
            f"{item.payoff_text}：{item.consequence}"
            for item in branch.payoff_impacts
            if item.delivery_chapter_id == chapter_id
        ]
        if impact.impact_type == "payoff":
            payoff.append(impact.planned_change)
        rationale_parts = [
            f"分支“{branch.label}”中的"
            f"{BRANCH_CHAPTER_IMPACT_LABELS.get(impact.impact_type, impact.impact_type)}",
            f"证据状态："
            f"{BRANCH_EVIDENCE_STATUS_LABELS.get(impact.evidence_status, impact.evidence_status)}",
        ]
        if impact.unresolved_assumption:
            rationale_parts.append(
                "仍需作者确认：" + impact.unresolved_assumption
            )
        return CausalBranchAdoptionItemSnapshot(
            item_key=f"{impact_index + 1:02d}:{chapter_id}",
            chapter_id=chapter_id,
            chapter_position=impact.chapter_position,
            chapter_title=impact.chapter_title,
            impact_type=impact.impact_type,
            planned_change=impact.planned_change,
            causal_role=impact.causal_role,
            evidence_status=impact.evidence_status,
            evidence_refs=list(impact.evidence_refs),
            unresolved_assumption=impact.unresolved_assumption,
            rationale="；".join(rationale_parts),
            proposed_patch=CausalBranchTaskPatch(
                must_happen=_unique(must_happen),
                plot_threads=_unique(plot_threads),
                foreshadow_setup=_unique(setup),
                foreshadow_payoff=_unique(payoff),
            ),
        )

    def create_adoption(
        self,
        *,
        user_id: int,
        simulation_id: str,
        branch_key: str,
        meaning_confirmed: bool,
    ) -> str:
        if branch_key not in BRANCH_KEY_LABELS:
            raise ValueError("请选择有效的长期因果分支")
        if not meaning_confirmed:
            raise ValueError(
                "请先确认已采纳链接的含义仍与这次分支推演一致"
            )
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    """
                    SELECT adoption.id
                    FROM novel_causal_branch_adoptions adoption
                    JOIN novel_projects project
                      ON project.id=adoption.project_id
                    WHERE adoption.simulation_id=?
                      AND adoption.branch_key=?
                      AND adoption.status IN ('draft', 'applied')
                      AND project.user_id=?
                    ORDER BY adoption.created_at DESC
                    LIMIT 1
                    """,
                    (simulation_id, branch_key, user_id),
                ).fetchone()
                if existing:
                    connection.commit()
                    return str(existing["id"])
                simulation, _context, result = (
                    self._load_completed_simulation(
                        connection,
                        user_id=user_id,
                        simulation_id=simulation_id,
                    )
                )
                review = self._load_accepted_source(
                    connection,
                    user_id=user_id,
                    simulation=simulation,
                )
                branch = next(
                    (
                        item
                        for item in result.branches
                        if item.branch_key == branch_key
                    ),
                    None,
                )
                if branch is None:
                    raise ValueError("所选长期因果分支不存在")
                items = [
                    self._build_item(
                        branch=branch,
                        impact_index=index,
                    )
                    for index in range(len(branch.chapter_impacts))
                ]
                chapter_ids = [item.chapter_id for item in items]
                self._ensure_future_chapters(
                    connection,
                    project_id=str(simulation["project_id"]),
                    chapter_ids=chapter_ids,
                )
                baseline_state = self._capture_state(
                    connection,
                    project_id=str(simulation["project_id"]),
                    chapter_ids=chapter_ids,
                )
                adoption_id = uuid.uuid4().hex
                connection.execute(
                    """
                    INSERT INTO novel_causal_branch_adoptions(
                        id, simulation_id, project_id, user_id,
                        source_causal_link_id, branch_key, status,
                        baseline_fingerprint, baseline_state_json,
                        branch_snapshot_json, accepted_item_count,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'draft', ?, ?, ?, 0, ?, ?)
                    """,
                    (
                        adoption_id,
                        simulation_id,
                        simulation["project_id"],
                        user_id,
                        review["causal_link_id"],
                        branch_key,
                        _fingerprint(baseline_state),
                        _json(baseline_state),
                        branch.model_dump_json(),
                        now,
                        now,
                    ),
                )
                for item in items:
                    connection.execute(
                        """
                        INSERT INTO novel_causal_branch_adoption_items(
                            id, adoption_id, item_key, chapter_id,
                            chapter_position, chapter_title, impact_type,
                            proposal_json, edited_patch_json, decision,
                            author_note, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', '',
                                  ?, ?)
                        """,
                        (
                            uuid.uuid4().hex,
                            adoption_id,
                            item.item_key,
                            item.chapter_id,
                            item.chapter_position,
                            item.chapter_title,
                            item.impact_type,
                            item.model_dump_json(),
                            item.proposed_patch.model_dump_json(),
                            now,
                            now,
                        ),
                    )
            except Exception:
                connection.rollback()
                raise
            connection.commit()
        return adoption_id

    def list_for_simulation(
        self,
        *,
        user_id: int,
        simulation_id: str,
    ) -> List[Dict[str, Any]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT adoption.id, adoption.branch_key, adoption.status,
                       adoption.accepted_item_count, adoption.created_at,
                       adoption.applied_at, adoption.reverted_at,
                       COUNT(item.id) AS item_count,
                       SUM(item.decision='pending') AS pending_count
                FROM novel_causal_branch_adoptions adoption
                JOIN novel_projects project
                  ON project.id=adoption.project_id
                LEFT JOIN novel_causal_branch_adoption_items item
                  ON item.adoption_id=adoption.id
                WHERE adoption.simulation_id=? AND project.user_id=?
                  AND adoption.status!='abandoned'
                GROUP BY adoption.id
                ORDER BY adoption.created_at DESC
                """,
                (simulation_id, user_id),
            ).fetchall()
        return [
            {
                **dict(row),
                "status_label": ADOPTION_STATUS_LABELS.get(
                    str(row["status"]), str(row["status"])
                ),
            }
            for row in rows
        ]

    def get_adoption(
        self,
        *,
        user_id: int,
        adoption_id: str,
    ) -> Optional[Dict[str, Any]]:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT adoption.*, project.title AS project_title,
                       link.status AS source_link_status,
                       link.cause_text AS source_cause_text,
                       link.effect_text AS source_effect_text,
                       link.source_chapter_id,
                       link.target_chapter_id
                FROM novel_causal_branch_adoptions adoption
                JOIN novel_projects project
                  ON project.id=adoption.project_id
                JOIN novel_chapter_causal_links link
                  ON link.id=adoption.source_causal_link_id
                WHERE adoption.id=? AND project.user_id=?
                """,
                (adoption_id, user_id),
            ).fetchone()
            if not row:
                return None
            item_rows = connection.execute(
                """
                SELECT item.*, chapter.title AS current_chapter_title,
                       chapter.position AS current_chapter_position,
                       plan.status AS current_plan_status
                FROM novel_causal_branch_adoption_items item
                JOIN novel_chapters chapter ON chapter.id=item.chapter_id
                LEFT JOIN novel_chapter_plans plan
                  ON plan.chapter_id=item.chapter_id
                WHERE item.adoption_id=?
                ORDER BY item.chapter_position, item.item_key
                """,
                (adoption_id,),
            ).fetchall()
            item_ids = [str(item["chapter_id"]) for item in item_rows]
            current_state = self._capture_state(
                connection,
                project_id=str(row["project_id"]),
                chapter_ids=item_ids,
            )
        item_data: List[Dict[str, Any]] = []
        for item_row in item_rows:
            item = dict(item_row)
            try:
                snapshot = CausalBranchAdoptionItemSnapshot.model_validate_json(
                    str(item.pop("proposal_json"))
                )
                patch = CausalBranchTaskPatch.model_validate_json(
                    str(item.pop("edited_patch_json"))
                )
            except ValueError:
                snapshot = None
                patch = None
            item["snapshot"] = (
                snapshot.model_dump(mode="json") if snapshot else None
            )
            item["patch"] = patch.model_dump(mode="json") if patch else None
            item["decision_label"] = ADOPTION_DECISION_LABELS.get(
                str(item["decision"]), str(item["decision"])
            )
            item["impact_label"] = BRANCH_CHAPTER_IMPACT_LABELS.get(
                str(item["impact_type"]), str(item["impact_type"])
            )
            item_data.append(item)
        result = dict(row)
        branch = CausalBranchOption.model_validate_json(
            str(result.pop("branch_snapshot_json"))
        )
        baseline_state = _load_json(
            result.pop("baseline_state_json"), {"chapters": {}}
        )
        result["branch"] = branch.model_dump(mode="json")
        result["branch_label"] = BRANCH_KEY_LABELS.get(
            str(result["branch_key"]), str(result["branch_key"])
        )
        result["status_label"] = ADOPTION_STATUS_LABELS.get(
            str(result["status"]), str(result["status"])
        )
        result["items"] = item_data
        result["pending_count"] = sum(
            item["decision"] == "pending" for item in item_data
        )
        result["accepted_count"] = sum(
            item["decision"] == "accepted" for item in item_data
        )
        result["rejected_count"] = sum(
            item["decision"] == "rejected" for item in item_data
        )
        relevant_ids = [
            str(item["chapter_id"])
            for item in item_data
            if item["decision"] != "rejected"
        ]
        if not relevant_ids:
            relevant_ids = [str(item["chapter_id"]) for item in item_data]
        result["baseline_changed"] = (
            _fingerprint(_subset_state(current_state, relevant_ids))
            != _fingerprint(_subset_state(baseline_state, relevant_ids))
            if result["status"] == "draft"
            else False
        )
        result["apply_blocker"] = ""
        if result["status"] != "draft":
            result["apply_blocker"] = "这份清单已经结束审核"
        elif str(result["source_link_status"]) != "active":
            result["apply_blocker"] = "原始因果链接已经归档"
        elif result["baseline_changed"]:
            result["apply_blocker"] = (
                "相关章节在清单建立后发生了变化，请放弃旧清单后重新建立"
            )
        elif result["pending_count"]:
            result["apply_blocker"] = "请先逐项决定采用或不采用"
        elif not result["accepted_count"]:
            result["apply_blocker"] = "至少需要采用一项章节修改"
        after_state = _load_json(
            result.pop("after_state_json"), {"chapters": {}}
        )
        applied_ids = [
            str(item["chapter_id"])
            for item in item_data
            if item.get("applied_at")
        ]
        result["can_revert"] = bool(
            result["status"] == "applied"
            and result.get("after_fingerprint")
            and _fingerprint(_subset_state(current_state, applied_ids))
            == str(result["after_fingerprint"])
            and after_state
        )
        result.pop("before_state_json", None)
        return result

    def review_item(
        self,
        *,
        user_id: int,
        adoption_id: str,
        item_id: str,
        decision: str,
        patch: CausalBranchTaskPatch,
        author_note: str,
    ) -> None:
        if decision not in {"accepted", "rejected"}:
            raise ValueError("请选择采用或不采用这项修改")
        note = author_note.strip()
        if len(note) > 1200:
            raise ValueError("单项作者备注不能超过 1,200 个字符")
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT item.*, adoption.status,
                       adoption.source_causal_link_id
                FROM novel_causal_branch_adoption_items item
                JOIN novel_causal_branch_adoptions adoption
                  ON adoption.id=item.adoption_id
                JOIN novel_projects project
                  ON project.id=adoption.project_id
                WHERE item.id=? AND item.adoption_id=?
                  AND project.user_id=?
                """,
                (item_id, adoption_id, user_id),
            ).fetchone()
            if not row:
                connection.rollback()
                raise ValueError("分支落地项不存在")
            if str(row["status"]) != "draft":
                connection.rollback()
                raise ValueError("这份落地清单已经结束审核")
            link = connection.execute(
                """
                SELECT status FROM novel_chapter_causal_links
                WHERE id=?
                """,
                (row["source_causal_link_id"],),
            ).fetchone()
            if not link or str(link["status"]) != "active":
                connection.rollback()
                raise ValueError("原始因果链接已经归档")
            snapshot = CausalBranchAdoptionItemSnapshot.model_validate_json(
                str(row["proposal_json"])
            )
            allowed_threads = set(
                snapshot.proposed_patch.plot_threads
            )
            unknown_threads = [
                title
                for title in patch.plot_threads
                if title not in allowed_threads
            ]
            if unknown_threads:
                connection.rollback()
                raise ValueError(
                    "落地项不能新增推演之外的剧情线："
                    + "、".join(unknown_threads)
                )
            connection.execute(
                """
                UPDATE novel_causal_branch_adoption_items
                SET edited_patch_json=?, decision=?, author_note=?,
                    decided_at=?, updated_at=?
                WHERE id=?
                """,
                (
                    patch.model_dump_json(),
                    decision,
                    note,
                    now,
                    now,
                    item_id,
                ),
            )
            connection.execute(
                """
                UPDATE novel_causal_branch_adoptions
                SET updated_at=? WHERE id=?
                """,
                (now, adoption_id),
            )
            connection.commit()

    @staticmethod
    def _apply_item(
        connection,
        *,
        item,
        project_id: str,
        now: str,
    ) -> Dict[str, int]:
        patch = CausalBranchTaskPatch.model_validate_json(
            str(item["edited_patch_json"])
        )
        chapter = connection.execute(
            """
            SELECT ch.*, project.target_chapter_chars
            FROM novel_chapters ch
            JOIN novel_projects project ON project.id=ch.project_id
            WHERE ch.id=? AND ch.project_id=?
            """,
            (item["chapter_id"], project_id),
        ).fetchone()
        if not chapter:
            raise ValueError("待应用章节已经不存在")
        plan = connection.execute(
            """
            SELECT * FROM novel_chapter_plans WHERE chapter_id=?
            """,
            (item["chapter_id"],),
        ).fetchone()
        skeleton_threads = _list(
            chapter["skeleton_arc_titles_json"]
        )
        skeleton_key_points = _text_lines(chapter["key_points"])
        if plan:
            plot_threads = _list(plan["plot_threads_json"])
            must_happen = _list(plan["must_happen_json"])
            setup = _list(plan["foreshadow_setup_json"])
            payoff = _list(plan["foreshadow_payoff_json"])
            plan_id = str(plan["id"])
        else:
            plot_threads = list(skeleton_threads)
            must_happen = list(skeleton_key_points)
            setup = []
            payoff = []
            plan_id = uuid.uuid4().hex
        plot_threads = _unique([*plot_threads, *patch.plot_threads])
        must_happen = _unique([*must_happen, *patch.must_happen])
        setup = _unique([*setup, *patch.foreshadow_setup])
        payoff = _unique([*payoff, *patch.foreshadow_payoff])
        skeleton_threads = _unique(
            [*skeleton_threads, *patch.plot_threads]
        )
        skeleton_key_points = _unique(
            [*skeleton_key_points, *patch.must_happen]
        )
        reset_count = int(
            bool(plan and str(plan["status"]) == "confirmed")
        )
        if plan:
            connection.execute(
                """
                UPDATE novel_chapter_plans
                SET plot_threads_json=?, must_happen_json=?,
                    foreshadow_setup_json=?,
                    foreshadow_payoff_json=?,
                    status='draft', source='branch_adoption',
                    confirmed_at=NULL, updated_at=?
                WHERE id=?
                """,
                (
                    _json(plot_threads),
                    _json(must_happen),
                    _json(setup),
                    _json(payoff),
                    now,
                    plan_id,
                ),
            )
        else:
            connection.execute(
                """
                INSERT INTO novel_chapter_plans(
                    id, project_id, chapter_id, purpose,
                    plot_threads_json, must_happen_json,
                    foreshadow_setup_json, foreshadow_payoff_json,
                    ending_hook, target_chars, status, source,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'draft',
                          'branch_adoption', ?, ?)
                """,
                (
                    plan_id,
                    project_id,
                    item["chapter_id"],
                    str(chapter["outline"] or ""),
                    _json(plot_threads),
                    _json(must_happen),
                    _json(setup),
                    _json(payoff),
                    str(chapter["skeleton_ending_hook"] or ""),
                    int(chapter["target_chapter_chars"] or 3000),
                    now,
                    now,
                ),
            )
        stale = connection.execute(
            """
            UPDATE novel_scene_beats
            SET draft_status='stale', updated_at=?
            WHERE plan_id=? AND beat_status='active'
              AND current_version_id IS NOT NULL
            """,
            (now, plan_id),
        )
        connection.execute(
            """
            UPDATE novel_chapters
            SET key_points=?, skeleton_arc_titles_json=?,
                skeleton_application_id=NULL,
                needs_recheck=CASE
                    WHEN head_version_id IS NOT NULL
                      OR char_count>0
                      OR EXISTS(
                          SELECT 1 FROM novel_scene_beats scene
                          WHERE scene.plan_id=?
                            AND scene.current_version_id IS NOT NULL
                      )
                    THEN 1
                    ELSE needs_recheck
                END,
                updated_at=?
            WHERE id=?
            """,
            (
                "\n".join(skeleton_key_points),
                _json(skeleton_threads),
                plan_id,
                now,
                item["chapter_id"],
            ),
        )
        return {
            "reset_task_card_count": reset_count,
            "stale_scene_count": int(stale.rowcount),
        }

    def apply_adoption(
        self,
        *,
        user_id: int,
        adoption_id: str,
        author_confirmed: bool,
    ) -> Dict[str, Any]:
        if not author_confirmed:
            raise ValueError(
                "请确认只会写入已采用条目，并让相关任务卡回到草稿"
            )
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                adoption = connection.execute(
                    """
                    SELECT adoption.*
                    FROM novel_causal_branch_adoptions adoption
                    JOIN novel_projects project
                      ON project.id=adoption.project_id
                    WHERE adoption.id=? AND project.user_id=?
                    """,
                    (adoption_id, user_id),
                ).fetchone()
                if not adoption:
                    raise ValueError("分支落地清单不存在")
                if str(adoption["status"]) != "draft":
                    raise ValueError("这份落地清单已经结束审核")
                link = connection.execute(
                    """
                    SELECT status FROM novel_chapter_causal_links
                    WHERE id=?
                    """,
                    (adoption["source_causal_link_id"],),
                ).fetchone()
                if not link or str(link["status"]) != "active":
                    raise ValueError("原始因果链接已经归档")
                items = connection.execute(
                    """
                    SELECT * FROM novel_causal_branch_adoption_items
                    WHERE adoption_id=?
                    ORDER BY chapter_position, item_key
                    """,
                    (adoption_id,),
                ).fetchall()
                if any(str(item["decision"]) == "pending" for item in items):
                    raise ValueError("请先逐项决定采用或不采用")
                accepted = [
                    item
                    for item in items
                    if str(item["decision"]) == "accepted"
                ]
                if not accepted:
                    raise ValueError("至少需要采用一项章节修改")
                chapter_ids = [
                    str(item["chapter_id"]) for item in accepted
                ]
                self._ensure_future_chapters(
                    connection,
                    project_id=str(adoption["project_id"]),
                    chapter_ids=chapter_ids,
                )
                self._ensure_jobs_idle(
                    connection,
                    chapter_ids=chapter_ids,
                )
                current_state = self._capture_state(
                    connection,
                    project_id=str(adoption["project_id"]),
                    chapter_ids=chapter_ids,
                )
                baseline_state = _load_json(
                    adoption["baseline_state_json"],
                    {"chapters": {}},
                )
                if _fingerprint(current_state) != _fingerprint(
                    _subset_state(baseline_state, chapter_ids)
                ):
                    raise ValueError(
                        "准备采用的章节在清单建立后发生了变化；"
                        "请放弃旧清单后重新建立"
                    )
                before_state = current_state
                reset_count = 0
                stale_count = 0
                for item in accepted:
                    result = self._apply_item(
                        connection,
                        item=item,
                        project_id=str(adoption["project_id"]),
                        now=now,
                    )
                    reset_count += result["reset_task_card_count"]
                    stale_count += result["stale_scene_count"]
                    connection.execute(
                        """
                        UPDATE novel_causal_branch_adoption_items
                        SET applied_at=?, updated_at=?
                        WHERE id=?
                        """,
                        (now, now, item["id"]),
                    )
                after_state = self._capture_state(
                    connection,
                    project_id=str(adoption["project_id"]),
                    chapter_ids=chapter_ids,
                )
                connection.execute(
                    """
                    UPDATE novel_causal_branch_adoptions
                    SET status='applied', accepted_item_count=?,
                        before_state_json=?, after_state_json=?,
                        after_fingerprint=?, applied_at=?, updated_at=?
                    WHERE id=?
                    """,
                    (
                        len(accepted),
                        _json(before_state),
                        _json(after_state),
                        _fingerprint(after_state),
                        now,
                        now,
                        adoption_id,
                    ),
                )
                connection.execute(
                    """
                    UPDATE novel_projects SET updated_at=? WHERE id=?
                    """,
                    (now, adoption["project_id"]),
                )
            except Exception:
                connection.rollback()
                raise
            connection.commit()
        return {
            "adoption_id": adoption_id,
            "project_id": str(adoption["project_id"]),
            "applied_item_count": len(accepted),
            "reset_task_card_count": reset_count,
            "stale_scene_count": stale_count,
        }

    def abandon_adoption(
        self,
        *,
        user_id: int,
        adoption_id: str,
    ) -> None:
        now = utc_now()
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE novel_causal_branch_adoptions
                SET status='abandoned', abandoned_at=?, updated_at=?
                WHERE id=? AND user_id=? AND status='draft'
                """,
                (now, now, adoption_id, user_id),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise ValueError("落地清单不存在或已经结束审核")
            connection.commit()

    def revert_adoption(
        self,
        *,
        user_id: int,
        adoption_id: str,
    ) -> Dict[str, Any]:
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                adoption = connection.execute(
                    """
                    SELECT adoption.*
                    FROM novel_causal_branch_adoptions adoption
                    JOIN novel_projects project
                      ON project.id=adoption.project_id
                    WHERE adoption.id=? AND project.user_id=?
                    """,
                    (adoption_id, user_id),
                ).fetchone()
                if not adoption:
                    raise ValueError("分支落地清单不存在")
                if str(adoption["status"]) != "applied":
                    raise ValueError("只有已应用且未撤销的清单可以撤销")
                before_state = _load_json(
                    adoption["before_state_json"], {"chapters": {}}
                )
                after_state = _load_json(
                    adoption["after_state_json"], {"chapters": {}}
                )
                chapter_ids = sorted(
                    dict(after_state.get("chapters") or {})
                )
                self._ensure_jobs_idle(
                    connection,
                    chapter_ids=chapter_ids,
                )
                current_state = self._capture_state(
                    connection,
                    project_id=str(adoption["project_id"]),
                    chapter_ids=chapter_ids,
                )
                if _fingerprint(current_state) != str(
                    adoption["after_fingerprint"] or ""
                ):
                    raise ValueError(
                        "应用后相关章节或任务卡已经继续修改；"
                        "为避免覆盖新工作，系统不能自动撤销"
                    )
                for chapter_id in chapter_ids:
                    before = before_state["chapters"][chapter_id]
                    after = after_state["chapters"][chapter_id]
                    before_plan = before.get("plan")
                    after_plan = after.get("plan")
                    if before_plan is None:
                        if after_plan is not None:
                            connection.execute(
                                """
                                DELETE FROM novel_chapter_plans
                                WHERE id=? AND chapter_id=?
                                """,
                                (after_plan["id"], chapter_id),
                            )
                    else:
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
                                chapter_id,
                            ),
                        )
                        scenes = {
                            str(scene["id"]): scene
                            for scene in before.get("scenes") or []
                        }
                        for scene_id, scene in scenes.items():
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
                    chapter = before["chapter"]
                    connection.execute(
                        """
                        UPDATE novel_chapters
                        SET key_points=?, skeleton_arc_titles_json=?,
                            skeleton_application_id=?, needs_recheck=?,
                            updated_at=?
                        WHERE id=? AND project_id=?
                        """,
                        (
                            chapter["key_points"],
                            chapter["skeleton_arc_titles_json"],
                            chapter["skeleton_application_id"],
                            chapter["needs_recheck"],
                            chapter["updated_at"],
                            chapter_id,
                            adoption["project_id"],
                        ),
                    )
                connection.execute(
                    """
                    UPDATE novel_causal_branch_adoptions
                    SET status='reverted', reverted_at=?, updated_at=?
                    WHERE id=?
                    """,
                    (now, now, adoption_id),
                )
                connection.execute(
                    """
                    UPDATE novel_projects SET updated_at=? WHERE id=?
                    """,
                    (now, adoption["project_id"]),
                )
            except Exception:
                connection.rollback()
                raise
            connection.commit()
        return {
            "adoption_id": adoption_id,
            "project_id": str(adoption["project_id"]),
            "reverted_item_count": len(chapter_ids),
        }
