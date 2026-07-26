from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Dict, List, Mapping, Optional

from .causal_branch_schema import CausalBranchSimulationSet
from .causal_suggestion_schema import (
    CausalSuggestionSet,
    ProposedCausalLink,
)
from .causal_suggestion_service import (
    CausalSuggestionService,
    _baseline_fingerprint as _causal_baseline_fingerprint,
    _task_card_fallbacks,
)
from .db import Database, utc_after, utc_now


BRANCH_KEY_LABELS = {
    "minimal_change": "最小改动",
    "distributed_consequences": "分散后果",
    "stress_test": "压力测试",
}
BRANCH_INTERVENTION_LABELS = {
    "low": "低改动",
    "medium": "中等改动",
    "high": "高改动",
}
BRANCH_EVIDENCE_STATUS_LABELS = {
    "supported": "当前证据支持",
    "uncertain": "证据不足",
    "conflict": "发现冲突",
}
BRANCH_CHAPTER_IMPACT_LABELS = {
    "setup": "补设前提",
    "escalation": "升级压力",
    "information_transfer": "传递信息",
    "choice": "改变选择",
    "reversal": "形成反转",
    "payoff": "兑现回报",
    "repair": "修复断点",
}
BRANCH_ARC_TRAJECTORY_LABELS = {
    "advances": "推进",
    "complicates": "复杂化",
    "delays": "延迟",
    "pays_off": "兑现",
    "risks_breaking": "可能破坏",
}
BRANCH_PAYOFF_TIMING_LABELS = {
    "unchanged": "时机不变",
    "earlier": "提前",
    "later": "延后",
    "reframed": "改换兑现方式",
    "at_risk": "可能无法兑现",
}


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _load_json(value: Any, fallback: Any) -> Any:
    try:
        return json.loads(str(value))
    except (TypeError, ValueError):
        return fallback


def _fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _proposal_signature(proposal: ProposedCausalLink) -> str:
    return hashlib.sha256(
        _json(proposal.model_dump(mode="json")).encode("utf-8")
    ).hexdigest()


class CausalBranchSimulationService:
    def __init__(self, database: Database):
        self.database = database

    @staticmethod
    def _load_source_proposal(
        connection,
        *,
        user_id: int,
        suggestion_id: str,
        proposal_index: int,
        require_pending: bool,
    ):
        source = connection.execute(
            """
            SELECT suggestion.*
            FROM novel_causal_link_suggestions suggestion
            JOIN novel_projects project
              ON project.id=suggestion.project_id
            WHERE suggestion.id=? AND project.user_id=?
            """,
            (suggestion_id, user_id),
        ).fetchone()
        if not source:
            raise ValueError("因果建议任务不存在")
        if str(source["status"]) != "completed":
            raise ValueError("因果建议尚未完成")
        if proposal_index < 0:
            raise ValueError("因果候选不存在或已经损坏")
        if require_pending:
            reviewed = connection.execute(
                """
                SELECT decision
                FROM novel_causal_link_suggestion_reviews
                WHERE suggestion_id=? AND proposal_index=?
                """,
                (suggestion_id, proposal_index),
            ).fetchone()
            if reviewed:
                label = (
                    "采纳"
                    if str(reviewed["decision"]) == "accepted"
                    else "忽略"
                )
                raise ValueError(
                    f"这条候选已经{label}；长期分支推演只能从待审候选发起"
                )
        try:
            source_result = CausalSuggestionSet.model_validate_json(
                str(source["result_json"] or "")
            )
            frozen = _load_json(source["context_snapshot_json"], {})
            source_result.ensure_context_compatible(frozen)
            proposal = source_result.proposals[proposal_index]
        except (ValueError, IndexError) as exc:
            raise ValueError("因果候选不存在或已经损坏") from exc
        return source, proposal

    @staticmethod
    def _source_baseline_is_current(
        connection,
        *,
        user_id: int,
        source,
    ) -> bool:
        current = CausalSuggestionService._build_context(
            connection,
            user_id=user_id,
            project_id=str(source["project_id"]),
            chapter_limit=int(source["chapter_limit"]),
        )
        if current is None:
            return False
        accepted_rows = connection.execute(
            """
            SELECT causal_link_id, source_chapter_id, target_chapter_id
            FROM novel_causal_link_suggestion_reviews
            WHERE suggestion_id=? AND decision='accepted'
              AND causal_link_id IS NOT NULL
            """,
            (source["id"],),
        ).fetchall()
        ignored_link_ids = {
            str(row["causal_link_id"]) for row in accepted_rows
        }
        affected_chapter_ids = {
            str(row[key])
            for row in accepted_rows
            for key in ("source_chapter_id", "target_chapter_id")
        }
        frozen = _load_json(source["context_snapshot_json"], {})
        return (
            _causal_baseline_fingerprint(
                current,
                ignored_link_ids=ignored_link_ids,
                task_card_fallbacks=_task_card_fallbacks(
                    frozen,
                    affected_chapter_ids,
                ),
            )
            == str(source["baseline_fingerprint"])
        )

    @staticmethod
    def _build_simulation_context(
        connection,
        *,
        user_id: int,
        source,
        proposal: ProposedCausalLink,
        proposal_index: int,
        horizon_chapter_count: int,
    ) -> Optional[Dict[str, Any]]:
        context = CausalSuggestionService._build_context(
            connection,
            user_id=user_id,
            project_id=str(source["project_id"]),
            chapter_limit=horizon_chapter_count,
        )
        if context is None:
            return None
        context["simulation_schema_version"] = 1
        context["horizon_chapter_count"] = horizon_chapter_count
        context["source_suggestion"] = {
            "id": str(source["id"]),
            "proposal_index": proposal_index,
            "chapter_limit": int(source["chapter_limit"]),
            "created_at": str(source["created_at"]),
        }
        context["selected_proposal"] = proposal.model_dump(mode="json")
        context["source_proposal_signature"] = _proposal_signature(
            proposal
        )
        context["simulation_policy"] = {
            "read_only": True,
            "branch_count": 3,
            "branch_order": [
                "minimal_change",
                "distributed_consequences",
                "stress_test",
            ],
            "future_plan_is_not_canon": True,
            "no_automatic_outline_mutation": True,
            "no_automatic_causal_link_creation": True,
            "uncertainty_must_remain_visible": True,
        }
        return context

    @staticmethod
    def _ensure_simulation_ready(
        context: Mapping[str, Any],
    ) -> None:
        horizon = int(context.get("horizon_chapter_count") or 0)
        future = list(context.get("future_chapters") or [])
        if len(future) != horizon:
            raise ValueError(
                f"当前只有 {len(future)} 章未来骨架；"
                f"请先补足到 {horizon} 章再做长期推演"
            )
        selected = dict(context.get("selected_proposal") or {})
        target_id = str(selected.get("target_chapter_id") or "")
        target = next(
            (
                item
                for item in future
                if str(item.get("id") or "") == target_id
            ),
            None,
        )
        if target is None:
            raise ValueError(
                "所选候选的结果章不在本次未来范围内；"
                "请选择更大的范围，最多 30 章"
            )
        last_position = max(
            int(item.get("position") or 0) for item in future
        )
        if int(target.get("position") or 0) >= last_position:
            raise ValueError(
                "长期推演范围至少要覆盖结果章之后一章；"
                "请扩大范围或选择更靠前的候选"
            )

    def create_simulation(
        self,
        *,
        user_id: int,
        suggestion_id: str,
        proposal_index: int,
        horizon_chapter_count: int,
        instruction: str,
        provider: str,
        model: str,
        credential_source: str,
    ) -> str:
        if not 10 <= horizon_chapter_count <= 30:
            raise ValueError("长期因果推演范围必须为未来 10–30 章")
        if credential_source not in {"default", "personal"}:
            raise ValueError("不支持的 API 凭据来源")
        clean_instruction = instruction.strip()
        if len(clean_instruction) > 4000:
            raise ValueError("本次推演重点不能超过 4,000 个字符")
        simulation_id = uuid.uuid4().hex
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            source, proposal = self._load_source_proposal(
                connection,
                user_id=user_id,
                suggestion_id=suggestion_id,
                proposal_index=proposal_index,
                require_pending=True,
            )
            if not self._source_baseline_is_current(
                connection,
                user_id=user_id,
                source=source,
            ):
                connection.rollback()
                raise ValueError(
                    "项目资料、正史、未来骨架或现有因果链接已经变化；"
                    "请重新生成因果建议后再推演"
                )
            context = self._build_simulation_context(
                connection,
                user_id=user_id,
                source=source,
                proposal=proposal,
                proposal_index=proposal_index,
                horizon_chapter_count=horizon_chapter_count,
            )
            if context is None:
                connection.rollback()
                raise ValueError("小说项目不存在")
            self._ensure_simulation_ready(context)
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
            active_self = connection.execute(
                """
                SELECT id, source_suggestion_id, proposal_index,
                       horizon_chapter_count, instruction
                FROM novel_causal_branch_simulations
                WHERE user_id=? AND status IN ('queued', 'running')
                ORDER BY created_at LIMIT 1
                """,
                (user_id,),
            ).fetchone()
            if active_self:
                if (
                    str(active_self["source_suggestion_id"])
                    == suggestion_id
                    and int(active_self["proposal_index"])
                    == proposal_index
                    and int(active_self["horizon_chapter_count"])
                    == horizon_chapter_count
                    and str(active_self["instruction"])
                    == clean_instruction
                ):
                    connection.rollback()
                    return str(active_self["id"])
                connection.rollback()
                raise ValueError(
                    "你已有一个长期因果推演正在排队或运行，请等待其完成"
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
                  (SELECT id FROM story_structure_suggestions
                   WHERE user_id=? AND status IN ('queued', 'running')
                   ORDER BY created_at LIMIT 1) AS structure_id,
                  (SELECT id FROM novel_causal_link_suggestions
                   WHERE user_id=? AND status IN ('queued', 'running')
                   ORDER BY created_at LIMIT 1) AS causal_id
                """,
                (user_id,) * 7,
            ).fetchone()
            if any(active_other[key] for key in active_other.keys()):
                connection.rollback()
                raise ValueError(
                    "你已有一个 AI 任务正在排队或运行，请等待其完成"
                )
            connection.execute(
                """
                INSERT INTO novel_causal_branch_simulations(
                    id, source_suggestion_id, project_id, user_id,
                    proposal_index, horizon_chapter_count, instruction,
                    provider, model, credential_source, status,
                    source_proposal_signature, baseline_fingerprint,
                    context_snapshot_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued',
                          ?, ?, ?, ?)
                """,
                (
                    simulation_id,
                    suggestion_id,
                    source["project_id"],
                    user_id,
                    proposal_index,
                    horizon_chapter_count,
                    clean_instruction,
                    provider,
                    model,
                    credential_source,
                    context["source_proposal_signature"],
                    _fingerprint(context),
                    _json(context),
                    now,
                ),
            )
            connection.commit()
        return simulation_id

    def list_for_suggestion(
        self,
        *,
        user_id: int,
        suggestion_id: str,
        limit: int = 30,
    ) -> List[Dict[str, Any]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT simulation.*
                FROM novel_causal_branch_simulations simulation
                JOIN novel_projects project
                  ON project.id=simulation.project_id
                WHERE simulation.source_suggestion_id=?
                  AND project.user_id=?
                ORDER BY simulation.created_at DESC, simulation.rowid DESC
                LIMIT ?
                """,
                (
                    suggestion_id,
                    user_id,
                    max(1, min(limit, 100)),
                ),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_simulation(
        self,
        *,
        user_id: int,
        simulation_id: str,
    ) -> Optional[Dict[str, Any]]:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT simulation.*, project.title AS project_title,
                       source.instruction AS source_instruction,
                       source.chapter_limit AS source_chapter_limit
                FROM novel_causal_branch_simulations simulation
                JOIN novel_projects project
                  ON project.id=simulation.project_id
                JOIN novel_causal_link_suggestions source
                  ON source.id=simulation.source_suggestion_id
                WHERE simulation.id=? AND project.user_id=?
                """,
                (simulation_id, user_id),
            ).fetchone()
            if not row:
                return None
            item = dict(row)
            frozen = _load_json(item.pop("context_snapshot_json"), {})
            item["context_snapshot"] = frozen
            item["selected_proposal"] = dict(
                frozen.get("selected_proposal") or {}
            )
            chapters = {
                str(chapter.get("id") or ""): dict(chapter)
                for chapter in (
                    frozen.get("allowed_source_chapters") or []
                )
                if str(chapter.get("id") or "")
            }
            item["source_chapter"] = chapters.get(
                str(
                    item["selected_proposal"].get(
                        "source_chapter_id"
                    )
                    or ""
                ),
                {},
            )
            item["target_chapter"] = chapters.get(
                str(
                    item["selected_proposal"].get(
                        "target_chapter_id"
                    )
                    or ""
                ),
                {},
            )
            item["result"] = None
            item["result_error"] = ""
            raw_result = item.pop("result_json")
            if raw_result:
                try:
                    result = CausalBranchSimulationSet.model_validate_json(
                        str(raw_result)
                    )
                    result.ensure_context_compatible(frozen)
                    item["result"] = self._decorate_result(
                        result=result.model_dump(mode="json"),
                        context=frozen,
                    )
                except ValueError:
                    item["result_error"] = (
                        "已保存的长期因果推演损坏或不再符合冻结上下文"
                    )
            current_context: Optional[Dict[str, Any]] = None
            source_baseline_current = False
            try:
                source, proposal = self._load_source_proposal(
                    connection,
                    user_id=user_id,
                    suggestion_id=str(item["source_suggestion_id"]),
                    proposal_index=int(item["proposal_index"]),
                    require_pending=False,
                )
                source_baseline_current = self._source_baseline_is_current(
                    connection,
                    user_id=user_id,
                    source=source,
                )
                current_context = self._build_simulation_context(
                    connection,
                    user_id=user_id,
                    source=source,
                    proposal=proposal,
                    proposal_index=int(item["proposal_index"]),
                    horizon_chapter_count=int(
                        item["horizon_chapter_count"]
                    ),
                )
            except ValueError:
                current_context = None
            review = connection.execute(
                """
                SELECT review.decision, review.decided_at,
                       review.causal_link_id,
                       link.status AS causal_link_status
                FROM novel_causal_link_suggestion_reviews review
                LEFT JOIN novel_chapter_causal_links link
                  ON link.id=review.causal_link_id
                WHERE review.suggestion_id=? AND review.proposal_index=?
                """,
                (
                    item["source_suggestion_id"],
                    item["proposal_index"],
                ),
            ).fetchone()
        item["baseline_changed"] = bool(
            current_context is None
            or _fingerprint(current_context)
            != str(item["baseline_fingerprint"])
        )
        item["source_review"] = dict(review) if review else None
        item["unexpected_baseline_changed"] = not source_baseline_current
        item["adoption_ready"] = bool(
            item["result"]
            and item["source_review"]
            and item["source_review"]["decision"] == "accepted"
            and item["source_review"]["causal_link_status"] == "active"
            and source_baseline_current
        )
        return item

    @staticmethod
    def _decorate_result(
        *,
        result: Dict[str, Any],
        context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        chapters = {
            str(item.get("id") or ""): dict(item)
            for item in (context.get("future_chapters") or [])
            if str(item.get("id") or "")
        }
        arcs = {
            str(item.get("id") or ""): dict(item)
            for item in (
                context.get("confirmed_planned_plot_arcs") or []
            )
            if str(item.get("id") or "")
        }
        evidence = {
            str(item.get("id") or ""): dict(item)
            for item in (context.get("evidence_catalog") or [])
            if str(item.get("id") or "")
        }
        for branch in result.get("branches") or []:
            branch["branch_label"] = BRANCH_KEY_LABELS.get(
                str(branch.get("branch_key") or ""),
                str(branch.get("label") or ""),
            )
            branch["intervention_label"] = (
                BRANCH_INTERVENTION_LABELS.get(
                    str(branch.get("intervention_level") or ""),
                    str(branch.get("intervention_level") or ""),
                )
            )
            for impact in branch.get("chapter_impacts") or []:
                impact["impact_label"] = (
                    BRANCH_CHAPTER_IMPACT_LABELS.get(
                        str(impact.get("impact_type") or ""),
                        str(impact.get("impact_type") or ""),
                    )
                )
                impact["evidence_status_label"] = (
                    BRANCH_EVIDENCE_STATUS_LABELS.get(
                        str(impact.get("evidence_status") or ""),
                        str(impact.get("evidence_status") or ""),
                    )
                )
                impact["chapter"] = chapters.get(
                    str(impact.get("chapter_id") or ""),
                    {},
                )
                impact["evidence"] = [
                    evidence[ref]
                    for ref in impact.get("evidence_refs") or []
                    if ref in evidence
                ]
            for trajectory in branch.get("arc_trajectories") or []:
                trajectory["trajectory_label"] = (
                    BRANCH_ARC_TRAJECTORY_LABELS.get(
                        str(trajectory.get("trajectory") or ""),
                        str(trajectory.get("trajectory") or ""),
                    )
                )
                trajectory["arc"] = arcs.get(
                    str(trajectory.get("arc_id") or ""),
                    {},
                )
                trajectory["affected_chapters"] = [
                    chapters[chapter_id]
                    for chapter_id in (
                        trajectory.get("affected_chapter_ids") or []
                    )
                    if chapter_id in chapters
                ]
                trajectory["evidence"] = [
                    evidence[ref]
                    for ref in trajectory.get("evidence_refs") or []
                    if ref in evidence
                ]
            for transfer in branch.get("knowledge_transfers") or []:
                transfer["evidence_status_label"] = (
                    BRANCH_EVIDENCE_STATUS_LABELS.get(
                        str(transfer.get("evidence_status") or ""),
                        str(transfer.get("evidence_status") or ""),
                    )
                )
                transfer["chapter"] = chapters.get(
                    str(transfer.get("chapter_id") or ""),
                    {},
                )
                transfer["evidence"] = [
                    evidence[ref]
                    for ref in transfer.get("evidence_refs") or []
                    if ref in evidence
                ]
            for payoff in branch.get("payoff_impacts") or []:
                payoff["timing_label"] = BRANCH_PAYOFF_TIMING_LABELS.get(
                    str(payoff.get("timing") or ""),
                    str(payoff.get("timing") or ""),
                )
                payoff["delivery_chapter"] = chapters.get(
                    str(payoff.get("delivery_chapter_id") or ""),
                    {},
                )
                payoff["evidence"] = [
                    evidence[ref]
                    for ref in payoff.get("evidence_refs") or []
                    if ref in evidence
                ]
        return result

    def get_status(
        self,
        *,
        user_id: int,
        simulation_id: str,
    ) -> Optional[str]:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT simulation.status
                FROM novel_causal_branch_simulations simulation
                JOIN novel_projects project
                  ON project.id=simulation.project_id
                WHERE simulation.id=? AND project.user_id=?
                """,
                (simulation_id, user_id),
            ).fetchone()
        return str(row["status"]) if row else None

    def claim_next_simulation(self) -> Optional[Dict[str, Any]]:
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = utc_now()
            connection.execute(
                """
                UPDATE novel_causal_branch_simulations
                SET status='queued', started_at=NULL, claim_token=NULL,
                    lease_expires_at=NULL,
                    error='上一次长期因果推演租约已过期，已自动重新排队'
                WHERE status='running' AND lease_expires_at IS NOT NULL
                  AND lease_expires_at<=?
                """,
                (now,),
            )
            row = connection.execute(
                """
                SELECT simulation.*
                FROM novel_causal_branch_simulations simulation
                JOIN novel_projects project
                  ON project.id=simulation.project_id
                WHERE simulation.status='queued'
                ORDER BY simulation.created_at
                LIMIT 1
                """
            ).fetchone()
            if not row:
                connection.commit()
                return None
            claim_token = uuid.uuid4().hex
            cursor = connection.execute(
                """
                UPDATE novel_causal_branch_simulations
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
            result.pop("context_snapshot_json"),
            {},
        )
        return result

    def complete_simulation(
        self,
        *,
        simulation_id: str,
        claim_token: str,
        result: CausalBranchSimulationSet,
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
                FROM novel_causal_branch_simulations
                WHERE id=? AND status='running' AND claim_token=?
                """,
                (simulation_id, claim_token),
            ).fetchone()
            if not row:
                return False
            context = _load_json(row["context_snapshot_json"], {})
            result.ensure_context_compatible(context)
            cursor = connection.execute(
                """
                UPDATE novel_causal_branch_simulations
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
                    simulation_id,
                    claim_token,
                ),
            )
            connection.commit()
        return cursor.rowcount == 1

    def fail_simulation(
        self,
        *,
        simulation_id: str,
        claim_token: str,
        error: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> bool:
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE novel_causal_branch_simulations
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
                    simulation_id,
                    claim_token,
                ),
            )
            connection.commit()
        return cursor.rowcount == 1

    def release_claim(
        self,
        simulation_id: str,
        claim_token: str,
        error: str,
    ) -> bool:
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE novel_causal_branch_simulations
                SET status='queued', started_at=NULL, claim_token=NULL,
                    lease_expires_at=NULL, error=?
                WHERE id=? AND status='running' AND claim_token=?
                """,
                (error[:2000], simulation_id, claim_token),
            )
            connection.commit()
        return cursor.rowcount == 1
