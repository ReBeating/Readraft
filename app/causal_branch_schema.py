from __future__ import annotations

from typing import Any, List, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator


CausalBranchKey = Literal[
    "minimal_change",
    "distributed_consequences",
    "stress_test",
]
CausalBranchEvidenceStatus = Literal[
    "supported",
    "uncertain",
    "conflict",
]
CausalBranchInterventionLevel = Literal["low", "medium", "high"]
CausalBranchChapterImpactType = Literal[
    "setup",
    "escalation",
    "information_transfer",
    "choice",
    "reversal",
    "payoff",
    "repair",
]
CausalBranchArcTrajectoryType = Literal[
    "advances",
    "complicates",
    "delays",
    "pays_off",
    "risks_breaking",
]
CausalBranchPayoffTiming = Literal[
    "unchanged",
    "earlier",
    "later",
    "reframed",
    "at_risk",
]


class StrictCausalBranchModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


def _ensure_unique(values: List[str], label: str) -> None:
    normalized = [value.casefold() for value in values]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label}不能重复")


class CausalBranchChapterImpact(StrictCausalBranchModel):
    chapter_id: str = Field(min_length=1, max_length=120)
    chapter_position: int = Field(ge=1)
    chapter_title: str = Field(min_length=1, max_length=300)
    impact_type: CausalBranchChapterImpactType
    planned_change: str = Field(min_length=8, max_length=1400)
    causal_role: str = Field(min_length=8, max_length=1200)
    evidence_status: CausalBranchEvidenceStatus
    evidence_refs: List[str] = Field(min_length=1, max_length=6)
    unresolved_assumption: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def validate_impact(self) -> "CausalBranchChapterImpact":
        _ensure_unique(self.evidence_refs, "章节影响证据引用")
        if (
            self.evidence_status in {"uncertain", "conflict"}
            and len(self.unresolved_assumption) < 8
        ):
            raise ValueError(
                "证据不足或冲突的章节影响必须说明未决假设"
            )
        return self


class CausalBranchArcTrajectory(StrictCausalBranchModel):
    arc_id: str = Field(min_length=1, max_length=120)
    arc_title: str = Field(min_length=1, max_length=240)
    trajectory: CausalBranchArcTrajectoryType
    before_state: str = Field(min_length=4, max_length=1200)
    after_state: str = Field(min_length=8, max_length=1400)
    affected_chapter_ids: List[str] = Field(min_length=1, max_length=8)
    evidence_refs: List[str] = Field(min_length=1, max_length=6)
    uncertainty: str = Field(min_length=8, max_length=1000)

    @model_validator(mode="after")
    def validate_trajectory(self) -> "CausalBranchArcTrajectory":
        _ensure_unique(self.affected_chapter_ids, "剧情线影响章节")
        _ensure_unique(self.evidence_refs, "剧情线影响证据引用")
        return self


class CausalBranchKnowledgeTransfer(StrictCausalBranchModel):
    character_name: str = Field(min_length=1, max_length=160)
    chapter_id: str = Field(min_length=1, max_length=120)
    before_state: str = Field(min_length=4, max_length=1000)
    after_state: str = Field(min_length=8, max_length=1200)
    channel: str = Field(min_length=4, max_length=800)
    evidence_status: CausalBranchEvidenceStatus
    evidence_refs: List[str] = Field(min_length=1, max_length=6)
    risk: str = Field(min_length=8, max_length=1000)

    @model_validator(mode="after")
    def validate_transfer(self) -> "CausalBranchKnowledgeTransfer":
        _ensure_unique(self.evidence_refs, "人物知情证据引用")
        return self


class CausalBranchPayoffImpact(StrictCausalBranchModel):
    payoff_ref: str = Field(min_length=1, max_length=240)
    payoff_text: str = Field(min_length=2, max_length=800)
    timing: CausalBranchPayoffTiming
    delivery_chapter_id: str = Field(min_length=1, max_length=120)
    consequence: str = Field(min_length=8, max_length=1200)
    evidence_refs: List[str] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def validate_payoff(self) -> "CausalBranchPayoffImpact":
        _ensure_unique(self.evidence_refs, "阶段回报证据引用")
        if self.payoff_ref not in self.evidence_refs:
            raise ValueError("阶段回报本身必须出现在证据引用中")
        return self


class CausalBranchOption(StrictCausalBranchModel):
    branch_key: CausalBranchKey
    label: str = Field(min_length=2, max_length=120)
    premise: str = Field(min_length=12, max_length=1800)
    intervention_level: CausalBranchInterventionLevel
    reader_experience: str = Field(min_length=12, max_length=1400)
    tradeoffs: List[str] = Field(min_length=2, max_length=5)
    chapter_impacts: List[CausalBranchChapterImpact] = Field(
        min_length=3,
        max_length=12,
    )
    arc_trajectories: List[CausalBranchArcTrajectory] = Field(
        default_factory=list,
        max_length=10,
    )
    knowledge_strategy: str = Field(min_length=12, max_length=1400)
    knowledge_transfers: List[CausalBranchKnowledgeTransfer] = Field(
        default_factory=list,
        max_length=8,
    )
    payoff_impacts: List[CausalBranchPayoffImpact] = Field(
        default_factory=list,
        max_length=8,
    )
    failure_conditions: List[str] = Field(min_length=1, max_length=5)
    overall_risk: str = Field(min_length=12, max_length=1400)

    @model_validator(mode="after")
    def validate_option(self) -> "CausalBranchOption":
        _ensure_unique(self.tradeoffs, "分支取舍")
        _ensure_unique(self.failure_conditions, "分支失效条件")
        chapter_ids = [
            impact.chapter_id for impact in self.chapter_impacts
        ]
        _ensure_unique(chapter_ids, "分支章节影响")
        positions = [
            impact.chapter_position for impact in self.chapter_impacts
        ]
        if positions != sorted(positions):
            raise ValueError("分支章节影响必须按章节位置升序排列")
        arc_ids = [impact.arc_id for impact in self.arc_trajectories]
        _ensure_unique(arc_ids, "分支剧情线推演")
        transfer_signatures = [
            f"{item.character_name.casefold()}::{item.chapter_id}"
            for item in self.knowledge_transfers
        ]
        _ensure_unique(transfer_signatures, "人物知情传递")
        payoff_refs = [item.payoff_ref for item in self.payoff_impacts]
        _ensure_unique(payoff_refs, "分支阶段回报")
        return self


class CausalBranchSimulationSet(StrictCausalBranchModel):
    analysis_summary: str = Field(min_length=20, max_length=2800)
    horizon_chapter_count: int = Field(ge=10, le=30)
    comparison_summary: str = Field(min_length=20, max_length=2200)
    shared_constraints: List[str] = Field(min_length=1, max_length=8)
    branches: List[CausalBranchOption] = Field(min_length=3, max_length=3)
    unresolved_gaps: List[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def validate_result(self) -> "CausalBranchSimulationSet":
        required_keys = {
            "minimal_change",
            "distributed_consequences",
            "stress_test",
        }
        keys = [branch.branch_key for branch in self.branches]
        if set(keys) != required_keys or len(keys) != len(required_keys):
            raise ValueError("长期因果推演必须恰好包含三种指定分支")
        if keys != [
            "minimal_change",
            "distributed_consequences",
            "stress_test",
        ]:
            raise ValueError(
                "三种分支必须按最小改动、分散后果、压力测试排序"
            )
        _ensure_unique(
            [branch.label for branch in self.branches],
            "分支名称",
        )
        _ensure_unique(
            [branch.premise for branch in self.branches],
            "分支前提",
        )
        _ensure_unique(self.shared_constraints, "共同约束")
        _ensure_unique(self.unresolved_gaps, "未决信息")
        impact_signatures = [
            "|".join(
                f"{item.chapter_id}:{item.planned_change.casefold()}"
                for item in branch.chapter_impacts
            )
            for branch in self.branches
        ]
        _ensure_unique(impact_signatures, "三种分支的章节变化")
        return self

    def ensure_context_compatible(
        self,
        context: Mapping[str, Any],
    ) -> None:
        future = {
            str(item.get("id") or ""): dict(item)
            for item in (context.get("future_chapters") or [])
            if str(item.get("id") or "")
        }
        if self.horizon_chapter_count != int(
            context.get("horizon_chapter_count") or 0
        ):
            raise ValueError("推演结果的章节范围与冻结任务不一致")
        if len(future) != self.horizon_chapter_count:
            raise ValueError("冻结未来章节数量与推演范围不一致")
        selected = dict(context.get("selected_proposal") or {})
        target_id = str(selected.get("target_chapter_id") or "")
        if target_id not in future:
            raise ValueError("冻结因果候选的结果章不在推演范围内")
        evidence = {
            str(item.get("id") or ""): dict(item)
            for item in (context.get("evidence_catalog") or [])
            if str(item.get("id") or "")
        }
        if not evidence:
            raise ValueError("长期因果推演缺少冻结证据目录")
        arcs = {
            str(item.get("id") or ""): dict(item)
            for item in (
                context.get("confirmed_planned_plot_arcs") or []
            )
            if str(item.get("id") or "")
        }
        character_names = {
            str(item.get("name") or "")
            for item in (context.get("characters") or [])
            if str(item.get("name") or "")
        }
        payoff_by_ref = {
            str(item.get("id") or ""): str(item.get("value") or "")
            for item in (context.get("evidence_catalog") or [])
            if str(item.get("kind") or "") == "blueprint_payoff"
            and str(item.get("id") or "")
        }
        required_arc_ids = {
            str(item.get("arc_id") or "")
            for item in (selected.get("arc_impacts") or [])
            if str(item.get("arc_id") or "")
        }
        for branch in self.branches:
            branch_chapter_ids = {
                impact.chapter_id for impact in branch.chapter_impacts
            }
            if target_id not in branch_chapter_ids:
                raise ValueError(
                    "每种分支都必须明确推演所选因果候选的结果章"
                )
            for impact in branch.chapter_impacts:
                chapter = future.get(impact.chapter_id)
                if chapter is None:
                    raise ValueError(
                        "章节影响引用了冻结范围之外的章节："
                        + impact.chapter_id
                    )
                if (
                    impact.chapter_position
                    != int(chapter.get("position") or 0)
                    or impact.chapter_title
                    != str(chapter.get("title") or "")
                ):
                    raise ValueError("章节影响的标题或位置与冻结资料不一致")
                self._ensure_evidence_refs(
                    impact.evidence_refs,
                    evidence,
                    "章节影响",
                )
            covered_arc_ids: set[str] = set()
            for trajectory in branch.arc_trajectories:
                arc = arcs.get(trajectory.arc_id)
                if arc is None or trajectory.arc_title != str(
                    arc.get("title") or ""
                ):
                    raise ValueError(
                        "剧情线推演引用了未确认或标题不匹配的剧情线"
                    )
                covered_arc_ids.add(trajectory.arc_id)
                self._ensure_evidence_refs(
                    trajectory.evidence_refs,
                    evidence,
                    "剧情线推演",
                )
                positions = []
                for chapter_id in trajectory.affected_chapter_ids:
                    chapter = future.get(chapter_id)
                    if chapter is None:
                        raise ValueError(
                            "剧情线推演引用了冻结范围之外的章节"
                        )
                    positions.append(int(chapter.get("position") or 0))
                if positions != sorted(positions):
                    raise ValueError(
                        "剧情线受影响章节必须按章节位置升序排列"
                    )
            missing_arc_ids = required_arc_ids - covered_arc_ids
            if missing_arc_ids:
                raise ValueError(
                    "分支没有覆盖因果候选已经识别的剧情线影响："
                    + sorted(missing_arc_ids)[0]
                )
            for transfer in branch.knowledge_transfers:
                if transfer.character_name not in character_names:
                    raise ValueError(
                        "人物知情推演引用了项目之外的人物："
                        + transfer.character_name
                    )
                if transfer.chapter_id not in future:
                    raise ValueError(
                        "人物知情推演引用了冻结范围之外的章节"
                    )
                self._ensure_evidence_refs(
                    transfer.evidence_refs,
                    evidence,
                    "人物知情推演",
                )
            if character_names and not branch.knowledge_transfers:
                raise ValueError(
                    "项目存在人物时，每种分支至少要落一条人物知情变化"
                )
            for payoff in branch.payoff_impacts:
                if payoff.payoff_ref not in payoff_by_ref:
                    raise ValueError(
                        "阶段回报引用了冻结蓝图之外的兑现项"
                    )
                if payoff.payoff_text != payoff_by_ref[payoff.payoff_ref]:
                    raise ValueError("阶段回报文本与冻结蓝图不一致")
                if payoff.delivery_chapter_id not in future:
                    raise ValueError(
                        "阶段回报引用了冻结范围之外的兑现章节"
                    )
                self._ensure_evidence_refs(
                    payoff.evidence_refs,
                    evidence,
                    "阶段回报推演",
                )
            if payoff_by_ref and not branch.payoff_impacts:
                raise ValueError(
                    "蓝图存在必须兑现项时，每种分支至少要评估一项"
                )

    @staticmethod
    def _ensure_evidence_refs(
        refs: List[str],
        evidence: Mapping[str, Any],
        label: str,
    ) -> None:
        missing = [ref for ref in refs if ref not in evidence]
        if missing:
            raise ValueError(
                f"{label}引用了冻结目录之外的证据：" + missing[0]
            )
