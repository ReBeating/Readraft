from __future__ import annotations

import json
from typing import Any, List, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator


CausalSuggestionConfidence = Literal["high", "medium", "low"]
CausalSuggestionRelation = Literal[
    "causes",
    "enables",
    "complicates",
    "pays_off",
]
CausalSuggestionBridgeReadiness = Literal[
    "direct",
    "needs_intermediate_steps",
]
CausalExplanationCompatibility = Literal[
    "exclusive",
    "can_coexist",
    "uncertain",
]
CausalSemanticCheckCategory = Literal[
    "canon_consistency",
    "character_knowledge",
    "timeline",
    "world_rules",
    "continuity",
]
CausalSemanticCheckStatus = Literal["supported", "uncertain", "conflict"]
CausalArcImpactType = Literal[
    "advances",
    "complicates",
    "delays",
    "pays_off",
    "risks_breaking",
]


class StrictCausalSuggestionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CausalSemanticCheck(StrictCausalSuggestionModel):
    category: CausalSemanticCheckCategory
    status: CausalSemanticCheckStatus
    finding: str = Field(min_length=8, max_length=1200)
    evidence_refs: List[str] = Field(min_length=1, max_length=6)
    required_resolution: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def validate_check(self) -> "CausalSemanticCheck":
        if len(set(self.evidence_refs)) != len(self.evidence_refs):
            raise ValueError("同一语义检查不能重复引用证据")
        if any(len(value) > 240 for value in self.evidence_refs):
            raise ValueError("语义检查证据 ID 不能超过 240 个字符")
        if self.status == "conflict" and len(self.required_resolution) < 8:
            raise ValueError("语义冲突必须说明作者需要解决什么")
        return self


class CausalArcImpact(StrictCausalSuggestionModel):
    arc_id: str = Field(min_length=1, max_length=120)
    arc_title: str = Field(min_length=1, max_length=240)
    impact_type: CausalArcImpactType
    before_state: str = Field(min_length=4, max_length=1000)
    after_state: str = Field(min_length=4, max_length=1000)
    evidence_refs: List[str] = Field(min_length=1, max_length=6)
    required_support_chapter_ids: List[str] = Field(
        default_factory=list,
        max_length=4,
    )
    risk: str = Field(min_length=8, max_length=1000)

    @model_validator(mode="after")
    def validate_impact(self) -> "CausalArcImpact":
        if len(set(self.evidence_refs)) != len(self.evidence_refs):
            raise ValueError("剧情线影响不能重复引用证据")
        if len(set(self.required_support_chapter_ids)) != len(
            self.required_support_chapter_ids
        ):
            raise ValueError("剧情线影响不能重复引用支持章节")
        return self


class ProposedCausalLink(StrictCausalSuggestionModel):
    source_chapter_id: str = Field(min_length=1, max_length=120)
    target_chapter_id: str = Field(min_length=1, max_length=120)
    relation_type: CausalSuggestionRelation
    cause_text: str = Field(min_length=8, max_length=1200)
    effect_text: str = Field(min_length=8, max_length=1200)
    bridge_purpose: str = Field(min_length=8, max_length=1200)
    source_evidence: List[str] = Field(min_length=1, max_length=4)
    target_evidence: List[str] = Field(min_length=1, max_length=4)
    risk_if_omitted: str = Field(min_length=8, max_length=1000)
    confidence: CausalSuggestionConfidence
    alternative_label: str = Field(default="", max_length=160)
    challenge_points: List[str] = Field(default_factory=list, max_length=4)
    missing_intermediate_steps: List[str] = Field(
        default_factory=list,
        max_length=4,
    )
    disconfirmation_test: str = Field(default="", max_length=1000)
    bridge_readiness: CausalSuggestionBridgeReadiness = "direct"
    semantic_checks: List[CausalSemanticCheck] = Field(
        default_factory=list,
        max_length=5,
    )
    arc_impacts: List[CausalArcImpact] = Field(
        default_factory=list,
        max_length=8,
    )

    @model_validator(mode="after")
    def validate_evidence(self) -> "ProposedCausalLink":
        for label, values in (
            ("起因证据", self.source_evidence),
            ("结果证据", self.target_evidence),
            ("反证或证据缺口", self.challenge_points),
            ("缺失中间步骤", self.missing_intermediate_steps),
        ):
            if any(len(item) > 600 for item in values):
                raise ValueError(f"{label}每项不能超过 600 个字符")
            normalized = [item.casefold() for item in values]
            if len(set(normalized)) != len(normalized):
                raise ValueError(f"{label}不能重复")
        if (
            self.bridge_readiness == "needs_intermediate_steps"
            and not self.missing_intermediate_steps
        ):
            raise ValueError("需要补中间步骤的候选必须列出具体缺口")
        if (
            self.bridge_readiness == "direct"
            and self.missing_intermediate_steps
        ):
            raise ValueError("存在缺失中间步骤时不能标记为直接可用")
        return self


class CausalExplanationGroup(StrictCausalSuggestionModel):
    group_id: str = Field(
        min_length=1,
        max_length=80,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    target_chapter_id: str = Field(min_length=1, max_length=120)
    target_outcome: str = Field(min_length=8, max_length=1200)
    proposal_indices: List[int] = Field(min_length=2, max_length=4)
    compatibility: CausalExplanationCompatibility
    comparison_summary: str = Field(min_length=8, max_length=1200)
    decision_question: str = Field(min_length=8, max_length=800)

    @model_validator(mode="after")
    def validate_indices(self) -> "CausalExplanationGroup":
        if len(set(self.proposal_indices)) != len(self.proposal_indices):
            raise ValueError("同一解释组不能重复引用候选索引")
        if self.proposal_indices != sorted(self.proposal_indices):
            raise ValueError("解释组候选索引必须按升序排列")
        if self.proposal_indices != list(
            range(
                self.proposal_indices[0],
                self.proposal_indices[-1] + 1,
            )
        ):
            raise ValueError("同一解释组的候选索引必须连续")
        if any(index < 0 for index in self.proposal_indices):
            raise ValueError("解释组候选索引不能为负数")
        return self


class CausalSuggestionSet(StrictCausalSuggestionModel):
    analysis_summary: str = Field(min_length=10, max_length=2400)
    proposals: List[ProposedCausalLink] = Field(
        default_factory=list,
        max_length=8,
    )
    comparison_groups: List[CausalExplanationGroup] = Field(
        default_factory=list,
        max_length=3,
    )
    unresolved_gaps: List[str] = Field(default_factory=list, max_length=8)
    no_proposal_reason: str = Field(default="", max_length=1200)

    @model_validator(mode="after")
    def validate_result(self) -> "CausalSuggestionSet":
        if not self.proposals and len(self.no_proposal_reason) < 8:
            raise ValueError(
                "没有安全候选时，必须说明为什么不应强行建立因果链接"
            )
        signatures = [
            (
                item.source_chapter_id,
                item.target_chapter_id,
                item.relation_type,
            )
            for item in self.proposals
        ]
        if len(set(signatures)) != len(signatures):
            raise ValueError("同一批建议不能重复相同章节与关系类型")
        required_semantic_categories = {
            "canon_consistency",
            "character_knowledge",
            "timeline",
            "world_rules",
            "continuity",
        }
        for proposal in self.proposals:
            categories = [
                check.category for check in proposal.semantic_checks
            ]
            if set(categories) != required_semantic_categories or len(
                categories
            ) != len(required_semantic_categories):
                raise ValueError(
                    "每条候选都必须完成五类且不重复的语义检查"
                )
        group_ids = [group.group_id for group in self.comparison_groups]
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("同一批建议的解释组 ID 不能重复")
        group_starts = [
            group.proposal_indices[0] for group in self.comparison_groups
        ]
        if group_starts != sorted(group_starts):
            raise ValueError("解释组必须按候选出现顺序排列")
        grouped_indices: set[int] = set()
        for group in self.comparison_groups:
            if any(index >= len(self.proposals) for index in group.proposal_indices):
                raise ValueError("解释组引用了不存在的候选索引")
            overlap = grouped_indices.intersection(group.proposal_indices)
            if overlap:
                raise ValueError("一条候选不能同时属于多个解释组")
            grouped_indices.update(group.proposal_indices)
            alternatives = [
                self.proposals[index] for index in group.proposal_indices
            ]
            if any(
                proposal.target_chapter_id != group.target_chapter_id
                for proposal in alternatives
            ):
                raise ValueError("同一解释组的候选必须指向同一结果章")
            labels = [proposal.alternative_label for proposal in alternatives]
            if any(len(label) < 2 for label in labels):
                raise ValueError("解释组中的每条候选都必须有可比较的名称")
            if len({label.casefold() for label in labels}) != len(labels):
                raise ValueError("同一解释组的候选名称不能重复")
            if any(not proposal.challenge_points for proposal in alternatives):
                raise ValueError(
                    "解释组中的每条候选都必须列出反证或证据缺口"
                )
            if any(
                len(proposal.disconfirmation_test) < 8
                for proposal in alternatives
            ):
                raise ValueError(
                    "解释组中的每条候选都必须说明如何推翻这项解释"
                )
        if any(len(item) > 600 for item in self.unresolved_gaps):
            raise ValueError("未决观察每项不能超过 600 个字符")
        return self

    def ensure_context_compatible(
        self,
        context: Mapping[str, Any],
    ) -> None:
        current_position = int(
            context.get("current_canonical_position") or 0
        )
        source_chapters = {
            str(item.get("id") or ""): dict(item)
            for item in (context.get("allowed_source_chapters") or [])
            if str(item.get("id") or "")
        }
        target_chapters = {
            str(item.get("id") or ""): dict(item)
            for item in (context.get("future_chapters") or [])
            if str(item.get("id") or "")
        }
        existing = {
            (
                str(item.get("source_chapter_id") or ""),
                str(item.get("target_chapter_id") or ""),
                str(item.get("relation_type") or ""),
            )
            for item in (context.get("active_causal_links") or [])
            if str(item.get("status") or "active") == "active"
        }
        evidence_catalog = {
            str(item.get("id") or ""): dict(item)
            for item in (context.get("evidence_catalog") or [])
            if str(item.get("id") or "")
        }
        confirmed_arcs_by_id = {
            str(item.get("id") or ""): dict(item)
            for item in (
                context.get("confirmed_planned_plot_arcs") or []
            )
            if str(item.get("id") or "")
        }
        confirmed_arcs_by_title = {
            str(item.get("title") or ""): dict(item)
            for item in confirmed_arcs_by_id.values()
            if str(item.get("title") or "")
        }
        if self.proposals and not evidence_catalog:
            raise ValueError("语义复核缺少冻结证据目录")
        for proposal in self.proposals:
            source = source_chapters.get(proposal.source_chapter_id)
            target = target_chapters.get(proposal.target_chapter_id)
            if source is None:
                raise ValueError(
                    "建议引用了冻结上下文之外的起因章节："
                    + proposal.source_chapter_id
                )
            if target is None:
                raise ValueError(
                    "建议引用了冻结上下文之外的结果章节："
                    + proposal.target_chapter_id
                )
            source_position = int(source.get("position") or 0)
            target_position = int(target.get("position") or 0)
            if source_position >= target_position:
                raise ValueError("建议的结果章必须晚于起因章")
            if target_position <= current_position or bool(
                target.get("is_canonical")
            ):
                raise ValueError("建议的结果章必须位于冻结的正史边界之后")
            signature = (
                proposal.source_chapter_id,
                proposal.target_chapter_id,
                proposal.relation_type,
            )
            if signature in existing:
                raise ValueError(
                    "建议重复了已经生效的同类型章节因果链接"
                )
            for check in proposal.semantic_checks:
                missing_refs = [
                    ref
                    for ref in check.evidence_refs
                    if ref not in evidence_catalog
                ]
                if missing_refs:
                    raise ValueError(
                        "语义检查引用了冻结目录之外的证据："
                        + missing_refs[0]
                    )
            involved_arc_titles = {
                str(value)
                for chapter in (source, target)
                for value in (chapter.get("skeleton_arc_titles") or [])
                if str(value).strip()
            }
            impact_arc_ids: set[str] = set()
            for impact in proposal.arc_impacts:
                arc = confirmed_arcs_by_id.get(impact.arc_id)
                if arc is None or str(arc.get("title") or "") != (
                    impact.arc_title
                ):
                    raise ValueError(
                        "剧情线影响引用了未确认或标题不匹配的规划剧情线"
                    )
                if impact.arc_id in impact_arc_ids:
                    raise ValueError("同一候选不能重复推演同一剧情线")
                impact_arc_ids.add(impact.arc_id)
                missing_refs = [
                    ref
                    for ref in impact.evidence_refs
                    if ref not in evidence_catalog
                ]
                if missing_refs:
                    raise ValueError(
                        "剧情线影响引用了冻结目录之外的证据："
                        + missing_refs[0]
                    )
                for chapter_id in impact.required_support_chapter_ids:
                    support = target_chapters.get(chapter_id)
                    if support is None:
                        raise ValueError(
                            "剧情线影响引用了冻结范围之外的支持章节"
                        )
                    support_position = int(support.get("position") or 0)
                    if not source_position < support_position < target_position:
                        raise ValueError(
                            "支持章节必须严格位于起因章和结果章之间"
                        )
            missing_arc_titles = [
                title
                for title in involved_arc_titles
                if title not in confirmed_arcs_by_title
                or str(
                    confirmed_arcs_by_title[title].get("id") or ""
                )
                not in impact_arc_ids
            ]
            if missing_arc_titles:
                raise ValueError(
                    "语义复核没有覆盖候选涉及的确认剧情线："
                    + missing_arc_titles[0]
                )

    def stable_signature(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
