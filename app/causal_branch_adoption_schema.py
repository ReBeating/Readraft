from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .causal_branch_schema import (
    CausalBranchChapterImpactType,
    CausalBranchEvidenceStatus,
)


class StrictCausalBranchAdoptionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


def _ensure_unique(values: List[str], label: str) -> None:
    normalized = [value.casefold() for value in values]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label}不能重复")


class CausalBranchTaskPatch(StrictCausalBranchAdoptionModel):
    """Author-reviewable additions to one future chapter task card."""

    must_happen: List[str] = Field(min_length=1, max_length=12)
    plot_threads: List[str] = Field(default_factory=list, max_length=10)
    foreshadow_setup: List[str] = Field(default_factory=list, max_length=10)
    foreshadow_payoff: List[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def validate_patch(self) -> "CausalBranchTaskPatch":
        for label, values in (
            ("本章必须落实", self.must_happen),
            ("推进剧情线", self.plot_threads),
            ("伏笔铺设", self.foreshadow_setup),
            ("伏笔回收", self.foreshadow_payoff),
        ):
            _ensure_unique(values, label)
            for value in values:
                if len(value) > 1200:
                    raise ValueError(f"{label}中每条不能超过 1,200 个字符")
        return self


class CausalBranchAdoptionItemSnapshot(
    StrictCausalBranchAdoptionModel
):
    """Immutable source material behind one review item."""

    item_key: str = Field(min_length=3, max_length=180)
    chapter_id: str = Field(min_length=1, max_length=120)
    chapter_position: int = Field(ge=1)
    chapter_title: str = Field(min_length=1, max_length=300)
    impact_type: CausalBranchChapterImpactType
    planned_change: str = Field(min_length=8, max_length=1400)
    causal_role: str = Field(min_length=8, max_length=1200)
    evidence_status: CausalBranchEvidenceStatus
    evidence_refs: List[str] = Field(min_length=1, max_length=6)
    unresolved_assumption: str = Field(default="", max_length=1000)
    rationale: str = Field(min_length=8, max_length=2400)
    proposed_patch: CausalBranchTaskPatch

    @model_validator(mode="after")
    def validate_item(self) -> "CausalBranchAdoptionItemSnapshot":
        _ensure_unique(self.evidence_refs, "落地项证据引用")
        return self
