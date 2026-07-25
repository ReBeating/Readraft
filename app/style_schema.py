from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


StyleIssueType = Literal[
    "abstract_emotion",
    "over_explanation",
    "uniform_rhythm",
    "generic_atmosphere",
    "cliche",
    "dialogue_convergence",
    "over_complete_paragraph",
    "unnecessary_summary",
    "repetition",
    "non_specific_detail",
]
StyleSeverity = Literal["low", "medium", "high"]


class StyleIssueProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    paragraph_index: int = Field(ge=1, le=1000)
    quote: str = Field(min_length=2, max_length=800)
    issue_type: StyleIssueType
    severity: StyleSeverity
    evidence: str = Field(min_length=2, max_length=600)
    reader_impact: str = Field(min_length=2, max_length=600)
    rewrite_direction: str = Field(min_length=2, max_length=600)


class StyleAuditResult(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    summary: str = Field(min_length=2, max_length=1000)
    issues: list[StyleIssueProposal] = Field(
        default_factory=list, max_length=12
    )


class RewriteAlternative(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    replacement_text: str = Field(min_length=1, max_length=4000)
    rationale: str = Field(min_length=2, max_length=500)


class TargetedRewriteResult(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    alternatives: list[RewriteAlternative] = Field(
        min_length=2, max_length=2
    )

    @model_validator(mode="after")
    def alternatives_must_differ(self) -> "TargetedRewriteResult":
        first, second = self.alternatives
        if first.replacement_text == second.replacement_text:
            raise ValueError("两个定点改写候选不能相同")
        return self
