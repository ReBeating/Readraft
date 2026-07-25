from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


AuditCategory = Literal[
    "continuity",
    "character_knowledge",
    "world_rule",
    "must_happen",
    "forbidden",
    "point_of_view",
    "length",
    "state_consistency",
    "scene_beat",
]
AuditSeverity = Literal["hard", "warning"]
CoverageStatus = Literal["met", "unclear", "missing"]


class AuditFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    code: str = Field(min_length=2, max_length=80)
    category: AuditCategory
    severity: AuditSeverity
    location: str = Field(default="", max_length=200)
    evidence: str = Field(default="", max_length=500)
    description: str = Field(min_length=2, max_length=500)
    violated_constraint: str = Field(default="", max_length=500)
    repair_instruction: str = Field(default="", max_length=500)


class CoverageItem(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    requirement: str = Field(min_length=1, max_length=500)
    status: CoverageStatus
    evidence: str = Field(default="", max_length=500)


class HardAuditAnalysis(BaseModel):
    """The model-produced portion of a hard chapter audit."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    summary: str = Field(min_length=2, max_length=1000)
    findings: list[AuditFinding] = Field(default_factory=list, max_length=80)
    must_happen_coverage: list[CoverageItem] = Field(
        default_factory=list, max_length=40
    )
    scene_coverage: list[CoverageItem] = Field(
        default_factory=list, max_length=10
    )


class QualityAuditReport(BaseModel):
    """The persisted, deterministic gate result."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: Literal[1] = 1
    verdict: Literal["pass", "block", "pending"]
    summary: str = Field(min_length=2, max_length=1000)
    effective_char_count: int = Field(ge=0)
    minimum_effective_chars: int = Field(ge=1)
    expansion_attempted: bool = False
    expansion_error: str = Field(default="", max_length=1000)
    audit_error: str = Field(default="", max_length=1000)
    findings: list[AuditFinding] = Field(default_factory=list, max_length=100)
    must_happen_coverage: list[CoverageItem] = Field(
        default_factory=list, max_length=40
    )
    scene_coverage: list[CoverageItem] = Field(
        default_factory=list, max_length=10
    )

    @model_validator(mode="after")
    def validate_verdict(self) -> "QualityAuditReport":
        hard_count = sum(
            finding.severity == "hard" for finding in self.findings
        )
        if self.verdict == "pass" and (
            hard_count or self.audit_error or self.effective_char_count
            < self.minimum_effective_chars
        ):
            raise ValueError("存在硬问题或审计失败时不能标记为通过")
        if self.verdict == "pending" and not self.audit_error:
            raise ValueError("待审计状态必须说明审计未完成原因")
        return self

    @property
    def hard_issue_count(self) -> int:
        return sum(
            finding.severity == "hard" for finding in self.findings
        )

    @property
    def warning_count(self) -> int:
        return sum(
            finding.severity == "warning" for finding in self.findings
        )
