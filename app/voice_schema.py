from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


VoiceDimension = Literal[
    "narration",
    "rhythm",
    "dialogue",
    "sensory",
    "metaphor",
    "omission",
]


class VoiceEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    dimension: VoiceDimension
    quote: str = Field(min_length=2, max_length=120)
    observation: str = Field(min_length=4, max_length=500)


class VoiceProfileSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    summary: str = Field(min_length=8, max_length=1000)
    narration_rules: str = Field(default="", max_length=2000)
    sentence_rhythm: str = Field(default="", max_length=1600)
    dialogue_voice: str = Field(default="", max_length=2000)
    sensory_palette: str = Field(default="", max_length=1600)
    metaphor_policy: str = Field(default="", max_length=1600)
    allowed_omissions: str = Field(default="", max_length=1600)
    preferred_patterns: list[str] = Field(
        default_factory=list, max_length=15
    )
    banned_expressions: list[str] = Field(
        default_factory=list, max_length=15
    )
    evidence: list[VoiceEvidence] = Field(min_length=2, max_length=12)
    uncertainties: list[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def require_executable_rules(self) -> "VoiceProfileSuggestion":
        rules = (
            self.narration_rules,
            self.sentence_rhythm,
            self.dialogue_voice,
            self.sensory_palette,
            self.metaphor_policy,
            self.allowed_omissions,
        )
        if sum(bool(rule.strip()) for rule in rules) < 2:
            raise ValueError("至少需要两个有样章依据的可执行声纹维度")
        if any(len(item) > 240 for item in self.preferred_patterns):
            raise ValueError("保留表达规则不能超过 240 个字符")
        if any(len(item) > 120 for item in self.banned_expressions):
            raise ValueError("禁用表达不能超过 120 个字符")
        if any(len(item) > 500 for item in self.uncertainties):
            raise ValueError("不确定项不能超过 500 个字符")
        return self
