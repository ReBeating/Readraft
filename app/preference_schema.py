from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


EditPreferenceCategory = Literal[
    "diction",
    "sentence_rhythm",
    "narration_distance",
    "dialogue",
    "emotional_expression",
    "sensory_detail",
    "metaphor",
    "omission",
    "paragraph_structure",
    "other",
]


class EditPreferenceCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    category: EditPreferenceCategory
    guidance: str = Field(min_length=8, max_length=600)
    applicability: str = Field(min_length=4, max_length=500)
    before_quote: str = Field(default="", max_length=240)
    after_quote: str = Field(default="", max_length=240)
    rationale: str = Field(min_length=4, max_length=500)

    @model_validator(mode="after")
    def require_changed_evidence(self) -> "EditPreferenceCandidate":
        if not self.before_quote and not self.after_quote:
            raise ValueError("至少需要一段修改前或修改后的逐字证据")
        if (
            self.before_quote
            and self.after_quote
            and self.before_quote == self.after_quote
        ):
            raise ValueError("修改前后证据不能完全相同")
        return self


class EditPreferenceSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    summary: str = Field(min_length=8, max_length=1000)
    preferences: list[EditPreferenceCandidate] = Field(
        min_length=1, max_length=6
    )
    uncertainties: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def validate_uncertainties(self) -> "EditPreferenceSuggestion":
        if any(len(item) > 500 for item in self.uncertainties):
            raise ValueError("不确定项不能超过 500 个字符")
        return self
