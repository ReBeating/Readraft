"""Validated intent-routing messages for the Agent runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

class AssistantIntentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    intent: Literal[
        "discuss",
        "analyze_work",
        "update_settings",
        "plan_story",
        "draft_prose",
        "draft_new_chapter",
        "revise_prose",
    ]
    workflow: list[
        Literal[
            "discuss",
            "analyze_work",
            "update_settings",
            "plan_story",
            "draft_prose",
            "draft_new_chapter",
            "revise_prose",
        ]
    ] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    target_chapter_id: Optional[str] = Field(
        default=None, min_length=1, max_length=80
    )
    reason: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_workflow(self) -> "AssistantIntentDecision":
        if not self.workflow:
            self.workflow = [self.intent]
            return self
        if self.workflow[0] != self.intent:
            raise ValueError("workflow 的第一项必须等于本轮 intent")
        if len(set(self.workflow)) != len(self.workflow):
            raise ValueError("workflow 不能重复同一任务")
        return self


@dataclass(frozen=True)
class AssistantIntentResponse:
    decision: AssistantIntentDecision
    raw_response: str
    input_tokens: int
    output_tokens: int
