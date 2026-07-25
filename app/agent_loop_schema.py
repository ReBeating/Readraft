from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from .assistant_chat_schema import AssistantCitationProposal


class AgentToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=100)
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentLoopDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: Literal["call_tool", "finish"]
    tool_call: Optional[AgentToolCall] = None
    answer: Optional[str] = Field(default=None, min_length=1, max_length=30_000)
    citations: list[AssistantCitationProposal] = Field(
        default_factory=list, max_length=8
    )

    @model_validator(mode="after")
    def validate_action_payload(self) -> "AgentLoopDecision":
        if self.action == "call_tool":
            if self.tool_call is None:
                raise ValueError("call_tool 必须包含 tool_call")
            if self.answer is not None or self.citations:
                raise ValueError("call_tool 不能同时返回最终回答")
        else:
            if self.tool_call is not None:
                raise ValueError("finish 不能包含 tool_call")
            if not self.answer:
                raise ValueError("finish 必须包含 answer")
        return self


class AssistantIntentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    intent: Literal[
        "discuss",
        "analyze_work",
        "update_settings",
        "plan_story",
        "draft_prose",
        "revise_prose",
    ]
    workflow: list[
        Literal[
            "discuss",
            "analyze_work",
            "update_settings",
            "plan_story",
            "draft_prose",
            "revise_prose",
        ]
    ] = Field(default_factory=list, max_length=4)
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
class AgentDecisionResponse:
    decision: AgentLoopDecision
    raw_response: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class AssistantIntentResponse:
    decision: AssistantIntentDecision
    raw_response: str
    input_tokens: int
    output_tokens: int
