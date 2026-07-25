from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .story_planning_schema import StoryBlueprint


class AssistantCitationProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source_id: str = Field(min_length=1, max_length=160)
    quote: str = Field(min_length=1, max_length=800)
    note: str = Field(default="", max_length=500)


class AssistantRewriteProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    replacement_text: str = Field(min_length=1, max_length=20_000)
    rationale: str = Field(min_length=2, max_length=800)


class AssistantDraftProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    mode: Literal["replace", "append"]
    content: str = Field(min_length=1, max_length=60_000)
    rationale: str = Field(min_length=2, max_length=800)


class AssistantSettingsPatch(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: Optional[str] = Field(default=None, min_length=1, max_length=120)
    genre: Optional[str] = Field(default=None, min_length=1, max_length=80)
    premise: Optional[str] = Field(default=None, min_length=2, max_length=4000)
    story_promise: Optional[str] = Field(
        default=None, min_length=2, max_length=4000
    )
    target_audience: Optional[str] = Field(
        default=None, min_length=1, max_length=1000
    )
    core_appeal: Optional[str] = Field(
        default=None, min_length=2, max_length=4000
    )
    ending_constraint: Optional[str] = Field(
        default=None, min_length=2, max_length=4000
    )
    world_setting: Optional[str] = Field(
        default=None, min_length=2, max_length=20_000
    )
    style_guide: Optional[str] = Field(
        default=None, min_length=2, max_length=10_000
    )
    ai_instructions: Optional[str] = Field(
        default=None, min_length=2, max_length=10_000
    )
    point_of_view: Optional[
        Literal["第三人称限知", "第一人称", "第三人称全知", "多视角"]
    ] = None
    target_chapter_chars: Optional[int] = Field(
        default=None, ge=2000, le=12_000
    )
    planning_horizon: Optional[int] = Field(
        default=None, ge=3, le=50
    )

    @model_validator(mode="after")
    def ensure_not_empty(self) -> "AssistantSettingsPatch":
        if not self.model_dump(exclude_none=True):
            raise ValueError("settings_patch 至少需要包含一个设定字段")
        return self


class AssistantStoryPlanProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    blueprint: StoryBlueprint
    rationale: str = Field(min_length=2, max_length=1200)

    @model_validator(mode="after")
    def ensure_confirmable_blueprint(self) -> "AssistantStoryPlanProposal":
        self.blueprint.ensure_confirmable()
        return self


class AssistantChatResult(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    answer: str = Field(min_length=1, max_length=30_000)
    citations: list[AssistantCitationProposal] = Field(
        default_factory=list, max_length=8
    )
    rewrite: Optional[AssistantRewriteProposal] = None
    draft: Optional[AssistantDraftProposal] = None
    settings_patch: Optional[AssistantSettingsPatch] = None
    story_plan: Optional[AssistantStoryPlanProposal] = None


@dataclass(frozen=True)
class AssistantChatResponse:
    result: AssistantChatResult
    raw_response: str
    input_tokens: int
    output_tokens: int
    provider: str
    model: str
    accessed_sources: list[dict[str, Any]] = field(default_factory=list)
    agent_trace: list[dict[str, Any]] = field(default_factory=list)
