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


class AssistantArchiveRuleProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    category: Literal[
        "core", "world", "character", "structure", "style"
    ] = Field(
        description=(
            "资料归属：作品概览、世界、人物、剧情与结构、叙事与文风之一"
        )
    )
    title: str = Field(default="", max_length=120)
    content: str = Field(min_length=2, max_length=6000)


class AssistantSettingsPatch(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=120,
        description="作品正式书名",
    )
    genre: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=80,
        description="作品题材或类型",
    )
    premise: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=4000,
        description="概括全书核心因果的一句话故事",
    )
    theme: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=2000,
        description="作品希望持续探索的主题",
    )
    story_promise: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=4000,
        description="作品向读者承诺的主要阅读体验",
    )
    target_audience: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=1000,
        description="目标读者",
    )
    core_appeal: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=4000,
        description="作品最核心、可持续兑现的吸引力",
    )
    ending_constraint: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=4000,
        description="结局必须满足的全局约束",
    )
    world_setting: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=20_000,
        description=(
            "只用于全书级时代、空间、社会秩序和世界运行逻辑的概述；"
            "不得塞入人物卡、具体情节、章节计划或文风要求"
        ),
    )
    style_guide: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=10_000,
        description="全书通用的叙事与语言规范",
    )
    point_of_view: Optional[
        Literal["第三人称限知", "第一人称", "第三人称全知", "多视角"]
    ] = None
    archive_rules: Optional[list[AssistantArchiveRuleProposal]] = Field(
        default=None,
        min_length=1,
        max_length=30,
        description=(
            "无法准确放入上述全局字段的具体作品资料。每条必须归入作品概览、"
            "世界、人物、剧情与结构、叙事与文风之一"
        ),
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
