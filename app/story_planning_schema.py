from __future__ import annotations

from typing import Annotated, List, Literal

from pydantic import BaseModel, ConfigDict, Field


StoryArcType = Literal[
    "main",
    "subplot",
    "character",
    "relationship",
    "mystery",
    "world",
]
StoryArcLifecycle = Literal[
    "planned",
    "active",
    "paused",
    "resolved",
    "abandoned",
]
PlanListItem = Annotated[str, Field(min_length=1, max_length=1200)]
CharacterName = Annotated[str, Field(min_length=1, max_length=120)]


class StrictStoryPlanningModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class StoryBlueprint(StrictStoryPlanningModel):
    central_question: str = Field(default="", max_length=2000)
    protagonist_goal: str = Field(default="", max_length=2000)
    core_conflict: str = Field(default="", max_length=3000)
    stakes: str = Field(default="", max_length=3000)
    opening_state: str = Field(default="", max_length=3000)
    ending_state: str = Field(default="", max_length=3000)
    major_turns: List[PlanListItem] = Field(
        default_factory=list, max_length=20
    )
    must_payoffs: List[PlanListItem] = Field(
        default_factory=list, max_length=30
    )
    forbidden_shortcuts: List[PlanListItem] = Field(
        default_factory=list, max_length=30
    )
    author_notes: str = Field(default="", max_length=6000)

    def ensure_confirmable(self) -> None:
        missing = []
        if not self.central_question:
            missing.append("核心悬问")
        if not self.protagonist_goal:
            missing.append("主角长期目标")
        if not self.core_conflict:
            missing.append("冲突引擎")
        if not self.ending_state:
            missing.append("终局状态")
        if not self.major_turns:
            missing.append("至少一个全书转折")
        if not self.must_payoffs:
            missing.append("至少一个必须兑现项")
        if missing:
            raise ValueError("确认全书蓝图前请补充：" + "、".join(missing))


class PlannedStoryArc(StrictStoryPlanningModel):
    arc_type: StoryArcType = "subplot"
    title: str = Field(min_length=1, max_length=160)
    dramatic_question: str = Field(default="", max_length=2000)
    promise: str = Field(default="", max_length=2000)
    start_state: str = Field(default="", max_length=2000)
    target_payoff: str = Field(default="", max_length=3000)
    involved_characters: List[CharacterName] = Field(
        default_factory=list, max_length=30
    )
    planned_turns: List[PlanListItem] = Field(
        default_factory=list, max_length=20
    )
    lifecycle_status: StoryArcLifecycle = "planned"
    priority: int = Field(default=3, ge=1, le=5)
    author_notes: str = Field(default="", max_length=4000)

    def ensure_confirmable(self) -> None:
        if self.lifecycle_status == "abandoned":
            return
        missing = []
        if not self.dramatic_question:
            missing.append("剧情线悬问")
        if not self.target_payoff:
            missing.append("目标回报")
        if self.lifecycle_status != "resolved":
            if not self.promise:
                missing.append("对读者的承诺")
        if missing:
            raise ValueError("确认剧情线前请补充：" + "、".join(missing))
