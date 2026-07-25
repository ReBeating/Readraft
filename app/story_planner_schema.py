from __future__ import annotations

import json
from typing import Annotated, List, Literal

from pydantic import Field, model_validator

from .story_planning_schema import (
    PlanListItem,
    PlannedStoryArc,
    StrictStoryPlanningModel,
    StoryBlueprint,
)


StoryPlanningMode = Literal["create", "refine", "rethink"]
OptionListItem = Annotated[str, Field(min_length=1, max_length=1200)]


class StoryVolumeSketch(StrictStoryPlanningModel):
    """A comparison aid only; applying a proposal does not create volumes."""

    position: int = Field(ge=1, le=12)
    title: str = Field(min_length=1, max_length=160)
    purpose: str = Field(min_length=1, max_length=2000)
    start_state: str = Field(min_length=1, max_length=2000)
    end_state: str = Field(min_length=1, max_length=2000)
    major_conflict: str = Field(min_length=1, max_length=2500)
    payoff: str = Field(min_length=1, max_length=2500)
    arc_titles: List[PlanListItem] = Field(
        min_length=1, max_length=8
    )


class StoryPlanOption(StrictStoryPlanningModel):
    label: str = Field(min_length=2, max_length=80)
    distinctive_choice: str = Field(min_length=10, max_length=2000)
    reader_experience: str = Field(min_length=10, max_length=2000)
    strengths: List[OptionListItem] = Field(min_length=2, max_length=5)
    tradeoffs: List[OptionListItem] = Field(min_length=2, max_length=5)
    blueprint: StoryBlueprint
    plot_arcs: List[PlannedStoryArc] = Field(min_length=3, max_length=8)
    volume_sketches: List[StoryVolumeSketch] = Field(
        min_length=2, max_length=6
    )

    @model_validator(mode="after")
    def validate_complete_option(self) -> "StoryPlanOption":
        self.blueprint.ensure_confirmable()
        normalized_titles: set[str] = set()
        has_main_arc = False
        for arc in self.plot_arcs:
            if arc.lifecycle_status not in {"planned", "active"}:
                raise ValueError(
                    "全书方案中的剧情线只能是 planned 或 active"
                )
            arc.ensure_confirmable()
            normalized = arc.title.casefold()
            if normalized in normalized_titles:
                raise ValueError("同一方案中的剧情线名称不能重复")
            normalized_titles.add(normalized)
            has_main_arc = has_main_arc or arc.arc_type == "main"
        if not has_main_arc:
            raise ValueError("每套全书方案至少需要一条 main 主线")

        positions: set[int] = set()
        volume_titles: set[str] = set()
        for sketch in self.volume_sketches:
            if sketch.position in positions:
                raise ValueError("分卷草图位置不能重复")
            positions.add(sketch.position)
            normalized_volume_title = sketch.title.casefold()
            if normalized_volume_title in volume_titles:
                raise ValueError("分卷草图名称不能重复")
            volume_titles.add(normalized_volume_title)
            unknown = [
                title
                for title in sketch.arc_titles
                if title.casefold() not in normalized_titles
            ]
            if unknown:
                raise ValueError(
                    "分卷草图只能引用本方案中的精确剧情线名称："
                    + "、".join(unknown)
                )
        expected_positions = list(
            range(1, len(self.volume_sketches) + 1)
        )
        if sorted(positions) != expected_positions:
            raise ValueError("分卷草图 position 必须从 1 连续排列")
        return self

    def structural_signature(self) -> str:
        payload = {
            "blueprint": self.blueprint.model_dump(mode="json"),
            "plot_arcs": [
                arc.model_dump(mode="json") for arc in self.plot_arcs
            ],
            "volume_sketches": [
                sketch.model_dump(mode="json")
                for sketch in self.volume_sketches
            ],
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


class StoryPlanProposalSet(StrictStoryPlanningModel):
    comparison_summary: str = Field(min_length=10, max_length=3000)
    options: List[StoryPlanOption] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_distinct_options(self) -> "StoryPlanProposalSet":
        labels = [option.label.casefold() for option in self.options]
        if len(set(labels)) != 3:
            raise ValueError("三套方案必须使用不同名称")
        choices = [
            option.distinctive_choice.casefold() for option in self.options
        ]
        if len(set(choices)) != 3:
            raise ValueError("三套方案必须有不同的结构选择")
        signatures = [
            option.structural_signature() for option in self.options
        ]
        if len(set(signatures)) != 3:
            raise ValueError("三套方案不能只是同一结构的重复")
        return self
