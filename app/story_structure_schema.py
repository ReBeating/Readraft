from __future__ import annotations

import json
from collections import Counter
from typing import Annotated, Any, List, Literal, Mapping

from pydantic import Field, model_validator

from .story_planning_schema import StrictStoryPlanningModel


StructureListItem = Annotated[str, Field(min_length=1, max_length=600)]
ChapterStructuralRole = Literal[
    "setup",
    "escalation",
    "reversal",
    "payoff",
    "transition",
]


class AuthorChapterSkeleton(StrictStoryPlanningModel):
    title: str = Field(min_length=1, max_length=120)
    structural_role: ChapterStructuralRole
    purpose: str = Field(min_length=8, max_length=800)
    key_points: List[StructureListItem] = Field(
        min_length=2, max_length=5
    )
    arc_titles: List[StructureListItem] = Field(
        min_length=1, max_length=4
    )
    ending_hook: str = Field(min_length=4, max_length=500)

    @model_validator(mode="after")
    def validate_lists(self) -> "AuthorChapterSkeleton":
        if len({item.casefold() for item in self.key_points}) != len(
            self.key_points
        ):
            raise ValueError("同一章的关键点不能重复")
        if len({item.casefold() for item in self.arc_titles}) != len(
            self.arc_titles
        ):
            raise ValueError("同一章不能重复引用同一条剧情线")
        return self


class ProposedVolumeStructure(StrictStoryPlanningModel):
    position: int = Field(ge=1, le=1000)
    title: str = Field(min_length=1, max_length=160)
    goal: str = Field(default="", max_length=1600)
    start_state: str = Field(default="", max_length=1600)
    end_state: str = Field(default="", max_length=1600)
    major_conflict: str = Field(default="", max_length=1800)
    payoff: str = Field(default="", max_length=1800)
    arc_titles: List[StructureListItem] = Field(
        min_length=1, max_length=8
    )

    @model_validator(mode="after")
    def validate_arc_titles(self) -> "ProposedVolumeStructure":
        normalized = [title.casefold() for title in self.arc_titles]
        if len(set(normalized)) != len(normalized):
            raise ValueError("同一分卷不能重复引用同一条剧情线")
        return self


class ProposedChapterSkeleton(StrictStoryPlanningModel):
    position: int = Field(ge=1, le=100_000)
    volume_position: int = Field(ge=1, le=1000)
    title: str = Field(min_length=1, max_length=120)
    structural_role: ChapterStructuralRole
    purpose: str = Field(min_length=8, max_length=800)
    key_points: List[StructureListItem] = Field(
        min_length=2, max_length=5
    )
    arc_titles: List[StructureListItem] = Field(
        min_length=1, max_length=4
    )
    ending_hook: str = Field(min_length=4, max_length=500)

    @model_validator(mode="after")
    def validate_lists(self) -> "ProposedChapterSkeleton":
        if len({item.casefold() for item in self.key_points}) != len(
            self.key_points
        ):
            raise ValueError("同一章的关键点不能重复")
        if len({item.casefold() for item in self.arc_titles}) != len(
            self.arc_titles
        ):
            raise ValueError("同一章不能重复引用同一条剧情线")
        return self


class StoryStructureOption(StrictStoryPlanningModel):
    label: str = Field(min_length=2, max_length=80)
    distinctive_choice: str = Field(min_length=10, max_length=1600)
    reader_experience: str = Field(min_length=10, max_length=1600)
    strengths: List[StructureListItem] = Field(min_length=2, max_length=5)
    tradeoffs: List[StructureListItem] = Field(min_length=2, max_length=5)
    volumes: List[ProposedVolumeStructure] = Field(
        min_length=2, max_length=6
    )
    chapters: List[ProposedChapterSkeleton] = Field(
        min_length=10, max_length=30
    )

    @model_validator(mode="after")
    def validate_structure(self) -> "StoryStructureOption":
        volume_positions = [volume.position for volume in self.volumes]
        if len(set(volume_positions)) != len(volume_positions):
            raise ValueError("同一方案的分卷位置不能重复")
        expected_volume_positions = list(
            range(min(volume_positions), max(volume_positions) + 1)
        )
        if sorted(volume_positions) != expected_volume_positions:
            raise ValueError("分卷 position 必须连续")

        chapter_positions = [chapter.position for chapter in self.chapters]
        if len(set(chapter_positions)) != len(chapter_positions):
            raise ValueError("同一方案的章节位置不能重复")
        expected_chapter_positions = list(
            range(min(chapter_positions), max(chapter_positions) + 1)
        )
        if sorted(chapter_positions) != expected_chapter_positions:
            raise ValueError("章节 position 必须连续")
        if chapter_positions != sorted(chapter_positions):
            raise ValueError("章节必须按 position 顺序输出")

        volume_by_position = {
            volume.position: volume for volume in self.volumes
        }
        assignments = [
            chapter.volume_position for chapter in self.chapters
        ]
        if assignments != sorted(assignments):
            raise ValueError("章节归卷必须按顺序推进，不能回到更早分卷")
        unknown_volumes = sorted(
            {
                position
                for position in assignments
                if position not in volume_by_position
            }
        )
        if unknown_volumes:
            raise ValueError(
                "章节引用了方案中不存在的分卷："
                + "、".join(str(item) for item in unknown_volumes)
            )
        counts = Counter(assignments)
        empty_volumes = [
            str(position)
            for position in volume_positions
            if not counts[position]
        ]
        if empty_volumes:
            raise ValueError(
                "每个分卷都必须承载至少一章："
                + "、".join(empty_volumes)
            )

        for chapter in self.chapters:
            volume = volume_by_position[chapter.volume_position]
            unknown_arcs = [
                title
                for title in chapter.arc_titles
                if title not in volume.arc_titles
            ]
            if unknown_arcs:
                raise ValueError(
                    "章节只能引用所属分卷列出的剧情线："
                    + "、".join(unknown_arcs)
                )
        roles = {chapter.structural_role for chapter in self.chapters}
        if "reversal" not in roles or "payoff" not in roles:
            raise ValueError("滚动章节骨架至少需要一次 reversal 和一次 payoff")
        return self

    def structural_signature(self) -> str:
        payload = {
            "volume_boundaries": [
                {
                    "position": volume.position,
                    "title": volume.title,
                    "chapter_positions": [
                        chapter.position
                        for chapter in self.chapters
                        if chapter.volume_position == volume.position
                    ],
                }
                for volume in self.volumes
            ],
            "chapter_engine": [
                {
                    "position": chapter.position,
                    "role": chapter.structural_role,
                    "arcs": chapter.arc_titles,
                }
                for chapter in self.chapters
            ],
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


class StoryStructureProposalSet(StrictStoryPlanningModel):
    comparison_summary: str = Field(min_length=10, max_length=3000)
    target_chapter_count: int = Field(ge=10, le=30)
    options: List[StoryStructureOption] = Field(
        min_length=3, max_length=3
    )

    @model_validator(mode="after")
    def validate_distinct_options(self) -> "StoryStructureProposalSet":
        if any(
            len(option.chapters) != self.target_chapter_count
            for option in self.options
        ):
            raise ValueError("每套方案都必须包含目标数量的章节骨架")
        labels = [option.label.casefold() for option in self.options]
        if len(set(labels)) != 3:
            raise ValueError("三套滚动结构方案必须使用不同名称")
        choices = [
            option.distinctive_choice.casefold() for option in self.options
        ]
        if len(set(choices)) != 3:
            raise ValueError("三套滚动结构方案必须有不同的核心选择")
        signatures = [
            option.structural_signature() for option in self.options
        ]
        if len(set(signatures)) != 3:
            raise ValueError("三套方案必须具有不同的卷界或章节发动机")
        return self

    def ensure_context_compatible(
        self, context: Mapping[str, Any]
    ) -> None:
        requested_count = int(context.get("requested_chapter_count") or 0)
        if self.target_chapter_count != requested_count:
            raise ValueError("模型返回的章节数量与作者请求不一致")
        current_position = int(
            context.get("current_canonical_position") or 0
        )
        expected_positions = list(
            range(
                current_position + 1,
                current_position + requested_count + 1,
            )
        )
        allowed_arcs = {
            str(item.get("title") or ""): str(
                item.get("arc_type") or ""
            )
            for item in (context.get("allowed_plot_arcs") or [])
            if str(item.get("title") or "")
        }
        main_arc_titles = {
            title
            for title, arc_type in allowed_arcs.items()
            if arc_type == "main"
        }
        if not main_arc_titles:
            raise ValueError("生成滚动结构前至少需要一条已确认主线")

        allowed_starts = {
            int(item)
            for item in (
                context.get("allowed_volume_start_positions") or []
            )
        }
        locked_volume = dict(context.get("locked_volume") or {})
        locked_position = int(locked_volume.get("position") or 0)
        locked_fields = (
            "title",
            "goal",
            "start_state",
            "end_state",
            "major_conflict",
            "payoff",
        )

        for option in self.options:
            if [chapter.position for chapter in option.chapters] != (
                expected_positions
            ):
                raise ValueError(
                    "每套方案必须从当前正史下一章连续覆盖目标窗口"
                )
            first_volume_position = min(
                volume.position for volume in option.volumes
            )
            if (
                allowed_starts
                and first_volume_position not in allowed_starts
            ):
                raise ValueError("方案的起始分卷与当前正史卷界不兼容")

            used_arc_titles: set[str] = set()
            for volume in option.volumes:
                unknown = [
                    title
                    for title in volume.arc_titles
                    if title not in allowed_arcs
                ]
                if unknown:
                    raise ValueError(
                        "分卷只能引用已确认且可推进的精确剧情线名称："
                        + "、".join(unknown)
                    )
                if locked_position and volume.position == locked_position:
                    for field in locked_fields:
                        if getattr(volume, field) != str(
                            locked_volume.get(field) or ""
                        ):
                            raise ValueError(
                                "继续当前正史分卷时必须原样保留其既有资料"
                            )
                elif any(
                    len(str(getattr(volume, field) or "").strip()) < 4
                    for field in locked_fields[1:]
                ):
                    raise ValueError(
                        "新建或重规划的分卷必须包含目标、起止状态、"
                        "主要冲突和回报"
                    )
            for chapter in option.chapters:
                unknown = [
                    title
                    for title in chapter.arc_titles
                    if title not in allowed_arcs
                ]
                if unknown:
                    raise ValueError(
                        "章节只能引用已确认且可推进的精确剧情线名称："
                        + "、".join(unknown)
                    )
                used_arc_titles.update(chapter.arc_titles)
            missing_main = sorted(main_arc_titles - used_arc_titles)
            if missing_main:
                raise ValueError(
                    "每套方案都必须在窗口内实际推进已确认主线："
                    + "、".join(missing_main)
                )
