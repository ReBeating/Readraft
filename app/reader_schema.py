from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field


ReaderRisk = Literal["low", "medium", "high"]


class StrictReaderModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class FutureChapterRevision(StrictReaderModel):
    chapter_position: int = Field(ge=1, le=100_000)
    title: str = Field(min_length=1, max_length=120)
    outline: str = Field(min_length=2, max_length=4000)
    key_points: List[str] = Field(
        default_factory=list, min_length=1, max_length=20
    )
    rationale: str = Field(min_length=2, max_length=1200)


class ReaderBranchProposal(StrictReaderModel):
    label: str = Field(min_length=2, max_length=80)
    summary: str = Field(min_length=4, max_length=1600)
    satisfies: List[str] = Field(
        default_factory=list, min_length=1, max_length=12
    )
    sacrifices: List[str] = Field(
        default_factory=list, min_length=1, max_length=12
    )
    affected_characters: List[str] = Field(
        default_factory=list, max_length=20
    )
    affected_plot_threads: List[str] = Field(
        default_factory=list, max_length=20
    )
    promise_impact: str = Field(min_length=2, max_length=1200)
    future_changes: List[FutureChapterRevision] = Field(
        min_length=1, max_length=12
    )
    affects_published_canon: bool = False
    published_canon_impact: str = Field(default="", max_length=1600)
    risk_level: ReaderRisk
    risks: List[str] = Field(
        default_factory=list, min_length=1, max_length=12
    )


class ReaderBranchSet(StrictReaderModel):
    analysis_summary: str = Field(min_length=4, max_length=1600)
    alternatives: List[ReaderBranchProposal] = Field(
        min_length=3, max_length=3
    )

    def ensure_applicable(
        self, *, current_position: int, planning_horizon: int
    ) -> None:
        maximum_position = current_position + planning_horizon
        labels: set[str] = set()
        summaries: set[str] = set()
        canon_safe = 0
        for alternative in self.alternatives:
            normalized_label = alternative.label.casefold()
            normalized_summary = alternative.summary.casefold()
            if normalized_label in labels or normalized_summary in summaries:
                raise ValueError("三个剧情方案必须彼此不同")
            labels.add(normalized_label)
            summaries.add(normalized_summary)
            if not alternative.affects_published_canon:
                canon_safe += 1
            positions: set[int] = set()
            for change in alternative.future_changes:
                if change.chapter_position in positions:
                    raise ValueError("同一方案不能重复修改同一章")
                positions.add(change.chapter_position)
                if not (
                    current_position
                    < change.chapter_position
                    <= maximum_position
                ):
                    raise ValueError(
                        "方案只能修改当前正史之后、滚动窗口以内的章节"
                    )
        if canon_safe < 2:
            raise ValueError("三个方案中至少两个必须不改动已发表正史")
