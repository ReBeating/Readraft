from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class StrictMemoryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CharacterStateChange(StrictMemoryModel):
    character_name: str = Field(min_length=1, max_length=80)
    aspect: Literal[
        "status",
        "location",
        "physical",
        "emotional",
        "goal",
        "ability",
        "possession",
        "other",
    ]
    before: Optional[str] = Field(default=None, max_length=1000)
    after: str = Field(min_length=1, max_length=1000)
    evidence: str = Field(min_length=1, max_length=500)


class RelationshipChange(StrictMemoryModel):
    character_a: str = Field(min_length=1, max_length=80)
    character_b: str = Field(min_length=1, max_length=80)
    before: Optional[str] = Field(default=None, max_length=1000)
    after: str = Field(min_length=1, max_length=1000)
    evidence: str = Field(min_length=1, max_length=500)


class LocationChange(StrictMemoryModel):
    subject_name: str = Field(min_length=1, max_length=120)
    from_location: Optional[str] = Field(default=None, max_length=200)
    to_location: str = Field(min_length=1, max_length=200)
    evidence: str = Field(min_length=1, max_length=500)


class ItemChange(StrictMemoryModel):
    item_name: str = Field(min_length=1, max_length=120)
    action: Literal[
        "created",
        "acquired",
        "lost",
        "transferred",
        "used",
        "destroyed",
        "changed",
        "other",
    ]
    from_holder: Optional[str] = Field(default=None, max_length=120)
    to_holder: Optional[str] = Field(default=None, max_length=120)
    state: str = Field(default="", max_length=1000)
    evidence: str = Field(min_length=1, max_length=500)


class KnowledgeChange(StrictMemoryModel):
    character_name: str = Field(min_length=1, max_length=80)
    fact: str = Field(min_length=1, max_length=1500)
    canonical_fact: str = Field(
        default="",
        max_length=1500,
        description=(
            "用于跨章节复用的标准事实表述；若既有身份账本中已有同一事实，"
            "必须逐字复用其标准表述"
        ),
    )
    state: Literal["knows", "suspects", "believes_false", "forgets"]
    learned_via: str = Field(default="", max_length=1000)
    evidence: str = Field(min_length=1, max_length=500)


class PlotThreadChange(StrictMemoryModel):
    thread_name: str = Field(min_length=1, max_length=160)
    thread_type: Literal[
        "main", "subplot", "relationship", "mystery", "promise", "other"
    ] = "subplot"
    action: Literal["opened", "advanced", "paused", "resolved", "abandoned"]
    update: str = Field(min_length=1, max_length=1500)
    promise: str = Field(default="", max_length=1000)
    target_payoff: str = Field(default="", max_length=1000)
    evidence: str = Field(min_length=1, max_length=500)


class ForeshadowingChange(StrictMemoryModel):
    hook_name: str = Field(min_length=1, max_length=160)
    action: Literal["setup", "advanced", "payoff", "abandoned"]
    description: str = Field(min_length=1, max_length=1500)
    intended_payoff: str = Field(default="", max_length=1000)
    evidence: str = Field(min_length=1, max_length=500)


class StoryEvent(StrictMemoryModel):
    event_key: str = Field(
        default="",
        max_length=160,
        description=(
            "跨章节稳定复用的简短事件键；同一事件再次被引用时必须复用"
        ),
    )
    summary: str = Field(min_length=1, max_length=1500)
    participants: List[str] = Field(default_factory=list, max_length=20)
    location: str = Field(default="", max_length=200)
    story_time: str = Field(default="", max_length=200)
    causes: List[str] = Field(default_factory=list, max_length=10)
    cause_event_keys: List[str] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "只填写本事件直接依赖的、此前已经发生的 event_key"
        ),
    )
    effects: List[str] = Field(default_factory=list, max_length=10)
    evidence: str = Field(min_length=1, max_length=500)


class TimeAdvance(StrictMemoryModel):
    from_time: str = Field(default="", max_length=200)
    to_time: str = Field(default="", max_length=200)
    elapsed: str = Field(default="", max_length=200)


class StoryDelta(StrictMemoryModel):
    chapter_summary: str = Field(min_length=1, max_length=3000)
    keywords: List[str] = Field(default_factory=list, max_length=30)
    unresolved_questions: List[str] = Field(
        default_factory=list, max_length=20
    )
    character_changes: List[CharacterStateChange] = Field(
        default_factory=list, max_length=30
    )
    relationship_changes: List[RelationshipChange] = Field(
        default_factory=list, max_length=20
    )
    location_changes: List[LocationChange] = Field(
        default_factory=list, max_length=20
    )
    item_changes: List[ItemChange] = Field(
        default_factory=list, max_length=20
    )
    knowledge_changes: List[KnowledgeChange] = Field(
        default_factory=list, max_length=30
    )
    plot_thread_changes: List[PlotThreadChange] = Field(
        default_factory=list, max_length=20
    )
    foreshadowing_changes: List[ForeshadowingChange] = Field(
        default_factory=list, max_length=20
    )
    events: List[StoryEvent] = Field(default_factory=list, max_length=30)
    time_advance: Optional[TimeAdvance] = None
