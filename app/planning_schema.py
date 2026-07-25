from __future__ import annotations

import hashlib
import json
from typing import Any, List, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictPlanningModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


SceneRequirementKind = Literal[
    "plot_thread",
    "must_happen",
    "foreshadow_setup",
    "foreshadow_payoff",
    "ending_hook",
]


class SceneRequirementRef(StrictPlanningModel):
    kind: SceneRequirementKind
    text: str = Field(min_length=1, max_length=2000)


class SceneBeat(StrictPlanningModel):
    pov_character: str = Field(default="", max_length=80)
    goal: str = Field(min_length=1, max_length=1200)
    obstacle: str = Field(min_length=1, max_length=1200)
    action: str = Field(min_length=1, max_length=2000)
    reveal: str = Field(default="", max_length=1200)
    conceal: str = Field(default="", max_length=1200)
    subtext: str = Field(default="", max_length=1200)
    location: str = Field(default="", max_length=200)
    key_items: List[str] = Field(default_factory=list, max_length=20)
    end_state: str = Field(min_length=1, max_length=1200)
    transition: str = Field(default="", max_length=1200)
    requirement_refs: List[SceneRequirementRef] = Field(
        default_factory=list,
        max_length=40,
    )

    @model_validator(mode="after")
    def ensure_unique_requirement_refs(self) -> "SceneBeat":
        keys = [(item.kind, item.text) for item in self.requirement_refs]
        if len(keys) != len(set(keys)):
            raise ValueError("同一场景不能重复绑定同一条任务要求")
        return self


class ChapterTaskCard(StrictPlanningModel):
    purpose: str = Field(default="", max_length=2000)
    start_state: str = Field(default="", max_length=2000)
    end_state: str = Field(default="", max_length=2000)
    central_conflict: str = Field(default="", max_length=2000)
    emotional_value: str = Field(default="", max_length=1200)
    plot_threads: List[str] = Field(default_factory=list, max_length=20)
    must_happen: List[str] = Field(default_factory=list, max_length=30)
    must_preserve: List[str] = Field(default_factory=list, max_length=30)
    forbidden: List[str] = Field(default_factory=list, max_length=30)
    foreshadow_setup: List[str] = Field(default_factory=list, max_length=20)
    foreshadow_payoff: List[str] = Field(default_factory=list, max_length=20)
    ending_hook: str = Field(default="", max_length=1200)
    target_chars: int = Field(default=3000, ge=2000, le=12000)
    scenes: List[SceneBeat] = Field(default_factory=list, max_length=5)

    def requirement_items(self) -> List[SceneRequirementRef]:
        items = [
            SceneRequirementRef(kind="plot_thread", text=text)
            for text in self.plot_threads
        ]
        items.extend(
            SceneRequirementRef(kind="must_happen", text=text)
            for text in self.must_happen
        )
        items.extend(
            SceneRequirementRef(kind="foreshadow_setup", text=text)
            for text in self.foreshadow_setup
        )
        items.extend(
            SceneRequirementRef(kind="foreshadow_payoff", text=text)
            for text in self.foreshadow_payoff
        )
        if self.ending_hook:
            items.append(
                SceneRequirementRef(
                    kind="ending_hook",
                    text=self.ending_hook,
                )
            )
        return items

    def ensure_scene_requirement_coverage(
        self,
        *,
        required: bool,
    ) -> None:
        expected = {
            (item.kind, item.text) for item in self.requirement_items()
        }
        actual = {
            (item.kind, item.text)
            for scene in self.scenes
            for item in scene.requirement_refs
        }
        unknown = sorted(actual - expected)
        if unknown:
            kind, text = unknown[0]
            raise ValueError(
                f"场景绑定了任务卡中不存在的要求：{kind} / {text}"
            )
        if not required and not actual:
            return
        missing = sorted(expected - actual)
        if missing:
            _, text = missing[0]
            raise ValueError(
                f"仍有任务要求没有分配到任何场景：{text}"
            )

    def ensure_confirmable(self) -> None:
        missing = []
        if not self.purpose:
            missing.append("本章作用")
        if not self.start_state:
            missing.append("开场状态")
        if not self.end_state:
            missing.append("结束状态")
        if not self.central_conflict:
            missing.append("核心冲突")
        if not self.ending_hook:
            missing.append("章末钩子")
        if len(self.scenes) < 2:
            missing.append("至少 2 个场景节拍")
        if missing:
            raise ValueError("确认任务卡前请补充：" + "、".join(missing))
        self.ensure_scene_requirement_coverage(required=True)


class SceneBeatPlan(StrictPlanningModel):
    scenes: List[SceneBeat] = Field(min_length=2, max_length=5)
    planning_note: str = Field(default="", max_length=1200)

    def ensure_covers(self, task_card: ChapterTaskCard) -> None:
        candidate = task_card.model_copy(
            update={"scenes": self.scenes}
        )
        candidate.ensure_scene_requirement_coverage(required=True)


def allocate_scene_requirement_refs(
    card: ChapterTaskCard,
) -> ChapterTaskCard:
    if not card.scenes:
        return card
    scene_refs: list[list[dict[str, str]]] = [
        [] for _ in card.scenes
    ]
    ordinary_position = 0
    for requirement in card.requirement_items():
        if requirement.kind in {"foreshadow_payoff", "ending_hook"}:
            scene_index = len(scene_refs) - 1
        else:
            scene_index = ordinary_position % len(scene_refs)
            ordinary_position += 1
        scene_refs[scene_index].append(
            requirement.model_dump(mode="json")
        )
    payload = card.model_dump(mode="json")
    for index, refs in enumerate(scene_refs):
        payload["scenes"][index]["requirement_refs"] = refs
    return ChapterTaskCard.model_validate(payload)


def chapter_task_card_payload(
    value: ChapterTaskCard | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(value, ChapterTaskCard):
        card = value
    else:
        data = dict(value)
        if "scenes" in data:
            data["scenes"] = [
                {
                    name: dict(scene)[name]
                    for name in SceneBeat.model_fields
                    if name in dict(scene)
                }
                for scene in data.get("scenes") or []
            ]
        card = ChapterTaskCard.model_validate(
            {
                name: data[name]
                for name in ChapterTaskCard.model_fields
                if name in data
            }
        )
    return card.model_dump(mode="json")


def chapter_task_card_fingerprint(
    value: ChapterTaskCard | Mapping[str, Any],
) -> str:
    serialized = json.dumps(
        chapter_task_card_payload(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
