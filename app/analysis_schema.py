from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .technique_schema import TechniqueObservation


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Character(StrictModel):
    name: str = Field(min_length=1, max_length=60)
    role: str = Field(min_length=1, max_length=100)
    development: str = Field(min_length=1, max_length=240)


class Scene(StrictModel):
    location: str = Field(min_length=1, max_length=100)
    time: str = Field(min_length=1, max_length=60)
    participants: List[str] = Field(min_length=1, max_length=12)
    summary: str = Field(min_length=1, max_length=300)


class KeyEvent(StrictModel):
    event: str = Field(min_length=1, max_length=240)
    impact: str = Field(min_length=1, max_length=240)


class Foreshadowing(StrictModel):
    type: Literal["setup", "payoff"]
    clue: str = Field(min_length=1, max_length=240)
    interpretation: str = Field(min_length=1, max_length=240)


class Conflict(StrictModel):
    parties: List[str] = Field(min_length=1, max_length=6)
    description: str = Field(min_length=1, max_length=240)
    status: Literal[
        "emerging",
        "escalating",
        "unresolved",
        "temporarily_resolved",
        "resolved",
    ]


class EndingHook(StrictModel):
    type: Literal[
        "suspense",
        "reversal",
        "crisis",
        "revelation",
        "new_goal",
        "emotional_cliffhanger",
    ]
    content: str = Field(min_length=1, max_length=240)


class ChapterAnalysis(StrictModel):
    chapter_title: str = Field(min_length=1, max_length=100)
    summary: str = Field(min_length=10, max_length=500)
    characters: List[Character] = Field(max_length=12)
    scenes: List[Scene] = Field(max_length=10)
    key_events: List[KeyEvent] = Field(max_length=12)
    foreshadowing: List[Foreshadowing] = Field(max_length=8)
    conflicts: List[Conflict] = Field(max_length=8)
    ending_hook: Optional[EndingHook]
    techniques: List[TechniqueObservation] = Field(
        default_factory=list, max_length=6
    )


ANALYSIS_JSON_EXAMPLE = {
    "chapter_title": "第三章 雨夜来客",
    "summary": "林川在雨夜收留受伤女子，并发现她与父亲旧案有关。女子离开后，林川决定追查真相。",
    "characters": [
        {
            "name": "林川",
            "role": "本章主视角人物",
            "development": "从被动卷入转为主动调查旧案",
        }
    ],
    "scenes": [
        {
            "location": "林川家中",
            "time": "深夜",
            "participants": ["林川", "受伤女子"],
            "summary": "林川救助女子并发现与父亲旧案相关的徽记",
        }
    ],
    "key_events": [
        {"event": "林川发现旧案徽记", "impact": "推动他开始调查父亲死亡真相"}
    ],
    "foreshadowing": [
        {
            "type": "setup",
            "clue": "受伤女子携带与林川父亲旧案有关的徽记",
            "interpretation": "徽记的来源与女子身份仍待后续解释",
        }
    ],
    "conflicts": [
        {
            "parties": ["林川", "受伤女子"],
            "description": "林川追问身份，女子拒绝说明",
            "status": "unresolved",
        }
    ],
    "ending_hook": {
        "type": "revelation",
        "content": "徽记背面刻着林川父亲的名字",
    },
    "techniques": [
        {
            "name": "关键物件先产生后果、后解释来源",
            "dimension": "information",
            "source_location": "徽记首次出现至章末揭示",
            "observation": "物件先改变人物的调查方向，章末才补充一项来源信息。",
            "effect": "让读者先感受到物件的叙事重量，再获得新的可验证问题。",
            "suitable_for": ["悬疑线索首次登场", "需要跨场景维持问题时"],
            "unsuitable_for": ["必须立即说明规则才能理解行动的场景"],
            "execution_rule": "让关键物件先触发人物行动或代价，延后解释一项来源信息，并保留下一步可核对的问题。",
            "originality_boundary": "只迁移信息释放顺序，不复用徽记、父亲旧案、具体揭示内容或原文措辞。",
        }
    ],
}
