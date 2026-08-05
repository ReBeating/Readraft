"""Schemas and evidence validation for reference-book analysis."""

from __future__ import annotations

import hashlib
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .technique_schema import TechniqueObservation


ANALYSIS_SCHEMA_VERSION = "3.0"
MODEL_LAYERS = ("facts", "narrative", "style", "techniques")
ALL_LAYERS = ("structure", *MODEL_LAYERS)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class EvidenceSpan(StrictModel):
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str = Field(min_length=1)

    @model_validator(mode="after")
    def ordered(self) -> "EvidenceSpan":
        if self.end <= self.start:
            raise ValueError("证据结束位置必须大于开始位置")
        return self


class EvidencedModel(StrictModel):
    evidence: list[EvidenceSpan] = Field(min_length=1)


class FactCharacter(EvidencedModel):
    name: str = Field(min_length=1)
    role: str = Field(min_length=1)
    development: str = Field(min_length=1)


class FactScene(EvidencedModel):
    location: str = Field(min_length=1)
    time: str = Field(min_length=1)
    participants: list[str] = Field(min_length=1)
    summary: str = Field(min_length=1)


class FactEvent(EvidencedModel):
    event: str = Field(min_length=1)
    impact: str = Field(min_length=1)


class FactForeshadowing(EvidencedModel):
    type: Literal["setup", "payoff"]
    thread_key: str = Field(min_length=1)
    clue: str = Field(min_length=1)
    interpretation: str = Field(min_length=1)


class FactsLayer(StrictModel):
    summary: str = Field(min_length=10)
    characters: list[FactCharacter] = Field(default_factory=list)
    scenes: list[FactScene] = Field(default_factory=list)
    events: list[FactEvent] = Field(default_factory=list)
    foreshadowing: list[FactForeshadowing] = Field(default_factory=list)


class NarrativeConflict(EvidencedModel):
    parties: list[str] = Field(min_length=1)
    description: str = Field(min_length=1)
    status: Literal[
        "emerging",
        "escalating",
        "unresolved",
        "temporarily_resolved",
        "resolved",
    ]


class NarrativeHook(EvidencedModel):
    type: Literal[
        "suspense",
        "reversal",
        "crisis",
        "revelation",
        "new_goal",
        "emotional_cliffhanger",
    ]
    content: str = Field(min_length=1)


class NarrativeObservation(EvidencedModel):
    label: str = Field(min_length=1)
    analysis: str = Field(min_length=5)


class NarrativeLayer(StrictModel):
    scene_functions: list[NarrativeObservation] = Field(default_factory=list)
    conflicts: list[NarrativeConflict] = Field(default_factory=list)
    relationship_changes: list[NarrativeObservation] = Field(default_factory=list)
    information_release: list[NarrativeObservation] = Field(default_factory=list)
    pacing: list[NarrativeObservation] = Field(default_factory=list)
    ending_hook: NarrativeHook | None = None


StyleAxis = Literal[
    "point_of_view",
    "narrative_distance",
    "sentence_rhythm",
    "paragraph_rhythm",
    "dialogue",
    "description",
    "information_flow",
    "emotion",
    "diction",
    "figurative_language",
    "transition",
    "scene_entry_exit",
]


class StyleObservation(EvidencedModel):
    """One evidenced, reusable style signal rather than a copied phrase."""

    axis: StyleAxis
    value: str = Field(min_length=1)
    analysis: str = Field(min_length=5)
    execution_rule: str = Field(min_length=5)
    originality_boundary: str = Field(min_length=5)

    @model_validator(mode="after")
    def explicit_originality_boundary(self) -> "StyleObservation":
        if not any(
            marker in self.originality_boundary
            for marker in ("不得", "不复用", "禁止", "避免", "不能照搬")
        ):
            raise ValueError("文风规则必须明确说明不可复用的原作元素")
        return self


class StyleLayer(StrictModel):
    observations: list[StyleObservation] = Field(default_factory=list)

    @model_validator(mode="after")
    def one_observation_per_axis(self) -> "StyleLayer":
        axes = [item.axis for item in self.observations]
        if len(axes) != len(set(axes)):
            raise ValueError("同一章节的同一文风维度只能给出一个综合判断")
        return self


class EvidencedTechnique(TechniqueObservation):
    evidence: list[EvidenceSpan] = Field(min_length=1)


class TechniquesLayer(StrictModel):
    techniques: list[EvidencedTechnique] = Field(default_factory=list)


LAYER_MODELS: dict[str, type[StrictModel]] = {
    "facts": FactsLayer,
    "narrative": NarrativeLayer,
    "style": StyleLayer,
    "techniques": TechniquesLayer,
}


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_segments(text: str, *, segment_chars: int = 1_200) -> list[dict[str, Any]]:
    """Return contiguous source slices whose offsets a model can cite exactly."""

    segments: list[dict[str, Any]] = []
    start = 0
    while start < len(text):
        hard_end = min(len(text), start + segment_chars)
        end = hard_end
        if hard_end < len(text):
            search_start = start + segment_chars // 2
            paragraph = text.rfind("\n\n", search_start, hard_end)
            line = text.rfind("\n", search_start, hard_end)
            sentence = max(
                text.rfind(mark, search_start, hard_end) for mark in "。！？"
            )
            cut = max(
                paragraph + 2 if paragraph >= 0 else -1,
                line + 1 if line >= 0 else -1,
                sentence + 1 if sentence >= 0 else -1,
            )
            if cut > start:
                end = cut
        segments.append({"start": start, "end": end, "text": text[start:end]})
        start = end
    return segments


def validate_evidence(result: Mapping[str, Any], text: str) -> int:
    """Reject approximate quotes; every evidence span must match frozen text."""

    count = 0

    def visit(value: Any) -> None:
        nonlocal count
        if isinstance(value, Mapping):
            if {"start", "end", "quote"}.issubset(value):
                start = int(value["start"])
                end = int(value["end"])
                quote = str(value["quote"])
                if start < 0 or end <= start or end > len(text):
                    raise ValueError("分析证据位置超出正文范围")
                if text[start:end] != quote:
                    raise ValueError("分析证据与冻结正文不一致")
                count += 1
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(result)
    return count


def _without_evidence(item: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key != "evidence"}


def combine_layers(
    *,
    chapter_title: str,
    text: str,
    layers: Mapping[str, Mapping[str, Any]],
    chapter_position: int | None = None,
) -> dict[str, Any]:
    """Build the export/UI record while preserving all evidence-backed layers."""

    facts = dict(layers["facts"])
    narrative = dict(layers["narrative"])
    style = dict(layers["style"])
    techniques = dict(layers["techniques"])
    result = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "content_hash": content_hash(text),
        "chapter_position": chapter_position,
        "chapter_title": chapter_title[:100],
        "summary": facts["summary"],
        "characters": [_without_evidence(item) for item in facts["characters"]],
        "scenes": [_without_evidence(item) for item in facts["scenes"]],
        "key_events": [_without_evidence(item) for item in facts["events"]],
        "foreshadowing": [
            {
                key: value
                for key, value in item.items()
                if key not in {"evidence", "thread_key"}
            }
            for item in facts["foreshadowing"]
        ],
        "conflicts": [_without_evidence(item) for item in narrative["conflicts"]],
        "ending_hook": (
            _without_evidence(narrative["ending_hook"])
            if narrative.get("ending_hook")
            else None
        ),
        "style_traits": [
            _without_evidence(item) for item in style["observations"]
        ],
        "techniques": [
            _without_evidence(item) for item in techniques["techniques"]
        ],
        "layers": {name: dict(value) for name, value in layers.items()},
    }
    return result
