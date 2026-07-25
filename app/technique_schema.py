from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


TechniqueDimension = Literal[
    "plot",
    "structure",
    "scene",
    "pacing",
    "information",
    "character",
    "dialogue",
    "language",
    "suspense",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class TechniqueObservation(StrictModel):
    name: str = Field(min_length=2, max_length=80)
    dimension: TechniqueDimension
    source_location: str = Field(min_length=1, max_length=200)
    observation: str = Field(min_length=10, max_length=600)
    effect: str = Field(min_length=10, max_length=600)
    suitable_for: List[str] = Field(default_factory=list, max_length=8)
    unsuitable_for: List[str] = Field(default_factory=list, max_length=8)
    execution_rule: str = Field(min_length=10, max_length=600)
    originality_boundary: str = Field(min_length=10, max_length=600)

    @field_validator("suitable_for", "unsuitable_for")
    @classmethod
    def validate_short_list(cls, value: List[str]) -> List[str]:
        cleaned: List[str] = []
        seen: set[str] = set()
        for item in value:
            text = item.strip()
            if not text or text in seen:
                continue
            if len(text) > 160:
                raise ValueError("适用条件单项不能超过 160 个字符")
            seen.add(text)
            cleaned.append(text)
        return cleaned

    @field_validator("originality_boundary")
    @classmethod
    def require_explicit_boundary(cls, value: str) -> str:
        if not any(
            marker in value
            for marker in ("不得", "不复用", "禁止", "避免", "不能照搬")
        ):
            raise ValueError(
                "原创性边界必须明确写出不得复用、禁止或避免的内容"
            )
        return value


class TechniqueContextItem(StrictModel):
    id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=2, max_length=80)
    dimension: TechniqueDimension
    execution_rule: str = Field(min_length=10, max_length=600)
    effect: str = Field(min_length=10, max_length=600)
    originality_boundary: str = Field(min_length=10, max_length=600)
    author_adaptation: str = Field(default="", max_length=1000)
    scope_type: Literal["project", "volume", "chapter", "scene"]
    scope_label: str = Field(default="", max_length=200)
    priority: int = Field(default=50, ge=0, le=100)
