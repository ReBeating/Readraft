from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .model_provider import ReasoningPolicy


QualityMode = Literal["low", "standard", "max"]
ModelTaskPolicy = Literal["fast", "discussion", "reasoning", "deep"]
ModelRole = Literal["fast", "quality"]

QUALITY_MODES: tuple[QualityMode, ...] = ("low", "standard", "max")


@dataclass(frozen=True)
class ModelRoutingDecision:
    model_role: ModelRole
    reasoning_policy: ReasoningPolicy


def normalize_quality_mode(
    value: object,
    *,
    default: QualityMode = "standard",
) -> QualityMode:
    normalized = str(value or "").strip().lower()
    if normalized in QUALITY_MODES:
        return normalized  # type: ignore[return-value]
    return default


def route_model_task(
    quality_mode: QualityMode,
    task_policy: ModelTaskPolicy,
) -> ModelRoutingDecision:
    """Resolve one user-facing quality mode into an internal model route.

    Low deliberately optimizes every call for latency and cost. Max deliberately
    applies the quality model's deepest verified reasoning contract to every
    call. Standard keeps mechanical work cheap, lets ordinary discussion use
    the fast model with reasoning, and reserves the quality model for writing,
    analysis, and long-range planning.
    """

    if quality_mode == "low":
        return ModelRoutingDecision("fast", "fast")
    if quality_mode == "max":
        return ModelRoutingDecision("quality", "deep")
    if task_policy == "fast":
        return ModelRoutingDecision("fast", "fast")
    if task_policy == "discussion":
        return ModelRoutingDecision("fast", "reasoning")
    if task_policy == "reasoning":
        return ModelRoutingDecision("quality", "reasoning")
    return ModelRoutingDecision("quality", "deep")
