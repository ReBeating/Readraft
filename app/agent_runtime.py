from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence


class AgentRunPhase(str, Enum):
    QUEUED = "queued"
    ROUTING = "routing"
    PREPARING_CONTEXT = "preparing_context"
    MODEL = "model"
    TOOL = "tool"
    ACTION = "action"
    RETRYING = "retrying"
    AUDITING = "auditing"
    FINALIZING = "finalizing"
    COMMITTING = "committing"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


TERMINAL_AGENT_PHASES = frozenset(
    {
        AgentRunPhase.CANCELLED,
        AgentRunPhase.COMPLETED,
        AgentRunPhase.FAILED,
    }
)


_ALLOWED_TRANSITIONS: dict[AgentRunPhase, frozenset[AgentRunPhase]] = {
    AgentRunPhase.QUEUED: frozenset(
        {AgentRunPhase.ROUTING, AgentRunPhase.CANCELLED}
    ),
    AgentRunPhase.ROUTING: frozenset(
        {
            AgentRunPhase.PREPARING_CONTEXT,
            AgentRunPhase.RETRYING,
            AgentRunPhase.CANCELLING,
            AgentRunPhase.CANCELLED,
            AgentRunPhase.FAILED,
        }
    ),
    AgentRunPhase.PREPARING_CONTEXT: frozenset(
        {
            AgentRunPhase.MODEL,
            AgentRunPhase.TOOL,
            AgentRunPhase.ACTION,
            AgentRunPhase.CANCELLING,
            AgentRunPhase.CANCELLED,
            AgentRunPhase.FAILED,
        }
    ),
    AgentRunPhase.MODEL: frozenset(
        {
            AgentRunPhase.MODEL,
            AgentRunPhase.TOOL,
            AgentRunPhase.ACTION,
            AgentRunPhase.RETRYING,
            AgentRunPhase.AUDITING,
            AgentRunPhase.FINALIZING,
            AgentRunPhase.CANCELLING,
            AgentRunPhase.CANCELLED,
            AgentRunPhase.FAILED,
        }
    ),
    AgentRunPhase.TOOL: frozenset(
        {
            AgentRunPhase.MODEL,
            AgentRunPhase.TOOL,
            AgentRunPhase.ACTION,
            AgentRunPhase.AUDITING,
            AgentRunPhase.FINALIZING,
            AgentRunPhase.CANCELLING,
            AgentRunPhase.CANCELLED,
            AgentRunPhase.FAILED,
        }
    ),
    AgentRunPhase.ACTION: frozenset(
        {
            AgentRunPhase.MODEL,
            AgentRunPhase.TOOL,
            AgentRunPhase.ACTION,
            AgentRunPhase.RETRYING,
            AgentRunPhase.AUDITING,
            AgentRunPhase.FINALIZING,
            AgentRunPhase.CANCELLING,
            AgentRunPhase.CANCELLED,
            AgentRunPhase.FAILED,
        }
    ),
    AgentRunPhase.RETRYING: frozenset(
        {
            AgentRunPhase.ROUTING,
            AgentRunPhase.PREPARING_CONTEXT,
            AgentRunPhase.MODEL,
            AgentRunPhase.ACTION,
            AgentRunPhase.AUDITING,
            AgentRunPhase.FINALIZING,
            AgentRunPhase.CANCELLING,
            AgentRunPhase.CANCELLED,
            AgentRunPhase.FAILED,
        }
    ),
    AgentRunPhase.AUDITING: frozenset(
        {
            AgentRunPhase.MODEL,
            AgentRunPhase.ACTION,
            AgentRunPhase.RETRYING,
            AgentRunPhase.FINALIZING,
            AgentRunPhase.CANCELLING,
            AgentRunPhase.CANCELLED,
            AgentRunPhase.FAILED,
        }
    ),
    AgentRunPhase.FINALIZING: frozenset(
        {
            AgentRunPhase.COMMITTING,
            AgentRunPhase.COMPLETED,
            AgentRunPhase.CANCELLING,
            AgentRunPhase.CANCELLED,
            AgentRunPhase.FAILED,
        }
    ),
    AgentRunPhase.COMMITTING: frozenset(
        {
            AgentRunPhase.COMPLETED,
            AgentRunPhase.CANCELLED,
            AgentRunPhase.FAILED,
        }
    ),
    AgentRunPhase.CANCELLING: frozenset(
        {AgentRunPhase.CANCELLED, AgentRunPhase.FAILED}
    ),
    AgentRunPhase.CANCELLED: frozenset(),
    AgentRunPhase.COMPLETED: frozenset(),
    AgentRunPhase.FAILED: frozenset(),
}


class AgentRunCancelled(asyncio.CancelledError):
    """Raised at a safe checkpoint after an author requested cancellation."""


TransitionCallable = Callable[..., bool]
CancellationCheckCallable = Callable[..., bool]


@dataclass
class AgentRun:
    message_id: str
    claim_token: str
    transition_callback: TransitionCallable
    cancellation_check: CancellationCheckCallable
    phase: AgentRunPhase = AgentRunPhase.QUEUED
    retry_resume_phase: AgentRunPhase | None = None

    async def checkpoint(self) -> None:
        requested = await asyncio.to_thread(
            self.cancellation_check,
            message_id=self.message_id,
            claim_token=self.claim_token,
        )
        if requested:
            raise AgentRunCancelled("作者已停止本轮生成")

    async def transition(
        self,
        next_phase: AgentRunPhase,
        *,
        event_type: str,
        label: str,
        status: str = "running",
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        if next_phase != self.phase:
            allowed = _ALLOWED_TRANSITIONS[self.phase]
            if next_phase not in allowed:
                raise ValueError(
                    f"Agent 状态不能从 {self.phase.value} "
                    f"切换到 {next_phase.value}"
                )
        accepted = await asyncio.to_thread(
            self.transition_callback,
            message_id=self.message_id,
            claim_token=self.claim_token,
            phase=next_phase.value,
            event_type=event_type,
            status=status,
            label=label,
            payload=dict(payload or {}),
        )
        if not accepted:
            raise AgentRunCancelled("Agent 任务租约已经失效")
        self.phase = next_phase

    async def emit(
        self,
        *,
        event_type: str,
        label: str,
        status: str = "running",
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        await self.transition(
            self.phase,
            event_type=event_type,
            label=label,
            status=status,
            payload=payload,
        )

    async def handle_model_runtime_event(
        self, event_type: str, payload: Mapping[str, Any]
    ) -> None:
        if event_type == "retry_scheduled":
            self.retry_resume_phase = self.phase
            await self.transition(
                AgentRunPhase.RETRYING,
                event_type="model.retry_scheduled",
                label="模型暂时不可用，正在重试",
                status="retrying",
                payload=payload,
            )
            return
        if event_type == "retry_resumed":
            resume_phase = self.retry_resume_phase or AgentRunPhase.MODEL
            self.retry_resume_phase = None
            await self.transition(
                resume_phase,
                event_type="model.retry_resumed",
                label="继续生成",
                payload=payload,
            )
            return
        await self.emit(
            event_type=f"model.{event_type}",
            label="模型处理中",
            payload=payload,
        )


def _normalize_text(value: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value.casefold())


def _normalize_argument(value: Any, *, key: str = "") -> Any:
    if isinstance(value, Mapping):
        return {
            str(item_key): _normalize_argument(
                item_value, key=str(item_key)
            )
            for item_key, item_value in sorted(value.items())
        }
    if isinstance(value, list):
        return [_normalize_argument(item, key=key) for item in value]
    if isinstance(value, str):
        if key in {"query", "keyword", "keywords"}:
            chunks = re.findall(
                r"[a-z0-9_]+|[\u4e00-\u9fff]+", value.casefold()
            )
            return "\u241f".join(sorted(chunks))
        return _normalize_text(value)
    return value


def semantic_tool_fingerprint(
    tool_name: str, arguments: Mapping[str, Any]
) -> str:
    normalized = {
        "tool": str(tool_name),
        "arguments": _normalize_argument(dict(arguments)),
    }
    return hashlib.sha256(
        json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _observation_fingerprint(
    tool_name: str,
    status: str,
    result: Mapping[str, Any] | None,
    error: str,
) -> str:
    compact = {
        "tool": tool_name,
        "status": status,
        "result": result or {},
        "error": _normalize_text(error)[:500],
    }
    return hashlib.sha256(
        json.dumps(
            compact,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ProgressAssessment:
    made_progress: bool
    should_stop: bool
    reason: str = ""


@dataclass
class AgentProgressTracker:
    """Stops repeated work based on new evidence, not a user-facing budget."""

    max_no_progress_steps: int = 3
    max_same_failure: int = 2
    call_counts: dict[str, int] = field(default_factory=dict)
    observation_fingerprints: set[str] = field(default_factory=set)
    failure_counts: dict[str, int] = field(default_factory=dict)
    consecutive_no_progress: int = 0

    def has_seen_call(
        self, *, tool_name: str, arguments: Mapping[str, Any]
    ) -> bool:
        return self.call_counts.get(
            semantic_tool_fingerprint(tool_name, arguments), 0
        ) > 0

    def assess(
        self,
        *,
        tool_name: str,
        arguments: Mapping[str, Any],
        status: str,
        result: Mapping[str, Any] | None = None,
        error: str = "",
        mutation_completed: bool = False,
    ) -> ProgressAssessment:
        call_fingerprint = semantic_tool_fingerprint(tool_name, arguments)
        self.call_counts[call_fingerprint] = (
            self.call_counts.get(call_fingerprint, 0) + 1
        )
        observation_fingerprint = _observation_fingerprint(
            tool_name, status, result, error
        )
        new_observation = (
            observation_fingerprint not in self.observation_fingerprints
        )
        self.observation_fingerprints.add(observation_fingerprint)

        made_progress = bool(
            mutation_completed
            or (
                status == "completed"
                and new_observation
                and bool(result)
            )
        )
        if made_progress:
            self.consecutive_no_progress = 0
        else:
            self.consecutive_no_progress += 1

        if (
            self.call_counts[call_fingerprint] >= 3
            and not new_observation
            and not mutation_completed
        ):
            return ProgressAssessment(
                made_progress,
                True,
                "semantic_repeated_tool_call",
            )

        if status in {"failed", "denied"}:
            failure_key = hashlib.sha256(
                (
                    tool_name
                    + "\u241f"
                    + _normalize_text(error)[:500]
                ).encode("utf-8")
            ).hexdigest()
            self.failure_counts[failure_key] = (
                self.failure_counts.get(failure_key, 0) + 1
            )
            if self.failure_counts[failure_key] >= self.max_same_failure:
                return ProgressAssessment(
                    made_progress,
                    True,
                    "repeated_tool_failure",
                )

        if self.consecutive_no_progress >= self.max_no_progress_steps:
            return ProgressAssessment(
                made_progress,
                True,
                "no_new_evidence",
            )
        return ProgressAssessment(made_progress, False)
