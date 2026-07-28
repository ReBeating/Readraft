from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Mapping

from .agent_tools import (
    AGENT_TOOL_SPECS,
    MUTATING_PROPOSAL_TOOLS,
    AgentToolExecutor,
    WebSearchCallable,
    available_agent_tools,
)
from .assistant_chat import AnswerUpdateCallback, BaseAssistantChatModel
from .agent_loop_schema import AssistantIntentDecision
from .assistant_chat_schema import (
    AssistantChatResponse,
    AssistantChatResult,
)
from .assistant_chat_service import AssistantChatService
from .deepseek import AnalyzerError


class AssistantAgentOrchestrator:
    def __init__(
        self,
        service: AssistantChatService,
        *,
        web_search: WebSearchCallable | None = None,
        max_model_turns: int = 32,
    ):
        if not 2 <= max_model_turns <= 64:
            raise ValueError("Agent Loop 故障保护轮次必须在 2–64 之间")
        self.service = service
        self.max_model_turns = max_model_turns
        self.executor = AgentToolExecutor(
            service.database,
            web_search=web_search,
        )

    async def run(
        self,
        *,
        model: BaseAssistantChatModel,
        routing_model: BaseAssistantChatModel | None = None,
        role_models: Mapping[str, BaseAssistantChatModel] | None = None,
        item: Mapping[str, Any],
        payload: Mapping[str, Any],
        provider_user_id: str,
        on_answer_update: AnswerUpdateCallback | None = None,
    ) -> AssistantChatResponse:
        context = dict(payload.get("context") or {})
        runtime_conversation_context = {
            key: context.get(key)
            for key in (
                "conversation_id",
                "current_user_message_id",
                "conversation_memory",
                "conversation_history_search_available",
            )
        }
        sources = [dict(item) for item in payload.get("sources") or []]
        observations: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        accessed_sources: list[dict[str, Any]] = []
        raw_steps: list[str] = []
        total_input_tokens = 0
        total_output_tokens = 0
        dispatch_context = dict(context.get("dispatch") or {})
        if (
            str(dispatch_context.get("requested_role") or "") == "auto"
            and str(dispatch_context.get("resolved_role") or "") == "pending"
        ):
            try:
                intent_response = await (
                    routing_model or model
                ).classify_intent(
                    context=context,
                    history=list(payload.get("history") or []),
                    question=str(payload.get("question") or ""),
                    has_selected_quote=bool(
                        payload.get("selected_quote")
                    ),
                    provider_user_id=provider_user_id,
                )
                intent_decision = intent_response.decision
                total_input_tokens += intent_response.input_tokens
                total_output_tokens += intent_response.output_tokens
                raw_steps.append(intent_response.raw_response)
            except AnalyzerError as exc:
                total_input_tokens += exc.input_tokens
                total_output_tokens += exc.output_tokens
                intent_decision = AssistantIntentDecision(
                    intent="discuss",
                    confidence=0,
                    target_chapter_id=None,
                    reason=(
                        "意图分类失败，已安全回退到只读讨论："
                        + str(exc)[:300]
                    ),
                )
                raw_steps.append(
                    json.dumps(
                        {
                            "intent_router": "fallback",
                            "error": str(exc),
                            "decision": intent_decision.model_dump(
                                mode="json"
                            ),
                        },
                        ensure_ascii=False,
                    )
                )
            resolved_snapshot = self.service.resolve_pending_dispatch(
                message_id=str(item["id"]),
                claim_token=str(item["claim_token"]),
                decision=intent_decision,
            )
            context = dict(resolved_snapshot.get("context") or {})
            context.update(
                {
                    key: value
                    for key, value in runtime_conversation_context.items()
                    if value not in (None, "")
                }
            )
            sources = [
                dict(source)
                for source in resolved_snapshot.get("sources") or []
            ]

        role = str((context.get("agent") or {}).get("role") or "advisor")
        if role_models:
            model = role_models.get(role, model)
        tool_specs = available_agent_tools(context)
        public_tools = [spec.public_schema() for spec in tool_specs]
        allowed_names = {spec.name for spec in tool_specs}
        settings_patch = None
        story_plan = None
        draft = None
        rewrite = None
        completed_mutation: str | None = None
        call_fingerprints: dict[str, int] = {}
        successful_tool_calls = 0
        invalid_tool_calls = 0
        web_search_attempts = 0
        model_turns = 0
        stop_reason = "loop_safety_limit_reached"
        goal = str(
            ((context.get("dispatch") or {}).get("goal") or "answer")
        )
        required_tool = {
            "propose_settings_patch": "propose_settings_patch",
            "propose_story_plan": "propose_story_plan",
            "create_chapter_draft": "create_chapter_draft",
            "replace_selected_text": "replace_selected_text",
        }.get(goal)
        if (
            required_tool == "create_chapter_draft"
            and "read_chapter" in allowed_names
        ):
            chapter_context = await asyncio.to_thread(
                self.executor.execute,
                user_id=int(item["user_id"]),
                tool_name="read_chapter",
                arguments={},
                context=context,
                sources=sources,
                selected_quote=str(payload.get("selected_quote") or ""),
            )
            observations.append(
                {
                    "tool_name": "read_chapter",
                    "status": "completed",
                    "automatic": True,
                    "result": chapter_context.result,
                }
            )
            accessed_sources.extend(chapter_context.accessed_sources)

        def build_response(
            *,
            answer: str,
            citations: list[Any],
        ) -> AssistantChatResponse:
            result = AssistantChatResult(
                answer=answer,
                citations=citations,
                rewrite=rewrite,
                draft=draft,
                settings_patch=settings_patch,
                story_plan=story_plan,
            )
            return AssistantChatResponse(
                result=result,
                raw_response="\n".join(raw_steps),
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                provider=model.provider,
                model=model.model,
                accessed_sources=_deduplicate_sources(
                    [*sources, *accessed_sources]
                ),
                agent_trace=trace,
            )

        def record_step(
            *,
            sequence: int,
            action: str,
            decision: Mapping[str, Any],
            outcome_status: str,
            input_tokens: int,
            output_tokens: int,
            latency_ms: int,
            available_tools: list[str],
            tool_name: str = "",
            tool_label: str = "",
            error: str = "",
        ) -> None:
            self.service.record_agent_step(
                message_id=str(item["id"]),
                claim_token=str(item["claim_token"]),
                sequence=sequence,
                agent_role=role,
                action=action,
                tool_name=tool_name,
                tool_label=tool_label,
                available_tools=available_tools,
                decision=decision,
                outcome_status=outcome_status,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                error=error,
            )

        while model_turns < self.max_model_turns:
            step = model_turns + 1
            call_started = time.monotonic()
            try:
                decision_response = await model.next_action(
                    context=context,
                    history=list(payload.get("history") or []),
                    question=str(payload.get("question") or ""),
                    selected_quote=str(
                        payload.get("selected_quote") or ""
                    ),
                    available_tools=public_tools,
                    observations=observations,
                    step=step,
                    provider_user_id=provider_user_id,
                    on_answer_update=(
                        on_answer_update
                        if required_tool is None
                        else None
                    ),
                )
            except AnalyzerError as exc:
                record_step(
                    sequence=step,
                    action="fallback",
                    decision={"phase": "model_decision"},
                    outcome_status="failed",
                    input_tokens=exc.input_tokens,
                    output_tokens=exc.output_tokens,
                    latency_ms=round(
                        (time.monotonic() - call_started) * 1000
                    ),
                    available_tools=sorted(allowed_names),
                    error=str(exc),
                )
                raise
            latency_ms = round(
                (time.monotonic() - call_started) * 1000
            )
            model_turns += 1
            total_input_tokens += decision_response.input_tokens
            total_output_tokens += decision_response.output_tokens
            raw_steps.append(decision_response.raw_response)
            decision = decision_response.decision
            if decision.action == "finish":
                if required_tool and completed_mutation != required_tool:
                    error = (
                        "任务目标尚未完成，需要先调用 "
                        f"{required_tool}"
                    )
                    invalid_tool_calls += 1
                    observations.append(
                        {
                            "tool_name": "agent_goal",
                            "status": "denied",
                            "error": error,
                            "result": {"goal": goal},
                        }
                    )
                    record_step(
                        sequence=step,
                        action="finish",
                        decision=decision.model_dump(mode="json"),
                        outcome_status="denied",
                        input_tokens=decision_response.input_tokens,
                        output_tokens=decision_response.output_tokens,
                        latency_ms=latency_ms,
                        available_tools=sorted(allowed_names),
                        error=error,
                    )
                    continue
                record_step(
                    sequence=step,
                    action="finish",
                    decision=decision.model_dump(mode="json"),
                    outcome_status="completed",
                    input_tokens=decision_response.input_tokens,
                    output_tokens=decision_response.output_tokens,
                    latency_ms=latency_ms,
                    available_tools=sorted(allowed_names),
                )
                return build_response(
                    answer=str(decision.answer),
                    citations=list(decision.citations),
                )

            tool_call = decision.tool_call
            if tool_call is None:
                record_step(
                    sequence=step,
                    action="fallback",
                    decision=decision.model_dump(mode="json"),
                    outcome_status="failed",
                    input_tokens=decision_response.input_tokens,
                    output_tokens=decision_response.output_tokens,
                    latency_ms=latency_ms,
                    available_tools=sorted(allowed_names),
                    error="Agent 调度缺少工具调用",
                )
                raise AnalyzerError(
                    "Agent 调度缺少工具调用",
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                )
            name = tool_call.name
            arguments = dict(tool_call.arguments)
            fingerprint = json.dumps(
                {"name": name, "arguments": arguments},
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            call_fingerprints[fingerprint] = (
                call_fingerprints.get(fingerprint, 0) + 1
            )
            spec = AGENT_TOOL_SPECS.get(name)
            status = "running"
            denied_error = ""
            if name not in allowed_names:
                status = "denied"
                denied_error = "当前角色没有调用该工具的权限"
            elif (
                name in MUTATING_PROPOSAL_TOOLS
                and completed_mutation is not None
            ):
                status = "denied"
                denied_error = (
                    "同一轮只能创建一个写入候选，已有工具："
                    + completed_mutation
                )
            elif call_fingerprints[fingerprint] > 1:
                status = "denied"
                denied_error = "相同工具调用已经执行过，请根据结果继续"

            call_id = self.service.start_tool_call(
                message_id=str(item["id"]),
                claim_token=str(item["claim_token"]),
                sequence=step,
                agent_role=role,
                tool_name=name,
                tool_label=(spec.label if spec else name),
                capability=(spec.capability if spec else ""),
                read_only=(spec.read_only if spec else True),
                arguments=arguments,
                initial_status=status,
                error=denied_error,
            )
            if status == "denied":
                invalid_tool_calls += 1
                observation = {
                    "tool_name": name,
                    "status": "denied",
                    "error": denied_error,
                }
                observations.append(observation)
                trace.append(
                    {
                        "sequence": step,
                        "tool_name": name,
                        "label": spec.label if spec else name,
                        "status": "denied",
                        "read_only": spec.read_only if spec else True,
                    }
                )
                record_step(
                    sequence=step,
                    action="call_tool",
                    tool_name=name,
                    tool_label=spec.label if spec else name,
                    decision=decision.model_dump(mode="json"),
                    outcome_status="denied",
                    input_tokens=decision_response.input_tokens,
                    output_tokens=decision_response.output_tokens,
                    latency_ms=latency_ms,
                    available_tools=sorted(allowed_names),
                    error=denied_error,
                )
                if call_fingerprints[fingerprint] > 1:
                    stop_reason = "repeated_tool_call"
                    break
                continue

            if name == "search_web":
                web_search_attempts += 1
            try:
                execution = await asyncio.to_thread(
                    self.executor.execute,
                    user_id=int(item["user_id"]),
                    tool_name=name,
                    arguments=arguments,
                    context=context,
                    sources=sources,
                    selected_quote=str(
                        payload.get("selected_quote") or ""
                    ),
                )
            except (OSError, PermissionError, UnicodeError, ValueError) as exc:
                error = str(exc)
                self.service.finish_tool_call(
                    call_id=call_id,
                    message_id=str(item["id"]),
                    claim_token=str(item["claim_token"]),
                    status="failed",
                    result={},
                    error=error,
                )
                observations.append(
                    {
                        "tool_name": name,
                        "status": "failed",
                        "error": error,
                    }
                )
                trace.append(
                    {
                        "sequence": step,
                        "tool_name": name,
                        "label": spec.label if spec else name,
                        "status": "failed",
                        "read_only": spec.read_only if spec else True,
                    }
                )
                invalid_tool_calls += 1
                record_step(
                    sequence=step,
                    action="call_tool",
                    tool_name=name,
                    tool_label=spec.label if spec else name,
                    decision=decision.model_dump(mode="json"),
                    outcome_status="failed",
                    input_tokens=decision_response.input_tokens,
                    output_tokens=decision_response.output_tokens,
                    latency_ms=latency_ms,
                    available_tools=sorted(allowed_names),
                    error=error,
                )
                continue

            self.service.finish_tool_call(
                call_id=call_id,
                message_id=str(item["id"]),
                claim_token=str(item["claim_token"]),
                status="completed",
                result=execution.result,
                error="",
            )
            observations.append(
                {
                    "tool_name": name,
                    "status": "completed",
                    "result": execution.result,
                }
            )
            trace.append(
                {
                    "sequence": step,
                    "tool_name": name,
                    "label": spec.label if spec else name,
                    "status": "completed",
                    "read_only": spec.read_only if spec else True,
                }
            )
            successful_tool_calls += 1
            record_step(
                sequence=step,
                action="call_tool",
                tool_name=name,
                tool_label=spec.label if spec else name,
                decision=decision.model_dump(mode="json"),
                outcome_status="completed",
                input_tokens=decision_response.input_tokens,
                output_tokens=decision_response.output_tokens,
                latency_ms=latency_ms,
                available_tools=sorted(allowed_names),
            )
            accessed_sources.extend(execution.accessed_sources)
            if execution.settings_patch is not None:
                settings_patch = execution.settings_patch
                completed_mutation = name
            if execution.story_plan is not None:
                story_plan = execution.story_plan
                completed_mutation = name
            if execution.draft is not None:
                draft = execution.draft
                completed_mutation = name
            if execution.rewrite is not None:
                rewrite = execution.rewrite
                completed_mutation = name
            if completed_mutation is not None:
                stop_reason = "mutation_completed"
                break

        finalization_observations = [
            *observations,
            {
                "tool_name": "agent_loop",
                "status": "finalization_required",
                "result": {
                    "reason": stop_reason,
                    "goal": goal,
                    "completed_mutation": completed_mutation,
                    "model_turns": model_turns,
                    "successful_tool_calls": successful_tool_calls,
                    "invalid_tool_calls": invalid_tool_calls,
                    "web_search_attempts": web_search_attempts,
                    "instruction": (
                        "不得再调用工具；请根据已有观察立即给出最终回答。"
                    ),
                },
            },
        ]
        final_step = model_turns + 1
        final_started = time.monotonic()
        fallback_sequence = final_step
        fallback_input_tokens = 0
        fallback_output_tokens = 0
        fallback_latency_ms = 0
        try:
            decision_response = await model.next_action(
                context=context,
                history=list(payload.get("history") or []),
                question=str(payload.get("question") or ""),
                selected_quote=str(payload.get("selected_quote") or ""),
                available_tools=[],
                observations=finalization_observations,
                step=final_step,
                provider_user_id=provider_user_id,
                on_answer_update=on_answer_update,
            )
            final_latency_ms = round(
                (time.monotonic() - final_started) * 1000
            )
            total_input_tokens += decision_response.input_tokens
            total_output_tokens += decision_response.output_tokens
            raw_steps.append(decision_response.raw_response)
            decision = decision_response.decision
            if decision.action == "finish":
                record_step(
                    sequence=final_step,
                    action="finish",
                    decision=decision.model_dump(mode="json"),
                    outcome_status="completed",
                    input_tokens=decision_response.input_tokens,
                    output_tokens=decision_response.output_tokens,
                    latency_ms=final_latency_ms,
                    available_tools=[],
                )
                return build_response(
                    answer=str(decision.answer),
                    citations=list(decision.citations),
                )
            final_error = "收束阶段禁止继续调用工具"
            record_step(
                sequence=final_step,
                action="call_tool",
                tool_name=(
                    decision.tool_call.name
                    if decision.tool_call
                    else ""
                ),
                tool_label=(
                    decision.tool_call.name
                    if decision.tool_call
                    else ""
                ),
                decision=decision.model_dump(mode="json"),
                outcome_status="denied",
                input_tokens=decision_response.input_tokens,
                output_tokens=decision_response.output_tokens,
                latency_ms=final_latency_ms,
                available_tools=[],
                error=final_error,
            )
            fallback_sequence = final_step + 1
        except AnalyzerError as exc:
            final_latency_ms = round(
                (time.monotonic() - final_started) * 1000
            )
            total_input_tokens += exc.input_tokens
            total_output_tokens += exc.output_tokens
            final_error = str(exc)
            fallback_input_tokens = exc.input_tokens
            fallback_output_tokens = exc.output_tokens
            fallback_latency_ms = final_latency_ms
            raw_steps.append(
                json.dumps(
                    {
                        "finalization": "fallback",
                        "error": final_error,
                    },
                    ensure_ascii=False,
                )
            )

        fallback_answer = _fallback_answer_after_step_limit(
            observations=observations,
            trace=trace,
            settings_patch=settings_patch,
            story_plan=story_plan,
            draft=draft,
            rewrite=rewrite,
        )
        raw_steps.append(
            json.dumps(
                {
                    "action": "finish",
                    "answer": fallback_answer,
                    "fallback": True,
                },
                ensure_ascii=False,
            )
        )
        record_step(
            sequence=fallback_sequence,
            action="fallback",
            decision={
                "action": "finish",
                "answer": fallback_answer,
                "fallback": True,
            },
            outcome_status="fallback",
            input_tokens=fallback_input_tokens,
            output_tokens=fallback_output_tokens,
            latency_ms=fallback_latency_ms,
            available_tools=[],
            error=final_error,
        )
        return build_response(answer=fallback_answer, citations=[])


def _deduplicate_sources(
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in sources:
        source_id = str(source.get("source_id") or "")
        if not source_id or source_id in seen:
            continue
        seen.add(source_id)
        result.append(dict(source))
    return result


def _fallback_answer_after_step_limit(
    *,
    observations: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    settings_patch: Any,
    story_plan: Any,
    draft: Any,
    rewrite: Any,
) -> str:
    if rewrite is not None:
        return (
            "已生成选区修订候选，并保留本轮读取结果。为避免继续重复调用工具，"
            "本轮已自动收束；你可以查看修订内容，或继续告诉我还要调整什么。"
        )
    if draft is not None:
        return (
            "已生成章节候选稿，并保留本轮读取结果。为避免继续重复调用工具，"
            "本轮已自动收束；你可以直接检查工作稿，或继续提出修改要求。"
        )
    if settings_patch is not None:
        return (
            "已把本轮讨论整理成候选设定。为避免继续重复调用工具，本轮已自动"
            "收束；候选仍需由你决定是否应用。"
        )
    if story_plan is not None:
        return (
            "已把已有设定整理成故事规划候选。为避免继续重复调用工具，本轮已"
            "自动收束；你可以检查转折与兑现关系，再决定是否采用。"
        )

    completed_labels = [
        str(item.get("label") or item.get("tool_name") or "")
        for item in trace
        if item.get("status") == "completed"
    ]
    failed_count = sum(
        1
        for item in observations
        if item.get("status") in {"failed", "denied"}
    )
    if completed_labels:
        completed = "、".join(completed_labels[-3:])
        suffix = (
            f"，另有 {failed_count} 次调用未执行"
            if failed_count
            else ""
        )
        return (
            f"本轮已完成{completed}{suffix}，并已自动收束。当前没有产生正文或"
            "设定候选；你可以基于这些结果继续追问，我会从现有进度继续。"
        )
    return (
        "本轮没有取得可用的工具结果，已自动停止继续调用，避免陷入循环。"
        "请换一种说法继续提问，或指定希望我讨论、创作、修订还是拆解。"
    )
