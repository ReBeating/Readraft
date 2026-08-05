from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

from .config import Settings
from .model_provider import ModelProtocol, resolve_model_protocol


class ModelProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class PreparedModelRequest:
    protocol: ModelProtocol
    endpoint: str
    payload: Dict[str, Any]


@dataclass(frozen=True)
class ModelStreamDelta:
    content: str = ""
    reasoning: str = ""
    tool_call_deltas: tuple["ModelToolCallDelta", ...] = ()
    finish_reason: str = ""
    usage: Mapping[str, int] | None = None
    done: bool = False
    completed_content: str = ""


@dataclass(frozen=True)
class ModelToolCallDelta:
    """One provider-neutral fragment of a native function call."""

    index: int = 0
    call_id: str = ""
    name: str = ""
    arguments_delta: str = ""
    arguments: str = ""
    done: bool = False


def _json_arguments(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "{}"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _parsed_arguments(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _canonical_tools(payload: Mapping[str, Any]) -> list[Dict[str, Any]]:
    raw_tools = payload.get("tools") or []
    if not isinstance(raw_tools, Sequence) or isinstance(
        raw_tools, (str, bytes)
    ):
        raise ModelProtocolError("模型请求中的工具格式不正确")
    tools: list[Dict[str, Any]] = []
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, Mapping):
            continue
        function = raw_tool.get("function")
        if raw_tool.get("type") == "function" and isinstance(
            function, Mapping
        ):
            name = str(function.get("name") or "").strip()
            if not name:
                continue
            parameters = function.get("parameters")
            tools.append(
                {
                    "name": name,
                    "description": str(function.get("description") or ""),
                    "parameters": (
                        dict(parameters)
                        if isinstance(parameters, Mapping)
                        else {"type": "object", "properties": {}}
                    ),
                    "strict": function.get("strict"),
                }
            )
            continue
        name = str(raw_tool.get("name") or "").strip()
        if not name:
            continue
        parameters = raw_tool.get(
            "parameters", raw_tool.get("input_schema")
        )
        tools.append(
            {
                "name": name,
                "description": str(raw_tool.get("description") or ""),
                "parameters": (
                    dict(parameters)
                    if isinstance(parameters, Mapping)
                    else {"type": "object", "properties": {}}
                ),
                "strict": raw_tool.get("strict"),
            }
        )
    return tools


def _canonical_tool_calls(message: Mapping[str, Any]) -> list[Dict[str, Any]]:
    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        return []
    calls: list[Dict[str, Any]] = []
    for index, raw_call in enumerate(raw_calls):
        if not isinstance(raw_call, Mapping):
            continue
        function = raw_call.get("function") or {}
        if not isinstance(function, Mapping):
            function = {}
        name = str(function.get("name") or raw_call.get("name") or "")
        if not name:
            continue
        calls.append(
            {
                "id": str(
                    raw_call.get("id")
                    or raw_call.get("call_id")
                    or f"call_{index}"
                ),
                "name": name,
                "arguments": _json_arguments(
                    function.get("arguments", raw_call.get("arguments"))
                ),
            }
        )
    return calls


def _openai_chat_messages(
    messages: Sequence[Any], *, include_reasoning: bool
) -> list[Dict[str, Any]]:
    prepared: list[Dict[str, Any]] = []
    for raw_message in messages:
        if not isinstance(raw_message, Mapping):
            continue
        role = str(raw_message.get("role") or "")
        if role in {"system", "developer", "user"}:
            prepared.append(
                {"role": role, "content": _message_text(raw_message)}
            )
            continue
        if role == "tool":
            call_id = str(raw_message.get("tool_call_id") or "")
            if call_id:
                prepared.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": _message_text(raw_message),
                    }
                )
            continue
        if role != "assistant":
            continue
        message: Dict[str, Any] = {
            "role": "assistant",
            "content": _message_text(raw_message),
        }
        calls = _canonical_tool_calls(raw_message)
        if calls:
            message["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": call["arguments"],
                    },
                }
                for call in calls
            ]
        reasoning = raw_message.get("reasoning_content")
        if include_reasoning and isinstance(reasoning, str) and reasoning:
            message["reasoning_content"] = reasoning
        prepared.append(message)
    return prepared


ANTHROPIC_PROTOCOL_DEFAULT_MAX_TOKENS = 64_000


def _optional_max_tokens(payload: Mapping[str, Any]) -> int | None:
    value = payload.get("max_completion_tokens", payload.get("max_tokens"))
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _canonical_usage(
    *, input_tokens: Any = 0, output_tokens: Any = 0
) -> Dict[str, int]:
    try:
        clean_input = max(0, int(input_tokens or 0))
    except (TypeError, ValueError):
        clean_input = 0
    try:
        clean_output = max(0, int(output_tokens or 0))
    except (TypeError, ValueError):
        clean_output = 0
    return {
        "prompt_tokens": clean_input,
        "completion_tokens": clean_output,
    }


def _anthropic_finish_reason(reason: Any) -> str:
    clean = str(reason or "")
    if clean == "tool_use":
        return "tool_calls"
    if clean in {"end_turn", "stop_sequence"}:
        return "stop"
    if clean == "max_tokens":
        return "length"
    if clean in {"refusal", "content_filter"}:
        return "content_filter"
    return clean


def _responses_finish_reason(body: Mapping[str, Any]) -> str:
    status = str(body.get("status") or "")
    if status == "completed":
        return "stop"
    if status == "incomplete":
        details = body.get("incomplete_details") or {}
        reason = (
            details.get("reason")
            if isinstance(details, Mapping)
            else ""
        )
        if reason in {"max_output_tokens", "max_tokens"}:
            return "length"
        if reason in {"content_filter", "safety"}:
            return "content_filter"
        return str(reason or "incomplete")
    if status in {"failed", "cancelled"}:
        return status
    return "stop" if body.get("output") else status


def _responses_text(body: Mapping[str, Any]) -> str:
    direct = body.get("output_text")
    if isinstance(direct, str) and direct:
        return direct
    parts: list[str] = []
    output = body.get("output") or []
    if not isinstance(output, list):
        return ""
    for item in output:
        if not isinstance(item, Mapping):
            continue
        content = item.get("content") or []
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, Mapping):
                continue
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _responses_tool_calls(body: Mapping[str, Any]) -> list[Dict[str, Any]]:
    calls: list[Dict[str, Any]] = []
    output = body.get("output") or []
    if not isinstance(output, list):
        return calls
    for index, item in enumerate(output):
        if not isinstance(item, Mapping):
            continue
        if item.get("type") not in {"function_call", "tool_call"}:
            continue
        name = str(item.get("name") or "")
        if not name:
            continue
        calls.append(
            {
                "id": str(
                    item.get("call_id")
                    or item.get("id")
                    or f"call_{index}"
                ),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": _json_arguments(item.get("arguments")),
                },
            }
        )
    return calls


def _message_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, Mapping):
            continue
        text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _responses_input(messages: Sequence[Any]) -> list[Dict[str, Any]]:
    items: list[Dict[str, Any]] = []
    for raw_message in messages:
        if not isinstance(raw_message, Mapping):
            continue
        role = str(raw_message.get("role") or "")
        if role == "tool":
            call_id = str(raw_message.get("tool_call_id") or "")
            if call_id:
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": _message_text(raw_message),
                    }
                )
            continue
        if role not in {"system", "developer", "user", "assistant"}:
            continue
        content = _message_text(raw_message)
        if content:
            items.append({"role": role, "content": content})
        if role == "assistant":
            for call in _canonical_tool_calls(raw_message):
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": call["name"],
                        "arguments": call["arguments"],
                    }
                )
    return items


def _responses_tools(
    canonical_tools: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    tools: list[Dict[str, Any]] = []
    for tool in canonical_tools:
        converted: Dict[str, Any] = {
            "type": "function",
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
        }
        if tool.get("strict") is not None:
            converted["strict"] = bool(tool["strict"])
        tools.append(converted)
    return tools


def _responses_tool_choice(value: Any) -> Any:
    if isinstance(value, str):
        return value
    if not isinstance(value, Mapping):
        return None
    function = value.get("function") or {}
    if value.get("type") == "function" and isinstance(function, Mapping):
        name = str(function.get("name") or "")
        return {"type": "function", "name": name} if name else None
    return dict(value)


def _append_anthropic_message(
    conversation: list[Dict[str, Any]],
    *,
    role: str,
    blocks: list[Dict[str, Any]],
) -> None:
    if not blocks:
        return
    if conversation and conversation[-1]["role"] == role:
        conversation[-1]["content"].extend(blocks)
        return
    conversation.append({"role": role, "content": blocks})


def _anthropic_messages(
    messages: Sequence[Any],
) -> tuple[list[str], list[Dict[str, Any]]]:
    system_parts: list[str] = []
    conversation: list[Dict[str, Any]] = []
    for raw_message in messages:
        if not isinstance(raw_message, Mapping):
            continue
        role = str(raw_message.get("role") or "")
        text = _message_text(raw_message)
        if role in {"system", "developer"}:
            if text:
                system_parts.append(text)
            continue
        if role == "tool":
            call_id = str(raw_message.get("tool_call_id") or "")
            if call_id:
                _append_anthropic_message(
                    conversation,
                    role="user",
                    blocks=[
                        {
                            "type": "tool_result",
                            "tool_use_id": call_id,
                            "content": text,
                            "is_error": bool(raw_message.get("is_error")),
                        }
                    ],
                )
            continue
        if role not in {"user", "assistant"}:
            continue
        blocks: list[Dict[str, Any]] = []
        if text:
            blocks.append({"type": "text", "text": text})
        if role == "assistant":
            for call in _canonical_tool_calls(raw_message):
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": call["name"],
                        "input": _parsed_arguments(call["arguments"]),
                    }
                )
        _append_anthropic_message(
            conversation, role=role, blocks=blocks
        )
    for message in conversation:
        blocks = message.get("content")
        if (
            isinstance(blocks, list)
            and len(blocks) == 1
            and isinstance(blocks[0], Mapping)
            and blocks[0].get("type") == "text"
        ):
            message["content"] = str(blocks[0].get("text") or "")
    return system_parts, conversation


def _anthropic_tool_choice(value: Any) -> Dict[str, Any] | None:
    if value == "auto":
        return {"type": "auto"}
    if value == "required":
        return {"type": "any"}
    if value == "none" or value is None:
        return None
    if not isinstance(value, Mapping):
        return None
    function = value.get("function") or {}
    if value.get("type") == "function" and isinstance(function, Mapping):
        name = str(function.get("name") or "")
        return {"type": "tool", "name": name} if name else None
    return dict(value)


def prepare_model_request(
    settings: Settings,
    canonical_payload: Mapping[str, Any],
) -> PreparedModelRequest:
    """Translate Readraft's canonical chat payload to the model protocol."""

    protocol = resolve_model_protocol(
        settings.model_provider, settings.model_name
    )
    payload = dict(canonical_payload)
    if protocol == "openai_chat":
        raw_messages = payload.get("messages") or []
        if not isinstance(raw_messages, list):
            raise ModelProtocolError("模型请求中的消息格式不正确")
        payload["messages"] = _openai_chat_messages(
            raw_messages,
            include_reasoning=(
                settings.model_provider in {"deepseek", "opencode_go"}
            ),
        )
        canonical_tools = _canonical_tools(payload)
        if canonical_tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"],
                        **(
                            {"strict": bool(tool["strict"])}
                            if tool.get("strict") is not None
                            else {}
                        ),
                    },
                }
                for tool in canonical_tools
            ]
        if (
            settings.model_provider == "opencode_go"
            and settings.model_thinking
        ):
            payload.pop("temperature", None)
            payload["reasoning_effort"] = (
                settings.model_reasoning_effort
            )
        return PreparedModelRequest(
            protocol=protocol,
            endpoint="chat/completions",
            payload=payload,
        )

    messages = payload.get("messages") or []
    if not isinstance(messages, list):
        raise ModelProtocolError("模型请求中的消息格式不正确")

    if protocol == "openai_responses":
        request: Dict[str, Any] = {
            "model": str(payload.get("model") or ""),
            "input": _responses_input(messages),
            "stream": bool(payload.get("stream")),
        }
        requested_max_tokens = _optional_max_tokens(payload)
        if requested_max_tokens is not None:
            request["max_output_tokens"] = requested_max_tokens
        canonical_tools = _canonical_tools(payload)
        if canonical_tools:
            request["tools"] = _responses_tools(canonical_tools)
            tool_choice = _responses_tool_choice(payload.get("tool_choice"))
            if tool_choice is not None:
                request["tool_choice"] = tool_choice
            if payload.get("parallel_tool_calls") is not None:
                request["parallel_tool_calls"] = bool(
                    payload["parallel_tool_calls"]
                )
        response_format = payload.get("response_format")
        if isinstance(response_format, Mapping):
            request["text"] = {"format": dict(response_format)}
        if settings.model_thinking:
            request["reasoning"] = {
                "effort": settings.model_reasoning_effort
            }
        elif payload.get("temperature") is not None:
            request["temperature"] = payload["temperature"]
        return PreparedModelRequest(
            protocol=protocol,
            endpoint="responses",
            payload=request,
        )

    system_parts, conversation = _anthropic_messages(messages)
    if not conversation:
        raise ModelProtocolError("模型请求缺少用户消息")
    if isinstance(payload.get("response_format"), Mapping):
        system_parts.append(
            "本次响应必须只包含一个语法有效的 JSON 对象，不要添加 Markdown 代码围栏。"
        )
    # Anthropic Messages requires max_tokens. In automatic mode use a broad
    # protocol default; unlike the other protocols the field cannot be
    # omitted. A user-supplied MODEL_MAX_TOKENS still takes precedence.
    requested_max_tokens = _optional_max_tokens(payload)
    anthropic_request: Dict[str, Any] = {
        "model": str(payload.get("model") or ""),
        "messages": conversation,
        "max_tokens": (
            requested_max_tokens
            if requested_max_tokens is not None
            else ANTHROPIC_PROTOCOL_DEFAULT_MAX_TOKENS
        ),
        "stream": bool(payload.get("stream")),
    }
    if system_parts:
        anthropic_request["system"] = "\n\n".join(system_parts)
    canonical_tools = _canonical_tools(payload)
    if canonical_tools:
        anthropic_request["tools"] = [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["parameters"],
            }
            for tool in canonical_tools
        ]
        tool_choice = _anthropic_tool_choice(payload.get("tool_choice"))
        if tool_choice is not None:
            if payload.get("parallel_tool_calls") is False:
                tool_choice["disable_parallel_tool_use"] = True
            anthropic_request["tool_choice"] = tool_choice
    if (
        settings.model_thinking
        and anthropic_request["max_tokens"] > 1_024
    ):
        anthropic_request["thinking"] = {
            "type": "enabled",
            "budget_tokens": (
                min(
                    31_999
                    if settings.model_reasoning_effort == "max"
                    else 16_000,
                    anthropic_request["max_tokens"] - 1,
                )
            ),
        }
    elif payload.get("temperature") is not None:
        anthropic_request["temperature"] = payload["temperature"]
    return PreparedModelRequest(
        protocol=protocol,
        endpoint="messages",
        payload=anthropic_request,
    )


def normalize_model_response(
    protocol: ModelProtocol,
    body: Mapping[str, Any],
) -> Dict[str, Any]:
    """Normalize provider responses to the existing chat-completion shape."""

    if protocol == "openai_chat":
        return dict(body)

    if protocol == "openai_responses":
        usage = body.get("usage") or {}
        if not isinstance(usage, Mapping):
            usage = {}
        tool_calls = _responses_tool_calls(body)
        message: Dict[str, Any] = {
            "role": "assistant",
            "content": _responses_text(body),
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        return {
            "choices": [
                {
                    "message": message,
                    "finish_reason": (
                        "tool_calls"
                        if tool_calls
                        else _responses_finish_reason(body)
                    ),
                }
            ],
            "usage": _canonical_usage(
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
            ),
        }

    content_parts: list[str] = []
    tool_calls: list[Dict[str, Any]] = []
    content = body.get("content") or []
    if isinstance(content, list):
        for index, block in enumerate(content):
            if not isinstance(block, Mapping):
                continue
            text = block.get("text")
            if isinstance(text, str):
                content_parts.append(text)
            if block.get("type") != "tool_use":
                continue
            name = str(block.get("name") or "")
            if not name:
                continue
            tool_calls.append(
                {
                    "id": str(block.get("id") or f"call_{index}"),
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": _json_arguments(block.get("input")),
                    },
                }
            )
    usage = body.get("usage") or {}
    if not isinstance(usage, Mapping):
        usage = {}
    message = {"role": "assistant", "content": "".join(content_parts)}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "choices": [
            {
                "message": message,
                "finish_reason": (
                    "tool_calls"
                    if tool_calls
                    else _anthropic_finish_reason(body.get("stop_reason"))
                ),
            }
        ],
        "usage": _canonical_usage(
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        ),
    }


def decode_model_stream_event(
    protocol: ModelProtocol,
    event: Mapping[str, Any],
) -> ModelStreamDelta:
    if isinstance(event.get("error"), Mapping):
        error = event["error"]
        raise ModelProtocolError(
            str(error.get("message") or "模型流式请求失败")
        )

    if protocol == "openai_chat":
        usage = event.get("usage")
        canonical_usage = None
        if isinstance(usage, Mapping):
            canonical_usage = _canonical_usage(
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
            )
        choices = event.get("choices") or []
        if not isinstance(choices, list) or not choices:
            return ModelStreamDelta(usage=canonical_usage)
        choice = choices[0]
        if not isinstance(choice, Mapping):
            return ModelStreamDelta(usage=canonical_usage)
        delta = choice.get("delta") or {}
        content = delta.get("content") if isinstance(delta, Mapping) else ""
        reasoning = ""
        tool_call_deltas: list[ModelToolCallDelta] = []
        if isinstance(delta, Mapping):
            raw_reasoning = delta.get(
                "reasoning_content", delta.get("reasoning")
            )
            if isinstance(raw_reasoning, str):
                reasoning = raw_reasoning
            raw_calls = delta.get("tool_calls") or []
            if isinstance(raw_calls, list):
                for fallback_index, raw_call in enumerate(raw_calls):
                    if not isinstance(raw_call, Mapping):
                        continue
                    function = raw_call.get("function") or {}
                    if not isinstance(function, Mapping):
                        function = {}
                    try:
                        index = int(
                            raw_call.get("index", fallback_index)
                        )
                    except (TypeError, ValueError):
                        index = fallback_index
                    arguments = function.get("arguments")
                    tool_call_deltas.append(
                        ModelToolCallDelta(
                            index=index,
                            call_id=str(raw_call.get("id") or ""),
                            name=str(function.get("name") or ""),
                            arguments_delta=(
                                arguments
                                if isinstance(arguments, str)
                                else ""
                            ),
                        )
                    )
        return ModelStreamDelta(
            content=content if isinstance(content, str) else "",
            reasoning=reasoning,
            tool_call_deltas=tuple(tool_call_deltas),
            finish_reason=str(choice.get("finish_reason") or ""),
            usage=canonical_usage,
        )

    event_type = str(event.get("type") or "")
    if protocol == "openai_responses":
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            return ModelStreamDelta(
                content=delta if isinstance(delta, str) else ""
            )
        if event_type in {
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        }:
            delta = event.get("delta")
            return ModelStreamDelta(
                reasoning=delta if isinstance(delta, str) else ""
            )
        if event_type == "response.output_item.added":
            item = event.get("item") or {}
            if not isinstance(item, Mapping) or item.get("type") not in {
                "function_call",
                "tool_call",
            }:
                return ModelStreamDelta()
            try:
                index = int(event.get("output_index", 0))
            except (TypeError, ValueError):
                index = 0
            return ModelStreamDelta(
                tool_call_deltas=(
                    ModelToolCallDelta(
                        index=index,
                        call_id=str(
                            item.get("call_id") or item.get("id") or ""
                        ),
                        name=str(item.get("name") or ""),
                        arguments=(
                            str(item.get("arguments") or "")
                            if item.get("arguments")
                            else ""
                        ),
                    ),
                )
            )
        if event_type == "response.function_call_arguments.delta":
            try:
                index = int(event.get("output_index", 0))
            except (TypeError, ValueError):
                index = 0
            delta = event.get("delta")
            return ModelStreamDelta(
                tool_call_deltas=(
                    ModelToolCallDelta(
                        index=index,
                        call_id=str(event.get("call_id") or ""),
                        arguments_delta=(
                            delta if isinstance(delta, str) else ""
                        ),
                    ),
                )
            )
        if event_type in {
            "response.function_call_arguments.done",
            "response.output_item.done",
        }:
            item = event.get("item") or {}
            if not isinstance(item, Mapping):
                item = {}
            if event_type == "response.output_item.done" and item.get(
                "type"
            ) not in {"function_call", "tool_call"}:
                return ModelStreamDelta()
            try:
                index = int(event.get("output_index", 0))
            except (TypeError, ValueError):
                index = 0
            return ModelStreamDelta(
                tool_call_deltas=(
                    ModelToolCallDelta(
                        index=index,
                        call_id=str(
                            event.get("call_id")
                            or item.get("call_id")
                            or item.get("id")
                            or ""
                        ),
                        name=str(item.get("name") or ""),
                        arguments=str(
                            event.get("arguments")
                            or item.get("arguments")
                            or ""
                        ),
                        done=True,
                    ),
                )
            )
        if event_type in {"response.completed", "response.incomplete"}:
            response = event.get("response") or {}
            if not isinstance(response, Mapping):
                response = {}
            normalized = normalize_model_response(protocol, response)
            choice = normalized["choices"][0]
            message = choice.get("message") or {}
            completed_calls: list[ModelToolCallDelta] = []
            if isinstance(message, Mapping):
                for index, call in enumerate(
                    _canonical_tool_calls(message)
                ):
                    completed_calls.append(
                        ModelToolCallDelta(
                            index=index,
                            call_id=call["id"],
                            name=call["name"],
                            arguments=call["arguments"],
                            done=True,
                        )
                    )
            return ModelStreamDelta(
                finish_reason=str(choice.get("finish_reason") or ""),
                usage=normalized["usage"],
                done=True,
                tool_call_deltas=tuple(completed_calls),
                completed_content=str(
                    message.get("content") or ""
                    if isinstance(message, Mapping)
                    else ""
                ),
            )
        if event_type in {"error", "response.failed"}:
            error = event.get("error") or {}
            message = (
                error.get("message")
                if isinstance(error, Mapping)
                else ""
            )
            raise ModelProtocolError(str(message or "模型流式请求失败"))
        return ModelStreamDelta()

    if event_type == "message_start":
        message = event.get("message") or {}
        usage = message.get("usage") if isinstance(message, Mapping) else {}
        if not isinstance(usage, Mapping):
            usage = {}
        return ModelStreamDelta(
            usage=_canonical_usage(
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
            )
        )
    if event_type == "content_block_delta":
        delta = event.get("delta") or {}
        text = delta.get("text") if isinstance(delta, Mapping) else ""
        if isinstance(delta, Mapping) and delta.get("type") == (
            "input_json_delta"
        ):
            partial_json = delta.get("partial_json")
            try:
                index = int(event.get("index", 0))
            except (TypeError, ValueError):
                index = 0
            return ModelStreamDelta(
                tool_call_deltas=(
                    ModelToolCallDelta(
                        index=index,
                        arguments_delta=(
                            partial_json
                            if isinstance(partial_json, str)
                            else ""
                        ),
                    ),
                )
            )
        if isinstance(delta, Mapping) and delta.get("type") == (
            "thinking_delta"
        ):
            thinking = delta.get("thinking")
            return ModelStreamDelta(
                reasoning=thinking if isinstance(thinking, str) else ""
            )
        return ModelStreamDelta(
            content=text if isinstance(text, str) else ""
        )
    if event_type == "content_block_start":
        block = event.get("content_block") or {}
        if not isinstance(block, Mapping) or block.get("type") != "tool_use":
            return ModelStreamDelta()
        try:
            index = int(event.get("index", 0))
        except (TypeError, ValueError):
            index = 0
        return ModelStreamDelta(
            tool_call_deltas=(
                ModelToolCallDelta(
                    index=index,
                    call_id=str(block.get("id") or ""),
                    name=str(block.get("name") or ""),
                    arguments=(
                        _json_arguments(block.get("input"))
                        if block.get("input")
                        else ""
                    ),
                ),
            )
        )
    if event_type == "content_block_stop":
        try:
            index = int(event.get("index", 0))
        except (TypeError, ValueError):
            index = 0
        return ModelStreamDelta(
            tool_call_deltas=(
                ModelToolCallDelta(index=index, done=True),
            )
        )
    if event_type == "message_delta":
        delta = event.get("delta") or {}
        usage = event.get("usage") or {}
        if not isinstance(delta, Mapping):
            delta = {}
        if not isinstance(usage, Mapping):
            usage = {}
        return ModelStreamDelta(
            finish_reason=_anthropic_finish_reason(
                delta.get("stop_reason")
            ),
            usage=_canonical_usage(
                output_tokens=usage.get("output_tokens")
            ),
        )
    if event_type == "message_stop":
        return ModelStreamDelta(done=True)
    if event_type == "error":
        error = event.get("error") or {}
        message = (
            error.get("message") if isinstance(error, Mapping) else ""
        )
        raise ModelProtocolError(str(message or "模型流式请求失败"))
    return ModelStreamDelta()
