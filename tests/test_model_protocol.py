from dataclasses import replace
from pathlib import Path

from app.config import Settings
from app.model_protocol import (
    decode_model_stream_event,
    normalize_model_response,
    prepare_model_request,
)


def make_settings(tmp_path: Path, model: str) -> Settings:
    return Settings(
        app_name="test",
        app_env="test",
        secret_key="test-secret-long-enough",
        data_dir=tmp_path,
        database_path=tmp_path / "test.db",
        cookie_secure=False,
        allow_registration=True,
        max_upload_bytes=1_000_000,
        max_text_chars=1_000_000,
        target_chapter_chars=10_000,
        max_chapter_chars=30_000,
        model_provider="opencode_go",
        model_api_key="go-key",
        model_base_url="https://opencode.ai/zen/go/v1",
        model_name=model,
        model_thinking=False,
        model_reasoning_effort="high",
        model_max_tokens=5_000,
        model_connect_timeout_seconds=1,
        model_read_timeout_seconds=1,
        model_max_retries=0,
    )


def canonical_payload() -> dict:
    return {
        "model": "placeholder",
        "messages": [
            {"role": "system", "content": "只输出 JSON。"},
            {"role": "user", "content": "分析文本"},
        ],
        "max_tokens": 800,
        "stream": False,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }


def tool_payload() -> dict:
    return {
        "model": "placeholder",
        "messages": [
            {"role": "system", "content": "按需使用工具。"},
            {"role": "user", "content": "读取第一章"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_read_1",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": '{"path":"book/manuscript/chapters/001.md"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_read_1",
                "name": "read",
                "content": '{"content":"第一章"}',
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "读取作品资源",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "max_tokens": 800,
        "stream": False,
    }


def test_opencode_go_openai_chat_request_keeps_compatible_shape(tmp_path):
    settings = replace(
        make_settings(tmp_path, "deepseek-v4-pro"),
        model_thinking=True,
        model_reasoning_effort="max",
    )
    payload = canonical_payload()
    payload["model"] = settings.model_name
    payload["max_tokens"] = 20_000

    prepared = prepare_model_request(settings, payload)

    assert prepared.endpoint == "chat/completions"
    assert prepared.payload["reasoning_effort"] == "max"
    assert "temperature" not in prepared.payload
    assert prepared.payload["messages"] == payload["messages"]


def test_opencode_go_responses_request_and_response_are_normalized(tmp_path):
    settings = make_settings(tmp_path, "gpt-5.6-luna")
    payload = canonical_payload()
    payload["model"] = settings.model_name

    prepared = prepare_model_request(settings, payload)
    normalized = normalize_model_response(
        prepared.protocol,
        {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": '{"ok":true}'}
                    ],
                }
            ],
            "usage": {"input_tokens": 12, "output_tokens": 5},
        },
    )

    assert prepared.endpoint == "responses"
    assert prepared.payload["input"] == payload["messages"]
    assert prepared.payload["max_output_tokens"] == 800
    assert prepared.payload["text"] == {
        "format": {"type": "json_object"}
    }
    assert normalized["choices"][0]["message"]["content"] == (
        '{"ok":true}'
    )
    assert normalized["choices"][0]["finish_reason"] == "stop"
    assert normalized["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 5,
    }


def test_opencode_go_anthropic_request_and_response_are_normalized(tmp_path):
    settings = replace(
        make_settings(tmp_path, "qwen3.7-plus"),
        model_thinking=True,
    )
    payload = canonical_payload()
    payload["model"] = settings.model_name
    payload["max_tokens"] = 20_000

    prepared = prepare_model_request(settings, payload)
    normalized = normalize_model_response(
        prepared.protocol,
        {
            "content": [{"type": "text", "text": '{"ok":true}'}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 9, "output_tokens": 4},
        },
    )

    assert prepared.endpoint == "messages"
    assert prepared.payload["messages"] == [
        {"role": "user", "content": "分析文本"}
    ]
    assert "只输出 JSON" in prepared.payload["system"]
    assert "语法有效的 JSON" in prepared.payload["system"]
    assert prepared.payload["thinking"] == {
        "type": "enabled",
        "budget_tokens": 16_000,
    }
    assert "temperature" not in prepared.payload
    assert normalized["choices"][0]["message"]["content"] == (
        '{"ok":true}'
    )
    assert normalized["choices"][0]["finish_reason"] == "stop"


def test_mixed_protocol_stream_events_become_common_deltas():
    responses = decode_model_stream_event(
        "openai_responses",
        {"type": "response.output_text.delta", "delta": "一段"},
    )
    anthropic = decode_model_stream_event(
        "anthropic_messages",
        {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "文字"},
        },
    )
    anthropic_done = decode_model_stream_event(
        "anthropic_messages",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "max_tokens"},
            "usage": {"output_tokens": 31},
        },
    )

    assert responses.content == "一段"
    assert anthropic.content == "文字"
    assert anthropic_done.finish_reason == "length"
    assert anthropic_done.usage == {
        "prompt_tokens": 0,
        "completion_tokens": 31,
    }


def test_responses_protocol_translates_native_tool_history(tmp_path):
    settings = make_settings(tmp_path, "gpt-5.6-luna")
    payload = tool_payload()
    payload["model"] = settings.model_name

    prepared = prepare_model_request(settings, payload)

    assert prepared.payload["tools"] == [
        {
            "type": "function",
            "name": "read",
            "description": "读取作品资源",
            "parameters": payload["tools"][0]["function"]["parameters"],
        }
    ]
    assert prepared.payload["input"][2] == {
        "type": "function_call",
        "call_id": "call_read_1",
        "name": "read",
        "arguments": '{"path":"book/manuscript/chapters/001.md"}',
    }
    assert prepared.payload["input"][3] == {
        "type": "function_call_output",
        "call_id": "call_read_1",
        "output": '{"content":"第一章"}',
    }


def test_openai_chat_protocol_strips_internal_tool_message_fields(tmp_path):
    settings = make_settings(tmp_path, "deepseek-v4-pro")
    payload = tool_payload()
    payload["model"] = settings.model_name
    payload["messages"][-1]["is_error"] = False

    prepared = prepare_model_request(settings, payload)

    assert prepared.protocol == "openai_chat"
    assert prepared.payload["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_read_1",
        "content": '{"content":"第一章"}',
    }


def test_anthropic_protocol_translates_native_tool_history(tmp_path):
    settings = make_settings(tmp_path, "qwen3.7-plus")
    payload = tool_payload()
    payload["model"] = settings.model_name

    prepared = prepare_model_request(settings, payload)

    assert prepared.payload["tools"][0]["input_schema"] == (
        payload["tools"][0]["function"]["parameters"]
    )
    assert prepared.payload["tool_choice"] == {
        "type": "auto",
        "disable_parallel_tool_use": True,
    }
    assert prepared.payload["messages"][1] == {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "call_read_1",
                "name": "read",
                "input": {
                    "path": "book/manuscript/chapters/001.md"
                },
            }
        ],
    }
    assert prepared.payload["messages"][2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "call_read_1",
        "content": '{"content":"第一章"}',
        "is_error": False,
    }


def test_provider_tool_responses_are_normalized_to_chat_shape():
    responses = normalize_model_response(
        "openai_responses",
        {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "grep",
                    "arguments": '{"query":"港口"}',
                }
            ],
        },
    )
    anthropic = normalize_model_response(
        "anthropic_messages",
        {
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_2",
                    "name": "read",
                    "input": {"path": "book/settings/core.json"},
                }
            ],
            "stop_reason": "tool_use",
        },
    )

    assert responses["choices"][0]["finish_reason"] == "tool_calls"
    assert responses["choices"][0]["message"]["tool_calls"][0] == {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "grep",
            "arguments": '{"query":"港口"}',
        },
    }
    assert anthropic["choices"][0]["finish_reason"] == "tool_calls"
    assert anthropic["choices"][0]["message"]["tool_calls"][0][
        "function"
    ]["arguments"] == '{"path":"book/settings/core.json"}'


def test_provider_tool_stream_events_become_common_deltas():
    chat = decode_model_stream_event(
        "openai_chat",
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {
                                    "name": "read",
                                    "arguments": '{"path":',
                                },
                            }
                        ]
                    }
                }
            ]
        },
    )
    responses = decode_model_stream_event(
        "openai_responses",
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 1,
            "delta": '{"query":"港',
        },
    )
    anthropic = decode_model_stream_event(
        "anthropic_messages",
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {
                "type": "tool_use",
                "id": "call_2",
                "name": "grep",
                "input": {},
            },
        },
    )

    assert chat.tool_call_deltas[0].call_id == "call_1"
    assert chat.tool_call_deltas[0].arguments_delta == '{"path":'
    assert responses.tool_call_deltas[0].index == 1
    assert responses.tool_call_deltas[0].arguments_delta == (
        '{"query":"港'
    )
    assert anthropic.tool_call_deltas[0].name == "grep"
