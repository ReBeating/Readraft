import asyncio
import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from app.analysis_schema import ANALYSIS_JSON_EXAMPLE, ChapterAnalysis
from app.config import Settings
from app.model_client import AnalyzerError, ProviderAnalyzer


def make_settings(tmp_path: Path, retries: int = 0) -> Settings:
    return Settings(
        app_name="test",
        app_env="test",
        secret_key="test-secret",
        data_dir=tmp_path,
        database_path=tmp_path / "test.db",
        cookie_secure=False,
        allow_registration=True,
        max_upload_bytes=1_000_000,
        max_text_chars=1_000_000,
        target_chapter_chars=10_000,
        max_chapter_chars=30_000,
        model_api_key="test-key",
        model_base_url="https://api.deepseek.com",
        model_name="deepseek-v4-flash",
        model_thinking=False,
        model_reasoning_effort="high",
        model_max_tokens=5_000,
        model_connect_timeout_seconds=1,
        model_read_timeout_seconds=1,
        model_max_retries=retries,
        worker_poll_seconds=0.01,
    )


def success_response() -> dict:
    return {
        "id": "chat-1",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps(ANALYSIS_JSON_EXAMPLE, ensure_ascii=False),
                },
            }
        ],
        "usage": {"prompt_tokens": 120, "completion_tokens": 88},
    }


def test_analysis_example_covers_foreshadowing_contract():
    parsed = ChapterAnalysis.model_validate(ANALYSIS_JSON_EXAMPLE)

    assert parsed.foreshadowing
    assert parsed.foreshadowing[0].type == "setup"
    assert set(ANALYSIS_JSON_EXAMPLE["foreshadowing"][0]) == {
        "type",
        "clue",
        "interpretation",
    }


def test_deepseek_provider_request_and_validation(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers["Authorization"]
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json=success_response())

    async def scenario():
        analyzer = ProviderAnalyzer(
            make_settings(tmp_path), transport=httpx.MockTransport(handler)
        )
        try:
            response = await analyzer.analyze("第三章 雨夜来客", "正文内容", "u_abc")
        finally:
            await analyzer.close()
        return response

    response = asyncio.run(scenario())
    assert response.result.chapter_title == "第三章 雨夜来客"
    assert response.input_tokens == 120
    assert seen["authorization"] == "Bearer test-key"
    assert seen["url"] == "https://api.deepseek.com/chat/completions"
    assert seen["payload"]["model"] == "deepseek-v4-flash"
    assert seen["payload"]["response_format"] == {"type": "json_object"}
    assert seen["payload"]["thinking"] == {"type": "disabled"}
    assert seen["payload"]["user_id"] == "u_abc"


def test_openai_compatible_analyzer_uses_provider_contract(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json=success_response())

    async def scenario():
        settings = replace(
            make_settings(tmp_path),
            model_provider="openai",
            model_base_url="https://api.openai.com/v1",
            model_name="gpt-4.1-mini",
        )
        analyzer = ProviderAnalyzer(
            settings, transport=httpx.MockTransport(handler)
        )
        try:
            return analyzer, await analyzer.analyze(
                "第三章", "正文内容", "u_abc"
            )
        finally:
            await analyzer.close()

    analyzer, response = asyncio.run(scenario())
    assert response.result.summary
    assert analyzer.provider == "openai"
    assert seen["url"] == "https://api.openai.com/v1/chat/completions"
    assert seen["payload"]["max_completion_tokens"] == 5_000
    assert seen["payload"]["user"] == "u_abc"
    assert "thinking" not in seen["payload"]
    assert "max_tokens" not in seen["payload"]


@pytest.mark.parametrize(
    ("model", "expected_path", "response_body", "auth_header"),
    [
        (
            "gpt-5.6-luna",
            "/zen/go/v1/responses",
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(
                                    ANALYSIS_JSON_EXAMPLE,
                                    ensure_ascii=False,
                                ),
                            }
                        ],
                    }
                ],
                "usage": {"input_tokens": 120, "output_tokens": 88},
            },
            "authorization",
        ),
        (
            "qwen3.7-plus",
            "/zen/go/v1/messages",
            {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            ANALYSIS_JSON_EXAMPLE,
                            ensure_ascii=False,
                        ),
                    }
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 120, "output_tokens": 88},
            },
            "x-api-key",
        ),
    ],
)
def test_opencode_go_analyzer_selects_model_protocol(
    tmp_path,
    model,
    expected_path,
    response_body,
    auth_header,
):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["headers"] = request.headers
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json=response_body)

    async def scenario():
        settings = replace(
            make_settings(tmp_path),
            model_provider="opencode_go",
            model_api_key="go-test-key",
            model_base_url="https://opencode.ai/zen/go/v1",
            model_name=model,
        )
        analyzer = ProviderAnalyzer(
            settings, transport=httpx.MockTransport(handler)
        )
        try:
            return await analyzer.analyze("第三章", "正文内容", "u_abc")
        finally:
            await analyzer.close()

    response = asyncio.run(scenario())

    assert response.result.summary
    assert response.input_tokens == 120
    assert seen["path"] == expected_path
    assert seen["headers"][auth_header] == (
        "Bearer go-test-key"
        if auth_header == "authorization"
        else "go-test-key"
    )
    if model == "gpt-5.6-luna":
        assert "input" in seen["payload"]
        assert "messages" not in seen["payload"]
    else:
        assert seen["headers"]["anthropic-version"] == "2023-06-01"
        assert isinstance(seen["payload"]["system"], str)


@pytest.mark.parametrize(
    ("model", "events"),
    [
        (
            "gpt-5.6-luna",
            [
                {
                    "type": "response.output_text.delta",
                    "delta": "流式",
                },
                {
                    "type": "response.output_text.delta",
                    "delta": "内容",
                },
                {
                    "type": "response.completed",
                    "response": {
                        "status": "completed",
                        "output": [],
                        "usage": {
                            "input_tokens": 13,
                            "output_tokens": 7,
                        },
                    },
                },
            ],
        ),
        (
            "qwen3.7-plus",
            [
                {
                    "type": "message_start",
                    "message": {"usage": {"input_tokens": 13}},
                },
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "流式"},
                },
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "内容"},
                },
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 7},
                },
                {"type": "message_stop"},
            ],
        ),
    ],
)
def test_opencode_go_streaming_protocols_share_callback_contract(
    tmp_path, model, events
):
    chunks = []

    def handler(_request: httpx.Request) -> httpx.Response:
        body = "".join(
            f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            for event in events
        )
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "text/event-stream"},
        )

    async def scenario():
        settings = replace(
            make_settings(tmp_path),
            model_provider="opencode_go",
            model_api_key="go-test-key",
            model_base_url="https://opencode.ai/zen/go/v1",
            model_name=model,
        )
        analyzer = ProviderAnalyzer(
            settings, transport=httpx.MockTransport(handler)
        )
        try:
            payload = analyzer._payload(
                [{"role": "user", "content": "测试"}],
                "u_abc",
                800,
            )
            return await analyzer._post_stream(
                payload,
                on_content_delta=chunks.append,
            )
        finally:
            await analyzer.close()

    body = asyncio.run(scenario())
    content, reason, input_tokens, output_tokens = ProviderAnalyzer._extract(
        body
    )

    assert chunks == ["流式", "内容"]
    assert content == "流式内容"
    assert reason == "stop"
    assert input_tokens == 13
    assert output_tokens == 7


def test_streaming_chat_completion_reassembles_native_tool_calls(tmp_path):
    def handler(_request: httpx.Request) -> httpx.Response:
        events = [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
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
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": (
                                            '"book/settings/core.json"}'
                                        )
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 15, "completion_tokens": 6},
            },
        ]
        body = "".join(
            f"data: {json.dumps(event)}\n\n" for event in events
        ) + "data: [DONE]\n\n"
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "text/event-stream"},
        )

    async def scenario():
        analyzer = ProviderAnalyzer(
            make_settings(tmp_path), transport=httpx.MockTransport(handler)
        )
        try:
            payload = analyzer._payload(
                [{"role": "user", "content": "读取作品"}],
                "u_abc",
                800,
                json_object=False,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "read",
                            "description": "读取资源",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                tool_choice="auto",
            )
            return await analyzer._post_stream(payload)
        finally:
            await analyzer.close()

    body = asyncio.run(scenario())
    message = body["choices"][0]["message"]

    assert body["choices"][0]["finish_reason"] == "tool_calls"
    assert message["tool_calls"][0]["id"] == "call_1"
    assert message["tool_calls"][0]["function"] == {
        "name": "read",
        "arguments": '{"path":"book/settings/core.json"}',
    }
    assert body["usage"] == {
        "prompt_tokens": 15,
        "completion_tokens": 6,
    }


def test_retries_transient_status(tmp_path):
    attempts = {"count": 0}
    runtime_events = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(503, json={"error": {"message": "busy"}})
        return httpx.Response(200, json=success_response())

    async def no_sleep(_seconds: float) -> None:
        return None

    async def scenario():
        analyzer = ProviderAnalyzer(
            make_settings(tmp_path, retries=1),
            transport=httpx.MockTransport(handler),
            sleep=no_sleep,
        )
        analyzer.set_runtime_event_callback(
            lambda event_type, payload: runtime_events.append(
                (event_type, dict(payload))
            )
        )
        try:
            return await analyzer.analyze("第三章", "足够长的章节正文内容。", "u_abc")
        finally:
            await analyzer.close()

    result = asyncio.run(scenario())
    assert result.result.summary
    assert attempts["count"] == 2
    assert [event[0] for event in runtime_events] == [
        "retry_scheduled",
        "retry_resumed",
    ]
    assert runtime_events[0][1]["category"] == "http_503"


def test_retries_truncated_output_and_accumulates_usage(tmp_path):
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": '{"chapter_title":'},
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                },
            )
        return httpx.Response(200, json=success_response())

    async def scenario():
        analyzer = ProviderAnalyzer(
            make_settings(tmp_path), transport=httpx.MockTransport(handler)
        )
        try:
            return await analyzer.analyze("第三章", "正文内容", "u_abc")
        finally:
            await analyzer.close()

    result = asyncio.run(scenario())
    assert attempts["count"] == 2
    assert result.input_tokens == 220
    assert result.output_tokens == 138


def test_content_filter_is_terminal_and_preserves_usage(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "content_filter",
                        "message": {"content": ""},
                    }
                ],
                "usage": {"prompt_tokens": 42, "completion_tokens": 3},
            },
        )

    async def scenario():
        analyzer = ProviderAnalyzer(
            make_settings(tmp_path), transport=httpx.MockTransport(handler)
        )
        try:
            await analyzer.analyze("第三章", "正文内容", "u_abc")
        finally:
            await analyzer.close()

    with pytest.raises(AnalyzerError) as caught:
        asyncio.run(scenario())
    assert caught.value.input_tokens == 42
    assert caught.value.output_tokens == 3
    assert "内容安全策略" in str(caught.value)
