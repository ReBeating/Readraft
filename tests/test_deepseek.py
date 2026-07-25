import asyncio
import json
from pathlib import Path

import httpx
import pytest

from app.analysis_schema import ANALYSIS_JSON_EXAMPLE
from app.config import Settings
from app.deepseek import AnalyzerError, DeepSeekAnalyzer


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
        deepseek_api_key="test-key",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-v4-flash",
        deepseek_thinking=False,
        deepseek_reasoning_effort="high",
        deepseek_max_tokens=5_000,
        deepseek_connect_timeout_seconds=1,
        deepseek_read_timeout_seconds=1,
        deepseek_max_retries=retries,
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


def test_deepseek_request_and_validation(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers["Authorization"]
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json=success_response())

    async def scenario():
        analyzer = DeepSeekAnalyzer(
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


def test_retries_transient_status(tmp_path):
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(503, json={"error": {"message": "busy"}})
        return httpx.Response(200, json=success_response())

    async def no_sleep(_seconds: float) -> None:
        return None

    async def scenario():
        analyzer = DeepSeekAnalyzer(
            make_settings(tmp_path, retries=1),
            transport=httpx.MockTransport(handler),
            sleep=no_sleep,
        )
        try:
            return await analyzer.analyze("第三章", "足够长的章节正文内容。", "u_abc")
        finally:
            await analyzer.close()

    result = asyncio.run(scenario())
    assert result.result.summary
    assert attempts["count"] == 2


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
        analyzer = DeepSeekAnalyzer(
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
        analyzer = DeepSeekAnalyzer(
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
