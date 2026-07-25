import asyncio

import httpx
import pytest

from app.model_catalog import (
    ModelCatalogError,
    fetch_deepseek_models,
    fetch_models,
)


def test_fetch_deepseek_models_uses_bearer_key_and_deduplicates():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("Authorization")
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "deepseek-chat"},
                    {"id": "deepseek-reasoner"},
                    {"id": "deepseek-chat"},
                    {"id": "bad model name"},
                ]
            },
        )

    result = asyncio.run(
        fetch_deepseek_models(
            api_key="sk-test-key",
            base_url="https://api.deepseek.com",
            transport=httpx.MockTransport(handler),
        )
    )
    assert result == ["deepseek-chat", "deepseek-reasoner"]
    assert seen["url"] == "https://api.deepseek.com/models"
    assert seen["authorization"] == "Bearer sk-test-key"


def test_fetch_deepseek_models_reports_invalid_key():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "invalid"}})

    with pytest.raises(ModelCatalogError, match="API Key 无效"):
        asyncio.run(
            fetch_deepseek_models(
                api_key="sk-test-key",
                base_url="https://api.deepseek.com",
                transport=httpx.MockTransport(handler),
            )
        )


def test_fetch_ollama_models_does_not_send_authorization():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"data": [{"id": "qwen3:8b"}]})

    models = asyncio.run(
        fetch_models(
            provider_id="ollama",
            api_key=None,
            transport=httpx.MockTransport(handler),
        )
    )

    assert models == ["qwen3:8b"]
    assert seen["url"] == "http://127.0.0.1:11434/v1/models"
    assert seen["authorization"] is None


def test_fetch_openai_compatible_models_uses_custom_url_and_optional_key():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"data": [{"id": "local-model"}]})

    models = asyncio.run(
        fetch_models(
            provider_id="openai_compatible",
            api_key="sk-test-key",
            base_url="https://gateway.example.com/openai/v1",
            transport=httpx.MockTransport(handler),
        )
    )

    assert models == ["local-model"]
    assert (
        seen["url"]
        == "https://gateway.example.com/openai/v1/models"
    )
    assert seen["authorization"] == "Bearer sk-test-key"
