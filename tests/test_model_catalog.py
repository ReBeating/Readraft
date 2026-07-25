import asyncio

import httpx
import pytest

from app.model_catalog import ModelCatalogError, fetch_deepseek_models


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
