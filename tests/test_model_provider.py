from dataclasses import replace
from pathlib import Path

import pytest

from app.config import Settings
from app.model_provider import (
    ProviderConfigError,
    build_chat_payload,
    build_provider_headers,
    get_provider,
    list_providers,
    normalize_provider_base_url,
    settings_for_credential,
)


def make_settings(tmp_path: Path) -> Settings:
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
        deepseek_api_key="test-key",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-v4-flash",
        deepseek_thinking=False,
        deepseek_reasoning_effort="high",
        deepseek_max_tokens=5_000,
        deepseek_connect_timeout_seconds=1,
        deepseek_read_timeout_seconds=1,
        deepseek_max_retries=0,
    )


def test_provider_registry_exposes_curated_capability_matrix():
    providers = {provider.id: provider for provider in list_providers()}

    assert set(providers) == {
        "deepseek",
        "openai",
        "gemini",
        "ollama",
        "openai_compatible",
    }
    assert providers["deepseek"].capabilities.thinking is True
    assert providers["ollama"].capabilities.api_key_required is False
    assert providers["ollama"].capabilities.configurable_base_url is True
    assert (
        providers["openai_compatible"].capabilities.configurable_base_url
        is True
    )
    assert providers["openai"].max_tokens_field == "max_completion_tokens"


def test_deepseek_payload_keeps_native_thinking_contract(tmp_path):
    settings = replace(make_settings(tmp_path), deepseek_thinking=True)

    payload = build_chat_payload(
        settings=settings,
        messages=[{"role": "user", "content": "hello"}],
        provider_user_id="u-safe",
        max_tokens=800,
        json_object=True,
        temperature=0.2,
    )

    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "high"
    assert payload["user_id"] == "u-safe"
    assert "temperature" not in payload


def test_openai_payload_uses_compatible_fields(tmp_path):
    settings = replace(
        make_settings(tmp_path),
        model_provider="openai",
        deepseek_model="gpt-4.1-mini",
        deepseek_base_url=get_provider("openai").base_url,
    )

    payload = build_chat_payload(
        settings=settings,
        messages=[{"role": "user", "content": "hello"}],
        provider_user_id="u-safe",
        max_tokens=800,
        json_object=True,
        temperature=0.2,
    )

    assert payload["max_completion_tokens"] == 800
    assert payload["user"] == "u-safe"
    assert payload["response_format"] == {"type": "json_object"}
    assert "thinking" not in payload
    assert "max_tokens" not in payload


def test_openai_compatible_payload_uses_portable_fields(tmp_path):
    settings = replace(
        make_settings(tmp_path),
        model_provider="openai_compatible",
        deepseek_model="custom-chat",
        deepseek_base_url="https://llm.example.com/v1",
    )

    payload = build_chat_payload(
        settings=settings,
        messages=[{"role": "user", "content": "hello"}],
        provider_user_id="u-safe",
        max_tokens=800,
        json_object=True,
        temperature=0.2,
    )

    assert payload["max_tokens"] == 800
    assert payload["temperature"] == 0.2
    assert payload["response_format"] == {"type": "json_object"}
    assert "user" not in payload
    assert "user_id" not in payload
    assert "thinking" not in payload


def test_ollama_does_not_require_authorization_header():
    provider = get_provider("ollama")

    assert build_provider_headers(provider, None) == {
        "Content-Type": "application/json"
    }


def test_remote_provider_requires_key():
    with pytest.raises(ProviderConfigError, match="API Key"):
        build_provider_headers(get_provider("openai"), None)


def test_personal_credential_selects_fixed_provider_url(
    tmp_path,
):
    settings = settings_for_credential(
        make_settings(tmp_path),
        credential={
            "provider": "gemini",
            "model": "gemini-2.5-flash",
            "thinking": 0,
            "reasoning_effort": "high",
            "system_prompt": "concise",
        },
        api_key="secret",
    )

    assert settings.model_provider == "gemini"
    assert settings.deepseek_base_url == get_provider("gemini").base_url
    assert settings.deepseek_model == "gemini-2.5-flash"
    assert settings.deepseek_system_prompt == "concise"


def test_openai_compatible_credential_uses_normalized_custom_url(tmp_path):
    settings = settings_for_credential(
        make_settings(tmp_path),
        credential={
            "provider": "openai_compatible",
            "base_url": "https://llm.example.com/api/v1/",
            "model": "custom-chat",
            "thinking": 0,
            "reasoning_effort": "high",
        },
        api_key="",
    )

    assert settings.model_provider == "openai_compatible"
    assert settings.deepseek_base_url == "https://llm.example.com/api/v1"
    assert settings.deepseek_model == "custom-chat"
    assert settings.deepseek_api_key is None


def test_official_provider_ignores_submitted_url_override():
    provider = get_provider("openai")

    assert (
        normalize_provider_base_url(
            provider,
            "http://127.0.0.1:9000/v1",
        )
        == provider.base_url
    )


def test_custom_url_rejects_endpoint_path_and_private_production_target():
    provider = get_provider("ollama")

    with pytest.raises(ProviderConfigError, match="API 根路径"):
        normalize_provider_base_url(
            provider,
            "http://127.0.0.1:11434/v1/chat/completions",
        )
    with pytest.raises(ProviderConfigError, match="生产环境默认禁止"):
        normalize_provider_base_url(
            provider,
            "http://127.0.0.1:11434/v1",
            allow_private=False,
            production=True,
        )


def test_unknown_provider_is_rejected():
    with pytest.raises(ProviderConfigError, match="不支持"):
        get_provider("arbitrary-url")
