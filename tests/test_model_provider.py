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
    resolve_model_protocol,
    settings_for_credential,
    settings_for_reasoning_policy,
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
        model_api_key="test-key",
        model_base_url="https://api.deepseek.com",
        model_name="deepseek-v4-flash",
        model_thinking=False,
        model_reasoning_effort="high",
        model_max_tokens=5_000,
        model_connect_timeout_seconds=1,
        model_read_timeout_seconds=1,
        model_max_retries=0,
    )


def test_provider_registry_exposes_curated_capability_matrix():
    providers = {provider.id: provider for provider in list_providers()}

    assert set(providers) == {
        "deepseek",
        "opencode_go",
        "openai_compatible",
        "openai",
        "gemini",
        "ollama",
    }
    assert providers["deepseek"].capabilities.thinking is True
    assert providers["opencode_go"].base_url == (
        "https://opencode.ai/zen/go/v1"
    )
    assert providers["openai_compatible"].label == "自定义 OpenAI"
    assert providers["ollama"].capabilities.api_key_required is False
    assert providers["ollama"].capabilities.configurable_base_url is True
    assert (
        providers["openai_compatible"].capabilities.configurable_base_url
        is True
    )
    assert providers["openai"].max_tokens_field == "max_completion_tokens"


def test_deepseek_payload_keeps_native_thinking_contract(tmp_path):
    settings = replace(make_settings(tmp_path), model_thinking=True)

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


def test_internal_reasoning_policy_is_task_and_capability_aware(tmp_path):
    settings = make_settings(tmp_path)

    fast = settings_for_reasoning_policy(settings, "fast")
    reasoning = settings_for_reasoning_policy(settings, "reasoning")
    deep = settings_for_reasoning_policy(settings, "deep")

    assert fast.model_thinking is False
    assert fast.model_reasoning_effort == "high"
    assert reasoning.model_thinking is True
    assert reasoning.model_reasoning_effort == "high"
    assert deep.model_thinking is True
    assert deep.model_reasoning_effort == "max"

    openai = replace(settings, model_provider="openai")
    portable = settings_for_reasoning_policy(openai, "deep")
    assert portable.model_thinking is False
    assert portable.model_reasoning_effort == "high"


def test_openai_payload_uses_compatible_fields(tmp_path):
    settings = replace(
        make_settings(tmp_path),
        model_provider="openai",
        model_name="gpt-4.1-mini",
        model_base_url=get_provider("openai").base_url,
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


def test_automatic_output_mode_omits_provider_token_field(tmp_path):
    settings = replace(make_settings(tmp_path), model_max_tokens=0)

    payload = build_chat_payload(
        settings=settings,
        messages=[{"role": "user", "content": "hello"}],
        provider_user_id="u-safe",
        max_tokens=None,
        json_object=False,
        temperature=0.2,
    )

    assert "max_tokens" not in payload
    assert "max_completion_tokens" not in payload


def test_openai_compatible_payload_uses_portable_fields(tmp_path):
    settings = replace(
        make_settings(tmp_path),
        model_provider="openai_compatible",
        model_name="custom-chat",
        model_base_url="https://llm.example.com/v1",
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


def test_opencode_go_resolves_protocol_and_auth_per_model():
    provider = get_provider("opencode_go")

    assert (
        resolve_model_protocol(provider, "deepseek-v4-flash")
        == "openai_chat"
    )
    assert (
        resolve_model_protocol(provider, "gpt-5.6-luna")
        == "openai_responses"
    )
    assert (
        resolve_model_protocol(provider, "qwen3.7-plus")
        == "anthropic_messages"
    )
    assert build_provider_headers(
        provider, "go-key", model="qwen3.7-plus"
    ) == {
        "Content-Type": "application/json",
        "x-api-key": "go-key",
        "anthropic-version": "2023-06-01",
    }
    assert build_provider_headers(
        provider, "go-key", model="deepseek-v4-flash"
    )["Authorization"] == "Bearer go-key"


def test_personal_credential_selects_fixed_provider_url(
    tmp_path,
):
    settings = settings_for_credential(
        make_settings(tmp_path),
        credential={
            "provider": "gemini",
            "model": "gemini-2.5-flash",
        },
        api_key="secret",
        model_adapter_prompt="受限时保留事件结果。",
    )

    assert settings.model_provider == "gemini"
    assert settings.model_base_url == get_provider("gemini").base_url
    assert settings.model_name == "gemini-2.5-flash"
    assert settings.model_adapter_prompt == "受限时保留事件结果。"


def test_model_adapter_is_injected_once_without_mutating_messages(
    tmp_path,
):
    messages = [
        {"role": "system", "content": "只输出 JSON。"},
        {"role": "user", "content": "分析虚构材料。"},
    ]
    settings = replace(
        make_settings(tmp_path),
        model_provider="openai_compatible",
        model_base_url="https://llm.example.com/v1",
        model_name="custom-chat",
        model_adapter_prompt=(
            "如果无法直接描述细节，保留事件因果并改用非露骨叙述。"
        ),
    )

    payload = build_chat_payload(
        settings=settings,
        messages=messages,
        provider_user_id="u-safe",
        max_tokens=800,
        json_object=True,
        temperature=0.2,
    )

    system_message = payload["messages"][0]["content"]
    assert system_message.startswith("只输出 JSON。")
    assert system_message.count("<model_adapter_policy>") == 1
    assert "保留事件因果并改用非露骨叙述" in system_message
    assert "不能覆盖本任务的系统规则" in system_message
    assert messages[0]["content"] == "只输出 JSON。"


def test_openai_compatible_credential_uses_normalized_custom_url(tmp_path):
    settings = settings_for_credential(
        make_settings(tmp_path),
        credential={
            "provider": "openai_compatible",
            "base_url": "https://llm.example.com/api/v1/",
            "model": "custom-chat",
        },
        api_key="",
    )

    assert settings.model_provider == "openai_compatible"
    assert settings.model_base_url == "https://llm.example.com/api/v1"
    assert settings.model_name == "custom-chat"
    assert settings.model_api_key is None


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
