from dataclasses import replace
from pathlib import Path

from app.config import Settings


def test_default_brand_and_database(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("APP_NAME", raising=False)
    monkeypatch.delenv("APP_DATABASE_PATH", raising=False)
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))

    settings = Settings.from_env()

    assert settings.app_name == "Readraft"
    assert settings.database_path == tmp_path / "readraft.db"


def test_model_output_and_product_quotas_default_to_automatic(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_MAX_TOKENS", "0")
    monkeypatch.setenv("APP_MAX_DOCUMENTS_PER_USER", "0")
    monkeypatch.setenv("APP_MAX_STORED_CHARS_PER_USER", "0")

    settings = Settings.from_env()

    assert settings.model_max_tokens == 0
    assert settings.max_documents_per_user is None
    assert settings.max_stored_chars_per_user is None
    settings.validate()
    replace(settings, model_max_tokens=100_000).validate()
    replace(settings, model_adapter_prompt="提示" * 20_001).validate()


def test_server_model_configuration_uses_provider_neutral_environment(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.setenv("MODEL_API_KEY", "server-key")
    monkeypatch.setenv("MODEL_NAME", "gpt-test")
    monkeypatch.setenv("MODEL_BASE_URL", "https://wrong.example/v1")
    monkeypatch.setenv(
        "DEEPSEEK_API_KEY",
        "ignored-old-key",  # pragma: allowlist secret -- test value
    )

    settings = Settings.from_env()

    assert settings.model_provider == "openai"
    assert settings.model_api_key == "server-key"  # pragma: allowlist secret
    assert settings.model_name == "gpt-test"
    assert settings.model_base_url == "https://api.openai.com/v1"


def test_old_provider_environment_is_not_a_runtime_alias(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_PROVIDER", "deepseek")
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ignored-old-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "ignored-old-model")

    settings = Settings.from_env()

    assert settings.model_api_key is None
    assert settings.model_name == "deepseek-v4-flash"
