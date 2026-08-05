from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _as_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数") from exc


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    path = Path(value).expanduser() if value else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_env: str
    secret_key: str
    data_dir: Path
    database_path: Path
    cookie_secure: bool
    allow_registration: bool
    max_upload_bytes: int
    max_text_chars: int
    target_chapter_chars: int
    max_chapter_chars: int
    model_api_key: str | None
    model_base_url: str
    model_name: str
    model_thinking: bool
    model_reasoning_effort: str
    model_max_tokens: int
    model_connect_timeout_seconds: int
    model_read_timeout_seconds: int
    model_max_retries: int
    worker_poll_seconds: float = 1.0
    max_documents_per_user: int | None = None
    max_stored_chars_per_user: int | None = None
    credential_encryption_key: str | None = None
    model_adapter_prompt: str = ""
    model_provider: str = "deepseek"
    max_work_archive_bytes: int = 256 * 1024 * 1024
    allow_private_model_base_urls: bool = False
    exa_api_key: str | None = None
    chapter_edit_buffer_max_chars: int = 200_000
    max_edit_buffer_chars_per_user: int = 2_000_000
    edit_buffer_retention_days: int = 30

    @property
    def documents_dir(self) -> Path:
        return self.data_dir / "documents"

    @property
    def novels_dir(self) -> Path:
        return self.data_dir / "novels"

    @property
    def import_previews_dir(self) -> Path:
        return self.data_dir / "import-previews"

    @property
    def uses_test_models(self) -> bool:
        """Return whether deterministic AI doubles may be used.

        Test doubles are deliberately restricted to the test environment. A
        development instance without a shared key must stay usable for account
        and personal-key setup, but it must never generate simulated creative
        output.
        """
        return self.app_env.lower() == "test" and not bool(self.model_api_key)

    @property
    def credential_secret(self) -> str:
        return self.credential_encryption_key or self.secret_key

    @property
    def permits_private_model_base_urls(self) -> bool:
        return (
            self.app_env.lower() != "production" or self.allow_private_model_base_urls
        )

    @classmethod
    def from_env(cls) -> "Settings":
        from .model_provider import get_provider

        data_dir = _path_from_env("APP_DATA_DIR", PROJECT_ROOT / "data")
        database_path = _path_from_env("APP_DATABASE_PATH", data_dir / "readraft.db")
        model_provider = os.getenv("MODEL_PROVIDER", "deepseek").strip().lower()
        provider = get_provider(model_provider)
        configured_base_url = os.getenv("MODEL_BASE_URL")
        model_base_url = (
            configured_base_url
            if provider.capabilities.configurable_base_url
            and configured_base_url is not None
            else provider.base_url
        ).rstrip("/")
        default_model = "deepseek-v4-flash" if model_provider == "deepseek" else ""
        return cls(
            app_name=os.getenv("APP_NAME", "Readraft"),
            app_env=os.getenv("APP_ENV", "development"),
            secret_key=os.getenv(
                "APP_SECRET_KEY", "dev-only-change-this-secret-before-deploy"
            ),
            data_dir=data_dir,
            database_path=database_path,
            cookie_secure=_as_bool("APP_COOKIE_SECURE", False),
            allow_registration=_as_bool("APP_ALLOW_REGISTRATION", True),
            max_upload_bytes=_as_int("APP_MAX_UPLOAD_BYTES", 20 * 1024 * 1024),
            max_text_chars=_as_int("APP_MAX_TEXT_CHARS", 4_000_000),
            target_chapter_chars=_as_int("APP_TARGET_CHAPTER_CHARS", 12_000),
            max_chapter_chars=_as_int("APP_MAX_CHAPTER_CHARS", 30_000),
            model_api_key=os.getenv("MODEL_API_KEY") or None,
            model_base_url=model_base_url,
            model_name=os.getenv("MODEL_NAME", default_model).strip(),
            # Runtime task policy owns these internal request fields. Keep the
            # Settings attributes for provider payload construction, but do
            # not expose deployment-wide switches that can weaken every task.
            model_thinking=False,
            model_reasoning_effort="high",
            # 0 means automatic: do not impose an application-level output
            # ceiling when the selected provider protocol allows omission.
            model_max_tokens=_as_int("MODEL_MAX_TOKENS", 0),
            model_connect_timeout_seconds=_as_int("MODEL_CONNECT_TIMEOUT_SECONDS", 10),
            model_read_timeout_seconds=_as_int("MODEL_READ_TIMEOUT_SECONDS", 300),
            model_max_retries=_as_int("MODEL_MAX_RETRIES", 3),
            max_documents_per_user=(
                _as_int("APP_MAX_DOCUMENTS_PER_USER", 0) or None
            ),
            max_stored_chars_per_user=(
                _as_int("APP_MAX_STORED_CHARS_PER_USER", 0) or None
            ),
            credential_encryption_key=(
                os.getenv("APP_CREDENTIAL_ENCRYPTION_KEY") or None
            ),
            model_adapter_prompt=os.getenv("MODEL_ADAPTER_PROMPT", "").strip(),
            model_provider=model_provider,
            max_work_archive_bytes=_as_int(
                "APP_MAX_WORK_ARCHIVE_BYTES", 256 * 1024 * 1024
            ),
            allow_private_model_base_urls=_as_bool(
                "APP_ALLOW_PRIVATE_MODEL_BASE_URLS",
                os.getenv("APP_ENV", "development").lower() != "production",
            ),
            exa_api_key=os.getenv("EXA_API_KEY") or None,
            chapter_edit_buffer_max_chars=_as_int(
                "APP_CHAPTER_EDIT_BUFFER_MAX_CHARS", 200_000
            ),
            max_edit_buffer_chars_per_user=_as_int(
                "APP_MAX_EDIT_BUFFER_CHARS_PER_USER", 2_000_000
            ),
            edit_buffer_retention_days=_as_int(
                "APP_EDIT_BUFFER_RETENTION_DAYS", 30
            ),
        )

    def validate(self) -> None:
        from .model_provider import (
            get_provider,
            normalize_provider_base_url,
        )

        positive_values = {
            "APP_MAX_UPLOAD_BYTES": self.max_upload_bytes,
            "APP_MAX_TEXT_CHARS": self.max_text_chars,
            "APP_TARGET_CHAPTER_CHARS": self.target_chapter_chars,
            "APP_MAX_CHAPTER_CHARS": self.max_chapter_chars,
            "APP_MAX_WORK_ARCHIVE_BYTES": self.max_work_archive_bytes,
            "APP_CHAPTER_EDIT_BUFFER_MAX_CHARS": (
                self.chapter_edit_buffer_max_chars
            ),
            "APP_MAX_EDIT_BUFFER_CHARS_PER_USER": (
                self.max_edit_buffer_chars_per_user
            ),
            "APP_EDIT_BUFFER_RETENTION_DAYS": self.edit_buffer_retention_days,
            "MODEL_CONNECT_TIMEOUT_SECONDS": self.model_connect_timeout_seconds,
            "MODEL_READ_TIMEOUT_SECONDS": self.model_read_timeout_seconds,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} 必须大于 0")
        optional_positive_values = {
            "APP_MAX_DOCUMENTS_PER_USER": self.max_documents_per_user,
            "APP_MAX_STORED_CHARS_PER_USER": self.max_stored_chars_per_user,
        }
        for name, value in optional_positive_values.items():
            if value is not None and value <= 0:
                raise ValueError(f"{name} 必须留空、设为 0，或大于 0")
        if self.target_chapter_chars > self.max_chapter_chars:
            raise ValueError("APP_TARGET_CHAPTER_CHARS 不能大于 APP_MAX_CHAPTER_CHARS")
        if not 0 <= self.model_max_retries <= 8:
            raise ValueError("MODEL_MAX_RETRIES 必须在 0–8 之间")
        if self.model_max_tokens < 0 or 0 < self.model_max_tokens < 256:
            raise ValueError("MODEL_MAX_TOKENS 必须为 0（自动）或至少 256")
        if self.model_reasoning_effort not in {"high", "max"}:
            raise ValueError("MODEL_REASONING_EFFORT 目前仅支持 high 或 max")
        if not self.model_name.strip():
            raise ValueError("MODEL_NAME 不能为空")
        if not self.model_provider.strip():
            raise ValueError("MODEL_PROVIDER 不能为空")
        provider = get_provider(self.model_provider)
        if self.model_thinking and not provider.capabilities.thinking:
            raise ValueError(f"{provider.label} 暂不支持 Readraft 的思考模式")
        if len(self.credential_secret) < 16:
            raise ValueError("API 凭据加密密钥至少需要 16 个字符")
        normalized_base_url = normalize_provider_base_url(
            provider,
            self.model_base_url,
            allow_private=self.permits_private_model_base_urls,
            production=self.app_env.lower() == "production",
        )
        if normalized_base_url != self.model_base_url.rstrip("/"):
            raise ValueError("MODEL_BASE_URL 与所选模型服务商不匹配")

        if self.app_env.lower() == "production":
            unsafe_secrets = {
                "dev-only-change-this-secret-before-deploy",
                "change-this-to-a-long-random-string",
            }
            if len(self.secret_key) < 32 or self.secret_key in unsafe_secrets:
                raise ValueError("生产环境必须设置至少 32 字符的随机 APP_SECRET_KEY")
            if not self.cookie_secure:
                raise ValueError("生产环境必须设置 APP_COOKIE_SECURE=true")
            if (
                self.credential_encryption_key is not None
                and len(self.credential_encryption_key) < 32
            ):
                raise ValueError(
                    "生产环境 APP_CREDENTIAL_ENCRYPTION_KEY 至少需要 32 个字符"
                )

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.novels_dir.mkdir(parents=True, exist_ok=True)
        self.import_previews_dir.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        for directory in {
            self.data_dir,
            self.documents_dir,
            self.novels_dir,
            self.import_previews_dir,
            self.database_path.parent,
        }:
            directory.chmod(0o700)
        for database_file in (
            self.database_path,
            Path(f"{self.database_path}-wal"),
            Path(f"{self.database_path}-shm"),
        ):
            if database_file.exists():
                database_file.chmod(0o600)
