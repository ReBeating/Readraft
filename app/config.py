from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

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
    deepseek_api_key: str | None
    deepseek_base_url: str
    deepseek_model: str
    deepseek_thinking: bool
    deepseek_reasoning_effort: str
    deepseek_max_tokens: int
    deepseek_connect_timeout_seconds: int
    deepseek_read_timeout_seconds: int
    deepseek_max_retries: int
    worker_poll_seconds: float = 1.0
    max_documents_per_user: int = 50
    max_stored_chars_per_user: int = 20_000_000
    max_jobs_per_day: int = 10
    credential_encryption_key: str | None = None
    model_adapter_prompt: str = ""
    model_provider: str = "deepseek"
    max_project_archive_bytes: int = 256 * 1024 * 1024
    allow_private_model_base_urls: bool = False

    @property
    def documents_dir(self) -> Path:
        return self.data_dir / "documents"

    @property
    def novels_dir(self) -> Path:
        return self.data_dir / "novels"

    @property
    def uses_test_models(self) -> bool:
        """Return whether deterministic AI doubles may be used.

        Test doubles are deliberately restricted to the test environment. A
        development instance without a shared key must stay usable for account
        and personal-key setup, but it must never generate simulated creative
        output.
        """
        return self.app_env.lower() == "test" and not bool(
            self.deepseek_api_key
        )

    @property
    def credential_secret(self) -> str:
        return self.credential_encryption_key or self.secret_key

    @property
    def permits_private_model_base_urls(self) -> bool:
        return (
            self.app_env.lower() != "production"
            or self.allow_private_model_base_urls
        )

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = _path_from_env("APP_DATA_DIR", PROJECT_ROOT / "data")
        database_path = _path_from_env(
            "APP_DATABASE_PATH", data_dir / "novelai.db"
        )
        return cls(
            app_name=os.getenv("APP_NAME", "novelAI"),
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
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY") or None,
            deepseek_base_url=os.getenv(
                "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
            ).rstrip("/"),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            # Runtime task policy owns these internal request fields. Keep the
            # Settings attributes for provider payload construction, but do
            # not expose deployment-wide switches that can weaken every task.
            deepseek_thinking=False,
            deepseek_reasoning_effort="high",
            deepseek_max_tokens=_as_int("DEEPSEEK_MAX_TOKENS", 5_000),
            deepseek_connect_timeout_seconds=_as_int(
                "DEEPSEEK_CONNECT_TIMEOUT_SECONDS", 10
            ),
            deepseek_read_timeout_seconds=_as_int(
                "DEEPSEEK_READ_TIMEOUT_SECONDS", 300
            ),
            deepseek_max_retries=_as_int("DEEPSEEK_MAX_RETRIES", 3),
            max_documents_per_user=_as_int("APP_MAX_DOCUMENTS_PER_USER", 50),
            max_stored_chars_per_user=_as_int(
                "APP_MAX_STORED_CHARS_PER_USER", 20_000_000
            ),
            max_jobs_per_day=_as_int("APP_MAX_JOBS_PER_DAY", 10),
            credential_encryption_key=(
                os.getenv("APP_CREDENTIAL_ENCRYPTION_KEY") or None
            ),
            model_adapter_prompt=os.getenv(
                "MODEL_ADAPTER_PROMPT", ""
            ).strip(),
            model_provider=os.getenv("MODEL_PROVIDER", "deepseek"),
            max_project_archive_bytes=_as_int(
                "APP_MAX_PROJECT_ARCHIVE_BYTES", 256 * 1024 * 1024
            ),
            allow_private_model_base_urls=_as_bool(
                "APP_ALLOW_PRIVATE_MODEL_BASE_URLS",
                os.getenv("APP_ENV", "development").lower() != "production",
            ),
        )

    def validate(self) -> None:
        from .model_provider import get_provider

        positive_values = {
            "APP_MAX_UPLOAD_BYTES": self.max_upload_bytes,
            "APP_MAX_TEXT_CHARS": self.max_text_chars,
            "APP_TARGET_CHAPTER_CHARS": self.target_chapter_chars,
            "APP_MAX_CHAPTER_CHARS": self.max_chapter_chars,
            "APP_MAX_DOCUMENTS_PER_USER": self.max_documents_per_user,
            "APP_MAX_STORED_CHARS_PER_USER": self.max_stored_chars_per_user,
            "APP_MAX_JOBS_PER_DAY": self.max_jobs_per_day,
            "APP_MAX_PROJECT_ARCHIVE_BYTES": self.max_project_archive_bytes,
            "DEEPSEEK_MAX_TOKENS": self.deepseek_max_tokens,
            "DEEPSEEK_CONNECT_TIMEOUT_SECONDS": self.deepseek_connect_timeout_seconds,
            "DEEPSEEK_READ_TIMEOUT_SECONDS": self.deepseek_read_timeout_seconds,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} 必须大于 0")
        if self.target_chapter_chars > self.max_chapter_chars:
            raise ValueError("APP_TARGET_CHAPTER_CHARS 不能大于 APP_MAX_CHAPTER_CHARS")
        if not 0 <= self.deepseek_max_retries <= 8:
            raise ValueError("DEEPSEEK_MAX_RETRIES 必须在 0–8 之间")
        if not 256 <= self.deepseek_max_tokens <= 20_000:
            raise ValueError("DEEPSEEK_MAX_TOKENS 必须在 256–20000 之间")
        if self.deepseek_reasoning_effort not in {"high", "max"}:
            raise ValueError("DEEPSEEK_REASONING_EFFORT 目前仅支持 high 或 max")
        if not self.deepseek_model.strip():
            raise ValueError("DEEPSEEK_MODEL 不能为空")
        if not self.model_provider.strip():
            raise ValueError("MODEL_PROVIDER 不能为空")
        provider = get_provider(self.model_provider)
        if self.deepseek_thinking and not provider.capabilities.thinking:
            raise ValueError(
                f"{provider.label} 暂不支持 novelAI 的思考模式"
            )
        if len(self.model_adapter_prompt) > 20_000:
            raise ValueError("MODEL_ADAPTER_PROMPT 不能超过 20000 个字符")
        if len(self.credential_secret) < 16:
            raise ValueError("API 凭据加密密钥至少需要 16 个字符")
        parsed_base_url = urlparse(self.deepseek_base_url)
        if parsed_base_url.scheme not in {"http", "https"} or not parsed_base_url.netloc:
            raise ValueError("DEEPSEEK_BASE_URL 必须是完整的 HTTP(S) 地址")

        if self.app_env.lower() == "production":
            unsafe_secrets = {
                "dev-only-change-this-secret-before-deploy",
                "change-this-to-a-long-random-string",
            }
            if len(self.secret_key) < 32 or self.secret_key in unsafe_secrets:
                raise ValueError(
                    "生产环境必须设置至少 32 字符的随机 APP_SECRET_KEY"
                )
            if not self.cookie_secure:
                raise ValueError("生产环境必须设置 APP_COOKIE_SECURE=true")
            if parsed_base_url.scheme != "https":
                raise ValueError("生产环境 DEEPSEEK_BASE_URL 必须使用 HTTPS")
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
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        for directory in {
            self.data_dir,
            self.documents_dir,
            self.novels_dir,
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
