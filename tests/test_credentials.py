from pathlib import Path

import pytest

from app.credentials import (
    CredentialCipher,
    CredentialError,
    key_hint,
    validate_api_key,
    validate_model,
)
from app.db import SCHEMA, Database
from app.migrations import _LEGACY_DEFAULT_SYSTEM_PROMPT_V24
from app.security import hash_password
from app.style_service import StyleService


def test_credential_cipher_round_trip_and_wrong_installation_key():
    cipher = CredentialCipher("installation-secret-a")
    encrypted = cipher.encrypt("sk-user-secret-1234")
    assert "sk-user-secret-1234" not in encrypted
    assert cipher.decrypt(encrypted) == "sk-user-secret-1234"

    with pytest.raises(CredentialError, match="重新保存"):
        CredentialCipher("installation-secret-b").decrypt(encrypted)


def test_api_credential_database_never_stores_plaintext(tmp_path: Path):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id = database.create_user("owner", hash_password("password-123"))
    other_user_id = database.create_user(
        "other-owner", hash_password("password-456")
    )
    cipher = CredentialCipher("installation-secret")
    raw_key = "sk-private-user-key-9876"

    database.upsert_api_credential(
        user_id=user_id,
        encrypted_key=cipher.encrypt(raw_key),
        key_hint=key_hint(raw_key),
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
    )
    database.upsert_model_adapter_prompt(
        user_id, "减少解释性总结。"
    )

    stored = database.get_api_credential(user_id)
    assert stored is not None
    assert raw_key not in stored["encrypted_key"]
    assert stored["key_hint"] == "sk-••••9876"
    assert "system_prompt" not in stored
    assert stored["base_url"] == "https://api.deepseek.com"
    assert cipher.decrypt(stored["encrypted_key"]) == raw_key
    assert (
        database.get_model_adapter_prompt(user_id)
        == "减少解释性总结。"
    )
    assert database.get_api_credential(other_user_id) is None
    assert database.get_model_adapter_prompt(other_user_id) is None


def test_credential_input_validation():
    assert validate_api_key("sk-valid-key") == "sk-valid-key"
    assert validate_model("deepseek-v4-flash") == "deepseek-v4-flash"
    assert validate_model("meta-llama/Llama-3.3@latest") == (
        "meta-llama/Llama-3.3@latest"
    )
    with pytest.raises(CredentialError):
        validate_api_key("sk-bad key")
    with pytest.raises(CredentialError):
        validate_model("http://internal/model")


def test_credentials_and_models_are_kept_per_provider(tmp_path: Path):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id = database.create_user(
        "multi-provider-owner", hash_password("password-123")
    )

    database.upsert_api_credential(
        user_id=user_id,
        provider="deepseek",
        base_url="https://api.deepseek.com",
        encrypted_key="encrypted-deepseek",
        key_hint="sk-••••deep",
        model="deepseek-chat",
        models=["deepseek-chat", "deepseek-reasoner"],
    )
    database.upsert_api_credential(
        user_id=user_id,
        provider="openai_compatible",
        base_url="https://models.example.com/v1",
        encrypted_key="encrypted-compatible",
        key_hint="••••comp",
        model="writer-pro",
        models=["writer-pro", "writer-fast"],
    )

    default = database.get_api_credential(user_id)
    deepseek = database.get_api_credential(user_id, "deepseek")
    compatible = database.get_api_credential(
        user_id, "openai_compatible"
    )
    assert default["provider"] == "openai_compatible"
    assert deepseek["key_hint"] == "sk-••••deep"
    assert compatible["base_url"] == "https://models.example.com/v1"
    assert database.list_api_models(user_id, "deepseek") == [
        "deepseek-chat",
        "deepseek-reasoner",
    ]
    assert database.list_api_models(user_id, "openai_compatible") == [
        "writer-pro",
        "writer-fast",
    ]
    assert [item["provider"] for item in database.list_api_credentials(user_id)] == [
        "openai_compatible",
        "deepseek",
    ]

    assert database.delete_api_credential(user_id, "openai_compatible")
    assert database.get_api_credential(user_id)["provider"] == "deepseek"
    assert database.list_api_models(user_id, "openai_compatible") == []


def test_v27_credential_is_migrated_to_provider_collection(tmp_path: Path):
    database = Database(tmp_path / "legacy.db")
    with database.connection() as connection:
        connection.executescript(SCHEMA)
        connection.executescript(
            """
            DROP TABLE api_credentials;
            CREATE TABLE api_credentials (
                user_id INTEGER PRIMARY KEY
                    REFERENCES users(id) ON DELETE CASCADE,
                provider TEXT NOT NULL,
                base_url TEXT NOT NULL DEFAULT '',
                encrypted_key TEXT NOT NULL,
                key_hint TEXT NOT NULL,
                model TEXT NOT NULL,
                thinking INTEGER NOT NULL DEFAULT 0,
                reasoning_effort TEXT NOT NULL DEFAULT 'high',
                system_prompt TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            INSERT INTO users(
                id, username, password_hash, created_at
            ) VALUES (
                7, 'legacy-model-owner', 'password-hash',
                '2026-01-01T00:00:00+00:00'
            );
            INSERT INTO api_credentials(
                user_id, provider, base_url, encrypted_key, key_hint,
                model, thinking, reasoning_effort, system_prompt,
                created_at, updated_at
            ) VALUES (
                7, 'deepseek', 'https://api.deepseek.com',
                'encrypted-secret', 'sk-••••1234', 'deepseek-chat',
                0, 'high', '保持克制',
                '2026-01-01T00:00:00+00:00',
                '2026-01-01T00:00:00+00:00'
            );
            """
        )
        connection.commit()

    database.initialize()

    credential = database.get_api_credential(7, "deepseek")
    assert credential["encrypted_key"] == "encrypted-secret"
    assert credential["is_default"] == 1
    assert database.list_api_models(7, "deepseek") == ["deepseek-chat"]
    assert database.get_model_adapter_prompt(7) == "保持克制"
    with database.connection() as connection:
        primary_key = [
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(api_credentials)"
            ).fetchall()
            if row["pk"]
        ]
        assert primary_key == ["user_id", "provider"]
        credential_columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(api_credentials)"
            ).fetchall()
        }
        assert "thinking" not in credential_columns
        assert "reasoning_effort" not in credential_columns
        assert "system_prompt" not in credential_columns
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_v28_reasoning_fields_are_removed_without_losing_provider_keys(
    tmp_path: Path,
):
    database = Database(tmp_path / "v28.db")
    database.initialize()
    with database.connection() as connection:
        connection.execute(
            "ALTER TABLE api_credentials "
            "ADD COLUMN thinking INTEGER NOT NULL DEFAULT 0"
        )
        connection.execute(
            "ALTER TABLE api_credentials "
            "ADD COLUMN reasoning_effort TEXT NOT NULL DEFAULT 'high'"
        )
        connection.execute(
            "ALTER TABLE api_credentials "
            "ADD COLUMN system_prompt TEXT NOT NULL DEFAULT ''"
        )
        connection.execute(
            "DELETE FROM schema_migrations WHERE version IN (29, 30)"
        )
        connection.execute(
            """
            INSERT INTO users(id, username, password_hash, created_at)
            VALUES (
                9, 'v28-owner', 'password-hash',
                '2026-01-01T00:00:00+00:00'
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO api_credentials(
                user_id, provider, base_url, encrypted_key, key_hint, model,
                system_prompt, is_default, created_at, updated_at,
                thinking, reasoning_effort
            ) VALUES (
                9, ?, ?, ?, ?, ?, ?, ?,
                '2026-01-01T00:00:00+00:00',
                '2026-01-01T00:00:00+00:00', ?, ?
            )
            """,
            [
                (
                    "deepseek",
                    "https://api.deepseek.com",
                    "encrypted-deepseek",
                    "sk-••••deep",
                    "deepseek-chat",
                    "保留",
                    0,
                    1,
                    "max",
                ),
                (
                    "openai_compatible",
                    "https://models.example.com/v1",
                    "encrypted-compatible",
                    "sk-••••comp",
                    "writer-pro",
                    _LEGACY_DEFAULT_SYSTEM_PROMPT_V24,
                    1,
                    0,
                    "high",
                ),
            ],
        )
        connection.executemany(
            """
            INSERT INTO api_models(
                user_id, provider, model, position, created_at
            ) VALUES (
                9, ?, ?, 0, '2026-01-01T00:00:00+00:00'
            )
            """,
            [
                ("deepseek", "deepseek-chat"),
                ("openai_compatible", "writer-pro"),
            ],
        )
        connection.execute(
            """
            INSERT INTO users(id, username, password_hash, created_at)
            VALUES (
                10, 'legacy-default-only', 'password-hash',
                '2026-01-01T00:00:00+00:00'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO api_credentials(
                user_id, provider, base_url, encrypted_key, key_hint, model,
                system_prompt, is_default, created_at, updated_at,
                thinking, reasoning_effort
            ) VALUES (
                10, 'openai_compatible', 'https://models.example.com/v1',
                'encrypted-default-only', 'sk-••••only', 'writer-fast',
                ?, 1,
                '2026-01-01T00:00:00+00:00',
                '2026-01-01T00:00:00+00:00', 0, 'high'
            )
            """,
            (_LEGACY_DEFAULT_SYSTEM_PROMPT_V24,),
        )
        connection.commit()

    database.initialize()

    assert (
        database.get_api_credential(9, "deepseek")["encrypted_key"]
        == "encrypted-deepseek"
    )
    assert (
        database.get_api_credential(
            9, "openai_compatible"
        )["encrypted_key"]
        == "encrypted-compatible"
    )
    assert database.list_api_models(9, "deepseek") == ["deepseek-chat"]
    assert database.list_api_models(9, "openai_compatible") == [
        "writer-pro"
    ]
    assert (
        database.get_model_adapter_prompt(9)
        == "保留"
    )
    assert database.get_model_adapter_prompt(10) is None
    assert (
        database.get_api_credential(
            10, "openai_compatible"
        )["encrypted_key"]
        == "encrypted-default-only"
    )
    with database.connection() as connection:
        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(api_credentials)"
            ).fetchall()
        }
        assert "thinking" not in columns
        assert "reasoning_effort" not in columns
        assert "system_prompt" not in columns
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_active_voice_task_protects_credentials_and_project(tmp_path: Path):
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id = database.create_user("voice-owner", hash_password("password-123"))
    project_id = "voice-project"
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="纸灯塔",
        genre="悬疑",
        premise="",
        world_setting="",
        style_guide="",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
    )
    database.upsert_api_credential(
        user_id=user_id,
        encrypted_key="encrypted-original-key",
        key_hint="sk-••••test",
        model="deepseek-v4-flash",
    )
    database.upsert_model_adapter_prompt(user_id, "原适配策略")
    StyleService(database).create_voice_suggestion(
        user_id=user_id,
        project_id=project_id,
        sample_title="合成样章",
        sample_text="只用于验证任务隔离，不包含真实作品内容。",
        author_intent="验证并发保护",
        provider="mock",
        model="mock-voice",
        credential_source="default",
    )

    with pytest.raises(ValueError, match="任务正在运行"):
        database.upsert_api_credential(
            user_id=user_id,
            encrypted_key="encrypted-replacement-key",
            key_hint="sk-••••next",
            model="deepseek-v4-pro",
        )
    with pytest.raises(ValueError, match="任务正在运行"):
        database.delete_api_credential(user_id)
    with pytest.raises(ValueError, match="任务正在运行"):
        database.upsert_model_adapter_prompt(user_id, "新适配策略")
    with pytest.raises(ValueError, match="正在排队或运行"):
        database.delete_novel_project(user_id, project_id)

    assert database.get_api_credential(user_id)["key_hint"] == "sk-••••test"
    assert database.get_model_adapter_prompt(user_id) == "原适配策略"
    assert database.get_novel_project(user_id, project_id) is not None
