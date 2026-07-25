from pathlib import Path

import pytest

from app.credentials import (
    CredentialCipher,
    CredentialError,
    key_hint,
    validate_api_key,
    validate_model,
)
from app.db import Database
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
        thinking=True,
        reasoning_effort="max",
        system_prompt="减少解释性总结。",
        base_url="https://api.deepseek.com",
    )

    stored = database.get_api_credential(user_id)
    assert stored is not None
    assert raw_key not in stored["encrypted_key"]
    assert stored["key_hint"] == "sk-••••9876"
    assert stored["thinking"] == 1
    assert stored["system_prompt"] == "减少解释性总结。"
    assert stored["base_url"] == "https://api.deepseek.com"
    assert cipher.decrypt(stored["encrypted_key"]) == raw_key
    assert database.get_api_credential(other_user_id) is None


def test_credential_input_validation():
    assert validate_api_key("sk-valid-key") == "sk-valid-key"
    assert validate_model("deepseek-v4-flash") == "deepseek-v4-flash"
    with pytest.raises(CredentialError):
        validate_api_key("sk-bad key")
    with pytest.raises(CredentialError):
        validate_model("http://internal/model")


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
        thinking=False,
        reasoning_effort="high",
    )
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
            thinking=False,
            reasoning_effort="high",
        )
    with pytest.raises(ValueError, match="任务正在运行"):
        database.delete_api_credential(user_id)
    with pytest.raises(ValueError, match="正在排队或运行"):
        database.delete_novel_project(user_id, project_id)

    assert database.get_api_credential(user_id)["key_hint"] == "sk-••••test"
    assert database.get_novel_project(user_id, project_id) is not None
