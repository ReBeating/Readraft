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
    )

    stored = database.get_api_credential(user_id)
    assert stored is not None
    assert raw_key not in stored["encrypted_key"]
    assert stored["key_hint"] == "sk-••••9876"
    assert stored["thinking"] == 1
    assert stored["system_prompt"] == "减少解释性总结。"
    assert cipher.decrypt(stored["encrypted_key"]) == raw_key
    assert database.get_api_credential(other_user_id) is None


def test_credential_input_validation():
    assert validate_api_key("sk-valid-key") == "sk-valid-key"
    assert validate_model("deepseek-v4-flash") == "deepseek-v4-flash"
    with pytest.raises(CredentialError):
        validate_api_key("sk-bad key")
    with pytest.raises(CredentialError):
        validate_model("http://internal/model")
