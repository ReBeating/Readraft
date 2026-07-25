import pytest

from app.security import (
    hash_password,
    stable_provider_user_id,
    validate_password,
    validate_username,
    verify_password,
)


def test_password_hash_round_trip():
    encoded = hash_password("a-good-password")
    assert encoded.startswith("pbkdf2_sha256$")
    assert verify_password("a-good-password", encoded)
    assert not verify_password("wrong-password", encoded)


def test_username_validation():
    assert validate_username("作者_01") == "作者_01"
    with pytest.raises(ValueError):
        validate_username("../bad")


def test_password_minimum():
    with pytest.raises(ValueError):
        validate_password("short")


def test_provider_user_id_is_stable_and_installation_scoped():
    first = stable_provider_user_id(7, "site-secret-a")
    assert first == stable_provider_user_id(7, "site-secret-a")
    assert first != stable_provider_user_id(7, "site-secret-b")
    assert first.startswith("u_")
