from __future__ import annotations

import base64
import hashlib
import re

from cryptography.fernet import Fernet, InvalidToken


MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,199}$")
# This protocol namespace predates the display-name change. It is intentionally
# stable: changing it would make every stored encrypted API key unreadable.
CREDENTIAL_KDF_NAMESPACE = b"xushu:credential:v1\0"

class CredentialError(ValueError):
    pass


class CredentialCipher:
    """Encrypts user API keys with an installation-scoped authenticated key."""

    def __init__(self, secret: str):
        if len(secret) < 16:
            raise ValueError("凭据加密密钥至少需要 16 个字符")
        derived = hashlib.sha256(
            CREDENTIAL_KDF_NAMESPACE + secret.encode("utf-8")
        ).digest()
        self._fernet = Fernet(base64.urlsafe_b64encode(derived))

    def encrypt(self, value: str) -> str:
        return self._fernet.encrypt(value.encode("utf-8")).decode("ascii")

    def decrypt(self, token: str) -> str:
        try:
            return self._fernet.decrypt(token.encode("ascii")).decode("utf-8")
        except (InvalidToken, UnicodeError, ValueError) as exc:
            raise CredentialError(
                "无法解密个人 API Key，请在 API 设置中重新保存"
            ) from exc


def validate_api_key(value: str) -> str:
    if value != value.strip() or any(character.isspace() for character in value):
        raise CredentialError("API Key 不能包含空格或换行")
    if not 8 <= len(value) <= 512:
        raise CredentialError("API Key 长度必须在 8–512 个字符之间")
    return value


def validate_model(value: str) -> str:
    value = value.strip()
    if "://" in value or not MODEL_RE.fullmatch(value):
        raise CredentialError(
            "模型名需为 1–200 位字母、数字或常用模型 ID 符号"
        )
    return value


def key_hint(value: str) -> str:
    prefix = "sk-" if value.startswith("sk-") else ""
    return f"{prefix}••••{value[-4:]}"
