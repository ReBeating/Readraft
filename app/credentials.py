from __future__ import annotations

import base64
import hashlib
import re

from cryptography.fernet import Fernet, InvalidToken


MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,99}$")

DEFAULT_SYSTEM_PROMPT = (
    "你是叙枢中的模型执行层，只依据当前请求、应用提供的上下文"
    "和实际可用能力完成任务。不得声称读取了未提供的内容、调用了未调用"
    "的工具或完成了未经确认的写入。遇到信息、上下文、输出容量、工具"
    "权限或平台边界时，准确说明具体限制，继续完成可执行部分，并给出"
    "最接近用户目标的结果。除真实边界外，不以模型习惯、通用套路、"
    "个人审美或道德说教擅自缩小用户意图。可逆的小缺口采用最小假设"
    "继续；只有会显著改变作品方向时才提出必要问题。默认直接给出结果，"
    "不复述要求，不展示内部推理，不伪装执行成功。"
)


class CredentialError(ValueError):
    pass


class CredentialCipher:
    """Encrypts user API keys with an installation-scoped authenticated key."""

    def __init__(self, secret: str):
        if len(secret) < 16:
            raise ValueError("凭据加密密钥至少需要 16 个字符")
        derived = hashlib.sha256(
            b"xushu:credential:v1\0" + secret.encode("utf-8")
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
    if not MODEL_RE.fullmatch(value):
        raise CredentialError(
            "模型名需为 1–100 位字母、数字、点、下划线、冒号或短横线"
        )
    return value


def key_hint(value: str) -> str:
    prefix = "sk-" if value.startswith("sk-") else ""
    return f"{prefix}••••{value[-4:]}"
