from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
from typing import Any

from fastapi import HTTPException, Request, status


USERNAME_RE = re.compile(r"^[A-Za-z0-9_\-\u4e00-\u9fff]{2,32}$")
PASSWORD_MIN_LENGTH = 8


def validate_username(username: str) -> str:
    username = username.strip()
    if not USERNAME_RE.fullmatch(username):
        raise ValueError("用户名需为 2–32 位中文、字母、数字、下划线或短横线")
    return username


def validate_password(password: str) -> str:
    if len(password) < PASSWORD_MIN_LENGTH:
        raise ValueError(f"密码至少需要 {PASSWORD_MIN_LENGTH} 位")
    if len(password) > 256:
        raise ValueError("密码过长")
    return password


def hash_password(password: str) -> str:
    validate_password(password)
    salt = os.urandom(16)
    iterations = 600_000
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, iterations, dklen=32
    )
    return "$".join(
        [
            "pbkdf2_sha256",
            str(iterations),
            base64.urlsafe_b64encode(salt).decode("ascii"),
            base64.urlsafe_b64encode(digest).decode("ascii"),
        ]
    )


def verify_password(password: str, encoded: str) -> bool:
    try:
        parts = encoded.split("$")
        algorithm = parts[0]
        if algorithm == "pbkdf2_sha256" and len(parts) == 4:
            _, iterations, salt_b64, digest_b64 = parts
            salt = base64.urlsafe_b64decode(salt_b64.encode("ascii"))
            expected = base64.urlsafe_b64decode(digest_b64.encode("ascii"))
            actual = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                salt,
                int(iterations),
                dklen=len(expected),
            )
        else:
            return False
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError, IndexError):
        return False


def csrf_token(request: Request) -> str:
    token = request.session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        request.session["csrf_token"] = token
    return str(token)


def verify_csrf(request: Request, submitted: str | None) -> None:
    expected = request.session.get("csrf_token")
    if not expected or not submitted or not hmac.compare_digest(str(expected), submitted):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="表单已过期，请刷新后重试"
        )


def signed_in_user(request: Request) -> dict[str, Any] | None:
    user = getattr(request.state, "user", None)
    return dict(user) if user else None


def stable_provider_user_id(user_id: int, secret_key: str) -> str:
    digest = hmac.new(
        secret_key.encode("utf-8"),
        f"xushu:user:{user_id}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"u_{digest[:32]}"
