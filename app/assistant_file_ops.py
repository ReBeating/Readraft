from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_text(path: Any) -> str:
    try:
        return Path(str(path)).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ""


def atomic_write(path: Path, content: str, token: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.parent / f".{path.name}.{token}.tmp"
    temporary.write_text(content, encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)
    path.chmod(0o600)


def clean_title(value: str, fallback: str) -> str:
    title = re.sub(r"\s+", " ", value).strip()
    return (title or fallback)[:120]
