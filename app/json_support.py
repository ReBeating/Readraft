from __future__ import annotations

import hashlib
import json
from typing import Any


def dump_json(value: Any) -> str:
    """Serialize application data compactly without escaping Chinese text."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def dump_canonical_json(value: Any) -> str:
    """Serialize data deterministically for signatures and snapshots."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def load_json(value: Any, fallback: Any) -> Any:
    """Decode stored JSON and return the caller's fallback when it is invalid."""

    try:
        return json.loads(str(value))
    except (TypeError, ValueError):
        return fallback


def load_json_dict(value: Any) -> dict[str, Any]:
    """Decode an object-shaped JSON column, rejecting other JSON types."""

    result = load_json(value, {})
    return result if isinstance(result, dict) else {}


def load_json_list(value: Any) -> list[Any]:
    """Decode an array-shaped JSON column, rejecting other JSON types."""

    result = load_json(value, [])
    return result if isinstance(result, list) else []


def json_fingerprint(value: Any) -> str:
    """Return a stable SHA-256 fingerprint for JSON-compatible data."""

    payload = dump_canonical_json(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
