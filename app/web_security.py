from __future__ import annotations

import time
from collections import deque
from typing import Any
from urllib.parse import quote

from fastapi import Request, status
from fastapi.responses import RedirectResponse

from .db import Database


class SlidingWindowLimiter:
    """Small in-process limiter for authentication endpoints."""

    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = {}

    def allow(self, key: str, *, limit: int, window_seconds: int) -> bool:
        now = time.monotonic()
        events = self._events.setdefault(key, deque())
        cutoff = now - window_seconds
        while events and events[0] <= cutoff:
            events.popleft()
        if len(events) >= limit:
            return False
        events.append(now)
        if len(self._events) > 10_000:
            self._events = {
                item_key: item_events
                for item_key, item_events in self._events.items()
                if item_events and item_events[-1] > cutoff
            }
        return True


def client_address(request: Request) -> str:
    direct = request.client.host if request.client else "unknown"
    forwarded = request.headers.get("x-forwarded-for", "")
    if direct in {"127.0.0.1", "::1"} and forwarded:
        return forwarded.split(",", 1)[0].strip()[:128]
    return direct[:128]


def current_user(request: Request) -> dict[str, Any] | None:
    user_id = request.session.get("user_id")
    if not isinstance(user_id, int):
        return None
    database: Database = request.app.state.database
    user = database.get_user(user_id)
    if not user:
        request.session.clear()
    return user


def login_redirect(request: Request) -> RedirectResponse:
    next_path = request.url.path
    return RedirectResponse(
        f"/login?next={quote(next_path, safe='/')}",
        status_code=status.HTTP_303_SEE_OTHER,
    )
