from __future__ import annotations

from typing import Any, Mapping
from urllib.parse import quote, urlencode, urlsplit

from fastapi import Request


def safe_next(value: str, fallback: str = "/") -> str:
    """Return a local redirect target or the supplied safe fallback."""
    clean_value = str(value or "").strip()
    if (
        not clean_value.startswith("/")
        or clean_value.startswith("//")
        or "\\" in clean_value
        or any(ord(character) < 32 for character in clean_value)
    ):
        return fallback
    parsed = urlsplit(clean_value)
    if parsed.scheme or parsed.netloc or not parsed.path.startswith("/"):
        return fallback
    return clean_value


def request_relative_url(request: Request) -> str:
    path = request.url.path
    if request.url.query:
        path = f"{path}?{request.url.query}"
    return safe_next(path, "/dashboard")


def wants_json(request: Request) -> bool:
    return "application/json" in request.headers.get("accept", "")


def api_settings_path(
    *,
    return_to: str = "/dashboard",
    **params: Any,
) -> str:
    query = [(key, str(value)) for key, value in params.items() if value is not None]
    safe_return_to = safe_next(return_to, "/dashboard")
    if safe_return_to != "/dashboard":
        query.append(("return_to", safe_return_to))
    return "/settings/api" + (f"?{urlencode(query)}" if query else "")


def workbench_path(
    project_id: str,
    *,
    settings_tab: str | None = None,
    chapter_id: str | None = None,
    archive_tab: str | None = None,
    **params: Any,
) -> str:
    query: list[tuple[str, str]] = []
    if settings_tab:
        query.extend(
            (
                ("view", "archive"),
                ("archive_tab", "creative"),
                ("settings_tab", settings_tab),
            )
        )
    elif archive_tab:
        query.extend(
            (
                ("view", "archive"),
                ("archive_tab", archive_tab),
            )
        )
    if chapter_id:
        query.append(("chapter_id", chapter_id))
    query.extend(
        (key, str(value)) for key, value in params.items() if value is not None
    )
    path = f"/novels/{quote(project_id, safe='')}/workbench"
    return f"{path}?{urlencode(query)}" if query else path


def document_workbench_path(
    document_id: str,
    *,
    chapter_id: str | None = None,
    conversation_id: str | None = None,
    view: str | None = None,
    **params: Any,
) -> str:
    query: list[tuple[str, str]] = []
    if chapter_id:
        query.append(("chapter_id", chapter_id))
    if conversation_id:
        query.append(("conversation_id", conversation_id))
    if view:
        query.append(("view", view))
    query.extend(
        (key, str(value)) for key, value in params.items() if value is not None
    )
    path = f"/documents/{quote(document_id, safe='')}"
    return f"{path}?{urlencode(query)}" if query else path


def append_query(path: str, **params: Any) -> str:
    query = [(key, str(value)) for key, value in params.items() if value is not None]
    if not query:
        return path
    separator = "&" if "?" in path else "?"
    return f"{path}{separator}{urlencode(query)}"


def work_archive_destination(
    work: Mapping[str, Any],
    *,
    saved: bool = False,
    error: str | None = None,
) -> str:
    current_version = work.get("current_version")
    main_version = work.get("main_version")
    if current_version and current_version.get("document_id"):
        destination = document_workbench_path(
            str(current_version["document_id"]),
            view="archive",
            archive_tab="analysis",
        )
    elif current_version and current_version.get("project_id"):
        destination = workbench_path(
            str(current_version["project_id"]),
            archive_tab="creative",
        )
    elif main_version:
        destination = workbench_path(
            str(main_version["project_id"]),
            archive_tab="creative",
        )
    elif work.get("source_version"):
        destination = document_workbench_path(
            str(work["source_version"]["document_id"]),
            view="archive",
            archive_tab="analysis",
        )
    else:
        destination = "/dashboard"
    return append_query(
        destination,
        saved="true" if saved else None,
        error=error,
    )
