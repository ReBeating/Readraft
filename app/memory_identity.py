from __future__ import annotations

import json
import sqlite3
import unicodedata
import uuid
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


IDENTITY_TYPES = (
    "character",
    "location",
    "item",
    "fact",
    "plot_thread",
    "foreshadowing",
    "event",
    "other",
)

IDENTITY_TYPE_LABELS = {
    "character": "人物",
    "location": "地点",
    "item": "物品",
    "fact": "事实",
    "plot_thread": "剧情线",
    "foreshadowing": "伏笔",
    "event": "事件",
    "other": "其他实体",
}


def normalize_identity_text(value: Any) -> str:
    """Conservative identity key: normalize form, case and punctuation only."""

    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return "".join(
        character
        for character in normalized
        if not unicodedata.category(character).startswith(("P", "Z", "C"))
    )


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _load_json(value: Any, fallback: Any) -> Any:
    try:
        return json.loads(str(value))
    except (TypeError, ValueError):
        return fallback


def _identity_row(
    connection: sqlite3.Connection,
    *,
    project_id: str,
    identity_type: str,
    text: str,
) -> Optional[sqlite3.Row]:
    alias_key = normalize_identity_text(text)
    if not alias_key:
        return None
    return connection.execute(
        """
        SELECT i.*
        FROM memory_identity_aliases a
        JOIN memory_identities i ON i.id=a.identity_id
        WHERE a.project_id=? AND a.identity_type=? AND a.alias_key=?
          AND i.status='active'
        LIMIT 1
        """,
        (project_id, identity_type, alias_key),
    ).fetchone()


def ensure_memory_identity(
    connection: sqlite3.Connection,
    *,
    project_id: str,
    identity_type: str,
    canonical_text: str,
    created_at: str,
    source: str = "story_delta",
    linked_record_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if identity_type not in IDENTITY_TYPES:
        raise ValueError("不支持的记忆身份类型")
    display = _clean(canonical_text)
    canonical_key = normalize_identity_text(display)
    if not canonical_key:
        return None
    existing = _identity_row(
        connection,
        project_id=project_id,
        identity_type=identity_type,
        text=display,
    )
    if existing:
        if linked_record_id and not _clean(existing["linked_record_id"]):
            connection.execute(
                """
                UPDATE memory_identities
                SET linked_record_id=?, updated_at=?
                WHERE id=?
                """,
                (linked_record_id, created_at, existing["id"]),
            )
        return dict(existing)

    same_canonical = connection.execute(
        """
        SELECT * FROM memory_identities
        WHERE project_id=? AND identity_type=? AND canonical_key=?
          AND status='active'
        LIMIT 1
        """,
        (project_id, identity_type, canonical_key),
    ).fetchone()
    identity_id = (
        str(same_canonical["id"]) if same_canonical else uuid.uuid4().hex
    )
    if not same_canonical:
        connection.execute(
            """
            INSERT INTO memory_identities(
                id, project_id, identity_type, canonical_text,
                canonical_key, linked_record_id, source, status,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
            """,
            (
                identity_id,
                project_id,
                identity_type,
                display,
                canonical_key,
                linked_record_id,
                source,
                created_at,
                created_at,
            ),
        )
    connection.execute(
        """
        INSERT OR IGNORE INTO memory_identity_aliases(
            id, identity_id, project_id, identity_type,
            alias_text, alias_key, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            uuid.uuid4().hex,
            identity_id,
            project_id,
            identity_type,
            display,
            canonical_key,
            created_at,
        ),
    )
    row = connection.execute(
        "SELECT * FROM memory_identities WHERE id=?",
        (identity_id,),
    ).fetchone()
    return dict(row) if row else None


_IDENTITY_REFERENCE_COLUMNS = (
    ("story_events", "event_identity_id"),
    ("story_facts", "subject_identity_id"),
    ("story_facts", "fact_identity_id"),
    ("character_knowledge", "character_identity_id"),
    ("character_knowledge", "fact_identity_id"),
    ("plot_threads", "thread_identity_id"),
    ("foreshadowing", "hook_identity_id"),
)


def _merge_identities(
    connection: sqlite3.Connection,
    *,
    target_id: str,
    source_id: str,
) -> None:
    if target_id == source_id:
        return
    for table, column in _IDENTITY_REFERENCE_COLUMNS:
        connection.execute(
            f"UPDATE {table} SET {column}=? WHERE {column}=?",
            (target_id, source_id),
        )
    connection.execute(
        """
        UPDATE memory_identity_aliases
        SET identity_id=?
        WHERE identity_id=?
        """,
        (target_id, source_id),
    )
    connection.execute(
        "DELETE FROM memory_identities WHERE id=?",
        (source_id,),
    )


def save_identity_rule(
    connection: sqlite3.Connection,
    *,
    project_id: str,
    identity_type: str,
    canonical_text: str,
    aliases: Iterable[str],
    created_at: str,
) -> Dict[str, Any]:
    if identity_type not in IDENTITY_TYPES:
        raise ValueError("不支持的归一类型")
    canonical = _clean(canonical_text)
    if not canonical:
        raise ValueError("标准称呼不能为空")
    unique_texts: List[str] = []
    seen_keys: set[str] = set()
    for text in (canonical, *aliases):
        cleaned = _clean(text)
        if len(cleaned) > 1500:
            raise ValueError("单个别名不能超过 1500 个字符")
        key = normalize_identity_text(cleaned)
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        unique_texts.append(cleaned)
    if len(unique_texts) < 2:
        raise ValueError("请至少填写一个不同于标准称呼的别名")

    existing_rows = [
        row
        for text in unique_texts
        if (
            row := _identity_row(
                connection,
                project_id=project_id,
                identity_type=identity_type,
                text=text,
            )
        )
    ]
    canonical_row = _identity_row(
        connection,
        project_id=project_id,
        identity_type=identity_type,
        text=canonical,
    )
    target_id = (
        str(canonical_row["id"])
        if canonical_row
        else str(existing_rows[0]["id"])
        if existing_rows
        else ""
    )
    if not target_id:
        created = ensure_memory_identity(
            connection,
            project_id=project_id,
            identity_type=identity_type,
            canonical_text=canonical,
            created_at=created_at,
            source="author",
        )
        if not created:
            raise ValueError("无法建立标准记忆身份")
        target_id = str(created["id"])

    for row in existing_rows:
        source_id = str(row["id"])
        if source_id != target_id:
            _merge_identities(
                connection,
                target_id=target_id,
                source_id=source_id,
            )

    old_target = connection.execute(
        "SELECT * FROM memory_identities WHERE id=?",
        (target_id,),
    ).fetchone()
    if not old_target:
        raise ValueError("标准记忆身份不存在")
    old_canonical = _clean(old_target["canonical_text"])
    old_key = normalize_identity_text(old_canonical)
    new_key = normalize_identity_text(canonical)
    connection.execute(
        """
        UPDATE memory_identities
        SET canonical_text=?, canonical_key=?, source='author', updated_at=?
        WHERE id=?
        """,
        (canonical, new_key, created_at, target_id),
    )
    for text in (old_canonical, *unique_texts):
        alias_key = normalize_identity_text(text)
        if not alias_key:
            continue
        connection.execute(
            """
            INSERT OR IGNORE INTO memory_identity_aliases(
                id, identity_id, project_id, identity_type,
                alias_text, alias_key, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uuid.uuid4().hex,
                target_id,
                project_id,
                identity_type,
                text,
                alias_key,
                created_at,
            ),
        )
        connection.execute(
            """
            UPDATE memory_identity_aliases
            SET identity_id=?, alias_text=?
            WHERE project_id=? AND identity_type=? AND alias_key=?
            """,
            (
                target_id,
                text,
                project_id,
                identity_type,
                alias_key,
            ),
        )
    if old_key and old_key != new_key:
        connection.execute(
            """
            UPDATE memory_identity_aliases
            SET identity_id=?
            WHERE project_id=? AND identity_type=? AND alias_key=?
            """,
            (target_id, project_id, identity_type, old_key),
        )
    result = connection.execute(
        "SELECT * FROM memory_identities WHERE id=?",
        (target_id,),
    ).fetchone()
    return dict(result)


def load_identity_index(
    connection: sqlite3.Connection, *, project_id: str
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    index: Dict[str, Dict[str, Dict[str, Any]]] = {
        identity_type: {} for identity_type in IDENTITY_TYPES
    }
    try:
        rows = connection.execute(
            """
            SELECT i.id, i.identity_type, i.canonical_text,
                   i.linked_record_id, a.alias_text, a.alias_key
            FROM memory_identities i
            JOIN memory_identity_aliases a ON a.identity_id=i.id
            WHERE i.project_id=? AND i.status='active'
            ORDER BY i.identity_type, i.canonical_text, a.alias_text
            """,
            (project_id,),
        ).fetchall()
    except sqlite3.OperationalError:
        return index
    for row in rows:
        index[str(row["identity_type"])][str(row["alias_key"])] = {
            "id": str(row["id"]),
            "identity_type": str(row["identity_type"]),
            "canonical_text": str(row["canonical_text"]),
            "linked_record_id": (
                str(row["linked_record_id"])
                if row["linked_record_id"]
                else None
            ),
            "matched_alias": str(row["alias_text"]),
        }
    return index


def resolve_identity(
    identity_index: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    identity_type: str,
    text: Any,
) -> Dict[str, Any]:
    display = _clean(text)
    key = normalize_identity_text(display)
    resolved = (identity_index.get(identity_type) or {}).get(key)
    if resolved:
        return dict(resolved)
    return {
        "id": None,
        "identity_type": identity_type,
        "canonical_text": display,
        "linked_record_id": None,
        "matched_alias": display,
    }


def list_identity_context(
    connection: sqlite3.Connection,
    *,
    project_id: str,
    limit: int = 160,
) -> List[Dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT i.id, i.identity_type, i.canonical_text,
               i.linked_record_id, i.source,
               COUNT(a.id) AS alias_count
        FROM memory_identities i
        JOIN memory_identity_aliases a ON a.identity_id=i.id
        WHERE i.project_id=? AND i.status='active'
        GROUP BY i.id
        ORDER BY CASE i.source WHEN 'author' THEN 0 ELSE 1 END,
                 alias_count DESC, i.updated_at DESC
        LIMIT ?
        """,
        (project_id, limit),
    ).fetchall()
    results: List[Dict[str, Any]] = []
    for row in rows:
        aliases = connection.execute(
            """
            SELECT alias_text FROM memory_identity_aliases
            WHERE identity_id=?
            ORDER BY alias_text
            """,
            (row["id"],),
        ).fetchall()
        item = dict(row)
        item["aliases"] = [str(alias["alias_text"]) for alias in aliases]
        results.append(item)
    return results


def expand_identity_terms(
    connection: sqlite3.Connection,
    *,
    project_id: str,
    terms: Sequence[str],
    max_terms: int = 96,
) -> List[str]:
    expanded: List[str] = []
    seen: set[str] = set()

    def append(value: Any) -> None:
        text = _clean(value)
        key = normalize_identity_text(text)
        if not text or not key or key in seen or len(expanded) >= max_terms:
            return
        seen.add(key)
        expanded.append(text)

    for term in terms:
        append(term)
    if not expanded:
        return expanded
    rows = connection.execute(
        """
        SELECT a.identity_id, a.alias_text, a.alias_key,
               i.canonical_text
        FROM memory_identity_aliases a
        JOIN memory_identities i ON i.id=a.identity_id
        WHERE a.project_id=? AND i.status='active'
        ORDER BY i.source='author' DESC, i.updated_at DESC
        LIMIT 1000
        """,
        (project_id,),
    ).fetchall()
    normalized_terms = [normalize_identity_text(term) for term in expanded]
    matched_ids = {
        str(row["identity_id"])
        for row in rows
        if any(
            row["alias_key"] == term_key
            or (
                len(str(row["alias_key"])) >= 2
                and str(row["alias_key"]) in term_key
            )
            for term_key in normalized_terms
        )
    }
    for row in rows:
        if str(row["identity_id"]) not in matched_ids:
            continue
        append(row["canonical_text"])
        append(row["alias_text"])
    return expanded


def backfill_memory_identities(
    connection: sqlite3.Connection,
    *,
    project_id: str,
    created_at: str,
) -> None:
    for character in connection.execute(
        "SELECT id, name FROM novel_characters WHERE project_id=?",
        (project_id,),
    ).fetchall():
        ensure_memory_identity(
            connection,
            project_id=project_id,
            identity_type="character",
            canonical_text=str(character["name"]),
            created_at=created_at,
            source="project",
            linked_record_id=str(character["id"]),
        )

    for row in connection.execute(
        """
        SELECT id, summary, event_key, participants_json, location
        FROM story_events
        WHERE project_id=? AND record_status='canon'
        """,
        (project_id,),
    ).fetchall():
        event_key = _clean(row["event_key"]) or _clean(row["summary"])
        identity = ensure_memory_identity(
            connection,
            project_id=project_id,
            identity_type="event",
            canonical_text=event_key,
            created_at=created_at,
        )
        connection.execute(
            """
            UPDATE story_events
            SET event_key=?, event_identity_id=?
            WHERE id=?
            """,
            (
                event_key,
                identity["id"] if identity else None,
                row["id"],
            ),
        )
        for participant in _load_json(row["participants_json"], []):
            ensure_memory_identity(
                connection,
                project_id=project_id,
                identity_type="character",
                canonical_text=str(participant),
                created_at=created_at,
            )
        if _clean(row["location"]):
            ensure_memory_identity(
                connection,
                project_id=project_id,
                identity_type="location",
                canonical_text=str(row["location"]),
                created_at=created_at,
            )

    for row in connection.execute(
        """
        SELECT id, subject_type, subject_id, subject_name, predicate,
               object_json
        FROM story_facts
        WHERE project_id=? AND fact_status='canon'
        """,
        (project_id,),
    ).fetchall():
        subject_type = str(row["subject_type"])
        identity_type = (
            "character"
            if subject_type == "character"
            or (subject_type == "entity" and row["subject_id"])
            else {
                "item": "item",
                "location": "location",
            }.get(subject_type, "other")
        )
        subject = ensure_memory_identity(
            connection,
            project_id=project_id,
            identity_type=identity_type,
            canonical_text=str(row["subject_name"]),
            created_at=created_at,
            linked_record_id=(
                str(row["subject_id"]) if row["subject_id"] else None
            ),
        )
        fact_text = (
            f"{row['subject_name']}｜{row['predicate']}｜{row['object_json']}"
        )
        fact = ensure_memory_identity(
            connection,
            project_id=project_id,
            identity_type="fact",
            canonical_text=fact_text,
            created_at=created_at,
        )
        connection.execute(
            """
            UPDATE story_facts
            SET subject_identity_id=?, fact_identity_id=?
            WHERE id=?
            """,
            (
                subject["id"] if subject else None,
                fact["id"] if fact else None,
                row["id"],
            ),
        )

    for row in connection.execute(
        """
        SELECT id, character_name, fact_text, fact_key
        FROM character_knowledge
        WHERE project_id=? AND record_status='canon'
        """,
        (project_id,),
    ).fetchall():
        character = ensure_memory_identity(
            connection,
            project_id=project_id,
            identity_type="character",
            canonical_text=str(row["character_name"]),
            created_at=created_at,
        )
        fact_key = _clean(row["fact_key"]) or _clean(row["fact_text"])
        fact = ensure_memory_identity(
            connection,
            project_id=project_id,
            identity_type="fact",
            canonical_text=fact_key,
            created_at=created_at,
        )
        connection.execute(
            """
            UPDATE character_knowledge
            SET character_identity_id=?, fact_identity_id=?, fact_key=?
            WHERE id=?
            """,
            (
                character["id"] if character else None,
                fact["id"] if fact else None,
                fact_key,
                row["id"],
            ),
        )

    for row in connection.execute(
        """
        SELECT id, thread_name FROM plot_threads
        WHERE project_id=? AND record_status='canon'
        """,
        (project_id,),
    ).fetchall():
        identity = ensure_memory_identity(
            connection,
            project_id=project_id,
            identity_type="plot_thread",
            canonical_text=str(row["thread_name"]),
            created_at=created_at,
        )
        connection.execute(
            "UPDATE plot_threads SET thread_identity_id=? WHERE id=?",
            (identity["id"] if identity else None, row["id"]),
        )

    for row in connection.execute(
        """
        SELECT id, hook_name FROM foreshadowing
        WHERE project_id=? AND record_status='canon'
        """,
        (project_id,),
    ).fetchall():
        identity = ensure_memory_identity(
            connection,
            project_id=project_id,
            identity_type="foreshadowing",
            canonical_text=str(row["hook_name"]),
            created_at=created_at,
        )
        connection.execute(
            "UPDATE foreshadowing SET hook_identity_id=? WHERE id=?",
            (identity["id"] if identity else None, row["id"]),
        )


class MemoryIdentityService:
    def __init__(self, database: Any):
        self.database = database

    def list_rules(
        self, *, user_id: int, project_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        with self.database.connection() as connection:
            owner = connection.execute(
                "SELECT 1 FROM novel_projects WHERE id=? AND user_id=?",
                (project_id, user_id),
            ).fetchone()
            if not owner:
                return None
            identities = list_identity_context(
                connection, project_id=project_id, limit=500
            )
        return [
            {
                **identity,
                "type_label": IDENTITY_TYPE_LABELS.get(
                    str(identity["identity_type"]),
                    str(identity["identity_type"]),
                ),
                "aliases": [
                    alias
                    for alias in identity["aliases"]
                    if normalize_identity_text(alias)
                    != normalize_identity_text(identity["canonical_text"])
                ],
            }
            for identity in identities
            if identity["source"] == "author"
            or int(identity["alias_count"]) > 1
        ]

    def save_rule(
        self,
        *,
        user_id: int,
        project_id: str,
        identity_type: str,
        canonical_text: str,
        aliases: Iterable[str],
        updated_at: str,
    ) -> Dict[str, Any]:
        from .continuity import replay_canonical_state

        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            project = connection.execute(
                """
                SELECT canonical_branch_id FROM novel_projects
                WHERE id=? AND user_id=?
                """,
                (project_id, user_id),
            ).fetchone()
            if not project:
                connection.rollback()
                raise ValueError("小说项目不存在")
            result = save_identity_rule(
                connection,
                project_id=project_id,
                identity_type=identity_type,
                canonical_text=canonical_text,
                aliases=aliases,
                created_at=updated_at,
            )
            replay_canonical_state(
                connection,
                project_id=project_id,
                branch_id=str(project["canonical_branch_id"] or "main"),
                trigger_type="identity_rule_updated",
                trigger_chapter_id=None,
                created_at=updated_at,
            )
            connection.execute(
                "UPDATE novel_projects SET updated_at=? WHERE id=?",
                (updated_at, project_id),
            )
            connection.commit()
        return result

    def remove_rule(
        self,
        *,
        user_id: int,
        project_id: str,
        identity_id: str,
        updated_at: str,
    ) -> bool:
        from .continuity import replay_canonical_state

        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT i.*, p.canonical_branch_id
                FROM memory_identities i
                JOIN novel_projects p ON p.id=i.project_id
                WHERE i.id=? AND i.project_id=? AND p.user_id=?
                """,
                (identity_id, project_id, user_id),
            ).fetchone()
            if not row:
                connection.rollback()
                return False
            connection.execute(
                """
                DELETE FROM memory_identity_aliases
                WHERE identity_id=? AND alias_key<>?
                """,
                (identity_id, row["canonical_key"]),
            )
            connection.execute(
                """
                UPDATE memory_identities
                SET source='story_delta', updated_at=?
                WHERE id=?
                """,
                (updated_at, identity_id),
            )
            replay_canonical_state(
                connection,
                project_id=project_id,
                branch_id=str(row["canonical_branch_id"] or "main"),
                trigger_type="identity_rule_removed",
                trigger_chapter_id=None,
                created_at=updated_at,
            )
            connection.execute(
                "UPDATE novel_projects SET updated_at=? WHERE id=?",
                (updated_at, project_id),
            )
            connection.commit()
        return True
