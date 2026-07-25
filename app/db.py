from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional

from .chapter_splitter import ChapterChunk
from .continuity import get_continuity_context, replay_canonical_state
from .memory_identity import (
    ensure_memory_identity,
    expand_identity_terms,
    list_identity_context,
)
from .memory_search import (
    SEARCH_ENGINE,
    SEARCH_SCOPE,
    build_query_concepts,
    build_query_terms,
    delete_chapter_search_documents,
    search_memory_documents,
)
from .migrations import apply_migrations


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def utc_after(seconds: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat(
        timespec="seconds"
    )


def _load_json(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    try:
        return json.loads(str(value))
    except (TypeError, ValueError):
        return fallback


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS api_credentials (
    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    encrypted_key TEXT NOT NULL,
    key_hint TEXT NOT NULL,
    model TEXT NOT NULL,
    thinking INTEGER NOT NULL DEFAULT 0,
    reasoning_effort TEXT NOT NULL DEFAULT 'high',
    system_prompt TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS novel_projects (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    genre TEXT NOT NULL,
    premise TEXT NOT NULL,
    world_setting TEXT NOT NULL DEFAULT '',
    style_guide TEXT NOT NULL DEFAULT '',
    ai_instructions TEXT NOT NULL DEFAULT '',
    point_of_view TEXT NOT NULL DEFAULT '第三人称限知',
    target_chapter_chars INTEGER NOT NULL DEFAULT 3000,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS novel_characters (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES novel_projects(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    name TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT '',
    traits TEXT NOT NULL DEFAULT '',
    background TEXT NOT NULL DEFAULT '',
    character_arc TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(project_id, name)
);

CREATE TABLE IF NOT EXISTS novel_chapters (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES novel_projects(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    title TEXT NOT NULL,
    outline TEXT NOT NULL DEFAULT '',
    key_points TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'planned',
    content_path TEXT NOT NULL,
    char_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(project_id, position)
);

CREATE TABLE IF NOT EXISTS generation_jobs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES novel_projects(id) ON DELETE CASCADE,
    chapter_id TEXT NOT NULL REFERENCES novel_chapters(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    operation TEXT NOT NULL,
    instruction TEXT NOT NULL DEFAULT '',
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    credential_source TEXT NOT NULL DEFAULT 'default',
    status TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    result_char_count INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    claim_token TEXT,
    lease_expires_at TEXT
);

CREATE TABLE IF NOT EXISTS novel_chapter_versions (
    id TEXT PRIMARY KEY,
    chapter_id TEXT NOT NULL REFERENCES novel_chapters(id) ON DELETE CASCADE,
    job_id TEXT REFERENCES generation_jobs(id) ON DELETE SET NULL,
    kind TEXT NOT NULL,
    content_path TEXT NOT NULL,
    char_count INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    source_path TEXT NOT NULL,
    source_encoding TEXT NOT NULL,
    char_count INTEGER NOT NULL,
    split_strategy TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ready',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chapters (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    title TEXT NOT NULL,
    kind TEXT NOT NULL,
    content_path TEXT NOT NULL,
    char_count INTEGER NOT NULL,
    source_start INTEGER NOT NULL,
    source_end INTEGER NOT NULL,
    part_number INTEGER NOT NULL DEFAULT 1,
    part_count INTEGER NOT NULL DEFAULT 1,
    UNIQUE(document_id, position)
);

CREATE TABLE IF NOT EXISTS analysis_jobs (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    credential_source TEXT NOT NULL DEFAULT 'default',
    status TEXT NOT NULL,
    total_chapters INTEGER NOT NULL,
    completed_chapters INTEGER NOT NULL DEFAULT 0,
    failed_chapters INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS chapter_analyses (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES analysis_jobs(id) ON DELETE CASCADE,
    chapter_id TEXT NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    result_json TEXT,
    raw_response TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    started_at TEXT,
    finished_at TEXT,
    claim_token TEXT,
    lease_expires_at TEXT,
    UNIQUE(job_id, chapter_id)
);

CREATE INDEX IF NOT EXISTS idx_documents_user_created
    ON documents(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_novel_projects_user_updated
    ON novel_projects(user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_novel_characters_project_position
    ON novel_characters(project_id, position);
CREATE INDEX IF NOT EXISTS idx_novel_chapters_project_position
    ON novel_chapters(project_id, position);
CREATE INDEX IF NOT EXISTS idx_generation_queue
    ON generation_jobs(status, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_generation_one_active_chapter
    ON generation_jobs(chapter_id)
    WHERE status IN ('queued', 'running');
CREATE INDEX IF NOT EXISTS idx_chapters_document_position
    ON chapters(document_id, position);
CREATE INDEX IF NOT EXISTS idx_jobs_user_created
    ON analysis_jobs(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analysis_queue
    ON chapter_analyses(status, job_id);
"""


class Database:
    def __init__(self, path: Path):
        self.path = path

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connection() as connection:
            connection.executescript(SCHEMA)
            columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(chapter_analyses)"
                ).fetchall()
            }
            if "claim_token" not in columns:
                connection.execute(
                    "ALTER TABLE chapter_analyses ADD COLUMN claim_token TEXT"
                )
            if "lease_expires_at" not in columns:
                connection.execute(
                    "ALTER TABLE chapter_analyses ADD COLUMN lease_expires_at TEXT"
                )
            job_columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(analysis_jobs)"
                ).fetchall()
            }
            if "credential_source" not in job_columns:
                connection.execute(
                    "ALTER TABLE analysis_jobs ADD COLUMN credential_source "
                    "TEXT NOT NULL DEFAULT 'default'"
                )
            connection.commit()
            apply_migrations(connection, utc_now())
            connection.execute(
                "UPDATE chapter_analyses SET status='queued', started_at=NULL, "
                "claim_token=NULL, lease_expires_at=NULL "
                "WHERE status='running'"
            )
            connection.execute(
                "UPDATE analysis_jobs SET status='queued', started_at=NULL "
                "WHERE status='running'"
            )
            connection.execute(
                """
                UPDATE generation_jobs
                SET status='queued', started_at=NULL, claim_token=NULL,
                    lease_expires_at=NULL
                WHERE status='running'
                """
            )
            connection.execute(
                """
                UPDATE story_plan_suggestions
                SET status='queued', started_at=NULL, claim_token=NULL,
                    lease_expires_at=NULL,
                    error='服务重启后已自动重新排队'
                WHERE status='running'
                """
            )
            connection.execute(
                """
                UPDATE story_structure_suggestions
                SET status='queued', started_at=NULL, claim_token=NULL,
                    lease_expires_at=NULL,
                    error='服务重启后已自动重新排队'
                WHERE status='running'
                """
            )
            connection.execute(
                """
                UPDATE novel_causal_link_suggestions
                SET status='queued', started_at=NULL, claim_token=NULL,
                    lease_expires_at=NULL,
                    error='服务重启后已自动重新排队'
                WHERE status='running'
                """
            )
            connection.execute(
                """
                UPDATE novel_causal_branch_simulations
                SET status='queued', started_at=NULL, claim_token=NULL,
                    lease_expires_at=NULL,
                    error='服务重启后已自动重新排队'
                WHERE status='running'
                """
            )
            connection.execute(
                """
                UPDATE assistant_messages
                SET status='queued', started_at=NULL, claim_token=NULL,
                    lease_expires_at=NULL,
                    error='服务重启后已自动重新排队'
                WHERE role='assistant' AND status='running'
                """
            )
            connection.commit()

    def ping(self) -> bool:
        with self.connection() as connection:
            row = connection.execute("SELECT 1 AS ok").fetchone()
        return bool(row and row["ok"] == 1)

    def user_count(self) -> int:
        with self.connection() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM users").fetchone()
        return int(row["count"])

    def create_user(self, username: str, password_hash: str) -> int:
        with self.connection() as connection:
            cursor = connection.execute(
                "INSERT INTO users(username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, password_hash, utc_now()),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE username = ? COLLATE NOCASE", (username,)
            ).fetchone()
        return dict(row) if row else None

    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT id, username, created_at FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_user_usage(self, user_id: int) -> Dict[str, int]:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS document_count,
                       COALESCE(SUM(char_count), 0) AS stored_chars
                FROM documents WHERE user_id=?
                """,
                (user_id,),
            ).fetchone()
        return {
            "document_count": int(row["document_count"] or 0),
            "stored_chars": int(row["stored_chars"] or 0),
        }

    def get_api_credential(self, user_id: int) -> Optional[Dict[str, Any]]:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT user_id, provider, encrypted_key, key_hint, model,
                       thinking, reasoning_effort, system_prompt,
                       created_at, updated_at
                FROM api_credentials WHERE user_id=?
                """,
                (user_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_api_credential_summary(
        self, user_id: int
    ) -> Optional[Dict[str, Any]]:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT user_id, provider, key_hint, model, thinking,
                       reasoning_effort, system_prompt, created_at, updated_at
                FROM api_credentials WHERE user_id=?
                """,
                (user_id,),
            ).fetchone()
        return dict(row) if row else None

    def has_api_credential(self, user_id: int) -> bool:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM api_credentials WHERE user_id=? LIMIT 1",
                (user_id,),
            ).fetchone()
        return row is not None

    def upsert_api_credential(
        self,
        *,
        user_id: int,
        encrypted_key: str,
        key_hint: str,
        model: str,
        thinking: bool,
        reasoning_effort: str,
        system_prompt: str = "",
        provider: str = "deepseek",
    ) -> None:
        now = utc_now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                """
                SELECT 1
                WHERE EXISTS(
                    SELECT 1 FROM analysis_jobs
                    WHERE user_id=? AND status IN ('queued', 'running')
                ) OR EXISTS(
                    SELECT 1 FROM generation_jobs
                    WHERE user_id=? AND status IN ('queued', 'running')
                ) OR EXISTS(
                    SELECT 1 FROM story_plan_suggestions
                    WHERE user_id=? AND status IN ('queued', 'running')
                ) OR EXISTS(
                    SELECT 1 FROM story_structure_suggestions
                    WHERE user_id=? AND status IN ('queued', 'running')
                ) OR EXISTS(
                    SELECT 1 FROM novel_causal_link_suggestions
                    WHERE user_id=? AND status IN ('queued', 'running')
                ) OR EXISTS(
                    SELECT 1 FROM novel_causal_branch_simulations
                    WHERE user_id=? AND status IN ('queued', 'running')
                ) OR EXISTS(
                    SELECT 1
                    FROM assistant_messages m
                    JOIN assistant_conversations c
                      ON c.id=m.conversation_id
                    WHERE c.user_id=?
                      AND m.role='assistant'
                      AND m.status IN ('queued', 'running')
                )
                """,
                (
                    user_id,
                    user_id,
                    user_id,
                    user_id,
                    user_id,
                    user_id,
                    user_id,
                ),
            ).fetchone()
            if active:
                connection.rollback()
                raise ValueError(
                    "有 AI 任务正在运行，请在任务结束后再修改 API 设置"
                )
            connection.execute(
                """
                INSERT INTO api_credentials(
                    user_id, provider, encrypted_key, key_hint, model,
                    thinking, reasoning_effort, system_prompt,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    provider=excluded.provider,
                    encrypted_key=excluded.encrypted_key,
                    key_hint=excluded.key_hint,
                    model=excluded.model,
                    thinking=excluded.thinking,
                    reasoning_effort=excluded.reasoning_effort,
                    system_prompt=excluded.system_prompt,
                    updated_at=excluded.updated_at
                """,
                (
                    user_id,
                    provider,
                    encrypted_key,
                    key_hint,
                    model,
                    int(thinking),
                    reasoning_effort,
                    system_prompt,
                    now,
                    now,
                ),
            )
            connection.commit()

    def delete_api_credential(self, user_id: int) -> bool:
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                """
                SELECT 1
                WHERE EXISTS(
                    SELECT 1 FROM analysis_jobs
                    WHERE user_id=? AND status IN ('queued', 'running')
                ) OR EXISTS(
                    SELECT 1 FROM generation_jobs
                    WHERE user_id=? AND status IN ('queued', 'running')
                ) OR EXISTS(
                    SELECT 1 FROM story_plan_suggestions
                    WHERE user_id=? AND status IN ('queued', 'running')
                ) OR EXISTS(
                    SELECT 1 FROM story_structure_suggestions
                    WHERE user_id=? AND status IN ('queued', 'running')
                ) OR EXISTS(
                    SELECT 1 FROM novel_causal_link_suggestions
                    WHERE user_id=? AND status IN ('queued', 'running')
                ) OR EXISTS(
                    SELECT 1 FROM novel_causal_branch_simulations
                    WHERE user_id=? AND status IN ('queued', 'running')
                ) OR EXISTS(
                    SELECT 1
                    FROM assistant_messages m
                    JOIN assistant_conversations c
                      ON c.id=m.conversation_id
                    WHERE c.user_id=?
                      AND m.role='assistant'
                      AND m.status IN ('queued', 'running')
                )
                """,
                (
                    user_id,
                    user_id,
                    user_id,
                    user_id,
                    user_id,
                    user_id,
                    user_id,
                ),
            ).fetchone()
            if active:
                connection.rollback()
                raise ValueError(
                    "有 AI 任务正在运行，请在任务结束后再删除 API Key"
                )
            cursor = connection.execute(
                "DELETE FROM api_credentials WHERE user_id=?", (user_id,)
            )
            connection.commit()
        return cursor.rowcount == 1

    def create_novel_project(
        self,
        *,
        user_id: int,
        project_id: str,
        title: str,
        genre: str,
        premise: str,
        world_setting: str,
        style_guide: str,
        point_of_view: str,
        target_chapter_chars: int,
        story_promise: str = "",
        target_audience: str = "",
        core_appeal: str = "",
        ending_constraint: str = "",
        planning_horizon: int = 20,
        ai_instructions: str = "",
    ) -> str:
        now = utc_now()
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO novel_projects(
                    id, user_id, title, genre, premise, world_setting,
                    style_guide, point_of_view, target_chapter_chars,
                    story_promise, target_audience, core_appeal,
                    ending_constraint, planning_horizon, ai_instructions,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    user_id,
                    title,
                    genre,
                    premise,
                    world_setting,
                    style_guide,
                    point_of_view,
                    target_chapter_chars,
                    story_promise,
                    target_audience,
                    core_appeal,
                    ending_constraint,
                    planning_horizon,
                    ai_instructions,
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO novel_voice_profiles(
                    id, project_id, narration_rules, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, 'draft', ?, ?)
                """,
                (uuid.uuid4().hex, project_id, style_guide, now, now),
            )
            connection.commit()
        return project_id

    def list_novel_projects(self, user_id: int) -> List[Dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT p.*,
                    (SELECT COUNT(*) FROM novel_characters c
                        WHERE c.project_id=p.id) AS character_count,
                    (SELECT COUNT(*) FROM novel_chapters ch
                        WHERE ch.project_id=p.id) AS chapter_count,
                    (SELECT COALESCE(SUM(ch.char_count), 0)
                        FROM novel_chapters ch
                        WHERE ch.project_id=p.id) AS total_chars
                FROM novel_projects p
                WHERE p.user_id=?
                ORDER BY p.updated_at DESC
                """,
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_novel_project(
        self, user_id: int, project_id: str
    ) -> Optional[Dict[str, Any]]:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT p.*,
                    (SELECT COUNT(*) FROM novel_characters c
                        WHERE c.project_id=p.id) AS character_count,
                    (SELECT COUNT(*) FROM novel_chapters ch
                        WHERE ch.project_id=p.id) AS chapter_count,
                    (SELECT COALESCE(SUM(ch.char_count), 0)
                        FROM novel_chapters ch
                        WHERE ch.project_id=p.id) AS total_chars
                FROM novel_projects p
                WHERE p.id=? AND p.user_id=?
                """,
                (project_id, user_id),
            ).fetchone()
        return dict(row) if row else None

    def delete_novel_project(self, user_id: int, project_id: str) -> bool:
        """Delete one owned novel after confirming no project task is active."""
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            project = connection.execute(
                """
                SELECT 1 FROM novel_projects
                WHERE id=? AND user_id=?
                """,
                (project_id, user_id),
            ).fetchone()
            if not project:
                connection.rollback()
                return False
            active = connection.execute(
                """
                SELECT 1
                WHERE EXISTS(
                    SELECT 1 FROM generation_jobs
                    WHERE project_id=? AND status IN ('queued', 'running')
                ) OR EXISTS(
                    SELECT 1 FROM story_plan_suggestions
                    WHERE project_id=? AND status IN ('queued', 'running')
                ) OR EXISTS(
                    SELECT 1 FROM story_structure_suggestions
                    WHERE project_id=? AND status IN ('queued', 'running')
                ) OR EXISTS(
                    SELECT 1 FROM novel_causal_link_suggestions
                    WHERE project_id=? AND status IN ('queued', 'running')
                ) OR EXISTS(
                    SELECT 1 FROM novel_causal_branch_simulations
                    WHERE project_id=? AND status IN ('queued', 'running')
                ) OR EXISTS(
                    SELECT 1
                    FROM assistant_messages m
                    JOIN assistant_conversations c
                      ON c.id=m.conversation_id
                    WHERE c.project_id=? AND c.user_id=?
                      AND m.role='assistant'
                      AND m.status IN ('queued', 'running')
                )
                """,
                (
                    project_id,
                    project_id,
                    project_id,
                    project_id,
                    project_id,
                    project_id,
                    user_id,
                ),
            ).fetchone()
            if active:
                connection.rollback()
                raise ValueError(
                    "作品有 AI 任务正在排队或运行，请完成后再删除"
                )
            cursor = connection.execute(
                """
                DELETE FROM novel_projects
                WHERE id=? AND user_id=?
                """,
                (project_id, user_id),
            )
            connection.commit()
        return cursor.rowcount == 1

    def update_novel_project(
        self,
        *,
        user_id: int,
        project_id: str,
        title: str,
        genre: str,
        premise: str,
        world_setting: str,
        style_guide: str,
        point_of_view: str,
        target_chapter_chars: int,
        story_promise: str = "",
        target_audience: str = "",
        core_appeal: str = "",
        ending_constraint: str = "",
        planning_horizon: int = 20,
        ai_instructions: str = "",
    ) -> bool:
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE novel_projects
                SET title=?, genre=?, premise=?, world_setting=?,
                    style_guide=?, point_of_view=?, target_chapter_chars=?,
                    story_promise=?, target_audience=?, core_appeal=?,
                    ending_constraint=?, planning_horizon=?,
                    ai_instructions=?,
                    updated_at=?
                WHERE id=? AND user_id=?
                """,
                (
                    title,
                    genre,
                    premise,
                    world_setting,
                    style_guide,
                    point_of_view,
                    target_chapter_chars,
                    story_promise,
                    target_audience,
                    core_appeal,
                    ending_constraint,
                    planning_horizon,
                    ai_instructions,
                    utc_now(),
                    project_id,
                    user_id,
                ),
            )
            connection.commit()
        return cursor.rowcount == 1

    def list_novel_characters(
        self, user_id: int, project_id: str
    ) -> List[Dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT c.* FROM novel_characters c
                JOIN novel_projects p ON p.id=c.project_id
                WHERE c.project_id=? AND p.user_id=?
                ORDER BY c.position, c.created_at
                """,
                (project_id, user_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def add_novel_character(
        self,
        *,
        user_id: int,
        project_id: str,
        name: str,
        role: str,
        traits: str,
        background: str,
        character_arc: str,
    ) -> str:
        character_id = uuid.uuid4().hex
        now = utc_now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            owner = connection.execute(
                "SELECT 1 FROM novel_projects WHERE id=? AND user_id=?",
                (project_id, user_id),
            ).fetchone()
            if not owner:
                connection.rollback()
                raise ValueError("小说项目不存在")
            position = connection.execute(
                """
                SELECT COALESCE(MAX(position), 0) + 1 AS next_position
                FROM novel_characters WHERE project_id=?
                """,
                (project_id,),
            ).fetchone()
            connection.execute(
                """
                INSERT INTO novel_characters(
                    id, project_id, position, name, role, traits, background,
                    character_arc, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    character_id,
                    project_id,
                    int(position["next_position"]),
                    name,
                    role,
                    traits,
                    background,
                    character_arc,
                    now,
                    now,
                ),
            )
            ensure_memory_identity(
                connection,
                project_id=project_id,
                identity_type="character",
                canonical_text=name,
                created_at=now,
                source="project",
                linked_record_id=character_id,
            )
            connection.execute(
                "UPDATE novel_projects SET updated_at=? WHERE id=?",
                (now, project_id),
            )
            connection.commit()
        return character_id

    def delete_novel_character(
        self, user_id: int, project_id: str, character_id: str
    ) -> bool:
        with self.connection() as connection:
            cursor = connection.execute(
                """
                DELETE FROM novel_characters
                WHERE id=? AND project_id=? AND EXISTS(
                    SELECT 1 FROM novel_projects p
                    WHERE p.id=novel_characters.project_id AND p.user_id=?
                )
                """,
                (character_id, project_id, user_id),
            )
            if cursor.rowcount:
                connection.execute(
                    """
                    UPDATE memory_identities
                    SET linked_record_id=NULL, updated_at=?
                    WHERE project_id=? AND linked_record_id=?
                    """,
                    (utc_now(), project_id, character_id),
                )
                connection.execute(
                    "UPDATE novel_projects SET updated_at=? WHERE id=?",
                    (utc_now(), project_id),
                )
            connection.commit()
        return cursor.rowcount == 1

    def list_novel_chapters(
        self, user_id: int, project_id: str
    ) -> List[Dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT ch.*,
                    v.title AS volume_title,
                    cp.status AS plan_status,
                    (SELECT COUNT(*) FROM novel_scene_beats sb
                        WHERE sb.plan_id=cp.id
                            AND sb.beat_status='active') AS scene_count,
                    (SELECT j.id FROM generation_jobs j
                        WHERE j.chapter_id=ch.id
                        ORDER BY j.created_at DESC LIMIT 1) AS latest_job_id,
                    (SELECT j.status FROM generation_jobs j
                        WHERE j.chapter_id=ch.id
                        ORDER BY j.created_at DESC LIMIT 1) AS latest_job_status
                FROM novel_chapters ch
                JOIN novel_projects p ON p.id=ch.project_id
                LEFT JOIN novel_volumes v ON v.id=ch.volume_id
                LEFT JOIN novel_chapter_plans cp ON cp.chapter_id=ch.id
                WHERE ch.project_id=? AND p.user_id=?
                ORDER BY ch.position
                """,
                (project_id, user_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def add_novel_chapter(
        self,
        *,
        user_id: int,
        project_id: str,
        chapter_id: str,
        title: str,
        outline: str,
        key_points: str,
        content_path: Path,
        volume_id: Optional[str] = None,
    ) -> str:
        now = utc_now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            owner = connection.execute(
                "SELECT 1 FROM novel_projects WHERE id=? AND user_id=?",
                (project_id, user_id),
            ).fetchone()
            if not owner:
                connection.rollback()
                raise ValueError("小说项目不存在")
            if volume_id:
                volume = connection.execute(
                    """
                    SELECT 1 FROM novel_volumes
                    WHERE id=? AND project_id=?
                    """,
                    (volume_id, project_id),
                ).fetchone()
                if not volume:
                    connection.rollback()
                    raise ValueError("所选分卷不存在")
            position = connection.execute(
                """
                SELECT COALESCE(MAX(position), 0) + 1 AS next_position
                FROM novel_chapters WHERE project_id=?
                """,
                (project_id,),
            ).fetchone()
            connection.execute(
                """
                INSERT INTO novel_chapters(
                    id, project_id, position, title, outline, key_points,
                    content_path, volume_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chapter_id,
                    project_id,
                    int(position["next_position"]),
                    title,
                    outline,
                    key_points,
                    str(content_path),
                    volume_id,
                    now,
                    now,
                ),
            )
            connection.execute(
                "UPDATE novel_projects SET updated_at=? WHERE id=?",
                (now, project_id),
            )
            connection.commit()
        return chapter_id

    def get_novel_chapter(
        self, user_id: int, project_id: str, chapter_id: str
    ) -> Optional[Dict[str, Any]]:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT ch.*, p.title AS project_title, p.genre, p.premise,
                       p.world_setting, p.style_guide, p.ai_instructions,
                       p.point_of_view,
                       p.target_chapter_chars,
                       v.title AS volume_title,
                       cp.status AS plan_status,
                       (SELECT COUNT(*) FROM novel_scene_beats sb
                        WHERE sb.plan_id=cp.id
                            AND sb.beat_status='active') AS scene_count,
                       (SELECT j.id FROM generation_jobs j
                            WHERE j.chapter_id=ch.id
                            ORDER BY j.created_at DESC LIMIT 1) AS latest_job_id,
                       (SELECT j.status FROM generation_jobs j
                            WHERE j.chapter_id=ch.id
                            ORDER BY j.created_at DESC LIMIT 1) AS latest_job_status
                FROM novel_chapters ch
                JOIN novel_projects p ON p.id=ch.project_id
                LEFT JOIN novel_volumes v ON v.id=ch.volume_id
                LEFT JOIN novel_chapter_plans cp ON cp.chapter_id=ch.id
                WHERE ch.id=? AND ch.project_id=? AND p.user_id=?
                """,
                (chapter_id, project_id, user_id),
            ).fetchone()
        return dict(row) if row else None

    def update_novel_chapter_plan(
        self,
        *,
        user_id: int,
        project_id: str,
        chapter_id: str,
        title: str,
        outline: str,
        key_points: str,
    ) -> bool:
        now = utc_now()
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE novel_chapters
                SET title=?, outline=?, key_points=?, updated_at=?
                WHERE id=? AND project_id=? AND EXISTS(
                    SELECT 1 FROM novel_projects p
                    WHERE p.id=novel_chapters.project_id AND p.user_id=?
                )
                """,
                (
                    title,
                    outline,
                    key_points,
                    now,
                    chapter_id,
                    project_id,
                    user_id,
                ),
            )
            if cursor.rowcount:
                connection.execute(
                    "UPDATE novel_projects SET updated_at=? WHERE id=?",
                    (now, project_id),
                )
            connection.commit()
        return cursor.rowcount == 1

    def record_manual_chapter_version(
        self,
        *,
        user_id: int,
        project_id: str,
        chapter_id: str,
        version_path: Path,
        char_count: int,
        effective_char_count: Optional[int] = None,
        content_hash: str = "",
        change_summary: str = "",
    ) -> Optional[str]:
        now = utc_now()
        version_id = uuid.uuid4().hex
        effective_count = (
            int(effective_char_count)
            if effective_char_count is not None
            else int(char_count)
        )
        quality_status = "block" if effective_count < 2000 else "pending"
        hard_issue_count = 1 if effective_count < 2000 else 0
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            chapter = connection.execute(
                """
                SELECT ch.id, ch.working_version_id FROM novel_chapters ch
                JOIN novel_projects p ON p.id=ch.project_id
                WHERE ch.id=? AND ch.project_id=? AND p.user_id=?
                """,
                (chapter_id, project_id, user_id),
            ).fetchone()
            if not chapter:
                connection.rollback()
                return None
            active = connection.execute(
                """
                SELECT 1 FROM generation_jobs
                WHERE chapter_id=? AND status IN ('queued', 'running')
                LIMIT 1
                """,
                (chapter_id,),
            ).fetchone()
            if active:
                connection.rollback()
                raise ValueError("AI 正在生成本章，请等待任务完成后再保存")
            connection.execute(
                """
                INSERT INTO novel_chapter_versions(
                    id, chapter_id, kind, content_path, char_count, created_at,
                    parent_version_id, status, source, content_hash,
                    change_summary, created_by, quality_status,
                    effective_char_count, hard_issue_count
                ) VALUES (?, ?, 'manual', ?, ?, ?, ?, 'candidate', 'manual',
                          ?, ?, 'author', ?, ?, ?)
                """,
                (
                    version_id,
                    chapter_id,
                    str(version_path),
                    char_count,
                    now,
                    chapter["working_version_id"],
                    content_hash,
                    change_summary[:1000],
                    quality_status,
                    effective_count,
                    hard_issue_count,
                ),
            )
            connection.execute(
                """
                UPDATE novel_chapters
                SET char_count=?, status=?, working_version_id=?, updated_at=?
                WHERE id=?
                """,
                (
                    char_count,
                    "draft" if char_count else "planned",
                    version_id,
                    now,
                    chapter_id,
                ),
            )
            connection.execute(
                "UPDATE novel_projects SET updated_at=? WHERE id=?",
                (now, project_id),
            )
            connection.commit()
        return version_id

    def list_chapter_versions(
        self, user_id: int, project_id: str, chapter_id: str
    ) -> List[Dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT v.*,
                       (
                           SELECT qa.id FROM chapter_quality_audits qa
                           WHERE qa.version_id=v.id
                           ORDER BY qa.created_at DESC, qa.rowid DESC
                           LIMIT 1
                       ) AS latest_quality_audit_id
                FROM novel_chapter_versions v
                JOIN novel_chapters ch ON ch.id=v.chapter_id
                JOIN novel_projects p ON p.id=ch.project_id
                WHERE v.chapter_id=? AND ch.project_id=? AND p.user_id=?
                ORDER BY v.created_at DESC, v.rowid DESC
                LIMIT 20
                """,
                (chapter_id, project_id, user_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_chapter_version(
        self,
        user_id: int,
        project_id: str,
        chapter_id: str,
        version_id: str,
    ) -> Optional[Dict[str, Any]]:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT v.*, ch.canonical_version_id, ch.working_version_id,
                       ch.position, ch.title AS chapter_title,
                       p.title AS project_title,
                       (
                           SELECT qa.id FROM chapter_quality_audits qa
                           WHERE qa.version_id=v.id
                           ORDER BY qa.created_at DESC, qa.rowid DESC
                           LIMIT 1
                       ) AS latest_quality_audit_id
                FROM novel_chapter_versions v
                JOIN novel_chapters ch ON ch.id=v.chapter_id
                JOIN novel_projects p ON p.id=ch.project_id
                WHERE v.id=? AND v.chapter_id=? AND ch.project_id=?
                    AND p.user_id=?
                """,
                (version_id, chapter_id, project_id, user_id),
            ).fetchone()
        return dict(row) if row else None

    def get_latest_quality_audit(
        self,
        *,
        user_id: int,
        project_id: str,
        chapter_id: str,
        version_id: str,
    ) -> Optional[Dict[str, Any]]:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT qa.*
                FROM chapter_quality_audits qa
                JOIN novel_chapter_versions v ON v.id=qa.version_id
                JOIN novel_chapters ch ON ch.id=v.chapter_id
                JOIN novel_projects p ON p.id=ch.project_id
                WHERE qa.version_id=? AND ch.id=? AND p.id=? AND p.user_id=?
                ORDER BY qa.created_at DESC, qa.rowid DESC
                LIMIT 1
                """,
                (version_id, chapter_id, project_id, user_id),
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["report"] = _load_json(result.get("result_json"), {})
        return result

    def accept_chapter_version(
        self,
        *,
        user_id: int,
        project_id: str,
        chapter_id: str,
        version_id: str,
        override_reason: str = "",
        expected_old_canonical_version_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        now = utc_now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT v.id, v.content_path, v.char_count, v.quality_status,
                       ch.canonical_version_id, ch.position,
                       p.canonical_branch_id
                FROM novel_chapter_versions v
                JOIN novel_chapters ch ON ch.id=v.chapter_id
                JOIN novel_projects p ON p.id=ch.project_id
                WHERE v.id=? AND v.chapter_id=? AND ch.project_id=?
                    AND p.user_id=?
                """,
                (version_id, chapter_id, project_id, user_id),
            ).fetchone()
            if not row:
                connection.rollback()
                return None
            if (
                expected_old_canonical_version_id is not None
                and str(row["canonical_version_id"] or "")
                != expected_old_canonical_version_id
            ):
                connection.rollback()
                raise ValueError(
                    "正史版本已在别处发生变化，请重新生成影响报告"
                )
            active = connection.execute(
                """
                SELECT 1 FROM generation_jobs
                WHERE chapter_id=? AND status IN ('queued', 'running')
                LIMIT 1
                """,
                (chapter_id,),
            ).fetchone()
            if active:
                connection.rollback()
                raise ValueError(
                    "AI 正在处理本章，请等待任务完成后再切换正史版本"
                )
            quality_status = str(row["quality_status"] or "pending")
            clean_override = override_reason.strip()
            if (
                row["canonical_version_id"] != version_id
                and quality_status != "pass"
                and len(clean_override) < 8
            ):
                connection.rollback()
                raise ValueError(
                    "硬审计尚未通过；如仍要设为正史，请填写至少 8 个字符的"
                    "作者覆盖原因"
                )
            if (
                row["canonical_version_id"] != version_id
                and quality_status != "pass"
            ):
                connection.execute(
                    """
                    UPDATE novel_chapter_versions
                    SET quality_status='overridden', quality_override_reason=?,
                        quality_overridden_at=?
                    WHERE id=?
                    """,
                    (clean_override[:1000], now, version_id),
                )
            old_canonical = row["canonical_version_id"]
            changed = old_canonical != version_id
            downstream_count = 0
            if changed:
                connection.execute(
                    """
                    UPDATE novel_chapter_versions
                    SET status='archived'
                    WHERE chapter_id=? AND status='canonical' AND id<>?
                    """,
                    (chapter_id, version_id),
                )
                connection.execute(
                    """
                    UPDATE novel_chapter_versions
                    SET status='canonical'
                    WHERE id=?
                    """,
                    (version_id,),
                )
                cursor = connection.execute(
                    """
                    UPDATE novel_chapters
                    SET needs_recheck=1, updated_at=?
                    WHERE project_id=? AND position>?
                        AND canonical_version_id IS NOT NULL
                    """,
                    (now, project_id, row["position"]),
                )
                downstream_count = int(cursor.rowcount)
                if old_canonical:
                    delete_chapter_search_documents(
                        connection, chapter_id=chapter_id
                    )
                    for table in (
                        "chapter_memory",
                        "story_events",
                        "character_knowledge",
                        "plot_threads",
                        "foreshadowing",
                    ):
                        connection.execute(
                            f"""
                            UPDATE {table}
                            SET record_status='retracted'
                            WHERE chapter_id=? AND record_status='canon'
                            """,
                            (chapter_id,),
                        )
                    connection.execute(
                        """
                        UPDATE story_facts
                        SET fact_status='retracted'
                        WHERE chapter_id=? AND fact_status='canon'
                        """,
                        (chapter_id,),
                    )
                    connection.execute(
                        """
                        UPDATE story_deltas
                        SET status='superseded', updated_at=?
                        WHERE chapter_id=? AND status='projected'
                        """,
                        (now, chapter_id),
                    )
            connection.execute(
                """
                UPDATE novel_chapters
                SET canonical_version_id=?, working_version_id=?,
                    char_count=?, status='canonical', needs_recheck=0,
                    updated_at=?
                WHERE id=?
                """,
                (
                    version_id,
                    version_id,
                    int(row["char_count"]),
                    now,
                    chapter_id,
                ),
            )
            connection.execute(
                "UPDATE novel_projects SET updated_at=? WHERE id=?",
                (now, project_id),
            )
            if changed:
                replay_canonical_state(
                    connection,
                    project_id=project_id,
                    branch_id=str(row["canonical_branch_id"] or "main"),
                    trigger_type="canon_version_changed",
                    trigger_chapter_id=chapter_id,
                    created_at=now,
                )
            connection.commit()
        return {
            "version_id": version_id,
            "content_path": str(row["content_path"]),
            "char_count": int(row["char_count"]),
            "old_canonical_version_id": (
                str(old_canonical) if old_canonical else None
            ),
            "changed": changed,
            "downstream_count": downstream_count,
        }

    def chapter_has_active_generation(
        self, user_id: int, project_id: str, chapter_id: str
    ) -> bool:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM generation_jobs j
                JOIN novel_chapters ch ON ch.id=j.chapter_id
                JOIN novel_projects p ON p.id=ch.project_id
                WHERE j.chapter_id=? AND ch.project_id=? AND p.user_id=?
                    AND j.status IN ('queued', 'running')
                LIMIT 1
                """,
                (chapter_id, project_id, user_id),
            ).fetchone()
        return row is not None

    def get_writing_context(
        self,
        user_id: int,
        chapter_id: str,
        scene_beat_id: Optional[str] = None,
        retrieval_hint: str = "",
    ) -> Optional[Dict[str, Any]]:
        with self.connection() as connection:
            chapter = connection.execute(
                """
                SELECT ch.*, p.title AS project_title, p.genre, p.premise,
                       p.world_setting, p.style_guide, p.ai_instructions,
                       p.point_of_view,
                       p.target_chapter_chars, p.canonical_branch_id,
                       p.story_promise, p.target_audience, p.core_appeal,
                       p.ending_constraint, p.planning_horizon,
                       v.title AS volume_title, v.goal AS volume_goal,
                       v.start_state AS volume_start_state,
                       v.end_state AS volume_end_state,
                       v.major_conflict AS volume_major_conflict,
                       v.payoff AS volume_payoff
                FROM novel_chapters ch
                JOIN novel_projects p ON p.id=ch.project_id
                LEFT JOIN novel_volumes v ON v.id=ch.volume_id
                WHERE ch.id=? AND p.user_id=?
                """,
                (chapter_id, user_id),
            ).fetchone()
            if not chapter:
                return None
            planned_causal_link_rows = connection.execute(
                """
                SELECT link.id, link.project_id,
                       link.source_chapter_id, link.target_chapter_id,
                       link.relation_type, link.cause_text,
                       link.effect_text, link.author_note, link.status,
                       source.position AS source_position,
                       source.title AS source_title,
                       source.skeleton_arc_titles_json
                           AS source_arc_titles_json,
                       source.canonical_version_id
                           AS source_canonical_version_id,
                       target.position AS target_position,
                       target.title AS target_title,
                       target.skeleton_arc_titles_json
                           AS target_arc_titles_json,
                       target.canonical_version_id
                           AS target_canonical_version_id
                FROM novel_chapter_causal_links link
                JOIN novel_chapters source
                  ON source.id=link.source_chapter_id
                JOIN novel_chapters target
                  ON target.id=link.target_chapter_id
                WHERE link.project_id=?
                  AND link.status='active'
                  AND target.canonical_version_id IS NULL
                  AND (
                    link.source_chapter_id=?
                    OR link.target_chapter_id=?
                  )
                ORDER BY target.position, source.position, link.created_at
                LIMIT 40
                """,
                (chapter["project_id"], chapter_id, chapter_id),
            ).fetchall()
            voice_profile = connection.execute(
                """
                SELECT narration_rules, sentence_rhythm, dialogue_voice,
                       sensory_palette, metaphor_policy, allowed_omissions,
                       preferred_patterns_json, banned_expressions_json,
                       author_notes, status, confirmed_at, updated_at
                FROM novel_voice_profiles
                WHERE project_id=? AND status='confirmed'
                """,
                (chapter["project_id"],),
            ).fetchone()
            editing_preferences = connection.execute(
                """
                SELECT id, category, guidance, applicability, updated_at,
                       source_type, support_count
                FROM (
                  SELECT aggregate.id, aggregate.category,
                         aggregate.guidance, aggregate.applicability,
                         aggregate.updated_at,
                         'stable_aggregate' AS source_type,
                         (
                           SELECT COUNT(*)
                           FROM
                             author_editing_preference_aggregate_evidence e
                           WHERE e.aggregate_id=aggregate.id
                             AND e.role='support'
                         ) AS support_count,
                         0 AS source_rank
                  FROM author_editing_preference_aggregates aggregate
                  WHERE aggregate.project_id=?
                    AND aggregate.status='active'

                  UNION ALL

                  SELECT pref.id, pref.category, pref.guidance,
                         pref.applicability, pref.updated_at,
                         'single_observation' AS source_type,
                         1 AS support_count,
                         1 AS source_rank
                  FROM author_editing_preferences pref
                  WHERE pref.project_id=? AND pref.status='active'
                    AND NOT EXISTS(
                      SELECT 1
                      FROM
                        author_editing_preference_aggregate_evidence e
                      JOIN
                        author_editing_preference_aggregates aggregate
                        ON aggregate.id=e.aggregate_id
                      WHERE e.preference_id=pref.id
                        AND aggregate.status='active'
                    )
                )
                ORDER BY source_rank, updated_at DESC, id DESC
                LIMIT 20
                """,
                (chapter["project_id"], chapter["project_id"]),
            ).fetchall()
            story_blueprint = connection.execute(
                """
                SELECT v.*
                FROM novel_story_blueprint_heads h
                JOIN novel_story_blueprint_versions v
                    ON v.id=h.confirmed_version_id
                WHERE h.project_id=? AND v.project_id=h.project_id
                    AND v.version_status='confirmed'
                """,
                (chapter["project_id"],),
            ).fetchone()
            planned_plot_arcs = connection.execute(
                """
                SELECT a.id, a.position, v.arc_type, v.title,
                       v.dramatic_question, v.promise, v.start_state,
                       v.target_payoff, v.involved_characters_json,
                       v.planned_turns_json, v.lifecycle_status,
                       v.priority
                FROM novel_plot_arcs a
                JOIN novel_plot_arc_versions v
                    ON v.id=a.confirmed_version_id
                WHERE a.project_id=? AND v.project_id=a.project_id
                    AND v.version_status='confirmed'
                ORDER BY v.priority DESC, a.position
                """,
                (chapter["project_id"],),
            ).fetchall()
            characters = connection.execute(
                """
                SELECT name, role, traits, background, character_arc
                FROM novel_characters
                WHERE project_id=?
                ORDER BY position
                """,
                (chapter["project_id"],),
            ).fetchall()
            plan = connection.execute(
                """
                SELECT * FROM novel_chapter_plans
                WHERE chapter_id=? AND status='confirmed'
                """,
                (chapter_id,),
            ).fetchone()
            scene_beats = []
            if plan:
                scene_beats = connection.execute(
                    """
                    SELECT id, position, pov_character, goal, obstacle,
                           action, reveal, conceal, subtext, location,
                           key_items_json, end_state, transition,
                           requirement_refs_json
                    FROM novel_scene_beats
                    WHERE plan_id=? AND beat_status='active'
                    ORDER BY position
                    """,
                    (plan["id"],),
                ).fetchall()
            previous = connection.execute(
                """
                SELECT ch.id, ch.position, ch.title, ch.outline,
                       v.content_path, v.char_count
                FROM novel_chapters ch
                JOIN novel_chapter_versions v
                    ON v.id=ch.canonical_version_id
                WHERE ch.project_id=? AND ch.position<?
                ORDER BY ch.position DESC LIMIT 1
                """,
                (chapter["project_id"], chapter["position"]),
            ).fetchone()
            recent_memory = connection.execute(
                """
                SELECT m.id AS source_id, ch.id AS chapter_id,
                       ch.position, ch.title, m.summary,
                       m.key_events_json, m.unresolved_questions_json,
                       m.keywords_json
                FROM chapter_memory m
                JOIN novel_chapters ch ON ch.id=m.chapter_id
                WHERE m.project_id=? AND m.branch_id=?
                    AND m.record_status='canon' AND ch.position<?
                ORDER BY ch.position DESC
                LIMIT 5
                """,
                (
                    chapter["project_id"],
                    chapter["canonical_branch_id"],
                    chapter["position"],
                ),
            ).fetchall()
            facts = connection.execute(
                """
                SELECT f.id AS source_id,
                       ch.position AS source_chapter_position,
                       ch.title AS source_chapter_title,
                       f.fact_type, f.subject_type, f.subject_name,
                       f.predicate, f.object_json, f.evidence
                FROM story_facts f
                JOIN novel_chapters ch ON ch.id=f.chapter_id
                WHERE f.project_id=? AND f.branch_id=?
                    AND f.fact_status='canon' AND ch.position<?
                ORDER BY ch.position DESC, f.created_at DESC
                LIMIT 120
                """,
                (
                    chapter["project_id"],
                    chapter["canonical_branch_id"],
                    chapter["position"],
                ),
            ).fetchall()
            knowledge = connection.execute(
                """
                SELECT k.id AS source_id,
                       ch.position AS source_chapter_position,
                       k.character_name, k.fact_text, k.knowledge_state,
                       k.learned_via, k.evidence
                FROM character_knowledge k
                JOIN novel_chapters ch ON ch.id=k.chapter_id
                WHERE k.project_id=? AND k.branch_id=?
                    AND k.record_status='canon' AND ch.position<?
                ORDER BY ch.position DESC, k.created_at DESC
                LIMIT 120
                """,
                (
                    chapter["project_id"],
                    chapter["canonical_branch_id"],
                    chapter["position"],
                ),
            ).fetchall()
            plot_threads = connection.execute(
                """
                SELECT t.id AS source_id,
                       ch.position AS source_chapter_position,
                       t.thread_name, t.thread_type, t.action,
                       t.update_text, t.promise, t.target_payoff, t.evidence
                FROM plot_threads t
                JOIN novel_chapters ch ON ch.id=t.chapter_id
                WHERE t.project_id=? AND t.branch_id=?
                    AND t.record_status='canon' AND ch.position<?
                ORDER BY ch.position DESC, t.created_at DESC
                LIMIT 80
                """,
                (
                    chapter["project_id"],
                    chapter["canonical_branch_id"],
                    chapter["position"],
                ),
            ).fetchall()
            hooks = connection.execute(
                """
                SELECT f.id AS source_id,
                       ch.position AS source_chapter_position,
                       f.hook_name, f.action, f.description,
                       f.intended_payoff, f.evidence
                FROM foreshadowing f
                JOIN novel_chapters ch ON ch.id=f.chapter_id
                WHERE f.project_id=? AND f.branch_id=?
                    AND f.record_status='canon' AND ch.position<?
                ORDER BY ch.position DESC, f.created_at DESC
                LIMIT 80
                """,
                (
                    chapter["project_id"],
                    chapter["canonical_branch_id"],
                    chapter["position"],
                ),
            ).fetchall()
            retrieval_scenes = []
            for row in scene_beats:
                scene = dict(row)
                scene["key_items"] = _load_json(
                    scene.pop("key_items_json"), []
                )
                scene["requirement_refs"] = _load_json(
                    scene.pop("requirement_refs_json"), []
                )
                retrieval_scenes.append(scene)
            retrieval_task_card = None
            if plan:
                retrieval_task_card = dict(plan)
                for stored, public in (
                    ("plot_threads_json", "plot_threads"),
                    ("must_happen_json", "must_happen"),
                    ("must_preserve_json", "must_preserve"),
                    ("forbidden_json", "forbidden"),
                    ("foreshadow_setup_json", "foreshadow_setup"),
                    ("foreshadow_payoff_json", "foreshadow_payoff"),
                ):
                    retrieval_task_card[public] = _load_json(
                        retrieval_task_card.pop(stored), []
                    )
                retrieval_task_card["scenes"] = retrieval_scenes
            retrieval_query_terms = build_query_terms(
                chapter=dict(chapter),
                characters=[dict(row) for row in characters],
                task_card=retrieval_task_card,
                scenes=retrieval_scenes,
                focused_scene_id=scene_beat_id,
                retrieval_hint=retrieval_hint,
            )
            retrieval_query_concepts = build_query_concepts(
                chapter=dict(chapter),
                characters=[dict(row) for row in characters],
                task_card=retrieval_task_card,
                scenes=retrieval_scenes,
                focused_scene_id=scene_beat_id,
                retrieval_hint=retrieval_hint,
            )
            retrieval_query_terms = expand_identity_terms(
                connection,
                project_id=str(chapter["project_id"]),
                terms=retrieval_query_terms,
                max_terms=96,
            )
            retrieval_query_concepts = expand_identity_terms(
                connection,
                project_id=str(chapter["project_id"]),
                terms=retrieval_query_concepts,
                max_terms=48,
            )
            memory_identities = list_identity_context(
                connection,
                project_id=str(chapter["project_id"]),
                limit=160,
            )
            retrieved_memory = search_memory_documents(
                connection,
                project_id=str(chapter["project_id"]),
                branch_id=str(chapter["canonical_branch_id"]),
                before_chapter_position=int(chapter["position"]),
                query_terms=retrieval_query_terms,
                excluded_chapter_ids=[
                    str(row["chapter_id"]) for row in recent_memory
                ],
            )
            continuity_context = get_continuity_context(
                connection,
                project_id=str(chapter["project_id"]),
                branch_id=str(chapter["canonical_branch_id"]),
                before_chapter_position=int(chapter["position"]),
                query_concepts=retrieval_query_concepts,
            )
            technique_rows = connection.execute(
                """
                SELECT tc.id, tc.name, tc.dimension, tc.effect,
                       tc.execution_rule, tc.originality_boundary,
                       b.id AS binding_id, b.scope_type,
                       b.usage_modes_json, b.author_adaptation, b.priority,
                       v.position AS volume_position,
                       v.title AS volume_title,
                       scoped_ch.position AS chapter_position,
                       scoped_ch.title AS chapter_title,
                       sb.position AS scene_position,
                       sb.goal AS scene_goal,
                       scene_ch.position AS scene_chapter_position,
                       scene_ch.title AS scene_chapter_title
                FROM novel_technique_bindings b
                JOIN reference_technique_cards tc ON tc.id=b.technique_id
                LEFT JOIN novel_volumes v ON v.id=b.volume_id
                LEFT JOIN novel_chapters scoped_ch ON scoped_ch.id=b.chapter_id
                LEFT JOIN novel_scene_beats sb ON sb.id=b.scene_beat_id
                LEFT JOIN novel_chapter_plans scene_plan
                    ON scene_plan.id=sb.plan_id
                LEFT JOIN novel_chapters scene_ch
                    ON scene_ch.id=scene_plan.chapter_id
                WHERE b.project_id=? AND b.status='enabled'
                    AND tc.status='active'
                    AND (
                        b.scope_type='project'
                        OR (b.scope_type='volume' AND b.volume_id=?)
                        OR (b.scope_type='chapter' AND b.chapter_id=?)
                        OR (
                            b.scope_type='scene'
                            AND scene_plan.chapter_id=?
                            AND sb.beat_status='active'
                            AND (? IS NULL OR b.scene_beat_id=?)
                        )
                    )
                ORDER BY b.priority DESC, b.created_at
                """,
                (
                    chapter["project_id"],
                    chapter["volume_id"],
                    chapter_id,
                    chapter_id,
                    scene_beat_id,
                    scene_beat_id,
                ),
            ).fetchall()
        recent_items = []
        for row in recent_memory:
            item = dict(row)
            item["key_events"] = _load_json(
                item.pop("key_events_json"), []
            )
            item["unresolved_questions"] = _load_json(
                item.pop("unresolved_questions_json"), []
            )
            item["keywords"] = _load_json(item.pop("keywords_json"), [])
            recent_items.append(item)
        fact_items = []
        for row in facts:
            item = dict(row)
            item["object"] = _load_json(item.pop("object_json"), {})
            fact_items.append(item)
        technique_items = []
        for row in technique_rows:
            item = dict(row)
            item["usage_modes"] = _load_json(
                item.pop("usage_modes_json"), []
            )
            if item["scope_type"] == "project":
                item["scope_label"] = "全书"
            elif item["scope_type"] == "volume":
                item["scope_label"] = (
                    f"第 {item['volume_position']} 卷"
                    f"《{item['volume_title']}》"
                )
            elif item["scope_type"] == "chapter":
                item["scope_label"] = (
                    f"第 {item['chapter_position']} 章"
                    f"《{item['chapter_title']}》"
                )
            else:
                item["scope_label"] = (
                    f"第 {item['scene_chapter_position']} 章 / "
                    f"场景 {item['scene_position']}：{item['scene_goal']}"
                )
            technique_items.append(item)
        task_card = None
        if plan:
            task_card = dict(plan)
            for stored, public in (
                ("plot_threads_json", "plot_threads"),
                ("must_happen_json", "must_happen"),
                ("must_preserve_json", "must_preserve"),
                ("forbidden_json", "forbidden"),
                ("foreshadow_setup_json", "foreshadow_setup"),
                ("foreshadow_payoff_json", "foreshadow_payoff"),
            ):
                task_card[public] = _load_json(
                    task_card.pop(stored), []
                )
            task_card["scenes"] = []
            for row in scene_beats:
                scene = dict(row)
                scene["key_items"] = _load_json(
                    scene.pop("key_items_json"), []
                )
                scene["requirement_refs"] = _load_json(
                    scene.pop("requirement_refs_json"), []
                )
                task_card["scenes"].append(scene)
        confirmed_voice_profile = None
        if voice_profile:
            confirmed_voice_profile = dict(voice_profile)
            confirmed_voice_profile["preferred_patterns"] = _load_json(
                confirmed_voice_profile.pop("preferred_patterns_json"), []
            )
            confirmed_voice_profile["banned_expressions"] = _load_json(
                confirmed_voice_profile.pop("banned_expressions_json"), []
            )
        confirmed_story_blueprint = None
        if story_blueprint:
            confirmed_story_blueprint = dict(story_blueprint)
            confirmed_story_blueprint["major_turns"] = _load_json(
                confirmed_story_blueprint.pop("major_turns_json"), []
            )
            confirmed_story_blueprint["must_payoffs"] = _load_json(
                confirmed_story_blueprint.pop("must_payoffs_json"), []
            )
            confirmed_story_blueprint["forbidden_shortcuts"] = _load_json(
                confirmed_story_blueprint.pop(
                    "forbidden_shortcuts_json"
                ),
                [],
            )
        confirmed_plot_arcs = []
        for row in planned_plot_arcs:
            item = dict(row)
            item["involved_characters"] = _load_json(
                item.pop("involved_characters_json"), []
            )
            item["planned_turns"] = _load_json(
                item.pop("planned_turns_json"), []
            )
            confirmed_plot_arcs.append(item)
        planned_causal_links = []
        for row in planned_causal_link_rows:
            item = dict(row)
            source_arcs = _load_json(
                item.pop("source_arc_titles_json"), []
            )
            target_arcs = _load_json(
                item.pop("target_arc_titles_json"), []
            )
            if not isinstance(source_arcs, list):
                source_arcs = []
            if not isinstance(target_arcs, list):
                target_arcs = []
            source_arcs = [
                str(value)
                for value in source_arcs
                if str(value).strip()
            ]
            target_arcs = [
                str(value)
                for value in target_arcs
                if str(value).strip()
            ]
            item["source_arc_titles"] = source_arcs
            item["target_arc_titles"] = target_arcs
            item["shared_arc_titles"] = sorted(
                set(source_arcs) & set(target_arcs)
            )
            item["cross_line"] = bool(
                source_arcs
                and target_arcs
                and not item["shared_arc_titles"]
            )
            item["source_is_canonical"] = bool(
                item.pop("source_canonical_version_id", None)
            )
            item["target_is_canonical"] = bool(
                item.pop("target_canonical_version_id", None)
            )
            planned_causal_links.append(item)
        chapter_item = dict(chapter)
        chapter_item["skeleton_arc_titles"] = _load_json(
            chapter_item.pop("skeleton_arc_titles_json", "[]"), []
        )
        return {
            "chapter": chapter_item,
            "characters": [dict(row) for row in characters],
            "previous_chapter": dict(previous) if previous else None,
            "task_card": task_card,
            "voice_profile": confirmed_voice_profile,
            "story_blueprint": confirmed_story_blueprint,
            "planned_plot_arcs": confirmed_plot_arcs,
            "planned_causal_links": planned_causal_links,
            "confirmed_editing_preferences": [
                dict(row) for row in editing_preferences
            ],
            "canonical_memory": {
                "recent_chapters": recent_items,
                "story_facts": fact_items,
                "character_knowledge": [
                    dict(row) for row in knowledge
                ],
                "plot_threads": [dict(row) for row in plot_threads],
                "foreshadowing": [dict(row) for row in hooks],
                "retrieved_memory": retrieved_memory,
                "current_state": continuity_context["current_state"],
                "continuity_issues": continuity_context[
                    "continuity_issues"
                ],
                "continuity_replay": continuity_context[
                    "continuity_replay"
                ],
                "retrieval": {
                    "engine": SEARCH_ENGINE,
                    "scope": SEARCH_SCOPE,
                    "query_terms": retrieval_query_terms,
                    "query_concepts": retrieval_query_concepts,
                    "matched_count": len(retrieved_memory),
                    "excluded_recent_chapter_count": len(
                        recent_memory
                    ),
                },
            },
            "memory_identities": memory_identities,
            "technique_cards": technique_items,
        }

    def create_generation_job(
        self,
        *,
        user_id: int,
        project_id: str,
        chapter_id: str,
        operation: str,
        instruction: str,
        provider: str,
        model: str,
        credential_source: str,
        subject_id: Optional[str] = None,
        max_jobs_per_day: Optional[int] = None,
    ) -> str:
        if operation not in {
            "draft",
            "continue",
            "rewrite",
            "polish",
            "generate_scene",
            "rewrite_scene",
            "audit_scene",
        }:
            raise ValueError("不支持的写作操作")
        if credential_source not in {"default", "personal"}:
            raise ValueError("不支持的 API 凭据来源")
        job_id = uuid.uuid4().hex
        now = utc_now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            chapter = connection.execute(
                """
                SELECT ch.id FROM novel_chapters ch
                JOIN novel_projects p ON p.id=ch.project_id
                WHERE ch.id=? AND ch.project_id=? AND p.user_id=?
                """,
                (chapter_id, project_id, user_id),
            ).fetchone()
            if not chapter:
                connection.rollback()
                raise ValueError("章节不存在")
            if operation in {"generate_scene", "rewrite_scene"}:
                if not subject_id:
                    connection.rollback()
                    raise ValueError("没有指定场景节拍")
                scene = connection.execute(
                    """
                    SELECT sb.current_version_id, cp.status
                    FROM novel_scene_beats sb
                    JOIN novel_chapter_plans cp ON cp.id=sb.plan_id
                    WHERE sb.id=? AND cp.chapter_id=?
                        AND sb.beat_status='active'
                    """,
                    (subject_id, chapter_id),
                ).fetchone()
                if not scene:
                    connection.rollback()
                    raise ValueError("场景节拍不存在")
                if str(scene["status"]) != "confirmed":
                    connection.rollback()
                    raise ValueError("章节任务卡尚未确认")
                if (
                    operation == "generate_scene"
                    and scene["current_version_id"]
                ):
                    connection.rollback()
                    raise ValueError("这个场景已有草稿，请选择重写")
                if (
                    operation == "rewrite_scene"
                    and not scene["current_version_id"]
                ):
                    connection.rollback()
                    raise ValueError("这个场景还没有草稿，请先生成")
            elif operation == "audit_scene":
                if not subject_id:
                    connection.rollback()
                    raise ValueError("没有指定场景版本")
                scene_version = connection.execute(
                    """
                    SELECT sv.id, sb.current_version_id, cp.status
                    FROM novel_scene_versions sv
                    JOIN novel_scene_beats sb ON sb.id=sv.scene_beat_id
                    JOIN novel_chapter_plans cp ON cp.id=sb.plan_id
                    WHERE sv.id=? AND cp.chapter_id=?
                        AND sb.beat_status='active'
                    """,
                    (subject_id, chapter_id),
                ).fetchone()
                if not scene_version:
                    connection.rollback()
                    raise ValueError("场景版本不存在")
                if (
                    str(scene_version["current_version_id"] or "")
                    != subject_id
                ):
                    connection.rollback()
                    raise ValueError("只能检查当前场景版本")
                if str(scene_version["status"]) != "confirmed":
                    connection.rollback()
                    raise ValueError("章节任务卡尚未确认")
            elif subject_id:
                connection.rollback()
                raise ValueError("整章写作任务不能指定场景")
            if credential_source == "personal":
                credential = connection.execute(
                    "SELECT 1 FROM api_credentials WHERE user_id=?",
                    (user_id,),
                ).fetchone()
                if not credential:
                    connection.rollback()
                    raise ValueError("个人 DeepSeek API Key 不存在，请重新配置")
            active = connection.execute(
                """
                SELECT id, chapter_id, operation, subject_id
                FROM generation_jobs
                WHERE user_id=? AND status IN ('queued', 'running')
                ORDER BY created_at LIMIT 1
                """,
                (user_id,),
            ).fetchone()
            if active:
                connection.rollback()
                if (
                    str(active["chapter_id"]) == chapter_id
                    and str(active["operation"]) == operation
                    and str(active["subject_id"] or "")
                    == str(subject_id or "")
                ):
                    return str(active["id"])
                raise ValueError("你已有一个写作任务正在排队或运行，请等待其完成")
            if max_jobs_per_day is not None:
                day_start = datetime.now(timezone.utc).replace(
                    hour=0, minute=0, second=0, microsecond=0
                ).isoformat(timespec="seconds")
                generation_count = connection.execute(
                    """
                    SELECT COUNT(*) AS count FROM generation_jobs
                    WHERE user_id=? AND created_at>=?
                    """,
                    (user_id, day_start),
                ).fetchone()
                analysis_count = connection.execute(
                    """
                    SELECT COUNT(*) AS count FROM analysis_jobs
                    WHERE user_id=? AND created_at>=?
                    """,
                    (user_id, day_start),
                ).fetchone()
                if (
                    int(generation_count["count"] or 0)
                    + int(analysis_count["count"] or 0)
                    >= max_jobs_per_day
                ):
                    connection.rollback()
                    raise ValueError(
                        f"今天已达到 {max_jobs_per_day} 个 AI 任务的上限"
                    )
            connection.execute(
                """
                INSERT INTO generation_jobs(
                    id, project_id, chapter_id, user_id, operation,
                    instruction, provider, model, credential_source,
                    status, created_at, subject_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)
                """,
                (
                    job_id,
                    project_id,
                    chapter_id,
                    user_id,
                    operation,
                    instruction,
                    provider,
                    model,
                    credential_source,
                    now,
                    subject_id,
                ),
            )
            connection.commit()
        return job_id

    def create_memory_extraction_job(
        self,
        *,
        user_id: int,
        project_id: str,
        chapter_id: str,
        version_id: str,
        provider: str,
        model: str,
        credential_source: str,
        max_jobs_per_day: Optional[int] = None,
    ) -> str:
        if credential_source not in {"default", "personal"}:
            raise ValueError("不支持的 API 凭据来源")
        job_id = uuid.uuid4().hex
        now = utc_now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            target = connection.execute(
                """
                SELECT ch.title, ch.position, p.title AS project_title
                FROM novel_chapter_versions v
                JOIN novel_chapters ch ON ch.id=v.chapter_id
                JOIN novel_projects p ON p.id=ch.project_id
                WHERE v.id=? AND v.chapter_id=? AND ch.project_id=?
                    AND p.user_id=? AND ch.canonical_version_id=v.id
                """,
                (version_id, chapter_id, project_id, user_id),
            ).fetchone()
            if not target:
                connection.rollback()
                raise ValueError("只能从当前正史版本提取故事记忆")
            if credential_source == "personal":
                credential = connection.execute(
                    "SELECT 1 FROM api_credentials WHERE user_id=?",
                    (user_id,),
                ).fetchone()
                if not credential:
                    connection.rollback()
                    raise ValueError(
                        "个人 DeepSeek API Key 不存在，请重新配置"
                    )
            active = connection.execute(
                """
                SELECT id, chapter_id, operation, version_id
                FROM generation_jobs
                WHERE user_id=? AND status IN ('queued', 'running')
                ORDER BY created_at LIMIT 1
                """,
                (user_id,),
            ).fetchone()
            if active:
                connection.rollback()
                if (
                    str(active["chapter_id"]) == chapter_id
                    and str(active["operation"]) == "extract_story_delta"
                    and str(active["version_id"]) == version_id
                ):
                    return str(active["id"])
                raise ValueError(
                    "你已有一个写作任务正在排队或运行，请等待其完成"
                )
            existing_delta = connection.execute(
                """
                SELECT id FROM story_deltas
                WHERE version_id=? AND status IN(
                    'proposed', 'author_edited', 'accepted', 'projected'
                )
                LIMIT 1
                """,
                (version_id,),
            ).fetchone()
            if existing_delta:
                connection.rollback()
                raise ValueError("这个正史版本已经有故事记忆提案")
            if max_jobs_per_day is not None:
                day_start = datetime.now(timezone.utc).replace(
                    hour=0, minute=0, second=0, microsecond=0
                ).isoformat(timespec="seconds")
                generation_count = connection.execute(
                    """
                    SELECT COUNT(*) AS count FROM generation_jobs
                    WHERE user_id=? AND created_at>=?
                    """,
                    (user_id, day_start),
                ).fetchone()
                analysis_count = connection.execute(
                    """
                    SELECT COUNT(*) AS count FROM analysis_jobs
                    WHERE user_id=? AND created_at>=?
                    """,
                    (user_id, day_start),
                ).fetchone()
                if (
                    int(generation_count["count"] or 0)
                    + int(analysis_count["count"] or 0)
                    >= max_jobs_per_day
                ):
                    connection.rollback()
                    raise ValueError(
                        f"今天已达到 {max_jobs_per_day} 个 AI 任务的上限"
                    )
            snapshot = {
                "project_id": project_id,
                "chapter_id": chapter_id,
                "version_id": version_id,
                "chapter_title": str(target["title"]),
                "chapter_position": int(target["position"]),
            }
            connection.execute(
                """
                INSERT INTO generation_jobs(
                    id, project_id, chapter_id, user_id, operation,
                    instruction, provider, model, credential_source,
                    status, created_at, version_id, context_snapshot_json
                ) VALUES (?, ?, ?, ?, 'extract_story_delta', '', ?, ?, ?,
                          'queued', ?, ?, ?)
                """,
                (
                    job_id,
                    project_id,
                    chapter_id,
                    user_id,
                    provider,
                    model,
                    credential_source,
                    now,
                    version_id,
                    json.dumps(snapshot, ensure_ascii=False),
                ),
            )
            connection.commit()
        return job_id

    def create_chapter_planning_job(
        self,
        *,
        user_id: int,
        project_id: str,
        chapter_id: str,
        instruction: str,
        provider: str,
        model: str,
        credential_source: str,
        max_jobs_per_day: Optional[int] = None,
        operation: str = "plan_chapter",
    ) -> str:
        if operation not in {"plan_chapter", "plan_scene_beats"}:
            raise ValueError("不支持的章节规划操作")
        if credential_source not in {"default", "personal"}:
            raise ValueError("不支持的 API 凭据来源")
        job_id = uuid.uuid4().hex
        now = utc_now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            chapter = connection.execute(
                """
                SELECT ch.title, ch.position
                FROM novel_chapters ch
                JOIN novel_projects p ON p.id=ch.project_id
                WHERE ch.id=? AND ch.project_id=? AND p.user_id=?
                """,
                (chapter_id, project_id, user_id),
            ).fetchone()
            if not chapter:
                connection.rollback()
                raise ValueError("章节不存在")
            if credential_source == "personal":
                credential = connection.execute(
                    "SELECT 1 FROM api_credentials WHERE user_id=?",
                    (user_id,),
                ).fetchone()
                if not credential:
                    connection.rollback()
                    raise ValueError(
                        "个人 DeepSeek API Key 不存在，请重新配置"
                    )
            active = connection.execute(
                """
                SELECT id, chapter_id, operation
                FROM generation_jobs
                WHERE user_id=? AND status IN ('queued', 'running')
                ORDER BY created_at LIMIT 1
                """,
                (user_id,),
            ).fetchone()
            if active:
                connection.rollback()
                if (
                    str(active["chapter_id"]) == chapter_id
                    and str(active["operation"]) == operation
                ):
                    return str(active["id"])
                raise ValueError(
                    "你已有一个写作任务正在排队或运行，请等待其完成"
                )
            if max_jobs_per_day is not None:
                day_start = datetime.now(timezone.utc).replace(
                    hour=0, minute=0, second=0, microsecond=0
                ).isoformat(timespec="seconds")
                generation_count = connection.execute(
                    """
                    SELECT COUNT(*) AS count FROM generation_jobs
                    WHERE user_id=? AND created_at>=?
                    """,
                    (user_id, day_start),
                ).fetchone()
                analysis_count = connection.execute(
                    """
                    SELECT COUNT(*) AS count FROM analysis_jobs
                    WHERE user_id=? AND created_at>=?
                    """,
                    (user_id, day_start),
                ).fetchone()
                if (
                    int(generation_count["count"] or 0)
                    + int(analysis_count["count"] or 0)
                    >= max_jobs_per_day
                ):
                    connection.rollback()
                    raise ValueError(
                        f"今天已达到 {max_jobs_per_day} 个 AI 任务的上限"
                    )
            snapshot = {
                "project_id": project_id,
                "chapter_id": chapter_id,
                "chapter_title": str(
                    chapter["title"] or "未命名章节"
                ),
                "chapter_position": int(chapter["position"]),
            }
            connection.execute(
                """
                INSERT INTO generation_jobs(
                    id, project_id, chapter_id, user_id, operation,
                    instruction, provider, model, credential_source,
                    status, created_at, context_snapshot_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,
                          'queued', ?, ?)
                """,
                (
                    job_id,
                    project_id,
                    chapter_id,
                    user_id,
                    operation,
                    instruction,
                    provider,
                    model,
                    credential_source,
                    now,
                    json.dumps(snapshot, ensure_ascii=False),
                ),
            )
            connection.commit()
        return job_id

    def create_reader_planning_job(
        self,
        *,
        user_id: int,
        project_id: str,
        request_id: str,
        provider: str,
        model: str,
        credential_source: str,
        max_jobs_per_day: Optional[int] = None,
    ) -> str:
        if credential_source not in {"default", "personal"}:
            raise ValueError("不支持的 API 凭据来源")
        job_id = uuid.uuid4().hex
        now = utc_now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            request = connection.execute(
                """
                SELECT r.id, r.status
                FROM reader_requests r
                JOIN novel_projects p ON p.id=r.project_id
                WHERE r.id=? AND r.project_id=? AND p.user_id=?
                  AND r.status IN
                      ('draft', 'failed', 'reviewing', 'proposing')
                """,
                (request_id, project_id, user_id),
            ).fetchone()
            if not request:
                connection.rollback()
                raise ValueError("读者意见不存在或已经处理")
            anchor = connection.execute(
                """
                SELECT id, title, position
                FROM novel_chapters
                WHERE project_id=?
                ORDER BY
                    CASE WHEN canonical_version_id IS NOT NULL
                         THEN 0 ELSE 1 END,
                    position DESC
                LIMIT 1
                """,
                (project_id,),
            ).fetchone()
            if not anchor:
                connection.rollback()
                raise ValueError("请先规划至少一章，再评估读者意见")
            if credential_source == "personal":
                credential = connection.execute(
                    "SELECT 1 FROM api_credentials WHERE user_id=?",
                    (user_id,),
                ).fetchone()
                if not credential:
                    connection.rollback()
                    raise ValueError(
                        "个人 DeepSeek API Key 不存在，请重新配置"
                    )
            active = connection.execute(
                """
                SELECT id, operation, subject_id
                FROM generation_jobs
                WHERE user_id=? AND status IN ('queued', 'running')
                ORDER BY created_at LIMIT 1
                """,
                (user_id,),
            ).fetchone()
            if active:
                connection.rollback()
                if (
                    str(active["operation"])
                    == "propose_reader_branches"
                    and str(active["subject_id"] or "") == request_id
                ):
                    return str(active["id"])
                raise ValueError(
                    "你已有一个写作任务正在排队或运行，请等待其完成"
                )
            if max_jobs_per_day is not None:
                day_start = datetime.now(timezone.utc).replace(
                    hour=0, minute=0, second=0, microsecond=0
                ).isoformat(timespec="seconds")
                generation_count = connection.execute(
                    """
                    SELECT COUNT(*) AS count FROM generation_jobs
                    WHERE user_id=? AND created_at>=?
                    """,
                    (user_id, day_start),
                ).fetchone()
                analysis_count = connection.execute(
                    """
                    SELECT COUNT(*) AS count FROM analysis_jobs
                    WHERE user_id=? AND created_at>=?
                    """,
                    (user_id, day_start),
                ).fetchone()
                if (
                    int(generation_count["count"] or 0)
                    + int(analysis_count["count"] or 0)
                    >= max_jobs_per_day
                ):
                    connection.rollback()
                    raise ValueError(
                        f"今天已达到 {max_jobs_per_day} 个 AI 任务的上限"
                    )
            snapshot = {
                "project_id": project_id,
                "request_id": request_id,
                "chapter_id": str(anchor["id"]),
                "chapter_title": str(anchor["title"]),
                "chapter_position": int(anchor["position"]),
            }
            connection.execute(
                """
                INSERT INTO generation_jobs(
                    id, project_id, chapter_id, user_id, operation,
                    instruction, provider, model, credential_source,
                    status, created_at, subject_id, context_snapshot_json
                ) VALUES (?, ?, ?, ?, 'propose_reader_branches', '',
                          ?, ?, ?, 'queued', ?, ?, ?)
                """,
                (
                    job_id,
                    project_id,
                    anchor["id"],
                    user_id,
                    provider,
                    model,
                    credential_source,
                    now,
                    request_id,
                    json.dumps(snapshot, ensure_ascii=False),
                ),
            )
            connection.execute(
                """
                UPDATE reader_requests
                SET status='proposing', updated_at=?
                WHERE id=?
                """,
                (now, request_id),
            )
            connection.commit()
        return job_id

    def create_quality_audit_job(
        self,
        *,
        user_id: int,
        project_id: str,
        chapter_id: str,
        version_id: str,
        provider: str,
        model: str,
        credential_source: str,
        max_jobs_per_day: Optional[int] = None,
    ) -> str:
        if credential_source not in {"default", "personal"}:
            raise ValueError("不支持的 API 凭据来源")
        job_id = uuid.uuid4().hex
        now = utc_now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            version = connection.execute(
                """
                SELECT v.id, ch.title, ch.position
                FROM novel_chapter_versions v
                JOIN novel_chapters ch ON ch.id=v.chapter_id
                JOIN novel_projects p ON p.id=ch.project_id
                WHERE v.id=? AND ch.id=? AND p.id=? AND p.user_id=?
                """,
                (version_id, chapter_id, project_id, user_id),
            ).fetchone()
            if not version:
                connection.rollback()
                raise ValueError("正文版本不存在")
            if credential_source == "personal":
                credential = connection.execute(
                    "SELECT 1 FROM api_credentials WHERE user_id=?",
                    (user_id,),
                ).fetchone()
                if not credential:
                    connection.rollback()
                    raise ValueError(
                        "个人 DeepSeek API Key 不存在，请重新配置"
                    )
            active = connection.execute(
                """
                SELECT id, chapter_id, operation, version_id
                FROM generation_jobs
                WHERE user_id=? AND status IN ('queued', 'running')
                ORDER BY created_at LIMIT 1
                """,
                (user_id,),
            ).fetchone()
            if active:
                connection.rollback()
                if (
                    str(active["chapter_id"]) == chapter_id
                    and str(active["operation"]) == "audit_chapter"
                    and str(active["version_id"] or "") == version_id
                ):
                    return str(active["id"])
                raise ValueError(
                    "你已有一个写作任务正在排队或运行，请等待其完成"
                )
            if max_jobs_per_day is not None:
                day_start = datetime.now(timezone.utc).replace(
                    hour=0, minute=0, second=0, microsecond=0
                ).isoformat(timespec="seconds")
                generation_count = connection.execute(
                    """
                    SELECT COUNT(*) AS count FROM generation_jobs
                    WHERE user_id=? AND created_at>=?
                    """,
                    (user_id, day_start),
                ).fetchone()
                analysis_count = connection.execute(
                    """
                    SELECT COUNT(*) AS count FROM analysis_jobs
                    WHERE user_id=? AND created_at>=?
                    """,
                    (user_id, day_start),
                ).fetchone()
                if (
                    int(generation_count["count"] or 0)
                    + int(analysis_count["count"] or 0)
                    >= max_jobs_per_day
                ):
                    connection.rollback()
                    raise ValueError(
                        f"今天已达到 {max_jobs_per_day} 个 AI 任务的上限"
                    )
            snapshot = {
                "project_id": project_id,
                "chapter_id": chapter_id,
                "version_id": version_id,
                "chapter_title": str(version["title"]),
                "chapter_position": int(version["position"]),
            }
            connection.execute(
                """
                INSERT INTO generation_jobs(
                    id, project_id, chapter_id, user_id, operation,
                    instruction, provider, model, credential_source,
                    status, created_at, version_id, context_snapshot_json
                ) VALUES (?, ?, ?, ?, 'audit_chapter', '', ?, ?, ?,
                          'queued', ?, ?, ?)
                """,
                (
                    job_id,
                    project_id,
                    chapter_id,
                    user_id,
                    provider,
                    model,
                    credential_source,
                    now,
                    version_id,
                    json.dumps(snapshot, ensure_ascii=False),
                ),
            )
            connection.commit()
        return job_id

    def create_style_job(
        self,
        *,
        user_id: int,
        project_id: str,
        chapter_id: str,
        version_id: str,
        operation: str,
        subject_id: str = "",
        instruction: str = "",
        provider: str,
        model: str,
        credential_source: str,
        max_jobs_per_day: Optional[int] = None,
    ) -> str:
        if operation not in {"audit_ai_style", "rewrite_style_issue"}:
            raise ValueError("不支持的文风编辑操作")
        if credential_source not in {"default", "personal"}:
            raise ValueError("不支持的 API 凭据来源")
        job_id = uuid.uuid4().hex
        now = utc_now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if operation == "audit_ai_style":
                target = connection.execute(
                    """
                    SELECT v.id AS version_id, ch.title, ch.position,
                           vp.status AS voice_status
                    FROM novel_chapter_versions v
                    JOIN novel_chapters ch ON ch.id=v.chapter_id
                    JOIN novel_projects p ON p.id=ch.project_id
                    JOIN novel_voice_profiles vp ON vp.project_id=p.id
                    WHERE v.id=? AND ch.id=? AND p.id=? AND p.user_id=?
                    """,
                    (version_id, chapter_id, project_id, user_id),
                ).fetchone()
            else:
                target = connection.execute(
                    """
                    SELECT i.version_id, ch.title, ch.position,
                           vp.status AS voice_status
                    FROM chapter_style_issues i
                    JOIN novel_chapters ch ON ch.id=i.chapter_id
                    JOIN novel_projects p ON p.id=i.project_id
                    JOIN novel_voice_profiles vp ON vp.project_id=p.id
                    WHERE i.id=? AND i.version_id=? AND i.chapter_id=?
                        AND i.project_id=? AND p.user_id=? AND i.status='open'
                    """,
                    (
                        subject_id,
                        version_id,
                        chapter_id,
                        project_id,
                        user_id,
                    ),
                ).fetchone()
            if not target:
                connection.rollback()
                raise ValueError("正文版本或待修改问题不存在")
            if str(target["voice_status"]) != "confirmed":
                connection.rollback()
                raise ValueError("请先确认作品声纹，再执行 AI 味编辑")
            if credential_source == "personal":
                credential = connection.execute(
                    "SELECT 1 FROM api_credentials WHERE user_id=?",
                    (user_id,),
                ).fetchone()
                if not credential:
                    connection.rollback()
                    raise ValueError(
                        "个人 DeepSeek API Key 不存在，请重新配置"
                    )
            active = connection.execute(
                """
                SELECT id, operation, version_id, subject_id
                FROM generation_jobs
                WHERE user_id=? AND status IN ('queued', 'running')
                ORDER BY created_at LIMIT 1
                """,
                (user_id,),
            ).fetchone()
            if active:
                connection.rollback()
                if (
                    str(active["operation"]) == operation
                    and str(active["version_id"] or "") == version_id
                    and str(active["subject_id"] or "") == subject_id
                ):
                    return str(active["id"])
                raise ValueError(
                    "你已有一个写作任务正在排队或运行，请等待其完成"
                )
            if max_jobs_per_day is not None:
                day_start = datetime.now(timezone.utc).replace(
                    hour=0, minute=0, second=0, microsecond=0
                ).isoformat(timespec="seconds")
                generation_count = connection.execute(
                    """
                    SELECT COUNT(*) AS count FROM generation_jobs
                    WHERE user_id=? AND created_at>=?
                    """,
                    (user_id, day_start),
                ).fetchone()
                analysis_count = connection.execute(
                    """
                    SELECT COUNT(*) AS count FROM analysis_jobs
                    WHERE user_id=? AND created_at>=?
                    """,
                    (user_id, day_start),
                ).fetchone()
                if (
                    int(generation_count["count"] or 0)
                    + int(analysis_count["count"] or 0)
                    >= max_jobs_per_day
                ):
                    connection.rollback()
                    raise ValueError(
                        f"今天已达到 {max_jobs_per_day} 个 AI 任务的上限"
                    )
            snapshot = {
                "project_id": project_id,
                "chapter_id": chapter_id,
                "version_id": version_id,
                "subject_id": subject_id,
                "chapter_title": str(target["title"]),
                "chapter_position": int(target["position"]),
                "operation": operation,
            }
            connection.execute(
                """
                INSERT INTO generation_jobs(
                    id, project_id, chapter_id, user_id, operation,
                    instruction, provider, model, credential_source,
                    status, created_at, version_id, subject_id,
                    context_snapshot_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?)
                """,
                (
                    job_id,
                    project_id,
                    chapter_id,
                    user_id,
                    operation,
                    instruction,
                    provider,
                    model,
                    credential_source,
                    now,
                    version_id,
                    subject_id or None,
                    json.dumps(snapshot, ensure_ascii=False),
                ),
            )
            connection.commit()
        return job_id

    def claim_next_generation(self) -> Optional[Dict[str, Any]]:
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = utc_now()
            connection.execute(
                """
                UPDATE generation_jobs
                SET status='queued', started_at=NULL, claim_token=NULL,
                    lease_expires_at=NULL,
                    error='上一次生成租约已过期，已自动重新排队'
                WHERE status='running' AND lease_expires_at IS NOT NULL
                    AND lease_expires_at<=?
                """,
                (now,),
            )
            row = connection.execute(
                """
                SELECT j.*, ch.title AS chapter_title, ch.outline,
                       ch.key_points, ch.content_path, ch.position,
                       ch.char_count
                FROM generation_jobs j
                JOIN novel_chapters ch ON ch.id=j.chapter_id
                WHERE j.status='queued'
                ORDER BY j.created_at
                LIMIT 1
                """
            ).fetchone()
            if not row:
                connection.commit()
                return None
            claim_token = uuid.uuid4().hex
            cursor = connection.execute(
                """
                UPDATE generation_jobs
                SET status='running', started_at=?, error=NULL,
                    claim_token=?, lease_expires_at=?
                WHERE id=? AND status='queued'
                """,
                (
                    now,
                    claim_token,
                    utc_after(2 * 60 * 60),
                    row["id"],
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return None
            connection.commit()
            claimed = dict(row)
            claimed["claim_token"] = claim_token
            return claimed

    def record_generation_context_snapshot(
        self,
        *,
        job_id: str,
        claim_token: str,
        snapshot: Mapping[str, Any],
    ) -> bool:
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE generation_jobs
                SET context_snapshot_json=?
                WHERE id=? AND status='running' AND claim_token=?
                """,
                (
                    json.dumps(snapshot, ensure_ascii=False),
                    job_id,
                    claim_token,
                ),
            )
            connection.commit()
        return cursor.rowcount == 1

    def complete_generation(
        self,
        *,
        job_id: str,
        claim_token: str,
        version_path: Path,
        result_char_count: int,
        input_tokens: int,
        output_tokens: int,
        warning: str = "",
        content_hash: str = "",
        quality_report: Optional[Mapping[str, Any]] = None,
        quality_provider: str = "local",
        quality_model: str = "deterministic-v1",
        quality_input_tokens: int = 0,
        quality_output_tokens: int = 0,
    ) -> bool:
        now = utc_now()
        version_id = uuid.uuid4().hex
        audit_id = uuid.uuid4().hex if quality_report else None
        report = dict(quality_report or {})
        quality_status = str(report.get("verdict") or "pending")
        if quality_status not in {"pass", "block", "pending"}:
            raise ValueError("不支持的正文质量状态")
        effective_count = int(
            report.get("effective_char_count") or result_char_count
        )
        findings = report.get("findings") or []
        hard_issue_count = sum(
            isinstance(item, Mapping) and item.get("severity") == "hard"
            for item in findings
        )
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = connection.execute(
                """
                SELECT j.id, j.project_id, j.chapter_id, j.operation,
                       ch.working_version_id
                FROM generation_jobs j
                JOIN novel_chapters ch ON ch.id=j.chapter_id
                WHERE j.id=? AND j.status='running' AND j.claim_token=?
                """,
                (job_id, claim_token),
            ).fetchone()
            if not job:
                connection.rollback()
                return False
            connection.execute(
                """
                INSERT INTO novel_chapter_versions(
                    id, chapter_id, job_id, kind, content_path,
                    char_count, created_at, parent_version_id, status,
                    source, content_hash, created_by, quality_status,
                    effective_char_count, hard_issue_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'candidate', 'generated',
                          ?, 'ai', ?, ?, ?)
                """,
                (
                    version_id,
                    job["chapter_id"],
                    job_id,
                    job["operation"],
                    str(version_path),
                    result_char_count,
                    now,
                    job["working_version_id"],
                    content_hash,
                    quality_status,
                    effective_count,
                    hard_issue_count,
                ),
            )
            if quality_report and audit_id:
                connection.execute(
                    """
                    INSERT INTO chapter_quality_audits(
                        id, project_id, chapter_id, version_id, verdict,
                        effective_char_count, minimum_char_count,
                        expansion_attempted, provider, model, result_json,
                        input_tokens, output_tokens, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        audit_id,
                        job["project_id"],
                        job["chapter_id"],
                        version_id,
                        quality_status,
                        effective_count,
                        int(report.get("minimum_effective_chars") or 2000),
                        int(bool(report.get("expansion_attempted"))),
                        quality_provider,
                        quality_model,
                        json.dumps(report, ensure_ascii=False),
                        quality_input_tokens,
                        quality_output_tokens,
                        now,
                    ),
                )
            connection.execute(
                """
                UPDATE generation_jobs
                SET status='completed', input_tokens=?, output_tokens=?,
                    result_char_count=?, error=?, finished_at=?,
                    claim_token=NULL, lease_expires_at=NULL, version_id=?,
                    result_json=?
                WHERE id=?
                """,
                (
                    input_tokens,
                    output_tokens,
                    result_char_count,
                    warning[:2000] or None,
                    now,
                    version_id,
                    json.dumps(
                        {
                            "version_id": version_id,
                            "quality_audit_id": audit_id,
                            "quality": report or None,
                        },
                        ensure_ascii=False,
                    ),
                    job_id,
                ),
            )
            connection.execute(
                """
                UPDATE novel_chapters
                SET char_count=?, status='draft', working_version_id=?,
                    updated_at=?
                WHERE id=?
                """,
                (result_char_count, version_id, now, job["chapter_id"]),
            )
            connection.execute(
                "UPDATE novel_projects SET updated_at=? WHERE id=?",
                (now, job["project_id"]),
            )
            connection.commit()
        return True

    def complete_memory_extraction(
        self,
        *,
        job_id: str,
        claim_token: str,
        result: Mapping[str, Any],
        input_tokens: int,
        output_tokens: int,
    ) -> bool:
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE generation_jobs
                SET status='completed', result_json=?, input_tokens=?,
                    output_tokens=?, finished_at=?, claim_token=NULL,
                    lease_expires_at=NULL, error=NULL
                WHERE id=? AND status='running' AND claim_token=?
                    AND operation='extract_story_delta'
                """,
                (
                    json.dumps(result, ensure_ascii=False),
                    input_tokens,
                    output_tokens,
                    utc_now(),
                    job_id,
                    claim_token,
                ),
            )
            connection.commit()
        return cursor.rowcount == 1

    def complete_chapter_planning(
        self,
        *,
        job_id: str,
        claim_token: str,
        result: Mapping[str, Any],
        input_tokens: int,
        output_tokens: int,
        operation: str = "plan_chapter",
    ) -> bool:
        if operation not in {"plan_chapter", "plan_scene_beats"}:
            raise ValueError("不支持的章节规划操作")
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE generation_jobs
                SET status='completed', result_json=?, input_tokens=?,
                    output_tokens=?, finished_at=?, claim_token=NULL,
                    lease_expires_at=NULL, error=NULL
                WHERE id=? AND status='running' AND claim_token=?
                    AND operation=?
                """,
                (
                    json.dumps(result, ensure_ascii=False),
                    input_tokens,
                    output_tokens,
                    utc_now(),
                    job_id,
                    claim_token,
                    operation,
                ),
            )
            connection.commit()
        return cursor.rowcount == 1

    def complete_quality_audit(
        self,
        *,
        job_id: str,
        claim_token: str,
        report: Mapping[str, Any],
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> bool:
        now = utc_now()
        audit_id = uuid.uuid4().hex
        verdict = str(report.get("verdict") or "")
        if verdict not in {"pass", "block", "pending"}:
            raise ValueError("不支持的正文质量状态")
        effective_count = int(report.get("effective_char_count") or 0)
        findings = report.get("findings") or []
        hard_issue_count = sum(
            isinstance(item, Mapping) and item.get("severity") == "hard"
            for item in findings
        )
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            job = connection.execute(
                """
                SELECT id, project_id, chapter_id, version_id
                FROM generation_jobs
                WHERE id=? AND status='running' AND claim_token=?
                    AND operation='audit_chapter'
                """,
                (job_id, claim_token),
            ).fetchone()
            if not job or not job["version_id"]:
                connection.rollback()
                return False
            version = connection.execute(
                """
                SELECT id FROM novel_chapter_versions
                WHERE id=? AND chapter_id=?
                """,
                (job["version_id"], job["chapter_id"]),
            ).fetchone()
            if not version:
                connection.rollback()
                return False
            connection.execute(
                """
                INSERT INTO chapter_quality_audits(
                    id, project_id, chapter_id, version_id, verdict,
                    effective_char_count, minimum_char_count,
                    expansion_attempted, provider, model, result_json,
                    input_tokens, output_tokens, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    audit_id,
                    job["project_id"],
                    job["chapter_id"],
                    job["version_id"],
                    verdict,
                    effective_count,
                    int(report.get("minimum_effective_chars") or 2000),
                    int(bool(report.get("expansion_attempted"))),
                    provider,
                    model,
                    json.dumps(dict(report), ensure_ascii=False),
                    input_tokens,
                    output_tokens,
                    now,
                ),
            )
            connection.execute(
                """
                UPDATE novel_chapter_versions
                SET quality_status=?, effective_char_count=?,
                    hard_issue_count=?, quality_override_reason='',
                    quality_overridden_at=NULL
                WHERE id=?
                """,
                (
                    verdict,
                    effective_count,
                    hard_issue_count,
                    job["version_id"],
                ),
            )
            connection.execute(
                """
                UPDATE generation_jobs
                SET status='completed', result_json=?, input_tokens=?,
                    output_tokens=?, result_char_count=?, error=NULL,
                    finished_at=?, claim_token=NULL, lease_expires_at=NULL
                WHERE id=?
                """,
                (
                    json.dumps(
                        {
                            "quality_audit_id": audit_id,
                            "version_id": str(job["version_id"]),
                            "quality": dict(report),
                        },
                        ensure_ascii=False,
                    ),
                    input_tokens,
                    output_tokens,
                    effective_count,
                    now,
                    job_id,
                ),
            )
            connection.commit()
        return True

    def complete_style_job(
        self,
        *,
        job_id: str,
        claim_token: str,
        result: Mapping[str, Any],
        input_tokens: int,
        output_tokens: int,
    ) -> bool:
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE generation_jobs
                SET status='completed', result_json=?, input_tokens=?,
                    output_tokens=?, finished_at=?, claim_token=NULL,
                    lease_expires_at=NULL, error=NULL
                WHERE id=? AND status='running' AND claim_token=?
                    AND operation IN ('audit_ai_style',
                                      'rewrite_style_issue')
                """,
                (
                    json.dumps(dict(result), ensure_ascii=False),
                    input_tokens,
                    output_tokens,
                    utc_now(),
                    job_id,
                    claim_token,
                ),
            )
            connection.commit()
        return cursor.rowcount == 1

    def complete_reader_planning(
        self,
        *,
        job_id: str,
        claim_token: str,
        result: Mapping[str, Any],
        input_tokens: int,
        output_tokens: int,
    ) -> bool:
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE generation_jobs
                SET status='completed', result_json=?, input_tokens=?,
                    output_tokens=?, finished_at=?, claim_token=NULL,
                    lease_expires_at=NULL, error=NULL
                WHERE id=? AND status='running' AND claim_token=?
                    AND operation='propose_reader_branches'
                """,
                (
                    json.dumps(dict(result), ensure_ascii=False),
                    input_tokens,
                    output_tokens,
                    utc_now(),
                    job_id,
                    claim_token,
                ),
            )
            connection.commit()
        return cursor.rowcount == 1

    def fail_generation(
        self,
        job_id: str,
        claim_token: str,
        error: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> bool:
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE generation_jobs
                SET status='failed', error=?, input_tokens=?,
                    output_tokens=?, finished_at=?, claim_token=NULL,
                    lease_expires_at=NULL
                WHERE id=? AND status='running' AND claim_token=?
                """,
                (
                    error[:2000],
                    input_tokens,
                    output_tokens,
                    utc_now(),
                    job_id,
                    claim_token,
                ),
            )
            if cursor.rowcount:
                connection.execute(
                    """
                    UPDATE reader_requests
                    SET status='failed', updated_at=?
                    WHERE id=(
                        SELECT subject_id FROM generation_jobs
                        WHERE id=? AND operation='propose_reader_branches'
                    )
                    """,
                    (utc_now(), job_id),
                )
            connection.commit()
        return cursor.rowcount == 1

    def release_generation_claim(
        self, job_id: str, claim_token: str, error: str
    ) -> bool:
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE generation_jobs
                SET status='queued', started_at=NULL, claim_token=NULL,
                    lease_expires_at=NULL, error=?
                WHERE id=? AND status='running' AND claim_token=?
                """,
                (error[:2000], job_id, claim_token),
            )
            connection.commit()
        return cursor.rowcount == 1

    def get_generation_job(
        self, user_id: int, job_id: str
    ) -> Optional[Dict[str, Any]]:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT j.*, p.title AS project_title,
                       ch.title AS chapter_title, ch.position
                FROM generation_jobs j
                JOIN novel_projects p ON p.id=j.project_id
                JOIN novel_chapters ch ON ch.id=j.chapter_id
                WHERE j.id=? AND j.user_id=?
                """,
                (job_id, user_id),
            ).fetchone()
        return dict(row) if row else None

    def create_document(
        self,
        *,
        user_id: int,
        title: str,
        original_filename: str,
        source_path: Path,
        source_encoding: str,
        text_length: int,
        chunks: Iterable[ChapterChunk],
        chapter_paths: Iterable[Path],
        max_documents: Optional[int] = None,
        max_stored_chars: Optional[int] = None,
    ) -> str:
        document_id = source_path.parent.name
        chunk_list = list(chunks)
        path_list = list(chapter_paths)
        if len(chunk_list) != len(path_list):
            raise ValueError("章节与文件数量不一致")

        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            usage = connection.execute(
                """
                SELECT COUNT(*) AS document_count,
                       COALESCE(SUM(char_count), 0) AS stored_chars
                FROM documents WHERE user_id=?
                """,
                (user_id,),
            ).fetchone()
            if (
                max_documents is not None
                and int(usage["document_count"] or 0) >= max_documents
            ):
                connection.rollback()
                raise ValueError(f"每个账号最多保存 {max_documents} 本文档")
            if (
                max_stored_chars is not None
                and int(usage["stored_chars"] or 0) + text_length > max_stored_chars
            ):
                connection.rollback()
                raise ValueError(
                    f"账号累计正文不能超过 {max_stored_chars:,} 字"
                )
            connection.execute(
                """
                INSERT INTO documents(
                    id, user_id, title, original_filename, source_path,
                    source_encoding, char_count, split_strategy, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    user_id,
                    title,
                    original_filename,
                    str(source_path),
                    source_encoding,
                    text_length,
                    "heading+smart-fallback-v1",
                    utc_now(),
                ),
            )
            for position, (chunk, content_path) in enumerate(
                zip(chunk_list, path_list), start=1
            ):
                connection.execute(
                    """
                    INSERT INTO chapters(
                        id, document_id, position, title, kind, content_path,
                        char_count, source_start, source_end, part_number, part_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid.uuid4().hex,
                        document_id,
                        position,
                        chunk.title,
                        chunk.kind,
                        str(content_path),
                        len(chunk.text),
                        chunk.source_start,
                        chunk.source_end,
                        chunk.part_number,
                        chunk.part_count,
                    ),
                )
            connection.commit()
        return document_id

    def list_documents(self, user_id: int) -> List[Dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT d.*,
                    (SELECT COUNT(*) FROM chapters c WHERE c.document_id=d.id)
                        AS chapter_count,
                    (SELECT j.id FROM analysis_jobs j WHERE j.document_id=d.id
                        ORDER BY j.created_at DESC LIMIT 1) AS latest_job_id,
                    (SELECT j.status FROM analysis_jobs j WHERE j.document_id=d.id
                        ORDER BY j.created_at DESC LIMIT 1) AS latest_job_status,
                    (SELECT j.completed_chapters FROM analysis_jobs j
                        WHERE j.document_id=d.id ORDER BY j.created_at DESC LIMIT 1)
                        AS completed_chapters
                FROM documents d
                WHERE d.user_id=?
                ORDER BY d.created_at DESC
                """,
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_document(self, user_id: int, document_id: str) -> Optional[Dict[str, Any]]:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT d.*,
                    (SELECT COUNT(*) FROM chapters c WHERE c.document_id=d.id)
                        AS chapter_count,
                    (SELECT j.id FROM analysis_jobs j WHERE j.document_id=d.id
                        ORDER BY j.created_at DESC LIMIT 1) AS latest_job_id,
                    (SELECT j.status FROM analysis_jobs j WHERE j.document_id=d.id
                        ORDER BY j.created_at DESC LIMIT 1) AS latest_job_status
                FROM documents d
                WHERE d.id=? AND d.user_id=?
                """,
                (document_id, user_id),
            ).fetchone()
        return dict(row) if row else None

    def list_chapters(
        self, user_id: int, document_id: str, job_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        params: List[Any] = [user_id, document_id]
        analysis_join = ""
        analysis_fields = (
            "NULL AS analysis_id, NULL AS analysis_status, NULL AS result_json"
        )
        if job_id:
            analysis_join = (
                "LEFT JOIN chapter_analyses a ON a.chapter_id=c.id AND a.job_id=?"
            )
            analysis_fields = (
                "a.id AS analysis_id, a.status AS analysis_status, a.result_json"
            )
            params = [job_id, user_id, document_id]
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT c.*, {analysis_fields}
                FROM chapters c
                JOIN documents d ON d.id=c.document_id
                {analysis_join}
                WHERE d.user_id=? AND d.id=?
                ORDER BY c.position
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def create_job(
        self,
        *,
        user_id: int,
        document_id: str,
        provider: str,
        model: str,
        credential_source: str = "default",
        max_jobs_per_day: Optional[int] = None,
    ) -> str:
        if credential_source not in {"default", "personal"}:
            raise ValueError("不支持的 API 凭据来源")
        job_id = uuid.uuid4().hex
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if credential_source == "personal":
                credential = connection.execute(
                    "SELECT 1 FROM api_credentials WHERE user_id=?",
                    (user_id,),
                ).fetchone()
                if not credential:
                    connection.rollback()
                    raise ValueError("个人 DeepSeek API Key 不存在，请重新配置")
            active = connection.execute(
                """
                SELECT id FROM analysis_jobs
                WHERE document_id=? AND user_id=? AND status IN ('queued', 'running')
                ORDER BY created_at DESC LIMIT 1
                """,
                (document_id, user_id),
            ).fetchone()
            if active:
                connection.rollback()
                return str(active["id"])
            other_active = connection.execute(
                """
                SELECT 1
                WHERE EXISTS(
                    SELECT 1 FROM analysis_jobs
                    WHERE user_id=? AND status IN ('queued', 'running')
                ) OR EXISTS(
                    SELECT 1 FROM generation_jobs
                    WHERE user_id=? AND status IN ('queued', 'running')
                )
                """,
                (user_id, user_id),
            ).fetchone()
            if other_active:
                connection.rollback()
                raise ValueError("你已有一个 AI 任务正在排队或运行，请等待其完成")
            if max_jobs_per_day is not None:
                day_start = datetime.now(timezone.utc).replace(
                    hour=0, minute=0, second=0, microsecond=0
                ).isoformat(timespec="seconds")
                job_count = connection.execute(
                    """
                    SELECT COUNT(*) AS count FROM analysis_jobs
                    WHERE user_id=? AND created_at>=?
                    """,
                    (user_id, day_start),
                ).fetchone()
                generation_count = connection.execute(
                    """
                    SELECT COUNT(*) AS count FROM generation_jobs
                    WHERE user_id=? AND created_at>=?
                    """,
                    (user_id, day_start),
                ).fetchone()
                if (
                    int(job_count["count"] or 0)
                    + int(generation_count["count"] or 0)
                    >= max_jobs_per_day
                ):
                    connection.rollback()
                    raise ValueError(
                        f"今天已达到 {max_jobs_per_day} 个分析任务的上限"
                    )
            chapters = connection.execute(
                """
                SELECT c.id FROM chapters c
                JOIN documents d ON d.id=c.document_id
                WHERE c.document_id=? AND d.user_id=?
                ORDER BY c.position
                """,
                (document_id, user_id),
            ).fetchall()
            if not chapters:
                connection.rollback()
                raise ValueError("文档没有可分析章节")
            connection.execute(
                """
                INSERT INTO analysis_jobs(
                    id, document_id, user_id, provider, model, credential_source,
                    status, total_chapters, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?)
                """,
                (
                    job_id,
                    document_id,
                    user_id,
                    provider,
                    model,
                    credential_source,
                    len(chapters),
                    utc_now(),
                ),
            )
            for chapter in chapters:
                connection.execute(
                    """
                    INSERT INTO chapter_analyses(id, job_id, chapter_id, status)
                    VALUES (?, ?, ?, 'queued')
                    """,
                    (uuid.uuid4().hex, job_id, chapter["id"]),
                )
            connection.commit()
        return job_id

    def get_job(self, user_id: int, job_id: str) -> Optional[Dict[str, Any]]:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT j.*, d.title AS document_title
                FROM analysis_jobs j
                JOIN documents d ON d.id=j.document_id
                WHERE j.id=? AND j.user_id=?
                """,
                (job_id, user_id),
            ).fetchone()
        return dict(row) if row else None

    def claim_next_analysis(self) -> Optional[Dict[str, Any]]:
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = utc_now()
            connection.execute(
                """
                UPDATE chapter_analyses
                SET status='queued', started_at=NULL, claim_token=NULL,
                    lease_expires_at=NULL,
                    error='上一次处理租约已过期，已自动重新排队'
                WHERE status='running' AND lease_expires_at IS NOT NULL
                    AND lease_expires_at<=?
                """,
                (now,),
            )
            row = connection.execute(
                """
                SELECT a.id AS analysis_id, a.job_id, a.chapter_id, a.attempts,
                       c.title AS chapter_title, c.content_path, c.position,
                       j.user_id, j.document_id, j.provider, j.model,
                       j.credential_source
                FROM chapter_analyses a
                JOIN analysis_jobs j ON j.id=a.job_id
                JOIN chapters c ON c.id=a.chapter_id
                WHERE a.status='queued' AND j.status IN ('queued', 'running')
                ORDER BY j.created_at, c.position
                LIMIT 1
                """
            ).fetchone()
            if not row:
                connection.commit()
                return None
            claim_token = uuid.uuid4().hex
            cursor = connection.execute(
                """
                UPDATE chapter_analyses
                SET status='running', attempts=attempts+1, started_at=?, error=NULL,
                    claim_token=?, lease_expires_at=?
                WHERE id=? AND status='queued'
                """,
                (
                    now,
                    claim_token,
                    utc_after(2 * 60 * 60),
                    row["analysis_id"],
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return None
            connection.execute(
                """
                UPDATE analysis_jobs
                SET status='running', started_at=COALESCE(started_at, ?), error=NULL
                WHERE id=?
                """,
                (now, row["job_id"]),
            )
            connection.commit()
            claimed = dict(row)
            claimed["claim_token"] = claim_token
            return claimed

    def release_claim(
        self, analysis_id: str, job_id: str, claim_token: str, error: str
    ) -> bool:
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE chapter_analyses
                SET status='queued', started_at=NULL, claim_token=NULL,
                    lease_expires_at=NULL, error=?
                WHERE id=? AND job_id=? AND status='running' AND claim_token=?
                """,
                (error[:2000], analysis_id, job_id, claim_token),
            )
            connection.commit()
            return cursor.rowcount == 1

    def _refresh_job(self, connection: sqlite3.Connection, job_id: str) -> None:
        counts = connection.execute(
            """
            SELECT
                SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) AS completed,
                SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed,
                SUM(CASE WHEN status IN ('queued', 'running') THEN 1 ELSE 0 END) AS active,
                COALESCE(SUM(input_tokens), 0) AS input_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens
            FROM chapter_analyses WHERE job_id=?
            """,
            (job_id,),
        ).fetchone()
        completed = int(counts["completed"] or 0)
        failed = int(counts["failed"] or 0)
        active = int(counts["active"] or 0)
        if active:
            status = "running"
            finished_at = None
        elif failed and completed:
            status = "partial"
            finished_at = utc_now()
        elif failed:
            status = "failed"
            finished_at = utc_now()
        else:
            status = "completed"
            finished_at = utc_now()
        connection.execute(
            """
            UPDATE analysis_jobs
            SET status=?, completed_chapters=?, failed_chapters=?,
                input_tokens=?, output_tokens=?, finished_at=?
            WHERE id=?
            """,
            (
                status,
                completed,
                failed,
                int(counts["input_tokens"] or 0),
                int(counts["output_tokens"] or 0),
                finished_at,
                job_id,
            ),
        )

    def complete_analysis(
        self,
        *,
        analysis_id: str,
        job_id: str,
        result: Mapping[str, Any],
        raw_response: str,
        input_tokens: int,
        output_tokens: int,
        claim_token: str,
    ) -> bool:
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE chapter_analyses
                SET status='completed', result_json=?, raw_response=?,
                    input_tokens=?, output_tokens=?, error=NULL, finished_at=?,
                    claim_token=NULL, lease_expires_at=NULL
                WHERE id=? AND job_id=? AND status='running' AND claim_token=?
                """,
                (
                    json.dumps(result, ensure_ascii=False),
                    raw_response,
                    input_tokens,
                    output_tokens,
                    utc_now(),
                    analysis_id,
                    job_id,
                    claim_token,
                ),
            )
            if cursor.rowcount == 1:
                self._refresh_job(connection, job_id)
            connection.commit()
            return cursor.rowcount == 1

    def fail_analysis(
        self,
        analysis_id: str,
        job_id: str,
        error: str,
        claim_token: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> bool:
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE chapter_analyses
                SET status='failed', error=?, input_tokens=?, output_tokens=?,
                    finished_at=?, claim_token=NULL, lease_expires_at=NULL
                WHERE id=? AND job_id=? AND status='running' AND claim_token=?
                """,
                (
                    error[:2000],
                    input_tokens,
                    output_tokens,
                    utc_now(),
                    analysis_id,
                    job_id,
                    claim_token,
                ),
            )
            if cursor.rowcount == 1:
                self._refresh_job(connection, job_id)
            connection.commit()
            return cursor.rowcount == 1

    def retry_failed(self, user_id: int, job_id: str) -> bool:
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            owner = connection.execute(
                "SELECT id, status FROM analysis_jobs WHERE id=? AND user_id=?",
                (job_id, user_id),
            ).fetchone()
            if not owner or owner["status"] not in {"partial", "failed"}:
                connection.rollback()
                return False
            other_active = connection.execute(
                """
                SELECT 1
                WHERE EXISTS(
                    SELECT 1 FROM analysis_jobs
                    WHERE user_id=? AND id<>?
                        AND status IN ('queued', 'running')
                ) OR EXISTS(
                    SELECT 1 FROM generation_jobs
                    WHERE user_id=? AND status IN ('queued', 'running')
                )
                """,
                (user_id, job_id, user_id),
            ).fetchone()
            if other_active:
                connection.rollback()
                return False
            cursor = connection.execute(
                """
                UPDATE chapter_analyses
                SET status='queued', error=NULL, started_at=NULL, finished_at=NULL,
                    claim_token=NULL, lease_expires_at=NULL
                WHERE job_id=? AND status='failed'
                """,
                (job_id,),
            )
            if cursor.rowcount:
                connection.execute(
                    """
                    UPDATE analysis_jobs
                    SET status='queued', failed_chapters=0, error=NULL, finished_at=NULL
                    WHERE id=?
                    """,
                    (job_id,),
                )
            connection.commit()
            return bool(cursor.rowcount)

    def get_analysis(
        self, user_id: int, analysis_id: str
    ) -> Optional[Dict[str, Any]]:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT a.*, c.title AS chapter_title, c.position, c.char_count,
                       c.content_path, d.title AS document_title, d.id AS document_id,
                       j.model, j.provider
                FROM chapter_analyses a
                JOIN chapters c ON c.id=a.chapter_id
                JOIN analysis_jobs j ON j.id=a.job_id
                JOIN documents d ON d.id=j.document_id
                WHERE a.id=? AND j.user_id=?
                """,
                (analysis_id, user_id),
            ).fetchone()
        return dict(row) if row else None

    def export_job(self, user_id: int, job_id: str) -> Optional[Dict[str, Any]]:
        job = self.get_job(user_id, job_id)
        if not job:
            return None
        chapters = self.list_chapters(user_id, job["document_id"], job_id)
        exported = []
        for chapter in chapters:
            result = (
                json.loads(chapter["result_json"]) if chapter.get("result_json") else None
            )
            exported.append(
                {
                    "position": chapter["position"],
                    "title": chapter["title"],
                    "status": chapter["analysis_status"],
                    "analysis": result,
                }
            )
        return {
            "schema_version": "1.0",
            "document": {
                "id": job["document_id"],
                "title": job["document_title"],
            },
            "job": {
                "id": job["id"],
                "provider": job["provider"],
                "model": job["model"],
                "status": job["status"],
                "created_at": job["created_at"],
                "finished_at": job["finished_at"],
            },
            "chapters": exported,
        }
