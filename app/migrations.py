from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Callable, Iterable

from .continuity import replay_canonical_state
from .memory_identity import backfill_memory_identities
from .memory_search import rebuild_memory_search_documents


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    apply: Callable[[sqlite3.Connection, str], None]


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row["name"])
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }


def _add_column(
    connection: sqlite3.Connection,
    table: str,
    name: str,
    definition: str,
) -> None:
    if name not in _columns(connection, table):
        connection.execute(
            f"ALTER TABLE {table} ADD COLUMN {name} {definition}"
        )


def _execute_statements(
    connection: sqlite3.Connection, statements: Iterable[str]
) -> None:
    for statement in statements:
        connection.execute(statement)


def _core_memory_v1(
    connection: sqlite3.Connection, applied_at: str
) -> None:
    _add_column(
        connection,
        "novel_projects",
        "story_promise",
        "TEXT NOT NULL DEFAULT ''",
    )
    _add_column(
        connection,
        "novel_projects",
        "target_audience",
        "TEXT NOT NULL DEFAULT ''",
    )
    _add_column(
        connection,
        "novel_projects",
        "core_appeal",
        "TEXT NOT NULL DEFAULT ''",
    )
    _add_column(
        connection,
        "novel_projects",
        "ending_constraint",
        "TEXT NOT NULL DEFAULT ''",
    )
    _add_column(
        connection,
        "novel_projects",
        "planning_horizon",
        "INTEGER NOT NULL DEFAULT 20",
    )
    _add_column(
        connection,
        "novel_projects",
        "canonical_branch_id",
        "TEXT NOT NULL DEFAULT 'main'",
    )
    _add_column(
        connection,
        "novel_projects",
        "workflow_version",
        "INTEGER NOT NULL DEFAULT 1",
    )

    _add_column(
        connection,
        "novel_chapters",
        "canonical_version_id",
        "TEXT REFERENCES novel_chapter_versions(id) ON DELETE SET NULL",
    )
    _add_column(
        connection,
        "novel_chapters",
        "working_version_id",
        "TEXT REFERENCES novel_chapter_versions(id) ON DELETE SET NULL",
    )
    _add_column(
        connection,
        "novel_chapters",
        "needs_recheck",
        "INTEGER NOT NULL DEFAULT 0",
    )

    _add_column(
        connection,
        "novel_chapter_versions",
        "parent_version_id",
        "TEXT REFERENCES novel_chapter_versions(id) ON DELETE SET NULL",
    )
    _add_column(
        connection,
        "novel_chapter_versions",
        "status",
        "TEXT NOT NULL DEFAULT 'candidate'",
    )
    _add_column(
        connection,
        "novel_chapter_versions",
        "source",
        "TEXT NOT NULL DEFAULT 'manual'",
    )
    _add_column(
        connection,
        "novel_chapter_versions",
        "content_hash",
        "TEXT NOT NULL DEFAULT ''",
    )
    _add_column(
        connection,
        "novel_chapter_versions",
        "change_summary",
        "TEXT NOT NULL DEFAULT ''",
    )
    _add_column(
        connection,
        "novel_chapter_versions",
        "created_by",
        "TEXT NOT NULL DEFAULT 'system'",
    )

    _add_column(
        connection,
        "generation_jobs",
        "version_id",
        "TEXT REFERENCES novel_chapter_versions(id) ON DELETE SET NULL",
    )
    _add_column(
        connection,
        "generation_jobs",
        "result_json",
        "TEXT",
    )
    _add_column(
        connection,
        "generation_jobs",
        "context_snapshot_json",
        "TEXT",
    )

    _execute_statements(
        connection,
        (
            """
            CREATE TABLE IF NOT EXISTS story_deltas (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                chapter_id TEXT NOT NULL
                    REFERENCES novel_chapters(id) ON DELETE CASCADE,
                version_id TEXT NOT NULL
                    REFERENCES novel_chapter_versions(id) ON DELETE CASCADE,
                branch_id TEXT NOT NULL DEFAULT 'main',
                status TEXT NOT NULL DEFAULT 'proposed',
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                reviewed_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS chapter_memory (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                chapter_id TEXT NOT NULL
                    REFERENCES novel_chapters(id) ON DELETE CASCADE,
                version_id TEXT NOT NULL
                    REFERENCES novel_chapter_versions(id) ON DELETE CASCADE,
                delta_id TEXT NOT NULL
                    REFERENCES story_deltas(id) ON DELETE CASCADE,
                branch_id TEXT NOT NULL DEFAULT 'main',
                summary TEXT NOT NULL,
                key_events_json TEXT NOT NULL DEFAULT '[]',
                unresolved_questions_json TEXT NOT NULL DEFAULT '[]',
                keywords_json TEXT NOT NULL DEFAULT '[]',
                record_status TEXT NOT NULL DEFAULT 'canon',
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS story_events (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                chapter_id TEXT NOT NULL
                    REFERENCES novel_chapters(id) ON DELETE CASCADE,
                version_id TEXT NOT NULL
                    REFERENCES novel_chapter_versions(id) ON DELETE CASCADE,
                delta_id TEXT NOT NULL
                    REFERENCES story_deltas(id) ON DELETE CASCADE,
                branch_id TEXT NOT NULL DEFAULT 'main',
                position INTEGER NOT NULL,
                summary TEXT NOT NULL,
                participants_json TEXT NOT NULL DEFAULT '[]',
                location TEXT NOT NULL DEFAULT '',
                story_time TEXT NOT NULL DEFAULT '',
                causes_json TEXT NOT NULL DEFAULT '[]',
                effects_json TEXT NOT NULL DEFAULT '[]',
                evidence TEXT NOT NULL DEFAULT '',
                record_status TEXT NOT NULL DEFAULT 'canon',
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS story_facts (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                chapter_id TEXT NOT NULL
                    REFERENCES novel_chapters(id) ON DELETE CASCADE,
                version_id TEXT NOT NULL
                    REFERENCES novel_chapter_versions(id) ON DELETE CASCADE,
                delta_id TEXT NOT NULL
                    REFERENCES story_deltas(id) ON DELETE CASCADE,
                branch_id TEXT NOT NULL DEFAULT 'main',
                fact_type TEXT NOT NULL,
                subject_type TEXT NOT NULL,
                subject_id TEXT,
                subject_name TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object_json TEXT NOT NULL,
                valid_from_chapter_id TEXT
                    REFERENCES novel_chapters(id) ON DELETE SET NULL,
                visibility_json TEXT NOT NULL DEFAULT '{}',
                confidence REAL NOT NULL DEFAULT 1.0,
                fact_status TEXT NOT NULL DEFAULT 'canon',
                evidence TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS character_knowledge (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                chapter_id TEXT NOT NULL
                    REFERENCES novel_chapters(id) ON DELETE CASCADE,
                version_id TEXT NOT NULL
                    REFERENCES novel_chapter_versions(id) ON DELETE CASCADE,
                delta_id TEXT NOT NULL
                    REFERENCES story_deltas(id) ON DELETE CASCADE,
                branch_id TEXT NOT NULL DEFAULT 'main',
                character_id TEXT
                    REFERENCES novel_characters(id) ON DELETE SET NULL,
                character_name TEXT NOT NULL,
                fact_text TEXT NOT NULL,
                knowledge_state TEXT NOT NULL,
                learned_via TEXT NOT NULL DEFAULT '',
                evidence TEXT NOT NULL DEFAULT '',
                record_status TEXT NOT NULL DEFAULT 'canon',
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS plot_threads (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                chapter_id TEXT NOT NULL
                    REFERENCES novel_chapters(id) ON DELETE CASCADE,
                version_id TEXT NOT NULL
                    REFERENCES novel_chapter_versions(id) ON DELETE CASCADE,
                delta_id TEXT NOT NULL
                    REFERENCES story_deltas(id) ON DELETE CASCADE,
                branch_id TEXT NOT NULL DEFAULT 'main',
                thread_name TEXT NOT NULL,
                thread_type TEXT NOT NULL DEFAULT 'plot',
                action TEXT NOT NULL,
                update_text TEXT NOT NULL,
                promise TEXT NOT NULL DEFAULT '',
                target_payoff TEXT NOT NULL DEFAULT '',
                evidence TEXT NOT NULL DEFAULT '',
                record_status TEXT NOT NULL DEFAULT 'canon',
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS foreshadowing (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                chapter_id TEXT NOT NULL
                    REFERENCES novel_chapters(id) ON DELETE CASCADE,
                version_id TEXT NOT NULL
                    REFERENCES novel_chapter_versions(id) ON DELETE CASCADE,
                delta_id TEXT NOT NULL
                    REFERENCES story_deltas(id) ON DELETE CASCADE,
                branch_id TEXT NOT NULL DEFAULT 'main',
                hook_name TEXT NOT NULL,
                action TEXT NOT NULL,
                description TEXT NOT NULL,
                intended_payoff TEXT NOT NULL DEFAULT '',
                evidence TEXT NOT NULL DEFAULT '',
                record_status TEXT NOT NULL DEFAULT 'canon',
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_chapter_versions_status
            ON novel_chapter_versions(chapter_id, status, created_at DESC)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_story_deltas_chapter
            ON story_deltas(chapter_id, status, created_at DESC)
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_story_delta_active_version
            ON story_deltas(version_id)
            WHERE status IN ('proposed', 'author_edited', 'accepted', 'projected')
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_chapter_memory_version
            ON chapter_memory(version_id)
            WHERE record_status='canon'
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_story_events_project
            ON story_events(project_id, branch_id, record_status, created_at)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_story_facts_project
            ON story_facts(
                project_id, branch_id, fact_status, subject_name, predicate
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_character_knowledge_project
            ON character_knowledge(
                project_id, branch_id, character_name, record_status
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_plot_threads_project
            ON plot_threads(project_id, branch_id, thread_name, record_status)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_foreshadowing_project
            ON foreshadowing(project_id, branch_id, hook_name, record_status)
            """,
        ),
    )

    connection.execute(
        """
        UPDATE novel_chapter_versions
        SET source=CASE
            WHEN kind='manual' THEN 'manual'
            ELSE 'generated'
        END
        WHERE source='manual'
        """
    )

    chapters = connection.execute(
        """
        SELECT id, content_path, char_count, canonical_version_id
        FROM novel_chapters
        """
    ).fetchall()
    for chapter in chapters:
        if chapter["canonical_version_id"]:
            continue
        latest = connection.execute(
            """
            SELECT id FROM novel_chapter_versions
            WHERE chapter_id=?
            ORDER BY created_at DESC, rowid DESC
            LIMIT 1
            """,
            (chapter["id"],),
        ).fetchone()
        if latest:
            version_id = str(latest["id"])
        elif int(chapter["char_count"] or 0) > 0:
            version_id = uuid.uuid4().hex
            connection.execute(
                """
                INSERT INTO novel_chapter_versions(
                    id, chapter_id, kind, content_path, char_count, created_at,
                    status, source, created_by
                ) VALUES (?, ?, 'legacy', ?, ?, ?, 'canonical', 'manual',
                          'migration')
                """,
                (
                    version_id,
                    chapter["id"],
                    chapter["content_path"],
                    int(chapter["char_count"] or 0),
                    applied_at,
                ),
            )
        else:
            continue
        connection.execute(
            """
            UPDATE novel_chapter_versions
            SET status=CASE WHEN id=? THEN 'canonical' ELSE 'archived' END
            WHERE chapter_id=?
            """,
            (version_id, chapter["id"]),
        )
        connection.execute(
            """
            UPDATE novel_chapters
            SET canonical_version_id=?, working_version_id=?,
                status=CASE WHEN char_count>0 THEN 'canonical' ELSE status END
            WHERE id=?
            """,
            (version_id, version_id, chapter["id"]),
        )


def _planning_v2(
    connection: sqlite3.Connection, applied_at: str
) -> None:
    _execute_statements(
        connection,
        (
            """
            CREATE TABLE IF NOT EXISTS novel_volumes (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                position INTEGER NOT NULL,
                title TEXT NOT NULL,
                goal TEXT NOT NULL DEFAULT '',
                start_state TEXT NOT NULL DEFAULT '',
                end_state TEXT NOT NULL DEFAULT '',
                major_conflict TEXT NOT NULL DEFAULT '',
                payoff TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'planning',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(project_id, position)
            )
            """,
        ),
    )
    _add_column(
        connection,
        "novel_chapters",
        "volume_id",
        "TEXT REFERENCES novel_volumes(id) ON DELETE SET NULL",
    )
    _execute_statements(
        connection,
        (
            """
            CREATE TABLE IF NOT EXISTS novel_chapter_plans (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                chapter_id TEXT NOT NULL UNIQUE
                    REFERENCES novel_chapters(id) ON DELETE CASCADE,
                purpose TEXT NOT NULL DEFAULT '',
                start_state TEXT NOT NULL DEFAULT '',
                end_state TEXT NOT NULL DEFAULT '',
                central_conflict TEXT NOT NULL DEFAULT '',
                emotional_value TEXT NOT NULL DEFAULT '',
                plot_threads_json TEXT NOT NULL DEFAULT '[]',
                must_happen_json TEXT NOT NULL DEFAULT '[]',
                must_preserve_json TEXT NOT NULL DEFAULT '[]',
                forbidden_json TEXT NOT NULL DEFAULT '[]',
                foreshadow_setup_json TEXT NOT NULL DEFAULT '[]',
                foreshadow_payoff_json TEXT NOT NULL DEFAULT '[]',
                ending_hook TEXT NOT NULL DEFAULT '',
                target_chars INTEGER NOT NULL DEFAULT 3000,
                status TEXT NOT NULL DEFAULT 'draft',
                source TEXT NOT NULL DEFAULT 'manual',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                confirmed_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS novel_scene_beats (
                id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL
                    REFERENCES novel_chapter_plans(id) ON DELETE CASCADE,
                position INTEGER NOT NULL,
                pov_character TEXT NOT NULL DEFAULT '',
                goal TEXT NOT NULL,
                obstacle TEXT NOT NULL,
                action TEXT NOT NULL,
                reveal TEXT NOT NULL DEFAULT '',
                conceal TEXT NOT NULL DEFAULT '',
                subtext TEXT NOT NULL DEFAULT '',
                location TEXT NOT NULL DEFAULT '',
                key_items_json TEXT NOT NULL DEFAULT '[]',
                end_state TEXT NOT NULL,
                transition TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(plan_id, position)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_novel_volumes_project_position
            ON novel_volumes(project_id, position)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_novel_chapter_plans_project
            ON novel_chapter_plans(project_id, status, updated_at DESC)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_novel_scene_beats_plan
            ON novel_scene_beats(plan_id, position)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_novel_chapters_volume
            ON novel_chapters(volume_id, position)
            """,
        ),
    )

    chapters = connection.execute(
        """
        SELECT ch.id, ch.project_id, ch.outline, ch.key_points,
               p.target_chapter_chars
        FROM novel_chapters ch
        JOIN novel_projects p ON p.id=ch.project_id
        WHERE NOT EXISTS(
            SELECT 1 FROM novel_chapter_plans cp
            WHERE cp.chapter_id=ch.id
        )
        """
    ).fetchall()
    for chapter in chapters:
        must_happen = [
            line.strip()
            for line in str(chapter["key_points"] or "").splitlines()
            if line.strip()
        ]
        connection.execute(
            """
            INSERT INTO novel_chapter_plans(
                id, project_id, chapter_id, purpose, must_happen_json,
                target_chars, status, source, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'draft', 'migration', ?, ?)
            """,
            (
                uuid.uuid4().hex,
                chapter["project_id"],
                chapter["id"],
                str(chapter["outline"] or ""),
                json.dumps(must_happen, ensure_ascii=False),
                int(chapter["target_chapter_chars"] or 3000),
                applied_at,
                applied_at,
            ),
        )


def _quality_gate_v3(
    connection: sqlite3.Connection, applied_at: str
) -> None:
    _add_column(
        connection,
        "novel_chapter_versions",
        "quality_status",
        "TEXT NOT NULL DEFAULT 'pending'",
    )
    _add_column(
        connection,
        "novel_chapter_versions",
        "effective_char_count",
        "INTEGER NOT NULL DEFAULT 0",
    )
    _add_column(
        connection,
        "novel_chapter_versions",
        "hard_issue_count",
        "INTEGER NOT NULL DEFAULT 0",
    )
    _add_column(
        connection,
        "novel_chapter_versions",
        "quality_override_reason",
        "TEXT NOT NULL DEFAULT ''",
    )
    _add_column(
        connection,
        "novel_chapter_versions",
        "quality_overridden_at",
        "TEXT",
    )
    _execute_statements(
        connection,
        (
            """
            CREATE TABLE IF NOT EXISTS chapter_quality_audits (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                chapter_id TEXT NOT NULL
                    REFERENCES novel_chapters(id) ON DELETE CASCADE,
                version_id TEXT NOT NULL
                    REFERENCES novel_chapter_versions(id) ON DELETE CASCADE,
                verdict TEXT NOT NULL,
                effective_char_count INTEGER NOT NULL,
                minimum_char_count INTEGER NOT NULL,
                expansion_attempted INTEGER NOT NULL DEFAULT 0,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                result_json TEXT NOT NULL,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_quality_audits_version
            ON chapter_quality_audits(version_id, created_at DESC)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_quality_audits_project
            ON chapter_quality_audits(project_id, verdict, created_at DESC)
            """,
        ),
    )
    connection.execute(
        """
        UPDATE novel_chapter_versions
        SET effective_char_count=CASE
                WHEN effective_char_count=0 THEN char_count
                ELSE effective_char_count
            END,
            quality_status=CASE
                WHEN status='canonical' THEN 'overridden'
                ELSE quality_status
            END,
            quality_override_reason=CASE
                WHEN status='canonical' AND quality_override_reason=''
                THEN '迁移前已由作者确认为正史'
                ELSE quality_override_reason
            END,
            quality_overridden_at=CASE
                WHEN status='canonical' AND quality_overridden_at IS NULL
                THEN ?
                ELSE quality_overridden_at
            END
        """,
        (applied_at,),
    )


def _style_editor_v4(
    connection: sqlite3.Connection, applied_at: str
) -> None:
    _add_column(
        connection,
        "generation_jobs",
        "subject_id",
        "TEXT",
    )
    _add_column(
        connection,
        "novel_chapter_versions",
        "style_status",
        "TEXT NOT NULL DEFAULT 'pending'",
    )
    _add_column(
        connection,
        "novel_chapter_versions",
        "style_issue_count",
        "INTEGER NOT NULL DEFAULT 0",
    )
    _execute_statements(
        connection,
        (
            """
            CREATE TABLE IF NOT EXISTS novel_voice_profiles (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL UNIQUE
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                narration_rules TEXT NOT NULL DEFAULT '',
                sentence_rhythm TEXT NOT NULL DEFAULT '',
                dialogue_voice TEXT NOT NULL DEFAULT '',
                sensory_palette TEXT NOT NULL DEFAULT '',
                metaphor_policy TEXT NOT NULL DEFAULT '',
                allowed_omissions TEXT NOT NULL DEFAULT '',
                preferred_patterns_json TEXT NOT NULL DEFAULT '[]',
                banned_expressions_json TEXT NOT NULL DEFAULT '[]',
                author_notes TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'draft',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                confirmed_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS chapter_style_audits (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                chapter_id TEXT NOT NULL
                    REFERENCES novel_chapters(id) ON DELETE CASCADE,
                version_id TEXT NOT NULL
                    REFERENCES novel_chapter_versions(id) ON DELETE CASCADE,
                voice_profile_id TEXT NOT NULL
                    REFERENCES novel_voice_profiles(id) ON DELETE RESTRICT,
                summary TEXT NOT NULL,
                issue_count INTEGER NOT NULL,
                dropped_issue_count INTEGER NOT NULL DEFAULT 0,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                result_json TEXT NOT NULL,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS chapter_style_issues (
                id TEXT PRIMARY KEY,
                audit_id TEXT NOT NULL
                    REFERENCES chapter_style_audits(id) ON DELETE CASCADE,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                chapter_id TEXT NOT NULL
                    REFERENCES novel_chapters(id) ON DELETE CASCADE,
                version_id TEXT NOT NULL
                    REFERENCES novel_chapter_versions(id) ON DELETE CASCADE,
                position INTEGER NOT NULL,
                paragraph_index INTEGER NOT NULL,
                start_offset INTEGER NOT NULL,
                end_offset INTEGER NOT NULL,
                quote TEXT NOT NULL,
                issue_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                evidence TEXT NOT NULL,
                reader_impact TEXT NOT NULL,
                rewrite_direction TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS style_rewrite_candidates (
                id TEXT PRIMARY KEY,
                issue_id TEXT NOT NULL
                    REFERENCES chapter_style_issues(id) ON DELETE CASCADE,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                chapter_id TEXT NOT NULL
                    REFERENCES novel_chapters(id) ON DELETE CASCADE,
                source_version_id TEXT NOT NULL
                    REFERENCES novel_chapter_versions(id) ON DELETE CASCADE,
                position INTEGER NOT NULL,
                replacement_text TEXT NOT NULL,
                rationale TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'candidate',
                result_version_id TEXT
                    REFERENCES novel_chapter_versions(id) ON DELETE SET NULL,
                created_at TEXT NOT NULL,
                decided_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS author_style_preferences (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                issue_id TEXT
                    REFERENCES chapter_style_issues(id) ON DELETE SET NULL,
                issue_type TEXT NOT NULL,
                decision TEXT NOT NULL,
                original_text TEXT NOT NULL,
                replacement_text TEXT NOT NULL DEFAULT '',
                guidance TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_style_audits_version
            ON chapter_style_audits(version_id, created_at DESC)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_style_issues_version
            ON chapter_style_issues(version_id, status, position)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_style_rewrites_issue
            ON style_rewrite_candidates(issue_id, status, position)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_style_preferences_project
            ON author_style_preferences(project_id, created_at DESC)
            """,
        ),
    )
    projects = connection.execute(
        """
        SELECT id, style_guide FROM novel_projects
        WHERE NOT EXISTS(
            SELECT 1 FROM novel_voice_profiles vp
            WHERE vp.project_id=novel_projects.id
        )
        """
    ).fetchall()
    for project in projects:
        connection.execute(
            """
            INSERT INTO novel_voice_profiles(
                id, project_id, narration_rules, status,
                created_at, updated_at
            ) VALUES (?, ?, ?, 'draft', ?, ?)
            """,
            (
                uuid.uuid4().hex,
                project["id"],
                str(project["style_guide"] or ""),
                applied_at,
                applied_at,
            ),
        )


def _reader_decisions_v5(
    connection: sqlite3.Connection, applied_at: str
) -> None:
    del applied_at
    _execute_statements(
        connection,
        (
            """
            CREATE TABLE IF NOT EXISTS reader_requests (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                raw_text TEXT NOT NULL,
                request_type TEXT NOT NULL,
                impact_scope TEXT NOT NULL,
                priority TEXT NOT NULL,
                constraints_json TEXT NOT NULL DEFAULT '[]',
                author_note TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'draft',
                chosen_proposal_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                decided_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS reader_branch_proposals (
                id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL
                    REFERENCES reader_requests(id) ON DELETE CASCADE,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                generation_job_id TEXT
                    REFERENCES generation_jobs(id) ON DELETE SET NULL,
                position INTEGER NOT NULL,
                label TEXT NOT NULL,
                summary TEXT NOT NULL,
                satisfies_json TEXT NOT NULL DEFAULT '[]',
                sacrifices_json TEXT NOT NULL DEFAULT '[]',
                affected_characters_json TEXT NOT NULL DEFAULT '[]',
                affected_plot_threads_json TEXT NOT NULL DEFAULT '[]',
                promise_impact TEXT NOT NULL,
                future_changes_json TEXT NOT NULL DEFAULT '[]',
                affects_published_canon INTEGER NOT NULL DEFAULT 0,
                published_canon_impact TEXT NOT NULL DEFAULT '',
                risk_level TEXT NOT NULL,
                risks_json TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'candidate',
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                created_at TEXT NOT NULL,
                decided_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS reader_plan_applications (
                id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL
                    REFERENCES reader_requests(id) ON DELETE CASCADE,
                proposal_id TEXT NOT NULL
                    REFERENCES reader_branch_proposals(id) ON DELETE CASCADE,
                chapter_id TEXT NOT NULL
                    REFERENCES novel_chapters(id) ON DELETE CASCADE,
                chapter_position INTEGER NOT NULL,
                action TEXT NOT NULL,
                before_json TEXT NOT NULL DEFAULT '{}',
                after_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS canon_impact_reports (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                chapter_id TEXT NOT NULL
                    REFERENCES novel_chapters(id) ON DELETE CASCADE,
                old_version_id TEXT NOT NULL
                    REFERENCES novel_chapter_versions(id) ON DELETE CASCADE,
                proposed_version_id TEXT NOT NULL
                    REFERENCES novel_chapter_versions(id) ON DELETE CASCADE,
                status TEXT NOT NULL DEFAULT 'pending',
                summary TEXT NOT NULL,
                downstream_count INTEGER NOT NULL DEFAULT 0,
                item_count INTEGER NOT NULL DEFAULT 0,
                override_reason TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                decided_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS canon_impact_items (
                id TEXT PRIMARY KEY,
                report_id TEXT NOT NULL
                    REFERENCES canon_impact_reports(id) ON DELETE CASCADE,
                position INTEGER NOT NULL,
                item_type TEXT NOT NULL,
                downstream_chapter_id TEXT
                    REFERENCES novel_chapters(id) ON DELETE SET NULL,
                source_record_id TEXT,
                title TEXT NOT NULL,
                evidence TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                recommended_action TEXT NOT NULL,
                decision TEXT NOT NULL DEFAULT 'recheck',
                decision_note TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                decided_at TEXT
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_reader_requests_project
            ON reader_requests(project_id, created_at DESC)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_reader_proposals_request
            ON reader_branch_proposals(request_id, status, position)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_reader_applications_proposal
            ON reader_plan_applications(proposal_id, chapter_position)
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_canon_impact_pending
            ON canon_impact_reports(chapter_id, old_version_id,
                                    proposed_version_id)
            WHERE status='pending'
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_canon_impact_project
            ON canon_impact_reports(project_id, created_at DESC)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_canon_impact_items_report
            ON canon_impact_items(report_id, position)
            """,
        ),
    )


def _technique_library_v6(
    connection: sqlite3.Connection, applied_at: str
) -> None:
    del applied_at
    _execute_statements(
        connection,
        (
            """
            CREATE TABLE IF NOT EXISTS reference_technique_cards (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL
                    REFERENCES users(id) ON DELETE CASCADE,
                source_document_id TEXT
                    REFERENCES documents(id) ON DELETE SET NULL,
                source_chapter_id TEXT
                    REFERENCES chapters(id) ON DELETE SET NULL,
                source_analysis_id TEXT
                    REFERENCES chapter_analyses(id) ON DELETE SET NULL,
                name TEXT NOT NULL,
                dimension TEXT NOT NULL,
                source_location TEXT NOT NULL,
                observation TEXT NOT NULL,
                effect TEXT NOT NULL,
                suitable_for_json TEXT NOT NULL DEFAULT '[]',
                unsuitable_for_json TEXT NOT NULL DEFAULT '[]',
                execution_rule TEXT NOT NULL,
                originality_boundary TEXT NOT NULL,
                author_note TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, source_analysis_id, name)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS novel_technique_bindings (
                id TEXT PRIMARY KEY,
                technique_id TEXT NOT NULL
                    REFERENCES reference_technique_cards(id)
                    ON DELETE CASCADE,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                scope_type TEXT NOT NULL,
                volume_id TEXT
                    REFERENCES novel_volumes(id) ON DELETE CASCADE,
                chapter_id TEXT
                    REFERENCES novel_chapters(id) ON DELETE CASCADE,
                scene_beat_id TEXT
                    REFERENCES novel_scene_beats(id) ON DELETE CASCADE,
                usage_modes_json TEXT NOT NULL DEFAULT '[]',
                author_adaptation TEXT NOT NULL DEFAULT '',
                priority INTEGER NOT NULL DEFAULT 50,
                status TEXT NOT NULL DEFAULT 'enabled',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_technique_cards_user
            ON reference_technique_cards(user_id, status, updated_at DESC)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_technique_cards_analysis
            ON reference_technique_cards(source_analysis_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_technique_bindings_project
            ON novel_technique_bindings(project_id, status, priority DESC)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_technique_bindings_card
            ON novel_technique_bindings(technique_id, status)
            """,
        ),
    )


def _scene_workbench_v7(
    connection: sqlite3.Connection, applied_at: str
) -> None:
    del applied_at
    _execute_statements(
        connection,
        (
            """
            CREATE TABLE IF NOT EXISTS novel_scene_versions (
                id TEXT PRIMARY KEY,
                scene_beat_id TEXT NOT NULL
                    REFERENCES novel_scene_beats(id) ON DELETE CASCADE,
                job_id TEXT
                    REFERENCES generation_jobs(id) ON DELETE SET NULL,
                parent_version_id TEXT
                    REFERENCES novel_scene_versions(id) ON DELETE SET NULL,
                kind TEXT NOT NULL,
                source TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'candidate',
                content_path TEXT NOT NULL,
                char_count INTEGER NOT NULL DEFAULT 0,
                effective_char_count INTEGER NOT NULL DEFAULT 0,
                content_hash TEXT NOT NULL DEFAULT '',
                plan_fingerprint TEXT NOT NULL,
                created_by TEXT NOT NULL DEFAULT 'system',
                quality_status TEXT NOT NULL DEFAULT 'pending',
                hard_issue_count INTEGER NOT NULL DEFAULT 0,
                quality_override_reason TEXT NOT NULL DEFAULT '',
                quality_overridden_at TEXT,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS scene_quality_audits (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                chapter_id TEXT NOT NULL
                    REFERENCES novel_chapters(id) ON DELETE CASCADE,
                scene_beat_id TEXT NOT NULL
                    REFERENCES novel_scene_beats(id) ON DELETE CASCADE,
                scene_version_id TEXT NOT NULL
                    REFERENCES novel_scene_versions(id) ON DELETE CASCADE,
                verdict TEXT NOT NULL,
                effective_char_count INTEGER NOT NULL,
                minimum_effective_chars INTEGER NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                result_json TEXT NOT NULL,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS novel_scene_assembly_items (
                chapter_version_id TEXT NOT NULL
                    REFERENCES novel_chapter_versions(id) ON DELETE CASCADE,
                scene_beat_id TEXT NOT NULL
                    REFERENCES novel_scene_beats(id) ON DELETE CASCADE,
                scene_version_id TEXT NOT NULL
                    REFERENCES novel_scene_versions(id) ON DELETE RESTRICT,
                position INTEGER NOT NULL,
                PRIMARY KEY(chapter_version_id, scene_beat_id),
                UNIQUE(chapter_version_id, position)
            )
            """,
        ),
    )
    _add_column(
        connection,
        "novel_scene_beats",
        "beat_status",
        "TEXT NOT NULL DEFAULT 'active'",
    )
    _add_column(
        connection,
        "novel_scene_beats",
        "current_version_id",
        "TEXT REFERENCES novel_scene_versions(id) ON DELETE SET NULL",
    )
    _add_column(
        connection,
        "novel_scene_beats",
        "draft_status",
        "TEXT NOT NULL DEFAULT 'empty'",
    )
    _add_column(
        connection,
        "novel_scene_beats",
        "draft_plan_fingerprint",
        "TEXT NOT NULL DEFAULT ''",
    )
    _add_column(
        connection,
        "novel_scene_beats",
        "draft_char_count",
        "INTEGER NOT NULL DEFAULT 0",
    )
    _add_column(
        connection,
        "novel_scene_beats",
        "draft_updated_at",
        "TEXT",
    )
    _execute_statements(
        connection,
        (
            """
            CREATE INDEX IF NOT EXISTS idx_scene_versions_beat_created
            ON novel_scene_versions(scene_beat_id, created_at DESC)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_scene_audits_version_created
            ON scene_quality_audits(scene_version_id, created_at DESC)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_scene_assembly_version
            ON novel_scene_assembly_items(chapter_version_id, position)
            """,
        ),
    )


def _memory_search_v8(
    connection: sqlite3.Connection, applied_at: str
) -> None:
    try:
        _execute_statements(
            connection,
            (
                """
                CREATE TABLE IF NOT EXISTS story_memory_search_documents (
                    id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    project_id TEXT NOT NULL
                        REFERENCES novel_projects(id) ON DELETE CASCADE,
                    branch_id TEXT NOT NULL DEFAULT 'main',
                    chapter_id TEXT NOT NULL
                        REFERENCES novel_chapters(id) ON DELETE CASCADE,
                    chapter_position INTEGER NOT NULL,
                    chapter_title TEXT NOT NULL,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    keywords TEXT NOT NULL DEFAULT '',
                    search_terms TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    UNIQUE(source_type, source_id)
                )
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_memory_search_scope
                ON story_memory_search_documents(
                    project_id, branch_id, chapter_position, chapter_id
                )
                """,
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS story_memory_fts
                USING fts5(
                    title,
                    body,
                    keywords,
                    search_terms,
                    content='story_memory_search_documents',
                    content_rowid='rowid',
                    tokenize='unicode61 remove_diacritics 2'
                )
                """,
                """
                CREATE TRIGGER IF NOT EXISTS memory_search_documents_ai
                AFTER INSERT ON story_memory_search_documents BEGIN
                    INSERT INTO story_memory_fts(
                        rowid, title, body, keywords, search_terms
                    ) VALUES (
                        new.rowid, new.title, new.body, new.keywords,
                        new.search_terms
                    );
                END
                """,
                """
                CREATE TRIGGER IF NOT EXISTS memory_search_documents_ad
                AFTER DELETE ON story_memory_search_documents BEGIN
                    INSERT INTO story_memory_fts(
                        story_memory_fts, rowid, title, body, keywords,
                        search_terms
                    ) VALUES (
                        'delete', old.rowid, old.title, old.body,
                        old.keywords, old.search_terms
                    );
                END
                """,
                """
                CREATE TRIGGER IF NOT EXISTS memory_search_documents_au
                AFTER UPDATE ON story_memory_search_documents BEGIN
                    INSERT INTO story_memory_fts(
                        story_memory_fts, rowid, title, body, keywords,
                        search_terms
                    ) VALUES (
                        'delete', old.rowid, old.title, old.body,
                        old.keywords, old.search_terms
                    );
                    INSERT INTO story_memory_fts(
                        rowid, title, body, keywords, search_terms
                    ) VALUES (
                        new.rowid, new.title, new.body, new.keywords,
                        new.search_terms
                    );
                END
                """,
            ),
        )
    except sqlite3.OperationalError as exc:
        raise RuntimeError(
            "当前 SQLite 未启用 FTS5，无法建立长篇故事记忆检索"
        ) from exc
    rebuild_memory_search_documents(connection, created_at=applied_at)


def _continuity_replay_v9(
    connection: sqlite3.Connection, applied_at: str
) -> None:
    _execute_statements(
        connection,
        (
            """
            CREATE TABLE IF NOT EXISTS continuity_replay_runs (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                branch_id TEXT NOT NULL DEFAULT 'main',
                trigger_type TEXT NOT NULL,
                trigger_chapter_id TEXT
                    REFERENCES novel_chapters(id) ON DELETE SET NULL,
                status TEXT NOT NULL DEFAULT 'running',
                replayed_chapter_count INTEGER NOT NULL DEFAULT 0,
                issue_count INTEGER NOT NULL DEFAULT 0,
                final_state_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                finished_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS story_state_snapshots (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                branch_id TEXT NOT NULL DEFAULT 'main',
                chapter_id TEXT NOT NULL
                    REFERENCES novel_chapters(id) ON DELETE CASCADE,
                version_id TEXT NOT NULL
                    REFERENCES novel_chapter_versions(id) ON DELETE CASCADE,
                delta_id TEXT NOT NULL
                    REFERENCES story_deltas(id) ON DELETE CASCADE,
                chapter_position INTEGER NOT NULL,
                state_json TEXT NOT NULL,
                issue_count INTEGER NOT NULL DEFAULT 0,
                replay_run_id TEXT NOT NULL
                    REFERENCES continuity_replay_runs(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL,
                UNIQUE(project_id, branch_id, chapter_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS continuity_issues (
                id TEXT PRIMARY KEY,
                fingerprint TEXT NOT NULL UNIQUE,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                branch_id TEXT NOT NULL DEFAULT 'main',
                chapter_id TEXT NOT NULL
                    REFERENCES novel_chapters(id) ON DELETE CASCADE,
                version_id TEXT NOT NULL
                    REFERENCES novel_chapter_versions(id) ON DELETE CASCADE,
                delta_id TEXT NOT NULL
                    REFERENCES story_deltas(id) ON DELETE CASCADE,
                chapter_position INTEGER NOT NULL,
                issue_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_name TEXT NOT NULL,
                field_name TEXT NOT NULL,
                expected_value TEXT NOT NULL,
                actual_value TEXT NOT NULL,
                message TEXT NOT NULL,
                evidence TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'open',
                author_note TEXT NOT NULL DEFAULT '',
                active INTEGER NOT NULL DEFAULT 1,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                resolved_at TEXT,
                acknowledged_at TEXT,
                replay_run_id TEXT NOT NULL
                    REFERENCES continuity_replay_runs(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_continuity_runs_project
            ON continuity_replay_runs(
                project_id, branch_id, created_at DESC
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_state_snapshots_scope
            ON story_state_snapshots(
                project_id, branch_id, chapter_position
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_continuity_issues_scope
            ON continuity_issues(
                project_id, branch_id, active, severity, chapter_position
            )
            """,
        ),
    )
    projects = connection.execute(
        """
        SELECT id, canonical_branch_id
        FROM novel_projects
        ORDER BY id
        """
    ).fetchall()
    for project in projects:
        replay_canonical_state(
            connection,
            project_id=str(project["id"]),
            branch_id=str(project["canonical_branch_id"] or "main"),
            trigger_type="migration_v9",
            trigger_chapter_id=None,
            created_at=applied_at,
        )
def _continuity_lifecycle_v10(
    connection: sqlite3.Connection, applied_at: str
) -> None:
    projects = connection.execute(
        """
        SELECT id, canonical_branch_id
        FROM novel_projects
        ORDER BY id
        """
    ).fetchall()
    for project in projects:
        replay_canonical_state(
            connection,
            project_id=str(project["id"]),
            branch_id=str(project["canonical_branch_id"] or "main"),
            trigger_type="migration_v10",
            trigger_chapter_id=None,
            created_at=applied_at,
        )


def _memory_identity_and_causality_v11(
    connection: sqlite3.Connection, applied_at: str
) -> None:
    _execute_statements(
        connection,
        (
            """
            CREATE TABLE IF NOT EXISTS memory_identities (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                identity_type TEXT NOT NULL,
                canonical_text TEXT NOT NULL,
                canonical_key TEXT NOT NULL,
                linked_record_id TEXT,
                source TEXT NOT NULL DEFAULT 'story_delta',
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(project_id, identity_type, canonical_key)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS memory_identity_aliases (
                id TEXT PRIMARY KEY,
                identity_id TEXT NOT NULL
                    REFERENCES memory_identities(id) ON DELETE CASCADE,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                identity_type TEXT NOT NULL,
                alias_text TEXT NOT NULL,
                alias_key TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(project_id, identity_type, alias_key)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_memory_identities_project
            ON memory_identities(
                project_id, identity_type, status, canonical_text
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_memory_alias_identity
            ON memory_identity_aliases(identity_id, alias_key)
            """,
        ),
    )
    _add_column(
        connection,
        "story_events",
        "event_identity_id",
        "TEXT REFERENCES memory_identities(id) ON DELETE SET NULL",
    )
    _add_column(
        connection,
        "story_events",
        "event_key",
        "TEXT NOT NULL DEFAULT ''",
    )
    _add_column(
        connection,
        "story_events",
        "cause_event_keys_json",
        "TEXT NOT NULL DEFAULT '[]'",
    )
    _add_column(
        connection,
        "story_facts",
        "subject_identity_id",
        "TEXT REFERENCES memory_identities(id) ON DELETE SET NULL",
    )
    _add_column(
        connection,
        "story_facts",
        "fact_identity_id",
        "TEXT REFERENCES memory_identities(id) ON DELETE SET NULL",
    )
    _add_column(
        connection,
        "character_knowledge",
        "character_identity_id",
        "TEXT REFERENCES memory_identities(id) ON DELETE SET NULL",
    )
    _add_column(
        connection,
        "character_knowledge",
        "fact_identity_id",
        "TEXT REFERENCES memory_identities(id) ON DELETE SET NULL",
    )
    _add_column(
        connection,
        "character_knowledge",
        "fact_key",
        "TEXT NOT NULL DEFAULT ''",
    )
    _add_column(
        connection,
        "plot_threads",
        "thread_identity_id",
        "TEXT REFERENCES memory_identities(id) ON DELETE SET NULL",
    )
    _add_column(
        connection,
        "foreshadowing",
        "hook_identity_id",
        "TEXT REFERENCES memory_identities(id) ON DELETE SET NULL",
    )

    projects = connection.execute(
        """
        SELECT id, canonical_branch_id
        FROM novel_projects
        ORDER BY id
        """
    ).fetchall()
    for project in projects:
        project_id = str(project["id"])
        backfill_memory_identities(
            connection,
            project_id=project_id,
            created_at=applied_at,
        )
        replay_canonical_state(
            connection,
            project_id=project_id,
            branch_id=str(project["canonical_branch_id"] or "main"),
            trigger_type="migration_v11",
            trigger_chapter_id=None,
            created_at=applied_at,
        )
    rebuild_memory_search_documents(connection, created_at=applied_at)


def _voice_profile_learning_v12(
    connection: sqlite3.Connection, applied_at: str
) -> None:
    del applied_at
    _execute_statements(
        connection,
        (
            """
            CREATE TABLE IF NOT EXISTS voice_profile_suggestions (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                user_id INTEGER NOT NULL
                    REFERENCES users(id) ON DELETE CASCADE,
                sample_title TEXT NOT NULL,
                sample_text TEXT NOT NULL,
                sample_hash TEXT NOT NULL,
                sample_char_count INTEGER NOT NULL,
                author_intent TEXT NOT NULL DEFAULT '',
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                credential_source TEXT NOT NULL DEFAULT 'default',
                status TEXT NOT NULL DEFAULT 'queued',
                suggestion_json TEXT,
                raw_response TEXT,
                valid_evidence_count INTEGER NOT NULL DEFAULT 0,
                dropped_evidence_count INTEGER NOT NULL DEFAULT 0,
                applied_profile_json TEXT,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                reviewed_at TEXT,
                claim_token TEXT,
                lease_expires_at TEXT
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_voice_suggestions_project
            ON voice_profile_suggestions(
                project_id, status, created_at DESC
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_voice_suggestions_queue
            ON voice_profile_suggestions(status, created_at)
            """,
        ),
    )


def _manual_edit_preference_learning_v13(
    connection: sqlite3.Connection, applied_at: str
) -> None:
    del applied_at
    _add_column(
        connection,
        "novel_scene_versions",
        "change_summary",
        "TEXT NOT NULL DEFAULT ''",
    )
    _execute_statements(
        connection,
        (
            """
            CREATE TABLE IF NOT EXISTS editing_preference_suggestions (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                user_id INTEGER NOT NULL
                    REFERENCES users(id) ON DELETE CASCADE,
                chapter_id TEXT NOT NULL
                    REFERENCES novel_chapters(id) ON DELETE CASCADE,
                source_type TEXT NOT NULL,
                scene_beat_id TEXT
                    REFERENCES novel_scene_beats(id) ON DELETE CASCADE,
                before_version_id TEXT NOT NULL,
                after_version_id TEXT NOT NULL,
                before_content_hash TEXT NOT NULL,
                after_content_hash TEXT NOT NULL,
                change_sample_json TEXT NOT NULL,
                changed_char_count INTEGER NOT NULL,
                author_change_summary TEXT NOT NULL DEFAULT '',
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                credential_source TEXT NOT NULL DEFAULT 'default',
                status TEXT NOT NULL DEFAULT 'queued',
                suggestion_json TEXT,
                raw_response TEXT,
                valid_evidence_count INTEGER NOT NULL DEFAULT 0,
                dropped_evidence_count INTEGER NOT NULL DEFAULT 0,
                applied_preference_ids_json TEXT,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                reviewed_at TEXT,
                claim_token TEXT,
                lease_expires_at TEXT
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_edit_pref_suggestions_project
            ON editing_preference_suggestions(
                project_id, status, created_at DESC
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_edit_pref_suggestions_queue
            ON editing_preference_suggestions(status, created_at)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_edit_pref_suggestions_source
            ON editing_preference_suggestions(
                source_type, after_version_id, status
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS author_editing_preferences (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                suggestion_id TEXT
                    REFERENCES editing_preference_suggestions(id)
                    ON DELETE SET NULL,
                category TEXT NOT NULL,
                guidance TEXT NOT NULL,
                applicability TEXT NOT NULL,
                before_quote TEXT NOT NULL DEFAULT '',
                after_quote TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_author_editing_preferences_project
            ON author_editing_preferences(
                project_id, status, updated_at DESC
            )
            """,
        ),
    )


def _story_blueprint_v14(
    connection: sqlite3.Connection, applied_at: str
) -> None:
    del applied_at
    _execute_statements(
        connection,
        (
            """
            CREATE TABLE IF NOT EXISTS novel_story_blueprint_versions (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                revision INTEGER NOT NULL,
                version_status TEXT NOT NULL,
                central_question TEXT NOT NULL DEFAULT '',
                protagonist_goal TEXT NOT NULL DEFAULT '',
                core_conflict TEXT NOT NULL DEFAULT '',
                stakes TEXT NOT NULL DEFAULT '',
                opening_state TEXT NOT NULL DEFAULT '',
                ending_state TEXT NOT NULL DEFAULT '',
                major_turns_json TEXT NOT NULL DEFAULT '[]',
                must_payoffs_json TEXT NOT NULL DEFAULT '[]',
                forbidden_shortcuts_json TEXT NOT NULL DEFAULT '[]',
                author_notes TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT 'manual',
                created_at TEXT NOT NULL,
                confirmed_at TEXT,
                UNIQUE(project_id, revision)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS novel_story_blueprint_heads (
                project_id TEXT PRIMARY KEY
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                current_version_id TEXT NOT NULL
                    REFERENCES novel_story_blueprint_versions(id)
                    ON DELETE CASCADE,
                confirmed_version_id TEXT
                    REFERENCES novel_story_blueprint_versions(id)
                    ON DELETE SET NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_story_blueprint_versions_project
            ON novel_story_blueprint_versions(
                project_id, revision DESC
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS novel_plot_arcs (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                position INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(project_id, position)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS novel_plot_arc_versions (
                id TEXT PRIMARY KEY,
                arc_id TEXT NOT NULL
                    REFERENCES novel_plot_arcs(id) ON DELETE CASCADE,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                revision INTEGER NOT NULL,
                version_status TEXT NOT NULL,
                arc_type TEXT NOT NULL,
                title TEXT NOT NULL,
                dramatic_question TEXT NOT NULL DEFAULT '',
                promise TEXT NOT NULL DEFAULT '',
                start_state TEXT NOT NULL DEFAULT '',
                target_payoff TEXT NOT NULL DEFAULT '',
                involved_characters_json TEXT NOT NULL DEFAULT '[]',
                planned_turns_json TEXT NOT NULL DEFAULT '[]',
                lifecycle_status TEXT NOT NULL DEFAULT 'planned',
                priority INTEGER NOT NULL DEFAULT 3,
                author_notes TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT 'manual',
                created_at TEXT NOT NULL,
                confirmed_at TEXT,
                UNIQUE(arc_id, revision)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_plot_arcs_project_position
            ON novel_plot_arcs(project_id, position)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_plot_arc_versions_project
            ON novel_plot_arc_versions(
                project_id, lifecycle_status, version_status, created_at DESC
            )
            """,
        ),
    )
    _add_column(
        connection,
        "novel_plot_arcs",
        "current_version_id",
        (
            "TEXT REFERENCES novel_plot_arc_versions(id) "
            "ON DELETE SET NULL"
        ),
    )
    _add_column(
        connection,
        "novel_plot_arcs",
        "confirmed_version_id",
        (
            "TEXT REFERENCES novel_plot_arc_versions(id) "
            "ON DELETE SET NULL"
        ),
    )


def _story_planner_suggestions_v15(
    connection: sqlite3.Connection, applied_at: str
) -> None:
    del applied_at
    _execute_statements(
        connection,
        (
            """
            CREATE TABLE IF NOT EXISTS story_plan_suggestions (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                user_id INTEGER NOT NULL
                    REFERENCES users(id) ON DELETE CASCADE,
                planning_mode TEXT NOT NULL,
                instruction TEXT NOT NULL DEFAULT '',
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                credential_source TEXT NOT NULL DEFAULT 'default',
                status TEXT NOT NULL DEFAULT 'queued',
                baseline_fingerprint TEXT NOT NULL,
                context_snapshot_json TEXT NOT NULL,
                result_json TEXT,
                raw_response TEXT,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                claim_token TEXT,
                lease_expires_at TEXT
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_story_plan_suggestions_project
            ON story_plan_suggestions(
                project_id, created_at DESC
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_story_plan_suggestions_queue
            ON story_plan_suggestions(status, created_at)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_story_plan_suggestions_user_status
            ON story_plan_suggestions(user_id, status, created_at)
            """,
            """
            CREATE TABLE IF NOT EXISTS
                story_plan_suggestion_applications (
                    id TEXT PRIMARY KEY,
                    suggestion_id TEXT NOT NULL
                        REFERENCES story_plan_suggestions(id)
                        ON DELETE CASCADE,
                    project_id TEXT NOT NULL
                        REFERENCES novel_projects(id) ON DELETE CASCADE,
                    option_index INTEGER NOT NULL,
                    apply_blueprint INTEGER NOT NULL DEFAULT 0,
                    selected_arc_indices_json TEXT NOT NULL DEFAULT '[]',
                    created_blueprint_version_id TEXT
                        REFERENCES novel_story_blueprint_versions(id)
                        ON DELETE SET NULL,
                    applied_arcs_json TEXT NOT NULL DEFAULT '[]',
                    baseline_changed INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_story_plan_applications_suggestion
            ON story_plan_suggestion_applications(
                suggestion_id, created_at
            )
            """,
        ),
    )


def _story_structure_planner_v16(
    connection: sqlite3.Connection, applied_at: str
) -> None:
    del applied_at
    _add_column(
        connection,
        "novel_chapters",
        "skeleton_role",
        "TEXT NOT NULL DEFAULT ''",
    )
    _add_column(
        connection,
        "novel_chapters",
        "skeleton_arc_titles_json",
        "TEXT NOT NULL DEFAULT '[]'",
    )
    _add_column(
        connection,
        "novel_chapters",
        "skeleton_ending_hook",
        "TEXT NOT NULL DEFAULT ''",
    )
    _add_column(
        connection,
        "novel_chapters",
        "skeleton_application_id",
        "TEXT",
    )
    _execute_statements(
        connection,
        (
            """
            CREATE TABLE IF NOT EXISTS story_structure_suggestions (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                user_id INTEGER NOT NULL
                    REFERENCES users(id) ON DELETE CASCADE,
                instruction TEXT NOT NULL DEFAULT '',
                chapter_count INTEGER NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                credential_source TEXT NOT NULL DEFAULT 'default',
                status TEXT NOT NULL DEFAULT 'queued',
                baseline_fingerprint TEXT NOT NULL,
                context_snapshot_json TEXT NOT NULL,
                result_json TEXT,
                raw_response TEXT,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                claim_token TEXT,
                lease_expires_at TEXT
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_story_structure_project
            ON story_structure_suggestions(project_id, created_at DESC)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_story_structure_queue
            ON story_structure_suggestions(status, created_at)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_story_structure_user_status
            ON story_structure_suggestions(user_id, status, created_at)
            """,
            """
            CREATE TABLE IF NOT EXISTS story_structure_applications (
                id TEXT PRIMARY KEY,
                suggestion_id TEXT NOT NULL
                    REFERENCES story_structure_suggestions(id)
                    ON DELETE CASCADE,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                option_index INTEGER NOT NULL,
                preview_fingerprint TEXT NOT NULL,
                before_state_json TEXT NOT NULL,
                after_state_json TEXT NOT NULL,
                after_fingerprint TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'applied',
                baseline_changed INTEGER NOT NULL DEFAULT 0,
                recovery_path TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                reverted_at TEXT
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_story_structure_app_suggestion
            ON story_structure_applications(
                suggestion_id, created_at DESC
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_story_structure_app_project
            ON story_structure_applications(
                project_id, status, created_at DESC
            )
            """,
        ),
    )


def _chapter_causal_links_v17(
    connection: sqlite3.Connection, applied_at: str
) -> None:
    del applied_at
    _execute_statements(
        connection,
        (
            """
            CREATE TABLE IF NOT EXISTS novel_chapter_causal_links (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                source_chapter_id TEXT NOT NULL
                    REFERENCES novel_chapters(id) ON DELETE CASCADE,
                target_chapter_id TEXT NOT NULL
                    REFERENCES novel_chapters(id) ON DELETE CASCADE,
                relation_type TEXT NOT NULL
                    CHECK(relation_type IN (
                        'causes', 'enables', 'complicates', 'pays_off'
                    )),
                cause_text TEXT NOT NULL,
                effect_text TEXT NOT NULL,
                author_note TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'active'
                    CHECK(status IN ('active', 'archived')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                archived_at TEXT,
                CHECK(source_chapter_id <> target_chapter_id)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_chapter_causal_links_project
            ON novel_chapter_causal_links(
                project_id, status, target_chapter_id
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_chapter_causal_links_source
            ON novel_chapter_causal_links(source_chapter_id, status)
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS
                idx_chapter_causal_links_active_exact
            ON novel_chapter_causal_links(
                source_chapter_id, target_chapter_id, relation_type,
                cause_text, effect_text
            )
            WHERE status='active'
            """,
        ),
    )


def _causal_link_suggestions_v18(
    connection: sqlite3.Connection, applied_at: str
) -> None:
    del applied_at
    _execute_statements(
        connection,
        (
            """
            CREATE TABLE IF NOT EXISTS novel_causal_link_suggestions (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                user_id INTEGER NOT NULL
                    REFERENCES users(id) ON DELETE CASCADE,
                instruction TEXT NOT NULL DEFAULT '',
                chapter_limit INTEGER NOT NULL
                    CHECK(chapter_limit BETWEEN 2 AND 80),
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                credential_source TEXT NOT NULL DEFAULT 'default'
                    CHECK(credential_source IN ('default', 'personal')),
                status TEXT NOT NULL DEFAULT 'queued'
                    CHECK(status IN (
                        'queued', 'running', 'completed', 'failed'
                    )),
                baseline_fingerprint TEXT NOT NULL,
                context_snapshot_json TEXT NOT NULL,
                result_json TEXT,
                raw_response TEXT,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                claim_token TEXT,
                lease_expires_at TEXT
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_causal_suggestions_project
            ON novel_causal_link_suggestions(
                project_id, created_at DESC
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_causal_suggestions_queue
            ON novel_causal_link_suggestions(status, created_at)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_causal_suggestions_user_status
            ON novel_causal_link_suggestions(
                user_id, status, created_at
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS
                novel_causal_link_suggestion_reviews (
                    id TEXT PRIMARY KEY,
                    suggestion_id TEXT NOT NULL
                        REFERENCES novel_causal_link_suggestions(id)
                        ON DELETE CASCADE,
                    project_id TEXT NOT NULL
                        REFERENCES novel_projects(id) ON DELETE CASCADE,
                    proposal_index INTEGER NOT NULL
                        CHECK(proposal_index >= 0),
                    decision TEXT NOT NULL
                        CHECK(decision IN ('accepted', 'dismissed')),
                    source_chapter_id TEXT NOT NULL,
                    target_chapter_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL
                        CHECK(relation_type IN (
                            'causes', 'enables', 'complicates', 'pays_off'
                        )),
                    cause_text TEXT NOT NULL,
                    effect_text TEXT NOT NULL,
                    author_note TEXT NOT NULL DEFAULT '',
                    causal_link_id TEXT
                        REFERENCES novel_chapter_causal_links(id)
                        ON DELETE SET NULL,
                    decided_at TEXT NOT NULL,
                    UNIQUE(suggestion_id, proposal_index)
                )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_causal_reviews_project
            ON novel_causal_link_suggestion_reviews(
                project_id, decided_at DESC
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_causal_reviews_link
            ON novel_causal_link_suggestion_reviews(causal_link_id)
            """,
        ),
    )


def _causal_branch_simulations_v19(
    connection: sqlite3.Connection, applied_at: str
) -> None:
    del applied_at
    _execute_statements(
        connection,
        (
            """
            CREATE TABLE IF NOT EXISTS novel_causal_branch_simulations (
                id TEXT PRIMARY KEY,
                source_suggestion_id TEXT NOT NULL
                    REFERENCES novel_causal_link_suggestions(id)
                    ON DELETE CASCADE,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                user_id INTEGER NOT NULL
                    REFERENCES users(id) ON DELETE CASCADE,
                proposal_index INTEGER NOT NULL
                    CHECK(proposal_index >= 0),
                horizon_chapter_count INTEGER NOT NULL
                    CHECK(horizon_chapter_count BETWEEN 10 AND 30),
                instruction TEXT NOT NULL DEFAULT '',
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                credential_source TEXT NOT NULL DEFAULT 'default'
                    CHECK(credential_source IN ('default', 'personal')),
                status TEXT NOT NULL DEFAULT 'queued'
                    CHECK(status IN (
                        'queued', 'running', 'completed', 'failed'
                    )),
                source_proposal_signature TEXT NOT NULL,
                baseline_fingerprint TEXT NOT NULL,
                context_snapshot_json TEXT NOT NULL,
                result_json TEXT,
                raw_response TEXT,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                claim_token TEXT,
                lease_expires_at TEXT
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_causal_branch_project
            ON novel_causal_branch_simulations(
                project_id, created_at DESC
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_causal_branch_source
            ON novel_causal_branch_simulations(
                source_suggestion_id, proposal_index, created_at DESC
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_causal_branch_queue
            ON novel_causal_branch_simulations(status, created_at)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_causal_branch_user_status
            ON novel_causal_branch_simulations(
                user_id, status, created_at
            )
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS
                idx_causal_branch_one_active_user
            ON novel_causal_branch_simulations(user_id)
            WHERE status IN ('queued', 'running')
            """,
        ),
    )


def _causal_branch_adoptions_v20(
    connection: sqlite3.Connection, applied_at: str
) -> None:
    del applied_at
    _execute_statements(
        connection,
        (
            """
            CREATE TABLE IF NOT EXISTS novel_causal_branch_adoptions (
                id TEXT PRIMARY KEY,
                simulation_id TEXT NOT NULL
                    REFERENCES novel_causal_branch_simulations(id)
                    ON DELETE CASCADE,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                user_id INTEGER NOT NULL
                    REFERENCES users(id) ON DELETE CASCADE,
                source_causal_link_id TEXT NOT NULL
                    REFERENCES novel_chapter_causal_links(id)
                    ON DELETE CASCADE,
                branch_key TEXT NOT NULL
                    CHECK(branch_key IN (
                        'minimal_change',
                        'distributed_consequences',
                        'stress_test'
                    )),
                status TEXT NOT NULL DEFAULT 'draft'
                    CHECK(status IN (
                        'draft', 'applied', 'abandoned', 'reverted'
                    )),
                baseline_fingerprint TEXT NOT NULL,
                baseline_state_json TEXT NOT NULL,
                branch_snapshot_json TEXT NOT NULL,
                before_state_json TEXT,
                after_state_json TEXT,
                after_fingerprint TEXT,
                accepted_item_count INTEGER NOT NULL DEFAULT 0
                    CHECK(accepted_item_count >= 0),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                applied_at TEXT,
                abandoned_at TEXT,
                reverted_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS
                novel_causal_branch_adoption_items (
                id TEXT PRIMARY KEY,
                adoption_id TEXT NOT NULL
                    REFERENCES novel_causal_branch_adoptions(id)
                    ON DELETE CASCADE,
                item_key TEXT NOT NULL,
                chapter_id TEXT NOT NULL
                    REFERENCES novel_chapters(id) ON DELETE CASCADE,
                chapter_position INTEGER NOT NULL
                    CHECK(chapter_position >= 1),
                chapter_title TEXT NOT NULL,
                impact_type TEXT NOT NULL
                    CHECK(impact_type IN (
                        'setup', 'escalation', 'information_transfer',
                        'choice', 'reversal', 'payoff', 'repair'
                    )),
                proposal_json TEXT NOT NULL,
                edited_patch_json TEXT NOT NULL,
                decision TEXT NOT NULL DEFAULT 'pending'
                    CHECK(decision IN (
                        'pending', 'accepted', 'rejected'
                    )),
                author_note TEXT NOT NULL DEFAULT '',
                decided_at TEXT,
                applied_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(adoption_id, item_key),
                UNIQUE(adoption_id, chapter_id)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_causal_branch_adoption_project
            ON novel_causal_branch_adoptions(
                project_id, created_at DESC
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_causal_branch_adoption_simulation
            ON novel_causal_branch_adoptions(
                simulation_id, branch_key, created_at DESC
            )
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS
                idx_causal_branch_one_live_adoption
            ON novel_causal_branch_adoptions(simulation_id, branch_key)
            WHERE status IN ('draft', 'applied')
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_causal_branch_adoption_items
            ON novel_causal_branch_adoption_items(
                adoption_id, chapter_position
            )
            """,
        ),
    )


def _scene_requirement_coverage_v21(
    connection: sqlite3.Connection, applied_at: str
) -> None:
    _add_column(
        connection,
        "novel_scene_beats",
        "requirement_refs_json",
        "TEXT NOT NULL DEFAULT '[]'",
    )
    connection.execute(
        """
        UPDATE novel_chapter_plans
        SET status='draft', confirmed_at=NULL, updated_at=?
        WHERE status='confirmed'
          AND NOT EXISTS (
              SELECT 1
              FROM novel_scene_beats scene
              WHERE scene.plan_id=novel_chapter_plans.id
                AND scene.beat_status='active'
                AND json_array_length(scene.requirement_refs_json)>0
          )
        """,
        (applied_at,),
    )


def _editing_preference_aggregation_v22(
    connection: sqlite3.Connection, applied_at: str
) -> None:
    del applied_at
    _execute_statements(
        connection,
        (
            """
            CREATE TABLE IF NOT EXISTS author_editing_preference_aggregates (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                category TEXT NOT NULL,
                guidance TEXT NOT NULL,
                applicability TEXT NOT NULL,
                author_note TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'active'
                    CHECK(status IN ('active', 'archived')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                confirmed_at TEXT NOT NULL,
                archived_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS
                author_editing_preference_aggregate_evidence (
                aggregate_id TEXT NOT NULL
                    REFERENCES author_editing_preference_aggregates(id)
                    ON DELETE CASCADE,
                preference_id TEXT NOT NULL
                    REFERENCES author_editing_preferences(id)
                    ON DELETE CASCADE,
                role TEXT NOT NULL
                    CHECK(role IN ('support', 'conflict')),
                created_at TEXT NOT NULL,
                PRIMARY KEY(aggregate_id, preference_id)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_edit_pref_aggregates_project
            ON author_editing_preference_aggregates(
                project_id, status, updated_at DESC
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_edit_pref_aggregate_evidence_source
            ON author_editing_preference_aggregate_evidence(preference_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_style_audits_project_time
            ON chapter_style_audits(project_id, created_at, version_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_style_issues_audit_type
            ON chapter_style_issues(audit_id, issue_type)
            """,
        ),
    )


def _assistant_chat_v23(
    connection: sqlite3.Connection, applied_at: str
) -> None:
    del applied_at
    _execute_statements(
        connection,
        (
            """
            CREATE TABLE IF NOT EXISTS assistant_conversations (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL
                    REFERENCES users(id) ON DELETE CASCADE,
                scope_type TEXT NOT NULL
                    CHECK(scope_type IN (
                        'project', 'chapter',
                        'document', 'reference_chapter'
                    )),
                project_id TEXT
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                document_id TEXT
                    REFERENCES documents(id) ON DELETE CASCADE,
                novel_chapter_id TEXT
                    REFERENCES novel_chapters(id) ON DELETE CASCADE,
                reference_chapter_id TEXT
                    REFERENCES chapters(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                CHECK(
                    (
                        scope_type='project'
                        AND project_id IS NOT NULL
                        AND document_id IS NULL
                        AND novel_chapter_id IS NULL
                        AND reference_chapter_id IS NULL
                    )
                    OR (
                        scope_type='chapter'
                        AND project_id IS NOT NULL
                        AND novel_chapter_id IS NOT NULL
                        AND document_id IS NULL
                        AND reference_chapter_id IS NULL
                    )
                    OR (
                        scope_type='document'
                        AND document_id IS NOT NULL
                        AND project_id IS NULL
                        AND novel_chapter_id IS NULL
                        AND reference_chapter_id IS NULL
                    )
                    OR (
                        scope_type='reference_chapter'
                        AND document_id IS NOT NULL
                        AND reference_chapter_id IS NOT NULL
                        AND project_id IS NULL
                        AND novel_chapter_id IS NULL
                    )
                )
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS assistant_messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL
                    REFERENCES assistant_conversations(id)
                    ON DELETE CASCADE,
                role TEXT NOT NULL
                    CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL DEFAULT '',
                parent_user_message_id TEXT
                    REFERENCES assistant_messages(id) ON DELETE CASCADE,
                status TEXT NOT NULL
                    CHECK(status IN (
                        'completed', 'queued', 'running', 'failed'
                    )),
                provider TEXT NOT NULL DEFAULT '',
                model TEXT NOT NULL DEFAULT '',
                credential_source TEXT NOT NULL DEFAULT 'default'
                    CHECK(credential_source IN ('default', 'personal')),
                context_snapshot_json TEXT,
                response_json TEXT,
                raw_response TEXT,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                claim_token TEXT,
                lease_expires_at TEXT,
                applied_version_id TEXT
                    REFERENCES novel_chapter_versions(id)
                    ON DELETE SET NULL,
                CHECK(
                    (role='user' AND status='completed'
                        AND parent_user_message_id IS NULL)
                    OR (role='assistant'
                        AND parent_user_message_id IS NOT NULL)
                )
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS assistant_message_quotes (
                id TEXT PRIMARY KEY,
                message_id TEXT NOT NULL UNIQUE
                    REFERENCES assistant_messages(id) ON DELETE CASCADE,
                source_type TEXT NOT NULL
                    CHECK(source_type IN (
                        'novel_version', 'reference_chapter'
                    )),
                project_id TEXT
                    REFERENCES novel_projects(id) ON DELETE CASCADE,
                document_id TEXT
                    REFERENCES documents(id) ON DELETE CASCADE,
                novel_chapter_id TEXT
                    REFERENCES novel_chapters(id) ON DELETE CASCADE,
                version_id TEXT
                    REFERENCES novel_chapter_versions(id) ON DELETE CASCADE,
                reference_chapter_id TEXT
                    REFERENCES chapters(id) ON DELETE CASCADE,
                start_offset INTEGER NOT NULL CHECK(start_offset>=0),
                end_offset INTEGER NOT NULL CHECK(end_offset>start_offset),
                quote_text TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                source_label TEXT NOT NULL,
                created_at TEXT NOT NULL,
                CHECK(
                    (
                        source_type='novel_version'
                        AND project_id IS NOT NULL
                        AND novel_chapter_id IS NOT NULL
                        AND version_id IS NOT NULL
                        AND document_id IS NULL
                        AND reference_chapter_id IS NULL
                    )
                    OR (
                        source_type='reference_chapter'
                        AND document_id IS NOT NULL
                        AND reference_chapter_id IS NOT NULL
                        AND project_id IS NULL
                        AND novel_chapter_id IS NULL
                        AND version_id IS NULL
                    )
                )
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_assistant_conversations_project
            ON assistant_conversations(
                user_id, project_id, updated_at DESC
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_assistant_conversations_document
            ON assistant_conversations(
                user_id, document_id, updated_at DESC
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_assistant_messages_conversation
            ON assistant_messages(conversation_id, created_at)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_assistant_messages_queue
            ON assistant_messages(status, created_at)
            WHERE role='assistant' AND status IN ('queued', 'running')
            """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS
                idx_assistant_reply_per_user_message
            ON assistant_messages(parent_user_message_id)
            WHERE role='assistant'
            """,
        ),
    )


def _workbench_prompts_v24(
    connection: sqlite3.Connection, applied_at: str
) -> None:
    del applied_at
    _add_column(
        connection,
        "api_credentials",
        "system_prompt",
        "TEXT NOT NULL DEFAULT ''",
    )
    _add_column(
        connection,
        "novel_projects",
        "ai_instructions",
        "TEXT NOT NULL DEFAULT ''",
    )


def _assistant_agent_tools_v25(
    connection: sqlite3.Connection, applied_at: str
) -> None:
    del applied_at
    _execute_statements(
        connection,
        (
            """
            CREATE TABLE IF NOT EXISTS assistant_tool_calls (
                id TEXT PRIMARY KEY,
                assistant_message_id TEXT NOT NULL
                    REFERENCES assistant_messages(id) ON DELETE CASCADE,
                sequence INTEGER NOT NULL CHECK(sequence>=1),
                agent_role TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                tool_label TEXT NOT NULL,
                capability TEXT NOT NULL DEFAULT '',
                read_only INTEGER NOT NULL DEFAULT 1
                    CHECK(read_only IN (0, 1)),
                arguments_json TEXT NOT NULL DEFAULT '{}',
                result_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL
                    CHECK(status IN (
                        'running', 'completed', 'failed', 'denied'
                    )),
                error TEXT,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                UNIQUE(assistant_message_id, sequence)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_assistant_tool_calls_message
            ON assistant_tool_calls(
                assistant_message_id, sequence
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_assistant_tool_calls_tool
            ON assistant_tool_calls(
                tool_name, status, started_at
            )
            """,
        ),
    )


def _assistant_agent_steps_v26(
    connection: sqlite3.Connection, applied_at: str
) -> None:
    del applied_at
    _execute_statements(
        connection,
        (
            """
            CREATE TABLE IF NOT EXISTS assistant_agent_steps (
                id TEXT PRIMARY KEY,
                assistant_message_id TEXT NOT NULL
                    REFERENCES assistant_messages(id) ON DELETE CASCADE,
                sequence INTEGER NOT NULL CHECK(sequence>=1),
                agent_role TEXT NOT NULL,
                action TEXT NOT NULL
                    CHECK(action IN ('call_tool', 'finish', 'fallback')),
                tool_name TEXT NOT NULL DEFAULT '',
                tool_label TEXT NOT NULL DEFAULT '',
                available_tools_json TEXT NOT NULL DEFAULT '[]',
                decision_json TEXT NOT NULL DEFAULT '{}',
                outcome_status TEXT NOT NULL
                    CHECK(outcome_status IN (
                        'completed', 'denied', 'failed', 'fallback'
                    )),
                error TEXT,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                latency_ms INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                UNIQUE(assistant_message_id, sequence)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_assistant_agent_steps_message
            ON assistant_agent_steps(
                assistant_message_id, sequence
            )
            """,
        ),
    )


def _model_base_url_v27(
    connection: sqlite3.Connection, applied_at: str
) -> None:
    del applied_at
    _add_column(
        connection,
        "api_credentials",
        "base_url",
        "TEXT NOT NULL DEFAULT ''",
    )


def _multi_provider_credentials_v28(
    connection: sqlite3.Connection, applied_at: str
) -> None:
    credential_columns = _columns(connection, "api_credentials")
    primary_key = [
        str(row["name"])
        for row in sorted(
            connection.execute(
                "PRAGMA table_info(api_credentials)"
            ).fetchall(),
            key=lambda row: int(row["pk"]),
        )
        if int(row["pk"]) > 0
    ]
    modern_schema = (
        "is_default" in credential_columns
        and primary_key == ["user_id", "provider"]
    )
    if not modern_schema:
        connection.execute(
            "ALTER TABLE api_credentials RENAME TO api_credentials_v27"
        )
        connection.execute(
            """
            CREATE TABLE api_credentials (
                user_id INTEGER NOT NULL
                    REFERENCES users(id) ON DELETE CASCADE,
                provider TEXT NOT NULL,
                base_url TEXT NOT NULL DEFAULT '',
                encrypted_key TEXT NOT NULL,
                key_hint TEXT NOT NULL,
                model TEXT NOT NULL,
                thinking INTEGER NOT NULL DEFAULT 0,
                reasoning_effort TEXT NOT NULL DEFAULT 'high',
                system_prompt TEXT NOT NULL DEFAULT '',
                is_default INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(user_id, provider)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO api_credentials(
                user_id, provider, base_url, encrypted_key, key_hint, model,
                thinking, reasoning_effort, system_prompt, is_default,
                created_at, updated_at
            )
            SELECT user_id, provider, base_url, encrypted_key, key_hint, model,
                   thinking, reasoning_effort, system_prompt, 1,
                   created_at, updated_at
            FROM api_credentials_v27
            """
        )
        connection.execute("DROP TABLE api_credentials_v27")

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS api_models (
            user_id INTEGER NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            position INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            PRIMARY KEY(user_id, provider, model),
            FOREIGN KEY(user_id, provider)
                REFERENCES api_credentials(user_id, provider)
                ON DELETE CASCADE
        )
        """
    )
    connection.execute(
        """
        INSERT OR IGNORE INTO api_models(
            user_id, provider, model, position, created_at
        )
        SELECT user_id, provider, model, 0, ?
        FROM api_credentials
        """,
        (applied_at,),
    )
    connection.execute(
        "DROP INDEX IF EXISTS idx_api_credentials_one_default"
    )
    connection.execute(
        """
        UPDATE api_credentials AS credential
        SET is_default = CASE
            WHEN credential.provider = (
                SELECT candidate.provider
                FROM api_credentials AS candidate
                WHERE candidate.user_id=credential.user_id
                ORDER BY candidate.is_default DESC,
                         candidate.updated_at DESC,
                         candidate.provider
                LIMIT 1
            ) THEN 1
            ELSE 0
        END
        """
    )
    connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS
            idx_api_credentials_one_default
        ON api_credentials(user_id)
        WHERE is_default=1
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_api_models_provider_position
        ON api_models(user_id, provider, position)
        """
    )


MIGRATIONS = (
    Migration(1, "core_memory_v1", _core_memory_v1),
    Migration(2, "planning_v2", _planning_v2),
    Migration(3, "quality_gate_v3", _quality_gate_v3),
    Migration(4, "style_editor_v4", _style_editor_v4),
    Migration(5, "reader_decisions_v5", _reader_decisions_v5),
    Migration(6, "technique_library_v6", _technique_library_v6),
    Migration(7, "scene_workbench_v7", _scene_workbench_v7),
    Migration(8, "memory_search_v8", _memory_search_v8),
    Migration(9, "continuity_replay_v9", _continuity_replay_v9),
    Migration(10, "continuity_lifecycle_v10", _continuity_lifecycle_v10),
    Migration(
        11,
        "memory_identity_and_causality_v11",
        _memory_identity_and_causality_v11,
    ),
    Migration(
        12,
        "voice_profile_learning_v12",
        _voice_profile_learning_v12,
    ),
    Migration(
        13,
        "manual_edit_preference_learning_v13",
        _manual_edit_preference_learning_v13,
    ),
    Migration(
        14,
        "story_blueprint_v14",
        _story_blueprint_v14,
    ),
    Migration(
        15,
        "story_planner_suggestions_v15",
        _story_planner_suggestions_v15,
    ),
    Migration(
        16,
        "story_structure_planner_v16",
        _story_structure_planner_v16,
    ),
    Migration(
        17,
        "chapter_causal_links_v17",
        _chapter_causal_links_v17,
    ),
    Migration(
        18,
        "causal_link_suggestions_v18",
        _causal_link_suggestions_v18,
    ),
    Migration(
        19,
        "causal_branch_simulations_v19",
        _causal_branch_simulations_v19,
    ),
    Migration(
        20,
        "causal_branch_adoptions_v20",
        _causal_branch_adoptions_v20,
    ),
    Migration(
        21,
        "scene_requirement_coverage_v21",
        _scene_requirement_coverage_v21,
    ),
    Migration(
        22,
        "editing_preference_aggregation_v22",
        _editing_preference_aggregation_v22,
    ),
    Migration(
        23,
        "assistant_chat_v23",
        _assistant_chat_v23,
    ),
    Migration(
        24,
        "workbench_prompts_v24",
        _workbench_prompts_v24,
    ),
    Migration(
        25,
        "assistant_agent_tools_v25",
        _assistant_agent_tools_v25,
    ),
    Migration(
        26,
        "assistant_agent_steps_v26",
        _assistant_agent_steps_v26,
    ),
    Migration(
        27,
        "model_base_url_v27",
        _model_base_url_v27,
    ),
    Migration(
        28,
        "multi_provider_credentials_v28",
        _multi_provider_credentials_v28,
    ),
)


def apply_migrations(
    connection: sqlite3.Connection, applied_at: str
) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
        """
    )
    connection.commit()
    applied = {
        int(row["version"])
        for row in connection.execute(
            "SELECT version FROM schema_migrations"
        ).fetchall()
    }
    for migration in MIGRATIONS:
        if migration.version in applied:
            continue
        try:
            connection.execute("BEGIN IMMEDIATE")
            migration.apply(connection, applied_at)
            connection.execute(
                """
                INSERT INTO schema_migrations(version, name, applied_at)
                VALUES (?, ?, ?)
                """,
                (migration.version, migration.name, applied_at),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
