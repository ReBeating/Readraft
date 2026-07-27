from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .continuity import replay_canonical_state
from .db import Database, utc_now
from .memory_identity import ensure_memory_identity
from .memory_schema import StoryDelta
from .memory_search import (
    delete_chapter_search_documents,
    replace_chapter_search_documents,
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class MemoryService:
    """Projects extracted chapter changes into canonical story memory."""

    def __init__(self, database: Database):
        self.database = database

    def create_proposal(
        self,
        *,
        user_id: int,
        project_id: str,
        chapter_id: str,
        version_id: str,
        payload: StoryDelta | Mapping[str, Any],
    ) -> str:
        delta = (
            payload
            if isinstance(payload, StoryDelta)
            else StoryDelta.model_validate(payload)
        )
        serialized = delta.model_dump_json()
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            target = self._owned_canonical_version(
                connection,
                user_id=user_id,
                project_id=project_id,
                chapter_id=chapter_id,
                version_id=version_id,
            )
            if not target:
                connection.rollback()
                raise ValueError(
                    "只能为当前正史版本提取故事记忆，请先确认正文版本"
                )
            existing = connection.execute(
                """
                SELECT id, status FROM story_deltas
                WHERE version_id=?
                    AND status IN (
                        'proposed', 'author_edited', 'accepted', 'projected'
                    )
                LIMIT 1
                """,
                (version_id,),
            ).fetchone()
            if existing:
                delta_id = str(existing["id"])
                if str(existing["status"]) in {"proposed", "author_edited"}:
                    connection.execute(
                        """
                        UPDATE story_deltas
                        SET payload_json=?, status='proposed', updated_at=?
                        WHERE id=?
                        """,
                        (serialized, now, delta_id),
                    )
                connection.commit()
                return delta_id
            delta_id = uuid.uuid4().hex
            connection.execute(
                """
                INSERT INTO story_deltas(
                    id, project_id, chapter_id, version_id, branch_id,
                    status, payload_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'proposed', ?, ?, ?)
                """,
                (
                    delta_id,
                    project_id,
                    chapter_id,
                    version_id,
                    str(target["branch_id"]),
                    serialized,
                    now,
                    now,
                ),
            )
            connection.commit()
        return delta_id

    def get_delta(
        self, *, user_id: int, delta_id: str
    ) -> Optional[Dict[str, Any]]:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT d.*, ch.title AS chapter_title,
                       ch.position AS chapter_position,
                       ch.canonical_version_id,
                       p.title AS project_title
                FROM story_deltas d
                JOIN novel_chapters ch ON ch.id=d.chapter_id
                JOIN novel_projects p ON p.id=d.project_id
                WHERE d.id=? AND p.user_id=?
                """,
                (delta_id, user_id),
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["payload"] = StoryDelta.model_validate_json(
            str(result["payload_json"])
        ).model_dump(mode="json")
        return result

    def list_chapter_deltas(
        self,
        *,
        user_id: int,
        project_id: str,
        chapter_id: str,
    ) -> List[Dict[str, Any]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT d.*
                FROM story_deltas d
                JOIN novel_projects p ON p.id=d.project_id
                WHERE d.project_id=? AND d.chapter_id=? AND p.user_id=?
                ORDER BY d.created_at DESC
                """,
                (project_id, chapter_id, user_id),
            ).fetchall()
        results: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["payload"] = StoryDelta.model_validate_json(
                str(item["payload_json"])
            ).model_dump(mode="json")
            results.append(item)
        return results

    def update_proposal(
        self,
        *,
        user_id: int,
        delta_id: str,
        payload: StoryDelta | Mapping[str, Any],
    ) -> bool:
        delta = (
            payload
            if isinstance(payload, StoryDelta)
            else StoryDelta.model_validate(payload)
        )
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE story_deltas
                SET payload_json=?, status='author_edited', updated_at=?
                WHERE id=? AND status IN ('proposed', 'author_edited')
                    AND EXISTS(
                        SELECT 1 FROM novel_projects p
                        WHERE p.id=story_deltas.project_id AND p.user_id=?
                    )
                """,
                (delta.model_dump_json(), utc_now(), delta_id, user_id),
            )
            connection.commit()
        return cursor.rowcount == 1

    def reject_delta(self, *, user_id: int, delta_id: str) -> bool:
        now = utc_now()
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE story_deltas
                SET status='rejected', reviewed_at=?, updated_at=?
                WHERE id=? AND status IN ('proposed', 'author_edited')
                    AND EXISTS(
                        SELECT 1 FROM novel_projects p
                        WHERE p.id=story_deltas.project_id AND p.user_id=?
                    )
                """,
                (now, now, delta_id, user_id),
            )
            connection.commit()
        return cursor.rowcount == 1

    def accept_delta(
        self, *, user_id: int, delta_id: str
    ) -> Optional[Dict[str, Any]]:
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT d.*, ch.canonical_version_id
                FROM story_deltas d
                JOIN novel_chapters ch ON ch.id=d.chapter_id
                JOIN novel_projects p ON p.id=d.project_id
                WHERE d.id=? AND p.user_id=?
                """,
                (delta_id, user_id),
            ).fetchone()
            if not row:
                connection.rollback()
                return None
            if str(row["status"]) == "projected":
                connection.commit()
                return {
                    "delta_id": delta_id,
                    "projected": True,
                    "already_projected": True,
                }
            if str(row["status"]) not in {"proposed", "author_edited"}:
                connection.rollback()
                raise ValueError("这份故事记忆提案已不能确认")
            if str(row["canonical_version_id"] or "") != str(
                row["version_id"]
            ):
                connection.rollback()
                raise ValueError("正文正史已经变化，请重新提取故事记忆")

            delta = StoryDelta.model_validate_json(str(row["payload_json"]))
            self._retract_previous_chapter_memory(
                connection,
                chapter_id=str(row["chapter_id"]),
                delta_id=delta_id,
            )
            counts = self._project(
                connection,
                delta_id=delta_id,
                project_id=str(row["project_id"]),
                chapter_id=str(row["chapter_id"]),
                version_id=str(row["version_id"]),
                branch_id=str(row["branch_id"]),
                delta=delta,
                created_at=now,
            )
            connection.execute(
                """
                UPDATE story_deltas
                SET status='projected', reviewed_at=?, updated_at=?
                WHERE id=?
                """,
                (now, now, delta_id),
            )
            connection.execute(
                "UPDATE novel_projects SET updated_at=? WHERE id=?",
                (now, row["project_id"]),
            )
            replay_canonical_state(
                connection,
                project_id=str(row["project_id"]),
                branch_id=str(row["branch_id"]),
                trigger_type="story_delta_projected",
                trigger_chapter_id=str(row["chapter_id"]),
                created_at=now,
            )
            connection.commit()
        return {
            "delta_id": delta_id,
            "projected": True,
            "already_projected": False,
            **counts,
        }

    def get_chapter_memory(
        self,
        *,
        user_id: int,
        project_id: str,
        chapter_id: str,
    ) -> Optional[Dict[str, Any]]:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT m.*
                FROM chapter_memory m
                JOIN novel_projects p ON p.id=m.project_id
                WHERE m.project_id=? AND m.chapter_id=? AND p.user_id=?
                    AND m.record_status='canon'
                ORDER BY m.created_at DESC
                LIMIT 1
                """,
                (project_id, chapter_id, user_id),
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        for field in (
            "key_events_json",
            "unresolved_questions_json",
            "keywords_json",
        ):
            result[field.removesuffix("_json")] = json.loads(
                str(result[field])
            )
        return result

    def list_project_chapter_memory_records(
        self,
        *,
        user_id: int,
        project_id: str,
    ) -> List[Dict[str, Any]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT ch.id AS chapter_id,
                       ch.position AS chapter_position,
                       ch.title AS chapter_title,
                       m.id AS memory_id,
                       m.summary,
                       m.key_events_json,
                       m.unresolved_questions_json,
                       m.keywords_json,
                       d.payload_json AS delta_payload_json,
                       j.status AS extraction_status
                FROM novel_chapters ch
                JOIN novel_projects p ON p.id=ch.project_id
                LEFT JOIN chapter_memory m
                  ON m.chapter_id=ch.id
                 AND m.version_id=ch.canonical_version_id
                 AND m.record_status='canon'
                LEFT JOIN story_deltas d
                  ON d.id=m.delta_id AND d.status='projected'
                LEFT JOIN generation_jobs j
                  ON j.id=(
                      SELECT latest.id
                      FROM generation_jobs latest
                      WHERE latest.chapter_id=ch.id
                        AND latest.version_id=ch.canonical_version_id
                        AND latest.operation='extract_story_delta'
                      ORDER BY latest.created_at DESC
                      LIMIT 1
                  )
                WHERE ch.project_id=? AND p.user_id=?
                    AND ch.canonical_version_id IS NOT NULL
                ORDER BY ch.position
                """,
                (project_id, user_id),
            ).fetchall()
        results: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            if item["memory_id"] and item["delta_payload_json"]:
                item["memory_status"] = "ready"
            else:
                item["memory_status"] = str(
                    item["extraction_status"] or "missing"
                )
            if item["memory_status"] == "completed":
                item["memory_status"] = "missing"
            if item["memory_status"] != "ready":
                item["payload"] = None
                results.append(item)
                continue
            for field in (
                "key_events_json",
                "unresolved_questions_json",
                "keywords_json",
            ):
                item[field.removesuffix("_json")] = json.loads(
                    str(item[field])
                )
            item["payload"] = StoryDelta.model_validate_json(
                str(item.pop("delta_payload_json"))
            ).model_dump(mode="json")
            results.append(item)
        return results

    @staticmethod
    def _owned_canonical_version(
        connection: sqlite3.Connection,
        *,
        user_id: int,
        project_id: str,
        chapter_id: str,
        version_id: str,
    ) -> Optional[sqlite3.Row]:
        return connection.execute(
            """
            SELECT p.canonical_branch_id AS branch_id
            FROM novel_chapter_versions v
            JOIN novel_chapters ch ON ch.id=v.chapter_id
            JOIN novel_projects p ON p.id=ch.project_id
            WHERE v.id=? AND v.chapter_id=? AND ch.project_id=?
                AND p.user_id=? AND ch.canonical_version_id=v.id
            """,
            (version_id, chapter_id, project_id, user_id),
        ).fetchone()

    @staticmethod
    def _retract_previous_chapter_memory(
        connection: sqlite3.Connection,
        *,
        chapter_id: str,
        delta_id: str,
    ) -> None:
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
                WHERE chapter_id=? AND delta_id<>? AND record_status='canon'
                """,
                (chapter_id, delta_id),
            )
        connection.execute(
            """
            UPDATE story_facts
            SET fact_status='retracted'
            WHERE chapter_id=? AND delta_id<>? AND fact_status='canon'
            """,
            (chapter_id, delta_id),
        )
        connection.execute(
            """
            UPDATE story_deltas
            SET status='superseded', updated_at=?
            WHERE chapter_id=? AND id<>? AND status='projected'
            """,
            (utc_now(), chapter_id, delta_id),
        )

    def _project(
        self,
        connection: sqlite3.Connection,
        *,
        delta_id: str,
        project_id: str,
        chapter_id: str,
        version_id: str,
        branch_id: str,
        delta: StoryDelta,
        created_at: str,
    ) -> Dict[str, int]:
        connection.execute(
            """
            INSERT INTO chapter_memory(
                id, project_id, chapter_id, version_id, delta_id, branch_id,
                summary, key_events_json, unresolved_questions_json,
                keywords_json, record_status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'canon', ?)
            """,
            (
                uuid.uuid4().hex,
                project_id,
                chapter_id,
                version_id,
                delta_id,
                branch_id,
                delta.chapter_summary,
                _json(
                    [
                        event.model_dump(mode="json")
                        for event in delta.events
                    ]
                ),
                _json(delta.unresolved_questions),
                _json(delta.keywords),
                created_at,
            ),
        )
        self._snapshot_matching_work_versions(
            connection,
            project_id=project_id,
            version_id=version_id,
            delta=delta,
            created_at=created_at,
        )

        characters = {
            str(row["name"]): str(row["id"])
            for row in connection.execute(
                "SELECT id, name FROM novel_characters WHERE project_id=?",
                (project_id,),
            ).fetchall()
        }
        for name, character_id in characters.items():
            ensure_memory_identity(
                connection,
                project_id=project_id,
                identity_type="character",
                canonical_text=name,
                created_at=created_at,
                source="project",
                linked_record_id=character_id,
            )

        for position, event in enumerate(delta.events, start=1):
            event_key = event.event_key or event.summary
            event_identity = ensure_memory_identity(
                connection,
                project_id=project_id,
                identity_type="event",
                canonical_text=event_key,
                created_at=created_at,
            )
            for participant in event.participants:
                ensure_memory_identity(
                    connection,
                    project_id=project_id,
                    identity_type="character",
                    canonical_text=participant,
                    created_at=created_at,
                )
            if event.location:
                ensure_memory_identity(
                    connection,
                    project_id=project_id,
                    identity_type="location",
                    canonical_text=event.location,
                    created_at=created_at,
                )
            for cause_key in event.cause_event_keys:
                ensure_memory_identity(
                    connection,
                    project_id=project_id,
                    identity_type="event",
                    canonical_text=cause_key,
                    created_at=created_at,
                )
            connection.execute(
                """
                INSERT INTO story_events(
                    id, project_id, chapter_id, version_id, delta_id,
                    branch_id, position, event_identity_id, event_key,
                    summary, participants_json, location, story_time,
                    causes_json, cause_event_keys_json, effects_json,
                    evidence, record_status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          ?, 'canon', ?)
                """,
                (
                    uuid.uuid4().hex,
                    project_id,
                    chapter_id,
                    version_id,
                    delta_id,
                    branch_id,
                    position,
                    event_identity["id"] if event_identity else None,
                    event_key,
                    event.summary,
                    _json(event.participants),
                    event.location,
                    event.story_time,
                    _json(event.causes),
                    _json(event.cause_event_keys),
                    _json(event.effects),
                    event.evidence,
                    created_at,
                ),
            )

        fact_count = 0
        for change in delta.character_changes:
            self._insert_fact(
                connection,
                project_id=project_id,
                chapter_id=chapter_id,
                version_id=version_id,
                delta_id=delta_id,
                branch_id=branch_id,
                fact_type="character_state",
                subject_type="character",
                subject_id=characters.get(change.character_name),
                subject_name=change.character_name,
                predicate=change.aspect,
                object_value={
                    "before": change.before,
                    "after": change.after,
                },
                evidence=change.evidence,
                created_at=created_at,
            )
            fact_count += 1

        for change in delta.relationship_changes:
            self._insert_fact(
                connection,
                project_id=project_id,
                chapter_id=chapter_id,
                version_id=version_id,
                delta_id=delta_id,
                branch_id=branch_id,
                fact_type="relationship",
                subject_type="relationship",
                subject_id=None,
                subject_name=f"{change.character_a} ↔ {change.character_b}",
                predicate="relationship_state",
                object_value=change.model_dump(mode="json"),
                evidence=change.evidence,
                created_at=created_at,
            )
            fact_count += 1

        for change in delta.location_changes:
            self._insert_fact(
                connection,
                project_id=project_id,
                chapter_id=chapter_id,
                version_id=version_id,
                delta_id=delta_id,
                branch_id=branch_id,
                fact_type="location",
                subject_type="entity",
                subject_id=characters.get(change.subject_name),
                subject_name=change.subject_name,
                predicate="current_location",
                object_value={
                    "before": change.from_location,
                    "after": change.to_location,
                },
                evidence=change.evidence,
                created_at=created_at,
            )
            fact_count += 1

        for change in delta.item_changes:
            self._insert_fact(
                connection,
                project_id=project_id,
                chapter_id=chapter_id,
                version_id=version_id,
                delta_id=delta_id,
                branch_id=branch_id,
                fact_type="item_state",
                subject_type="item",
                subject_id=None,
                subject_name=change.item_name,
                predicate=change.action,
                object_value=change.model_dump(mode="json"),
                evidence=change.evidence,
                created_at=created_at,
            )
            fact_count += 1

        if delta.time_advance:
            self._insert_fact(
                connection,
                project_id=project_id,
                chapter_id=chapter_id,
                version_id=version_id,
                delta_id=delta_id,
                branch_id=branch_id,
                fact_type="timeline",
                subject_type="project",
                subject_id=project_id,
                subject_name="故事时间",
                predicate="time_advance",
                object_value=delta.time_advance.model_dump(mode="json"),
                evidence="由本章时间推进提案确认",
                created_at=created_at,
            )
            fact_count += 1

        for change in delta.knowledge_changes:
            character_identity = ensure_memory_identity(
                connection,
                project_id=project_id,
                identity_type="character",
                canonical_text=change.character_name,
                created_at=created_at,
            )
            fact_key = change.canonical_fact or change.fact
            fact_identity = ensure_memory_identity(
                connection,
                project_id=project_id,
                identity_type="fact",
                canonical_text=fact_key,
                created_at=created_at,
            )
            connection.execute(
                """
                INSERT INTO character_knowledge(
                    id, project_id, chapter_id, version_id, delta_id,
                    branch_id, character_id, character_identity_id,
                    character_name, fact_identity_id, fact_key, fact_text,
                    knowledge_state, learned_via, evidence,
                    record_status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          'canon', ?)
                """,
                (
                    uuid.uuid4().hex,
                    project_id,
                    chapter_id,
                    version_id,
                    delta_id,
                    branch_id,
                    (
                        character_identity.get("linked_record_id")
                        if character_identity
                        else characters.get(change.character_name)
                    ),
                    (
                        character_identity["id"]
                        if character_identity
                        else None
                    ),
                    change.character_name,
                    fact_identity["id"] if fact_identity else None,
                    fact_key,
                    change.fact,
                    change.state,
                    change.learned_via,
                    change.evidence,
                    created_at,
                ),
            )

        for change in delta.plot_thread_changes:
            thread_identity = ensure_memory_identity(
                connection,
                project_id=project_id,
                identity_type="plot_thread",
                canonical_text=change.thread_name,
                created_at=created_at,
            )
            connection.execute(
                """
                INSERT INTO plot_threads(
                    id, project_id, chapter_id, version_id, delta_id,
                    branch_id, thread_identity_id, thread_name, thread_type,
                    action, update_text, promise, target_payoff, evidence,
                    record_status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'canon',
                          ?)
                """,
                (
                    uuid.uuid4().hex,
                    project_id,
                    chapter_id,
                    version_id,
                    delta_id,
                    branch_id,
                    thread_identity["id"] if thread_identity else None,
                    change.thread_name,
                    change.thread_type,
                    change.action,
                    change.update,
                    change.promise,
                    change.target_payoff,
                    change.evidence,
                    created_at,
                ),
            )

        for change in delta.foreshadowing_changes:
            hook_identity = ensure_memory_identity(
                connection,
                project_id=project_id,
                identity_type="foreshadowing",
                canonical_text=change.hook_name,
                created_at=created_at,
            )
            connection.execute(
                """
                INSERT INTO foreshadowing(
                    id, project_id, chapter_id, version_id, delta_id,
                    branch_id, hook_identity_id, hook_name, action,
                    description, intended_payoff, evidence, record_status,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'canon', ?)
                """,
                (
                    uuid.uuid4().hex,
                    project_id,
                    chapter_id,
                    version_id,
                    delta_id,
                    branch_id,
                    hook_identity["id"] if hook_identity else None,
                    change.hook_name,
                    change.action,
                    change.description,
                    change.intended_payoff,
                    change.evidence,
                    created_at,
                ),
            )

        replace_chapter_search_documents(
            connection,
            chapter_id=chapter_id,
            created_at=created_at,
        )
        return {
            "memory_count": 1,
            "event_count": len(delta.events),
            "fact_count": fact_count,
            "knowledge_count": len(delta.knowledge_changes),
            "plot_thread_count": len(delta.plot_thread_changes),
            "foreshadowing_count": len(delta.foreshadowing_changes),
        }

    @staticmethod
    def _snapshot_matching_work_versions(
        connection: sqlite3.Connection,
        *,
        project_id: str,
        version_id: str,
        delta: StoryDelta,
        created_at: str,
    ) -> None:
        source = connection.execute(
            """
            SELECT version.content_hash, version.content_path,
                   chapter.position AS chapter_position
            FROM novel_chapter_versions version
            JOIN novel_chapters chapter
              ON chapter.id=version.chapter_id
            WHERE version.id=?
            """,
            (version_id,),
        ).fetchone()
        if not source:
            return
        try:
            source_body = Path(str(source["content_path"])).read_text(
                encoding="utf-8"
            )
        except (OSError, UnicodeError):
            return
        source_hash = str(source["content_hash"] or "")
        if not source_hash:
            source_hash = hashlib.sha256(
                source_body.encode("utf-8")
            ).hexdigest()
        targets = connection.execute(
            """
            SELECT tag.id AS work_version_id,
                   chapter.id AS document_chapter_id,
                   chapter.title,
                   chapter.content_path
            FROM work_versions tag
            JOIN work_versions base ON base.id=tag.base_version_id
            JOIN chapters chapter
              ON chapter.document_id=tag.document_id
            LEFT JOIN work_version_story_memories snapshot
              ON snapshot.work_version_id=tag.id
             AND snapshot.document_chapter_id=chapter.id
            WHERE base.project_id=?
              AND tag.ref_type='tag'
              AND tag.intent='snapshot'
              AND chapter.position=?
              AND snapshot.id IS NULL
            """,
            (project_id, int(source["chapter_position"])),
        ).fetchall()
        payload_json = delta.model_dump_json()
        keywords_json = _json(delta.keywords)
        for target in targets:
            try:
                body = Path(str(target["content_path"])).read_text(
                    encoding="utf-8"
                )
            except (OSError, UnicodeError):
                continue
            variants = [body]
            title = str(target["title"] or "")
            for separator in ("\r\n", "\n"):
                prefix = f"{title}{separator}"
                if title and body.startswith(prefix):
                    variants.append(body[len(prefix) :])
                    break
            matching_body = next(
                (
                    candidate
                    for candidate in variants
                    if hashlib.sha256(
                        candidate.encode("utf-8")
                    ).hexdigest()
                    == source_hash
                ),
                None,
            )
            if matching_body is None:
                matching_body = next(
                    (
                        candidate
                        for candidate in variants
                        if candidate.strip() == source_body.strip()
                    ),
                    None,
                )
            if matching_body is None:
                continue
            connection.execute(
                """
                INSERT OR IGNORE INTO work_version_story_memories(
                    id, work_version_id, document_chapter_id,
                    content_hash, summary, keywords_json,
                    payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    str(target["work_version_id"]),
                    str(target["document_chapter_id"]),
                    hashlib.sha256(
                        matching_body.encode("utf-8")
                    ).hexdigest(),
                    delta.chapter_summary,
                    keywords_json,
                    payload_json,
                    created_at,
                ),
            )

    @staticmethod
    def _insert_fact(
        connection: sqlite3.Connection,
        *,
        project_id: str,
        chapter_id: str,
        version_id: str,
        delta_id: str,
        branch_id: str,
        fact_type: str,
        subject_type: str,
        subject_id: Optional[str],
        subject_name: str,
        predicate: str,
        object_value: Mapping[str, Any],
        evidence: str,
        created_at: str,
    ) -> None:
        identity_type = (
            "character"
            if subject_type == "character"
            or (subject_type == "entity" and subject_id)
            else {
                "item": "item",
                "location": "location",
            }.get(subject_type, "other")
        )
        subject_identity = ensure_memory_identity(
            connection,
            project_id=project_id,
            identity_type=identity_type,
            canonical_text=subject_name,
            created_at=created_at,
            linked_record_id=subject_id,
        )
        serialized_object = _json(object_value)
        fact_identity = ensure_memory_identity(
            connection,
            project_id=project_id,
            identity_type="fact",
            canonical_text=(
                f"{subject_name}｜{predicate}｜{serialized_object}"
            ),
            created_at=created_at,
        )
        connection.execute(
            """
            INSERT INTO story_facts(
                id, project_id, chapter_id, version_id, delta_id, branch_id,
                fact_type, subject_type, subject_id, subject_identity_id,
                subject_name, predicate, object_json, fact_identity_id,
                valid_from_chapter_id, visibility_json, confidence,
                fact_status, evidence, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '{}',
                      1.0, 'canon', ?, ?)
            """,
            (
                uuid.uuid4().hex,
                project_id,
                chapter_id,
                version_id,
                delta_id,
                branch_id,
                fact_type,
                subject_type,
                subject_id,
                (
                    subject_identity["id"]
                    if subject_identity
                    else None
                ),
                subject_name,
                predicate,
                serialized_object,
                fact_identity["id"] if fact_identity else None,
                chapter_id,
                evidence,
                created_at,
            ),
        )
