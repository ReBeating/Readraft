from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Mapping, Optional

from .db import Database, utc_now
from .story_planning_schema import PlannedStoryArc, StoryBlueprint


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _load_list(value: Any) -> List[str]:
    try:
        parsed = json.loads(str(value or "[]"))
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item).strip()]


BLUEPRINT_FIELDS = (
    "central_question",
    "protagonist_goal",
    "core_conflict",
    "stakes",
    "opening_state",
    "ending_state",
    "author_notes",
)

ARC_FIELDS = (
    "arc_type",
    "title",
    "dramatic_question",
    "promise",
    "start_state",
    "target_payoff",
    "lifecycle_status",
    "priority",
    "author_notes",
)


def _blueprint_signature(value: Mapping[str, Any]) -> str:
    return _json(
        {
            **{field: value.get(field) or "" for field in BLUEPRINT_FIELDS},
            "major_turns": list(value.get("major_turns") or []),
            "must_payoffs": list(value.get("must_payoffs") or []),
            "forbidden_shortcuts": list(
                value.get("forbidden_shortcuts") or []
            ),
        }
    )


def _arc_signature(value: Mapping[str, Any]) -> str:
    return _json(
        {
            **{field: value.get(field) for field in ARC_FIELDS},
            "involved_characters": list(
                value.get("involved_characters") or []
            ),
            "planned_turns": list(value.get("planned_turns") or []),
        }
    )


def decode_blueprint_version(
    row: Mapping[str, Any] | None,
) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    item = dict(row)
    item["major_turns"] = _load_list(item.pop("major_turns_json", "[]"))
    item["must_payoffs"] = _load_list(
        item.pop("must_payoffs_json", "[]")
    )
    item["forbidden_shortcuts"] = _load_list(
        item.pop("forbidden_shortcuts_json", "[]")
    )
    return item


def decode_arc_version(
    row: Mapping[str, Any] | None,
) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    item = dict(row)
    item["involved_characters"] = _load_list(
        item.pop("involved_characters_json", "[]")
    )
    item["planned_turns"] = _load_list(
        item.pop("planned_turns_json", "[]")
    )
    return item


class StoryPlanningService:
    def __init__(self, database: Database):
        self.database = database

    def get_blueprint(
        self, *, user_id: int, project_id: str
    ) -> Optional[Dict[str, Any]]:
        with self.database.connection() as connection:
            project = connection.execute(
                "SELECT 1 FROM novel_projects WHERE id=? AND user_id=?",
                (project_id, user_id),
            ).fetchone()
            if not project:
                return None
            head = connection.execute(
                """
                SELECT current_version_id, confirmed_version_id, updated_at
                FROM novel_story_blueprint_heads
                WHERE project_id=?
                """,
                (project_id,),
            ).fetchone()
            current = None
            confirmed = None
            if head:
                current = connection.execute(
                    """
                    SELECT * FROM novel_story_blueprint_versions
                    WHERE id=? AND project_id=?
                    """,
                    (head["current_version_id"], project_id),
                ).fetchone()
                if head["confirmed_version_id"]:
                    confirmed = connection.execute(
                        """
                        SELECT * FROM novel_story_blueprint_versions
                        WHERE id=? AND project_id=?
                        """,
                        (head["confirmed_version_id"], project_id),
                    ).fetchone()
        current_item = decode_blueprint_version(current)
        confirmed_item = decode_blueprint_version(confirmed)
        return {
            "project_id": project_id,
            "current": current_item,
            "confirmed": confirmed_item,
            "current_version_id": (
                str(head["current_version_id"]) if head else None
            ),
            "confirmed_version_id": (
                str(head["confirmed_version_id"])
                if head and head["confirmed_version_id"]
                else None
            ),
            "has_unconfirmed_changes": bool(
                head
                and head["confirmed_version_id"]
                and head["current_version_id"]
                != head["confirmed_version_id"]
            ),
        }

    def list_blueprint_versions(
        self, *, user_id: int, project_id: str, limit: int = 12
    ) -> List[Dict[str, Any]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT v.*
                FROM novel_story_blueprint_versions v
                JOIN novel_projects p ON p.id=v.project_id
                WHERE v.project_id=? AND p.user_id=?
                ORDER BY v.revision DESC
                LIMIT ?
                """,
                (project_id, user_id, max(1, min(limit, 50))),
            ).fetchall()
        return [
            item
            for row in rows
            if (item := decode_blueprint_version(row)) is not None
        ]

    def save_blueprint(
        self,
        *,
        user_id: int,
        project_id: str,
        blueprint: StoryBlueprint,
        confirm: bool,
        source: str = "manual",
    ) -> str:
        if source not in {
            "manual",
            "restore",
            "migration",
            "story_planner",
        }:
            raise ValueError("不支持的全书蓝图来源")
        if confirm:
            blueprint.ensure_confirmable()
        now = utc_now()
        version_id = uuid.uuid4().hex
        version_status = "confirmed" if confirm else "draft"
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            owner = connection.execute(
                "SELECT 1 FROM novel_projects WHERE id=? AND user_id=?",
                (project_id, user_id),
            ).fetchone()
            if not owner:
                connection.rollback()
                raise ValueError("小说项目不存在")
            revision_row = connection.execute(
                """
                SELECT COALESCE(MAX(revision), 0) + 1 AS revision
                FROM novel_story_blueprint_versions
                WHERE project_id=?
                """,
                (project_id,),
            ).fetchone()
            revision = int(revision_row["revision"])
            connection.execute(
                """
                INSERT INTO novel_story_blueprint_versions(
                    id, project_id, revision, version_status,
                    central_question, protagonist_goal, core_conflict,
                    stakes, opening_state, ending_state,
                    major_turns_json, must_payoffs_json,
                    forbidden_shortcuts_json, author_notes, source,
                    created_at, confirmed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          ?, ?)
                """,
                (
                    version_id,
                    project_id,
                    revision,
                    version_status,
                    blueprint.central_question,
                    blueprint.protagonist_goal,
                    blueprint.core_conflict,
                    blueprint.stakes,
                    blueprint.opening_state,
                    blueprint.ending_state,
                    _json(blueprint.major_turns),
                    _json(blueprint.must_payoffs),
                    _json(blueprint.forbidden_shortcuts),
                    blueprint.author_notes,
                    source,
                    now,
                    now if confirm else None,
                ),
            )
            head = connection.execute(
                """
                SELECT confirmed_version_id
                FROM novel_story_blueprint_heads
                WHERE project_id=?
                """,
                (project_id,),
            ).fetchone()
            previous_confirmed = None
            if head and head["confirmed_version_id"]:
                previous_confirmed = decode_blueprint_version(
                    connection.execute(
                        """
                        SELECT * FROM novel_story_blueprint_versions
                        WHERE id=? AND project_id=?
                        """,
                        (head["confirmed_version_id"], project_id),
                    ).fetchone()
                )
            if confirm:
                active_job = connection.execute(
                    """
                    SELECT 1 FROM generation_jobs
                    WHERE project_id=? AND status IN ('queued', 'running')
                    LIMIT 1
                    """,
                    (project_id,),
                ).fetchone()
                if active_job:
                    connection.rollback()
                    raise ValueError(
                        "AI 正在处理本书，请任务结束后再确认全书蓝图"
                    )
            confirmed_version_id = (
                version_id
                if confirm
                else (
                    str(head["confirmed_version_id"])
                    if head and head["confirmed_version_id"]
                    else None
                )
            )
            if head:
                connection.execute(
                    """
                    UPDATE novel_story_blueprint_heads
                    SET current_version_id=?, confirmed_version_id=?,
                        updated_at=?
                    WHERE project_id=?
                    """,
                    (
                        version_id,
                        confirmed_version_id,
                        now,
                        project_id,
                    ),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO novel_story_blueprint_heads(
                        project_id, current_version_id,
                        confirmed_version_id, updated_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        project_id,
                        version_id,
                        confirmed_version_id,
                        now,
                    ),
                )
            connection.execute(
                "UPDATE novel_projects SET updated_at=? WHERE id=?",
                (now, project_id),
            )
            if (
                confirm
                and previous_confirmed
                and _blueprint_signature(previous_confirmed)
                != _blueprint_signature(blueprint.model_dump(mode="json"))
            ):
                self._invalidate_future_plans(
                    connection,
                    project_id=project_id,
                    now=now,
                )
            connection.commit()
        return version_id

    def restore_blueprint_version(
        self,
        *,
        user_id: int,
        project_id: str,
        version_id: str,
    ) -> str:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT v.*
                FROM novel_story_blueprint_versions v
                JOIN novel_projects p ON p.id=v.project_id
                WHERE v.id=? AND v.project_id=? AND p.user_id=?
                """,
                (version_id, project_id, user_id),
            ).fetchone()
        item = decode_blueprint_version(row)
        if not item:
            raise ValueError("全书蓝图版本不存在")
        blueprint = StoryBlueprint.model_validate(
            {
                **{field: item[field] for field in BLUEPRINT_FIELDS},
                "major_turns": item["major_turns"],
                "must_payoffs": item["must_payoffs"],
                "forbidden_shortcuts": item["forbidden_shortcuts"],
            }
        )
        return self.save_blueprint(
            user_id=user_id,
            project_id=project_id,
            blueprint=blueprint,
            confirm=False,
            source="restore",
        )

    def apply_draft_bundle(
        self,
        *,
        user_id: int,
        project_id: str,
        blueprint: Optional[StoryBlueprint],
        arcs: List[PlannedStoryArc],
        source: str = "story_planner",
    ) -> Dict[str, Any]:
        """Apply a selected proposal atomically without changing confirmation."""

        if blueprint is None and not arcs:
            raise ValueError("至少选择全书蓝图或一条剧情线")
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            owner = connection.execute(
                "SELECT 1 FROM novel_projects WHERE id=? AND user_id=?",
                (project_id, user_id),
            ).fetchone()
            if not owner:
                connection.rollback()
                raise ValueError("小说项目不存在")
            result = self._apply_draft_bundle_in_connection(
                connection,
                project_id=project_id,
                blueprint=blueprint,
                arcs=arcs,
                source=source,
                now=now,
            )
            connection.execute(
                "UPDATE novel_projects SET updated_at=? WHERE id=?",
                (now, project_id),
            )
            connection.commit()
        return result

    @classmethod
    def _apply_draft_bundle_in_connection(
        cls,
        connection,
        *,
        project_id: str,
        blueprint: Optional[StoryBlueprint],
        arcs: List[PlannedStoryArc],
        source: str,
        now: str,
    ) -> Dict[str, Any]:
        if source != "story_planner":
            raise ValueError("不支持的方案草稿来源")
        blueprint_version_id: Optional[str] = None
        if blueprint is not None:
            blueprint_version_id = uuid.uuid4().hex
            revision_row = connection.execute(
                """
                SELECT COALESCE(MAX(revision), 0) + 1 AS revision
                FROM novel_story_blueprint_versions
                WHERE project_id=?
                """,
                (project_id,),
            ).fetchone()
            connection.execute(
                """
                INSERT INTO novel_story_blueprint_versions(
                    id, project_id, revision, version_status,
                    central_question, protagonist_goal, core_conflict,
                    stakes, opening_state, ending_state,
                    major_turns_json, must_payoffs_json,
                    forbidden_shortcuts_json, author_notes, source,
                    created_at, confirmed_at
                ) VALUES (?, ?, ?, 'draft', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          ?, ?, NULL)
                """,
                (
                    blueprint_version_id,
                    project_id,
                    int(revision_row["revision"]),
                    blueprint.central_question,
                    blueprint.protagonist_goal,
                    blueprint.core_conflict,
                    blueprint.stakes,
                    blueprint.opening_state,
                    blueprint.ending_state,
                    _json(blueprint.major_turns),
                    _json(blueprint.must_payoffs),
                    _json(blueprint.forbidden_shortcuts),
                    blueprint.author_notes,
                    source,
                    now,
                ),
            )
            head = connection.execute(
                """
                SELECT confirmed_version_id
                FROM novel_story_blueprint_heads
                WHERE project_id=?
                """,
                (project_id,),
            ).fetchone()
            if head:
                connection.execute(
                    """
                    UPDATE novel_story_blueprint_heads
                    SET current_version_id=?, updated_at=?
                    WHERE project_id=?
                    """,
                    (blueprint_version_id, now, project_id),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO novel_story_blueprint_heads(
                        project_id, current_version_id,
                        confirmed_version_id, updated_at
                    ) VALUES (?, ?, NULL, ?)
                    """,
                    (project_id, blueprint_version_id, now),
                )

        existing_rows = connection.execute(
            """
            SELECT a.id, a.position, a.confirmed_version_id,
                   current_v.title AS current_title,
                   confirmed_v.title AS confirmed_title
            FROM novel_plot_arcs a
            JOIN novel_plot_arc_versions current_v
                ON current_v.id=a.current_version_id
            LEFT JOIN novel_plot_arc_versions confirmed_v
                ON confirmed_v.id=a.confirmed_version_id
            WHERE a.project_id=?
            ORDER BY a.position
            """,
            (project_id,),
        ).fetchall()
        title_to_arc: Dict[str, str] = {}
        for row in existing_rows:
            for raw_title in (
                row["current_title"],
                row["confirmed_title"],
            ):
                normalized = str(raw_title or "").strip().casefold()
                if normalized and normalized not in title_to_arc:
                    title_to_arc[normalized] = str(row["id"])
        next_position_row = connection.execute(
            """
            SELECT COALESCE(MAX(position), 0) + 1 AS position
            FROM novel_plot_arcs WHERE project_id=?
            """,
            (project_id,),
        ).fetchone()
        next_position = int(next_position_row["position"])
        applied_arcs: List[Dict[str, Any]] = []
        selected_titles: set[str] = set()
        selected_arc_ids: set[str] = set()
        for arc in arcs:
            if arc.lifecycle_status not in {"planned", "active"}:
                raise ValueError(
                    "方案剧情线只能以 planned 或 active 草稿应用"
                )
            normalized_title = arc.title.strip().casefold()
            if normalized_title in selected_titles:
                raise ValueError("所选剧情线名称不能重复")
            selected_titles.add(normalized_title)
            arc_id = title_to_arc.get(normalized_title)
            created = arc_id is None
            if arc_id and arc_id in selected_arc_ids:
                raise ValueError("多条所选剧情线匹配到同一现有规划线")
            version_id = uuid.uuid4().hex
            if arc_id:
                revision_row = connection.execute(
                    """
                    SELECT COALESCE(MAX(revision), 0) + 1 AS revision
                    FROM novel_plot_arc_versions
                    WHERE arc_id=?
                    """,
                    (arc_id,),
                ).fetchone()
                cls._insert_arc_version(
                    connection,
                    arc_id=arc_id,
                    project_id=project_id,
                    version_id=version_id,
                    revision=int(revision_row["revision"]),
                    arc=arc,
                    confirm=False,
                    source=source,
                    now=now,
                )
                connection.execute(
                    """
                    UPDATE novel_plot_arcs
                    SET current_version_id=?, updated_at=?
                    WHERE id=? AND project_id=?
                    """,
                    (version_id, now, arc_id, project_id),
                )
            else:
                arc_id = uuid.uuid4().hex
                connection.execute(
                    """
                    INSERT INTO novel_plot_arcs(
                        id, project_id, position, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (arc_id, project_id, next_position, now, now),
                )
                next_position += 1
                cls._insert_arc_version(
                    connection,
                    arc_id=arc_id,
                    project_id=project_id,
                    version_id=version_id,
                    revision=1,
                    arc=arc,
                    confirm=False,
                    source=source,
                    now=now,
                )
                connection.execute(
                    """
                    UPDATE novel_plot_arcs
                    SET current_version_id=?, confirmed_version_id=NULL,
                        updated_at=?
                    WHERE id=?
                    """,
                    (version_id, now, arc_id),
                )
                title_to_arc[normalized_title] = arc_id
            selected_arc_ids.add(arc_id)
            applied_arcs.append(
                {
                    "arc_id": arc_id,
                    "version_id": version_id,
                    "title": arc.title,
                    "created": created,
                }
            )
        return {
            "blueprint_version_id": blueprint_version_id,
            "arcs": applied_arcs,
        }

    def list_arcs(
        self, *, user_id: int, project_id: str
    ) -> List[Dict[str, Any]]:
        with self.database.connection() as connection:
            owned = connection.execute(
                "SELECT 1 FROM novel_projects WHERE id=? AND user_id=?",
                (project_id, user_id),
            ).fetchone()
            if not owned:
                return []
            heads = connection.execute(
                """
                SELECT id, position, current_version_id,
                       confirmed_version_id, created_at, updated_at
                FROM novel_plot_arcs
                WHERE project_id=?
                ORDER BY position
                """,
                (project_id,),
            ).fetchall()
            result = []
            for head in heads:
                current = connection.execute(
                    """
                    SELECT * FROM novel_plot_arc_versions
                    WHERE id=? AND arc_id=?
                    """,
                    (head["current_version_id"], head["id"]),
                ).fetchone()
                confirmed = None
                if head["confirmed_version_id"]:
                    confirmed = connection.execute(
                        """
                        SELECT * FROM novel_plot_arc_versions
                        WHERE id=? AND arc_id=?
                        """,
                        (head["confirmed_version_id"], head["id"]),
                    ).fetchone()
                current_item = decode_arc_version(current)
                if not current_item:
                    continue
                result.append(
                    {
                        **current_item,
                        "id": str(head["id"]),
                        "position": int(head["position"]),
                        "current_version_id": str(
                            head["current_version_id"]
                        ),
                        "confirmed_version_id": (
                            str(head["confirmed_version_id"])
                            if head["confirmed_version_id"]
                            else None
                        ),
                        "confirmed": decode_arc_version(confirmed),
                        "has_unconfirmed_changes": bool(
                            head["confirmed_version_id"]
                            and head["current_version_id"]
                            != head["confirmed_version_id"]
                        ),
                    }
                )
        return result

    def list_arc_versions(
        self,
        *,
        user_id: int,
        project_id: str,
        arc_id: str,
        limit: int = 12,
    ) -> List[Dict[str, Any]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT v.*
                FROM novel_plot_arc_versions v
                JOIN novel_plot_arcs a ON a.id=v.arc_id
                JOIN novel_projects p ON p.id=a.project_id
                WHERE v.arc_id=? AND a.project_id=? AND p.user_id=?
                ORDER BY v.revision DESC
                LIMIT ?
                """,
                (
                    arc_id,
                    project_id,
                    user_id,
                    max(1, min(limit, 50)),
                ),
            ).fetchall()
        return [
            item
            for row in rows
            if (item := decode_arc_version(row)) is not None
        ]

    def create_arc(
        self,
        *,
        user_id: int,
        project_id: str,
        arc: PlannedStoryArc,
        confirm: bool,
    ) -> str:
        if confirm:
            arc.ensure_confirmable()
        arc_id = uuid.uuid4().hex
        version_id = uuid.uuid4().hex
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            owner = connection.execute(
                "SELECT 1 FROM novel_projects WHERE id=? AND user_id=?",
                (project_id, user_id),
            ).fetchone()
            if not owner:
                connection.rollback()
                raise ValueError("小说项目不存在")
            if confirm:
                active_job = connection.execute(
                    """
                    SELECT 1 FROM generation_jobs
                    WHERE project_id=? AND status IN ('queued', 'running')
                    LIMIT 1
                    """,
                    (project_id,),
                ).fetchone()
                if active_job:
                    connection.rollback()
                    raise ValueError(
                        "AI 正在处理本书，请任务结束后再确认剧情线"
                    )
            position_row = connection.execute(
                """
                SELECT COALESCE(MAX(position), 0) + 1 AS position
                FROM novel_plot_arcs
                WHERE project_id=?
                """,
                (project_id,),
            ).fetchone()
            position = int(position_row["position"])
            connection.execute(
                """
                INSERT INTO novel_plot_arcs(
                    id, project_id, position, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (arc_id, project_id, position, now, now),
            )
            self._insert_arc_version(
                connection,
                arc_id=arc_id,
                project_id=project_id,
                version_id=version_id,
                revision=1,
                arc=arc,
                confirm=confirm,
                source="manual",
                now=now,
            )
            connection.execute(
                """
                UPDATE novel_plot_arcs
                SET current_version_id=?, confirmed_version_id=?,
                    updated_at=?
                WHERE id=?
                """,
                (
                    version_id,
                    version_id if confirm else None,
                    now,
                    arc_id,
                ),
            )
            connection.execute(
                "UPDATE novel_projects SET updated_at=? WHERE id=?",
                (now, project_id),
            )
            connection.commit()
        return arc_id

    def update_arc(
        self,
        *,
        user_id: int,
        project_id: str,
        arc_id: str,
        arc: PlannedStoryArc,
        confirm: bool,
        source: str = "manual",
    ) -> str:
        if source not in {
            "manual",
            "restore",
            "lifecycle",
            "story_planner",
        }:
            raise ValueError("不支持的剧情线来源")
        if confirm:
            arc.ensure_confirmable()
        version_id = uuid.uuid4().hex
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            head = connection.execute(
                """
                SELECT a.confirmed_version_id
                FROM novel_plot_arcs a
                JOIN novel_projects p ON p.id=a.project_id
                WHERE a.id=? AND a.project_id=? AND p.user_id=?
                """,
                (arc_id, project_id, user_id),
            ).fetchone()
            if not head:
                connection.rollback()
                raise ValueError("规划剧情线不存在")
            previous_confirmed = None
            if head["confirmed_version_id"]:
                previous_confirmed = decode_arc_version(
                    connection.execute(
                        """
                        SELECT * FROM novel_plot_arc_versions
                        WHERE id=? AND arc_id=?
                        """,
                        (head["confirmed_version_id"], arc_id),
                    ).fetchone()
                )
            if confirm:
                active_job = connection.execute(
                    """
                    SELECT 1 FROM generation_jobs
                    WHERE project_id=? AND status IN ('queued', 'running')
                    LIMIT 1
                    """,
                    (project_id,),
                ).fetchone()
                if active_job:
                    connection.rollback()
                    raise ValueError(
                        "AI 正在处理本书，请任务结束后再确认剧情线"
                    )
            revision_row = connection.execute(
                """
                SELECT COALESCE(MAX(revision), 0) + 1 AS revision
                FROM novel_plot_arc_versions
                WHERE arc_id=?
                """,
                (arc_id,),
            ).fetchone()
            revision = int(revision_row["revision"])
            self._insert_arc_version(
                connection,
                arc_id=arc_id,
                project_id=project_id,
                version_id=version_id,
                revision=revision,
                arc=arc,
                confirm=confirm,
                source=source,
                now=now,
            )
            confirmed_version_id = (
                version_id
                if confirm
                else (
                    str(head["confirmed_version_id"])
                    if head["confirmed_version_id"]
                    else None
                )
            )
            connection.execute(
                """
                UPDATE novel_plot_arcs
                SET current_version_id=?, confirmed_version_id=?,
                    updated_at=?
                WHERE id=?
                """,
                (
                    version_id,
                    confirmed_version_id,
                    now,
                    arc_id,
                ),
            )
            connection.execute(
                "UPDATE novel_projects SET updated_at=? WHERE id=?",
                (now, project_id),
            )
            if (
                confirm
                and previous_confirmed
                and _arc_signature(previous_confirmed)
                != _arc_signature(arc.model_dump(mode="json"))
            ):
                self._invalidate_future_plans(
                    connection,
                    project_id=project_id,
                    now=now,
                    arc_titles={
                        str(previous_confirmed["title"]),
                        arc.title,
                    },
                )
            connection.commit()
        return version_id

    def restore_arc_version(
        self,
        *,
        user_id: int,
        project_id: str,
        arc_id: str,
        version_id: str,
    ) -> str:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT v.*
                FROM novel_plot_arc_versions v
                JOIN novel_plot_arcs a ON a.id=v.arc_id
                JOIN novel_projects p ON p.id=a.project_id
                WHERE v.id=? AND v.arc_id=? AND a.project_id=?
                    AND p.user_id=?
                """,
                (version_id, arc_id, project_id, user_id),
            ).fetchone()
        item = decode_arc_version(row)
        if not item:
            raise ValueError("剧情线版本不存在")
        arc = PlannedStoryArc.model_validate(
            {
                **{field: item[field] for field in ARC_FIELDS},
                "involved_characters": item["involved_characters"],
                "planned_turns": item["planned_turns"],
            }
        )
        return self.update_arc(
            user_id=user_id,
            project_id=project_id,
            arc_id=arc_id,
            arc=arc,
            confirm=False,
            source="restore",
        )

    def archive_arc(
        self, *, user_id: int, project_id: str, arc_id: str
    ) -> str:
        arcs = self.list_arcs(user_id=user_id, project_id=project_id)
        item = next((row for row in arcs if row["id"] == arc_id), None)
        if not item:
            raise ValueError("规划剧情线不存在")
        arc = PlannedStoryArc.model_validate(
            {
                **{field: item[field] for field in ARC_FIELDS},
                "involved_characters": item["involved_characters"],
                "planned_turns": item["planned_turns"],
                "lifecycle_status": "abandoned",
            }
        )
        return self.update_arc(
            user_id=user_id,
            project_id=project_id,
            arc_id=arc_id,
            arc=arc,
            confirm=True,
            source="lifecycle",
        )

    @staticmethod
    def _insert_arc_version(
        connection,
        *,
        arc_id: str,
        project_id: str,
        version_id: str,
        revision: int,
        arc: PlannedStoryArc,
        confirm: bool,
        source: str,
        now: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO novel_plot_arc_versions(
                id, arc_id, project_id, revision, version_status,
                arc_type, title, dramatic_question, promise,
                start_state, target_payoff, involved_characters_json,
                planned_turns_json, lifecycle_status, priority,
                author_notes, source, created_at, confirmed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?)
            """,
            (
                version_id,
                arc_id,
                project_id,
                revision,
                "confirmed" if confirm else "draft",
                arc.arc_type,
                arc.title,
                arc.dramatic_question,
                arc.promise,
                arc.start_state,
                arc.target_payoff,
                _json(arc.involved_characters),
                _json(arc.planned_turns),
                arc.lifecycle_status,
                arc.priority,
                arc.author_notes,
                source,
                now,
                now if confirm else None,
            ),
        )

    @staticmethod
    def _invalidate_future_plans(
        connection,
        *,
        project_id: str,
        now: str,
        arc_titles: Optional[set[str]] = None,
    ) -> int:
        rows = connection.execute(
            """
            SELECT cp.id, cp.plot_threads_json
            FROM novel_chapter_plans cp
            JOIN novel_chapters ch ON ch.id=cp.chapter_id
            WHERE cp.project_id=? AND cp.status='confirmed'
                AND ch.canonical_version_id IS NULL
            """,
            (project_id,),
        ).fetchall()
        plan_ids = []
        normalized_titles = {
            title.strip().casefold()
            for title in (arc_titles or set())
            if title.strip()
        }
        for row in rows:
            if normalized_titles:
                plan_threads = {
                    item.strip().casefold()
                    for item in _load_list(row["plot_threads_json"])
                }
                if not plan_threads.intersection(normalized_titles):
                    continue
            plan_ids.append(str(row["id"]))
        for plan_id in plan_ids:
            connection.execute(
                """
                UPDATE novel_chapter_plans
                SET status='draft', confirmed_at=NULL, updated_at=?
                WHERE id=?
                """,
                (now, plan_id),
            )
            connection.execute(
                """
                UPDATE novel_chapters
                SET needs_recheck=1, updated_at=?
                WHERE id=(
                    SELECT chapter_id FROM novel_chapter_plans WHERE id=?
                )
                """,
                (now, plan_id),
            )
            connection.execute(
                """
                UPDATE novel_scene_beats
                SET draft_status=CASE
                        WHEN current_version_id IS NOT NULL THEN 'stale'
                        ELSE draft_status
                    END,
                    updated_at=?
                WHERE plan_id=? AND beat_status='active'
                """,
                (now, plan_id),
            )
        return len(plan_ids)
