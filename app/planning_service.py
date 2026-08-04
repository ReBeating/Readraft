from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Mapping, Optional

from .db import Database, utc_now
from .json_support import dump_json as _json, load_json as _load_json
from .planning_schema import (
    ChapterTaskCard,
    SceneBeat,
    chapter_task_card_fingerprint,
)
from .scene_service import scene_plan_fingerprint
from .story_structure_schema import AuthorChapterSkeleton


PLAN_JSON_FIELDS = {
    "plot_threads_json": "plot_threads",
    "must_happen_json": "must_happen",
    "must_preserve_json": "must_preserve",
    "forbidden_json": "forbidden",
    "foreshadow_setup_json": "foreshadow_setup",
    "foreshadow_payoff_json": "foreshadow_payoff",
}


def _card_from_storage(
    plan: Mapping[str, Any],
    scenes: List[Mapping[str, Any]],
) -> ChapterTaskCard:
    plan_data = dict(plan)
    card_data: Dict[str, Any] = {
        "purpose": str(plan_data.get("purpose") or ""),
        "start_state": str(plan_data.get("start_state") or ""),
        "end_state": str(plan_data.get("end_state") or ""),
        "central_conflict": str(
            plan_data.get("central_conflict") or ""
        ),
        "emotional_value": str(
            plan_data.get("emotional_value") or ""
        ),
        "ending_hook": str(plan_data.get("ending_hook") or ""),
        "target_chars": int(plan_data.get("target_chars") or 3000),
        "scenes": [],
    }
    for stored, public in PLAN_JSON_FIELDS.items():
        card_data[public] = _load_json(plan_data.get(stored), [])
    for raw in scenes:
        scene = dict(raw)
        scene["key_items"] = _load_json(
            scene.get("key_items_json"), []
        )
        scene["requirement_refs"] = _load_json(
            scene.get("requirement_refs_json"), []
        )
        card_data["scenes"].append(
            {
                name: scene[name]
                for name in SceneBeat.model_fields
                if name in scene
            }
        )
    return ChapterTaskCard.model_validate(
        {
            name: card_data[name]
            for name in ChapterTaskCard.model_fields
            if name in card_data
        }
    )


class PlanningService:
    def __init__(self, database: Database):
        self.database = database

    def list_volumes(
        self, *, user_id: int, project_id: str
    ) -> List[Dict[str, Any]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT v.*,
                       (SELECT COUNT(*) FROM novel_chapters ch
                        WHERE ch.volume_id=v.id) AS chapter_count,
                       (SELECT COUNT(*) FROM novel_chapters ch
                        WHERE ch.volume_id=v.id
                          AND ch.head_version_id IS NOT NULL)
                           AS canonical_chapter_count,
                       (SELECT COUNT(*) FROM novel_chapters ch
                        WHERE ch.volume_id=v.id
                          AND ch.head_version_id IS NULL)
                           AS future_chapter_count
                FROM novel_volumes v
                JOIN novel_projects p ON p.id=v.project_id
                WHERE v.project_id=? AND p.user_id=?
                ORDER BY v.position
                """,
                (project_id, user_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_rolling_plan(
        self, *, user_id: int, project_id: str
    ) -> Optional[Dict[str, Any]]:
        with self.database.connection() as connection:
            project = connection.execute(
                """
                SELECT planning_horizon FROM novel_projects
                WHERE id=? AND user_id=?
                """,
                (project_id, user_id),
            ).fetchone()
            if not project:
                return None
            current = connection.execute(
                """
                SELECT COALESCE(MAX(position), 0) AS position
                FROM novel_chapters
                WHERE project_id=? AND head_version_id IS NOT NULL
                """,
                (project_id,),
            ).fetchone()
            horizon = int(project["planning_horizon"] or 20)
            rows = connection.execute(
                """
                SELECT ch.id, ch.position, ch.title, ch.outline,
                       ch.needs_recheck, v.title AS volume_title,
                       ch.skeleton_role,
                       ch.skeleton_arc_titles_json,
                       ch.skeleton_ending_hook,
                       cp.status AS plan_status,
                       (SELECT COUNT(*) FROM novel_scene_beats sb
                        WHERE sb.plan_id=cp.id
                            AND sb.beat_status='active') AS scene_count
                FROM novel_chapters ch
                LEFT JOIN novel_volumes v ON v.id=ch.volume_id
                LEFT JOIN novel_chapter_plans cp ON cp.chapter_id=ch.id
                WHERE ch.project_id=? AND ch.position>?
                ORDER BY ch.position
                LIMIT ?
                """,
                (project_id, int(current["position"]), horizon),
            ).fetchall()
        chapters = []
        for row in rows:
            item = dict(row)
            try:
                item["skeleton_arc_titles"] = json.loads(
                    str(item.pop("skeleton_arc_titles_json", "[]"))
                )
            except (TypeError, ValueError):
                item["skeleton_arc_titles"] = []
            chapters.append(item)
        return {
            "current_position": int(current["position"]),
            "horizon": horizon,
            "planned_count": len(chapters),
            "confirmed_count": sum(
                1 for row in chapters if row["plan_status"] == "confirmed"
            ),
            "missing_count": max(0, horizon - len(chapters)),
            "chapters": chapters,
        }

    def create_volume(
        self,
        *,
        user_id: int,
        project_id: str,
        title: str,
        goal: str,
        start_state: str,
        end_state: str,
        major_conflict: str,
        payoff: str,
    ) -> str:
        volume_id = uuid.uuid4().hex
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
            position = connection.execute(
                """
                SELECT COALESCE(MAX(position), 0) + 1 AS next_position
                FROM novel_volumes WHERE project_id=?
                """,
                (project_id,),
            ).fetchone()
            connection.execute(
                """
                INSERT INTO novel_volumes(
                    id, project_id, position, title, goal, start_state,
                    end_state, major_conflict, payoff, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'planning', ?, ?)
                """,
                (
                    volume_id,
                    project_id,
                    int(position["next_position"]),
                    title,
                    goal,
                    start_state,
                    end_state,
                    major_conflict,
                    payoff,
                    now,
                    now,
                ),
            )
            connection.execute(
                "UPDATE novel_projects SET updated_at=? WHERE id=?",
                (now, project_id),
            )
            connection.commit()
        return volume_id

    def update_volume(
        self,
        *,
        user_id: int,
        project_id: str,
        volume_id: str,
        title: str,
        goal: str,
        start_state: str,
        end_state: str,
        major_conflict: str,
        payoff: str,
    ) -> Dict[str, Any]:
        now = utc_now()
        fields = {
            "title": title,
            "goal": goal,
            "start_state": start_state,
            "end_state": end_state,
            "major_conflict": major_conflict,
            "payoff": payoff,
        }
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            volume = connection.execute(
                """
                SELECT v.*,
                       COALESCE((
                           SELECT MAX(ch.position)
                           FROM novel_chapters ch
                           WHERE ch.project_id=v.project_id
                             AND ch.head_version_id IS NOT NULL
                       ), 0) AS current_canonical_position
                FROM novel_volumes v
                JOIN novel_projects p ON p.id=v.project_id
                WHERE v.id=? AND v.project_id=? AND p.user_id=?
                """,
                (volume_id, project_id, user_id),
            ).fetchone()
            if not volume:
                connection.rollback()
                raise ValueError("分卷不存在")
            changed = any(
                str(volume[key] or "") != value
                for key, value in fields.items()
            )
            if not changed:
                connection.rollback()
                return {
                    "changed": False,
                    "affected_chapter_count": 0,
                    "reset_task_card_count": 0,
                    "stale_scene_count": 0,
                }
            current_position = int(
                volume["current_canonical_position"] or 0
            )
            active = connection.execute(
                """
                SELECT 1
                FROM generation_jobs j
                JOIN novel_chapters ch ON ch.id=j.chapter_id
                WHERE ch.volume_id=? AND ch.position>?
                  AND j.status IN ('queued', 'running')
                LIMIT 1
                """,
                (volume_id, current_position),
            ).fetchone()
            if active:
                connection.rollback()
                raise ValueError(
                    "AI 正在处理本卷的未来章节，请完成任务后再修改分卷"
                )
            affected_row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM novel_chapters
                WHERE volume_id=? AND position>?
                  AND head_version_id IS NULL
                """,
                (volume_id, current_position),
            ).fetchone()
            task_card_row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM novel_chapter_plans cp
                JOIN novel_chapters ch ON ch.id=cp.chapter_id
                WHERE ch.volume_id=? AND ch.position>?
                  AND ch.head_version_id IS NULL
                  AND cp.status='confirmed'
                """,
                (volume_id, current_position),
            ).fetchone()
            connection.execute(
                """
                UPDATE novel_volumes
                SET title=?, goal=?, start_state=?, end_state=?,
                    major_conflict=?, payoff=?, updated_at=?
                WHERE id=?
                """,
                (
                    title,
                    goal,
                    start_state,
                    end_state,
                    major_conflict,
                    payoff,
                    now,
                    volume_id,
                ),
            )
            connection.execute(
                """
                UPDATE novel_chapter_plans
                SET status='draft', source='manual', confirmed_at=NULL,
                    updated_at=?
                WHERE chapter_id IN (
                    SELECT ch.id
                    FROM novel_chapters ch
                    WHERE ch.volume_id=? AND ch.position>?
                      AND ch.head_version_id IS NULL
                )
                """,
                (now, volume_id, current_position),
            )
            stale_cursor = connection.execute(
                """
                UPDATE novel_scene_beats
                SET draft_status='stale', updated_at=?
                WHERE current_version_id IS NOT NULL
                  AND plan_id IN (
                    SELECT cp.id
                    FROM novel_chapter_plans cp
                    JOIN novel_chapters ch ON ch.id=cp.chapter_id
                    WHERE ch.volume_id=? AND ch.position>?
                      AND ch.head_version_id IS NULL
                  )
                """,
                (now, volume_id, current_position),
            )
            connection.execute(
                """
                UPDATE novel_chapters
                SET needs_recheck=CASE
                        WHEN head_version_id IS NOT NULL
                          OR char_count>0
                          OR EXISTS(
                              SELECT 1
                              FROM novel_scene_beats sb
                              JOIN novel_chapter_plans cp
                                ON cp.id=sb.plan_id
                              WHERE cp.chapter_id=novel_chapters.id
                                AND sb.current_version_id IS NOT NULL
                          )
                        THEN 1
                        ELSE needs_recheck
                    END,
                    updated_at=?
                WHERE volume_id=? AND position>?
                  AND head_version_id IS NULL
                """,
                (now, volume_id, current_position),
            )
            connection.execute(
                "UPDATE novel_projects SET updated_at=? WHERE id=?",
                (now, project_id),
            )
            connection.commit()
        return {
            "changed": True,
            "affected_chapter_count": int(affected_row["count"]),
            "reset_task_card_count": int(task_card_row["count"]),
            "stale_scene_count": int(stale_cursor.rowcount),
        }

    def update_future_chapter_skeleton(
        self,
        *,
        user_id: int,
        project_id: str,
        chapter_id: str,
        volume_id: Optional[str],
        skeleton: AuthorChapterSkeleton,
    ) -> Dict[str, Any]:
        now = utc_now()
        arc_titles = list(dict.fromkeys(skeleton.arc_titles))
        key_points = list(skeleton.key_points)
        key_points_text = "\n".join(key_points)
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            chapter = connection.execute(
                """
                SELECT ch.*, p.target_chapter_chars,
                       COALESCE((
                           SELECT MAX(canon.position)
                           FROM novel_chapters canon
                           WHERE canon.project_id=ch.project_id
                             AND canon.head_version_id IS NOT NULL
                       ), 0) AS current_canonical_position
                FROM novel_chapters ch
                JOIN novel_projects p ON p.id=ch.project_id
                WHERE ch.id=? AND ch.project_id=? AND p.user_id=?
                """,
                (chapter_id, project_id, user_id),
            ).fetchone()
            if not chapter:
                connection.rollback()
                raise ValueError("章节不存在")
            if (
                chapter["head_version_id"]
                or int(chapter["position"])
                <= int(chapter["current_canonical_position"] or 0)
            ):
                connection.rollback()
                raise ValueError(
                    "已进入正史范围的章节骨架不能在这里修改"
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
                    "AI 正在处理本章，请完成任务后再修改滚动骨架"
                )
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
            confirmed_arc_rows = connection.execute(
                """
                SELECT version.title
                FROM novel_plot_arcs arc
                JOIN novel_plot_arc_versions version
                  ON version.id=arc.confirmed_version_id
                WHERE arc.project_id=?
                  AND version.lifecycle_status IN ('planned', 'active')
                """,
                (project_id,),
            ).fetchall()
            confirmed_arc_titles = {
                str(row["title"]) for row in confirmed_arc_rows
            }
            unknown_arcs = [
                title
                for title in arc_titles
                if title not in confirmed_arc_titles
            ]
            if unknown_arcs:
                connection.rollback()
                raise ValueError(
                    "滚动骨架只能引用已确认且可推进的剧情线："
                    + "、".join(unknown_arcs)
                )
            plan = connection.execute(
                """
                SELECT id, status
                FROM novel_chapter_plans
                WHERE chapter_id=?
                """,
                (chapter_id,),
            ).fetchone()
            try:
                current_arcs = json.loads(
                    str(chapter["skeleton_arc_titles_json"] or "[]")
                )
            except (TypeError, ValueError):
                current_arcs = []
            changed = (
                str(chapter["title"] or "") != skeleton.title
                or str(chapter["volume_id"] or "") != str(volume_id or "")
                or str(chapter["outline"] or "") != skeleton.purpose
                or str(chapter["key_points"] or "") != key_points_text
                or str(chapter["skeleton_role"] or "")
                != skeleton.structural_role
                or list(current_arcs) != arc_titles
                or str(chapter["skeleton_ending_hook"] or "")
                != skeleton.ending_hook
                or plan is None
            )
            if not changed:
                connection.rollback()
                return {
                    "changed": False,
                    "task_card_reset": False,
                    "stale_scene_count": 0,
                }
            task_card_reset = bool(
                plan and str(plan["status"]) == "confirmed"
            )
            if plan:
                plan_id = str(plan["id"])
                connection.execute(
                    """
                    UPDATE novel_chapter_plans
                    SET purpose=?, plot_threads_json=?,
                        must_happen_json=?, ending_hook=?,
                        status='draft', source='manual',
                        confirmed_at=NULL, updated_at=?
                    WHERE id=?
                    """,
                    (
                        skeleton.purpose,
                        _json(arc_titles),
                        _json(key_points),
                        skeleton.ending_hook,
                        now,
                        plan_id,
                    ),
                )
            else:
                plan_id = uuid.uuid4().hex
                connection.execute(
                    """
                    INSERT INTO novel_chapter_plans(
                        id, project_id, chapter_id, purpose,
                        plot_threads_json, must_happen_json, ending_hook,
                        target_chars, status, source,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'draft', 'manual',
                              ?, ?)
                    """,
                    (
                        plan_id,
                        project_id,
                        chapter_id,
                        skeleton.purpose,
                        _json(arc_titles),
                        _json(key_points),
                        skeleton.ending_hook,
                        int(chapter["target_chapter_chars"] or 3000),
                        now,
                        now,
                    ),
                )
            stale_cursor = connection.execute(
                """
                UPDATE novel_scene_beats
                SET draft_status='stale', updated_at=?
                WHERE plan_id=? AND current_version_id IS NOT NULL
                """,
                (now, plan_id),
            )
            connection.execute(
                """
                UPDATE novel_chapters
                SET volume_id=?, title=?, outline=?, key_points=?,
                    skeleton_role=?, skeleton_arc_titles_json=?,
                    skeleton_ending_hook=?,
                    skeleton_application_id=NULL,
                    needs_recheck=CASE
                        WHEN head_version_id IS NOT NULL
                          OR char_count>0
                          OR EXISTS(
                              SELECT 1
                              FROM novel_scene_beats sb
                              JOIN novel_chapter_plans cp
                                ON cp.id=sb.plan_id
                              WHERE cp.chapter_id=novel_chapters.id
                                AND sb.current_version_id IS NOT NULL
                          )
                        THEN 1
                        ELSE needs_recheck
                    END,
                    updated_at=?
                WHERE id=?
                """,
                (
                    volume_id,
                    skeleton.title,
                    skeleton.purpose,
                    key_points_text,
                    skeleton.structural_role,
                    _json(arc_titles),
                    skeleton.ending_hook,
                    now,
                    chapter_id,
                ),
            )
            connection.execute(
                "UPDATE novel_projects SET updated_at=? WHERE id=?",
                (now, project_id),
            )
            connection.commit()
        return {
            "changed": True,
            "task_card_reset": task_card_reset,
            "stale_scene_count": int(stale_cursor.rowcount),
        }

    def get_task_card(
        self,
        *,
        user_id: int,
        project_id: str,
        chapter_id: str,
    ) -> Optional[Dict[str, Any]]:
        with self.database.connection() as connection:
            chapter = connection.execute(
                """
                SELECT ch.id, ch.project_id, ch.volume_id, ch.position,
                       ch.title, ch.outline, ch.key_points,
                       ch.head_version_id,
                       ch.skeleton_role,
                       ch.skeleton_arc_titles_json,
                       ch.skeleton_ending_hook,
                       p.title AS project_title, p.target_chapter_chars,
                       COALESCE((
                           SELECT MAX(canon.position)
                           FROM novel_chapters canon
                           WHERE canon.project_id=ch.project_id
                             AND canon.head_version_id IS NOT NULL
                       ), 0) AS current_canonical_position
                FROM novel_chapters ch
                JOIN novel_projects p ON p.id=ch.project_id
                WHERE ch.id=? AND ch.project_id=? AND p.user_id=?
                """,
                (chapter_id, project_id, user_id),
            ).fetchone()
            if not chapter:
                return None
            plan = connection.execute(
                """
                SELECT * FROM novel_chapter_plans WHERE chapter_id=?
                """,
                (chapter_id,),
            ).fetchone()
            scenes = []
            if plan:
                scenes = [
                    dict(row)
                    for row in connection.execute(
                        """
                        SELECT * FROM novel_scene_beats
                        WHERE plan_id=? AND beat_status='active'
                        ORDER BY position
                        """,
                        (plan["id"],),
                    ).fetchall()
                ]
        chapter_data = dict(chapter)
        try:
            chapter_data["skeleton_arc_titles"] = json.loads(
                str(
                    chapter_data.pop(
                        "skeleton_arc_titles_json", "[]"
                    )
                )
            )
        except (TypeError, ValueError):
            chapter_data["skeleton_arc_titles"] = []
        if not plan:
            return {
                **chapter_data,
                "plan_id": None,
                "purpose": str(chapter["outline"] or ""),
                "start_state": "",
                "end_state": "",
                "central_conflict": "",
                "emotional_value": "",
                "plot_threads": list(
                    chapter_data["skeleton_arc_titles"]
                ),
                "must_happen": [
                    line.strip()
                    for line in str(chapter["key_points"] or "").splitlines()
                    if line.strip()
                ],
                "must_preserve": [],
                "forbidden": [],
                "foreshadow_setup": [],
                "foreshadow_payoff": [],
                "ending_hook": str(
                    chapter_data["skeleton_ending_hook"] or ""
                ),
                "target_chars": int(chapter["target_chapter_chars"]),
                "status": "draft",
                "source": "manual",
                "scenes": [],
            }
        plan_data = dict(plan)
        plan_id = str(plan_data.pop("id"))
        plan_data.pop("project_id", None)
        plan_data.pop("chapter_id", None)
        result = {**chapter_data, **plan_data}
        result["plan_id"] = plan_id
        for stored, public in PLAN_JSON_FIELDS.items():
            try:
                result[public] = json.loads(str(result[stored]))
            except (TypeError, ValueError):
                result[public] = []
        for scene in scenes:
            try:
                scene["key_items"] = json.loads(
                    str(scene["key_items_json"])
                )
            except (TypeError, ValueError):
                scene["key_items"] = []
            scene["requirement_refs"] = _load_json(
                scene.get("requirement_refs_json"), []
            )
        result["scenes"] = scenes
        return result

    def upsert_task_card(
        self,
        *,
        user_id: int,
        project_id: str,
        chapter_id: str,
        volume_id: Optional[str],
        card: ChapterTaskCard,
        confirm: bool,
        source: str = "manual",
        allow_active_plan_job: bool = False,
        expected_card_fingerprint: Optional[str] = None,
    ) -> str:
        if source not in {"manual", "ai", "migration"}:
            raise ValueError("不支持的任务卡来源")
        if confirm:
            card.ensure_confirmable()
        now = utc_now()
        status = "confirmed" if confirm else "draft"
        with self.database.connection() as connection:
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
            active = connection.execute(
                """
                SELECT operation FROM generation_jobs
                WHERE chapter_id=? AND status IN ('queued', 'running')
                LIMIT 1
                """,
                (chapter_id,),
            ).fetchone()
            if active and not (
                allow_active_plan_job
                and str(active["operation"])
                in {"plan_chapter", "plan_scene_beats"}
            ):
                connection.rollback()
                raise ValueError(
                    "AI 正在处理本章，请完成任务后再修改任务卡"
                )
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
            existing = connection.execute(
                """
                SELECT id, created_at FROM novel_chapter_plans
                WHERE chapter_id=?
                """,
                (chapter_id,),
            ).fetchone()
            if expected_card_fingerprint:
                if not existing:
                    connection.rollback()
                    raise ValueError(
                        "任务卡基线已经不存在，请重新打开页面"
                    )
                stored_plan = connection.execute(
                    """
                    SELECT * FROM novel_chapter_plans WHERE id=?
                    """,
                    (existing["id"],),
                ).fetchone()
                stored_scenes = connection.execute(
                    """
                    SELECT * FROM novel_scene_beats
                    WHERE plan_id=? AND beat_status='active'
                    ORDER BY position
                    """,
                    (existing["id"],),
                ).fetchall()
                current_fingerprint = chapter_task_card_fingerprint(
                    _card_from_storage(stored_plan, list(stored_scenes))
                )
                if current_fingerprint != expected_card_fingerprint:
                    connection.rollback()
                    raise ValueError(
                        "任务卡或场景在拆解期间已经变化，未覆盖较新的作者修改"
                    )
            plan_id = str(existing["id"]) if existing else uuid.uuid4().hex
            values = (
                card.purpose,
                card.start_state,
                card.end_state,
                card.central_conflict,
                card.emotional_value,
                _json(card.plot_threads),
                _json(card.must_happen),
                _json(card.must_preserve),
                _json(card.forbidden),
                _json(card.foreshadow_setup),
                _json(card.foreshadow_payoff),
                card.ending_hook,
                card.target_chars,
                status,
                source,
                now if confirm else None,
                now,
            )
            if existing:
                connection.execute(
                    """
                    UPDATE novel_chapter_plans
                    SET purpose=?, start_state=?, end_state=?,
                        central_conflict=?, emotional_value=?,
                        plot_threads_json=?, must_happen_json=?,
                        must_preserve_json=?, forbidden_json=?,
                        foreshadow_setup_json=?,
                        foreshadow_payoff_json=?, ending_hook=?,
                        target_chars=?, status=?, source=?, confirmed_at=?,
                        updated_at=?
                    WHERE id=?
                    """,
                    (*values, plan_id),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO novel_chapter_plans(
                        id, project_id, chapter_id, purpose, start_state,
                        end_state, central_conflict, emotional_value,
                        plot_threads_json, must_happen_json,
                        must_preserve_json, forbidden_json,
                        foreshadow_setup_json, foreshadow_payoff_json,
                        ending_hook, target_chars, status, source,
                        confirmed_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                              ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        plan_id,
                        project_id,
                        chapter_id,
                        *values[:-1],
                        now,
                        now,
                    ),
                )
            existing_scenes = {
                int(row["position"]): row
                for row in connection.execute(
                    """
                    SELECT * FROM novel_scene_beats
                    WHERE plan_id=?
                    ORDER BY position
                    """,
                    (plan_id,),
                ).fetchall()
            }
            for position, scene in enumerate(card.scenes, start=1):
                stored = existing_scenes.get(position)
                if stored:
                    changed = (
                        scene_plan_fingerprint(dict(stored))
                        != scene_plan_fingerprint(
                            scene.model_dump(mode="json")
                        )
                    )
                    connection.execute(
                        """
                        UPDATE novel_scene_beats
                        SET pov_character=?, goal=?, obstacle=?, action=?,
                            reveal=?, conceal=?, subtext=?, location=?,
                            key_items_json=?, end_state=?, transition=?,
                            requirement_refs_json=?,
                            beat_status='active',
                            draft_status=CASE
                                WHEN ? AND current_version_id IS NOT NULL
                                THEN 'stale'
                                ELSE draft_status
                            END,
                            updated_at=?
                        WHERE id=?
                        """,
                        (
                            scene.pov_character,
                            scene.goal,
                            scene.obstacle,
                            scene.action,
                            scene.reveal,
                            scene.conceal,
                            scene.subtext,
                            scene.location,
                            _json(scene.key_items),
                            scene.end_state,
                            scene.transition,
                            _json(
                                [
                                    item.model_dump(mode="json")
                                    for item in scene.requirement_refs
                                ]
                            ),
                            int(changed),
                            now,
                            stored["id"],
                        ),
                    )
                else:
                    connection.execute(
                        """
                        INSERT INTO novel_scene_beats(
                            id, plan_id, position, pov_character, goal,
                            obstacle, action, reveal, conceal, subtext,
                            location, key_items_json, end_state, transition,
                            requirement_refs_json, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                  ?, ?, ?)
                        """,
                        (
                            uuid.uuid4().hex,
                            plan_id,
                            position,
                            scene.pov_character,
                            scene.goal,
                            scene.obstacle,
                            scene.action,
                            scene.reveal,
                            scene.conceal,
                            scene.subtext,
                            scene.location,
                            _json(scene.key_items),
                            scene.end_state,
                            scene.transition,
                            _json(
                                [
                                    item.model_dump(mode="json")
                                    for item in scene.requirement_refs
                                ]
                            ),
                            now,
                            now,
                        ),
                    )
            connection.execute(
                """
                UPDATE novel_scene_beats
                SET beat_status='retired',
                    draft_status=CASE
                        WHEN current_version_id IS NOT NULL THEN 'stale'
                        ELSE draft_status
                    END,
                    updated_at=?
                WHERE plan_id=? AND position>?
                """,
                (now, plan_id, len(card.scenes)),
            )
            connection.execute(
                """
                UPDATE novel_technique_bindings
                SET status='disabled', updated_at=?
                WHERE scene_beat_id IN (
                    SELECT id FROM novel_scene_beats
                    WHERE plan_id=? AND beat_status='retired'
                )
                """,
                (now, plan_id),
            )
            connection.execute(
                """
                UPDATE novel_chapters
                SET volume_id=?, outline=?, key_points=?, updated_at=?
                WHERE id=?
                """,
                (
                    volume_id,
                    card.purpose,
                    "\n".join(card.must_happen),
                    now,
                    chapter_id,
                ),
            )
            connection.execute(
                "UPDATE novel_projects SET updated_at=? WHERE id=?",
                (now, project_id),
            )
            connection.commit()
        return plan_id
