from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any, Iterable, Mapping, Optional

from .analysis_schema import ChapterAnalysis
from .db import Database, utc_now
from .technique_schema import TechniqueObservation


ALLOWED_USAGE_MODES = {"plan", "write", "audit"}
ALLOWED_STATUSES = {"active", "archived"}
ALLOWED_BINDING_STATUSES = {"enabled", "disabled"}


def _json_list(value: Any) -> list[str]:
    try:
        parsed = json.loads(str(value or "[]"))
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item).strip()]


def _dump_list(value: Iterable[str]) -> str:
    return json.dumps(list(value), ensure_ascii=False, separators=(",", ":"))


def _card_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    card = dict(row)
    card["suitable_for"] = _json_list(
        card.pop("suitable_for_json", "[]")
    )
    card["unsuitable_for"] = _json_list(
        card.pop("unsuitable_for_json", "[]")
    )
    return card


def _binding_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    binding = dict(row)
    binding["usage_modes"] = _json_list(
        binding.pop("usage_modes_json", "[]")
    )
    return binding


class TechniqueService:
    def __init__(self, database: Database):
        self.database = database

    def count_cards(self, *, user_id: int) -> int:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM reference_technique_cards
                WHERE user_id=? AND status='active'
                """,
                (user_id,),
            ).fetchone()
        return int(row["count"] or 0)

    def create_from_analysis(
        self,
        *,
        user_id: int,
        analysis_id: str,
        technique_index: int,
    ) -> tuple[str, bool]:
        analysis = self.database.get_analysis(user_id, analysis_id)
        if not analysis or str(analysis.get("status")) != "completed":
            raise ValueError("拆文分析不存在或尚未完成")
        try:
            result = ChapterAnalysis.model_validate_json(
                str(analysis.get("result_json") or "{}")
            )
        except ValueError as exc:
            raise ValueError("拆文结果结构不完整，请重新分析") from exc
        if technique_index < 0 or technique_index >= len(result.techniques):
            raise ValueError("技法建议不存在")
        observation = result.techniques[technique_index]
        card_id = uuid.uuid4().hex
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            owner = connection.execute(
                """
                SELECT a.id, a.chapter_id, j.document_id
                FROM chapter_analyses a
                JOIN analysis_jobs j ON j.id=a.job_id
                WHERE a.id=? AND j.user_id=? AND a.status='completed'
                """,
                (analysis_id, user_id),
            ).fetchone()
            if not owner:
                connection.rollback()
                raise ValueError("拆文分析不存在或尚未完成")
            try:
                connection.execute(
                    """
                    INSERT INTO reference_technique_cards(
                        id, user_id, source_document_id, source_chapter_id,
                        source_analysis_id, name, dimension, source_location,
                        observation, effect, suitable_for_json,
                        unsuitable_for_json, execution_rule,
                        originality_boundary, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                              'active', ?, ?)
                    """,
                    (
                        card_id,
                        user_id,
                        owner["document_id"],
                        owner["chapter_id"],
                        analysis_id,
                        observation.name,
                        observation.dimension,
                        observation.source_location,
                        observation.observation,
                        observation.effect,
                        _dump_list(observation.suitable_for),
                        _dump_list(observation.unsuitable_for),
                        observation.execution_rule,
                        observation.originality_boundary,
                        now,
                        now,
                    ),
                )
                connection.commit()
                return card_id, True
            except sqlite3.IntegrityError:
                existing = connection.execute(
                    """
                    SELECT id FROM reference_technique_cards
                    WHERE user_id=? AND source_analysis_id=? AND name=?
                    """,
                    (user_id, analysis_id, observation.name),
                ).fetchone()
                connection.rollback()
                if not existing:
                    raise
                return str(existing["id"]), False

    def create_manual(
        self,
        *,
        user_id: int,
        observation: TechniqueObservation,
        author_note: str = "",
    ) -> str:
        card_id = uuid.uuid4().hex
        now = utc_now()
        with self.database.connection() as connection:
            owner = connection.execute(
                "SELECT 1 FROM users WHERE id=?", (user_id,)
            ).fetchone()
            if not owner:
                raise ValueError("用户不存在")
            connection.execute(
                """
                INSERT INTO reference_technique_cards(
                    id, user_id, name, dimension, source_location,
                    observation, effect, suitable_for_json,
                    unsuitable_for_json, execution_rule,
                    originality_boundary, author_note, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
                """,
                (
                    card_id,
                    user_id,
                    observation.name,
                    observation.dimension,
                    observation.source_location,
                    observation.observation,
                    observation.effect,
                    _dump_list(observation.suitable_for),
                    _dump_list(observation.unsuitable_for),
                    observation.execution_rule,
                    observation.originality_boundary,
                    author_note,
                    now,
                    now,
                ),
            )
            connection.commit()
        return card_id

    def list_cards(self, *, user_id: int) -> list[dict[str, Any]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT tc.*, d.title AS source_document_title,
                       c.title AS source_chapter_title,
                       c.position AS source_chapter_position,
                       (
                           SELECT COUNT(*) FROM novel_technique_bindings b
                           WHERE b.technique_id=tc.id
                       ) AS binding_count,
                       (
                           SELECT COUNT(*) FROM novel_technique_bindings b
                           WHERE b.technique_id=tc.id AND b.status='enabled'
                       ) AS enabled_binding_count
                FROM reference_technique_cards tc
                LEFT JOIN documents d ON d.id=tc.source_document_id
                LEFT JOIN chapters c ON c.id=tc.source_chapter_id
                WHERE tc.user_id=?
                ORDER BY CASE tc.status WHEN 'active' THEN 0 ELSE 1 END,
                         tc.updated_at DESC, tc.rowid DESC
                """,
                (user_id,),
            ).fetchall()
        return [_card_from_row(row) for row in rows]

    def list_saved_names_for_analysis(
        self, *, user_id: int, analysis_id: str
    ) -> set[str]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT name FROM reference_technique_cards
                WHERE user_id=? AND source_analysis_id=?
                """,
                (user_id, analysis_id),
            ).fetchall()
        return {str(row["name"]) for row in rows}

    def get_card(
        self, *, user_id: int, technique_id: str
    ) -> Optional[dict[str, Any]]:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT tc.*, d.title AS source_document_title,
                       c.title AS source_chapter_title,
                       c.position AS source_chapter_position
                FROM reference_technique_cards tc
                LEFT JOIN documents d ON d.id=tc.source_document_id
                LEFT JOIN chapters c ON c.id=tc.source_chapter_id
                WHERE tc.id=? AND tc.user_id=?
                """,
                (technique_id, user_id),
            ).fetchone()
            if not row:
                return None
            bindings = connection.execute(
                """
                SELECT b.*, p.title AS project_title,
                       v.position AS volume_position,
                       v.title AS volume_title,
                       ch.position AS chapter_position,
                       ch.title AS chapter_title,
                       sb.position AS scene_position,
                       sb.goal AS scene_goal,
                       scene_ch.position AS scene_chapter_position,
                       scene_ch.title AS scene_chapter_title
                FROM novel_technique_bindings b
                JOIN novel_projects p ON p.id=b.project_id
                LEFT JOIN novel_volumes v ON v.id=b.volume_id
                LEFT JOIN novel_chapters ch ON ch.id=b.chapter_id
                LEFT JOIN novel_scene_beats sb ON sb.id=b.scene_beat_id
                LEFT JOIN novel_chapter_plans cp ON cp.id=sb.plan_id
                LEFT JOIN novel_chapters scene_ch ON scene_ch.id=cp.chapter_id
                WHERE b.technique_id=? AND p.user_id=?
                ORDER BY b.status, b.priority DESC, b.created_at
                """,
                (technique_id, user_id),
            ).fetchall()
        result = _card_from_row(row)
        result["bindings"] = [_binding_from_row(item) for item in bindings]
        return result

    def update_card(
        self,
        *,
        user_id: int,
        technique_id: str,
        observation: TechniqueObservation,
        author_note: str,
        status: str,
    ) -> bool:
        if status not in ALLOWED_STATUSES:
            raise ValueError("不支持的技法卡状态")
        now = utc_now()
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE reference_technique_cards
                SET name=?, dimension=?, source_location=?, observation=?,
                    effect=?, suitable_for_json=?, unsuitable_for_json=?,
                    execution_rule=?, originality_boundary=?,
                    author_note=?, status=?, updated_at=?
                WHERE id=? AND user_id=?
                """,
                (
                    observation.name,
                    observation.dimension,
                    observation.source_location,
                    observation.observation,
                    observation.effect,
                    _dump_list(observation.suitable_for),
                    _dump_list(observation.unsuitable_for),
                    observation.execution_rule,
                    observation.originality_boundary,
                    author_note,
                    status,
                    now,
                    technique_id,
                    user_id,
                ),
            )
            if cursor.rowcount and status == "archived":
                connection.execute(
                    """
                    UPDATE novel_technique_bindings
                    SET status='disabled', updated_at=?
                    WHERE technique_id=?
                    """,
                    (now, technique_id),
                )
            connection.commit()
        return cursor.rowcount == 1

    def binding_targets(self, *, user_id: int) -> list[dict[str, Any]]:
        with self.database.connection() as connection:
            projects = connection.execute(
                """
                SELECT id, title FROM novel_projects
                WHERE user_id=? ORDER BY updated_at DESC
                """,
                (user_id,),
            ).fetchall()
            volumes = connection.execute(
                """
                SELECT v.id, v.project_id, v.position, v.title,
                       p.title AS project_title
                FROM novel_volumes v
                JOIN novel_projects p ON p.id=v.project_id
                WHERE p.user_id=? ORDER BY p.title, v.position
                """,
                (user_id,),
            ).fetchall()
            chapters = connection.execute(
                """
                SELECT ch.id, ch.project_id, ch.position, ch.title,
                       p.title AS project_title
                FROM novel_chapters ch
                JOIN novel_projects p ON p.id=ch.project_id
                WHERE p.user_id=? ORDER BY p.title, ch.position
                """,
                (user_id,),
            ).fetchall()
            scenes = connection.execute(
                """
                SELECT sb.id, cp.project_id, sb.position, sb.goal,
                       ch.position AS chapter_position,
                       ch.title AS chapter_title,
                       p.title AS project_title
                FROM novel_scene_beats sb
                JOIN novel_chapter_plans cp ON cp.id=sb.plan_id
                JOIN novel_chapters ch ON ch.id=cp.chapter_id
                JOIN novel_projects p ON p.id=cp.project_id
                WHERE p.user_id=? AND sb.beat_status='active'
                ORDER BY p.title, ch.position, sb.position
                """,
                (user_id,),
            ).fetchall()
        targets: list[dict[str, Any]] = []
        for row in projects:
            targets.append(
                {
                    "value": f"project:{row['id']}",
                    "project_id": str(row["id"]),
                    "label": f"《{row['title']}》· 全书",
                }
            )
        for row in volumes:
            targets.append(
                {
                    "value": f"volume:{row['id']}",
                    "project_id": str(row["project_id"]),
                    "label": (
                        f"《{row['project_title']}》· 第 {row['position']} 卷"
                        f"《{row['title']}》"
                    ),
                }
            )
        for row in chapters:
            targets.append(
                {
                    "value": f"chapter:{row['id']}",
                    "project_id": str(row["project_id"]),
                    "label": (
                        f"《{row['project_title']}》· 第 {row['position']} 章"
                        f"《{row['title']}》"
                    ),
                }
            )
        for row in scenes:
            targets.append(
                {
                    "value": f"scene:{row['id']}",
                    "project_id": str(row["project_id"]),
                    "label": (
                        f"《{row['project_title']}》· 第 "
                        f"{row['chapter_position']} 章 / 场景 "
                        f"{row['position']}：{str(row['goal'])[:60]}"
                    ),
                }
            )
        return targets

    def bind(
        self,
        *,
        user_id: int,
        technique_id: str,
        target: str,
        usage_modes: Iterable[str],
        author_adaptation: str,
        priority: int,
    ) -> str:
        modes = list(dict.fromkeys(str(item) for item in usage_modes))
        if not modes or any(item not in ALLOWED_USAGE_MODES for item in modes):
            raise ValueError("至少选择一个有效的使用阶段")
        if priority < 0 or priority > 100:
            raise ValueError("技法优先级必须在 0–100 之间")
        try:
            scope_type, target_id = target.split(":", 1)
        except ValueError as exc:
            raise ValueError("技法应用范围不正确") from exc
        if scope_type not in {"project", "volume", "chapter", "scene"}:
            raise ValueError("技法应用范围不正确")
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            card = connection.execute(
                """
                SELECT id, status FROM reference_technique_cards
                WHERE id=? AND user_id=?
                """,
                (technique_id, user_id),
            ).fetchone()
            if not card:
                connection.rollback()
                raise ValueError("技法卡不存在")
            if str(card["status"]) != "active":
                connection.rollback()
                raise ValueError("归档技法卡不能启用")

            volume_id = chapter_id = scene_beat_id = None
            if scope_type == "project":
                project = connection.execute(
                    "SELECT id FROM novel_projects WHERE id=? AND user_id=?",
                    (target_id, user_id),
                ).fetchone()
            elif scope_type == "volume":
                project = connection.execute(
                    """
                    SELECT p.id FROM novel_volumes v
                    JOIN novel_projects p ON p.id=v.project_id
                    WHERE v.id=? AND p.user_id=?
                    """,
                    (target_id, user_id),
                ).fetchone()
                volume_id = target_id
            elif scope_type == "chapter":
                project = connection.execute(
                    """
                    SELECT p.id FROM novel_chapters ch
                    JOIN novel_projects p ON p.id=ch.project_id
                    WHERE ch.id=? AND p.user_id=?
                    """,
                    (target_id, user_id),
                ).fetchone()
                chapter_id = target_id
            else:
                project = connection.execute(
                    """
                    SELECT p.id FROM novel_scene_beats sb
                    JOIN novel_chapter_plans cp ON cp.id=sb.plan_id
                    JOIN novel_projects p ON p.id=cp.project_id
                    WHERE sb.id=? AND p.user_id=?
                        AND sb.beat_status='active'
                    """,
                    (target_id, user_id),
                ).fetchone()
                scene_beat_id = target_id
            if not project:
                connection.rollback()
                raise ValueError("技法应用目标不存在")
            project_id = str(project["id"])
            existing = connection.execute(
                """
                SELECT id FROM novel_technique_bindings
                WHERE technique_id=? AND project_id=? AND scope_type=?
                  AND COALESCE(volume_id, '')=COALESCE(?, '')
                  AND COALESCE(chapter_id, '')=COALESCE(?, '')
                  AND COALESCE(scene_beat_id, '')=COALESCE(?, '')
                LIMIT 1
                """,
                (
                    technique_id,
                    project_id,
                    scope_type,
                    volume_id,
                    chapter_id,
                    scene_beat_id,
                ),
            ).fetchone()
            if existing:
                binding_id = str(existing["id"])
                connection.execute(
                    """
                    UPDATE novel_technique_bindings
                    SET usage_modes_json=?, author_adaptation=?, priority=?,
                        status='enabled', updated_at=?
                    WHERE id=?
                    """,
                    (
                        _dump_list(modes),
                        author_adaptation,
                        priority,
                        now,
                        binding_id,
                    ),
                )
            else:
                binding_id = uuid.uuid4().hex
                connection.execute(
                    """
                    INSERT INTO novel_technique_bindings(
                        id, technique_id, project_id, scope_type, volume_id,
                        chapter_id, scene_beat_id, usage_modes_json,
                        author_adaptation, priority, status,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'enabled', ?, ?)
                    """,
                    (
                        binding_id,
                        technique_id,
                        project_id,
                        scope_type,
                        volume_id,
                        chapter_id,
                        scene_beat_id,
                        _dump_list(modes),
                        author_adaptation,
                        priority,
                        now,
                        now,
                    ),
                )
            connection.execute(
                "UPDATE novel_projects SET updated_at=? WHERE id=?",
                (now, project_id),
            )
            connection.commit()
        return binding_id

    def set_binding_status(
        self,
        *,
        user_id: int,
        binding_id: str,
        status: str,
    ) -> Optional[str]:
        if status not in ALLOWED_BINDING_STATUSES:
            raise ValueError("不支持的绑定状态")
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT b.technique_id
                FROM novel_technique_bindings b
                JOIN reference_technique_cards tc ON tc.id=b.technique_id
                WHERE b.id=? AND tc.user_id=?
                """,
                (binding_id, user_id),
            ).fetchone()
            if not row:
                connection.rollback()
                return None
            if status == "enabled":
                card = connection.execute(
                    """
                    SELECT status FROM reference_technique_cards WHERE id=?
                    """,
                    (row["technique_id"],),
                ).fetchone()
                if not card or str(card["status"]) != "active":
                    connection.rollback()
                    raise ValueError("归档技法卡不能启用")
            connection.execute(
                """
                UPDATE novel_technique_bindings
                SET status=?, updated_at=? WHERE id=?
                """,
                (status, now, binding_id),
            )
            connection.commit()
        return str(row["technique_id"])

    def list_project_bindings(
        self, *, user_id: int, project_id: str
    ) -> list[dict[str, Any]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT b.*, tc.name, tc.dimension, tc.execution_rule,
                       tc.originality_boundary, tc.status AS technique_status,
                       v.position AS volume_position,
                       v.title AS volume_title,
                       ch.position AS chapter_position,
                       ch.title AS chapter_title,
                       sb.position AS scene_position,
                       sb.goal AS scene_goal,
                       scene_ch.position AS scene_chapter_position
                FROM novel_technique_bindings b
                JOIN reference_technique_cards tc ON tc.id=b.technique_id
                JOIN novel_projects p ON p.id=b.project_id
                LEFT JOIN novel_volumes v ON v.id=b.volume_id
                LEFT JOIN novel_chapters ch ON ch.id=b.chapter_id
                LEFT JOIN novel_scene_beats sb ON sb.id=b.scene_beat_id
                LEFT JOIN novel_chapter_plans cp ON cp.id=sb.plan_id
                LEFT JOIN novel_chapters scene_ch ON scene_ch.id=cp.chapter_id
                WHERE b.project_id=? AND p.user_id=?
                ORDER BY b.status, b.priority DESC, b.created_at
                """,
                (project_id, user_id),
            ).fetchall()
        return [_binding_from_row(row) for row in rows]
