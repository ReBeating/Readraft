from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .db import Database, utc_now
from .reader_schema import ReaderBranchSet


REQUEST_TYPES = {
    "pace",
    "character",
    "relationship",
    "plot",
    "world",
    "payoff",
    "other",
}
REQUEST_SCOPES = {
    "next_chapter",
    "next_three",
    "current_volume",
    "long_term",
}
REQUEST_PRIORITIES = {"soft", "hard"}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _load_json(value: Any, fallback: Any) -> Any:
    try:
        return json.loads(str(value))
    except (TypeError, ValueError):
        return fallback


class ReaderDecisionService:
    def __init__(self, database: Database, novels_dir: Path):
        self.database = database
        self.novels_dir = novels_dir

    def create_request(
        self,
        *,
        user_id: int,
        project_id: str,
        raw_text: str,
        request_type: str,
        impact_scope: str,
        priority: str,
        constraints: List[str],
        author_note: str,
    ) -> str:
        if request_type not in REQUEST_TYPES:
            raise ValueError("不支持的读者意见类型")
        if impact_scope not in REQUEST_SCOPES:
            raise ValueError("不支持的影响范围")
        if priority not in REQUEST_PRIORITIES:
            raise ValueError("优先级只能是硬性或软性")
        request_id = uuid.uuid4().hex
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
            connection.execute(
                """
                INSERT INTO reader_requests(
                    id, project_id, raw_text, request_type, impact_scope,
                    priority, constraints_json, author_note, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'draft', ?, ?)
                """,
                (
                    request_id,
                    project_id,
                    raw_text,
                    request_type,
                    impact_scope,
                    priority,
                    _json(constraints),
                    author_note,
                    now,
                    now,
                ),
            )
            connection.execute(
                "UPDATE novel_projects SET updated_at=? WHERE id=?",
                (now, project_id),
            )
            connection.commit()
        return request_id

    def list_requests(
        self, *, user_id: int, project_id: str
    ) -> List[Dict[str, Any]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT r.*,
                       (SELECT COUNT(*) FROM reader_branch_proposals p
                        WHERE p.request_id=r.id
                          AND p.status IN ('candidate', 'accepted'))
                           AS proposal_count
                FROM reader_requests r
                JOIN novel_projects p ON p.id=r.project_id
                WHERE r.project_id=? AND p.user_id=?
                ORDER BY r.created_at DESC, r.rowid DESC
                """,
                (project_id, user_id),
            ).fetchall()
        return [self._decode_request(dict(row)) for row in rows]

    def get_request(
        self, *, user_id: int, request_id: str
    ) -> Optional[Dict[str, Any]]:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT r.*, p.title AS project_title,
                       p.planning_horizon,
                       (SELECT j.id FROM generation_jobs j
                        WHERE j.subject_id=r.id
                          AND j.operation='propose_reader_branches'
                        ORDER BY j.created_at DESC LIMIT 1) AS latest_job_id,
                       (SELECT j.status FROM generation_jobs j
                        WHERE j.subject_id=r.id
                          AND j.operation='propose_reader_branches'
                        ORDER BY j.created_at DESC LIMIT 1) AS latest_job_status
                FROM reader_requests r
                JOIN novel_projects p ON p.id=r.project_id
                WHERE r.id=? AND p.user_id=?
                """,
                (request_id, user_id),
            ).fetchone()
            if not row:
                return None
            proposals = connection.execute(
                """
                SELECT * FROM reader_branch_proposals
                WHERE request_id=?
                  AND status IN ('candidate', 'accepted', 'rejected')
                ORDER BY created_at DESC, position
                """,
                (request_id,),
            ).fetchall()
            applications = connection.execute(
                """
                SELECT a.*, ch.title AS chapter_title
                FROM reader_plan_applications a
                JOIN novel_chapters ch ON ch.id=a.chapter_id
                WHERE a.request_id=?
                ORDER BY a.chapter_position
                """,
                (request_id,),
            ).fetchall()
        result = self._decode_request(dict(row))
        result["proposals"] = [
            self._decode_proposal(dict(proposal))
            for proposal in proposals
        ]
        result["applications"] = [dict(item) for item in applications]
        return result

    def build_planning_context(
        self, *, user_id: int, request_id: str
    ) -> Optional[Dict[str, Any]]:
        with self.database.connection() as connection:
            request = connection.execute(
                """
                SELECT r.*, p.title, p.genre, p.premise, p.story_promise,
                       p.target_audience, p.core_appeal,
                       p.ending_constraint, p.world_setting,
                       p.point_of_view, p.planning_horizon,
                       p.canonical_branch_id
                FROM reader_requests r
                JOIN novel_projects p ON p.id=r.project_id
                WHERE r.id=? AND p.user_id=?
                """,
                (request_id, user_id),
            ).fetchone()
            if not request:
                return None
            current_row = connection.execute(
                """
                SELECT COALESCE(MAX(position), 0) AS position
                FROM novel_chapters
                WHERE project_id=? AND canonical_version_id IS NOT NULL
                """,
                (request["project_id"],),
            ).fetchone()
            current_position = int(current_row["position"])
            horizon = int(request["planning_horizon"] or 20)
            future = connection.execute(
                """
                SELECT ch.id, ch.position, ch.title, ch.outline,
                       ch.key_points, ch.volume_id, v.title AS volume_title,
                       cp.status AS task_card_status
                FROM novel_chapters ch
                LEFT JOIN novel_volumes v ON v.id=ch.volume_id
                LEFT JOIN novel_chapter_plans cp ON cp.chapter_id=ch.id
                WHERE ch.project_id=? AND ch.position>?
                ORDER BY ch.position
                LIMIT ?
                """,
                (request["project_id"], current_position, horizon),
            ).fetchall()
            characters = connection.execute(
                """
                SELECT name, role, traits, background, character_arc
                FROM novel_characters
                WHERE project_id=? ORDER BY position
                """,
                (request["project_id"],),
            ).fetchall()
            volumes = connection.execute(
                """
                SELECT position, title, goal, start_state, end_state,
                       major_conflict, payoff, status
                FROM novel_volumes
                WHERE project_id=? ORDER BY position
                """,
                (request["project_id"],),
            ).fetchall()
            story_blueprint = connection.execute(
                """
                SELECT v.central_question, v.protagonist_goal,
                       v.core_conflict, v.stakes, v.opening_state,
                       v.ending_state, v.major_turns_json,
                       v.must_payoffs_json, v.forbidden_shortcuts_json
                FROM novel_story_blueprint_heads h
                JOIN novel_story_blueprint_versions v
                    ON v.id=h.confirmed_version_id
                WHERE h.project_id=? AND v.version_status='confirmed'
                """,
                (request["project_id"],),
            ).fetchone()
            planned_arcs = connection.execute(
                """
                SELECT a.id, a.position, v.arc_type, v.title,
                       v.dramatic_question, v.promise, v.start_state,
                       v.target_payoff, v.involved_characters_json,
                       v.planned_turns_json, v.lifecycle_status,
                       v.priority
                FROM novel_plot_arcs a
                JOIN novel_plot_arc_versions v
                    ON v.id=a.confirmed_version_id
                WHERE a.project_id=? AND v.version_status='confirmed'
                    AND v.lifecycle_status!='abandoned'
                ORDER BY v.priority DESC, a.position
                """,
                (request["project_id"],),
            ).fetchall()
            recent_memory = connection.execute(
                """
                SELECT ch.position, ch.title, m.summary,
                       m.key_events_json, m.unresolved_questions_json,
                       m.keywords_json
                FROM chapter_memory m
                JOIN novel_chapters ch ON ch.id=m.chapter_id
                WHERE m.project_id=? AND m.branch_id=?
                    AND m.record_status='canon'
                ORDER BY ch.position DESC LIMIT 8
                """,
                (
                    request["project_id"],
                    request["canonical_branch_id"],
                ),
            ).fetchall()
            facts = connection.execute(
                """
                SELECT ch.position AS source_chapter_position,
                       f.fact_type, f.subject_name, f.predicate,
                       f.object_json, f.evidence
                FROM story_facts f
                JOIN novel_chapters ch ON ch.id=f.chapter_id
                WHERE f.project_id=? AND f.branch_id=?
                    AND f.fact_status='canon'
                ORDER BY ch.position DESC, f.created_at DESC LIMIT 120
                """,
                (
                    request["project_id"],
                    request["canonical_branch_id"],
                ),
            ).fetchall()
            knowledge = connection.execute(
                """
                SELECT ch.position AS source_chapter_position,
                       k.character_name, k.fact_text, k.knowledge_state,
                       k.learned_via, k.evidence
                FROM character_knowledge k
                JOIN novel_chapters ch ON ch.id=k.chapter_id
                WHERE k.project_id=? AND k.branch_id=?
                    AND k.record_status='canon'
                ORDER BY ch.position DESC, k.created_at DESC LIMIT 120
                """,
                (
                    request["project_id"],
                    request["canonical_branch_id"],
                ),
            ).fetchall()
            threads = connection.execute(
                """
                SELECT ch.position AS source_chapter_position,
                       t.thread_name, t.thread_type, t.action,
                       t.update_text, t.promise, t.target_payoff, t.evidence
                FROM plot_threads t
                JOIN novel_chapters ch ON ch.id=t.chapter_id
                WHERE t.project_id=? AND t.branch_id=?
                    AND t.record_status='canon'
                ORDER BY ch.position DESC, t.created_at DESC LIMIT 80
                """,
                (
                    request["project_id"],
                    request["canonical_branch_id"],
                ),
            ).fetchall()
            hooks = connection.execute(
                """
                SELECT ch.position AS source_chapter_position,
                       f.hook_name, f.action, f.description,
                       f.intended_payoff, f.evidence
                FROM foreshadowing f
                JOIN novel_chapters ch ON ch.id=f.chapter_id
                WHERE f.project_id=? AND f.branch_id=?
                    AND f.record_status='canon'
                ORDER BY ch.position DESC, f.created_at DESC LIMIT 80
                """,
                (
                    request["project_id"],
                    request["canonical_branch_id"],
                ),
            ).fetchall()
        memory_items = []
        for row in recent_memory:
            item = dict(row)
            item["key_events"] = _load_json(
                item.pop("key_events_json"), []
            )
            item["unresolved_questions"] = _load_json(
                item.pop("unresolved_questions_json"), []
            )
            item["keywords"] = _load_json(item.pop("keywords_json"), [])
            memory_items.append(item)
        fact_items = []
        for row in facts:
            item = dict(row)
            item["object"] = _load_json(item.pop("object_json"), {})
            fact_items.append(item)
        blueprint_item = dict(story_blueprint) if story_blueprint else {}
        if blueprint_item:
            blueprint_item["major_turns"] = _load_json(
                blueprint_item.pop("major_turns_json"), []
            )
            blueprint_item["must_payoffs"] = _load_json(
                blueprint_item.pop("must_payoffs_json"), []
            )
            blueprint_item["forbidden_shortcuts"] = _load_json(
                blueprint_item.pop("forbidden_shortcuts_json"), []
            )
        planned_arc_items = []
        for row in planned_arcs:
            item = dict(row)
            item["involved_characters"] = _load_json(
                item.pop("involved_characters_json"), []
            )
            item["planned_turns"] = _load_json(
                item.pop("planned_turns_json"), []
            )
            planned_arc_items.append(item)
        request_data = self._decode_request(dict(request))
        return {
            "request": request_data,
            "project": {
                key: request[key]
                for key in (
                    "project_id",
                    "title",
                    "genre",
                    "premise",
                    "story_promise",
                    "target_audience",
                    "core_appeal",
                    "ending_constraint",
                    "world_setting",
                    "point_of_view",
                )
            },
            "current_position": current_position,
            "planning_horizon": horizon,
            "future_chapters": [dict(row) for row in future],
            "characters": [dict(row) for row in characters],
            "volumes": [dict(row) for row in volumes],
            "confirmed_story_blueprint": blueprint_item,
            "planned_plot_arcs": planned_arc_items,
            "canonical_memory": {
                "recent_chapters": memory_items,
                "story_facts": fact_items,
                "character_knowledge": [
                    dict(row) for row in knowledge
                ],
                "plot_threads": [dict(row) for row in threads],
                "foreshadowing": [dict(row) for row in hooks],
            },
            "open_plot_threads": [dict(row) for row in threads],
            "open_foreshadowing": [dict(row) for row in hooks],
        }

    def save_proposals(
        self,
        *,
        user_id: int,
        request_id: str,
        job_id: str,
        result: ReaderBranchSet,
        provider: str,
        model: str,
    ) -> List[str]:
        context = self.build_planning_context(
            user_id=user_id, request_id=request_id
        )
        if not context:
            raise ValueError("读者意见不存在")
        result.ensure_applicable(
            current_position=int(context["current_position"]),
            planning_horizon=int(context["planning_horizon"]),
        )
        now = utc_now()
        proposal_ids: List[str] = []
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            request = connection.execute(
                """
                SELECT r.id, r.project_id
                FROM reader_requests r
                JOIN novel_projects p ON p.id=r.project_id
                WHERE r.id=? AND p.user_id=?
                  AND r.status IN ('draft', 'proposing', 'reviewing')
                """,
                (request_id, user_id),
            ).fetchone()
            job = connection.execute(
                """
                SELECT id FROM generation_jobs
                WHERE id=? AND user_id=? AND subject_id=?
                  AND operation='propose_reader_branches'
                  AND status='running'
                """,
                (job_id, user_id, request_id),
            ).fetchone()
            if not request or not job:
                connection.rollback()
                raise ValueError("读者意见规划任务已经失效")
            connection.execute(
                """
                UPDATE reader_branch_proposals
                SET status='superseded', decided_at=?
                WHERE request_id=? AND status='candidate'
                """,
                (now, request_id),
            )
            for position, proposal in enumerate(
                result.alternatives, start=1
            ):
                proposal_id = uuid.uuid4().hex
                proposal_ids.append(proposal_id)
                payload = proposal.model_dump(mode="json")
                connection.execute(
                    """
                    INSERT INTO reader_branch_proposals(
                        id, request_id, project_id, generation_job_id,
                        position, label, summary, satisfies_json,
                        sacrifices_json, affected_characters_json,
                        affected_plot_threads_json, promise_impact,
                        future_changes_json, affects_published_canon,
                        published_canon_impact, risk_level, risks_json,
                        status, provider, model, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                              ?, ?, ?, 'candidate', ?, ?, ?)
                    """,
                    (
                        proposal_id,
                        request_id,
                        request["project_id"],
                        job_id,
                        position,
                        payload["label"],
                        payload["summary"],
                        _json(payload["satisfies"]),
                        _json(payload["sacrifices"]),
                        _json(payload["affected_characters"]),
                        _json(payload["affected_plot_threads"]),
                        payload["promise_impact"],
                        _json(payload["future_changes"]),
                        int(payload["affects_published_canon"]),
                        payload["published_canon_impact"],
                        payload["risk_level"],
                        _json(payload["risks"]),
                        provider,
                        model,
                        now,
                    ),
                )
            connection.execute(
                """
                UPDATE reader_requests
                SET status='reviewing', updated_at=?
                WHERE id=?
                """,
                (now, request_id),
            )
            connection.commit()
        return proposal_ids

    def accept_proposal(
        self, *, user_id: int, proposal_id: str
    ) -> Optional[Dict[str, Any]]:
        now = utc_now()
        created_directories: List[Path] = []
        with self.database.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                proposal = connection.execute(
                    """
                    SELECT bp.*, r.status AS request_status,
                           p.planning_horizon
                    FROM reader_branch_proposals bp
                    JOIN reader_requests r ON r.id=bp.request_id
                    JOIN novel_projects p ON p.id=bp.project_id
                    WHERE bp.id=? AND p.user_id=?
                    """,
                    (proposal_id, user_id),
                ).fetchone()
                if not proposal:
                    connection.rollback()
                    return None
                if (
                    str(proposal["status"]) != "candidate"
                    or str(proposal["request_status"]) != "reviewing"
                ):
                    raise ValueError("这个剧情方案已经处理或失效")
                if bool(proposal["affects_published_canon"]):
                    raise ValueError(
                        "这个方案要求回改已发表正史，不能直接更新未来细纲；"
                        "请先通过旧章影响报告处理正史修改"
                    )
                current = connection.execute(
                    """
                    SELECT COALESCE(MAX(position), 0) AS position
                    FROM novel_chapters
                    WHERE project_id=? AND canonical_version_id IS NOT NULL
                    """,
                    (proposal["project_id"],),
                ).fetchone()
                current_position = int(current["position"])
                maximum_position = current_position + int(
                    proposal["planning_horizon"] or 20
                )
                changes = _load_json(
                    proposal["future_changes_json"], []
                )
                if not isinstance(changes, list) or not changes:
                    raise ValueError("剧情方案没有可应用的未来章节变化")
                positions: set[int] = set()
                applied: List[Dict[str, Any]] = []
                for change in changes:
                    if not isinstance(change, Mapping):
                        raise ValueError("剧情方案章节变化结构不正确")
                    position = int(change.get("chapter_position") or 0)
                    if position in positions:
                        raise ValueError("剧情方案重复修改了同一章")
                    positions.add(position)
                    if not (
                        current_position < position <= maximum_position
                    ):
                        raise ValueError(
                            "剧情方案已经超出当前滚动规划窗口，请重新生成"
                        )
                    title = str(change.get("title") or "").strip()[:120]
                    outline = str(change.get("outline") or "").strip()[:6000]
                    key_points = [
                        str(item).strip()[:400]
                        for item in change.get("key_points") or []
                        if str(item).strip()
                    ][:20]
                    if not title or not outline or not key_points:
                        raise ValueError("未来章节变化缺少标题、大纲或关键点")
                    chapter = connection.execute(
                        """
                        SELECT id, title, outline, key_points,
                               canonical_version_id
                        FROM novel_chapters
                        WHERE project_id=? AND position=?
                        """,
                        (proposal["project_id"], position),
                    ).fetchone()
                    if chapter and chapter["canonical_version_id"]:
                        raise ValueError("方案试图修改已经确认的正史章节")
                    action = "revised"
                    if chapter:
                        chapter_id = str(chapter["id"])
                        active = connection.execute(
                            """
                            SELECT 1 FROM generation_jobs
                            WHERE chapter_id=?
                              AND status IN ('queued', 'running')
                            LIMIT 1
                            """,
                            (chapter_id,),
                        ).fetchone()
                        if active:
                            raise ValueError(
                                f"第 {position} 章有 AI 任务正在运行，"
                                "请完成后再采纳方案"
                            )
                        before = {
                            "title": chapter["title"],
                            "outline": chapter["outline"],
                            "key_points": chapter["key_points"],
                        }
                        connection.execute(
                            """
                            UPDATE novel_chapters
                            SET title=?, outline=?, key_points=?, updated_at=?
                            WHERE id=?
                            """,
                            (
                                title,
                                outline,
                                "\n".join(key_points),
                                now,
                                chapter_id,
                            ),
                        )
                        connection.execute(
                            """
                            UPDATE novel_chapter_plans
                            SET status='draft', source='reader_request',
                                updated_at=?
                            WHERE chapter_id=?
                            """,
                            (now, chapter_id),
                        )
                    else:
                        action = "created"
                        chapter_id = uuid.uuid4().hex
                        chapter_dir = (
                            self.novels_dir
                            / str(user_id)
                            / str(proposal["project_id"])
                            / "chapters"
                            / chapter_id
                        )
                        chapter_dir.mkdir(
                            parents=True, exist_ok=False, mode=0o700
                        )
                        created_directories.append(chapter_dir)
                        content_path = chapter_dir / "content.txt"
                        content_path.write_text("", encoding="utf-8")
                        content_path.chmod(0o600)
                        before = {}
                        connection.execute(
                            """
                            INSERT INTO novel_chapters(
                                id, project_id, position, title, outline,
                                key_points, status, content_path, char_count,
                                created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, 'planned', ?, 0, ?, ?)
                            """,
                            (
                                chapter_id,
                                proposal["project_id"],
                                position,
                                title,
                                outline,
                                "\n".join(key_points),
                                str(content_path),
                                now,
                                now,
                            ),
                        )
                    after = {
                        "title": title,
                        "outline": outline,
                        "key_points": key_points,
                        "rationale": str(
                            change.get("rationale") or ""
                        )[:1200],
                    }
                    connection.execute(
                        """
                        INSERT INTO reader_plan_applications(
                            id, request_id, proposal_id, chapter_id,
                            chapter_position, action, before_json,
                            after_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            uuid.uuid4().hex,
                            proposal["request_id"],
                            proposal_id,
                            chapter_id,
                            position,
                            action,
                            _json(before),
                            _json(after),
                            now,
                        ),
                    )
                    applied.append(
                        {
                            "chapter_id": chapter_id,
                            "position": position,
                            "action": action,
                            **after,
                        }
                    )
                connection.execute(
                    """
                    UPDATE reader_branch_proposals
                    SET status=CASE WHEN id=? THEN 'accepted'
                                    ELSE 'rejected' END,
                        decided_at=?
                    WHERE request_id=? AND status='candidate'
                    """,
                    (proposal_id, now, proposal["request_id"]),
                )
                connection.execute(
                    """
                    UPDATE reader_requests
                    SET status='adopted', chosen_proposal_id=?,
                        updated_at=?, decided_at=?
                    WHERE id=?
                    """,
                    (
                        proposal_id,
                        now,
                        now,
                        proposal["request_id"],
                    ),
                )
                connection.execute(
                    "UPDATE novel_projects SET updated_at=? WHERE id=?",
                    (now, proposal["project_id"]),
                )
                connection.commit()
                return {
                    "request_id": str(proposal["request_id"]),
                    "project_id": str(proposal["project_id"]),
                    "proposal_id": proposal_id,
                    "applied": applied,
                }
            except Exception:
                connection.rollback()
                for directory in reversed(created_directories):
                    shutil.rmtree(directory, ignore_errors=True)
                raise

    def dismiss_request(
        self, *, user_id: int, request_id: str
    ) -> Optional[str]:
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            request = connection.execute(
                """
                SELECT r.project_id
                FROM reader_requests r
                JOIN novel_projects p ON p.id=r.project_id
                WHERE r.id=? AND p.user_id=?
                """,
                (request_id, user_id),
            ).fetchone()
            if not request:
                connection.rollback()
                return None
            active = connection.execute(
                """
                SELECT 1 FROM generation_jobs
                WHERE subject_id=? AND operation='propose_reader_branches'
                  AND status IN ('queued', 'running')
                """,
                (request_id,),
            ).fetchone()
            if active:
                raise ValueError("剧情方案仍在生成，暂时不能归档这条意见")
            connection.execute(
                """
                UPDATE reader_requests
                SET status='dismissed', updated_at=?, decided_at=?
                WHERE id=?
                """,
                (now, now, request_id),
            )
            connection.execute(
                """
                UPDATE reader_branch_proposals
                SET status='rejected', decided_at=?
                WHERE request_id=? AND status='candidate'
                """,
                (now, request_id),
            )
            connection.commit()
        return str(request["project_id"])

    @staticmethod
    def _decode_request(row: Dict[str, Any]) -> Dict[str, Any]:
        row["constraints"] = _load_json(
            row.pop("constraints_json", "[]"), []
        )
        return row

    @staticmethod
    def _decode_proposal(row: Dict[str, Any]) -> Dict[str, Any]:
        for stored, public in (
            ("satisfies_json", "satisfies"),
            ("sacrifices_json", "sacrifices"),
            ("affected_characters_json", "affected_characters"),
            ("affected_plot_threads_json", "affected_plot_threads"),
            ("future_changes_json", "future_changes"),
            ("risks_json", "risks"),
        ):
            row[public] = _load_json(row.pop(stored, "[]"), [])
        row["affects_published_canon"] = bool(
            row["affects_published_canon"]
        )
        return row
