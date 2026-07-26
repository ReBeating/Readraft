from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Dict, List, Mapping, Optional

from .causal_suggestion_schema import CausalSuggestionSet
from .continuity import get_continuity_context
from .db import Database, utc_after, utc_now
from .story_plan_suggestion_service import StoryPlanSuggestionService
from .structure_link_schema import CAUSAL_RELATION_LABELS
from .structure_link_service import StructureLinkService


CAUSAL_EXPLANATION_COMPATIBILITY_LABELS = {
    "exclusive": "互斥解释",
    "can_coexist": "可以共同成立",
    "uncertain": "现有证据无法判断",
}
CAUSAL_SEMANTIC_CATEGORY_LABELS = {
    "canon_consistency": "正史事实",
    "character_knowledge": "人物知情",
    "timeline": "时间与先后",
    "world_rules": "世界规则",
    "continuity": "连续性账本",
}
CAUSAL_SEMANTIC_STATUS_LABELS = {
    "supported": "有证据支持",
    "uncertain": "证据不足",
    "conflict": "发现冲突",
}
CAUSAL_ARC_IMPACT_LABELS = {
    "advances": "推进",
    "complicates": "复杂化",
    "delays": "延迟",
    "pays_off": "兑现",
    "risks_breaking": "可能破坏",
}


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _load_json(value: Any, fallback: Any) -> Any:
    try:
        return json.loads(str(value))
    except (TypeError, ValueError):
        return fallback


def _fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _evidence_value(value: Any, *, limit: int = 1400) -> str:
    if isinstance(value, (dict, list)):
        text = _json(value)
    else:
        text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _build_evidence_catalog(
    context: Mapping[str, Any],
    *,
    max_items: int = 260,
) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    seen: set[str] = set()

    def add(
        evidence_id: str,
        kind: str,
        label: str,
        value: Any,
    ) -> None:
        clean_id = str(evidence_id).strip()
        clean_value = _evidence_value(value)
        if (
            not clean_id
            or not clean_value
            or clean_id in seen
            or len(items) >= max_items
        ):
            return
        seen.add(clean_id)
        items.append(
            {
                "id": clean_id,
                "kind": kind,
                "label": str(label).strip()[:240],
                "value": clean_value,
            }
        )

    project = dict(context.get("project") or {})
    for key, label in (
        ("premise", "项目梗概"),
        ("story_promise", "作品承诺"),
        ("ending_constraint", "结局约束"),
        ("world_setting", "世界规则"),
        ("point_of_view", "叙事视角"),
    ):
        add(f"project:{key}", "project", label, project.get(key))

    blueprint = dict(context.get("confirmed_story_blueprint") or {})
    for key, label in (
        ("central_question", "全书核心悬问"),
        ("protagonist_goal", "主角长期目标"),
        ("core_conflict", "全书冲突引擎"),
        ("stakes", "长期代价"),
        ("ending_state", "终局状态"),
        ("forbidden_shortcuts", "禁止捷径"),
    ):
        add(f"blueprint:{key}", "blueprint", label, blueprint.get(key))

    for index, payoff in enumerate(
        blueprint.get("must_payoffs") or [],
        start=1,
    ):
        add(
            f"blueprint:must-payoff:{index}",
            "blueprint_payoff",
            f"全书必须兑现 {index}",
            payoff,
        )

    for index, character in enumerate(
        context.get("characters") or [],
        start=1,
    ):
        add(
            f"character:{index}:profile",
            "character_profile",
            f"人物《{character.get('name') or ''}》",
            {
                "name": character.get("name"),
                "role": character.get("role"),
                "traits": character.get("traits"),
                "background": character.get("background"),
                "character_arc": character.get("character_arc"),
            },
        )

    for arc in context.get("confirmed_planned_plot_arcs") or []:
        arc_id = str(arc.get("id") or "")
        add(
            f"arc:{arc_id}:summary",
            "planned_arc",
            f"规划剧情线《{arc.get('title') or ''}》",
            {
                "title": arc.get("title"),
                "type": arc.get("arc_type"),
                "lifecycle_status": arc.get("lifecycle_status"),
                "promise": arc.get("promise"),
                "start_state": arc.get("start_state"),
                "target_payoff": arc.get("target_payoff"),
                "planned_turns": arc.get("planned_turns") or [],
                "involved_characters": (
                    arc.get("involved_characters") or []
                ),
            },
        )

    all_chapters = [
        *(context.get("canonical_source_chapters") or []),
        *(context.get("future_chapters") or []),
    ]
    for chapter in all_chapters:
        chapter_id = str(chapter.get("id") or "")
        add(
            f"chapter:{chapter_id}:summary",
            "canonical_chapter"
            if chapter.get("is_canonical")
            else "future_chapter",
            (
                f"第 {chapter.get('position') or 0} 章"
                f"《{chapter.get('title') or ''}》"
            ),
            {
                "outline": chapter.get("outline"),
                "memory_summary": chapter.get("memory_summary"),
                "key_points": chapter.get("key_points") or [],
                "skeleton_role": chapter.get("skeleton_role"),
                "skeleton_arc_titles": (
                    chapter.get("skeleton_arc_titles") or []
                ),
                "ending_hook": chapter.get("skeleton_ending_hook"),
                "confirmed_task_card": (
                    chapter.get("confirmed_task_card") or {}
                ),
            },
        )

    for chapter in context.get("canonical_source_chapters") or []:
        chapter_id = str(chapter.get("id") or "")
        for index, event in enumerate(
            chapter.get("canonical_events") or [],
            start=1,
        ):
            add(
                f"chapter:{chapter_id}:event:{index}",
                "canonical_event",
                f"第 {chapter.get('position') or 0} 章正史事件",
                event,
            )

    memory = dict(context.get("canonical_memory") or {})
    for category, rows, label in (
        ("fact", memory.get("story_facts") or [], "正史事实"),
        (
            "knowledge",
            memory.get("character_knowledge") or [],
            "人物知情",
        ),
        ("thread", memory.get("plot_threads") or [], "正史剧情线"),
        (
            "foreshadowing",
            memory.get("foreshadowing") or [],
            "正史伏笔",
        ),
    ):
        for index, row in enumerate(rows, start=1):
            add(
                f"canon:{category}:{index}",
                f"canonical_{category}",
                f"{label} {index}",
                row,
            )

    current_state = dict(memory.get("current_state") or {})
    for category in (
        "characters",
        "relationships",
        "locations",
        "items",
        "knowledge",
        "plot_threads",
        "foreshadowing",
        "events",
        "causal_edges",
    ):
        values = current_state.get(category) or {}
        if not isinstance(values, Mapping):
            continue
        for index, (name, value) in enumerate(
            sorted(values.items(), key=lambda item: str(item[0])),
            start=1,
        ):
            add(
                f"state:{category}:{index}",
                f"current_state_{category}",
                f"当前状态 · {name}",
                {str(name): value},
            )

    issues = list(memory.get("continuity_issues") or [])
    add(
        "continuity:active-issue-count",
        "continuity",
        "当前连续性问题数量",
        {
            "active_issue_count": len(issues),
            "hard_issue_count": sum(
                str(item.get("severity") or "") == "hard"
                for item in issues
            ),
        },
    )
    for index, issue in enumerate(issues, start=1):
        issue_id = str(issue.get("id") or index)
        add(
            f"continuity:issue:{issue_id}",
            "continuity",
            f"连续性问题 {index}",
            issue,
        )

    for chapter in context.get("future_chapters") or []:
        chapter_id = str(chapter.get("id") or "")
        for index, key_point in enumerate(
            chapter.get("key_points") or [],
            start=1,
        ):
            add(
                f"chapter:{chapter_id}:key-point:{index}",
                "future_chapter_detail",
                (
                    f"第 {chapter.get('position') or 0} 章"
                    f"关键行动 {index}"
                ),
                key_point,
            )
    return items


def _baseline_fingerprint(
    context: Mapping[str, Any],
    *,
    ignored_link_ids: set[str] | None = None,
    task_card_fallbacks: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    ignored = ignored_link_ids or set()
    fallbacks = task_card_fallbacks or {}
    future = []
    for raw in context.get("future_chapters") or []:
        item = dict(raw)
        chapter_id = str(item.get("id") or "")
        if (
            chapter_id in fallbacks
            and not item.get("confirmed_task_card")
        ):
            item["confirmed_task_card"] = dict(fallbacks[chapter_id])
        future.append(item)
    payload = {
        "project": dict(context.get("project") or {}),
        "current_canonical_position": int(
            context.get("current_canonical_position") or 0
        ),
        "characters": list(context.get("characters") or []),
        "confirmed_story_blueprint": dict(
            context.get("confirmed_story_blueprint") or {}
        ),
        "confirmed_planned_plot_arcs": list(
            context.get("confirmed_planned_plot_arcs") or []
        ),
        "canonical_source_chapters": list(
            context.get("canonical_source_chapters") or []
        ),
        "canonical_memory": dict(context.get("canonical_memory") or {}),
        "future_chapters": future,
        "active_causal_links": [
            dict(item)
            for item in (context.get("active_causal_links") or [])
            if str(item.get("id") or "") not in ignored
        ],
    }
    return _fingerprint(payload)


def _lines(value: Any, *, limit: int = 12) -> List[str]:
    if isinstance(value, list):
        raw_values = value
    else:
        raw_values = str(value or "").splitlines()
    result: List[str] = []
    for raw in raw_values:
        item = str(raw).strip()
        if item and item not in result:
            result.append(item)
        if len(result) >= limit:
            break
    return result


class CausalSuggestionService:
    def __init__(self, database: Database):
        self.database = database
        self.structure_links = StructureLinkService(database)

    @staticmethod
    def _build_context(
        connection,
        *,
        user_id: int,
        project_id: str,
        chapter_limit: int,
    ) -> Optional[Dict[str, Any]]:
        base = StoryPlanSuggestionService._build_context(
            connection,
            user_id=user_id,
            project_id=project_id,
        )
        if base is None:
            return None
        current_position = int(
            base.get("current_canonical_position") or 0
        )
        future_rows = connection.execute(
            """
            SELECT ch.id, ch.position, ch.title, ch.outline, ch.key_points,
                   ch.skeleton_role, ch.skeleton_arc_titles_json,
                   ch.skeleton_ending_hook, ch.skeleton_application_id,
                   ch.canonical_version_id,
                   v.position AS volume_position,
                   v.title AS volume_title, v.goal AS volume_goal,
                   v.payoff AS volume_payoff,
                   cp.status AS task_card_status,
                   cp.purpose AS task_card_purpose,
                   cp.start_state AS task_card_start_state,
                   cp.end_state AS task_card_end_state,
                   cp.central_conflict AS task_card_central_conflict,
                   cp.must_happen_json, cp.must_preserve_json,
                   cp.forbidden_json, cp.ending_hook AS task_card_ending_hook
            FROM novel_chapters ch
            LEFT JOIN novel_volumes v ON v.id=ch.volume_id
            LEFT JOIN novel_chapter_plans cp ON cp.chapter_id=ch.id
            WHERE ch.project_id=? AND ch.position>?
                AND ch.canonical_version_id IS NULL
            ORDER BY ch.position
            LIMIT ?
            """,
            (project_id, current_position, chapter_limit),
        ).fetchall()
        future_chapters: List[Dict[str, Any]] = []
        for row in future_rows:
            raw = dict(row)
            task_card: Dict[str, Any] = {}
            if str(raw.pop("task_card_status") or "") == "confirmed":
                task_card = {
                    "purpose": raw.pop("task_card_purpose") or "",
                    "start_state": raw.pop("task_card_start_state") or "",
                    "end_state": raw.pop("task_card_end_state") or "",
                    "central_conflict": (
                        raw.pop("task_card_central_conflict") or ""
                    ),
                    "must_happen": _load_json(
                        raw.pop("must_happen_json"), []
                    ),
                    "must_preserve": _load_json(
                        raw.pop("must_preserve_json"), []
                    ),
                    "forbidden": _load_json(
                        raw.pop("forbidden_json"), []
                    ),
                    "ending_hook": (
                        raw.pop("task_card_ending_hook") or ""
                    ),
                }
            else:
                for key in (
                    "task_card_purpose",
                    "task_card_start_state",
                    "task_card_end_state",
                    "task_card_central_conflict",
                    "must_happen_json",
                    "must_preserve_json",
                    "forbidden_json",
                    "task_card_ending_hook",
                ):
                    raw.pop(key, None)
            raw["key_points"] = _lines(raw.get("key_points"))
            raw["skeleton_arc_titles"] = _load_json(
                raw.pop("skeleton_arc_titles_json"), []
            )
            raw["is_canonical"] = bool(
                raw.pop("canonical_version_id", None)
            )
            raw["confirmed_task_card"] = task_card
            future_chapters.append(raw)

        canonical_rows = connection.execute(
            """
            SELECT ch.id, ch.position, ch.title, ch.outline, ch.key_points,
                   (
                       SELECT m.summary
                       FROM chapter_memory m
                       WHERE m.chapter_id=ch.id
                         AND m.record_status='canon'
                       ORDER BY m.created_at DESC LIMIT 1
                   ) AS memory_summary
            FROM novel_chapters ch
            WHERE ch.project_id=? AND ch.canonical_version_id IS NOT NULL
            ORDER BY ch.position DESC
            LIMIT 6
            """,
            (project_id,),
        ).fetchall()
        canonical_sources: List[Dict[str, Any]] = []
        canonical_by_id: Dict[str, Dict[str, Any]] = {}
        for row in reversed(canonical_rows):
            item = dict(row)
            item["key_points"] = _lines(item.get("key_points"))
            item["is_canonical"] = True
            item["canonical_events"] = []
            canonical_by_id[str(item["id"])] = item
        if canonical_by_id:
            placeholders = ",".join("?" for _ in canonical_by_id)
            event_rows = connection.execute(
                f"""
                SELECT chapter_id, event_key, summary, effects_json,
                       evidence
                FROM story_events
                WHERE project_id=? AND record_status='canon'
                  AND chapter_id IN ({placeholders})
                ORDER BY chapter_id, position, id
                """,
                (project_id, *canonical_by_id.keys()),
            ).fetchall()
            for row in event_rows:
                owner = canonical_by_id.get(str(row["chapter_id"]))
                if owner is None:
                    continue
                owner["canonical_events"].append(
                    {
                        "event_key": str(row["event_key"] or ""),
                        "summary": str(row["summary"] or ""),
                        "effects": _load_json(row["effects_json"], []),
                        "evidence": str(row["evidence"] or ""),
                    }
                )
        canonical_sources = [
            item
            for item in canonical_by_id.values()
            if str(item.get("memory_summary") or "").strip()
            or item.get("canonical_events")
        ]

        target_ids = {str(item["id"]) for item in future_chapters}
        active_links: List[Dict[str, Any]] = []
        if target_ids:
            target_placeholders = ",".join("?" for _ in target_ids)
            rows = connection.execute(
                f"""
                SELECT id, source_chapter_id, target_chapter_id,
                       relation_type, cause_text, effect_text, status
                FROM novel_chapter_causal_links
                WHERE project_id=? AND status='active'
                  AND target_chapter_id IN ({target_placeholders})
                ORDER BY created_at, id
                """,
                (project_id, *target_ids),
            ).fetchall()
            active_links = [dict(row) for row in rows]

        observations = CausalSuggestionService._observations(
            future_chapters=future_chapters,
            allowed_sources=[*canonical_sources, *future_chapters],
            active_links=active_links,
        )
        project = dict(base.get("project") or {})
        branch_row = connection.execute(
            """
            SELECT canonical_branch_id
            FROM novel_projects
            WHERE id=? AND user_id=?
            """,
            (project_id, user_id),
        ).fetchone()
        branch_id = str(
            (branch_row["canonical_branch_id"] if branch_row else "")
            or "main"
        )
        query_concepts = [
            str(project.get("title") or ""),
            str(project.get("premise") or ""),
            *[
                str(item.get("name") or "")
                for item in (base.get("characters") or [])
            ],
            *[
                str(item.get("title") or "")
                for item in (
                    base.get("confirmed_planned_plot_arcs") or []
                )
            ],
            *[
                str(item.get("title") or "")
                for item in future_chapters[:40]
            ],
            *[
                str(value)
                for item in future_chapters[:30]
                for value in (item.get("key_points") or [])[:2]
            ],
        ]
        continuity = get_continuity_context(
            connection,
            project_id=project_id,
            branch_id=branch_id,
            before_chapter_position=current_position + 1,
            query_concepts=query_concepts,
        )
        canonical_memory = dict(base.get("canonical_memory") or {})
        canonical_memory["current_state"] = continuity["current_state"]
        canonical_memory["continuity_issues"] = continuity[
            "continuity_issues"
        ]
        canonical_memory["continuity_replay"] = continuity[
            "continuity_replay"
        ]
        context = {
            "schema_version": 3,
            "context_policy": {
                "canon_source": "author_confirmed_canon_only",
                "future_source": "author_managed_future_structure",
                "draft_task_cards_excluded": True,
                "proposal_only_until_author_accepts": True,
                "semantic_similarity_is_not_causality": True,
                "compare_plausible_causes_for_same_outcome": True,
                "missing_steps_are_not_assumed_true": True,
                "semantic_claims_require_frozen_evidence_ids": True,
                "model_conflicts_require_author_override": True,
            },
            "project": {
                key: project.get(key)
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
            "current_canonical_position": current_position,
            "chapter_limit": chapter_limit,
            "characters": list(base.get("characters") or []),
            "confirmed_story_blueprint": dict(
                base.get("confirmed_story_blueprint") or {}
            ),
            "confirmed_planned_plot_arcs": list(
                base.get("confirmed_planned_plot_arcs") or []
            ),
            "canonical_memory": canonical_memory,
            "canonical_source_chapters": canonical_sources,
            "future_chapters": future_chapters,
            "allowed_source_chapters": [
                *canonical_sources,
                *future_chapters,
            ],
            "active_causal_links": active_links,
            "deterministic_observations": observations,
        }
        context["evidence_catalog"] = _build_evidence_catalog(context)
        return context

    @staticmethod
    def _observations(
        *,
        future_chapters: List[Dict[str, Any]],
        allowed_sources: List[Dict[str, Any]],
        active_links: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        observations: List[Dict[str, Any]] = []
        incoming = {
            str(item.get("target_chapter_id") or "")
            for item in active_links
        }
        for target in future_chapters:
            target_id = str(target.get("id") or "")
            role = str(target.get("skeleton_role") or "")
            if role not in {"reversal", "payoff"} or target_id in incoming:
                continue
            earlier = [
                item
                for item in allowed_sources
                if int(item.get("position") or 0)
                < int(target.get("position") or 0)
            ]
            if not earlier:
                continue
            source = max(
                earlier,
                key=lambda item: int(item.get("position") or 0),
            )
            observations.append(
                {
                    "code": "FUTURE_OUTCOME_WITHOUT_INCOMING_CAUSE",
                    "source_chapter_id": str(source.get("id") or ""),
                    "target_chapter_id": target_id,
                    "reason": (
                        f"第 {target.get('position')} 章职责为 {role}，"
                        "但尚无作者显式指定的 incoming 因果。"
                    ),
                }
            )
        for left, right in zip(future_chapters, future_chapters[1:]):
            left_source = str(
                left.get("skeleton_application_id") or "author"
            )
            right_source = str(
                right.get("skeleton_application_id") or "author"
            )
            if left_source == right_source:
                continue
            left_arcs = {
                str(value)
                for value in (left.get("skeleton_arc_titles") or [])
                if str(value).strip()
            }
            right_arcs = {
                str(value)
                for value in (right.get("skeleton_arc_titles") or [])
                if str(value).strip()
            }
            if not left_arcs or not right_arcs or left_arcs & right_arcs:
                continue
            left_position = int(left.get("position") or 0)
            right_position = int(right.get("position") or 0)
            bridged = any(
                _chapter_position(
                    allowed_sources,
                    str(item.get("source_chapter_id") or ""),
                )
                <= left_position
                and _chapter_position(
                    future_chapters,
                    str(item.get("target_chapter_id") or ""),
                )
                >= right_position
                for item in active_links
            )
            if not bridged:
                observations.append(
                    {
                        "code": "WINDOW_ARC_HARD_CUT",
                        "source_chapter_id": str(left.get("id") or ""),
                        "target_chapter_id": str(right.get("id") or ""),
                        "reason": (
                            f"第 {left_position}–{right_position} 章切换"
                            "结构来源且两侧确认剧情线没有交集。"
                        ),
                    }
                )
        return observations[:16]

    @staticmethod
    def _ensure_ready(context: Mapping[str, Any]) -> None:
        if not context.get("confirmed_story_blueprint"):
            raise ValueError("请先确认一版全书蓝图，再审查跨章因果")
        if len(context.get("future_chapters") or []) < 2:
            raise ValueError("至少需要两章正史之后的未来骨架才能审查因果")

    def create_suggestion(
        self,
        *,
        user_id: int,
        project_id: str,
        chapter_limit: int,
        instruction: str,
        provider: str,
        model: str,
        credential_source: str,
    ) -> str:
        if not 2 <= chapter_limit <= 80:
            raise ValueError("因果审查范围必须为未来 2–80 章")
        if credential_source not in {"default", "personal"}:
            raise ValueError("不支持的 API 凭据来源")
        clean_instruction = instruction.strip()
        if len(clean_instruction) > 4000:
            raise ValueError("本次因果审查重点不能超过 4,000 个字符")
        suggestion_id = uuid.uuid4().hex
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            context = self._build_context(
                connection,
                user_id=user_id,
                project_id=project_id,
                chapter_limit=chapter_limit,
            )
            if context is None:
                connection.rollback()
                raise ValueError("小说项目不存在")
            self._ensure_ready(context)
            if credential_source == "personal":
                credential = connection.execute(
                    """
                    SELECT 1 FROM api_credentials
                    WHERE user_id=? AND provider=?
                    """,
                    (user_id, provider),
                ).fetchone()
                if not credential:
                    connection.rollback()
                    raise ValueError(
                        "所选模型服务 API Key 或凭据不存在，请重新配置"
                    )
            active_self = connection.execute(
                """
                SELECT id, project_id, chapter_limit, instruction
                FROM novel_causal_link_suggestions
                WHERE user_id=? AND status IN ('queued', 'running')
                ORDER BY created_at LIMIT 1
                """,
                (user_id,),
            ).fetchone()
            if active_self:
                if (
                    str(active_self["project_id"]) == project_id
                    and int(active_self["chapter_limit"]) == chapter_limit
                    and str(active_self["instruction"])
                    == clean_instruction
                ):
                    connection.rollback()
                    return str(active_self["id"])
                connection.rollback()
                raise ValueError(
                    "你已有一个因果审查任务正在排队或运行，请等待其完成"
                )
            active_other = connection.execute(
                """
                SELECT
                  (SELECT id FROM generation_jobs
                   WHERE user_id=? AND status IN ('queued', 'running')
                   ORDER BY created_at LIMIT 1) AS generation_id,
                  (SELECT id FROM analysis_jobs
                   WHERE user_id=? AND status IN ('queued', 'running')
                   ORDER BY created_at LIMIT 1) AS analysis_id,
                  (SELECT id FROM voice_profile_suggestions
                   WHERE user_id=? AND status IN ('queued', 'running')
                   ORDER BY created_at LIMIT 1) AS voice_id,
                  (SELECT id FROM editing_preference_suggestions
                   WHERE user_id=? AND status IN ('queued', 'running')
                   ORDER BY created_at LIMIT 1) AS preference_id,
                  (SELECT id FROM story_plan_suggestions
                   WHERE user_id=? AND status IN ('queued', 'running')
                   ORDER BY created_at LIMIT 1) AS story_plan_id,
                  (SELECT id FROM story_structure_suggestions
                   WHERE user_id=? AND status IN ('queued', 'running')
                   ORDER BY created_at LIMIT 1) AS structure_id,
                  (SELECT id FROM novel_causal_branch_simulations
                   WHERE user_id=? AND status IN ('queued', 'running')
                   ORDER BY created_at LIMIT 1) AS causal_branch_id
                """,
                (user_id,) * 7,
            ).fetchone()
            if any(active_other[key] for key in active_other.keys()):
                connection.rollback()
                raise ValueError(
                    "你已有一个 AI 任务正在排队或运行，请等待其完成"
                )
            connection.execute(
                """
                INSERT INTO novel_causal_link_suggestions(
                    id, project_id, user_id, instruction, chapter_limit,
                    provider, model, credential_source, status,
                    baseline_fingerprint, context_snapshot_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?)
                """,
                (
                    suggestion_id,
                    project_id,
                    user_id,
                    clean_instruction,
                    chapter_limit,
                    provider,
                    model,
                    credential_source,
                    _baseline_fingerprint(context),
                    _json(context),
                    now,
                ),
            )
            connection.commit()
        return suggestion_id

    def list_suggestions(
        self,
        *,
        user_id: int,
        project_id: str,
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT suggestion.*,
                       (
                           SELECT COUNT(*) FROM
                               novel_causal_link_suggestion_reviews review
                           WHERE review.suggestion_id=suggestion.id
                             AND review.decision='accepted'
                       ) AS accepted_count,
                       (
                           SELECT COUNT(*) FROM
                               novel_causal_link_suggestion_reviews review
                           WHERE review.suggestion_id=suggestion.id
                             AND review.decision='dismissed'
                       ) AS dismissed_count
                FROM novel_causal_link_suggestions suggestion
                JOIN novel_projects project
                  ON project.id=suggestion.project_id
                WHERE suggestion.project_id=? AND project.user_id=?
                ORDER BY suggestion.created_at DESC, suggestion.rowid DESC
                LIMIT ?
                """,
                (project_id, user_id, max(1, min(limit, 30))),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_suggestion(
        self,
        *,
        user_id: int,
        suggestion_id: str,
    ) -> Optional[Dict[str, Any]]:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT suggestion.*, project.title AS project_title,
                       project.genre, project.premise
                FROM novel_causal_link_suggestions suggestion
                JOIN novel_projects project
                  ON project.id=suggestion.project_id
                WHERE suggestion.id=? AND project.user_id=?
                """,
                (suggestion_id, user_id),
            ).fetchone()
            if not row:
                return None
            review_rows = connection.execute(
                """
                SELECT review.*, link.status AS causal_link_status
                FROM novel_causal_link_suggestion_reviews review
                LEFT JOIN novel_chapter_causal_links link
                  ON link.id=review.causal_link_id
                WHERE review.suggestion_id=?
                ORDER BY review.proposal_index
                """,
                (suggestion_id,),
            ).fetchall()
            item = dict(row)
            frozen_context = _load_json(
                item.pop("context_snapshot_json"), {}
            )
            item["context_snapshot"] = frozen_context
            raw_result = item.pop("result_json")
            item["result"] = None
            item["result_error"] = ""
            proposal_set: Optional[CausalSuggestionSet] = None
            if raw_result:
                try:
                    proposal_set = CausalSuggestionSet.model_validate_json(
                        str(raw_result)
                    )
                    proposal_set.ensure_context_compatible(frozen_context)
                except ValueError:
                    item["result_error"] = (
                        "已保存因果建议损坏或不再符合冻结上下文"
                    )
            reviews = {
                int(row["proposal_index"]): dict(row)
                for row in review_rows
            }
            if proposal_set:
                result = proposal_set.model_dump(mode="json")
                comparison_by_index: Dict[int, Dict[str, Any]] = {}
                for group in result["comparison_groups"]:
                    group["compatibility_label"] = (
                        CAUSAL_EXPLANATION_COMPATIBILITY_LABELS.get(
                            str(group["compatibility"]),
                            str(group["compatibility"]),
                        )
                    )
                    group_size = len(group["proposal_indices"])
                    for position, proposal_index in enumerate(
                        group["proposal_indices"],
                        start=1,
                    ):
                        comparison_by_index[int(proposal_index)] = {
                            **group,
                            "position": position,
                            "size": group_size,
                            "is_first": position == 1,
                        }
                decorated_proposals = []
                for index, proposal in enumerate(result["proposals"]):
                    decorated = self._decorate_proposal(
                        raw=proposal,
                        context=frozen_context,
                        review=reviews.get(index),
                    )
                    decorated["proposal_index"] = index
                    decorated["comparison_group"] = (
                        comparison_by_index.get(index)
                    )
                    decorated_proposals.append(decorated)
                result["proposals"] = decorated_proposals
                result["comparison_group_count"] = len(
                    result["comparison_groups"]
                )
                result["semantic_conflict_count"] = sum(
                    int(proposal["semantic_conflict_count"])
                    for proposal in decorated_proposals
                )
                result["semantic_uncertain_count"] = sum(
                    int(proposal["semantic_uncertain_count"])
                    for proposal in decorated_proposals
                )
                item["result"] = result
            current_context = self._build_context(
                connection,
                user_id=user_id,
                project_id=str(item["project_id"]),
                chapter_limit=int(item["chapter_limit"]),
            )
            own_accepted_link_ids = {
                str(row["causal_link_id"])
                for row in review_rows
                if str(row["decision"]) == "accepted"
                and str(row["causal_link_id"] or "")
            }
            own_affected_chapter_ids = {
                str(row[key])
                for row in review_rows
                if str(row["decision"]) == "accepted"
                for key in ("source_chapter_id", "target_chapter_id")
            }
            frozen_task_cards = _task_card_fallbacks(
                frozen_context,
                own_affected_chapter_ids,
            )
        item["baseline_changed"] = bool(
            current_context is None
            or _baseline_fingerprint(
                current_context,
                ignored_link_ids=own_accepted_link_ids,
                task_card_fallbacks=frozen_task_cards,
            )
            != str(item["baseline_fingerprint"])
        )
        item["application_blocker"] = (
            "项目资料、正史、未来骨架或现有因果链接已经变化；"
            "请重新生成建议"
            if item["baseline_changed"]
            else ""
        )
        item["reviews"] = [dict(row) for row in review_rows]
        return item

    @staticmethod
    def _decorate_proposal(
        *,
        raw: Dict[str, Any],
        context: Mapping[str, Any],
        review: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        chapters = {
            str(item.get("id") or ""): dict(item)
            for item in (context.get("allowed_source_chapters") or [])
        }
        source = chapters.get(str(raw["source_chapter_id"]), {})
        target = chapters.get(str(raw["target_chapter_id"]), {})
        source_arcs = {
            str(value)
            for value in (source.get("skeleton_arc_titles") or [])
            if str(value).strip()
        }
        target_arcs = {
            str(value)
            for value in (target.get("skeleton_arc_titles") or [])
            if str(value).strip()
        }
        raw["source_chapter"] = source
        raw["target_chapter"] = target
        raw["relation_label"] = CAUSAL_RELATION_LABELS.get(
            str(raw["relation_type"]),
            str(raw["relation_type"]),
        )
        raw["cross_line"] = bool(
            source_arcs and target_arcs and not source_arcs & target_arcs
        )
        evidence_catalog = {
            str(item.get("id") or ""): dict(item)
            for item in (context.get("evidence_catalog") or [])
            if str(item.get("id") or "")
        }
        for check in raw.get("semantic_checks") or []:
            check["category_label"] = CAUSAL_SEMANTIC_CATEGORY_LABELS.get(
                str(check.get("category") or ""),
                str(check.get("category") or ""),
            )
            check["status_label"] = CAUSAL_SEMANTIC_STATUS_LABELS.get(
                str(check.get("status") or ""),
                str(check.get("status") or ""),
            )
            check["evidence"] = [
                evidence_catalog[reference]
                for reference in check.get("evidence_refs") or []
                if reference in evidence_catalog
            ]
        future_chapters = {
            str(item.get("id") or ""): dict(item)
            for item in (context.get("future_chapters") or [])
            if str(item.get("id") or "")
        }
        for impact in raw.get("arc_impacts") or []:
            impact["impact_label"] = CAUSAL_ARC_IMPACT_LABELS.get(
                str(impact.get("impact_type") or ""),
                str(impact.get("impact_type") or ""),
            )
            impact["evidence"] = [
                evidence_catalog[reference]
                for reference in impact.get("evidence_refs") or []
                if reference in evidence_catalog
            ]
            impact["support_chapters"] = [
                future_chapters[chapter_id]
                for chapter_id in (
                    impact.get("required_support_chapter_ids") or []
                )
                if chapter_id in future_chapters
            ]
        raw["semantic_conflict_count"] = sum(
            str(item.get("status") or "") == "conflict"
            for item in (raw.get("semantic_checks") or [])
        )
        raw["semantic_uncertain_count"] = sum(
            str(item.get("status") or "") == "uncertain"
            for item in (raw.get("semantic_checks") or [])
        )
        raw["review"] = review
        return raw

    def get_status(
        self,
        *,
        user_id: int,
        suggestion_id: str,
    ) -> Optional[str]:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT suggestion.status
                FROM novel_causal_link_suggestions suggestion
                JOIN novel_projects project
                  ON project.id=suggestion.project_id
                WHERE suggestion.id=? AND project.user_id=?
                """,
                (suggestion_id, user_id),
            ).fetchone()
        return str(row["status"]) if row else None

    def claim_next_suggestion(self) -> Optional[Dict[str, Any]]:
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = utc_now()
            connection.execute(
                """
                UPDATE novel_causal_link_suggestions
                SET status='queued', started_at=NULL, claim_token=NULL,
                    lease_expires_at=NULL,
                    error='上一次因果审查租约已过期，已自动重新排队'
                WHERE status='running' AND lease_expires_at IS NOT NULL
                  AND lease_expires_at<=?
                """,
                (now,),
            )
            row = connection.execute(
                """
                SELECT suggestion.*
                FROM novel_causal_link_suggestions suggestion
                JOIN novel_projects project
                  ON project.id=suggestion.project_id
                WHERE suggestion.status='queued'
                ORDER BY suggestion.created_at
                LIMIT 1
                """
            ).fetchone()
            if not row:
                connection.commit()
                return None
            claim_token = uuid.uuid4().hex
            cursor = connection.execute(
                """
                UPDATE novel_causal_link_suggestions
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
        result = dict(row)
        result["claim_token"] = claim_token
        result["context_snapshot"] = _load_json(
            result.pop("context_snapshot_json"), {}
        )
        return result

    def complete_suggestion(
        self,
        *,
        suggestion_id: str,
        claim_token: str,
        result: CausalSuggestionSet,
        raw_response: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> bool:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT context_snapshot_json
                FROM novel_causal_link_suggestions
                WHERE id=? AND status='running' AND claim_token=?
                """,
                (suggestion_id, claim_token),
            ).fetchone()
            if not row:
                return False
            context = _load_json(row["context_snapshot_json"], {})
            result.ensure_context_compatible(context)
            cursor = connection.execute(
                """
                UPDATE novel_causal_link_suggestions
                SET status='completed', result_json=?, raw_response=?,
                    provider=?, model=?, input_tokens=?, output_tokens=?,
                    finished_at=?, claim_token=NULL, lease_expires_at=NULL,
                    error=NULL
                WHERE id=? AND status='running' AND claim_token=?
                """,
                (
                    result.model_dump_json(),
                    raw_response,
                    provider,
                    model,
                    input_tokens,
                    output_tokens,
                    utc_now(),
                    suggestion_id,
                    claim_token,
                ),
            )
            connection.commit()
        return cursor.rowcount == 1

    def fail_suggestion(
        self,
        *,
        suggestion_id: str,
        claim_token: str,
        error: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> bool:
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE novel_causal_link_suggestions
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
                    suggestion_id,
                    claim_token,
                ),
            )
            connection.commit()
        return cursor.rowcount == 1

    def release_claim(
        self,
        suggestion_id: str,
        claim_token: str,
        error: str,
    ) -> bool:
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE novel_causal_link_suggestions
                SET status='queued', started_at=NULL, claim_token=NULL,
                    lease_expires_at=NULL, error=?
                WHERE id=? AND status='running' AND claim_token=?
                """,
                (error[:2000], suggestion_id, claim_token),
            )
            connection.commit()
        return cursor.rowcount == 1

    def accept_proposal(
        self,
        *,
        user_id: int,
        suggestion_id: str,
        proposal_index: int,
        cause_text: str,
        effect_text: str,
        author_note: str,
        comparison_confirmed: bool = False,
        semantic_review_confirmed: bool = False,
        semantic_override_reason: str = "",
    ) -> Dict[str, Any]:
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                suggestion, proposal, comparison_group = (
                    self._proposal_for_review(
                        connection,
                        user_id=user_id,
                        suggestion_id=suggestion_id,
                        proposal_index=proposal_index,
                    )
                )
                if comparison_group and not comparison_confirmed:
                    raise ValueError(
                        "请先确认已比较同一结果的其他可能前因"
                    )
                if (
                    proposal.semantic_checks
                    and not semantic_review_confirmed
                ):
                    raise ValueError(
                        "请先核对五类语义复核和跨剧情线影响"
                    )
                semantic_conflicts = [
                    check
                    for check in proposal.semantic_checks
                    if check.status == "conflict"
                ]
                clean_override = semantic_override_reason.strip()
                if semantic_conflicts and len(clean_override) < 8:
                    raise ValueError(
                        "模型预检发现语义冲突；请记录作者覆盖理由或"
                        "先修改规划后重新生成"
                    )
                if (
                    proposal.bridge_readiness
                    == "needs_intermediate_steps"
                    and len(author_note.strip()) < 8
                ):
                    raise ValueError(
                        "这条候选仍缺少中间步骤；请在作者备注中说明"
                        "准备如何补写或为什么可以直接建立链接"
                    )
                note_parts = [
                    value
                    for value in (author_note.strip(),)
                    if value
                ]
                if semantic_conflicts:
                    note_parts.append(
                        "语义冲突覆盖：" + clean_override
                    )
                effective_author_note = "\n".join(note_parts)
                current_context = self._build_context(
                    connection,
                    user_id=user_id,
                    project_id=str(suggestion["project_id"]),
                    chapter_limit=int(suggestion["chapter_limit"]),
                )
                own_link_rows = connection.execute(
                    """
                    SELECT causal_link_id, source_chapter_id,
                           target_chapter_id
                    FROM novel_causal_link_suggestion_reviews
                    WHERE suggestion_id=? AND decision='accepted'
                      AND causal_link_id IS NOT NULL
                    """,
                    (suggestion_id,),
                ).fetchall()
                own_link_ids = {
                    str(row["causal_link_id"]) for row in own_link_rows
                }
                own_affected_ids = {
                    str(row[key])
                    for row in own_link_rows
                    for key in ("source_chapter_id", "target_chapter_id")
                }
                frozen_context = _load_json(
                    suggestion["context_snapshot_json"], {}
                )
                if current_context is None or _baseline_fingerprint(
                    current_context,
                    ignored_link_ids=own_link_ids,
                    task_card_fallbacks=_task_card_fallbacks(
                        frozen_context,
                        own_affected_ids,
                    ),
                ) != str(suggestion["baseline_fingerprint"]):
                    raise ValueError(
                        "项目资料、正史、未来骨架或现有因果链接已经变化；"
                        "请重新生成建议后再确认"
                    )
                link_result = (
                    self.structure_links._create_link_in_connection(
                        connection,
                        user_id=user_id,
                        project_id=str(suggestion["project_id"]),
                        source_chapter_id=proposal.source_chapter_id,
                        target_chapter_id=proposal.target_chapter_id,
                        relation_type=proposal.relation_type,
                        cause_text=cause_text,
                        effect_text=effect_text,
                        author_note=effective_author_note,
                        now=now,
                    )
                )
                connection.execute(
                    """
                    INSERT INTO novel_causal_link_suggestion_reviews(
                        id, suggestion_id, project_id, proposal_index,
                        decision, source_chapter_id, target_chapter_id,
                        relation_type, cause_text, effect_text, author_note,
                        causal_link_id, decided_at
                    ) VALUES (?, ?, ?, ?, 'accepted', ?, ?, ?, ?, ?, ?,
                              ?, ?)
                    """,
                    (
                        uuid.uuid4().hex,
                        suggestion_id,
                        suggestion["project_id"],
                        proposal_index,
                        proposal.source_chapter_id,
                        proposal.target_chapter_id,
                        proposal.relation_type,
                        cause_text.strip(),
                        effect_text.strip(),
                        effective_author_note,
                        link_result["id"],
                        now,
                    ),
                )
            except Exception:
                connection.rollback()
                raise
            connection.commit()
        return {
            **link_result,
            "suggestion_id": suggestion_id,
            "proposal_index": proposal_index,
            "semantic_conflict_count": len(semantic_conflicts),
        }

    def dismiss_proposal(
        self,
        *,
        user_id: int,
        suggestion_id: str,
        proposal_index: int,
    ) -> Dict[str, Any]:
        now = utc_now()
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                suggestion, proposal, _comparison_group = (
                    self._proposal_for_review(
                        connection,
                        user_id=user_id,
                        suggestion_id=suggestion_id,
                        proposal_index=proposal_index,
                    )
                )
                connection.execute(
                    """
                    INSERT INTO novel_causal_link_suggestion_reviews(
                        id, suggestion_id, project_id, proposal_index,
                        decision, source_chapter_id, target_chapter_id,
                        relation_type, cause_text, effect_text, author_note,
                        causal_link_id, decided_at
                    ) VALUES (?, ?, ?, ?, 'dismissed', ?, ?, ?, ?, ?, '',
                              NULL, ?)
                    """,
                    (
                        uuid.uuid4().hex,
                        suggestion_id,
                        suggestion["project_id"],
                        proposal_index,
                        proposal.source_chapter_id,
                        proposal.target_chapter_id,
                        proposal.relation_type,
                        proposal.cause_text,
                        proposal.effect_text,
                        now,
                    ),
                )
            except Exception:
                connection.rollback()
                raise
            connection.commit()
        return {
            "suggestion_id": suggestion_id,
            "proposal_index": proposal_index,
            "decision": "dismissed",
        }

    @staticmethod
    def _proposal_for_review(
        connection,
        *,
        user_id: int,
        suggestion_id: str,
        proposal_index: int,
    ):
        suggestion = connection.execute(
            """
            SELECT suggestion.*
            FROM novel_causal_link_suggestions suggestion
            JOIN novel_projects project
              ON project.id=suggestion.project_id
            WHERE suggestion.id=? AND project.user_id=?
            """,
            (suggestion_id, user_id),
        ).fetchone()
        if not suggestion:
            raise ValueError("因果建议任务不存在")
        if str(suggestion["status"]) != "completed":
            raise ValueError("因果建议尚未完成")
        if proposal_index < 0:
            raise ValueError("因果候选不存在或已经损坏")
        reviewed = connection.execute(
            """
            SELECT decision
            FROM novel_causal_link_suggestion_reviews
            WHERE suggestion_id=? AND proposal_index=?
            """,
            (suggestion_id, proposal_index),
        ).fetchone()
        if reviewed:
            label = (
                "采纳"
                if str(reviewed["decision"]) == "accepted"
                else "忽略"
            )
            raise ValueError(f"这条候选已经{label}")
        try:
            result = CausalSuggestionSet.model_validate_json(
                str(suggestion["result_json"] or "")
            )
            context = _load_json(suggestion["context_snapshot_json"], {})
            result.ensure_context_compatible(context)
            proposal = result.proposals[proposal_index]
            comparison_group = next(
                (
                    group
                    for group in result.comparison_groups
                    if proposal_index in group.proposal_indices
                ),
                None,
            )
        except (ValueError, IndexError) as exc:
            raise ValueError("因果候选不存在或已经损坏") from exc
        return suggestion, proposal, comparison_group


def _chapter_position(
    chapters: List[Dict[str, Any]],
    chapter_id: str,
) -> int:
    for item in chapters:
        if str(item.get("id") or "") == chapter_id:
            return int(item.get("position") or 0)
    return 0


def _task_card_fallbacks(
    context: Mapping[str, Any],
    chapter_ids: set[str],
) -> Dict[str, Mapping[str, Any]]:
    return {
        str(item.get("id") or ""): dict(
            item.get("confirmed_task_card") or {}
        )
        for item in (context.get("future_chapters") or [])
        if str(item.get("id") or "") in chapter_ids
        and item.get("confirmed_task_card")
    }
