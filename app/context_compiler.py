from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from pydantic import ValidationError

from .structure_link_schema import CAUSAL_RELATION_LABELS
from .technique_schema import TechniqueContextItem


DEFAULT_MEMORY_CHAR_BUDGET = 12_000
DEFAULT_TECHNIQUE_CHAR_BUDGET = 6_000
DEFAULT_STORY_PLAN_CHAR_BUDGET = 12_000
DEFAULT_CAUSAL_LINK_CHAR_BUDGET = 8_000


def _serialized_length(value: Mapping[str, Any]) -> int:
    return len(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    )


def _deduplicate(
    items: Iterable[Mapping[str, Any]], key_fields: Tuple[str, ...]
) -> List[Dict[str, Any]]:
    seen: set[Tuple[str, ...]] = set()
    result: List[Dict[str, Any]] = []
    for item in items:
        key = tuple(str(item.get(field) or "") for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(item))
    return result


def compile_canonical_memory(
    raw_memory: Mapping[str, Any],
    *,
    max_chars: int = DEFAULT_MEMORY_CHAR_BUDGET,
) -> Dict[str, Any]:
    """Select only confirmed memory in a deterministic, auditable budget."""

    retrieved_memory = _deduplicate(
        raw_memory.get("retrieved_memory") or [],
        ("source_type", "source_id"),
    )
    retrieved_keys = {
        (str(item.get("source_type") or ""), str(item.get("source_id") or ""))
        for item in retrieved_memory
    }

    def without_retrieved(
        items: Iterable[Mapping[str, Any]], source_type: str
    ) -> List[Mapping[str, Any]]:
        return [
            item
            for item in items
            if (
                source_type,
                str(item.get("source_id") or ""),
            )
            not in retrieved_keys
        ]

    candidates = {
        "continuity_issues": _deduplicate(
            raw_memory.get("continuity_issues") or [],
            ("fingerprint", "id"),
        ),
        "recent_chapters": list(raw_memory.get("recent_chapters") or []),
        "story_facts": _deduplicate(
            without_retrieved(
                raw_memory.get("story_facts") or [], "fact"
            ),
            ("subject_name", "predicate"),
        ),
        "character_knowledge": _deduplicate(
            without_retrieved(
                raw_memory.get("character_knowledge") or [], "knowledge"
            ),
            ("character_name", "fact_text"),
        ),
        "plot_threads": _deduplicate(
            without_retrieved(
                raw_memory.get("plot_threads") or [], "plot_thread"
            ),
            ("thread_name",),
        ),
        "foreshadowing": _deduplicate(
            without_retrieved(
                raw_memory.get("foreshadowing") or [], "foreshadowing"
            ),
            ("hook_name",),
        ),
        "retrieved_memory": retrieved_memory,
    }
    raw_retrieval = dict(raw_memory.get("retrieval") or {})
    current_state = dict(raw_memory.get("current_state") or {})
    if int(current_state.get("schema_version") or 0) >= 2:
        # Lifecycle replay is the authoritative current view. Keeping the
        # historical action rows as well would reintroduce stale "knows",
        # resolved-thread and paid-off-hook states into the same prompt.
        candidates["character_knowledge"] = []
        candidates["plot_threads"] = []
        candidates["foreshadowing"] = []
    continuity_replay = raw_memory.get("continuity_replay")
    result: Dict[str, Any] = {
        "source": "author_confirmed_canon_only",
        "current_state": current_state,
        "continuity_issues": [],
        "continuity_replay": (
            dict(continuity_replay)
            if isinstance(continuity_replay, Mapping)
            else None
        ),
        "recent_chapters": [],
        "story_facts": [],
        "character_knowledge": [],
        "plot_threads": [],
        "foreshadowing": [],
        "retrieved_memory": [],
        "retrieval": {
            "engine": str(raw_retrieval.get("engine") or "not_available"),
            "scope": str(raw_retrieval.get("scope") or ""),
            "query_terms": [
                str(term)
                for term in (raw_retrieval.get("query_terms") or [])[:48]
            ],
            "query_concepts": [
                str(term)
                for term in (
                    raw_retrieval.get("query_concepts") or []
                )[:16]
            ],
            "matched_count": int(
                raw_retrieval.get("matched_count")
                or len(retrieved_memory)
            ),
            "excluded_recent_chapter_count": int(
                raw_retrieval.get("excluded_recent_chapter_count") or 0
            ),
        },
        "truncated": False,
    }
    if _serialized_length(result) > max_chars:
        result["current_state"] = {
            "story_time": current_state.get("story_time"),
            "last_chapter": current_state.get("last_chapter"),
            "truncated": True,
        }
        result["truncated"] = True
    # Current state, continuity conflicts and recent causal context get budget
    # before older state rows.
    category_order = (
        "recent_chapters",
        "plot_threads",
        "foreshadowing",
        "retrieved_memory",
        "continuity_issues",
        "story_facts",
        "character_knowledge",
    )
    for category in category_order:
        for item in candidates[category]:
            result[category].append(dict(item))
            if _serialized_length(result) > max_chars:
                result[category].pop()
                result["truncated"] = True
                break
        if len(result[category]) < len(candidates[category]):
            result["truncated"] = True

    # Database retrieval is newest-first; prose context reads naturally oldest-first.
    result["recent_chapters"].reverse()
    result["included_counts"] = {
        category: len(result[category]) for category in category_order
    }
    return result


def compile_active_techniques(
    raw_techniques: Iterable[Mapping[str, Any]],
    *,
    usage: str,
    max_chars: int = DEFAULT_TECHNIQUE_CHAR_BUDGET,
    max_cards: int = 12,
) -> Dict[str, Any]:
    """Select author-enabled abstract techniques without source prose."""

    if usage not in {"plan", "write", "audit"}:
        raise ValueError("不支持的技法使用阶段")
    specificity = {"project": 0, "volume": 1, "chapter": 2, "scene": 3}
    selected: dict[str, tuple[tuple[int, int], Dict[str, Any]]] = {}
    for raw in raw_techniques:
        modes = raw.get("usage_modes") or []
        if usage not in modes:
            continue
        try:
            item = TechniqueContextItem.model_validate(
                {
                    "id": raw.get("id") or raw.get("technique_id"),
                    "name": raw.get("name"),
                    "dimension": raw.get("dimension"),
                    "execution_rule": raw.get("execution_rule"),
                    "effect": raw.get("effect"),
                    "originality_boundary": raw.get(
                        "originality_boundary"
                    ),
                    "author_adaptation": raw.get("author_adaptation") or "",
                    "scope_type": raw.get("scope_type"),
                    "scope_label": raw.get("scope_label") or "",
                    "priority": int(raw.get("priority") or 0),
                }
            ).model_dump(mode="json")
        except (TypeError, ValueError, ValidationError):
            continue
        card_id = str(item["id"])
        score = (
            int(item["priority"]),
            specificity.get(str(item["scope_type"]), -1),
        )
        previous = selected.get(card_id)
        if previous is None or score > previous[0]:
            selected[card_id] = (score, item)
    ordered = sorted(
        (item for _score, item in selected.values()),
        key=lambda item: (
            -int(item["priority"]),
            -specificity.get(str(item["scope_type"]), -1),
            str(item["name"]),
        ),
    )
    result: Dict[str, Any] = {
        "source": "author_enabled_abstract_techniques",
        "usage": usage,
        "items": [],
        "truncated": False,
    }
    for item in ordered:
        if len(result["items"]) >= max_cards:
            result["truncated"] = True
            break
        result["items"].append(item)
        if _serialized_length(result) > max_chars:
            result["items"].pop()
            result["truncated"] = True
            break
    result["included_count"] = len(result["items"])
    return result


def compile_story_plan_context(
    context: Mapping[str, Any],
    *,
    usage: str,
    max_chars: int = DEFAULT_STORY_PLAN_CHAR_BUDGET,
) -> Dict[str, Any]:
    """Expose confirmed long-form plans without drafts or revision history."""

    if usage not in {"plan", "write", "audit"}:
        raise ValueError("不支持的全书规划使用阶段")
    raw_blueprint = context.get("story_blueprint") or {}
    blueprint_fields = (
        "central_question",
        "protagonist_goal",
        "core_conflict",
        "stakes",
        "ending_state",
        "must_payoffs",
        "forbidden_shortcuts",
        "opening_state",
        "major_turns",
    )
    raw_arcs = [
        dict(item)
        for item in (context.get("planned_plot_arcs") or [])
        if str(item.get("lifecycle_status") or "") != "abandoned"
    ]
    task_threads = [
        str(item).strip()
        for item in (
            (context.get("task_card") or {}).get("plot_threads") or []
        )
        if str(item).strip()
    ]
    if usage in {"write", "audit"}:
        selected_titles = {item.casefold() for item in task_threads}
        raw_arcs = [
            item
            for item in raw_arcs
            if str(item.get("title") or "").strip().casefold()
            in selected_titles
        ]
    ordered = sorted(
        raw_arcs,
        key=lambda item: (
            -int(item.get("priority") or 0),
            int(item.get("position") or 0),
            str(item.get("title") or ""),
        ),
    )
    result: Dict[str, Any] = {
        "source": "author_confirmed_story_plan_only",
        "usage": usage,
        "blueprint": {},
        "plot_arcs": [],
        "task_card_plot_threads": task_threads,
        "unmatched_task_threads": [],
        "truncated": False,
    }
    blueprint_budget = max(1000, int(max_chars * 0.6))
    for field in blueprint_fields:
        value = raw_blueprint.get(field)
        if not value:
            continue
        if isinstance(value, list):
            result["blueprint"][field] = []
            for raw_item in value:
                result["blueprint"][field].append(str(raw_item))
                if _serialized_length(result) > blueprint_budget:
                    result["blueprint"][field].pop()
                    result["truncated"] = True
                    break
            if not result["blueprint"][field]:
                result["blueprint"].pop(field)
        else:
            result["blueprint"][field] = str(value)
            if _serialized_length(result) > blueprint_budget:
                result["blueprint"].pop(field)
                result["truncated"] = True
    matched_titles = set()
    per_arc_budget = max(1800, min(4500, max_chars // 2))
    for raw_item in ordered:
        item = {
            field: raw_item.get(field)
            for field in (
                "id",
                "position",
                "arc_type",
                "title",
                "lifecycle_status",
                "priority",
            )
            if raw_item.get(field) not in (None, "", [])
        }
        for field in (
            "dramatic_question",
            "promise",
            "target_payoff",
            "start_state",
            "involved_characters",
            "planned_turns",
        ):
            value = raw_item.get(field)
            if not value:
                continue
            if isinstance(value, list):
                item[field] = []
                for raw_value in value:
                    item[field].append(str(raw_value))
                    if _serialized_length(item) > per_arc_budget:
                        item[field].pop()
                        result["truncated"] = True
                        break
                if not item[field]:
                    item.pop(field)
            else:
                item[field] = str(value)
                if _serialized_length(item) > per_arc_budget:
                    item.pop(field)
                    result["truncated"] = True
        result["plot_arcs"].append(item)
        if _serialized_length(result) > max_chars:
            result["plot_arcs"].pop()
            result["truncated"] = True
            break
        matched_titles.add(str(item.get("title") or "").strip().casefold())
    if usage in {"write", "audit"}:
        result["unmatched_task_threads"] = [
            item for item in task_threads if item.casefold() not in matched_titles
        ]
    result["included_arc_count"] = len(result["plot_arcs"])
    return result


def compile_planned_causal_links(
    context: Mapping[str, Any],
    *,
    usage: str,
    max_chars: int = DEFAULT_CAUSAL_LINK_CHAR_BUDGET,
    max_links: int = 16,
) -> Dict[str, Any]:
    """Expose only active author-authored links involving this chapter."""

    if usage not in {"plan", "write", "audit"}:
        raise ValueError("不支持的章节因果使用阶段")
    chapter_id = str((context.get("chapter") or {}).get("id") or "")
    raw_links = [
        dict(item)
        for item in (context.get("planned_causal_links") or [])
        if str(item.get("status") or "active") == "active"
        and not bool(item.get("target_is_canonical"))
        and chapter_id
        in {
            str(item.get("source_chapter_id") or ""),
            str(item.get("target_chapter_id") or ""),
        }
    ]
    raw_links.sort(
        key=lambda item: (
            0
            if str(item.get("target_chapter_id") or "") == chapter_id
            else 1,
            int(item.get("source_position") or 0),
            int(item.get("target_position") or 0),
            str(item.get("id") or ""),
        )
    )
    result: Dict[str, Any] = {
        "source": "author_confirmed_future_causality_only",
        "usage": usage,
        "chapter_id": chapter_id,
        "incoming": [],
        "outgoing": [],
        "truncated": False,
    }
    for raw in raw_links:
        relation_type = str(raw.get("relation_type") or "")
        source_arcs = [
            str(value)
            for value in (raw.get("source_arc_titles") or [])
            if str(value).strip()
        ]
        target_arcs = [
            str(value)
            for value in (raw.get("target_arc_titles") or [])
            if str(value).strip()
        ]
        shared_arcs = [
            str(value)
            for value in (raw.get("shared_arc_titles") or [])
            if str(value).strip()
        ]
        item = {
            "id": str(raw.get("id") or ""),
            "relation_type": relation_type,
            "relation_label": CAUSAL_RELATION_LABELS.get(
                relation_type, relation_type
            ),
            "source_chapter": {
                "id": str(raw.get("source_chapter_id") or ""),
                "position": int(raw.get("source_position") or 0),
                "title": str(raw.get("source_title") or ""),
                "arc_titles": source_arcs,
            },
            "target_chapter": {
                "id": str(raw.get("target_chapter_id") or ""),
                "position": int(raw.get("target_position") or 0),
                "title": str(raw.get("target_title") or ""),
                "arc_titles": target_arcs,
            },
            "cause": str(raw.get("cause_text") or ""),
            "effect": str(raw.get("effect_text") or ""),
            "author_note": str(raw.get("author_note") or ""),
            "cross_line": bool(raw.get("cross_line")),
            "shared_arc_titles": shared_arcs,
        }
        bucket = (
            "incoming"
            if item["target_chapter"]["id"] == chapter_id
            else "outgoing"
        )
        if (
            len(result["incoming"]) + len(result["outgoing"])
            >= max_links
        ):
            result["truncated"] = True
            break
        result[bucket].append(item)
        if _serialized_length(result) > max_chars:
            result[bucket].pop()
            result["truncated"] = True
            break
    result["included_count"] = (
        len(result["incoming"]) + len(result["outgoing"])
    )
    return result


def build_writing_context_snapshot(
    *,
    context: Mapping[str, Any],
    operation: str,
    instruction: str,
    current_content: str,
    previous_content: str,
) -> Dict[str, Any]:
    chapter = dict(context["chapter"])
    return {
        "schema_version": 1,
        "operation": operation,
        "instruction": instruction,
        "project": {
            "id": chapter.get("project_id"),
            "title": chapter.get("project_title"),
            "genre": chapter.get("genre"),
            "premise": chapter.get("premise"),
            "story_promise": chapter.get("story_promise"),
            "target_audience": chapter.get("target_audience"),
            "core_appeal": chapter.get("core_appeal"),
            "ending_constraint": chapter.get("ending_constraint"),
            "world_setting": chapter.get("world_setting"),
            "style_guide": chapter.get("style_guide"),
            "ai_instructions": chapter.get("ai_instructions"),
            "point_of_view": chapter.get("point_of_view"),
            "target_chapter_chars": chapter.get("target_chapter_chars"),
        },
        "chapter": {
            "id": chapter.get("id"),
            "position": chapter.get("position"),
            "title": chapter.get("title") or "未命名章节",
            "outline": chapter.get("outline"),
            "key_points": chapter.get("key_points"),
        },
        "volume": {
            "id": chapter.get("volume_id"),
            "title": chapter.get("volume_title"),
            "goal": chapter.get("volume_goal"),
            "start_state": chapter.get("volume_start_state"),
            "end_state": chapter.get("volume_end_state"),
            "major_conflict": chapter.get("volume_major_conflict"),
            "payoff": chapter.get("volume_payoff"),
        },
        "confirmed_task_card": context.get("task_card"),
        "characters": list(context.get("characters") or []),
        "confirmed_voice_profile": dict(
            context.get("voice_profile") or {}
        ),
        "confirmed_editing_preferences": [
            dict(item)
            for item in (
                context.get("confirmed_editing_preferences") or []
            )
        ],
        "confirmed_story_plan": compile_story_plan_context(
            context, usage="write"
        ),
        "planned_causal_links": compile_planned_causal_links(
            context, usage="write"
        ),
        "canonical_memory": dict(
            context.get("canonical_memory")
            or compile_canonical_memory({})
        ),
        "active_techniques": dict(
            context.get("active_techniques")
            or compile_active_techniques(
                context.get("technique_cards") or [], usage="write"
            )
        ),
        "previous_chapter_excerpt": previous_content[-16000:],
        "current_chapter_excerpt": current_content[-30000:],
    }


def build_scene_context_snapshot(
    *,
    context: Mapping[str, Any],
    operation: str,
    instruction: str,
    current_scene_content: str,
    previous_scene_content: str,
    previous_chapter_content: str,
) -> Dict[str, Any]:
    """Persist the exact, bounded inputs used for one scene operation."""

    snapshot = build_writing_context_snapshot(
        context=context,
        operation=operation,
        instruction=instruction,
        current_content="",
        previous_content=previous_chapter_content,
    )
    snapshot.pop("current_chapter_excerpt", None)
    snapshot["focused_scene"] = dict(context.get("focused_scene") or {})
    snapshot["scene_sequence"] = list(
        context.get("scene_sequence") or []
    )
    snapshot["previous_scene"] = dict(
        context.get("previous_scene") or {}
    )
    snapshot["next_scene"] = dict(context.get("next_scene") or {})
    snapshot["previous_scene_excerpt"] = previous_scene_content[-6000:]
    snapshot["current_scene_excerpt"] = current_scene_content[-12000:]
    snapshot["scene_target_chars"] = int(
        context.get("scene_target_chars") or 0
    )
    snapshot["scene_minimum_chars"] = int(
        context.get("scene_minimum_chars") or 0
    )
    return snapshot
