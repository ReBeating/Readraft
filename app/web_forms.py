from __future__ import annotations

from .story_planning_schema import PlannedStoryArc, StoryBlueprint
from .technique_schema import TechniqueObservation
from .workbench_view import STORY_ARC_TYPE_OPTIONS


STORY_ARC_LIFECYCLE_OPTIONS = (
    "planned",
    "active",
    "paused",
    "resolved",
    "abandoned",
)


def _clean_field(
    value: str,
    label: str,
    *,
    max_length: int,
    required: bool = False,
    min_length: int = 1,
) -> str:
    cleaned = value.strip()
    if required and len(cleaned) < min_length:
        raise ValueError(f"{label}至少需要 {min_length} 个字符")
    if len(cleaned) > max_length:
        raise ValueError(f"{label}不能超过 {max_length:,} 个字符")
    return cleaned


def _split_lines(value: str, *, limit: int = 30) -> list[str]:
    items = [line.strip() for line in value.splitlines() if line.strip()]
    if len(items) > limit:
        raise ValueError(f"逐行条目不能超过 {limit} 条")
    return items


def _story_plan_lines(
    value: str,
    label: str,
    *,
    limit: int,
    item_max_length: int = 1200,
) -> list[str]:
    items = _split_lines(value, limit=limit)
    for item in items:
        if len(item) > item_max_length:
            raise ValueError(f"{label}中每条不能超过 {item_max_length:,} 个字符")
    return items


def _story_blueprint_from_form(
    *,
    central_question: str,
    protagonist_goal: str,
    core_conflict: str,
    stakes: str,
    opening_state: str,
    ending_state: str,
    major_turns: str,
    must_payoffs: str,
    forbidden_shortcuts: str,
    author_notes: str,
) -> StoryBlueprint:
    return StoryBlueprint.model_validate(
        {
            "central_question": _clean_field(
                central_question, "核心悬问", max_length=2000
            ),
            "protagonist_goal": _clean_field(
                protagonist_goal, "主角长期目标", max_length=2000
            ),
            "core_conflict": _clean_field(
                core_conflict, "全书冲突引擎", max_length=3000
            ),
            "stakes": _clean_field(stakes, "长期代价与风险", max_length=3000),
            "opening_state": _clean_field(opening_state, "开篇状态", max_length=3000),
            "ending_state": _clean_field(ending_state, "终局状态", max_length=3000),
            "major_turns": _story_plan_lines(major_turns, "全书转折", limit=20),
            "must_payoffs": _story_plan_lines(must_payoffs, "必须兑现项", limit=30),
            "forbidden_shortcuts": _story_plan_lines(
                forbidden_shortcuts, "禁止捷径", limit=30
            ),
            "author_notes": _clean_field(author_notes, "蓝图作者备注", max_length=6000),
        }
    )


def _planned_story_arc_from_form(
    *,
    arc_type: str,
    title: str,
    dramatic_question: str,
    promise: str,
    start_state: str,
    target_payoff: str,
    involved_characters: str,
    planned_turns: str,
    lifecycle_status: str,
    priority: int,
    author_notes: str,
) -> PlannedStoryArc:
    if arc_type not in STORY_ARC_TYPE_OPTIONS:
        raise ValueError("请选择有效的剧情线类型")
    if lifecycle_status not in STORY_ARC_LIFECYCLE_OPTIONS:
        raise ValueError("请选择有效的剧情线阶段")
    return PlannedStoryArc.model_validate(
        {
            "arc_type": arc_type,
            "title": _clean_field(title, "剧情线名称", max_length=160, required=True),
            "dramatic_question": _clean_field(
                dramatic_question, "剧情线悬问", max_length=2000
            ),
            "promise": _clean_field(promise, "剧情线读者承诺", max_length=2000),
            "start_state": _clean_field(start_state, "剧情线起始状态", max_length=2000),
            "target_payoff": _clean_field(
                target_payoff, "剧情线目标回报", max_length=3000
            ),
            "involved_characters": _story_plan_lines(
                involved_characters, "涉及人物", limit=30, item_max_length=120
            ),
            "planned_turns": _story_plan_lines(planned_turns, "剧情线转折", limit=20),
            "lifecycle_status": lifecycle_status,
            "priority": priority,
            "author_notes": _clean_field(
                author_notes, "剧情线作者备注", max_length=4000
            ),
        }
    )


def _technique_observation_from_form(
    *,
    name: str,
    dimension: str,
    source_location: str,
    observation: str,
    effect: str,
    suitable_for: str,
    unsuitable_for: str,
    execution_rule: str,
    originality_boundary: str,
) -> TechniqueObservation:
    return TechniqueObservation.model_validate(
        {
            "name": _clean_field(
                name, "技法名称", max_length=80, required=True, min_length=2
            ),
            "dimension": dimension,
            "source_location": _clean_field(
                source_location,
                "来源位置",
                max_length=200,
                required=True,
            ),
            "observation": _clean_field(
                observation,
                "文本观察",
                max_length=600,
                required=True,
                min_length=10,
            ),
            "effect": _clean_field(
                effect,
                "读者效果",
                max_length=600,
                required=True,
                min_length=10,
            ),
            "suitable_for": _split_lines(suitable_for, limit=8),
            "unsuitable_for": _split_lines(unsuitable_for, limit=8),
            "execution_rule": _clean_field(
                execution_rule,
                "执行规则",
                max_length=600,
                required=True,
                min_length=10,
            ),
            "originality_boundary": _clean_field(
                originality_boundary,
                "原创性边界",
                max_length=600,
                required=True,
                min_length=10,
            ),
        }
    )
