from __future__ import annotations

import json
import re
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .assistant_chat_schema import AssistantStructuredSettingEdit
from .db import Database, utc_now
from .memory_identity import ensure_memory_identity
from .story_planning_schema import PlannedStoryArc, StoryBlueprint
from .story_planning_service import StoryPlanningService


ENTITY_LABELS = {
    "world_entry": "世界资料",
    "character": "人物",
    "relationship": "人物关系",
    "story_blueprint": "全书蓝图",
    "plot_arc": "剧情线",
    "voice_profile": "叙事与文风",
    "archive_rule": "已确认规则",
}

ACTION_LABELS = {
    "create": "新增",
    "update": "修改",
    "delete": "删除",
}

FIELD_LABELS = {
    "entry_type": "类型",
    "name": "名称",
    "description": "内容",
    "constraints": "规则与边界",
    "role": "人物定位",
    "external_goal": "外在目标",
    "internal_need": "内在需求与动机",
    "central_conflict": "核心矛盾",
    "secret": "秘密",
    "traits": "性格特征",
    "speech_style": "说话方式",
    "background": "背景",
    "initial_state": "初始状态",
    "character_arc": "人物弧光",
    "character_a_name": "人物 A",
    "character_b_name": "人物 B",
    "relationship": "当前关系",
    "tension": "关系张力",
    "change_direction": "变化方向",
    "central_question": "核心悬问",
    "protagonist_goal": "主角长期目标",
    "core_conflict": "冲突引擎",
    "stakes": "失败代价",
    "opening_state": "开篇状态",
    "ending_state": "终局状态",
    "major_turns": "全书转折",
    "must_payoffs": "必须兑现",
    "forbidden_shortcuts": "禁止捷径",
    "author_notes": "作者备注",
    "arc_type": "剧情线类型",
    "title": "名称",
    "dramatic_question": "剧情线悬问",
    "promise": "对读者的承诺",
    "start_state": "起始状态",
    "target_payoff": "目标回报",
    "involved_characters": "涉及人物",
    "planned_turns": "计划转折",
    "lifecycle_status": "状态",
    "priority": "优先级",
    "narrative_tense": "叙事时态",
    "narrative_distance": "叙事距离",
    "tone": "整体基调",
    "narration_rules": "叙述规则",
    "sentence_rhythm": "句段与节奏",
    "dialogue_voice": "对白方式",
    "sensory_palette": "感官与意象",
    "metaphor_policy": "比喻策略",
    "allowed_omissions": "允许的省略与留白",
    "preferred_patterns": "偏好表达",
    "banned_expressions": "禁用表达",
    "style_examples": "作者文风示例",
    "category": "分类",
    "content": "内容",
}

_WORLD_TYPES = {"background", "rule", "faction", "location", "element"}
_ARC_TYPES = {
    "main",
    "subplot",
    "character",
    "relationship",
    "mystery",
    "world",
}
_ARC_STATUSES = {"planned", "active", "paused", "resolved", "abandoned"}
_ARCHIVE_CATEGORIES = {"core", "world", "character", "structure", "style"}

_CHOICE_ALIASES = {
    "arc_type": {
        "主线": "main",
        "支线": "subplot",
        "人物线": "character",
        "角色线": "character",
        "关系线": "relationship",
        "情感线": "relationship",
        "谜题线": "mystery",
        "悬疑线": "mystery",
        "世界线": "world",
    },
    "lifecycle_status": {
        "planning": "planned",
        "计划": "planned",
        "计划中": "planned",
        "进行中": "active",
        "推进中": "active",
        "暂停": "paused",
        "已解决": "resolved",
        "已完成": "resolved",
        "已放弃": "abandoned",
    },
    "category": {
        "作品概览": "core",
        "核心": "core",
        "世界": "world",
        "人物": "character",
        "角色": "character",
        "剧情": "structure",
        "结构": "structure",
        "叙事": "style",
        "文风": "style",
    },
}

_PRIORITY_ALIASES = {
    "high": 1,
    "最高": 1,
    "高": 1,
    "medium": 3,
    "normal": 3,
    "中": 3,
    "low": 5,
    "低": 5,
}


@dataclass(frozen=True)
class FieldRule:
    value_type: str = "str"
    max_length: int = 0
    max_items: int = 0
    item_max_length: int = 0
    choices: frozenset[str] = frozenset()
    minimum: int = 0
    maximum: int = 0
    required: bool = False


def _text(max_length: int, *, required: bool = False) -> FieldRule:
    return FieldRule(
        value_type="str", max_length=max_length, required=required
    )


def _choice(values: Iterable[str], *, required: bool = False) -> FieldRule:
    return FieldRule(
        value_type="str",
        choices=frozenset(values),
        required=required,
    )


def _lines(
    max_items: int, item_max_length: int, *, required: bool = False
) -> FieldRule:
    return FieldRule(
        value_type="list",
        max_items=max_items,
        item_max_length=item_max_length,
        required=required,
    )


FIELD_RULES: dict[str, dict[str, FieldRule]] = {
    "world_entry": {
        "entry_type": _choice(_WORLD_TYPES, required=True),
        "name": _text(120, required=True),
        "description": _text(6000),
        "constraints": _text(4000),
    },
    "character": {
        "name": _text(60, required=True),
        "role": _text(300),
        "external_goal": _text(2000),
        "internal_need": _text(2000),
        "central_conflict": _text(2000),
        "secret": _text(2000),
        "traits": _text(1000),
        "speech_style": _text(2000),
        "background": _text(4000),
        "initial_state": _text(2000),
        "character_arc": _text(2000),
    },
    "relationship": {
        "character_a_name": _text(60, required=True),
        "character_b_name": _text(60, required=True),
        "relationship": _text(3000),
        "tension": _text(3000),
        "change_direction": _text(3000),
    },
    "story_blueprint": {
        "central_question": _text(2000),
        "protagonist_goal": _text(2000),
        "core_conflict": _text(3000),
        "stakes": _text(3000),
        "opening_state": _text(3000),
        "ending_state": _text(3000),
        "major_turns": _lines(20, 1200),
        "must_payoffs": _lines(30, 1200),
        "forbidden_shortcuts": _lines(30, 1200),
        "author_notes": _text(6000),
    },
    "plot_arc": {
        "arc_type": _choice(_ARC_TYPES, required=True),
        "title": _text(160, required=True),
        "dramatic_question": _text(2000, required=True),
        "promise": _text(2000, required=True),
        "start_state": _text(2000),
        "target_payoff": _text(3000, required=True),
        "involved_characters": _lines(30, 120),
        "planned_turns": _lines(20, 1200),
        "lifecycle_status": _choice(_ARC_STATUSES),
        "priority": FieldRule(
            value_type="int", minimum=1, maximum=5
        ),
        "author_notes": _text(4000),
    },
    "voice_profile": {
        "narrative_tense": _text(200),
        "narrative_distance": _text(1000),
        "tone": _text(1000),
        "narration_rules": _text(6000),
        "sentence_rhythm": _text(4000),
        "dialogue_voice": _text(6000),
        "sensory_palette": _text(4000),
        "metaphor_policy": _text(4000),
        "allowed_omissions": _text(4000),
        "preferred_patterns": _lines(50, 500),
        "banned_expressions": _lines(100, 500),
        "style_examples": _lines(30, 1000),
        "author_notes": _text(6000),
    },
    "archive_rule": {
        "category": _choice(_ARCHIVE_CATEGORIES, required=True),
        "title": _text(120),
        "content": _text(6000, required=True),
    },
}

_ENTITY_BUCKETS = {
    "world_entry": "world_entries",
    "character": "characters",
    "relationship": "relationships",
    "plot_arc": "plot_arcs",
    "archive_rule": "archive_rules",
}

_ENTITY_NAME_FIELDS = {
    "world_entry": ("name",),
    "character": ("name",),
    "relationship": ("label",),
    "story_blueprint": ("label",),
    "plot_arc": ("title",),
    "voice_profile": ("label",),
    "archive_rule": ("title",),
}


def _load_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        decoded = json.loads(str(value or "[]"))
    except (TypeError, ValueError):
        return []
    return [str(item) for item in decoded] if isinstance(decoded, list) else []


def _normalize_entry_type(value: str) -> str:
    cleaned = value.strip()
    if cleaned in _WORLD_TYPES:
        return cleaned
    lowered = cleaned.casefold()
    if any(token in cleaned for token in ("规则", "机制", "科技", "能力")):
        return "rule"
    if any(token in cleaned for token in ("组织", "阵营", "机构", "派系")):
        return "faction"
    if any(token in cleaned for token in ("地点", "场所", "区域", "城市")):
        return "location"
    if any(token in cleaned for token in ("背景", "时代", "社会")):
        return "background"
    if lowered in {"technology", "system", "mechanism"}:
        return "rule"
    return "element"


def _normalize_choice_value(field: str, value: str) -> str:
    cleaned = value.strip()
    if field == "entry_type":
        return _normalize_entry_type(cleaned)
    aliases = _CHOICE_ALIASES.get(field, {})
    return aliases.get(cleaned, aliases.get(cleaned.casefold(), cleaned))


def _split_model_lines(value: str) -> list[str]:
    cleaned = value.strip()
    if not cleaned:
        return []
    parts = [
        item.strip()
        for item in re.split(r"(?:\r?\n+|[；;]+)", cleaned)
        if item.strip()
    ]
    return parts or [cleaned]


def _normalize_value(field: str, value: Any, rule: FieldRule) -> Any:
    if rule.value_type == "int":
        if isinstance(value, bool):
            raise ValueError(f"{FIELD_LABELS.get(field, field)}必须是整数")
        if isinstance(value, str):
            alias = _PRIORITY_ALIASES.get(value.strip().casefold())
            if alias is not None:
                value = alias
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{FIELD_LABELS.get(field, field)}必须是整数"
            ) from exc
        if result < rule.minimum or result > rule.maximum:
            raise ValueError(
                f"{FIELD_LABELS.get(field, field)}必须在"
                f"{rule.minimum} 到 {rule.maximum} 之间"
            )
        return result
    if rule.value_type == "list":
        if isinstance(value, str):
            value = _split_model_lines(value)
        if not isinstance(value, list):
            raise ValueError(
                f"{FIELD_LABELS.get(field, field)}必须是字符串列表"
            )
        if len(value) > rule.max_items:
            raise ValueError(
                f"{FIELD_LABELS.get(field, field)}最多包含"
                f"{rule.max_items} 项"
            )
        result = []
        for item in value:
            cleaned = str(item or "").strip()
            if not cleaned:
                continue
            if len(cleaned) > rule.item_max_length:
                raise ValueError(
                    f"{FIELD_LABELS.get(field, field)}单项过长"
                )
            result.append(cleaned)
        if rule.required and not result:
            raise ValueError(f"{FIELD_LABELS.get(field, field)}不能为空")
        return result
    if isinstance(value, list) and all(
        isinstance(item, str) for item in value
    ):
        value = "、".join(
            item.strip() for item in value if item.strip()
        )
    if not isinstance(value, str):
        raise ValueError(f"{FIELD_LABELS.get(field, field)}必须是文本")
    result = value.strip()
    if rule.required and not result:
        raise ValueError(f"{FIELD_LABELS.get(field, field)}不能为空")
    if rule.max_length and len(result) > rule.max_length:
        raise ValueError(f"{FIELD_LABELS.get(field, field)}内容过长")
    if rule.choices:
        result = _normalize_choice_value(field, result)
    if rule.choices and result not in rule.choices:
        raise ValueError(f"{FIELD_LABELS.get(field, field)}取值无效")
    return result


def normalize_changes(
    entity_type: str, changes: Mapping[str, Any]
) -> dict[str, Any]:
    rules = FIELD_RULES.get(entity_type)
    if not rules:
        raise ValueError("不支持的结构化资料类型")
    unknown = sorted(set(changes) - set(rules))
    if unknown:
        raise ValueError(
            f"{ENTITY_LABELS[entity_type]}不支持字段："
            + "、".join(unknown)
        )
    return {
        field: _normalize_value(field, value, rules[field])
        for field, value in changes.items()
    }


def _entity_label(entity_type: str, item: Mapping[str, Any] | None) -> str:
    if not item:
        return ENTITY_LABELS.get(entity_type, entity_type)
    for field in _ENTITY_NAME_FIELDS.get(entity_type, ()):
        if item.get(field):
            return str(item[field])
    return ENTITY_LABELS.get(entity_type, entity_type)


def _ensure_confirmable_settings(
    entity_type: str,
    values: Mapping[str, Any],
) -> None:
    allowed = FIELD_RULES[entity_type]
    payload = {
        field: values[field]
        for field in allowed
        if field in values
    }
    if entity_type == "story_blueprint":
        blueprint = StoryBlueprint.model_validate(payload)
        blueprint.ensure_confirmable()
    elif entity_type == "plot_arc":
        arc = PlannedStoryArc.model_validate(payload)
        arc.ensure_confirmable()


def _matching_entity(
    snapshot: Mapping[str, Any],
    edit: AssistantStructuredSettingEdit,
) -> Mapping[str, Any] | None:
    if edit.entity_type == "story_blueprint":
        item = snapshot.get("story_blueprint")
        return item if isinstance(item, Mapping) else None
    if edit.entity_type == "voice_profile":
        item = snapshot.get("voice_profile")
        return item if isinstance(item, Mapping) else None
    bucket = _ENTITY_BUCKETS[edit.entity_type]
    items = [
        item
        for item in (snapshot.get(bucket) or [])
        if isinstance(item, Mapping)
    ]
    if edit.target_id:
        return next(
            (
                item
                for item in items
                if str(item.get("id") or "") == edit.target_id
            ),
            None,
        )
    target = str(edit.target_name or "").strip().casefold()
    matches = [
        item
        for item in items
        if any(
            str(item.get(field) or "").strip().casefold() == target
            for field in _ENTITY_NAME_FIELDS[edit.entity_type]
        )
    ]
    if len(matches) > 1:
        raise ValueError(
            f"{edit.target_name} 匹配到多个资料对象，请使用对象 id"
        )
    return matches[0] if matches else None


def preview_structured_edits(
    edits: Sequence[AssistantStructuredSettingEdit | Mapping[str, Any]],
    snapshot: Mapping[str, Any],
    *,
    allow_noop: bool = False,
) -> list[dict[str, Any]]:
    previews: list[dict[str, Any]] = []
    seen_targets: set[tuple[str, str]] = set()
    for index, raw in enumerate(edits):
        edit = (
            raw
            if isinstance(raw, AssistantStructuredSettingEdit)
            else AssistantStructuredSettingEdit.model_validate(raw)
        )
        changes = normalize_changes(edit.entity_type, edit.changes)
        before = _matching_entity(snapshot, edit)
        if edit.action == "create" and before:
            raise ValueError(
                f"{_entity_label(edit.entity_type, before)}已经存在"
            )
        if edit.action in {"update", "delete"} and not before:
            raise ValueError(
                f"找不到要{ACTION_LABELS[edit.action]}的"
                f"{ENTITY_LABELS[edit.entity_type]}："
                f"{edit.target_name or edit.target_id}"
            )
        target_key = (
            edit.entity_type,
            str((before or {}).get("id") or edit.target_name or index),
        )
        if target_key in seen_targets:
            raise ValueError("同一份资料不能在一轮候选中重复修改")
        seen_targets.add(target_key)
        if edit.action == "create":
            after = dict(changes)
        elif edit.action == "delete":
            after = None
        else:
            after = {**dict(before or {}), **changes}
        if after is not None and edit.action != "delete":
            _ensure_confirmable_settings(edit.entity_type, after)
        fields = [
            {
                "field": field,
                "label": FIELD_LABELS.get(field, field),
                "before": (before or {}).get(field),
                "after": value,
                "changed": (before or {}).get(field) != value,
            }
            for field, value in changes.items()
            if (before or {}).get(field) != value
        ]
        if edit.action == "update" and not fields and not allow_noop:
            raise ValueError(
                f"{_entity_label(edit.entity_type, before)}没有实际变化"
            )
        previews.append(
            {
                "index": index,
                "entity_type": edit.entity_type,
                "entity_label": ENTITY_LABELS[edit.entity_type],
                "action": edit.action,
                "action_label": ACTION_LABELS[edit.action],
                "target_id": str((before or {}).get("id") or ""),
                "target_label": (
                    _entity_label(edit.entity_type, before)
                    if before
                    else str(
                        changes.get("name")
                        or changes.get("title")
                        or edit.target_name
                        or ENTITY_LABELS[edit.entity_type]
                    )
                ),
                "reason": edit.reason,
                "fields": fields,
                "before": dict(before) if before else None,
                "after": after,
            }
        )
    return previews


class StructuredSettingsEditor:
    def __init__(self, database: Database):
        self.database = database

    def snapshot(
        self, *, user_id: int, project_id: str
    ) -> dict[str, Any]:
        with self.database.connection() as connection:
            return self.snapshot_in_connection(
                connection, user_id=user_id, project_id=project_id
            )

    @staticmethod
    def snapshot_in_connection(
        connection: sqlite3.Connection,
        *,
        user_id: int,
        project_id: str,
    ) -> dict[str, Any]:
        project = connection.execute(
            """
            SELECT id FROM novel_projects WHERE id=? AND user_id=?
            """,
            (project_id, user_id),
        ).fetchone()
        if not project:
            raise ValueError("小说项目不存在")
        world_rows = connection.execute(
            """
            SELECT id, entry_type, name, description, constraints, updated_at
            FROM novel_world_entries
            WHERE project_id=? ORDER BY position
            """,
            (project_id,),
        ).fetchall()
        character_rows = connection.execute(
            """
            SELECT id, name, role, external_goal, internal_need,
                   central_conflict, hidden_fact AS secret, traits,
                   speech_style, background, initial_state, character_arc,
                   updated_at
            FROM novel_characters
            WHERE project_id=? ORDER BY position
            """,
            (project_id,),
        ).fetchall()
        relationship_rows = connection.execute(
            """
            SELECT relation.id, relation.character_a_id,
                   relation.character_b_id,
                   first.name AS character_a_name,
                   second.name AS character_b_name,
                   relation.relationship, relation.tension,
                   relation.change_direction, relation.updated_at
            FROM novel_character_relationships relation
            JOIN novel_characters first
              ON first.id=relation.character_a_id
            JOIN novel_characters second
              ON second.id=relation.character_b_id
            WHERE relation.project_id=? ORDER BY relation.position
            """,
            (project_id,),
        ).fetchall()
        relationships = []
        for row in relationship_rows:
            item = dict(row)
            item["label"] = (
                f"{item['character_a_name']} × {item['character_b_name']}"
            )
            relationships.append(item)
        voice_row = connection.execute(
            """
            SELECT * FROM novel_voice_profiles WHERE project_id=?
            """,
            (project_id,),
        ).fetchone()
        voice = dict(voice_row) if voice_row else {}
        if voice:
            for stored, public in (
                ("preferred_patterns_json", "preferred_patterns"),
                ("banned_expressions_json", "banned_expressions"),
                ("style_examples_json", "style_examples"),
            ):
                voice[public] = _load_list(voice.pop(stored, "[]"))
            voice["label"] = "叙事与文风"
        blueprint_row = connection.execute(
            """
            SELECT version.*
            FROM novel_story_blueprint_heads head
            JOIN novel_story_blueprint_versions version
              ON version.id=head.current_version_id
            WHERE head.project_id=?
            """,
            (project_id,),
        ).fetchone()
        blueprint = dict(blueprint_row) if blueprint_row else {}
        if blueprint:
            for stored, public in (
                ("major_turns_json", "major_turns"),
                ("must_payoffs_json", "must_payoffs"),
                ("forbidden_shortcuts_json", "forbidden_shortcuts"),
            ):
                blueprint[public] = _load_list(
                    blueprint.pop(stored, "[]")
                )
            blueprint["label"] = "全书蓝图"
        arc_rows = connection.execute(
            """
            SELECT arc.id, arc.updated_at,
                   version.id AS current_version_id,
                   version.arc_type, version.title,
                   version.dramatic_question, version.promise,
                   version.start_state, version.target_payoff,
                   version.involved_characters_json,
                   version.planned_turns_json,
                   version.lifecycle_status, version.priority,
                   version.author_notes, version.version_status,
                   version.revision
            FROM novel_plot_arcs arc
            JOIN novel_plot_arc_versions version
              ON version.id=arc.current_version_id
            WHERE arc.project_id=? ORDER BY arc.position
            """,
            (project_id,),
        ).fetchall()
        arcs = []
        for row in arc_rows:
            item = dict(row)
            item["involved_characters"] = _load_list(
                item.pop("involved_characters_json", "[]")
            )
            item["planned_turns"] = _load_list(
                item.pop("planned_turns_json", "[]")
            )
            arcs.append(item)
        rule_rows = connection.execute(
            """
            SELECT entry.id, entry.category, entry.title, entry.content,
                   entry.updated_at
            FROM work_versions version
            JOIN work_archive_entries entry
              ON entry.work_id=version.work_id
            WHERE version.project_id=?
              AND version.ref_type='branch'
              AND version.ref_name='main'
              AND entry.entry_type='creative_rule'
              AND entry.status='confirmed'
            ORDER BY entry.updated_at DESC
            """,
            (project_id,),
        ).fetchall()
        return {
            "schema_version": 1,
            "world_entries": [dict(row) for row in world_rows],
            "characters": [dict(row) for row in character_rows],
            "relationships": relationships,
            "story_blueprint": blueprint or None,
            "plot_arcs": arcs,
            "voice_profile": voice or None,
            "archive_rules": [dict(row) for row in rule_rows],
        }

    def apply_in_connection(
        self,
        connection: sqlite3.Connection,
        *,
        user_id: int,
        project_id: str,
        edits: Sequence[AssistantStructuredSettingEdit],
        baseline_snapshot: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        current_snapshot = self.snapshot_in_connection(
            connection, user_id=user_id, project_id=project_id
        )
        previews = preview_structured_edits(edits, baseline_snapshot)
        current_previews = preview_structured_edits(
            edits, current_snapshot, allow_noop=True
        )
        for baseline, current in zip(
            previews, current_previews, strict=True
        ):
            if baseline["action"] == "create":
                continue
            baseline_before = baseline.get("before") or {}
            current_before = current.get("before") or {}
            if baseline["action"] == "delete":
                if (
                    current_before.get("updated_at")
                    != baseline_before.get("updated_at")
                ):
                    raise ValueError(
                        f"{baseline['target_label']}在讨论后已经变化，"
                        "请让 AI 基于最新资料重新修改"
                    )
                continue
            for field in (item["field"] for item in baseline["fields"]):
                if (
                    current_before.get(field)
                    != baseline_before.get(field)
                    and current_before.get(field)
                    != (baseline.get("after") or {}).get(field)
                ):
                    raise ValueError(
                        f"{baseline['target_label']}的"
                        f"{FIELD_LABELS.get(field, field)}在讨论后已经变化，"
                        "请让 AI 基于最新资料重新修改"
                    )
        applied: list[dict[str, Any]] = []
        now = utc_now()
        for edit, preview in zip(edits, previews, strict=True):
            changes = normalize_changes(edit.entity_type, edit.changes)
            target_id = self._apply_one(
                connection,
                user_id=user_id,
                project_id=project_id,
                edit=edit,
                preview=preview,
                changes=changes,
                now=now,
            )
            applied.append(
                {
                    **preview,
                    "target_id": target_id or preview.get("target_id") or "",
                }
            )
        connection.execute(
            "UPDATE novel_projects SET updated_at=? WHERE id=? AND user_id=?",
            (now, project_id, user_id),
        )
        return applied

    def _apply_one(
        self,
        connection: sqlite3.Connection,
        *,
        user_id: int,
        project_id: str,
        edit: AssistantStructuredSettingEdit,
        preview: Mapping[str, Any],
        changes: Mapping[str, Any],
        now: str,
    ) -> str:
        target_id = str(preview.get("target_id") or "")
        if edit.entity_type == "world_entry":
            return self._apply_world(
                connection, project_id, edit.action, target_id, changes, now
            )
        if edit.entity_type == "character":
            return self._apply_character(
                connection, project_id, edit.action, target_id, changes, now
            )
        if edit.entity_type == "relationship":
            return self._apply_relationship(
                connection,
                project_id,
                edit.action,
                target_id,
                changes,
                preview.get("before") or {},
                now,
            )
        if edit.entity_type == "voice_profile":
            return self._apply_voice(
                connection,
                project_id,
                edit.action,
                target_id,
                changes,
                preview.get("before") or {},
                now,
            )
        if edit.entity_type == "archive_rule":
            return self._apply_archive_rule(
                connection,
                project_id,
                edit.action,
                target_id,
                changes,
                preview.get("before") or {},
                now,
            )
        if edit.entity_type == "story_blueprint":
            return self._apply_blueprint(
                connection,
                project_id,
                edit.action,
                changes,
                preview.get("before") or {},
                now,
            )
        if edit.entity_type == "plot_arc":
            return self._apply_plot_arc(
                connection,
                project_id,
                edit.action,
                target_id,
                changes,
                preview.get("before") or {},
                now,
            )
        raise ValueError("不支持的结构化资料类型")

    @staticmethod
    def _apply_world(
        connection: sqlite3.Connection,
        project_id: str,
        action: str,
        target_id: str,
        changes: Mapping[str, Any],
        now: str,
    ) -> str:
        defaults = {
            "entry_type": "background",
            "name": "",
            "description": "",
            "constraints": "",
        }
        if action == "create":
            values = {**defaults, **changes}
            if not values["name"]:
                raise ValueError("世界资料名称不能为空")
            target_id = uuid.uuid4().hex
            position = connection.execute(
                """
                SELECT COALESCE(MAX(position), 0) + 1 AS value
                FROM novel_world_entries WHERE project_id=?
                """,
                (project_id,),
            ).fetchone()["value"]
            connection.execute(
                """
                INSERT INTO novel_world_entries(
                    id, project_id, position, entry_type, name,
                    description, constraints, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    target_id,
                    project_id,
                    int(position),
                    values["entry_type"],
                    values["name"],
                    values["description"],
                    values["constraints"],
                    now,
                    now,
                ),
            )
            return target_id
        if action == "delete":
            connection.execute(
                "DELETE FROM novel_world_entries WHERE id=? AND project_id=?",
                (target_id, project_id),
            )
            return target_id
        assignments = ", ".join(f"{field}=?" for field in changes)
        connection.execute(
            f"""
            UPDATE novel_world_entries
            SET {assignments}, updated_at=?
            WHERE id=? AND project_id=?
            """,
            (*changes.values(), now, target_id, project_id),
        )
        return target_id

    @staticmethod
    def _apply_character(
        connection: sqlite3.Connection,
        project_id: str,
        action: str,
        target_id: str,
        changes: Mapping[str, Any],
        now: str,
    ) -> str:
        defaults = {
            "name": "",
            "role": "",
            "external_goal": "",
            "internal_need": "",
            "central_conflict": "",
            "secret": "",
            "traits": "",
            "speech_style": "",
            "background": "",
            "initial_state": "",
            "character_arc": "",
        }
        storage_fields = {
            "secret": "hidden_fact",
        }
        if action == "create":
            values = {**defaults, **changes}
            if not values["name"]:
                raise ValueError("人物名不能为空")
            target_id = uuid.uuid4().hex
            position = connection.execute(
                """
                SELECT COALESCE(MAX(position), 0) + 1 AS value
                FROM novel_characters WHERE project_id=?
                """,
                (project_id,),
            ).fetchone()["value"]
            connection.execute(
                """
                INSERT INTO novel_characters(
                    id, project_id, position, name, role, traits, background,
                    character_arc, external_goal, internal_need,
                    central_conflict, hidden_fact, speech_style, initial_state,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    target_id,
                    project_id,
                    int(position),
                    values["name"],
                    values["role"],
                    values["traits"],
                    values["background"],
                    values["character_arc"],
                    values["external_goal"],
                    values["internal_need"],
                    values["central_conflict"],
                    values["secret"],
                    values["speech_style"],
                    values["initial_state"],
                    now,
                    now,
                ),
            )
            ensure_memory_identity(
                connection,
                project_id=project_id,
                identity_type="character",
                canonical_text=str(values["name"]),
                created_at=now,
                source="project",
                linked_record_id=target_id,
            )
            return target_id
        if action == "delete":
            connection.execute(
                "DELETE FROM novel_characters WHERE id=? AND project_id=?",
                (target_id, project_id),
            )
            connection.execute(
                """
                UPDATE memory_identities
                SET linked_record_id=NULL, updated_at=?
                WHERE project_id=? AND linked_record_id=?
                """,
                (now, project_id, target_id),
            )
            return target_id
        assignments = ", ".join(
            f"{storage_fields.get(field, field)}=?" for field in changes
        )
        connection.execute(
            f"""
            UPDATE novel_characters
            SET {assignments}, updated_at=?
            WHERE id=? AND project_id=?
            """,
            (*changes.values(), now, target_id, project_id),
        )
        if "name" in changes:
            connection.execute(
                """
                UPDATE memory_identities
                SET linked_record_id=NULL, updated_at=?
                WHERE project_id=? AND linked_record_id=?
                """,
                (now, project_id, target_id),
            )
            ensure_memory_identity(
                connection,
                project_id=project_id,
                identity_type="character",
                canonical_text=str(changes["name"]),
                created_at=now,
                source="project",
                linked_record_id=target_id,
            )
        return target_id

    @staticmethod
    def _character_id(
        connection: sqlite3.Connection,
        project_id: str,
        name: str,
    ) -> str:
        row = connection.execute(
            """
            SELECT id FROM novel_characters
            WHERE project_id=? AND lower(name)=lower(?)
            """,
            (project_id, name),
        ).fetchone()
        if not row:
            raise ValueError(f"找不到人物：{name}")
        return str(row["id"])

    @classmethod
    def _apply_relationship(
        cls,
        connection: sqlite3.Connection,
        project_id: str,
        action: str,
        target_id: str,
        changes: Mapping[str, Any],
        before: Mapping[str, Any],
        now: str,
    ) -> str:
        if action == "delete":
            connection.execute(
                """
                DELETE FROM novel_character_relationships
                WHERE id=? AND project_id=?
                """,
                (target_id, project_id),
            )
            return target_id
        values = {
            "character_a_name": str(
                before.get("character_a_name") or ""
            ),
            "character_b_name": str(
                before.get("character_b_name") or ""
            ),
            "relationship": str(before.get("relationship") or ""),
            "tension": str(before.get("tension") or ""),
            "change_direction": str(before.get("change_direction") or ""),
            **changes,
        }
        first_id, second_id = sorted(
            (
                cls._character_id(
                    connection,
                    project_id,
                    str(values["character_a_name"]),
                ),
                cls._character_id(
                    connection,
                    project_id,
                    str(values["character_b_name"]),
                ),
            )
        )
        if first_id == second_id:
            raise ValueError("人物关系必须连接两个不同人物")
        if action == "create":
            target_id = uuid.uuid4().hex
            position = connection.execute(
                """
                SELECT COALESCE(MAX(position), 0) + 1 AS value
                FROM novel_character_relationships WHERE project_id=?
                """,
                (project_id,),
            ).fetchone()["value"]
            connection.execute(
                """
                INSERT INTO novel_character_relationships(
                    id, project_id, position,
                    character_a_id, character_b_id,
                    relationship, tension, change_direction,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    target_id,
                    project_id,
                    int(position),
                    first_id,
                    second_id,
                    values["relationship"],
                    values["tension"],
                    values["change_direction"],
                    now,
                    now,
                ),
            )
            return target_id
        connection.execute(
            """
            UPDATE novel_character_relationships
            SET character_a_id=?, character_b_id=?, relationship=?,
                tension=?, change_direction=?, updated_at=?
            WHERE id=? AND project_id=?
            """,
            (
                first_id,
                second_id,
                values["relationship"],
                values["tension"],
                values["change_direction"],
                now,
                target_id,
                project_id,
            ),
        )
        return target_id

    @staticmethod
    def _apply_voice(
        connection: sqlite3.Connection,
        project_id: str,
        action: str,
        target_id: str,
        changes: Mapping[str, Any],
        before: Mapping[str, Any],
        now: str,
    ) -> str:
        if action != "update":
            raise ValueError("叙事与文风只支持局部修改")
        list_fields = {
            "preferred_patterns",
            "banned_expressions",
            "style_examples",
        }
        storage = {
            field: f"{field}_json" if field in list_fields else field
            for field in changes
        }
        assignments = ", ".join(
            f"{storage[field]}=?" for field in changes
        )
        values = [
            (
                json.dumps(value, ensure_ascii=False)
                if field in list_fields
                else value
            )
            for field, value in changes.items()
        ]
        connection.execute(
            f"""
            UPDATE novel_voice_profiles
            SET {assignments}, status='confirmed', confirmed_at=?,
                updated_at=?
            WHERE project_id=?
            """,
            (*values, now, now, project_id),
        )
        return str(target_id or before.get("id") or project_id)

    @staticmethod
    def _main_work(
        connection: sqlite3.Connection, project_id: str
    ) -> sqlite3.Row:
        row = connection.execute(
            """
            SELECT version.id AS content_version_id, version.work_id
            FROM work_versions version
            WHERE version.project_id=? AND version.ref_type='branch'
              AND version.ref_name='main' AND version.is_editable=1
            """,
            (project_id,),
        ).fetchone()
        if not row:
            raise ValueError("作品缺少可写的 main 分支")
        return row

    @classmethod
    def _apply_archive_rule(
        cls,
        connection: sqlite3.Connection,
        project_id: str,
        action: str,
        target_id: str,
        changes: Mapping[str, Any],
        before: Mapping[str, Any],
        now: str,
    ) -> str:
        main = cls._main_work(connection, project_id)
        if action == "delete":
            connection.execute(
                """
                DELETE FROM work_archive_entries
                WHERE id=? AND work_id=? AND entry_type='creative_rule'
                """,
                (target_id, main["work_id"]),
            )
            return target_id
        values = {
            "category": str(before.get("category") or "core"),
            "title": str(before.get("title") or ""),
            "content": str(before.get("content") or ""),
            **changes,
        }
        if action == "create":
            target_id = uuid.uuid4().hex
            connection.execute(
                """
                INSERT INTO work_archive_entries(
                    id, work_id, content_version_id,
                    entry_type, title, content, provenance, status,
                    evidence, source_ref, category, adopted_at,
                    created_at, updated_at
                ) VALUES (
                    ?, ?, ?, 'creative_rule', ?, ?, 'assistant',
                    'confirmed', '', '', ?, ?, ?, ?
                )
                """,
                (
                    target_id,
                    main["work_id"],
                    main["content_version_id"],
                    values["title"],
                    values["content"],
                    values["category"],
                    now,
                    now,
                    now,
                ),
            )
            return target_id
        connection.execute(
            """
            UPDATE work_archive_entries
            SET category=?, title=?, content=?, updated_at=?
            WHERE id=? AND work_id=? AND entry_type='creative_rule'
            """,
            (
                values["category"],
                values["title"],
                values["content"],
                now,
                target_id,
                main["work_id"],
            ),
        )
        return target_id

    @staticmethod
    def _apply_blueprint(
        connection: sqlite3.Connection,
        project_id: str,
        action: str,
        changes: Mapping[str, Any],
        before: Mapping[str, Any],
        now: str,
    ) -> str:
        if action == "delete":
            raise ValueError("全书蓝图不能删除，可以局部修改")
        fields = (
            "central_question",
            "protagonist_goal",
            "core_conflict",
            "stakes",
            "opening_state",
            "ending_state",
            "major_turns",
            "must_payoffs",
            "forbidden_shortcuts",
            "author_notes",
        )
        values = {
            field: before.get(field, [] if field in {
                "major_turns", "must_payoffs", "forbidden_shortcuts"
            } else "")
            for field in fields
        }
        values.update(changes)
        blueprint = StoryBlueprint.model_validate(values)
        blueprint.ensure_confirmable()
        version_id = uuid.uuid4().hex
        revision = connection.execute(
            """
            SELECT COALESCE(MAX(revision), 0) + 1 AS value
            FROM novel_story_blueprint_versions WHERE project_id=?
            """,
            (project_id,),
        ).fetchone()["value"]
        connection.execute(
            """
            INSERT INTO novel_story_blueprint_versions(
                id, project_id, revision, version_status,
                central_question, protagonist_goal, core_conflict,
                stakes, opening_state, ending_state,
                major_turns_json, must_payoffs_json,
                forbidden_shortcuts_json, author_notes, source,
                created_at, confirmed_at
            ) VALUES (
                ?, ?, ?, 'confirmed', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                'assistant', ?, ?
            )
            """,
            (
                version_id,
                project_id,
                int(revision),
                blueprint.central_question,
                blueprint.protagonist_goal,
                blueprint.core_conflict,
                blueprint.stakes,
                blueprint.opening_state,
                blueprint.ending_state,
                json.dumps(blueprint.major_turns, ensure_ascii=False),
                json.dumps(blueprint.must_payoffs, ensure_ascii=False),
                json.dumps(
                    blueprint.forbidden_shortcuts, ensure_ascii=False
                ),
                blueprint.author_notes,
                now,
                now,
            ),
        )
        head = connection.execute(
            """
            SELECT 1 FROM novel_story_blueprint_heads WHERE project_id=?
            """,
            (project_id,),
        ).fetchone()
        if head:
            connection.execute(
                """
                UPDATE novel_story_blueprint_heads
                SET current_version_id=?, confirmed_version_id=?,
                    updated_at=?
                WHERE project_id=?
                """,
                (version_id, version_id, now, project_id),
            )
        else:
            connection.execute(
                """
                INSERT INTO novel_story_blueprint_heads(
                    project_id, current_version_id,
                    confirmed_version_id, updated_at
                ) VALUES (?, ?, ?, ?)
                """,
                (project_id, version_id, version_id, now),
            )
        StoryPlanningService._invalidate_future_plans(
            connection, project_id=project_id, now=now
        )
        return version_id

    @staticmethod
    def _apply_plot_arc(
        connection: sqlite3.Connection,
        project_id: str,
        action: str,
        target_id: str,
        changes: Mapping[str, Any],
        before: Mapping[str, Any],
        now: str,
    ) -> str:
        fields = (
            "arc_type",
            "title",
            "dramatic_question",
            "promise",
            "start_state",
            "target_payoff",
            "involved_characters",
            "planned_turns",
            "lifecycle_status",
            "priority",
            "author_notes",
        )
        values = {
            field: before.get(
                field,
                [] if field in {"involved_characters", "planned_turns"}
                else 3 if field == "priority"
                else "subplot" if field == "arc_type"
                else "planned" if field == "lifecycle_status"
                else "",
            )
            for field in fields
        }
        values.update(changes)
        if action == "delete":
            values["lifecycle_status"] = "abandoned"
        arc = PlannedStoryArc.model_validate(values)
        arc.ensure_confirmable()
        created = action == "create"
        if created:
            target_id = uuid.uuid4().hex
            position = connection.execute(
                """
                SELECT COALESCE(MAX(position), 0) + 1 AS value
                FROM novel_plot_arcs WHERE project_id=?
                """,
                (project_id,),
            ).fetchone()["value"]
            connection.execute(
                """
                INSERT INTO novel_plot_arcs(
                    id, project_id, position, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (target_id, project_id, int(position), now, now),
            )
            revision = 1
        else:
            revision = connection.execute(
                """
                SELECT COALESCE(MAX(revision), 0) + 1 AS value
                FROM novel_plot_arc_versions WHERE arc_id=?
                """,
                (target_id,),
            ).fetchone()["value"]
        version_id = uuid.uuid4().hex
        connection.execute(
            """
            INSERT INTO novel_plot_arc_versions(
                id, arc_id, project_id, revision, version_status,
                arc_type, title, dramatic_question, promise,
                start_state, target_payoff, involved_characters_json,
                planned_turns_json, lifecycle_status, priority,
                author_notes, source, created_at, confirmed_at
            ) VALUES (
                ?, ?, ?, ?, 'confirmed', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, 'assistant', ?, ?
            )
            """,
            (
                version_id,
                target_id,
                project_id,
                int(revision),
                arc.arc_type,
                arc.title,
                arc.dramatic_question,
                arc.promise,
                arc.start_state,
                arc.target_payoff,
                json.dumps(arc.involved_characters, ensure_ascii=False),
                json.dumps(arc.planned_turns, ensure_ascii=False),
                arc.lifecycle_status,
                arc.priority,
                arc.author_notes,
                now,
                now,
            ),
        )
        connection.execute(
            """
            UPDATE novel_plot_arcs
            SET current_version_id=?, confirmed_version_id=?, updated_at=?
            WHERE id=? AND project_id=?
            """,
            (version_id, version_id, now, target_id, project_id),
        )
        StoryPlanningService._invalidate_future_plans(
            connection,
            project_id=project_id,
            now=now,
            arc_titles={
                str(before.get("title") or ""),
                arc.title,
            },
        )
        return target_id


def filter_structured_edits(
    edits: Sequence[AssistantStructuredSettingEdit],
    selected_paths: set[str] | None,
) -> list[AssistantStructuredSettingEdit]:
    if selected_paths is None:
        return list(edits)
    result: list[AssistantStructuredSettingEdit] = []
    for index, edit in enumerate(edits):
        if edit.action in {"create", "delete"}:
            if f"structured:{index}:__action__" in selected_paths:
                result.append(edit)
            continue
        selected_changes = {
            field: value
            for field, value in edit.changes.items()
            if f"structured:{index}:{field}" in selected_paths
        }
        if selected_changes:
            result.append(edit.model_copy(update={"changes": selected_changes}))
    return result
