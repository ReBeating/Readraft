from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import uuid
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .memory_identity import (
    list_identity_context,
    load_identity_index,
    normalize_identity_text,
    resolve_identity,
)
from .memory_schema import StoryDelta


STATE_SCHEMA_VERSION = 3


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _load_json(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    try:
        return json.loads(str(value))
    except (TypeError, ValueError):
        return fallback


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _same(left: Any, right: Any) -> bool:
    return _clean(left).casefold() == _clean(right).casefold()


def _empty_state() -> Dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "story_time": None,
        "characters": {},
        "relationships": {},
        "locations": {},
        "items": {},
        "knowledge": {},
        "plot_threads": {},
        "foreshadowing": {},
        "events": {},
        "causal_edges": {},
        "last_chapter": None,
    }


def _source(row: Mapping[str, Any], evidence: str = "") -> Dict[str, Any]:
    return {
        "chapter_id": str(row["chapter_id"]),
        "chapter_position": int(row["chapter_position"]),
        "chapter_title": str(row["chapter_title"]),
        "version_id": str(row["version_id"]),
        "delta_id": str(row["id"]),
        "evidence": _clean(evidence),
    }


def _value(entry: Any) -> Optional[str]:
    if isinstance(entry, Mapping):
        value = _clean(entry.get("value"))
        return value or None
    value = _clean(entry)
    return value or None


def _relationship_key(character_a: str, character_b: str) -> str:
    names = sorted(
        (_clean(character_a), _clean(character_b)),
        key=lambda value: value.casefold(),
    )
    return "\u241f".join(names)


def _existing_key(values: Mapping[str, Any], proposed: str) -> str:
    normalized = _clean(proposed).casefold()
    for key in values:
        if str(key).casefold() == normalized:
            return str(key)
    return _clean(proposed)


def _resolved(
    identity_index: Mapping[str, Mapping[str, Mapping[str, Any]]],
    identity_type: str,
    text: Any,
) -> Dict[str, Any]:
    return resolve_identity(
        identity_index,
        identity_type=identity_type,
        text=text,
    )


def _canonical(
    identity_index: Mapping[str, Mapping[str, Mapping[str, Any]]],
    identity_type: str,
    text: Any,
) -> str:
    return _clean(
        _resolved(identity_index, identity_type, text).get(
            "canonical_text"
        )
    )


def _issue_fingerprint(
    *,
    project_id: str,
    branch_id: str,
    chapter_id: str,
    issue_type: str,
    entity_type: str,
    entity_name: str,
    field_name: str,
    expected_value: str,
    actual_value: str,
) -> str:
    material = "\u241e".join(
        (
            project_id,
            branch_id,
            chapter_id,
            issue_type,
            entity_type,
            entity_name,
            field_name,
            expected_value,
            actual_value,
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _add_issue(
    issues: Dict[str, Dict[str, Any]],
    *,
    row: Mapping[str, Any],
    project_id: str,
    branch_id: str,
    issue_type: str,
    severity: str,
    entity_type: str,
    entity_name: str,
    field_name: str,
    expected_value: str,
    actual_value: str,
    message: str,
    evidence: str,
) -> None:
    fingerprint = _issue_fingerprint(
        project_id=project_id,
        branch_id=branch_id,
        chapter_id=str(row["chapter_id"]),
        issue_type=issue_type,
        entity_type=entity_type,
        entity_name=entity_name,
        field_name=field_name,
        expected_value=expected_value,
        actual_value=actual_value,
    )
    if fingerprint in issues:
        return
    issues[fingerprint] = {
        "id": f"continuity-{fingerprint[:32]}",
        "fingerprint": fingerprint,
        "project_id": project_id,
        "branch_id": branch_id,
        "chapter_id": str(row["chapter_id"]),
        "version_id": str(row["version_id"]),
        "delta_id": str(row["id"]),
        "chapter_position": int(row["chapter_position"]),
        "issue_type": issue_type,
        "severity": severity,
        "entity_type": entity_type,
        "entity_name": entity_name,
        "field_name": field_name,
        "expected_value": expected_value,
        "actual_value": actual_value,
        "message": message,
        "evidence": _clean(evidence),
    }


def _check_declared_before(
    issues: Dict[str, Dict[str, Any]],
    *,
    row: Mapping[str, Any],
    project_id: str,
    branch_id: str,
    declared: Any,
    current: Any,
    chapter_start: Any,
    has_prior_chapter: bool,
    mismatch_type: str,
    severity: str,
    entity_type: str,
    entity_name: str,
    field_name: str,
    evidence: str,
) -> None:
    declared_value = _clean(declared)
    if not declared_value:
        return
    current_value = _value(current)
    start_value = _value(chapter_start)
    if current_value and (
        _same(declared_value, current_value)
        or (start_value and _same(declared_value, start_value))
    ):
        return
    if current_value:
        _add_issue(
            issues,
            row=row,
            project_id=project_id,
            branch_id=branch_id,
            issue_type=mismatch_type,
            severity=severity,
            entity_type=entity_type,
            entity_name=entity_name,
            field_name=field_name,
            expected_value=current_value,
            actual_value=declared_value,
            message=(
                f"{entity_name} 的“变化前”状态与此前正史不一致："
                f"正史为“{current_value}”，本章声明为“{declared_value}”。"
            ),
            evidence=evidence,
        )
        return
    if has_prior_chapter:
        _add_issue(
            issues,
            row=row,
            project_id=project_id,
            branch_id=branch_id,
            issue_type="missing_baseline",
            severity="warning",
            entity_type=entity_type,
            entity_name=entity_name,
            field_name=field_name,
            expected_value="此前正史中可核对的状态",
            actual_value=declared_value,
            message=(
                f"{entity_name} 在本章声明了变化前状态“{declared_value}”，"
                "但此前已确认记忆中没有可核对的基线。"
            ),
            evidence=evidence,
        )


def _character_attribute(
    state: Mapping[str, Any], character_name: str, aspect: str
) -> Any:
    character = (state.get("characters") or {}).get(character_name) or {}
    return (character.get("attributes") or {}).get(aspect)


def _apply_knowledge_changes(
    *,
    state: Dict[str, Any],
    delta: StoryDelta,
    row: Mapping[str, Any],
    project_id: str,
    branch_id: str,
    has_prior_chapter: bool,
    issues: Dict[str, Dict[str, Any]],
    identity_index: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    for change in delta.knowledge_changes:
        character_identity = _resolved(
            identity_index, "character", change.character_name
        )
        character_name = _clean(character_identity["canonical_text"])
        fact_identity = _resolved(
            identity_index,
            "fact",
            change.canonical_fact or change.fact,
        )
        fact = _clean(fact_identity["canonical_text"])
        character_knowledge = state["knowledge"].setdefault(
            character_name, {}
        )
        fact_key = _existing_key(character_knowledge, fact)
        current = character_knowledge.get(fact_key)
        if (
            str(change.state) == "forgets"
            and not current
            and has_prior_chapter
        ):
            _add_issue(
                issues,
                row=row,
                project_id=project_id,
                branch_id=branch_id,
                issue_type="knowledge_without_baseline",
                severity="warning",
                entity_type="knowledge",
                entity_name=character_name,
                field_name=fact,
                expected_value="此前有获得或相信该信息的记录",
                actual_value="forgets",
                message=(
                    f"{character_name} 在本章遗忘“{fact}”，"
                    "但此前正史中没有可核对的知情记录。"
                ),
                evidence=change.evidence,
            )
        character_knowledge[fact_key] = {
            "fact": fact,
            "statement": _clean(change.fact),
            "identity_id": fact_identity.get("id"),
            "character_identity_id": character_identity.get("id"),
            "state": str(change.state),
            "learned_via": _clean(change.learned_via),
            "source": _source(row, change.evidence),
        }


def _apply_plot_thread_changes(
    *,
    state: Dict[str, Any],
    delta: StoryDelta,
    row: Mapping[str, Any],
    project_id: str,
    branch_id: str,
    has_prior_chapter: bool,
    issues: Dict[str, Dict[str, Any]],
    identity_index: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    status_for_action = {
        "opened": "open",
        "advanced": "active",
        "paused": "paused",
        "resolved": "resolved",
        "abandoned": "abandoned",
    }
    closed_statuses = {"resolved", "abandoned"}
    for change in delta.plot_thread_changes:
        identity = _resolved(
            identity_index, "plot_thread", change.thread_name
        )
        name = _clean(identity["canonical_text"])
        key = _existing_key(state["plot_threads"], name)
        current = state["plot_threads"].get(key)
        action = str(change.action)
        current_status = _clean((current or {}).get("status"))
        if not current and action != "opened" and has_prior_chapter:
            _add_issue(
                issues,
                row=row,
                project_id=project_id,
                branch_id=branch_id,
                issue_type="plot_thread_without_setup",
                severity="warning",
                entity_type="plot_thread",
                entity_name=name,
                field_name="status",
                expected_value="此前已建立的剧情线",
                actual_value=action,
                message=(
                    f"剧情线“{name}”在本章执行 {action}，"
                    "但此前正史中没有建立记录。"
                ),
                evidence=change.evidence,
            )
        elif current:
            if action == "opened" and current_status not in closed_statuses:
                _add_issue(
                    issues,
                    row=row,
                    project_id=project_id,
                    branch_id=branch_id,
                    issue_type="plot_thread_duplicate_open",
                    severity="warning",
                    entity_type="plot_thread",
                    entity_name=name,
                    field_name="status",
                    expected_value=current_status,
                    actual_value=action,
                    message=f"剧情线“{name}”尚未关闭，本章却再次建立。",
                    evidence=change.evidence,
                )
            elif action == "opened" and current_status in closed_statuses:
                _add_issue(
                    issues,
                    row=row,
                    project_id=project_id,
                    branch_id=branch_id,
                    issue_type="plot_thread_reopened",
                    severity="warning",
                    entity_type="plot_thread",
                    entity_name=name,
                    field_name="status",
                    expected_value=current_status,
                    actual_value=action,
                    message=(
                        f"剧情线“{name}”已经{current_status}，"
                        "本章重新建立；请确认这是有意重启。"
                    ),
                    evidence=change.evidence,
                )
            elif (
                current_status in closed_statuses
                and status_for_action.get(action) != current_status
            ):
                _add_issue(
                    issues,
                    row=row,
                    project_id=project_id,
                    branch_id=branch_id,
                    issue_type="plot_thread_after_closed",
                    severity="hard",
                    entity_type="plot_thread",
                    entity_name=name,
                    field_name="status",
                    expected_value=current_status,
                    actual_value=action,
                    message=(
                        f"剧情线“{name}”已经{current_status}，"
                        f"本章却继续执行 {action}。"
                    ),
                    evidence=change.evidence,
                )
        source = _source(row, change.evidence)
        state["plot_threads"][key] = {
            "name": name,
            "identity_id": identity.get("id"),
            "thread_type": str(change.thread_type),
            "status": status_for_action[action],
            "last_action": action,
            "update": _clean(change.update),
            "promise": (
                _clean(change.promise)
                or _clean((current or {}).get("promise"))
            ),
            "target_payoff": (
                _clean(change.target_payoff)
                or _clean((current or {}).get("target_payoff"))
            ),
            "opened_source": (
                source
                if action == "opened"
                else (current or {}).get("opened_source") or source
            ),
            "source": source,
        }


def _apply_foreshadowing_changes(
    *,
    state: Dict[str, Any],
    delta: StoryDelta,
    row: Mapping[str, Any],
    project_id: str,
    branch_id: str,
    has_prior_chapter: bool,
    issues: Dict[str, Dict[str, Any]],
    identity_index: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    closed_statuses = {"payoff", "abandoned"}
    for change in delta.foreshadowing_changes:
        identity = _resolved(
            identity_index, "foreshadowing", change.hook_name
        )
        name = _clean(identity["canonical_text"])
        key = _existing_key(state["foreshadowing"], name)
        current = state["foreshadowing"].get(key)
        action = str(change.action)
        current_status = _clean((current or {}).get("status"))
        if not current and action != "setup" and has_prior_chapter:
            _add_issue(
                issues,
                row=row,
                project_id=project_id,
                branch_id=branch_id,
                issue_type="foreshadow_without_setup",
                severity="warning",
                entity_type="foreshadowing",
                entity_name=name,
                field_name="status",
                expected_value="此前已埋设的伏笔",
                actual_value=action,
                message=(
                    f"伏笔“{name}”在本章执行 {action}，"
                    "但此前正史中没有埋设记录。"
                ),
                evidence=change.evidence,
            )
        elif current:
            if action == "setup" and current_status not in closed_statuses:
                _add_issue(
                    issues,
                    row=row,
                    project_id=project_id,
                    branch_id=branch_id,
                    issue_type="foreshadow_duplicate_setup",
                    severity="warning",
                    entity_type="foreshadowing",
                    entity_name=name,
                    field_name="status",
                    expected_value=current_status,
                    actual_value=action,
                    message=f"伏笔“{name}”尚未回收，本章却再次埋设。",
                    evidence=change.evidence,
                )
            elif action == "setup" and current_status in closed_statuses:
                _add_issue(
                    issues,
                    row=row,
                    project_id=project_id,
                    branch_id=branch_id,
                    issue_type="foreshadow_reopened",
                    severity="warning",
                    entity_type="foreshadowing",
                    entity_name=name,
                    field_name="status",
                    expected_value=current_status,
                    actual_value=action,
                    message=(
                        f"伏笔“{name}”已经{current_status}，"
                        "本章重新埋设；请确认这是新的伏笔周期。"
                    ),
                    evidence=change.evidence,
                )
            elif current_status in closed_statuses and action != current_status:
                _add_issue(
                    issues,
                    row=row,
                    project_id=project_id,
                    branch_id=branch_id,
                    issue_type="foreshadow_after_closed",
                    severity="hard",
                    entity_type="foreshadowing",
                    entity_name=name,
                    field_name="status",
                    expected_value=current_status,
                    actual_value=action,
                    message=(
                        f"伏笔“{name}”已经{current_status}，"
                        f"本章却继续执行 {action}。"
                    ),
                    evidence=change.evidence,
                )
        source = _source(row, change.evidence)
        state["foreshadowing"][key] = {
            "name": name,
            "identity_id": identity.get("id"),
            "status": action,
            "last_action": action,
            "description": _clean(change.description),
            "intended_payoff": (
                _clean(change.intended_payoff)
                or _clean((current or {}).get("intended_payoff"))
            ),
            "setup_source": (
                source
                if action == "setup"
                else (current or {}).get("setup_source") or source
            ),
            "source": source,
        }


def _apply_events(
    *,
    state: Dict[str, Any],
    delta: StoryDelta,
    row: Mapping[str, Any],
    project_id: str,
    branch_id: str,
    issues: Dict[str, Dict[str, Any]],
    identity_index: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    for event in delta.events:
        identity = _resolved(
            identity_index,
            "event",
            event.event_key or event.summary,
        )
        event_key = _clean(identity["canonical_text"])
        current = state["events"].get(event_key)
        if current:
            same_summary = _same(current.get("summary"), event.summary)
            _add_issue(
                issues,
                row=row,
                project_id=project_id,
                branch_id=branch_id,
                issue_type="duplicate_event_identity",
                severity="warning" if same_summary else "hard",
                entity_type="event",
                entity_name=event_key,
                field_name="event_key",
                expected_value=_clean(current.get("summary")),
                actual_value=_clean(event.summary),
                message=(
                    f"事件键“{event_key}”已被此前正史使用；"
                    + (
                        "本章又把它登记为一次新事件。"
                        if same_summary
                        else "本章却用它表示不同事件，请拆分或修正归一规则。"
                    )
                ),
                evidence=event.evidence,
            )

        cause_keys: List[str] = []
        for raw_cause in event.cause_event_keys:
            cause_identity = _resolved(
                identity_index, "event", raw_cause
            )
            cause_key = _clean(cause_identity["canonical_text"])
            if not cause_key or cause_key in cause_keys:
                continue
            cause_keys.append(cause_key)
            if cause_key == event_key:
                _add_issue(
                    issues,
                    row=row,
                    project_id=project_id,
                    branch_id=branch_id,
                    issue_type="causal_self_reference",
                    severity="hard",
                    entity_type="event",
                    entity_name=event_key,
                    field_name="cause_event_keys",
                    expected_value="此前发生的另一事件",
                    actual_value=cause_key,
                    message=f"事件“{event_key}”不能把自身列为直接原因。",
                    evidence=event.evidence,
                )
                continue
            cause_event = state["events"].get(cause_key)
            if not cause_event:
                _add_issue(
                    issues,
                    row=row,
                    project_id=project_id,
                    branch_id=branch_id,
                    issue_type="causal_reference_missing",
                    severity="warning",
                    entity_type="event",
                    entity_name=event_key,
                    field_name="cause_event_keys",
                    expected_value="此前正史中已经发生的事件键",
                    actual_value=cause_key,
                    message=(
                        f"事件“{event_key}”引用“{cause_key}”作为直接原因，"
                        "但此前正史中没有该事件。"
                    ),
                    evidence=event.evidence,
                )
                continue
            edge_key = f"{cause_key}\u241f{event_key}"
            state["causal_edges"][edge_key] = {
                "cause": cause_key,
                "cause_identity_id": cause_event.get("identity_id"),
                "effect": event_key,
                "effect_identity_id": identity.get("id"),
                "source": _source(row, event.evidence),
            }

        participants = [
            _canonical(identity_index, "character", name)
            for name in event.participants
            if _clean(name)
        ]
        location = _canonical(identity_index, "location", event.location)
        state["events"][event_key] = {
            "identity_id": identity.get("id"),
            "event_key": event_key,
            "summary": _clean(event.summary),
            "participants": participants,
            "location": location,
            "story_time": _clean(event.story_time),
            "causes": [_clean(value) for value in event.causes if _clean(value)],
            "cause_event_keys": cause_keys,
            "effects": [
                _clean(value) for value in event.effects if _clean(value)
            ],
            "source": _source(row, event.evidence),
        }


def _apply_delta(
    *,
    state: Dict[str, Any],
    delta: StoryDelta,
    row: Mapping[str, Any],
    project_id: str,
    branch_id: str,
    has_prior_chapter: bool,
    issues: Dict[str, Dict[str, Any]],
    identity_index: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    start = copy.deepcopy(state)

    for change in delta.character_changes:
        identity = _resolved(
            identity_index, "character", change.character_name
        )
        name = _clean(identity["canonical_text"])
        aspect = _clean(change.aspect)
        current = _character_attribute(state, name, aspect)
        at_start = _character_attribute(start, name, aspect)
        declared_before = (
            _canonical(identity_index, "location", change.before)
            if aspect == "location" and change.before
            else change.before
        )
        _check_declared_before(
            issues,
            row=row,
            project_id=project_id,
            branch_id=branch_id,
            declared=declared_before,
            current=current,
            chapter_start=at_start,
            has_prior_chapter=has_prior_chapter,
            mismatch_type="state_before_mismatch",
            severity="hard" if aspect == "location" else "warning",
            entity_type="character",
            entity_name=name,
            field_name=aspect,
            evidence=change.evidence,
        )
        character = state["characters"].setdefault(
            name,
            {
                "identity_id": identity.get("id"),
                "attributes": {},
                "location": None,
            },
        )
        character["identity_id"] = identity.get("id")
        after_value = (
            _canonical(identity_index, "location", change.after)
            if aspect == "location"
            else _clean(change.after)
        )
        entry = {
            "value": after_value,
            "source": _source(row, change.evidence),
        }
        character["attributes"][aspect] = entry
        if aspect == "location":
            character["location"] = entry
            state["locations"][name] = entry

    for change in delta.relationship_changes:
        character_a = _canonical(
            identity_index, "character", change.character_a
        )
        character_b = _canonical(
            identity_index, "character", change.character_b
        )
        key = _relationship_key(character_a, character_b)
        current = state["relationships"].get(key)
        at_start = start["relationships"].get(key)
        _check_declared_before(
            issues,
            row=row,
            project_id=project_id,
            branch_id=branch_id,
            declared=change.before,
            current=current,
            chapter_start=at_start,
            has_prior_chapter=has_prior_chapter,
            mismatch_type="relationship_before_mismatch",
            severity="hard",
            entity_type="relationship",
            entity_name=f"{character_a} ↔ {character_b}",
            field_name="relationship",
            evidence=change.evidence,
        )
        state["relationships"][key] = {
            "characters": [character_a, character_b],
            "value": _clean(change.after),
            "source": _source(row, change.evidence),
        }

    for change in delta.location_changes:
        subject_identity = _resolved(
            identity_index, "character", change.subject_name
        )
        name = _clean(subject_identity["canonical_text"])
        current = state["locations"].get(name)
        at_start = start["locations"].get(name)
        _check_declared_before(
            issues,
            row=row,
            project_id=project_id,
            branch_id=branch_id,
            declared=(
                _canonical(
                    identity_index, "location", change.from_location
                )
                if change.from_location
                else change.from_location
            ),
            current=current,
            chapter_start=at_start,
            has_prior_chapter=has_prior_chapter,
            mismatch_type="location_before_mismatch",
            severity="hard",
            entity_type="location",
            entity_name=name,
            field_name="location",
            evidence=change.evidence,
        )
        entry = {
            "value": _canonical(
                identity_index, "location", change.to_location
            ),
            "source": _source(row, change.evidence),
        }
        state["locations"][name] = entry
        character = state["characters"].setdefault(
            name,
            {
                "identity_id": subject_identity.get("id"),
                "attributes": {},
                "location": None,
            },
        )
        character["identity_id"] = subject_identity.get("id")
        character["location"] = entry
        character["attributes"]["location"] = entry

    for change in delta.item_changes:
        item_identity = _resolved(
            identity_index, "item", change.item_name
        )
        name = _clean(item_identity["canonical_text"])
        current = state["items"].get(name) or {}
        at_start = start["items"].get(name) or {}
        declared_holder = (
            _canonical(
                identity_index, "character", change.from_holder
            )
            if change.from_holder
            else ""
        )
        if declared_holder:
            _check_declared_before(
                issues,
                row=row,
                project_id=project_id,
                branch_id=branch_id,
                declared=declared_holder,
                current=current.get("holder"),
                chapter_start=at_start.get("holder"),
                has_prior_chapter=has_prior_chapter,
                mismatch_type="item_holder_mismatch",
                severity="hard",
                entity_type="item",
                entity_name=name,
                field_name="holder",
                evidence=change.evidence,
            )
        if (
            current.get("status") == "destroyed"
            and change.action not in {"created", "changed"}
        ):
            _add_issue(
                issues,
                row=row,
                project_id=project_id,
                branch_id=branch_id,
                issue_type="item_after_destroyed",
                severity="hard",
                entity_type="item",
                entity_name=name,
                field_name="status",
                expected_value="已毁坏",
                actual_value=str(change.action),
                message=f"{name} 已在此前正史中毁坏，本章却再次被使用或转移。",
                evidence=change.evidence,
            )
        holder: Optional[str] = _value(current.get("holder"))
        status = _clean(current.get("status")) or "active"
        if change.action in {"created", "acquired", "transferred"}:
            holder = (
                _canonical(
                    identity_index, "character", change.to_holder
                )
                if change.to_holder
                else holder
            )
            status = "active"
        elif change.action == "lost":
            holder = None
            status = "lost"
        elif change.action == "destroyed":
            holder = None
            status = "destroyed"
        elif change.to_holder:
            holder = _canonical(
                identity_index, "character", change.to_holder
            )
        state["items"][name] = {
            "identity_id": item_identity.get("id"),
            "holder": (
                {
                    "value": holder,
                    "source": _source(row, change.evidence),
                }
                if holder
                else None
            ),
            "state": _clean(change.state),
            "status": status,
            "last_action": str(change.action),
            "source": _source(row, change.evidence),
        }

    _apply_knowledge_changes(
        state=state,
        delta=delta,
        row=row,
        project_id=project_id,
        branch_id=branch_id,
        has_prior_chapter=has_prior_chapter,
        issues=issues,
        identity_index=identity_index,
    )
    _apply_plot_thread_changes(
        state=state,
        delta=delta,
        row=row,
        project_id=project_id,
        branch_id=branch_id,
        has_prior_chapter=has_prior_chapter,
        issues=issues,
        identity_index=identity_index,
    )
    _apply_foreshadowing_changes(
        state=state,
        delta=delta,
        row=row,
        project_id=project_id,
        branch_id=branch_id,
        has_prior_chapter=has_prior_chapter,
        issues=issues,
        identity_index=identity_index,
    )
    _apply_events(
        state=state,
        delta=delta,
        row=row,
        project_id=project_id,
        branch_id=branch_id,
        issues=issues,
        identity_index=identity_index,
    )

    if delta.time_advance:
        advance = delta.time_advance
        current_time = state.get("story_time")
        start_time = start.get("story_time")
        _check_declared_before(
            issues,
            row=row,
            project_id=project_id,
            branch_id=branch_id,
            declared=advance.from_time,
            current=current_time,
            chapter_start=start_time,
            has_prior_chapter=has_prior_chapter,
            mismatch_type="story_time_mismatch",
            severity="warning",
            entity_type="timeline",
            entity_name="故事时间",
            field_name="story_time",
            evidence=advance.elapsed,
        )
        if _clean(advance.to_time):
            state["story_time"] = {
                "value": _clean(advance.to_time),
                "elapsed": _clean(advance.elapsed),
                "source": _source(row, advance.elapsed),
            }
    elif delta.events:
        event_time = next(
            (
                _clean(event.story_time)
                for event in reversed(delta.events)
                if _clean(event.story_time)
            ),
            "",
        )
        if event_time:
            state["story_time"] = {
                "value": event_time,
                "elapsed": "",
                "source": _source(row, delta.events[-1].evidence),
            }

    state["last_chapter"] = {
        "id": str(row["chapter_id"]),
        "position": int(row["chapter_position"]),
        "title": str(row["chapter_title"]),
        "version_id": str(row["version_id"]),
        "delta_id": str(row["id"]),
    }


def replay_canonical_state(
    connection: sqlite3.Connection,
    *,
    project_id: str,
    branch_id: str,
    trigger_type: str,
    trigger_chapter_id: Optional[str],
    created_at: str,
) -> Dict[str, Any]:
    """Replay confirmed deltas in chapter order inside the caller's transaction."""

    run_id = uuid.uuid4().hex
    connection.execute(
        """
        INSERT INTO continuity_replay_runs(
            id, project_id, branch_id, trigger_type, trigger_chapter_id,
            status, replayed_chapter_count, issue_count, final_state_json,
            created_at
        ) VALUES (?, ?, ?, ?, ?, 'running', 0, 0, '{}', ?)
        """,
        (
            run_id,
            project_id,
            branch_id,
            trigger_type,
            trigger_chapter_id,
            created_at,
        ),
    )
    rows = connection.execute(
        """
        SELECT d.id, d.project_id, d.chapter_id, d.version_id, d.branch_id,
               d.payload_json, ch.position AS chapter_position,
               ch.title AS chapter_title
        FROM story_deltas d
        JOIN novel_chapters ch ON ch.id=d.chapter_id
        WHERE d.project_id=? AND d.branch_id=? AND d.status='projected'
          AND ch.canonical_version_id=d.version_id
        ORDER BY ch.position, d.reviewed_at, d.created_at, d.id
        """,
        (project_id, branch_id),
    ).fetchall()
    identity_index = load_identity_index(
        connection, project_id=project_id
    )

    connection.execute(
        "DELETE FROM story_state_snapshots WHERE project_id=? AND branch_id=?",
        (project_id, branch_id),
    )
    connection.execute(
        """
        UPDATE continuity_issues
        SET active=0, resolved_at=?
        WHERE project_id=? AND branch_id=? AND active=1
        """,
        (created_at, project_id, branch_id),
    )

    state = _empty_state()
    issues: Dict[str, Dict[str, Any]] = {}
    snapshots: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        row_data = dict(row)
        delta = StoryDelta.model_validate_json(str(row["payload_json"]))
        before_count = len(issues)
        _apply_delta(
            state=state,
            delta=delta,
            row=row_data,
            project_id=project_id,
            branch_id=branch_id,
            has_prior_chapter=(
                index > 0 or int(row["chapter_position"]) > 1
            ),
            issues=issues,
            identity_index=identity_index,
        )
        snapshots.append(
            {
                "id": uuid.uuid4().hex,
                "project_id": project_id,
                "branch_id": branch_id,
                "chapter_id": str(row["chapter_id"]),
                "version_id": str(row["version_id"]),
                "delta_id": str(row["id"]),
                "chapter_position": int(row["chapter_position"]),
                "state_json": _json(state),
                "issue_count": len(issues) - before_count,
            }
        )

    for snapshot in snapshots:
        connection.execute(
            """
            INSERT INTO story_state_snapshots(
                id, project_id, branch_id, chapter_id, version_id, delta_id,
                chapter_position, state_json, issue_count, replay_run_id,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot["id"],
                snapshot["project_id"],
                snapshot["branch_id"],
                snapshot["chapter_id"],
                snapshot["version_id"],
                snapshot["delta_id"],
                snapshot["chapter_position"],
                snapshot["state_json"],
                snapshot["issue_count"],
                run_id,
                created_at,
            ),
        )

    for issue in issues.values():
        connection.execute(
            """
            INSERT INTO continuity_issues(
                id, fingerprint, project_id, branch_id, chapter_id,
                version_id, delta_id, chapter_position, issue_type, severity,
                entity_type, entity_name, field_name, expected_value,
                actual_value, message, evidence, status, author_note, active,
                first_seen_at, last_seen_at, resolved_at, acknowledged_at,
                replay_run_id
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                'open', '', 1, ?, ?, NULL, NULL, ?
            )
            ON CONFLICT(fingerprint) DO UPDATE SET
                chapter_id=excluded.chapter_id,
                version_id=excluded.version_id,
                delta_id=excluded.delta_id,
                chapter_position=excluded.chapter_position,
                severity=excluded.severity,
                expected_value=excluded.expected_value,
                actual_value=excluded.actual_value,
                message=excluded.message,
                evidence=excluded.evidence,
                active=1,
                last_seen_at=excluded.last_seen_at,
                resolved_at=NULL,
                replay_run_id=excluded.replay_run_id
            """,
            (
                issue["id"],
                issue["fingerprint"],
                issue["project_id"],
                issue["branch_id"],
                issue["chapter_id"],
                issue["version_id"],
                issue["delta_id"],
                issue["chapter_position"],
                issue["issue_type"],
                issue["severity"],
                issue["entity_type"],
                issue["entity_name"],
                issue["field_name"],
                issue["expected_value"],
                issue["actual_value"],
                issue["message"],
                issue["evidence"],
                created_at,
                created_at,
                run_id,
            ),
        )

    if issues:
        earliest_issue = min(
            int(issue["chapter_position"]) for issue in issues.values()
        )
        connection.execute(
            """
            UPDATE novel_chapters
            SET needs_recheck=1, updated_at=?
            WHERE project_id=? AND position>=?
              AND canonical_version_id IS NOT NULL
            """,
            (created_at, project_id, earliest_issue),
        )

    connection.execute(
        """
        UPDATE continuity_replay_runs
        SET status='completed', replayed_chapter_count=?, issue_count=?,
            final_state_json=?, finished_at=?
        WHERE id=?
        """,
        (len(rows), len(issues), _json(state), created_at, run_id),
    )
    return {
        "run_id": run_id,
        "replayed_chapter_count": len(rows),
        "issue_count": len(issues),
        "state": state,
    }


def _source_position(value: Any) -> int:
    if isinstance(value, Mapping):
        source = value.get("source") or {}
        if isinstance(source, Mapping):
            return int(source.get("chapter_position") or 0)
    return 0


def _ranked_subset(
    values: Mapping[str, Any],
    *,
    needles: Sequence[str],
    limit: int,
    active_statuses: Sequence[str] = (),
) -> Dict[str, Any]:
    ranked = []
    for key, value in values.items():
        haystack = f"{key} {_json(value)}".casefold()
        match_score = sum(1 for needle in needles if needle in haystack)
        status_rank = (
            0
            if not active_statuses
            or _clean(
                value.get("status") if isinstance(value, Mapping) else ""
            )
            in active_statuses
            else 1
        )
        ranked.append(
            (
                -match_score,
                status_rank,
                -_source_position(value),
                str(key).casefold(),
                str(key),
                value,
            )
        )
    ranked.sort()
    return {
        key: value for _, _, _, _, key, value in ranked[:limit]
    }


def _focused_knowledge(
    values: Mapping[str, Any],
    *,
    needles: Sequence[str],
    limit: int,
) -> Dict[str, Any]:
    ranked = []
    for character_name, facts in values.items():
        if not isinstance(facts, Mapping):
            continue
        for fact_key, entry in facts.items():
            haystack = (
                f"{character_name} {fact_key} {_json(entry)}".casefold()
            )
            match_score = sum(
                1 for needle in needles if needle in haystack
            )
            ranked.append(
                (
                    -match_score,
                    -_source_position(entry),
                    str(character_name).casefold(),
                    str(fact_key).casefold(),
                    str(character_name),
                    str(fact_key),
                    entry,
                )
            )
    ranked.sort()
    focused: Dict[str, Any] = {}
    for _, _, _, _, character_name, fact_key, entry in ranked[:limit]:
        focused.setdefault(character_name, {})[fact_key] = entry
    return focused


def _pop_last_focused_entry(
    focused: Dict[str, Any], category: str
) -> bool:
    values = focused[category]
    if not values:
        return False
    if category != "knowledge":
        values.pop(next(reversed(values)))
        return True
    character_name = next(reversed(values))
    facts = values[character_name]
    if facts:
        facts.pop(next(reversed(facts)))
    if not facts:
        values.pop(character_name)
    return True


def focus_state_for_context(
    state: Mapping[str, Any],
    *,
    query_concepts: Iterable[str],
    max_chars: int = 4_500,
) -> Dict[str, Any]:
    needles = [
        _clean(value).casefold()
        for value in query_concepts
        if len(_clean(value)) >= 2
    ][:24]
    focused = {
        "schema_version": int(
            state.get("schema_version") or STATE_SCHEMA_VERSION
        ),
        "story_time": state.get("story_time"),
        "last_chapter": state.get("last_chapter"),
        "characters": _ranked_subset(
            state.get("characters") or {}, needles=needles, limit=12
        ),
        "relationships": _ranked_subset(
            state.get("relationships") or {}, needles=needles, limit=10
        ),
        "locations": _ranked_subset(
            state.get("locations") or {}, needles=needles, limit=12
        ),
        "items": _ranked_subset(
            state.get("items") or {}, needles=needles, limit=12
        ),
        "knowledge": _focused_knowledge(
            state.get("knowledge") or {}, needles=needles, limit=24
        ),
        "plot_threads": _ranked_subset(
            state.get("plot_threads") or {},
            needles=needles,
            limit=12,
            active_statuses=("open", "active", "paused"),
        ),
        "foreshadowing": _ranked_subset(
            state.get("foreshadowing") or {},
            needles=needles,
            limit=12,
            active_statuses=("setup", "advanced"),
        ),
        "events": _ranked_subset(
            state.get("events") or {},
            needles=needles,
            limit=18,
        ),
        "causal_edges": _ranked_subset(
            state.get("causal_edges") or {},
            needles=needles,
            limit=24,
        ),
        "truncated": False,
    }
    categories = (
        "items",
        "locations",
        "relationships",
        "characters",
        "knowledge",
        "events",
        "causal_edges",
        "foreshadowing",
        "plot_threads",
    )
    while len(_json(focused)) > max_chars:
        removed = False
        for category in categories:
            if _pop_last_focused_entry(focused, category):
                focused["truncated"] = True
                removed = True
                break
        if not removed:
            break
    return focused


def get_continuity_context(
    connection: sqlite3.Connection,
    *,
    project_id: str,
    branch_id: str,
    before_chapter_position: int,
    query_concepts: Iterable[str],
) -> Dict[str, Any]:
    snapshot = connection.execute(
        """
        SELECT s.*, ch.title AS chapter_title
        FROM story_state_snapshots s
        JOIN novel_chapters ch ON ch.id=s.chapter_id
        WHERE s.project_id=? AND s.branch_id=?
          AND s.chapter_position<?
        ORDER BY s.chapter_position DESC LIMIT 1
        """,
        (project_id, branch_id, before_chapter_position),
    ).fetchone()
    issues = connection.execute(
        """
        SELECT i.id, i.fingerprint, i.chapter_id, i.chapter_position,
               ch.title AS chapter_title, i.issue_type, i.severity,
               i.entity_type, i.entity_name, i.field_name,
               i.expected_value, i.actual_value, i.message, i.evidence,
               i.status, i.author_note
        FROM continuity_issues i
        JOIN novel_chapters ch ON ch.id=i.chapter_id
        WHERE i.project_id=? AND i.branch_id=? AND i.active=1
          AND i.chapter_position<?
        ORDER BY CASE i.severity WHEN 'hard' THEN 0 ELSE 1 END,
                 i.chapter_position DESC, i.first_seen_at
        LIMIT 8
        """,
        (project_id, branch_id, before_chapter_position),
    ).fetchall()
    run = connection.execute(
        """
        SELECT id, trigger_type, replayed_chapter_count, issue_count,
               created_at, finished_at
        FROM continuity_replay_runs
        WHERE project_id=? AND branch_id=? AND status='completed'
        ORDER BY created_at DESC, rowid DESC LIMIT 1
        """,
        (project_id, branch_id),
    ).fetchone()
    full_state = (
        _load_json(snapshot["state_json"], _empty_state())
        if snapshot
        else _empty_state()
    )
    return {
        "current_state": focus_state_for_context(
            full_state, query_concepts=query_concepts
        ),
        "continuity_issues": [dict(row) for row in issues],
        "continuity_replay": dict(run) if run else None,
    }


class ContinuityService:
    def __init__(self, database: Any):
        self.database = database

    def get_dashboard(
        self, *, user_id: int, project_id: str
    ) -> Optional[Dict[str, Any]]:
        with self.database.connection() as connection:
            project = connection.execute(
                """
                SELECT id, title, canonical_branch_id
                FROM novel_projects
                WHERE id=? AND user_id=?
                """,
                (project_id, user_id),
            ).fetchone()
            if not project:
                return None
            branch_id = str(project["canonical_branch_id"])
            run = connection.execute(
                """
                SELECT * FROM continuity_replay_runs
                WHERE project_id=? AND branch_id=? AND status='completed'
                ORDER BY created_at DESC, rowid DESC LIMIT 1
                """,
                (project_id, branch_id),
            ).fetchone()
            snapshot = connection.execute(
                """
                SELECT s.*, ch.title AS chapter_title
                FROM story_state_snapshots s
                JOIN novel_chapters ch ON ch.id=s.chapter_id
                WHERE s.project_id=? AND s.branch_id=?
                ORDER BY s.chapter_position DESC LIMIT 1
                """,
                (project_id, branch_id),
            ).fetchone()
            issues = connection.execute(
                """
                SELECT i.*, ch.title AS chapter_title
                FROM continuity_issues i
                JOIN novel_chapters ch ON ch.id=i.chapter_id
                WHERE i.project_id=? AND i.branch_id=? AND i.active=1
                ORDER BY CASE i.severity WHEN 'hard' THEN 0 ELSE 1 END,
                         i.chapter_position DESC, i.first_seen_at
                """,
                (project_id, branch_id),
            ).fetchall()
            snapshots = connection.execute(
                """
                SELECT s.id, s.chapter_id, s.chapter_position, s.issue_count,
                       s.created_at, ch.title AS chapter_title
                FROM story_state_snapshots s
                JOIN novel_chapters ch ON ch.id=s.chapter_id
                WHERE s.project_id=? AND s.branch_id=?
                ORDER BY s.chapter_position DESC LIMIT 100
                """,
                (project_id, branch_id),
            ).fetchall()
            identities = list_identity_context(
                connection, project_id=project_id, limit=500
            )
        issue_items = [dict(row) for row in issues]
        state = (
            _load_json(snapshot["state_json"], _empty_state())
            if snapshot
            else _load_json(run["final_state_json"], _empty_state())
            if run
            else _empty_state()
        )
        knowledge_fact_count = sum(
            len(facts)
            for facts in (state.get("knowledge") or {}).values()
            if isinstance(facts, Mapping)
        )
        open_plot_threads = sum(
            1
            for item in (state.get("plot_threads") or {}).values()
            if _clean(item.get("status")) not in {"resolved", "abandoned"}
        )
        open_foreshadowing = sum(
            1
            for item in (state.get("foreshadowing") or {}).values()
            if _clean(item.get("status")) not in {"payoff", "abandoned"}
        )
        identity_rules = []
        for identity in identities:
            aliases = [
                alias
                for alias in identity["aliases"]
                if normalize_identity_text(alias)
                != normalize_identity_text(identity["canonical_text"])
            ]
            if identity["source"] != "author" and not aliases:
                continue
            identity_rules.append({**identity, "aliases": aliases})
        return {
            "project": dict(project),
            "latest_run": dict(run) if run else None,
            "latest_snapshot": dict(snapshot) if snapshot else None,
            "state": state,
            "issues": issue_items,
            "snapshots": [dict(row) for row in snapshots],
            "identity_rules": identity_rules,
            "counts": {
                "active": len(issue_items),
                "hard": sum(
                    1 for item in issue_items if item["severity"] == "hard"
                ),
                "warning": sum(
                    1
                    for item in issue_items
                    if item["severity"] == "warning"
                ),
                "acknowledged": sum(
                    1
                    for item in issue_items
                    if item["status"] == "acknowledged"
                ),
                "knowledge_facts": knowledge_fact_count,
                "open_plot_threads": open_plot_threads,
                "open_foreshadowing": open_foreshadowing,
                "events": len(state.get("events") or {}),
                "causal_edges": len(state.get("causal_edges") or {}),
                "identity_rules": len(identity_rules),
            },
        }

    def set_issue_status(
        self,
        *,
        user_id: int,
        project_id: str,
        issue_id: str,
        action: str,
        author_note: str,
        updated_at: str,
    ) -> bool:
        if action not in {"acknowledge", "reopen"}:
            raise ValueError("不支持的连续性问题操作")
        note = _clean(author_note)
        if action == "acknowledge" and len(note) < 4:
            raise ValueError("请用至少 4 个字符说明作者判断")
        with self.database.connection() as connection:
            if action == "acknowledge":
                cursor = connection.execute(
                    """
                    UPDATE continuity_issues
                    SET status='acknowledged', author_note=?,
                        acknowledged_at=?
                    WHERE id=? AND project_id=? AND active=1
                      AND EXISTS(
                          SELECT 1 FROM novel_projects p
                          WHERE p.id=continuity_issues.project_id
                            AND p.user_id=?
                      )
                    """,
                    (note[:2000], updated_at, issue_id, project_id, user_id),
                )
            else:
                cursor = connection.execute(
                    """
                    UPDATE continuity_issues
                    SET status='open', acknowledged_at=NULL
                    WHERE id=? AND project_id=? AND active=1
                      AND EXISTS(
                          SELECT 1 FROM novel_projects p
                          WHERE p.id=continuity_issues.project_id
                            AND p.user_id=?
                      )
                    """,
                    (issue_id, project_id, user_id),
                )
            connection.commit()
        return cursor.rowcount == 1
