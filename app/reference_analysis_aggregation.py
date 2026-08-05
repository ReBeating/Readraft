"""Whole-book aggregation for validated reference-analysis layers."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping

from .reference_analysis_schema import ANALYSIS_SCHEMA_VERSION


def _normalise_style_value(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _most_common_text(values: Iterable[str]) -> str:
    counter = Counter(item.strip() for item in values if item.strip())
    return counter.most_common(1)[0][0] if counter else ""


def _aggregate_style_profile(
    chapter_layers: list[tuple[int, Mapping[str, Any]]],
) -> dict[str, Any]:
    structures: list[Mapping[str, Any]] = []
    signals: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for position, layers in chapter_layers:
        structure = layers.get("structure")
        if isinstance(structure, Mapping):
            structures.append(structure)
        style = layers.get("style")
        if not isinstance(style, Mapping):
            continue
        for raw in style.get("observations") or []:
            if not isinstance(raw, Mapping):
                continue
            axis = str(raw.get("axis") or "").strip()
            value = str(raw.get("value") or "").strip()
            key = _normalise_style_value(value)
            if not axis or not key:
                continue
            bucket = signals[axis].setdefault(
                key,
                {
                    "value": value,
                    "chapters": [],
                    "analyses": [],
                    "execution_rules": [],
                    "originality_boundaries": [],
                },
            )
            bucket["chapters"].append(position)
            bucket["analyses"].append(str(raw.get("analysis") or ""))
            bucket["execution_rules"].append(
                str(raw.get("execution_rule") or "")
            )
            bucket["originality_boundaries"].append(
                str(raw.get("originality_boundary") or "")
            )

    analyzed_chapters = len(chapter_layers)
    traits: list[dict[str, Any]] = []
    for axis, variants_by_key in signals.items():
        variants = sorted(
            variants_by_key.values(),
            key=lambda item: (-len(item["chapters"]), item["chapters"][0]),
        )
        dominant = variants[0]
        count = len(dominant["chapters"])
        traits.append(
            {
                "axis": axis,
                "value": dominant["value"],
                "chapter_count": count,
                "coverage": round(count / max(1, analyzed_chapters), 4),
                "chapters": dominant["chapters"],
                "analysis": _most_common_text(dominant["analyses"]),
                "execution_rule": _most_common_text(
                    dominant["execution_rules"]
                ),
                "originality_boundary": _most_common_text(
                    dominant["originality_boundaries"]
                ),
                "variants": [
                    {
                        "value": item["value"],
                        "chapter_count": len(item["chapters"]),
                        "chapters": item["chapters"],
                    }
                    for item in variants
                ],
            }
        )
    traits.sort(key=lambda item: item["axis"])

    char_count = sum(int(item.get("char_count") or 0) for item in structures)
    paragraph_count = sum(
        int(item.get("paragraph_count") or 0) for item in structures
    )
    sentence_count = sum(
        int(item.get("sentence_count") or 0) for item in structures
    )
    sentence_chars = sum(
        int(
            item.get("sentence_char_count")
            or round(
                float(item.get("average_sentence_chars") or 0)
                * int(item.get("sentence_count") or 0)
            )
        )
        for item in structures
    )
    sentence_squares = sum(
        int(
            item.get("sentence_length_sum_squares")
            or round(
                (
                    float(item.get("sentence_length_stddev") or 0) ** 2
                    + float(item.get("average_sentence_chars") or 0) ** 2
                )
                * int(item.get("sentence_count") or 0)
            )
        )
        for item in structures
    )
    dialogue_chars = sum(
        int(
            item.get("dialogue_char_count")
            or round(
                float(item.get("dialogue_ratio") or 0)
                * int(item.get("char_count") or 0)
            )
        )
        for item in structures
    )
    paragraph_chars = sum(
        int(
            item.get("paragraph_char_count")
            or round(
                float(item.get("average_paragraph_chars") or 0)
                * int(item.get("paragraph_count") or 0)
            )
            or int(item.get("char_count") or 0)
        )
        for item in structures
    )
    average_sentence = sentence_chars / max(1, sentence_count)
    sentence_variance = max(
        0.0,
        sentence_squares / max(1, sentence_count) - average_sentence**2,
    )
    return {
        "analyzed_chapters": analyzed_chapters,
        "scope_note": (
            "这是从参考文本证据抽象出的写作信号；可迁移规则不包含原作措辞、"
            "专有名词、具体物件或具体情节。"
        ),
        "quantitative": {
            "char_count": char_count,
            "paragraph_count": paragraph_count,
            "sentence_count": sentence_count,
            "average_sentence_chars": round(average_sentence, 1),
            "sentence_length_stddev": round(math.sqrt(sentence_variance), 1),
            "average_paragraph_chars": round(
                paragraph_chars / max(1, paragraph_count), 1
            ),
            "dialogue_ratio": round(dialogue_chars / max(1, char_count), 4),
        },
        "traits": traits,
    }


def aggregate_document(
    chapter_results: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate completed chapters without inventing cross-chapter facts."""

    character_chapters: dict[str, list[int]] = defaultdict(list)
    character_roles: dict[str, Counter[str]] = defaultdict(Counter)
    timeline: list[dict[str, Any]] = []
    threads: dict[str, dict[str, Any]] = {}
    pacing_labels: Counter[str] = Counter()
    chapter_layers: list[tuple[int, Mapping[str, Any]]] = []
    for fallback_position, result in enumerate(chapter_results, start=1):
        layers = result.get("layers") if isinstance(result, Mapping) else None
        if not isinstance(layers, Mapping):
            continue
        position = int(result.get("chapter_position") or fallback_position)
        facts = layers.get("facts")
        narrative = layers.get("narrative")
        if not isinstance(facts, Mapping) or not isinstance(narrative, Mapping):
            continue
        chapter_layers.append((position, layers))
        for character in facts.get("characters") or []:
            name = str(character.get("name") or "").strip()
            if not name:
                continue
            character_chapters[name].append(position)
            character_roles[name][str(character.get("role") or "")] += 1
        for event in facts.get("events") or []:
            timeline.append(
                {
                    "chapter": position,
                    "event": str(event.get("event") or ""),
                    "impact": str(event.get("impact") or ""),
                }
            )
        for item in facts.get("foreshadowing") or []:
            key = str(item.get("thread_key") or item.get("clue") or "").strip()
            if not key:
                continue
            thread = threads.setdefault(
                key,
                {"key": key, "setups": [], "payoffs": [], "status": "open"},
            )
            target = "payoffs" if item.get("type") == "payoff" else "setups"
            thread[target].append(
                {"chapter": position, "clue": str(item.get("clue") or "")}
            )
            if thread["payoffs"]:
                thread["status"] = "paid_off"
        for item in narrative.get("pacing") or []:
            pacing_labels[str(item.get("label") or "未标注")] += 1
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "completed_chapters": len(chapter_layers),
        "characters": [
            {
                "name": name,
                "chapters": positions,
                "primary_role": roles.most_common(1)[0][0] if roles else "",
            }
            for name, positions in sorted(
                character_chapters.items(), key=lambda item: item[1][0]
            )
            for roles in [character_roles[name]]
        ],
        "event_timeline": timeline,
        "foreshadow_threads": list(threads.values()),
        "pacing_distribution": dict(pacing_labels),
        "style_profile": _aggregate_style_profile(chapter_layers),
    }
