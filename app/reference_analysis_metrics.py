"""Deterministic prose measurements used by reference analysis."""

from __future__ import annotations

import math
import re
import statistics
from typing import Any


def deterministic_structure(text: str) -> dict[str, Any]:
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
    sentences = [
        item.strip()
        for item in re.findall(
            r"[^。！？!?]+[。！？!?]+[”’\"']?|[^。！？!?]+$",
            text,
        )
        if item.strip()
    ]
    sentence_lengths = [len(item) for item in sentences]
    ordered_lengths = sorted(sentence_lengths)
    paragraph_lengths = [len(item) for item in paragraphs]
    dialogue_chars = sum(
        len(match.group(1))
        for match in re.finditer(r"[“\"]([^”\"]+)[”\"]", text)
    )
    total = len(text)
    return {
        "char_count": total,
        "paragraph_count": len(paragraphs),
        "sentence_count": len(sentences),
        "sentence_char_count": sum(sentence_lengths),
        "sentence_length_sum_squares": sum(
            length * length for length in sentence_lengths
        ),
        "average_sentence_chars": (
            round(statistics.fmean(sentence_lengths), 1)
            if sentence_lengths
            else 0.0
        ),
        "median_sentence_chars": (
            round(float(statistics.median(sentence_lengths)), 1)
            if sentence_lengths
            else 0.0
        ),
        "sentence_length_stddev": (
            round(statistics.pstdev(sentence_lengths), 1)
            if len(sentence_lengths) > 1
            else 0.0
        ),
        "p90_sentence_chars": (
            ordered_lengths[max(0, math.ceil(len(ordered_lengths) * 0.9) - 1)]
            if ordered_lengths
            else 0
        ),
        "paragraph_char_count": sum(paragraph_lengths),
        "average_paragraph_chars": (
            round(statistics.fmean(paragraph_lengths), 1)
            if paragraph_lengths
            else 0.0
        ),
        "dialogue_char_count": dialogue_chars,
        "dialogue_ratio": round(dialogue_chars / max(1, total), 4),
        "opening_excerpt": text[:160].strip(),
        "closing_excerpt": text[-160:].strip(),
    }
