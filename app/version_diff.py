from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any


def _effective_length(value: str) -> int:
    return sum(1 for character in value if not character.isspace())


def _append_segment(
    segments: list[dict[str, str]], kind: str, text: str
) -> None:
    if not text:
        return
    if segments and segments[-1]["kind"] == kind:
        segments[-1]["text"] += text
        return
    segments.append({"kind": kind, "text": text})


def _append_character_diff(
    before_segments: list[dict[str, str]],
    after_segments: list[dict[str, str]],
    original: str,
    replacement: str,
) -> tuple[int, int, int]:
    added = 0
    removed = 0
    unchanged = 0
    matcher = SequenceMatcher(None, original, replacement, autojunk=False)
    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        old_text = original[old_start:old_end]
        new_text = replacement[new_start:new_end]
        if tag == "equal":
            _append_segment(before_segments, "same", old_text)
            _append_segment(after_segments, "same", new_text)
            unchanged += _effective_length(old_text)
        elif tag == "delete":
            _append_segment(before_segments, "removed", old_text)
            removed += _effective_length(old_text)
        elif tag == "insert":
            _append_segment(after_segments, "added", new_text)
            added += _effective_length(new_text)
        else:
            _append_segment(before_segments, "removed", old_text)
            _append_segment(after_segments, "added", new_text)
            removed += _effective_length(old_text)
            added += _effective_length(new_text)
    return added, removed, unchanged


def build_version_diff(
    original: str,
    replacement: str,
    *,
    detailed_block_limit: int = 12_000,
) -> dict[str, Any]:
    """Build a bounded, display-safe whole-chapter comparison.

    Chapters are aligned by lines first so a large rewrite cannot make a
    character-level SequenceMatcher consume excessive CPU. Smaller replacement
    blocks still receive character-level highlighting.
    """

    original = original.replace("\r\n", "\n").replace("\r", "\n")
    replacement = replacement.replace("\r\n", "\n").replace("\r", "\n")
    original_lines = original.splitlines(keepends=True)
    replacement_lines = replacement.splitlines(keepends=True)
    before_segments: list[dict[str, str]] = []
    after_segments: list[dict[str, str]] = []
    added = 0
    removed = 0
    unchanged = 0
    changed_blocks = 0

    matcher = SequenceMatcher(
        None, original_lines, replacement_lines, autojunk=False
    )
    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        old_text = "".join(original_lines[old_start:old_end])
        new_text = "".join(replacement_lines[new_start:new_end])
        if tag == "equal":
            _append_segment(before_segments, "same", old_text)
            _append_segment(after_segments, "same", new_text)
            unchanged += _effective_length(old_text)
            continue

        changed_blocks += 1
        if tag == "delete":
            _append_segment(before_segments, "removed", old_text)
            removed += _effective_length(old_text)
        elif tag == "insert":
            _append_segment(after_segments, "added", new_text)
            added += _effective_length(new_text)
        elif len(old_text) + len(new_text) <= detailed_block_limit:
            block_added, block_removed, block_unchanged = (
                _append_character_diff(
                    before_segments,
                    after_segments,
                    old_text,
                    new_text,
                )
            )
            added += block_added
            removed += block_removed
            unchanged += block_unchanged
        else:
            _append_segment(before_segments, "removed", old_text)
            _append_segment(after_segments, "added", new_text)
            removed += _effective_length(old_text)
            added += _effective_length(new_text)

    original_effective = _effective_length(original)
    replacement_effective = _effective_length(replacement)
    return {
        "before": before_segments,
        "after": after_segments,
        "added_chars": added,
        "removed_chars": removed,
        "unchanged_chars": unchanged,
        "changed_blocks": changed_blocks,
        "original_chars": original_effective,
        "replacement_chars": replacement_effective,
        "delta_chars": replacement_effective - original_effective,
        "identical": original == replacement,
    }
