from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Iterable, List, Sequence, Tuple


NUMBER_CHARS = "0-9零〇一二三四五六七八九十百千万两壹贰叁肆伍陆柒捌玖拾佰仟"
NUMBER = rf"[{NUMBER_CHARS}]+"
TITLE_SUFFIX = r"(?:(?:\s+|[：:、.·—\-]\s*)\S.{0,40})?"

NUMBERED_HEADING = re.compile(
    rf"^(?:第\s*{NUMBER}\s*(?:章|回|卷|节|部|篇|幕|集)|"
    rf"(?:卷|篇)\s*{NUMBER}){TITLE_SUFFIX}$"
)
SPECIAL_HEADING = re.compile(
    rf"^(?:序章|序言|前言|楔子|引子|尾声|终章|终曲|后记|跋|"
    rf"番外(?:篇)?(?:\s*{NUMBER})?){TITLE_SUFFIX}$"
)
VOLUME_HEADING = re.compile(
    rf"^(?:第\s*{NUMBER}\s*(?:卷|部|篇)|(?:卷|篇)\s*{NUMBER}){TITLE_SUFFIX}$"
)


@dataclass(frozen=True)
class ChapterChunk:
    title: str
    text: str
    kind: str
    source_start: int
    source_end: int
    part_number: int = 1
    part_count: int = 1


@dataclass(frozen=True)
class _Heading:
    start: int
    end: int
    title: str
    kind: str


def normalize_text(text: str) -> str:
    if text.startswith("\ufeff"):
        text = text[1:]
    return text.replace("\r\n", "\n").replace("\r", "\n")


def decode_upload(raw: bytes) -> Tuple[str, str]:
    if raw.startswith(b"\xef\xbb\xbf"):
        return normalize_text(raw.decode("utf-8-sig")), "utf-8-sig"
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return normalize_text(raw.decode("utf-16")), "utf-16"

    for encoding in ("utf-8", "gb18030", "big5"):
        try:
            return normalize_text(raw.decode(encoding)), encoding
        except UnicodeDecodeError:
            continue
    raise ValueError("无法识别文本编码，请转换为 UTF-8 或 GB18030 后重试")


def _heading_for_line(line: str, start: int, end: int) -> _Heading | None:
    candidate = unicodedata.normalize("NFKC", line.strip())
    if not candidate or len(candidate) > 80:
        return None
    if NUMBERED_HEADING.fullmatch(candidate):
        kind = "volume" if VOLUME_HEADING.fullmatch(candidate) else "chapter"
        return _Heading(start=start, end=end, title=candidate, kind=kind)
    if SPECIAL_HEADING.fullmatch(candidate):
        return _Heading(start=start, end=end, title=candidate, kind="special")
    return None


def _find_headings(text: str) -> List[_Heading]:
    headings: List[_Heading] = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        end = cursor + len(line)
        heading = _heading_for_line(line.rstrip("\n"), cursor, end)
        if heading:
            headings.append(heading)
        cursor = end
    if cursor < len(text):
        heading = _heading_for_line(text[cursor:], cursor, len(text))
        if heading:
            headings.append(heading)
    return headings


def _suppress_table_of_contents(text: str, headings: Sequence[_Heading]) -> List[_Heading]:
    if len(headings) < 6:
        return list(headings)
    search_end = min(len(text), max(50_000, len(text) // 10))
    catalog_pos = text[:search_end].find("目录")
    if catalog_pos < 0:
        return list(headings)

    early = [h for h in headings if catalog_pos <= h.start < search_end]
    if len(early) < 5:
        return list(headings)

    seen_titles = set()
    for index, heading in enumerate(headings):
        if heading.title in seen_titles and index >= 5:
            return list(headings[index:])
        seen_titles.add(heading.title)
    return list(headings)


def _merge_volume_markers(headings: Sequence[_Heading]) -> List[_Heading]:
    merged: List[_Heading] = []
    index = 0
    while index < len(headings):
        current = headings[index]
        if current.kind == "volume" and index + 1 < len(headings):
            following = headings[index + 1]
            if following.kind != "volume" and following.start - current.end <= 500:
                merged.append(
                    replace(
                        following,
                        start=current.start,
                        title=f"{current.title} · {following.title}",
                    )
                )
                index += 2
                continue
        merged.append(current)
        index += 1
    return merged


def _best_cut(text: str, start: int, target: int, hard: int) -> int:
    hard_end = min(len(text), start + hard)
    if hard_end == len(text):
        return hard_end
    earliest = min(hard_end, start + max(1, int(target * 0.65)))
    preferred_end = min(hard_end, start + target)
    search_windows = [
        ("\n\n", earliest, hard_end),
        ("\n", earliest, hard_end),
        ("。", earliest, hard_end),
        ("！", earliest, hard_end),
        ("？", earliest, hard_end),
        ("；", earliest, hard_end),
        ("，", preferred_end, hard_end),
        ("、", preferred_end, hard_end),
    ]
    for separator, left, right in search_windows:
        position = text.rfind(separator, left, right)
        if position >= left:
            return position + len(separator)
    return hard_end


def _split_long_chunk(
    chunk: ChapterChunk, target_chars: int, max_chars: int
) -> List[ChapterChunk]:
    if len(chunk.text) <= max_chars:
        return [chunk]
    ranges: List[Tuple[int, int]] = []
    local_start = 0
    while local_start < len(chunk.text):
        local_end = _best_cut(chunk.text, local_start, target_chars, max_chars)
        if local_end <= local_start:
            local_end = min(len(chunk.text), local_start + max_chars)
        ranges.append((local_start, local_end))
        local_start = local_end

    total = len(ranges)
    return [
        ChapterChunk(
            title=f"{chunk.title}（{index}/{total}）",
            text=chunk.text[start:end],
            kind=chunk.kind,
            source_start=chunk.source_start + start,
            source_end=chunk.source_start + end,
            part_number=index,
            part_count=total,
        )
        for index, (start, end) in enumerate(ranges, start=1)
    ]


def split_chapters(
    raw_text: str, target_chars: int = 12_000, max_chars: int = 30_000
) -> List[ChapterChunk]:
    if target_chars <= 0 or max_chars <= 0 or target_chars > max_chars:
        raise ValueError("章节分块参数不合法")
    text = normalize_text(raw_text)
    if not text.strip():
        return []

    headings = _merge_volume_markers(
        _suppress_table_of_contents(text, _find_headings(text))
    )
    base_chunks: List[ChapterChunk] = []

    if not headings:
        base_chunks.append(
            ChapterChunk(
                title="全文",
                text=text,
                kind="fallback",
                source_start=0,
                source_end=len(text),
            )
        )
    else:
        first_start = headings[0].start
        if first_start > 0 and text[:first_start].strip():
            base_chunks.append(
                ChapterChunk(
                    title="书前内容",
                    text=text[:first_start],
                    kind="preamble",
                    source_start=0,
                    source_end=first_start,
                )
            )
        for index, heading in enumerate(headings):
            end = headings[index + 1].start if index + 1 < len(headings) else len(text)
            if end <= heading.start:
                continue
            base_chunks.append(
                ChapterChunk(
                    title=heading.title,
                    text=text[heading.start:end],
                    kind=heading.kind,
                    source_start=heading.start,
                    source_end=end,
                )
            )

    chunks: List[ChapterChunk] = []
    for chunk in base_chunks:
        chunks.extend(_split_long_chunk(chunk, target_chars, max_chars))
    return chunks


def joined_text(chunks: Iterable[ChapterChunk]) -> str:
    return "".join(chunk.text for chunk in chunks)
