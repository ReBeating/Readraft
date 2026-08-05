from __future__ import annotations

import json
import re
import sqlite3
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


SEARCH_ENGINE = "sqlite_fts5_unicode61_cjk_ngrams_rrf_v2"
SEARCH_SCOPE = "author_confirmed_canon_before_current_chapter"
DEFAULT_QUERY_TERM_LIMIT = 48
DEFAULT_RESULT_LIMIT = 18
DEFAULT_CANDIDATE_MULTIPLIER = 6
RRF_K = 60

_CAUSAL_SOURCE_TYPES = {"event", "foreshadowing", "plot_thread"}
_CAUSAL_INTENT_MARKERS = {
    "为什么",
    "原因",
    "因果",
    "导致",
    "结果",
    "影响",
    "伏笔",
    "回收",
    "呼应",
}
_SOURCE_TYPE_PRIORITY = {
    "fact": 1,
    "knowledge": 2,
    "event": 3,
    "foreshadowing": 4,
    "plot_thread": 5,
    "chapter_memory": 6,
}

_LEXEME_PATTERN = re.compile(
    r"[A-Za-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]+"
)
_CJK_PATTERN = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff]+$")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_STOP_TERMS = {
    "一个",
    "一些",
    "以及",
    "不能",
    "人物",
    "任务",
    "场景",
    "开始",
    "当前",
    "必须",
    "情节",
    "故事",
    "本章",
    "目标",
    "章节",
    "结束",
    "进行",
    "需要",
}


def _flatten_text(value: Any) -> Iterator[str]:
    if value is None:
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _flatten_text(item)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _flatten_text(item)
        return
    text = str(value).strip()
    if not text:
        return
    if text[:1] in {"[", "{"}:
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            pass
        else:
            yield from _flatten_text(parsed)
            return
    yield text


def _lexeme_terms(lexeme: str) -> Iterator[str]:
    if _CJK_PATTERN.fullmatch(lexeme):
        if len(lexeme) == 1:
            return
        if len(lexeme) <= 8 and lexeme not in _STOP_TERMS:
            yield lexeme
        for width in (2, 3):
            if len(lexeme) < width:
                continue
            for offset in range(len(lexeme) - width + 1):
                term = lexeme[offset : offset + width]
                if term not in _STOP_TERMS:
                    yield term
        return
    normalized = lexeme.casefold()
    if len(normalized) >= 2:
        yield normalized


def build_search_terms(
    values: Iterable[Any],
    *,
    max_terms: int | None = None,
) -> list[str]:
    """Build deterministic Chinese-friendly terms for an FTS5 unicode index."""

    seen: set[str] = set()
    terms: list[str] = []
    for value in values:
        for text in _flatten_text(value):
            for lexeme in _LEXEME_PATTERN.findall(text):
                for term in _lexeme_terms(lexeme):
                    if term in seen:
                        continue
                    seen.add(term)
                    terms.append(term)
                    if max_terms is not None and len(terms) >= max_terms:
                        return terms
    return terms


def build_query_terms(
    *,
    chapter: Mapping[str, Any],
    characters: Sequence[Mapping[str, Any]],
    task_card: Mapping[str, Any] | None,
    scenes: Sequence[Mapping[str, Any]],
    focused_scene_id: str | None,
    retrieval_hint: str,
    max_terms: int = DEFAULT_QUERY_TERM_LIMIT,
) -> list[str]:
    """Prioritize concrete scene/task nouns before broad chapter prose."""

    values = _prioritized_query_values(
        chapter=chapter,
        characters=characters,
        task_card=task_card,
        scenes=scenes,
        focused_scene_id=focused_scene_id,
        retrieval_hint=retrieval_hint,
    )
    return build_search_terms(values, max_terms=max_terms)


def _prioritized_query_values(
    *,
    chapter: Mapping[str, Any],
    characters: Sequence[Mapping[str, Any]],
    task_card: Mapping[str, Any] | None,
    scenes: Sequence[Mapping[str, Any]],
    focused_scene_id: str | None,
    retrieval_hint: str,
) -> list[Any]:
    focused_scenes = [
        scene
        for scene in scenes
        if not focused_scene_id
        or str(scene.get("id") or "") == focused_scene_id
    ]
    if focused_scene_id and not focused_scenes:
        focused_scenes = list(scenes)
    values: list[Any] = []
    for scene in focused_scenes:
        values.extend(
            [
                scene.get("key_items"),
                scene.get("pov_character"),
                scene.get("location"),
                scene.get("goal"),
                scene.get("reveal"),
                scene.get("conceal"),
            ]
        )
    if task_card:
        values.extend(
            [
                task_card.get("plot_threads"),
                task_card.get("foreshadow_payoff"),
                task_card.get("foreshadow_setup"),
                task_card.get("must_preserve"),
            ]
        )
    values.append(retrieval_hint)
    for scene in focused_scenes:
        values.extend(
            [
                scene.get("obstacle"),
                scene.get("action"),
                scene.get("end_state"),
                scene.get("transition"),
            ]
        )
    if task_card:
        values.extend(
            [
                task_card.get("must_happen"),
                task_card.get("forbidden"),
                task_card.get("purpose"),
                task_card.get("central_conflict"),
                task_card.get("ending_hook"),
            ]
        )
    values.extend(
        [
            chapter.get("key_points"),
            chapter.get("title"),
            chapter.get("outline"),
            [character.get("name") for character in characters],
        ]
    )
    return values


def build_query_concepts(
    *,
    chapter: Mapping[str, Any],
    characters: Sequence[Mapping[str, Any]],
    task_card: Mapping[str, Any] | None,
    scenes: Sequence[Mapping[str, Any]],
    focused_scene_id: str | None,
    retrieval_hint: str,
    max_concepts: int = 16,
) -> list[str]:
    """Return a concise, human-readable trace separate from FTS n-grams."""

    values = _prioritized_query_values(
        chapter=chapter,
        characters=characters,
        task_card=task_card,
        scenes=scenes,
        focused_scene_id=focused_scene_id,
        retrieval_hint=retrieval_hint,
    )
    seen: set[str] = set()
    concepts: list[str] = []
    for value in values:
        for text in _flatten_text(value):
            for lexeme in _LEXEME_PATTERN.findall(text):
                normalized = (
                    lexeme if _CJK_PATTERN.fullmatch(lexeme) else lexeme.casefold()
                )
                if (
                    len(normalized) < 2
                    or len(normalized) > 12
                    or normalized in _STOP_TERMS
                    or normalized in seen
                ):
                    continue
                seen.add(normalized)
                concepts.append(normalized)
                if len(concepts) >= max_concepts:
                    return concepts
    return concepts


def _compact_text(value: Any, *, max_chars: int | None = None) -> str:
    parts = [
        _WHITESPACE_PATTERN.sub(" ", part).strip()
        for part in _flatten_text(value)
    ]
    text = "；".join(part for part in parts if part)
    if max_chars is not None and len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "…"
    return text


def _upsert_document(
    connection: sqlite3.Connection,
    *,
    source_type: str,
    source_id: str,
    project_id: str,
    branch_id: str,
    chapter_id: str,
    chapter_position: int,
    chapter_title: str,
    title: str,
    body: Any,
    keywords: Any,
    created_at: str,
) -> None:
    clean_body = _compact_text(body)
    clean_keywords = _compact_text(keywords)
    search_terms = " ".join(
        build_search_terms(
            (title, clean_body, clean_keywords), max_terms=768
        )
    )
    connection.execute(
        """
        INSERT INTO story_memory_search_documents(
            id, source_type, source_id, project_id, branch_id,
            chapter_id, chapter_position, chapter_title, title, body,
            keywords, search_terms, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_type, source_id) DO UPDATE SET
            project_id=excluded.project_id,
            branch_id=excluded.branch_id,
            chapter_id=excluded.chapter_id,
            chapter_position=excluded.chapter_position,
            chapter_title=excluded.chapter_title,
            title=excluded.title,
            body=excluded.body,
            keywords=excluded.keywords,
            search_terms=excluded.search_terms,
            created_at=excluded.created_at
        """,
        (
            f"{source_type}:{source_id}",
            source_type,
            source_id,
            project_id,
            branch_id,
            chapter_id,
            chapter_position,
            chapter_title,
            title,
            clean_body,
            clean_keywords,
            search_terms,
            created_at,
        ),
    )


def delete_chapter_search_documents(
    connection: sqlite3.Connection, *, chapter_id: str
) -> None:
    connection.execute(
        "DELETE FROM story_memory_search_documents WHERE chapter_id=?",
        (chapter_id,),
    )


def _chapter_filter(
    alias: str, chapter_id: str | None
) -> tuple[str, tuple[str, ...]]:
    if chapter_id is None:
        return "", ()
    return f" AND {alias}.chapter_id=?", (chapter_id,)


def _index_canonical_documents(
    connection: sqlite3.Connection,
    *,
    chapter_id: str | None,
    fallback_created_at: str,
) -> None:
    clause, parameters = _chapter_filter("m", chapter_id)
    for row in connection.execute(
        f"""
        SELECT m.*, ch.position AS chapter_position,
               ch.title AS chapter_title
        FROM chapter_memory m
        JOIN novel_chapters ch ON ch.id=m.chapter_id
        WHERE m.record_status='canon'{clause}
        ORDER BY ch.position, m.created_at, m.id
        """,
        parameters,
    ).fetchall():
        _upsert_document(
            connection,
            source_type="chapter_memory",
            source_id=str(row["id"]),
            project_id=str(row["project_id"]),
            branch_id=str(row["branch_id"]),
            chapter_id=str(row["chapter_id"]),
            chapter_position=int(row["chapter_position"]),
            chapter_title=str(row["chapter_title"]),
            title=f"章节摘要 · {row['chapter_title']}",
            body=(
                row["summary"],
                row["key_events_json"],
                row["unresolved_questions_json"],
            ),
            keywords=row["keywords_json"],
            created_at=str(row["created_at"] or fallback_created_at),
        )

    clause, parameters = _chapter_filter("e", chapter_id)
    for row in connection.execute(
        f"""
        SELECT e.*, ch.position AS chapter_position,
               ch.title AS chapter_title
        FROM story_events e
        JOIN novel_chapters ch ON ch.id=e.chapter_id
        WHERE e.record_status='canon'{clause}
        ORDER BY ch.position, e.position, e.id
        """,
        parameters,
    ).fetchall():
        event_key = (
            row["event_key"] if "event_key" in row.keys() else ""
        )
        cause_event_keys = (
            row["cause_event_keys_json"]
            if "cause_event_keys_json" in row.keys()
            else "[]"
        )
        _upsert_document(
            connection,
            source_type="event",
            source_id=str(row["id"]),
            project_id=str(row["project_id"]),
            branch_id=str(row["branch_id"]),
            chapter_id=str(row["chapter_id"]),
            chapter_position=int(row["chapter_position"]),
            chapter_title=str(row["chapter_title"]),
            title=f"事件 · {str(row['summary'])[:80]}",
            body=(
                event_key,
                row["summary"],
                row["participants_json"],
                row["location"],
                row["story_time"],
                row["causes_json"],
                cause_event_keys,
                row["effects_json"],
                row["evidence"],
            ),
            keywords=(
                event_key,
                row["participants_json"],
                row["location"],
                row["story_time"],
            ),
            created_at=str(row["created_at"] or fallback_created_at),
        )

    clause, parameters = _chapter_filter("f", chapter_id)
    for row in connection.execute(
        f"""
        SELECT f.*, ch.position AS chapter_position,
               ch.title AS chapter_title
        FROM story_facts f
        JOIN novel_chapters ch ON ch.id=f.chapter_id
        WHERE f.fact_status='canon'{clause}
        ORDER BY ch.position, f.created_at, f.id
        """,
        parameters,
    ).fetchall():
        _upsert_document(
            connection,
            source_type="fact",
            source_id=str(row["id"]),
            project_id=str(row["project_id"]),
            branch_id=str(row["branch_id"]),
            chapter_id=str(row["chapter_id"]),
            chapter_position=int(row["chapter_position"]),
            chapter_title=str(row["chapter_title"]),
            title=f"事实 · {row['subject_name']} · {row['predicate']}",
            body=(
                row["subject_name"],
                row["predicate"],
                row["object_json"],
                row["evidence"],
            ),
            keywords=(
                row["fact_type"],
                row["subject_type"],
                row["subject_name"],
                row["predicate"],
            ),
            created_at=str(row["created_at"] or fallback_created_at),
        )

    clause, parameters = _chapter_filter("k", chapter_id)
    for row in connection.execute(
        f"""
        SELECT k.*, ch.position AS chapter_position,
               ch.title AS chapter_title
        FROM character_knowledge k
        JOIN novel_chapters ch ON ch.id=k.chapter_id
        WHERE k.record_status='canon'{clause}
        ORDER BY ch.position, k.created_at, k.id
        """,
        parameters,
    ).fetchall():
        _upsert_document(
            connection,
            source_type="knowledge",
            source_id=str(row["id"]),
            project_id=str(row["project_id"]),
            branch_id=str(row["branch_id"]),
            chapter_id=str(row["chapter_id"]),
            chapter_position=int(row["chapter_position"]),
            chapter_title=str(row["chapter_title"]),
            title=f"人物知情 · {row['character_name']}",
            body=(
                row["fact_text"],
                row["knowledge_state"],
                row["learned_via"],
                row["evidence"],
            ),
            keywords=(row["character_name"], row["knowledge_state"]),
            created_at=str(row["created_at"] or fallback_created_at),
        )

    clause, parameters = _chapter_filter("t", chapter_id)
    for row in connection.execute(
        f"""
        SELECT t.*, ch.position AS chapter_position,
               ch.title AS chapter_title
        FROM plot_threads t
        JOIN novel_chapters ch ON ch.id=t.chapter_id
        WHERE t.record_status='canon'{clause}
        ORDER BY ch.position, t.created_at, t.id
        """,
        parameters,
    ).fetchall():
        _upsert_document(
            connection,
            source_type="plot_thread",
            source_id=str(row["id"]),
            project_id=str(row["project_id"]),
            branch_id=str(row["branch_id"]),
            chapter_id=str(row["chapter_id"]),
            chapter_position=int(row["chapter_position"]),
            chapter_title=str(row["chapter_title"]),
            title=f"剧情线 · {row['thread_name']}",
            body=(
                row["action"],
                row["update_text"],
                row["promise"],
                row["target_payoff"],
                row["evidence"],
            ),
            keywords=(row["thread_name"], row["thread_type"]),
            created_at=str(row["created_at"] or fallback_created_at),
        )

    clause, parameters = _chapter_filter("f", chapter_id)
    for row in connection.execute(
        f"""
        SELECT f.*, ch.position AS chapter_position,
               ch.title AS chapter_title
        FROM foreshadowing f
        JOIN novel_chapters ch ON ch.id=f.chapter_id
        WHERE f.record_status='canon'{clause}
        ORDER BY ch.position, f.created_at, f.id
        """,
        parameters,
    ).fetchall():
        _upsert_document(
            connection,
            source_type="foreshadowing",
            source_id=str(row["id"]),
            project_id=str(row["project_id"]),
            branch_id=str(row["branch_id"]),
            chapter_id=str(row["chapter_id"]),
            chapter_position=int(row["chapter_position"]),
            chapter_title=str(row["chapter_title"]),
            title=f"伏笔 · {row['hook_name']}",
            body=(
                row["action"],
                row["description"],
                row["intended_payoff"],
                row["evidence"],
            ),
            keywords=(row["hook_name"], row["action"]),
            created_at=str(row["created_at"] or fallback_created_at),
        )


def replace_chapter_search_documents(
    connection: sqlite3.Connection,
    *,
    chapter_id: str,
    created_at: str,
) -> None:
    delete_chapter_search_documents(connection, chapter_id=chapter_id)
    _index_canonical_documents(
        connection,
        chapter_id=chapter_id,
        fallback_created_at=created_at,
    )


def rebuild_memory_search_documents(
    connection: sqlite3.Connection, *, created_at: str
) -> None:
    connection.execute("DELETE FROM story_memory_search_documents")
    _index_canonical_documents(
        connection,
        chapter_id=None,
        fallback_created_at=created_at,
    )


def search_memory_documents(
    connection: sqlite3.Connection,
    *,
    project_id: str,
    branch_id: str,
    before_chapter_position: int,
    query_terms: Sequence[str],
    query_concepts: Sequence[str] = (),
    excluded_chapter_ids: Sequence[str] = (),
    limit: int = DEFAULT_RESULT_LIMIT,
) -> list[dict[str, Any]]:
    clean_terms = [
        term
        for term in dict.fromkeys(str(term).strip() for term in query_terms)
        if term
    ]
    if not clean_terms or limit <= 0:
        return []
    match_query = " OR ".join(
        f'"{term.replace(chr(34), chr(34) * 2)}"' for term in clean_terms
    )
    excluded = [
        str(chapter_id)
        for chapter_id in dict.fromkeys(excluded_chapter_ids)
        if chapter_id
    ]
    exclusion_sql = ""
    parameters: list[Any] = [
        match_query,
        project_id,
        branch_id,
        before_chapter_position,
    ]
    if excluded:
        exclusion_sql = (
            " AND d.chapter_id NOT IN ("
            + ",".join("?" for _ in excluded)
            + ")"
        )
        parameters.extend(excluded)
    candidate_limit = min(
        240,
        max(limit * DEFAULT_CANDIDATE_MULTIPLIER, 60),
    )
    parameters.append(candidate_limit)
    rows = connection.execute(
        f"""
        SELECT d.source_type, d.source_id, d.chapter_id,
               d.chapter_position, d.chapter_title, d.title, d.body,
               d.keywords, d.search_terms,
               bm25(story_memory_fts, 6.0, 1.0, 3.0, 0.5) AS score
        FROM story_memory_fts
        JOIN story_memory_search_documents d
          ON d.rowid=story_memory_fts.rowid
        WHERE story_memory_fts MATCH ?
          AND d.project_id=? AND d.branch_id=?
          AND d.chapter_position<?{exclusion_sql}
        ORDER BY score ASC, d.chapter_position DESC,
                 d.source_type, d.source_id
        LIMIT ?
        """,
        tuple(parameters),
    ).fetchall()
    concepts = [
        concept
        for concept in dict.fromkeys(
            str(concept).strip().casefold() for concept in query_concepts
        )
        if len(concept) >= 2
    ][:48]
    causal_intent = any(
        marker in value
        for value in (*concepts, *clean_terms)
        for marker in _CAUSAL_INTENT_MARKERS
    )
    candidates: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        indexed_terms = set(str(row["search_terms"] or "").split())
        matched_terms = [
            term for term in clean_terms if term in indexed_terms
        ][:8]
        title = str(row["title"])
        body = str(row["body"])
        keywords = str(row["keywords"])
        folded_title = title.casefold()
        folded_body = body.casefold()
        folded_keywords = keywords.casefold()
        matched_concepts = [
            concept
            for concept in concepts
            if concept in folded_title
            or concept in folded_keywords
            or concept in folded_body
        ]
        exact_strength = sum(
            3 * int(concept in folded_title)
            + 2 * int(concept in folded_keywords)
            + int(concept in folded_body)
            for concept in matched_concepts
        )
        source_type = str(row["source_type"])
        candidates.append(
            {
                "bm25_rank": rank,
                "source_type": source_type,
                "source_id": str(row["source_id"]),
                "source_chapter_id": str(row["chapter_id"]),
                "source_chapter_position": int(row["chapter_position"]),
                "source_chapter_title": str(row["chapter_title"]),
                "title": title,
                "excerpt": _compact_text(body, max_chars=900),
                "keywords": _compact_text(keywords, max_chars=240),
                "matched_terms": matched_terms,
                "matched_concepts": matched_concepts[:8],
                "exact_strength": exact_strength,
                "causal_match": bool(
                    causal_intent
                    and matched_concepts
                    and source_type in _CAUSAL_SOURCE_TYPES
                ),
                "bm25_score": round(-float(row["score"] or 0.0), 8),
            }
        )
    return _fuse_memory_candidates(candidates, limit=limit)


def _rank_map(
    candidates: Sequence[Mapping[str, Any]],
    *,
    predicate: Callable[[Mapping[str, Any]], bool],
    sort_key: Callable[[Mapping[str, Any]], Any],
) -> dict[str, int]:
    ranked = sorted(
        (candidate for candidate in candidates if predicate(candidate)),
        key=sort_key,
    )
    return {
        _candidate_key(candidate): rank
        for rank, candidate in enumerate(ranked, start=1)
    }


def _candidate_key(candidate: Mapping[str, Any]) -> str:
    return f"{candidate.get('source_type', '')}:{candidate.get('source_id', '')}"


def _fuse_memory_candidates(
    candidates: Sequence[Mapping[str, Any]], *, limit: int
) -> list[dict[str, Any]]:
    """Fuse lexical, exact, causal and recency ranks without a vector service."""

    if limit <= 0:
        return []
    exact_ranks = _rank_map(
        candidates,
        predicate=lambda item: int(item.get("exact_strength") or 0) > 0,
        sort_key=lambda item: (
            -int(item.get("exact_strength") or 0),
            int(item.get("bm25_rank") or 10**9),
        ),
    )
    causal_ranks = _rank_map(
        candidates,
        predicate=lambda item: bool(item.get("causal_match")),
        sort_key=lambda item: (
            -int(item.get("exact_strength") or 0),
            -int(item.get("source_chapter_position") or 0),
            int(item.get("bm25_rank") or 10**9),
        ),
    )
    recency_ranks = _rank_map(
        candidates,
        predicate=lambda _item: True,
        sort_key=lambda item: (
            -int(item.get("source_chapter_position") or 0),
            int(item.get("bm25_rank") or 10**9),
        ),
    )
    fused: list[dict[str, Any]] = []
    for raw in candidates:
        item = dict(raw)
        candidate_key = _candidate_key(item)
        bm25_rank = int(item.get("bm25_rank") or 10**9)
        exact_rank = exact_ranks.get(candidate_key)
        causal_rank = causal_ranks.get(candidate_key)
        recency_rank = recency_ranks[candidate_key]
        source_priority = _SOURCE_TYPE_PRIORITY.get(
            str(item.get("source_type") or ""), 12
        )
        fusion_score = 1.0 / (RRF_K + bm25_rank)
        if exact_rank is not None:
            fusion_score += 1.8 / (RRF_K + exact_rank)
        if causal_rank is not None:
            fusion_score += 0.9 / (RRF_K + causal_rank)
        fusion_score += 0.35 / (RRF_K + recency_rank)
        fusion_score += 0.15 / (RRF_K + source_priority)
        item["fusion_score"] = round(fusion_score, 8)
        item["score"] = item["fusion_score"]
        item["ranking_signals"] = {
            "bm25_rank": bm25_rank,
            "exact_rank": exact_rank,
            "causal_rank": causal_rank,
            "recency_rank": recency_rank,
            "source_priority": source_priority,
        }
        fused.append(item)
    fused.sort(
        key=lambda item: (
            -float(item["fusion_score"]),
            -int(item.get("exact_strength") or 0),
            int(item.get("bm25_rank") or 10**9),
            -int(item.get("source_chapter_position") or 0),
            str(item.get("source_type") or ""),
            str(item.get("source_id") or ""),
        )
    )
    results = fused[:limit]
    for rank, item in enumerate(results, start=1):
        item["rank"] = rank
    return results
