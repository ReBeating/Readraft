"""Repeatable retrieval benchmark for long-form story memory."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from .memory_search import (
    SEARCH_ENGINE,
    build_search_terms,
    search_memory_documents,
)


CASE_PATH = Path(__file__).resolve().parent / "benchmark_cases" / "retrieval.json"


def load_suite(path: Path = CASE_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("检索基准必须是 JSON object")
    corpus = payload.get("corpus")
    cases = payload.get("cases")
    if not isinstance(corpus, list) or not isinstance(cases, list):
        raise ValueError("检索基准必须包含 corpus 与 cases 数组")
    return payload


def _connection(corpus: list[Mapping[str, Any]]) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE story_memory_search_documents (
            id TEXT PRIMARY KEY,
            source_type TEXT NOT NULL,
            source_id TEXT NOT NULL,
            project_id TEXT NOT NULL,
            branch_id TEXT NOT NULL,
            chapter_id TEXT NOT NULL,
            chapter_position INTEGER NOT NULL,
            chapter_title TEXT NOT NULL,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            keywords TEXT NOT NULL,
            search_terms TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(source_type, source_id)
        );
        CREATE VIRTUAL TABLE story_memory_fts USING fts5(
            title, body, keywords, search_terms,
            content='story_memory_search_documents',
            content_rowid='rowid',
            tokenize='unicode61 remove_diacritics 2'
        );
        """
    )
    for index, item in enumerate(corpus, start=1):
        title = str(item.get("title") or "")
        body = str(item.get("body") or "")
        keywords = str(item.get("keywords") or "")
        terms = " ".join(build_search_terms((title, body, keywords), max_terms=768))
        connection.execute(
            """
            INSERT INTO story_memory_search_documents(
                id, source_type, source_id, project_id, branch_id,
                chapter_id, chapter_position, chapter_title, title, body,
                keywords, search_terms, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"{item['source_type']}:{item['source_id']}",
                item["source_type"],
                item["source_id"],
                item["project_id"],
                item["branch_id"],
                item["chapter_id"],
                int(item["chapter_position"]),
                item["chapter_title"],
                title,
                body,
                keywords,
                terms,
                f"2026-01-{index:02d}T00:00:00+00:00",
            ),
        )
    connection.execute("INSERT INTO story_memory_fts(story_memory_fts) VALUES ('rebuild')")
    return connection


def evaluate_case(
    connection: sqlite3.Connection, case: Mapping[str, Any], *, limit: int = 5
) -> dict[str, Any]:
    expected = [str(item) for item in case.get("expected") or []]
    forbidden = set(str(item) for item in case.get("forbidden") or [])
    query_values = [case.get("query"), case.get("expansions")]
    results = search_memory_documents(
        connection,
        project_id=str(case["project_id"]),
        branch_id=str(case["branch_id"]),
        before_chapter_position=int(case["before_chapter_position"]),
        query_terms=build_search_terms(query_values, max_terms=96),
        query_concepts=[str(item) for item in case.get("concepts") or []],
        excluded_chapter_ids=[
            str(item) for item in case.get("excluded_chapter_ids") or []
        ],
        limit=limit,
    )
    retrieved = [
        f"{item['source_type']}:{item['source_id']}" for item in results
    ]
    hits = [item for item in expected if item in retrieved]
    first_rank = min(
        (retrieved.index(item) + 1 for item in expected if item in retrieved),
        default=0,
    )
    forbidden_hits = [item for item in retrieved if item in forbidden]
    return {
        "id": str(case["id"]),
        "query": str(case["query"]),
        "recall_at_5": round(len(hits) / max(1, len(expected)), 4),
        "reciprocal_rank": round(1 / first_rank, 4) if first_rank else 0.0,
        "scope_violations": forbidden_hits,
        "expected": expected,
        "retrieved": retrieved,
        "ranking": [
            {
                "id": retrieved[index],
                "score": item["fusion_score"],
                "signals": item["ranking_signals"],
                "matched_concepts": item["matched_concepts"],
            }
            for index, item in enumerate(results)
        ],
    }


def run_benchmark(
    suite: Mapping[str, Any], *, limit: int = 5
) -> dict[str, Any]:
    connection = _connection(list(suite["corpus"]))
    try:
        evaluations = [
            evaluate_case(connection, case, limit=limit)
            for case in suite["cases"]
        ]
    finally:
        connection.close()
    return {
        "schema_version": "1.0",
        "engine": SEARCH_ENGINE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(evaluations),
        "recall_at_5": round(mean(item["recall_at_5"] for item in evaluations), 4),
        "mrr": round(mean(item["reciprocal_rank"] for item in evaluations), 4),
        "scope_violations": sum(
            len(item["scope_violations"]) for item in evaluations
        ),
        "evaluations": evaluations,
    }


def write_report(report: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = [
        f"| {item['id']} | {item['recall_at_5']:.2f} | "
        f"{item['reciprocal_rank']:.2f} | {len(item['scope_violations'])} |"
        for item in report["evaluations"]
    ]
    markdown = "\n".join(
        [
            "# Readraft 长篇记忆检索基准",
            "",
            f"引擎：`{report['engine']}`",
            "",
            f"Recall@5：**{report['recall_at_5']:.3f}** · "
            f"MRR：**{report['mrr']:.3f}** · "
            f"越界命中：**{report['scope_violations']}**",
            "",
            "| 用例 | Recall@5 | MRR | 越界命中 |",
            "|---|---:|---:|---:|",
            *rows,
            "",
        ]
    )
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 Readraft 长篇记忆检索基准")
    parser.add_argument("--cases", type=Path, default=CASE_PATH)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/retrieval"))
    args = parser.parse_args()
    report = run_benchmark(load_suite(args.cases))
    write_report(report, args.output)
    print(args.output / "report.md")


if __name__ == "__main__":
    main()
