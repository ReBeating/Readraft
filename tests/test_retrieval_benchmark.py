from __future__ import annotations

from app.retrieval_benchmark import load_suite, run_benchmark


def test_retrieval_benchmark_has_full_scope_safe_recall():
    report = run_benchmark(load_suite())

    assert report["case_count"] >= 6
    assert report["recall_at_5"] == 1.0
    assert report["mrr"] >= 0.75
    assert report["scope_violations"] == 0
    assert all(item["ranking"] for item in report["evaluations"])
