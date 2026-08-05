import json
from pathlib import Path

from app.writing_benchmark import (
    BenchmarkOutput,
    build_report,
    evaluate_output,
    load_cases,
    write_report,
)


def test_benchmark_detects_constraints_and_is_deterministic(tmp_path: Path):
    case = load_cases()[0]
    good = BenchmarkOutput(
        case_id=case["id"],
        quality_mode="standard",
        content=case["baseline_output"],
    )
    bad = BenchmarkOutput(
        case_id=case["id"],
        quality_mode="low",
        content="周岚心想，她早已知道录音内容。林越拿出U盘。",
    )

    good_result = evaluate_output(case, good)
    bad_result = evaluate_output(case, bad)

    assert good_result["score"] > bad_result["score"]
    assert bad_result["hard_failures"] == [
        "continuity_or_constraint",
        "viewpoint_leak",
    ]
    assert evaluate_output(case, good) == good_result


def test_instruction_score_never_exceeds_one_hundred():
    case = load_cases()[2]
    result = evaluate_output(
        case,
        BenchmarkOutput(
            case_id=case["id"],
            quality_mode="max",
            content=case["baseline_output"],
        ),
    )

    assert result["dimensions"]["instruction_fulfillment"] <= 100
    assert result["score"] <= 100


def test_benchmark_writes_machine_human_and_html_reports(tmp_path: Path):
    cases = load_cases()
    outputs = [
        BenchmarkOutput(
            case_id=case["id"],
            quality_mode="standard",
            content=case["baseline_output"],
        )
        for case in cases
    ]
    report = build_report(cases, outputs, seed=42)

    write_report(report, tmp_path)

    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "report.html").exists()
    assert (tmp_path / "blind-review.csv").exists()
    key = json.loads((tmp_path / "blind-key.json").read_text(encoding="utf-8"))
    assert len(key) == len(cases)
    assert {item["sample_id"] for item in key} == {"S001", "S002", "S003"}
