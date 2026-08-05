"""Repeatable Chinese-novel writing benchmark and report generator."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import html
import importlib.metadata
import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

from .agent_model import ProviderAgentModel
from .config import Settings
from .model_provider import (
    settings_for_credential,
    settings_for_reasoning_policy,
)
from .model_routing import route_model_task
from .prose_pipeline import ProseDraftPipeline
from .text_metrics import effective_char_count


CASE_PATH = Path(__file__).resolve().parent / "benchmark_cases" / "core.json"
QUALITY_MODES = ("low", "standard", "max")
META_ARTIFACTS = (
    "以下是",
    "作为一个AI",
    "希望这段",
    "写作说明",
    "创作思路",
    "```",
)
CLICHES = (
    "不禁倒吸一口凉气",
    "嘴角勾起一抹",
    "眼神中闪过一丝",
    "空气仿佛凝固",
    "时间仿佛静止",
    "内心五味杂陈",
)


@dataclass(frozen=True)
class BenchmarkOutput:
    case_id: str
    quality_mode: str
    content: str
    latency_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    provider: str = "saved"
    model: str = "saved-response"


def load_cases(path: Path = CASE_PATH) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("基准用例文件必须是 JSON 数组")
    cases = []
    seen: set[str] = set()
    for raw in payload:
        if not isinstance(raw, dict):
            raise ValueError("基准用例必须是 JSON object")
        case_id = str(raw.get("id") or "").strip()
        if not case_id or case_id in seen:
            raise ValueError("基准用例编号为空或重复")
        seen.add(case_id)
        cases.append(dict(raw))
    return cases


def _contains_any(text: str, values: Iterable[str]) -> bool:
    return any(value and value in text for value in values)


def _ratio_score(hits: int, total: int) -> float:
    return 100.0 if total == 0 else round(hits / total * 100, 1)


def _repetition_score(text: str) -> tuple[float, dict[str, Any]]:
    sentences = [
        item.strip()
        for item in re.split(r"[。！？!?\n]+", text)
        if len(item.strip()) >= 4
    ]
    repeated_sentences = len(sentences) - len(set(sentences))
    compact = re.sub(r"[^\w\u4e00-\u9fff]", "", text)
    grams = [compact[index : index + 6] for index in range(max(0, len(compact) - 5))]
    counts: dict[str, int] = {}
    for gram in grams:
        counts[gram] = counts.get(gram, 0) + 1
    repeated_grams = sum(count - 1 for count in counts.values() if count > 1)
    denominator = max(1, len(sentences) + len(grams) * 0.08)
    penalty = (repeated_sentences * 18 + repeated_grams * 0.35) / denominator * 100
    return max(0.0, round(100 - min(100, penalty), 1)), {
        "repeated_sentences": repeated_sentences,
        "repeated_sixgrams": repeated_grams,
    }


def evaluate_output(
    case: Mapping[str, Any], output: BenchmarkOutput
) -> dict[str, Any]:
    text = str(output.content or "").strip()
    checks = dict(case.get("checks") or {})
    required_all = [str(item) for item in checks.get("required_all") or []]
    required_any = [str(item) for item in checks.get("required_any") or []]
    final_state_any = [str(item) for item in checks.get("final_state_any") or []]
    forbidden = [str(item) for item in checks.get("forbidden") or []]
    forbidden_viewpoint = [
        str(item) for item in checks.get("forbidden_viewpoint") or []
    ]
    all_hits = sum(item in text for item in required_all)
    any_hit = not required_any or _contains_any(text, required_any)
    final_hit = not final_state_any or _contains_any(text[-400:], final_state_any)
    instruction_parts = len(required_all) + int(bool(required_any)) + int(bool(final_state_any))
    instruction_hits = (
        all_hits
        + int(bool(required_any) and any_hit)
        + int(bool(final_state_any) and final_hit)
    )
    instruction_score = _ratio_score(instruction_hits, instruction_parts)

    forbidden_hits = [item for item in forbidden if item in text]
    viewpoint_hits = [item for item in forbidden_viewpoint if item in text]
    continuity_score = max(0.0, 100.0 - len(forbidden_hits) * 35.0)
    viewpoint_score = max(0.0, 100.0 - len(viewpoint_hits) * 50.0)

    char_count = effective_char_count(text)
    target_min = int(checks.get("target_min") or 1)
    target_max = int(checks.get("target_max") or max(target_min, 1))
    if target_min <= char_count <= target_max:
        length_score = 100.0
    elif char_count < target_min:
        length_score = round(max(0, char_count / max(1, target_min) * 100), 1)
    else:
        length_score = round(max(0, target_max / max(1, char_count) * 100), 1)

    artifact_hits = [item for item in (*META_ARTIFACTS, *CLICHES) if item in text]
    artifact_score = max(0.0, 100.0 - len(artifact_hits) * 18.0)
    repetition_score, repetition_detail = _repetition_score(text)
    dialogue_chars = sum(
        len(match.group(1))
        for match in re.finditer(r"[“\"]([^”\"]+)[”\"]", text)
    )
    paragraph_count = len([item for item in re.split(r"\n\s*\n", text) if item.strip()])

    dimensions = {
        "instruction_fulfillment": instruction_score,
        "continuity": continuity_score,
        "viewpoint": viewpoint_score,
        "length_control": length_score,
        "non_repetition": repetition_score,
        "artifact_free": artifact_score,
    }
    weights = {
        "instruction_fulfillment": 0.25,
        "continuity": 0.25,
        "viewpoint": 0.15,
        "length_control": 0.05,
        "non_repetition": 0.15,
        "artifact_free": 0.15,
    }
    total = round(sum(dimensions[key] * weights[key] for key in weights), 1)
    hard_failures = []
    if forbidden_hits:
        hard_failures.append("continuity_or_constraint")
    if viewpoint_hits:
        hard_failures.append("viewpoint_leak")
    if not text:
        hard_failures.append("empty")
    return {
        "case_id": output.case_id,
        "case_title": str(case.get("title") or output.case_id),
        "quality_mode": output.quality_mode,
        "provider": output.provider,
        "model": output.model,
        "score": total,
        "hard_failures": hard_failures,
        "dimensions": dimensions,
        "violations": {
            "forbidden": forbidden_hits,
            "viewpoint": viewpoint_hits,
            "artifacts": artifact_hits,
        },
        "metrics": {
            "effective_chars": char_count,
            "paragraphs": paragraph_count,
            "dialogue_ratio": round(dialogue_chars / max(1, len(text)), 4),
            **repetition_detail,
            "latency_seconds": round(output.latency_seconds, 3),
            "input_tokens": output.input_tokens,
            "output_tokens": output.output_tokens,
        },
        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "content": text,
    }


def _response_path(directory: Path, case_id: str, mode: str) -> Path:
    return directory / f"{case_id}.{mode}.txt"


def load_saved_outputs(
    cases: Iterable[Mapping[str, Any]],
    modes: Iterable[str],
    directory: Path | None,
) -> list[BenchmarkOutput]:
    outputs = []
    for case in cases:
        case_id = str(case["id"])
        for mode in modes:
            path = _response_path(directory, case_id, mode) if directory else None
            if path and path.exists():
                content = path.read_text(encoding="utf-8")
                metadata_path = path.with_suffix(".json")
                metadata = (
                    json.loads(metadata_path.read_text(encoding="utf-8"))
                    if metadata_path.exists()
                    else {}
                )
            elif directory:
                raise ValueError(f"缺少基准响应：{path}")
            else:
                content = str(case.get("baseline_output") or "")
                metadata = {}
            outputs.append(
                BenchmarkOutput(
                    case_id=case_id,
                    quality_mode=mode,
                    content=content,
                    latency_seconds=float(metadata.get("latency_seconds") or 0),
                    input_tokens=int(metadata.get("input_tokens") or 0),
                    output_tokens=int(metadata.get("output_tokens") or 0),
                    provider=str(metadata.get("provider") or "fixture"),
                    model=str(metadata.get("model") or "baseline"),
                )
            )
    return outputs


def _credential_from_env(role: str, base: Settings) -> tuple[dict[str, str], str]:
    prefix = f"READRAFT_BENCH_{role.upper()}_"
    api_key = os.getenv(prefix + "API_KEY", "").strip()
    if not api_key:
        raise ValueError(f"实时基准缺少 {prefix}API_KEY")
    provider = os.getenv(prefix + "PROVIDER", base.model_provider).strip()
    model = os.getenv(prefix + "MODEL", base.model_name).strip()
    base_url = os.getenv(prefix + "BASE_URL", base.model_base_url).strip()
    if not model:
        raise ValueError(f"实时基准缺少 {prefix}MODEL")
    return {"provider": provider, "model": model, "base_url": base_url}, api_key


async def generate_live_outputs(
    cases: Iterable[Mapping[str, Any]], modes: Iterable[str]
) -> list[BenchmarkOutput]:
    base = Settings.from_env()
    credentials = {
        role: _credential_from_env(role, base) for role in ("fast", "quality")
    }
    pipeline = ProseDraftPipeline()
    outputs: list[BenchmarkOutput] = []
    for case in cases:
        for mode in modes:
            route = route_model_task(mode, "prose")
            credential, api_key = credentials[route.model_role]
            selected = settings_for_credential(
                base, credential=credential, api_key=api_key
            )
            selected = settings_for_reasoning_policy(selected, route.reasoning_policy)
            model = ProviderAgentModel(selected)
            packet = dict(case["writing_packet"])
            packet["author_request"] = str(case["prompt"])
            packet["quality_mode"] = mode
            started = time.monotonic()
            try:
                result = await pipeline.generate(
                    model=model,
                    packet=packet,
                    provider_user_id=f"readraft-benchmark-{case['id']}-{mode}",
                )
            finally:
                await model.close()
            outputs.append(
                BenchmarkOutput(
                    case_id=str(case["id"]),
                    quality_mode=mode,
                    content=result.content,
                    latency_seconds=time.monotonic() - started,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    provider=result.provider,
                    model=result.model,
                )
            )
    return outputs


def build_report(
    cases: list[dict[str, Any]], outputs: list[BenchmarkOutput], *, seed: int
) -> dict[str, Any]:
    by_id = {str(case["id"]): case for case in cases}
    evaluations = [evaluate_output(by_id[item.case_id], item) for item in outputs]
    mode_summary = {}
    for mode in QUALITY_MODES:
        selected = [item for item in evaluations if item["quality_mode"] == mode]
        if selected:
            mode_summary[mode] = {
                "mean_score": round(mean(item["score"] for item in selected), 1),
                "hard_failures": sum(bool(item["hard_failures"]) for item in selected),
                "mean_latency_seconds": round(
                    mean(item["metrics"]["latency_seconds"] for item in selected), 3
                ),
                "input_tokens": sum(item["metrics"]["input_tokens"] for item in selected),
                "output_tokens": sum(item["metrics"]["output_tokens"] for item in selected),
            }
    try:
        version = importlib.metadata.version("readraft")
    except importlib.metadata.PackageNotFoundError:
        version = "working-tree"
    return {
        "schema_version": "1.0",
        "benchmark": "readraft-core-writing",
        "case_set_sha256": hashlib.sha256(CASE_PATH.read_bytes()).hexdigest(),
        "seed": seed,
        "readraft_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode_summary": mode_summary,
        "evaluations": evaluations,
    }


def write_report(report: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary_rows = []
    for mode, item in report["mode_summary"].items():
        summary_rows.append(
            f"| {mode} | {item['mean_score']} | {item['hard_failures']} | "
            f"{item['mean_latency_seconds']} | {item['input_tokens']} / {item['output_tokens']} |"
        )
    markdown = "\n".join(
        [
            "# Readraft 写作质量基准",
            "",
            f"用例集：`{report['case_set_sha256'][:12]}` · seed `{report['seed']}`",
            "",
            "| 模式 | 平均分 | 硬失败 | 平均延迟（秒） | 输入/输出 tokens |",
            "|---|---:|---:|---:|---:|",
            *summary_rows,
            "",
            "自动分只检查可机械验证的要求，不能替代盲评。",
        ]
    )
    (output_dir / "report.md").write_text(markdown + "\n", encoding="utf-8")
    table_rows = "".join(
        "<tr><td>{}</td><td>{:.1f}</td><td>{}</td><td>{:.3f}</td></tr>".format(
            html.escape(mode),
            float(item["mean_score"]),
            int(item["hard_failures"]),
            float(item["mean_latency_seconds"]),
        )
        for mode, item in report["mode_summary"].items()
    )
    document = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>Readraft 写作质量基准</title><style>
body{{font:16px/1.6 system-ui;max-width:960px;margin:40px auto;padding:0 20px;color:#263b33}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;border-bottom:1px solid #ccd5cf;text-align:left}}
</style><h1>Readraft 写作质量基准</h1><p>用例集 {html.escape(str(report['case_set_sha256'])[:12])}</p>
<table><thead><tr><th>模式</th><th>平均分</th><th>硬失败</th><th>平均延迟</th></tr></thead><tbody>{table_rows}</tbody></table>
<p>自动分只检查可机械验证的要求，不能替代盲评。</p></html>"""
    (output_dir / "report.html").write_text(document, encoding="utf-8")
    _write_blind_review(report, output_dir)


def _write_blind_review(report: Mapping[str, Any], output_dir: Path) -> None:
    rng = random.Random(int(report["seed"]))
    evaluations = list(report["evaluations"])
    rng.shuffle(evaluations)
    with (output_dir / "blind-review.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "readability_1_5",
                "character_agency_1_5",
                "continuity_1_5",
                "scene_movement_1_5",
                "specificity_1_5",
                "dialogue_1_5",
                "style_artifacts_1_5",
                "accept_or_edit_or_reject",
                "notes",
            ]
        )
        for index, item in enumerate(evaluations, start=1):
            sample_id = f"S{index:03d}"
            writer.writerow([sample_id, "", "", "", "", "", "", "", "", ""])
            (output_dir / f"{sample_id}.txt").write_text(
                str(item["content"]), encoding="utf-8"
            )
    key = [
        {
            "sample_id": f"S{index:03d}",
            "case_id": item["case_id"],
            "quality_mode": item["quality_mode"],
            "provider": item["provider"],
            "model": item["model"],
        }
        for index, item in enumerate(evaluations, start=1)
    ]
    (output_dir / "blind-key.json").write_text(
        json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 Readraft 写作质量基准")
    parser.add_argument("--cases", type=Path, default=CASE_PATH)
    parser.add_argument("--responses", type=Path)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results"))
    parser.add_argument("--modes", nargs="+", choices=QUALITY_MODES, default=list(QUALITY_MODES))
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    cases = load_cases(args.cases)
    if args.live and args.responses:
        parser.error("--live 与 --responses 不能同时使用")
    if args.live:
        outputs = asyncio.run(generate_live_outputs(cases, args.modes))
        response_dir = args.output / "responses"
        response_dir.mkdir(parents=True, exist_ok=True)
        for item in outputs:
            path = _response_path(response_dir, item.case_id, item.quality_mode)
            path.write_text(item.content, encoding="utf-8")
            path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "latency_seconds": item.latency_seconds,
                        "input_tokens": item.input_tokens,
                        "output_tokens": item.output_tokens,
                        "provider": item.provider,
                        "model": item.model,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
    else:
        outputs = load_saved_outputs(cases, args.modes, args.responses)
    report = build_report(cases, outputs, seed=args.seed)
    write_report(report, args.output)
    print(args.output / "report.html")


if __name__ == "__main__":
    main()
