from __future__ import annotations

from pathlib import Path

import pytest

from app.analysis_repository import AnalysisRepository
from app.chapter_splitter import split_chapters
from app.db import Database
from app.reference_analysis_aggregation import aggregate_document
from app.reference_analysis_metrics import deterministic_structure
from app.reference_analysis_schema import (
    StyleLayer,
    StyleObservation,
    combine_layers,
    validate_evidence,
)
from app.security import hash_password


def _chapter_result(position: int, text: str):
    quote = "门外的脚步忽然停了"
    start = text.index(quote)
    evidence = {"start": start, "end": start + len(quote), "quote": quote}
    style = StyleLayer(
        observations=[
            StyleObservation(
                axis="information_flow",
                value="先呈现后果，再补充解释",
                analysis="先给出听觉变化，解释被留到后续动作。",
                execution_rule="先呈现未知事物造成的后果，再补充一项来源信息。",
                originality_boundary="不得复用原作人名、物件、事件或措辞。",
                evidence=[evidence],
            )
        ]
    ).model_dump(mode="json")
    layers = {
        "structure": deterministic_structure(text),
        "facts": {
            "summary": "人物听见门外脚步停止，并继续判断来者的动向。",
            "characters": [],
            "scenes": [],
            "events": [],
            "foreshadowing": [],
        },
        "narrative": {
            "scene_functions": [],
            "conflicts": [],
            "relationship_changes": [],
            "information_release": [],
            "pacing": [],
            "ending_hook": None,
        },
        "style": style,
        "techniques": {"techniques": []},
    }
    return combine_layers(
        chapter_title=f"第{position}章",
        chapter_position=position,
        text=text,
        layers=layers,
    )


def test_style_analysis_is_evidence_backed_and_aggregated_without_quotes():
    first = _chapter_result(1, "门外的脚步忽然停了。她没有立刻开门。")
    third = _chapter_result(3, "门外的脚步忽然停了。灯影随后越过窗沿。")

    aggregate = aggregate_document([first, third])

    profile = aggregate["style_profile"]
    assert profile["analyzed_chapters"] == 2
    assert profile["traits"][0]["axis"] == "information_flow"
    assert profile["traits"][0]["chapters"] == [1, 3]
    assert profile["traits"][0]["coverage"] == 1.0
    assert "门外的脚步" not in str(profile)
    assert profile["quantitative"]["average_sentence_chars"] > 0


def test_evidence_offsets_must_match_frozen_source_exactly():
    source = "雨落在窗上。"
    invalid = {
        "observations": [
            {
                "evidence": [
                    {"start": 0, "end": 1, "quote": "雪"},
                ]
            }
        ]
    }
    with pytest.raises(ValueError, match="证据与冻结正文不一致"):
        validate_evidence(invalid, source)


def test_structure_metrics_capture_rhythm_without_model_inference():
    metrics = deterministic_structure("短句。\n\n这是一个明显更长的句子。\n\n“收到。”")
    assert metrics["paragraph_count"] == 3
    assert metrics["sentence_count"] == 3
    assert metrics["sentence_length_stddev"] > 0
    assert metrics["dialogue_ratio"] > 0


def test_migration_adds_style_layer_only_to_resumable_analysis(tmp_path: Path):
    database = Database(tmp_path / "analysis.db")
    database.initialize()
    user_id = database.create_user("style-migration", hash_password("password-123"))
    text = "第一章 雨夜\n门外的脚步忽然停了。"
    document_dir = tmp_path / "documents"
    chapter_dir = document_dir / "chapters"
    chapter_dir.mkdir(parents=True)
    source_path = document_dir / "source.txt"
    source_path.write_text(text, encoding="utf-8")
    chunks = split_chapters(text)
    chapter_paths = []
    for index, chunk in enumerate(chunks, start=1):
        path = chapter_dir / f"{index:05d}.txt"
        path.write_text(chunk.text, encoding="utf-8")
        chapter_paths.append(path)
    document_id = database.create_document(
        user_id=user_id,
        title="迁移测试",
        original_filename="source.txt",
        source_path=source_path,
        source_encoding="utf-8",
        text_length=len(text),
        chunks=chunks,
        chapter_paths=chapter_paths,
    )
    repository = AnalysisRepository(database)
    job_id = repository.create_job(
        user_id=user_id,
        document_id=document_id,
        provider="mock",
        model="mock",
    )
    with database.connection() as connection:
        analysis_id = str(
            connection.execute(
                "SELECT id FROM chapter_analyses WHERE job_id=?", (job_id,)
            ).fetchone()["id"]
        )
        connection.execute(
            "DELETE FROM chapter_analysis_layers WHERE analysis_id=? AND layer='style'",
            (analysis_id,),
        )
        connection.execute(
            "UPDATE chapter_analyses SET schema_version='2.0' WHERE id=?",
            (analysis_id,),
        )
        connection.execute(
            "UPDATE analysis_jobs SET schema_version='2.0' WHERE id=?",
            (job_id,),
        )
        connection.execute("DELETE FROM schema_migrations WHERE version=58")
        connection.commit()

    database.initialize()

    with database.connection() as connection:
        layer = connection.execute(
            """
            SELECT status FROM chapter_analysis_layers
            WHERE analysis_id=? AND layer='style'
            """,
            (analysis_id,),
        ).fetchone()
        analysis_schema = connection.execute(
            "SELECT schema_version FROM chapter_analyses WHERE id=?",
            (analysis_id,),
        ).fetchone()["schema_version"]
        job_schema = connection.execute(
            "SELECT schema_version FROM analysis_jobs WHERE id=?", (job_id,)
        ).fetchone()["schema_version"]
    assert layer and layer["status"] == "queued"
    assert analysis_schema == "3.0"
    assert job_schema == "3.0"
