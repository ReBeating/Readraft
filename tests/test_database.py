from pathlib import Path

import pytest

from app.chapter_splitter import split_chapters
from app.analysis_repository import AnalysisRepository
from app.reference_analysis_schema import ALL_LAYERS
from app.db import Database
from app.security import hash_password


def test_database_job_lifecycle(tmp_path: Path):
    database = Database(tmp_path / "app.db")
    database.initialize()
    analyses = AnalysisRepository(database)
    user_id = database.create_user("tester", hash_password("password-123"))

    document_dir = tmp_path / "documents" / str(user_id) / ("a" * 32)
    chapter_dir = document_dir / "chapters"
    chapter_dir.mkdir(parents=True)
    source_path = document_dir / "source.txt"
    text = "第一章 开始\n正文一。\n第二章 继续\n正文二。"
    source_path.write_text(text, encoding="utf-8")
    chunks = split_chapters(text)
    chapter_paths = []
    for index, chunk in enumerate(chunks, 1):
        path = chapter_dir / f"{index:05d}.txt"
        path.write_text(chunk.text, encoding="utf-8")
        chapter_paths.append(path)

    document_id = database.create_document(
        user_id=user_id,
        title="测试小说",
        original_filename="test.txt",
        source_path=source_path,
        source_encoding="utf-8",
        text_length=len(text),
        chunks=chunks,
        chapter_paths=chapter_paths,
    )
    job_id = analyses.create_job(
        user_id=user_id,
        document_id=document_id,
        provider="mock",
        model="mock",
    )
    with pytest.raises(ValueError, match="任务正在运行"):
        database.upsert_api_credential(
            user_id=user_id,
            encrypted_key="encrypted-test-key",
            key_hint="sk-••••test",
            model="deepseek-v4-flash",
        )

    first = analyses.claim_next()
    assert first is not None
    assert analyses.start_layer(
        analysis_id=first["analysis_id"],
        layer="structure",
        claim_token=first["claim_token"],
    )
    assert not analyses.complete_layer(
        analysis_id=first["analysis_id"],
        content_hash=first["content_hash"],
        layer="structure",
        provider="local",
        model="deterministic-v1",
        result={"char_count": 10},
        raw_response="{}",
        input_tokens=999,
        output_tokens=999,
        claim_token="stale-token",
    )
    assert set(analyses.layers(first["analysis_id"])) == set(ALL_LAYERS)
    for index, layer in enumerate(ALL_LAYERS):
        if layer != "structure":
            assert analyses.start_layer(
                analysis_id=first["analysis_id"],
                layer=layer,
                claim_token=first["claim_token"],
            )
        assert analyses.complete_layer(
            analysis_id=first["analysis_id"],
            content_hash=first["content_hash"],
            layer=layer,
            provider="local" if layer == "structure" else "mock",
            model="deterministic-v2" if layer == "structure" else "mock",
            result={"layer": layer},
            raw_response="{}",
            input_tokens=10 if index == 1 else 0,
            output_tokens=5 if index == 1 else 0,
            claim_token=first["claim_token"],
        )
    assert analyses.complete(
        analysis_id=first["analysis_id"],
        job_id=job_id,
        result={"chapter_title": first["chapter_title"]},
        claim_token=first["claim_token"],
    )
    second = analyses.claim_next()
    assert second is not None
    assert analyses.start_layer(
        analysis_id=second["analysis_id"],
        layer="structure",
        claim_token=second["claim_token"],
    )
    analyses.fail_layer(
        analysis_id=second["analysis_id"],
        layer="structure",
        error="test failure",
        input_tokens=7,
        output_tokens=3,
    )
    analyses.fail(
        analysis_id=second["analysis_id"],
        job_id=job_id,
        error="test failure",
        claim_token=second["claim_token"],
    )

    job = analyses.get_job(user_id, job_id)
    assert job is not None
    assert job["status"] == "partial"
    assert job["completed_chapters"] == 1
    assert job["failed_chapters"] == 1
    assert job["input_tokens"] == 17
    assert job["output_tokens"] == 8
    assert analyses.retry_failed(user_id, job_id)
