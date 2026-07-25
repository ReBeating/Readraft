import asyncio
import json
from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.db import Database
from app.quality_audit import (
    DeepSeekQualityAuditor,
    effective_char_count,
    finalize_hard_audit,
)
from app.quality_schema import HardAuditAnalysis
from app.security import hash_password


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        app_name="test",
        app_env="test",
        secret_key="test-secret-long-enough",
        data_dir=tmp_path,
        database_path=tmp_path / "test.db",
        cookie_secure=False,
        allow_registration=True,
        max_upload_bytes=1_000_000,
        max_text_chars=1_000_000,
        target_chapter_chars=10_000,
        max_chapter_chars=30_000,
        deepseek_api_key="test-key",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-v4-flash",
        deepseek_thinking=False,
        deepseek_reasoning_effort="high",
        deepseek_max_tokens=5_000,
        deepseek_connect_timeout_seconds=1,
        deepseek_read_timeout_seconds=1,
        deepseek_max_retries=0,
        worker_poll_seconds=0.01,
    )


def test_effective_char_count_ignores_all_whitespace():
    assert effective_char_count("林 岚\n拆\t信\r\n") == 4


def test_local_gate_blocks_short_text_and_missing_task_coverage():
    analysis = HardAuditAnalysis.model_validate(
        {
            "summary": "一项必须事件无法在正文中确认。",
            "findings": [],
            "must_happen_coverage": [
                {
                    "requirement": "林岚核对邮戳",
                    "status": "missing",
                    "evidence": "正文没有核对邮戳的行动。",
                }
            ],
            "scene_coverage": [],
        }
    )
    report = finalize_hard_audit(
        analysis=analysis,
        chapter_text="林岚拆开信。",
        expansion_attempted=True,
    )
    assert report.verdict == "block"
    assert report.effective_char_count == 6
    assert report.hard_issue_count == 2
    assert {
        finding.category for finding in report.findings
    } == {"length", "must_happen"}


def test_deepseek_hard_auditor_uses_strict_json_output(tmp_path: Path):
    seen = {}
    raw_result = {
        "summary": "任务卡与正史约束均已核对。",
        "findings": [],
        "must_happen_coverage": [
            {
                "requirement": "林岚核对邮戳",
                "status": "met",
                "evidence": "她把邮戳日期抄进笔记。",
            }
        ],
        "scene_coverage": [
            {
                "requirement": "场景 1：核对来信",
                "status": "met",
                "evidence": "她逐项核对信封与旧信。",
            }
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                raw_result, ensure_ascii=False
                            )
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 320,
                    "completion_tokens": 110,
                },
            },
        )

    async def scenario():
        auditor = DeepSeekQualityAuditor(
            _settings(tmp_path),
            transport=httpx.MockTransport(handler),
        )
        try:
            return await auditor.audit(
                context={
                    "chapter": {
                        "project_title": "雾港来信",
                        "genre": "悬疑",
                        "premise": "记者返回故乡调查父亲失踪案。",
                        "world_setting": "当代海港。",
                        "point_of_view": "第三人称限知",
                        "position": 1,
                        "title": "第一章",
                    },
                    "characters": [{"name": "林岚"}],
                    "canonical_memory": {},
                    "task_card": {
                        "must_happen": ["林岚核对邮戳"],
                        "scenes": [{"goal": "核对来信"}],
                    },
                },
                chapter_text="她把邮戳日期抄进笔记。",
                provider_user_id="u_quality",
            )
        finally:
            await auditor.close()

    response = asyncio.run(scenario())
    assert response.result.must_happen_coverage[0].status == "met"
    assert response.input_tokens == 320
    assert seen["payload"]["response_format"] == {"type": "json_object"}
    assert seen["payload"]["temperature"] == 0.2


def test_deepseek_scene_auditor_checks_only_focused_scene(tmp_path: Path):
    seen = {}
    raw_result = {
        "summary": "当前场景落实了目标、阻力和结束状态。",
        "findings": [],
        "must_happen_coverage": [],
        "scene_coverage": [
            {
                "requirement": "场景 1：确认来信",
                "status": "met",
                "evidence": "她致电邮局核对投递记录。",
            }
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                raw_result, ensure_ascii=False
                            )
                        },
                    }
                ],
                "usage": {"prompt_tokens": 180, "completion_tokens": 60},
            },
        )

    async def scenario():
        auditor = DeepSeekQualityAuditor(
            _settings(tmp_path),
            transport=httpx.MockTransport(handler),
        )
        try:
            return await auditor.audit(
                context={
                    "audit_scope": "scene",
                    "chapter": {
                        "project_title": "雾港来信",
                        "genre": "悬疑",
                        "premise": "调查异常来信。",
                        "world_setting": "当代海港。",
                        "point_of_view": "第三人称限知",
                        "position": 1,
                        "title": "第一章",
                    },
                    "characters": [{"name": "林岚"}],
                    "canonical_memory": {},
                    "focused_scene": {
                        "position": 1,
                        "goal": "确认来信",
                        "obstacle": "邮戳日期异常",
                        "action": "致电邮局",
                        "end_state": "确认投递记录存在",
                    },
                    "next_scene": {
                        "position": 2,
                        "goal": "购买车票",
                    },
                    "task_card": {
                        "must_happen": [],
                        "forbidden": ["揭晓寄信人"],
                        "scenes": [{"goal": "确认来信"}],
                    },
                },
                chapter_text="她致电邮局核对投递记录。",
                provider_user_id="u_scene_quality",
            )
        finally:
            await auditor.close()

    response = asyncio.run(scenario())
    assert response.result.must_happen_coverage == []
    system_prompt = seen["payload"]["messages"][0]["content"]
    user_prompt = seen["payload"]["messages"][1]["content"]
    assert "Scene Auditor" in system_prompt
    assert "must_happen_coverage 必须为空" in system_prompt
    assert "candidate_scene_text" in user_prompt
    assert "购买车票" in user_prompt


def test_manual_candidate_requires_audit_or_explicit_author_override(
    tmp_path: Path,
):
    database = Database(tmp_path / "quality.db")
    database.initialize()
    user_id = database.create_user(
        "quality-author", hash_password("password-123")
    )
    project_id = "p" * 32
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="雾港来信",
        genre="悬疑",
        premise="记者返回故乡调查父亲失踪案。",
        world_setting="当代海港。",
        style_guide="克制。",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
    )
    chapter_id = "c" * 32
    chapter_dir = tmp_path / "chapters" / chapter_id
    chapter_dir.mkdir(parents=True)
    content_path = chapter_dir / "content.txt"
    content_path.write_text("", encoding="utf-8")
    database.add_novel_chapter(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        title="第一章",
        outline="林岚收到来信。",
        key_points="核对邮戳",
        content_path=content_path,
    )
    candidate_path = chapter_dir / "manual.txt"
    candidate_path.write_text("太短的候选稿。", encoding="utf-8")
    version_id = database.record_manual_chapter_version(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        version_path=candidate_path,
        char_count=7,
        effective_char_count=7,
    )
    version = database.get_chapter_version(
        user_id, project_id, chapter_id, str(version_id)
    )
    assert version["quality_status"] == "block"
    with pytest.raises(ValueError, match="硬审计尚未通过"):
        database.accept_chapter_version(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            version_id=str(version_id),
        )

    accepted = database.accept_chapter_version(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        version_id=str(version_id),
        override_reason="作者有意使用极短章节制造突然中断效果",
    )
    assert accepted and accepted["changed"]
    version = database.get_chapter_version(
        user_id, project_id, chapter_id, str(version_id)
    )
    assert version["quality_status"] == "overridden"
    assert "极短章节" in version["quality_override_reason"]
