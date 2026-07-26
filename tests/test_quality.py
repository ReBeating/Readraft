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
                "scene_id": "scene-1",
                "position": 1,
                "goal": "核对来信",
                "checks": [
                    {
                        "aspect": "goal",
                        "requirement": "核对来信",
                        "status": "met",
                        "evidence": "她逐项核对信封与旧信。",
                    }
                ],
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
                        "scenes": [
                            {
                                "id": "scene-1",
                                "position": 1,
                                "goal": "核对来信",
                            }
                        ],
                    },
                },
                chapter_text="她把邮戳日期抄进笔记。",
                provider_user_id="u_quality",
            )
        finally:
            await auditor.close()

    response = asyncio.run(scenario())
    assert response.result.must_happen_coverage[0].status == "met"
    assert response.result.scene_coverage[0].scene_id == "scene-1"
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
                "scene_id": "focused-scene",
                "position": 1,
                "goal": "确认来信",
                "checks": [
                    {
                        "aspect": "goal",
                        "requirement": "确认来信",
                        "status": "met",
                        "evidence": "她开始核对来信。",
                    },
                    {
                        "aspect": "obstacle",
                        "requirement": "邮戳日期异常",
                        "status": "met",
                        "evidence": "她发现邮戳日期异常。",
                    },
                    {
                        "aspect": "action",
                        "requirement": "致电邮局",
                        "status": "met",
                        "evidence": "她致电邮局。",
                    },
                    {
                        "aspect": "end_state",
                        "requirement": "确认投递记录存在",
                        "status": "met",
                        "evidence": "邮局确认了投递记录。",
                    },
                ],
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
                        "id": "focused-scene",
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
                        "scenes": [
                            {
                                "id": "focused-scene",
                                "position": 1,
                                "goal": "确认来信",
                                "obstacle": "邮戳日期异常",
                                "action": "致电邮局",
                                "end_state": "确认投递记录存在",
                            }
                        ],
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
    assert "scene_coverage_contract" in user_prompt
    assert "购买车票" in user_prompt


def test_scene_beat_gap_is_advisory_in_chapter_gate():
    analysis = HardAuditAnalysis.model_validate(
        {
            "summary": "场景动作与任务卡存在偏差。",
            "findings": [
                {
                    "code": "scene_action_changed",
                    "category": "scene_beat",
                    "severity": "hard",
                    "location": "场景 1",
                    "evidence": "主角改为发短信。",
                    "description": "没有执行原计划的电话动作。",
                    "violated_constraint": "致电邮局",
                    "repair_instruction": "按需改回电话动作。",
                }
            ],
            "must_happen_coverage": [],
            "scene_coverage": [
                {
                    "scene_id": "scene-1",
                    "position": 1,
                    "goal": "核对来信",
                    "checks": [
                        {
                            "aspect": "action",
                            "requirement": "致电邮局",
                            "status": "missing",
                            "evidence": "正文改为发送短信。",
                        }
                    ],
                }
            ],
        }
    )
    report = finalize_hard_audit(
        analysis=analysis,
        chapter_text="正文" * 1000,
        expansion_attempted=False,
    )
    assert report.verdict == "pass"
    assert report.hard_issue_count == 0
    assert report.warning_count == 2
    assert all(
        finding.severity == "warning" for finding in report.findings
    )


def test_scene_coverage_has_no_fixed_ten_scene_limit():
    scenes = [
        {
            "scene_id": f"scene-{position}",
            "position": position,
            "goal": f"完成场景 {position}",
            "checks": [
                {
                    "aspect": "goal",
                    "requirement": f"完成场景 {position}",
                    "status": "met",
                    "evidence": "正文已经落实。",
                }
            ],
        }
        for position in range(1, 13)
    ]
    analysis = HardAuditAnalysis.model_validate(
        {
            "summary": "十二个场景均已检查。",
            "findings": [],
            "must_happen_coverage": [],
            "scene_coverage": scenes,
        }
    )
    assert len(analysis.scene_coverage) == 12


def test_scene_coverage_is_normalized_to_task_card_order(tmp_path: Path):
    raw_result = {
        "summary": "两个场景均已检查。",
        "findings": [],
        "must_happen_coverage": [],
        "scene_coverage": [
            {
                "scene_id": "scene-2",
                "position": 99,
                "goal": "模型改写的标题",
                "checks": [
                    {
                        "aspect": "action",
                        "requirement": "推门",
                        "status": "met",
                        "evidence": "她推开门。",
                    },
                    {
                        "aspect": "goal",
                        "requirement": "进入房间",
                        "status": "met",
                        "evidence": "她进入房间。",
                    },
                ],
            },
            {
                "scene_id": "scene-1",
                "position": 88,
                "goal": "模型改写的标题",
                "checks": [
                    {
                        "aspect": "goal",
                        "requirement": "找到钥匙",
                        "status": "met",
                        "evidence": "她找到了钥匙。",
                    }
                ],
            },
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
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
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
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
                        "project_title": "测试",
                        "position": 1,
                        "title": "第一章",
                    },
                    "task_card": {
                        "must_happen": [],
                        "scenes": [
                            {
                                "id": "scene-1",
                                "position": 1,
                                "goal": "找到钥匙",
                            },
                            {
                                "id": "scene-2",
                                "position": 2,
                                "goal": "进入房间",
                                "action": "推门",
                            },
                        ],
                    },
                },
                chapter_text="她找到钥匙并推门进入房间。",
                provider_user_id="u_scene_order",
            )
        finally:
            await auditor.close()

    response = asyncio.run(scenario())
    assert [
        item.scene_id for item in response.result.scene_coverage
    ] == ["scene-1", "scene-2"]
    assert response.result.scene_coverage[1].position == 2
    assert response.result.scene_coverage[1].goal == "进入房间"
    assert [
        check.aspect
        for check in response.result.scene_coverage[1].checks
    ] == ["goal", "action"]


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
