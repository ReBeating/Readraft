import asyncio
import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.deepseek import AnalyzerError
from app.writing import DeepSeekWriter, build_writing_messages


def make_settings(tmp_path: Path) -> Settings:
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


def writing_context() -> dict:
    return {
        "chapter": {
            "project_title": "雾港来信",
            "genre": "悬疑",
            "premise": "记者返回故乡调查父亲失踪案。",
            "world_setting": "当代海港。",
            "style_guide": "克制冷峻。",
            "point_of_view": "第三人称限知",
            "target_chapter_chars": 3000,
            "position": 1,
            "title": "第一章",
            "outline": "主角收到一封迟到的信。",
            "key_points": "信上有新的邮戳",
        },
        "characters": [{"name": "林岚", "role": "主角"}],
    }


def test_deepseek_writer_sends_plain_text_request(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "林岚拆开信封，潮湿的纸边划过指尖。"},
                    }
                ],
                "usage": {"prompt_tokens": 200, "completion_tokens": 30},
            },
        )

    async def scenario():
        writer = DeepSeekWriter(
            make_settings(tmp_path), transport=httpx.MockTransport(handler)
        )
        try:
            return await writer.write(
                context=writing_context(),
                operation="draft",
                instruction="从拆信开始",
                current_content="",
                previous_content="",
                provider_user_id="u_test",
            )
        finally:
            await writer.close()

    result = asyncio.run(scenario())
    assert result.content.startswith("林岚")
    assert result.input_tokens == 200
    assert seen["payload"]["temperature"] == 0.85
    assert seen["payload"]["user_id"] == "u_test"
    assert "response_format" not in seen["payload"]
    assert seen["payload"]["messages"][0]["role"] == "system"


def test_writing_prompt_only_includes_confirmed_archive_settings():
    context = writing_context()
    context["confirmed_archive_rules"] = [
        {
            "category": "world",
            "title": "潮汐限制",
            "content": "退潮后旧港入口只开放二十分钟。",
            "status": "confirmed",
        }
    ]
    messages = build_writing_messages(
        context=context,
        operation="draft",
        instruction="",
        current_content="",
        previous_content="",
    )
    prompt = messages[1]["content"]
    assert "<confirmed_work_archive_settings>" in prompt
    assert "退潮后旧港入口只开放二十分钟" in prompt


def test_deepseek_writer_reports_content_filter_with_usage(tmp_path):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "content_filter",
                        "message": {"content": None},
                    }
                ],
                "usage": {"prompt_tokens": 42, "completion_tokens": 3},
            },
        )

    async def scenario():
        writer = DeepSeekWriter(
            make_settings(tmp_path), transport=httpx.MockTransport(handler)
        )
        try:
            await writer.write(
                context=writing_context(),
                operation="draft",
                instruction="",
                current_content="",
                previous_content="",
                provider_user_id="u_test",
            )
        finally:
            await writer.close()

    with pytest.raises(AnalyzerError) as caught:
        asyncio.run(scenario())
    assert "内容安全策略" in str(caught.value)
    assert caught.value.input_tokens == 42
    assert caught.value.output_tokens == 3


def test_openai_compatible_writer_uses_active_provider(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "她把船票收进口袋。"},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 8},
            },
        )

    async def scenario():
        settings = replace(
            make_settings(tmp_path),
            model_provider="openai",
            deepseek_base_url="https://api.openai.com/v1",
            deepseek_model="gpt-4.1-mini",
        )
        writer = DeepSeekWriter(
            settings, transport=httpx.MockTransport(handler)
        )
        try:
            response = await writer.write(
                context=writing_context(),
                operation="draft",
                instruction="",
                current_content="",
                previous_content="",
                provider_user_id="u_test",
            )
            return writer, response
        finally:
            await writer.close()

    writer, result = asyncio.run(scenario())
    assert result.content
    assert writer.provider == "openai"
    assert seen["payload"]["max_completion_tokens"] == 5_000
    assert seen["payload"]["user"] == "u_test"
    assert "thinking" not in seen["payload"]


def test_deepseek_writer_retries_resource_exhaustion_once(tmp_path):
    attempts = {"count": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "insufficient_system_resource",
                        "message": {"content": None},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 0},
            },
        )

    async def no_sleep(_seconds: float) -> None:
        return None

    async def scenario():
        writer = DeepSeekWriter(
            make_settings(tmp_path),
            transport=httpx.MockTransport(handler),
            sleep=no_sleep,
        )
        try:
            await writer.write(
                context=writing_context(),
                operation="draft",
                instruction="",
                current_content="",
                previous_content="",
                provider_user_id="u_test",
            )
        finally:
            await writer.close()

    with pytest.raises(AnalyzerError) as caught:
        asyncio.run(scenario())
    assert "系统资源不足" in str(caught.value)
    assert caught.value.input_tokens == 20
    assert attempts["count"] == 2


def test_scene_writer_prompt_is_bounded_to_one_scene():
    context = writing_context()
    context.update(
        {
            "task_card": {
                "purpose": "迫使林岚返回雾港",
                "must_preserve": ["林岚不知道寄信人身份"],
                "forbidden": ["揭晓父亲下落"],
            },
            "focused_scene": {
                "id": "scene-1",
                "position": 1,
                "goal": "确认来信真假",
                "obstacle": "邮戳日期不可能",
                "action": "对照旧信并致电邮局",
                "end_state": "承认信件值得调查",
                "transition": "查询当夜车票",
            },
            "scene_sequence": [
                {"id": "scene-1", "position": 1, "goal": "确认来信真假"},
                {"id": "scene-2", "position": 2, "goal": "购买车票"},
            ],
            "next_scene": {"position": 2, "goal": "购买车票"},
            "scene_target_chars": 1400,
            "scene_minimum_chars": 600,
            "previous_chapter_content": "上一章正史末尾。",
        }
    )
    messages = build_writing_messages(
        context=context,
        operation="rewrite_scene",
        instruction="不要提前解释邮戳来源",
        current_content="当前场景旧稿。",
        previous_content="上一场景以电话挂断结束。",
    )
    prompt = str(messages[1]["content"])
    assert "只创作当前指定场景" not in prompt
    assert "重写当前指定场景的完整正文" in prompt
    assert "只输出当前场景正文" in prompt
    assert "不要代写下一个场景" in prompt
    assert "当前场景旧稿" in prompt
    assert "上一场景以电话挂断结束" in prompt
    assert "购买车票" in prompt


def test_writer_system_prompt_combines_base_and_book_instructions():
    context = writing_context()
    context["chapter"]["ai_instructions"] = "本书始终让人物先行动，再解释。"
    messages = build_writing_messages(
        context=context,
        operation="draft",
        instruction="",
        current_content="",
        previous_content="",
    )
    system_prompt = str(messages[0]["content"])
    assert "只输出小说正文" in system_prompt
    assert "本书始终让人物先行动，再解释" in system_prompt
