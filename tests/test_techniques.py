from __future__ import annotations

import asyncio
import json
from pathlib import Path

from app.analysis_schema import ANALYSIS_JSON_EXAMPLE
from app.chapter_splitter import split_chapters
from app.config import Settings
from app.context_compiler import compile_active_techniques
from app.db import Database
from app.planning_ai import ProviderChapterPlanner
from app.planning_schema import (
    ChapterTaskCard,
    allocate_scene_requirement_refs,
)
from app.planning_service import PlanningService
from app.security import hash_password
from app.style_editor import ProviderStyleEditor
from app.technique_schema import TechniqueObservation
from app.technique_service import TechniqueService
from app.writing import build_writing_messages


def _observation() -> TechniqueObservation:
    return TechniqueObservation.model_validate(
        {
            "name": "先让线索产生后果，再解释来源",
            "dimension": "information",
            "source_location": "参考文本第三章的两个连续场景",
            "observation": "来源作品中的蓝色徽记先改变调查目标，后续场景才说明来历。",
            "effect": "读者先看到信息造成的现实后果，再带着可核对的问题继续阅读。",
            "suitable_for": ["线索首次登场", "跨场景维持问题"],
            "unsuitable_for": ["规则不解释就无法理解行动的场景"],
            "execution_rule": "先让关键线索改变人物目标或代价，至少一个节拍后再解释一项来源。",
            "originality_boundary": "不得复用蓝色徽记、来源人物、具体事件、揭示答案或参考原句。",
        }
    )


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        app_name="test",
        app_env="test",
        secret_key="test-secret-long-enough",
        data_dir=tmp_path,
        database_path=tmp_path / "model.db",
        cookie_secure=False,
        allow_registration=True,
        max_upload_bytes=1_000_000,
        max_text_chars=1_000_000,
        target_chapter_chars=10_000,
        max_chapter_chars=30_000,
        model_api_key="test-key",
        model_base_url="https://api.deepseek.com",
        model_name="deepseek-v4-flash",
        model_thinking=False,
        model_reasoning_effort="high",
        model_max_tokens=5_000,
        model_connect_timeout_seconds=1,
        model_read_timeout_seconds=1,
        model_max_retries=0,
        worker_poll_seconds=0.01,
    )


def _build_novel(tmp_path: Path):
    database = Database(tmp_path / "techniques.db")
    database.initialize()
    user_id = database.create_user(
        "technique-writer", hash_password("password-123")
    )
    project_id = "technique-project"
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="雾港来信",
        genre="悬疑",
        premise="记者追查一封不可能出现的信。",
        world_setting="当代海港小城。",
        style_guide="克制、具体。",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
    )
    planning = PlanningService(database)
    volume_id = planning.create_volume(
        user_id=user_id,
        project_id=project_id,
        title="第一卷",
        goal="确认来信来源。",
        start_state="尚未介入。",
        end_state="主动调查。",
        major_conflict="证据互相冲突。",
        payoff="确认信件并非伪造。",
    )
    chapter_id = "technique-chapter"
    chapter_dir = tmp_path / "chapter"
    chapter_dir.mkdir()
    content_path = chapter_dir / "content.txt"
    content_path.write_text("", encoding="utf-8")
    database.add_novel_chapter(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        title="第一章 来信",
        outline="林岚核对来信。",
        key_points="发现邮戳异常",
        content_path=content_path,
        volume_id=volume_id,
    )
    card = ChapterTaskCard.model_validate(
        {
            "purpose": "迫使林岚开始调查。",
            "start_state": "林岚认为来信是恶作剧。",
            "end_state": "林岚决定核对寄信地点。",
            "central_conflict": "来信证据与旧结论冲突。",
            "ending_hook": "邮戳日期晚于寄件人失踪日期。",
            "scenes": [
                {
                    "goal": "检查信封",
                    "obstacle": "没有寄件地址",
                    "action": "核对邮戳和封口",
                    "end_state": "发现日期异常",
                },
                {
                    "goal": "验证日期",
                    "obstacle": "旧记录缺失",
                    "action": "比对照片备份",
                    "end_state": "决定追查寄信地点",
                },
            ],
        }
    )
    card = allocate_scene_requirement_refs(card)
    planning.upsert_task_card(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        volume_id=volume_id,
        card=card,
        confirm=True,
    )
    return database, user_id, project_id, chapter_id, planning


def test_enabled_techniques_enter_only_selected_contexts(tmp_path: Path):
    database, user_id, project_id, chapter_id, planning = _build_novel(
        tmp_path
    )
    service = TechniqueService(database)
    technique_id = service.create_manual(
        user_id=user_id,
        observation=_observation(),
        author_note="只学习信息顺序。",
    )
    project_binding = service.bind(
        user_id=user_id,
        technique_id=technique_id,
        target=f"project:{project_id}",
        usage_modes=["plan", "write"],
        author_adaptation="改成潮湿信纸先影响判断，下一场景再解释来源。",
        priority=70,
    )
    task_card = planning.get_task_card(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
    )
    scene_id = str(task_card["scenes"][0]["id"])
    service.bind(
        user_id=user_id,
        technique_id=technique_id,
        target=f"scene:{scene_id}",
        usage_modes=["audit"],
        author_adaptation="审校时只检查是否过早解释线索来源。",
        priority=90,
    )

    context = database.get_writing_context(user_id, chapter_id)
    assert len(context["technique_cards"]) == 2
    planning_context = compile_active_techniques(
        context["technique_cards"], usage="plan"
    )
    writing_context = compile_active_techniques(
        context["technique_cards"], usage="write"
    )
    audit_context = compile_active_techniques(
        context["technique_cards"], usage="audit"
    )
    assert planning_context["included_count"] == 1
    assert writing_context["items"][0]["scope_type"] == "project"
    assert audit_context["items"][0]["scope_type"] == "scene"
    assert "场景 1" in audit_context["items"][0]["scope_label"]

    context["active_techniques"] = writing_context
    messages = build_writing_messages(
        context=context,
        operation="draft",
        instruction="",
        current_content="",
        previous_content="",
    )
    prompt = str(messages[1]["content"])
    assert "<active_writing_techniques>" in prompt
    assert "先让关键线索改变人物目标或代价" in prompt
    assert "改成潮湿信纸先影响判断" in prompt
    assert "不得复用蓝色徽记" in prompt
    assert "来源作品中的蓝色徽记" not in prompt
    assert "参考文本第三章" not in prompt

    assert service.set_binding_status(
        user_id=user_id,
        binding_id=project_binding,
        status="disabled",
    ) == technique_id
    context = database.get_writing_context(user_id, chapter_id)
    assert compile_active_techniques(
        context["technique_cards"], usage="write"
    )["included_count"] == 0
    assert service.update_card(
        user_id=user_id,
        technique_id=technique_id,
        observation=_observation(),
        author_note="归档测试",
        status="archived",
    )
    archived = service.get_card(
        user_id=user_id, technique_id=technique_id
    )
    assert archived["status"] == "archived"
    assert {item["status"] for item in archived["bindings"]} == {"disabled"}
    assert database.get_writing_context(user_id, chapter_id)[
        "technique_cards"
    ] == []


def test_analysis_observation_can_be_saved_once_with_source_trace(
    tmp_path: Path,
):
    database = Database(tmp_path / "analysis-techniques.db")
    database.initialize()
    user_id = database.create_user(
        "reference-reader", hash_password("password-123")
    )
    document_dir = tmp_path / "documents" / str(user_id) / ("a" * 32)
    chapter_dir = document_dir / "chapters"
    chapter_dir.mkdir(parents=True)
    text = "第一章 开始\n线索先改变人物行动，随后才解释来源。"
    source_path = document_dir / "source.txt"
    source_path.write_text(text, encoding="utf-8")
    chunks = split_chapters(text)
    paths = []
    for index, chunk in enumerate(chunks, start=1):
        path = chapter_dir / f"{index:05d}.txt"
        path.write_text(chunk.text, encoding="utf-8")
        paths.append(path)
    document_id = database.create_document(
        user_id=user_id,
        title="参考小说",
        original_filename="reference.txt",
        source_path=source_path,
        source_encoding="utf-8",
        text_length=len(text),
        chunks=chunks,
        chapter_paths=paths,
    )
    job_id = database.create_job(
        user_id=user_id,
        document_id=document_id,
        provider="mock",
        model="mock",
    )
    claimed = database.claim_next_analysis()
    payload = dict(ANALYSIS_JSON_EXAMPLE)
    assert database.complete_analysis(
        analysis_id=str(claimed["analysis_id"]),
        job_id=job_id,
        result=payload,
        raw_response="{}",
        input_tokens=0,
        output_tokens=0,
        claim_token=str(claimed["claim_token"]),
    )
    service = TechniqueService(database)
    first_id, created = service.create_from_analysis(
        user_id=user_id,
        analysis_id=str(claimed["analysis_id"]),
        technique_index=0,
    )
    assert created
    repeated_id, repeated_created = service.create_from_analysis(
        user_id=user_id,
        analysis_id=str(claimed["analysis_id"]),
        technique_index=0,
    )
    assert repeated_id == first_id
    assert not repeated_created
    card = service.get_card(user_id=user_id, technique_id=first_id)
    assert card["source_document_title"] == "参考小说"
    assert card["source_analysis_id"] == claimed["analysis_id"]
    assert card["execution_rule"].startswith("让关键物件先触发")


def test_planner_and_style_auditor_receive_safe_abstract_techniques(
    tmp_path: Path,
):
    active = {
        "source": "author_enabled_abstract_techniques",
        "usage": "plan",
        "items": [
            {
                "id": "card-safe",
                "name": "延迟解释线索来源",
                "dimension": "information",
                "execution_rule": "先让线索改变人物目标，下一场景再解释一项来源。",
                "effect": "让读者带着可核对的问题继续阅读。",
                "originality_boundary": "不得复用参考作品的蓝色徽记或具体情节。",
                "author_adaptation": "本书改用异常邮戳。",
                "scope_type": "project",
                "scope_label": "全书",
                "priority": 80,
            }
        ],
        "included_count": 1,
        "truncated": False,
    }
    context = {
        "chapter": {
            "project_title": "雾港来信",
            "genre": "悬疑",
            "premise": "记者追查异常来信。",
            "story_promise": "每章一项新证据。",
            "target_audience": "悬疑读者",
            "core_appeal": "证据推理",
            "ending_constraint": "解释来信来源",
            "world_setting": "当代海港",
            "style_guide": "克制具体",
            "point_of_view": "第三人称限知",
            "position": 1,
            "title": "第一章",
            "outline": "检查来信",
            "key_points": "核对邮戳",
            "target_chapter_chars": 3000,
        },
        "characters": [],
        "canonical_memory": {},
        "active_techniques": active,
    }
    planner_result = ChapterTaskCard.model_validate(
        {
            "purpose": "推动调查",
            "start_state": "尚未相信来信",
            "end_state": "决定追查",
            "central_conflict": "证据互相冲突",
            "ending_hook": "出现新的邮戳",
            "scenes": [
                {
                    "goal": "检查信封",
                    "obstacle": "地址缺失",
                    "action": "核对邮戳",
                    "end_state": "发现异常",
                },
                {
                    "goal": "验证异常",
                    "obstacle": "记录缺失",
                    "action": "比对备份",
                    "end_state": "决定追查",
                },
            ],
        }
    )
    captured: dict[str, dict] = {}

    async def scenario():
        planner = ProviderChapterPlanner(_settings(tmp_path))

        async def planner_post(payload):
            captured["planner"] = dict(payload)
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": planner_result.model_dump_json()
                        },
                    }
                ],
                "usage": {},
            }

        planner._analyzer._post = planner_post
        style = ProviderStyleEditor(_settings(tmp_path))

        async def style_post(payload):
            captured["style"] = dict(payload)
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "summary": "没有明显问题。",
                                    "issues": [],
                                },
                                ensure_ascii=False,
                            )
                        },
                    }
                ],
                "usage": {},
            }

        style._analyzer._post = style_post
        try:
            await planner.propose(
                context=context,
                instruction="",
                provider_user_id="u_test",
            )
            await style.audit(
                context={
                    **context,
                    "active_techniques": {**active, "usage": "audit"},
                },
                chapter_text="她把信封移到窗边，重新核对邮戳。",
                voice_profile={"narration_rules": "紧贴主角视角"},
                preferences=[],
                provider_user_id="u_test",
            )
        finally:
            await planner.close()
            await style.close()

    asyncio.run(scenario())
    planner_prompt = str(captured["planner"]["messages"][1]["content"])
    style_prompt = str(captured["style"]["messages"][1]["content"])
    for prompt in (planner_prompt, style_prompt):
        assert "先让线索改变人物目标" in prompt
        assert "本书改用异常邮戳" in prompt
        assert "不得复用参考作品的蓝色徽记" in prompt
        assert "来源作品中的蓝色徽记" not in prompt
