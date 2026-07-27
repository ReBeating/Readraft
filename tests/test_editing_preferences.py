import asyncio
import hashlib
import json
import re
import time
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.context_compiler import (
    build_scene_context_snapshot,
    build_writing_context_snapshot,
)
from app.credentials import CredentialCipher, key_hint
from app.db import Database
from app.deepseek import MockAnalyzer
from app.main import create_app
from app.preference_extraction import (
    BaseEditPreferenceExtractor,
    DeepSeekEditPreferenceExtractor,
    EditPreferenceExtractionResponse,
    MockEditPreferenceExtractor,
    build_edit_sample,
)
from app.preference_schema import EditPreferenceSuggestion
from app.preference_service import PreferenceService
from app.security import hash_password
from app.style_service import StyleService
from app.worker import AnalysisWorker
from app.writing import MockWriter, build_writing_messages


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        app_name="novelAI 测试",
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
        deepseek_api_key=None,
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


def _create_project(
    database: Database, *, user_id: int, project_id: str
) -> None:
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="雾港来信",
        genre="悬疑",
        premise="林岚收到父亲署名的新信。",
        world_setting="当代海港。",
        style_guide="克制、具体。",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
    )


def _manual_edit_pair(
    database: Database,
    tmp_path: Path,
    *,
    user_id: int,
    project_id: str,
    chapter_id: str = "chapter-one",
) -> tuple[str, str, str, str]:
    chapter_dir = tmp_path / project_id / chapter_id
    (chapter_dir / "versions").mkdir(parents=True)
    content_path = chapter_dir / "content.txt"
    content_path.write_text("", encoding="utf-8")
    database.add_novel_chapter(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        title="第一章",
        outline="林岚拆开信。",
        key_points="北塔地址",
        content_path=content_path,
    )
    before = (
        "她感到十分难过，内心深处充满了复杂的情绪。"
        "她知道这封信意味着命运再次向她发出了召唤。"
    )
    after = (
        "她把湿信折了两次，塞回外套。窗框响了一声，"
        "她只把北塔的地址抄进本子。"
    )
    before_path = chapter_dir / "versions" / "manual-before.txt"
    before_path.write_text(before, encoding="utf-8")
    before_id = database.record_manual_chapter_version(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        version_path=before_path,
        char_count=len(before),
        effective_char_count=len(before),
        content_hash=hashlib.sha256(before.encode()).hexdigest(),
    )
    after_path = chapter_dir / "versions" / "manual-after.txt"
    after_path.write_text(after, encoding="utf-8")
    after_id = database.record_manual_chapter_version(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        version_path=after_path,
        char_count=len(after),
        effective_char_count=len(after),
        content_hash=hashlib.sha256(after.encode()).hexdigest(),
        change_summary="删去抽象解释，让动作和物件承担反应。",
    )
    assert before_id and after_id
    return before_id, after_id, before, after


def _csrf(html: str) -> str:
    match = re.search(r'name="csrf" value="([^"]+)"', html)
    assert match
    return match.group(1)


def _confirm_manual_edit_preference(
    database: Database,
    *,
    user_id: int,
    project_id: str,
    chapter_id: str,
    after_version_id: str,
    category: str = "emotional_expression",
    guidance: str = "人物反应已由动作表达时，不再追加抽象情绪总结。",
    applicability: str = "适用于即时受刺激的反应段落，不用于复杂推理。",
) -> str:
    service = PreferenceService(database)
    suggestion_id = service.create_suggestion(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        source_type="chapter",
        after_version_id=after_version_id,
        provider="mock",
        model="mock-edit-preference-learner",
        credential_source="default",
    )
    claimed = service.claim_next_suggestion()
    if claimed:
        assert claimed["id"] == suggestion_id
        completed = service.complete_suggestion(
            suggestion_id=suggestion_id,
            claim_token=str(claimed["claim_token"]),
            suggestion={
                "summary": "作者用具体动作替代抽象总结。",
                "preferences": [
                    {
                        "category": category,
                        "guidance": guidance,
                        "applicability": applicability,
                        "before_quote": "她感到十分难过",
                        "after_quote": "她把湿信折了两次",
                        "rationale": (
                            "改稿证据显示作者主动删除抽象解释。"
                        ),
                    }
                ],
                "uncertainties": [],
            },
            raw_response="{}",
            provider="mock",
            model="mock-edit-preference-learner",
            valid_evidence_count=1,
            dropped_evidence_count=0,
            input_tokens=0,
            output_tokens=0,
        )
        assert completed
    else:
        terminal = None
        for _ in range(100):
            terminal = service.get_suggestion(
                user_id=user_id, suggestion_id=suggestion_id
            )
            if terminal and terminal["status"] in {"ready", "failed"}:
                break
            time.sleep(0.01)
        assert terminal and terminal["status"] == "ready"
    assert (
        service.apply_suggestion(
            user_id=user_id,
            suggestion_id=suggestion_id,
            selections=[
                {
                    "index": 0,
                    "category": category,
                    "guidance": guidance,
                    "applicability": applicability,
                }
            ],
        )
        == project_id
    )
    with database.connection() as connection:
        row = connection.execute(
            """
            SELECT id FROM author_editing_preferences
            WHERE suggestion_id=?
            """,
            (suggestion_id,),
        ).fetchone()
    assert row
    return str(row["id"])


def test_deepseek_edit_preference_extractor_uses_bounded_diff_and_schema(
    tmp_path,
):
    before = "她感到十分难过，内心深处充满了复杂的情绪。"
    after = "她把湿信折了两次，塞回外套，没再看窗外。"
    change_sample = build_edit_sample(before, after)
    seen = {}
    result = {
        "summary": "作者把抽象情绪解释改成了可观察动作，形成一条待确认候选。",
        "preferences": [
            {
                "category": "emotional_expression",
                "guidance": "人物反应已经能由动作表达时，不再追加抽象情绪总结。",
                "applicability": "适用于即时受刺激后的反应段落，不用于必要的复杂推理。",
                "before_quote": before,
                "after_quote": after,
                "rationale": "修改前后显示作者主动用动作替代概括。",
            }
        ],
        "uncertainties": ["单次改稿仍需作者确认是否具有长期代表性。"],
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                result, ensure_ascii=False
                            )
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 260,
                    "completion_tokens": 110,
                },
            },
        )

    settings = replace(
        _settings(tmp_path), deepseek_api_key="sk-edit-pref-test"
    )

    async def scenario():
        extractor = DeepSeekEditPreferenceExtractor(
            settings, transport=httpx.MockTransport(handler)
        )
        try:
            return await extractor.extract(
                project={
                    "title": "雾港来信",
                    "genre": "悬疑",
                    "point_of_view": "第三人称限知",
                    "style_guide": "克制具体",
                },
                source={
                    "source_type": "chapter",
                    "chapter_title": "第一章",
                    "author_change_summary": "减少解释。",
                },
                change_sample=change_sample,
                before_text=before,
                after_text=after,
                provider_user_id="stable-user",
            )
        finally:
            await extractor.close()

    response = asyncio.run(scenario())
    assert response.input_tokens == 260
    assert response.output_tokens == 110
    assert response.result.preferences[0].category == "emotional_expression"
    assert seen["payload"]["response_format"] == {"type": "json_object"}
    assert seen["payload"]["user_id"] == "stable-user"
    system_prompt = seen["payload"]["messages"][0]["content"]
    assert "只能成为待审核建议" in system_prompt
    prompt_payload = json.loads(
        seen["payload"]["messages"][1]["content"].split("\n", 1)[1]
    )
    assert prompt_payload["bounded_change_sample"] == change_sample
    assert "before_text" not in prompt_payload
    assert "after_text" not in prompt_payload


def test_edit_sample_counts_actual_change_instead_of_whole_paragraph():
    before = "潮声压在窗外。" * 80 + "旧"
    after = "潮声压在窗外。" * 80 + "新"
    sample = build_edit_sample(before, after)
    assert sample["changed_char_count"] == 2


def test_manual_edit_preference_is_evidence_gated_and_author_confirmed(
    tmp_path,
):
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    user_id = database.create_user(
        "edit-owner", hash_password("password-123")
    )
    other_user_id = database.create_user(
        "edit-other", hash_password("password-123")
    )
    project_id = "edit-project"
    _create_project(database, user_id=user_id, project_id=project_id)
    before_id, after_id, before, after = _manual_edit_pair(
        database,
        tmp_path,
        user_id=user_id,
        project_id=project_id,
    )
    service = PreferenceService(database)
    suggestion_id = service.create_suggestion(
        user_id=user_id,
        project_id=project_id,
        chapter_id="chapter-one",
        source_type="chapter",
        after_version_id=after_id,
        provider="mock",
        model="mock-edit-preference-learner",
        credential_source="default",
    )
    claimed = service.claim_next_suggestion()
    assert claimed["id"] == suggestion_id
    assert claimed["before_version_id"] == before_id

    async def scenario():
        worker = AnalysisWorker(
            database,
            MockAnalyzer(),
            MockWriter(),
            settings.secret_key,
            settings,
            CredentialCipher(settings.credential_secret),
            edit_preference_extractor=MockEditPreferenceExtractor(),
            poll_seconds=0.01,
        )
        await worker._process_edit_preference_suggestion(claimed)

    asyncio.run(scenario())
    ready = service.get_suggestion(
        user_id=user_id, suggestion_id=suggestion_id
    )
    assert ready["status"] == "ready"
    assert ready["valid_evidence_count"] == 1
    assert ready["suggestion"]["preferences"][0][
        "before_start_offset"
    ] >= 0
    assert (
        service.get_suggestion(
            user_id=other_user_id, suggestion_id=suggestion_id
        )
        is None
    )
    context_before = database.get_writing_context(
        user_id, "chapter-one"
    )
    assert context_before["confirmed_editing_preferences"] == []

    candidate = ready["suggestion"]["preferences"][0]
    project = service.apply_suggestion(
        user_id=user_id,
        suggestion_id=suggestion_id,
        selections=[
            {
                "index": 0,
                "category": candidate["category"],
                "guidance": (
                    "人物反应已由动作表达时，不再追加抽象情绪总结。"
                ),
                "applicability": (
                    "适用于即时受刺激的反应段落，不用于复杂推理。"
                ),
            }
        ],
    )
    assert project == project_id
    active = service.list_active_preferences(
        user_id=user_id, project_id=project_id
    )
    assert len(active) == 1
    context = database.get_writing_context(user_id, "chapter-one")
    assert context["confirmed_editing_preferences"][0][
        "guidance"
    ].startswith("人物反应")
    serialized_context = json.dumps(context, ensure_ascii=False)
    assert candidate["before_quote"] not in serialized_context
    assert candidate["after_quote"] not in serialized_context
    messages = build_writing_messages(
        context=context,
        operation="draft",
        instruction="",
        current_content="",
        previous_content="",
    )
    assert "人物反应已由动作表达时" in messages[1]["content"]
    assert before not in messages[1]["content"]
    assert after not in messages[1]["content"]
    chapter_snapshot = build_writing_context_snapshot(
        context=context,
        operation="draft",
        instruction="",
        current_content="",
        previous_content="",
    )
    scene_snapshot = build_scene_context_snapshot(
        context=context,
        operation="generate_scene",
        instruction="",
        current_scene_content="",
        previous_scene_content="",
        previous_chapter_content="",
    )
    assert chapter_snapshot["confirmed_editing_preferences"][0][
        "guidance"
    ].startswith("人物反应")
    assert scene_snapshot["confirmed_editing_preferences"] == (
        chapter_snapshot["confirmed_editing_preferences"]
    )
    assert before not in json.dumps(chapter_snapshot, ensure_ascii=False)
    audit_preferences = StyleService(database).list_preferences(
        user_id=user_id, project_id=project_id
    )
    assert audit_preferences[0]["decision"] == "confirmed_manual_edit"

    assert service.archive_preference(
        user_id=user_id, preference_id=active[0]["id"]
    ) == project_id
    assert database.get_writing_context(
        user_id, "chapter-one"
    )["confirmed_editing_preferences"] == []


def test_edit_preference_hallucinated_evidence_fails_without_mutation(
    tmp_path,
):
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    user_id = database.create_user(
        "edit-evidence", hash_password("password-123")
    )
    project_id = "edit-evidence-project"
    _create_project(database, user_id=user_id, project_id=project_id)
    _, after_id, _, _ = _manual_edit_pair(
        database,
        tmp_path,
        user_id=user_id,
        project_id=project_id,
    )
    service = PreferenceService(database)
    suggestion_id = service.create_suggestion(
        user_id=user_id,
        project_id=project_id,
        chapter_id="chapter-one",
        source_type="chapter",
        after_version_id=after_id,
        provider="mock",
        model="bad-edit-evidence",
        credential_source="default",
    )
    claimed = service.claim_next_suggestion()

    class BadEvidenceExtractor(BaseEditPreferenceExtractor):
        provider = "mock"
        model = "bad-edit-evidence"

        async def extract(self, **kwargs):
            del kwargs
            result = EditPreferenceSuggestion.model_validate(
                {
                    "summary": "这份结果故意提供改稿中不存在的证据用于门禁测试。",
                    "preferences": [
                        {
                            "category": "diction",
                            "guidance": "删除所有不必要的抽象总结表达。",
                            "applicability": "人物即时反应场景。",
                            "before_quote": "修改前不存在的句子。",
                            "after_quote": "修改后也不存在的句子。",
                            "rationale": "故意制造无效证据。",
                        }
                    ],
                }
            )
            return EditPreferenceExtractionResponse(
                result=result,
                raw_response=result.model_dump_json(),
                input_tokens=10,
                output_tokens=20,
                provider=self.provider,
                model=self.model,
            )

    async def scenario():
        worker = AnalysisWorker(
            database,
            MockAnalyzer(),
            MockWriter(),
            settings.secret_key,
            settings,
            CredentialCipher(settings.credential_secret),
            edit_preference_extractor=BadEvidenceExtractor(),
        )
        await worker._process_edit_preference_suggestion(claimed)

    asyncio.run(scenario())
    failed = service.get_suggestion(
        user_id=user_id, suggestion_id=suggestion_id
    )
    assert failed["status"] == "failed"
    assert "逐字核对" in failed["error"]
    assert service.list_active_preferences(
        user_id=user_id, project_id=project_id
    ) == []


def test_expired_edit_preference_lease_is_requeued(tmp_path):
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    user_id = database.create_user(
        "edit-lease", hash_password("password-123")
    )
    project_id = "edit-lease-project"
    _create_project(database, user_id=user_id, project_id=project_id)
    _, after_id, _, _ = _manual_edit_pair(
        database,
        tmp_path,
        user_id=user_id,
        project_id=project_id,
    )
    service = PreferenceService(database)
    suggestion_id = service.create_suggestion(
        user_id=user_id,
        project_id=project_id,
        chapter_id="chapter-one",
        source_type="chapter",
        after_version_id=after_id,
        provider="mock",
        model="mock-edit-preference-learner",
        credential_source="default",
    )
    first = service.claim_next_suggestion()
    with database.connection() as connection:
        connection.execute(
            """
            UPDATE editing_preference_suggestions
            SET lease_expires_at='2000-01-01T00:00:00+00:00'
            WHERE id=?
            """,
            (suggestion_id,),
        )
        connection.commit()
    second = service.claim_next_suggestion()
    assert second["id"] == suggestion_id
    assert second["claim_token"] != first["claim_token"]


def test_scene_manual_edit_source_is_bound_to_exact_scene(tmp_path):
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    user_id = database.create_user(
        "scene-edit-owner", hash_password("password-123")
    )
    project_id = "scene-edit-project"
    _create_project(database, user_id=user_id, project_id=project_id)
    chapter_dir = tmp_path / project_id / "scene-chapter"
    chapter_dir.mkdir(parents=True)
    content_path = chapter_dir / "content.txt"
    content_path.write_text("", encoding="utf-8")
    database.add_novel_chapter(
        user_id=user_id,
        project_id=project_id,
        chapter_id="scene-chapter",
        title="第一章",
        outline="核对来信。",
        key_points="邮戳",
        content_path=content_path,
    )
    before = "她觉得自己十分震惊，也明白这封信必然意味着什么。"
    after = "她捏住信封一角，指腹在新邮戳上停了两秒。"
    before_path = chapter_dir / "scene-before.txt"
    after_path = chapter_dir / "scene-after.txt"
    before_path.write_text(before, encoding="utf-8")
    after_path.write_text(after, encoding="utf-8")
    timestamp = "2026-07-23T00:00:00+00:00"
    with database.connection() as connection:
        connection.execute(
            """
            INSERT INTO novel_chapter_plans(
                id, project_id, chapter_id, purpose, start_state,
                end_state, central_conflict, emotional_value, target_chars,
                status, source, created_at, updated_at, confirmed_at
            ) VALUES (
                'scene-plan', ?, 'scene-chapter', '核对信件',
                '不相信', '决定调查', '邮戳矛盾', '希望重现',
                3000, 'confirmed', 'manual', ?, ?, ?
            )
            """,
            (project_id, timestamp, timestamp, timestamp),
        )
        connection.execute(
            """
            INSERT INTO novel_scene_beats(
                id, plan_id, position, pov_character, goal, obstacle,
                action, end_state, created_at, updated_at
            ) VALUES (
                'scene-one', 'scene-plan', 1, '林岚', '核对邮戳',
                '日期矛盾', '对照旧信', '承认值得调查', ?, ?
            )
            """,
            (timestamp, timestamp),
        )
        connection.execute(
            """
            INSERT INTO novel_scene_versions(
                id, scene_beat_id, parent_version_id, kind, source, status,
                content_path, char_count, effective_char_count, content_hash,
                plan_fingerprint, created_by, quality_status,
                hard_issue_count, created_at
            ) VALUES (
                'scene-before-v', 'scene-one', NULL, 'manual', 'manual',
                'candidate', ?, ?, ?, ?, 'fingerprint', 'author',
                'pending', 0, ?
            )
            """,
            (
                str(before_path),
                len(before),
                len(before),
                hashlib.sha256(before.encode()).hexdigest(),
                timestamp,
            ),
        )
        connection.execute(
            """
            INSERT INTO novel_scene_versions(
                id, scene_beat_id, parent_version_id, kind, source, status,
                content_path, char_count, effective_char_count, content_hash,
                plan_fingerprint, created_by, quality_status,
                hard_issue_count, created_at
            ) VALUES (
                'scene-after-v', 'scene-one', 'scene-before-v', 'manual',
                'manual', 'candidate', ?, ?, ?, ?, 'fingerprint', 'author',
                'pending', 0, ?
            )
            """,
            (
                str(after_path),
                len(after),
                len(after),
                hashlib.sha256(after.encode()).hexdigest(),
                timestamp,
            ),
        )
        connection.commit()

    service = PreferenceService(database)
    suggestion_id = service.create_suggestion(
        user_id=user_id,
        project_id=project_id,
        chapter_id="scene-chapter",
        source_type="scene",
        after_version_id="scene-after-v",
        expected_scene_beat_id="scene-one",
        provider="mock",
        model="mock-edit-preference-learner",
        credential_source="default",
    )
    suggestion = service.get_suggestion(
        user_id=user_id, suggestion_id=suggestion_id
    )
    assert suggestion["scene_beat_id"] == "scene-one"
    assert suggestion["scene_goal"] == "核对邮戳"


def test_edit_preference_extractor_uses_owning_personal_key(
    tmp_path, monkeypatch
):
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    user_id = database.create_user(
        "edit-key-owner", hash_password("password-123")
    )
    cipher = CredentialCipher(settings.credential_secret)
    raw_key = "sk-edit-owner-2468"
    database.upsert_api_credential(
        user_id=user_id,
        encrypted_key=cipher.encrypt(raw_key),
        key_hint=key_hint(raw_key),
        model="deepseek-v4-pro",
    )
    seen = {}

    class FakePersonalExtractor(MockEditPreferenceExtractor):
        provider = "deepseek"

        def __init__(self, personal_settings):
            self.model = personal_settings.deepseek_model
            seen["api_key"] = personal_settings.deepseek_api_key
            seen["model"] = personal_settings.deepseek_model
            seen["thinking"] = personal_settings.deepseek_thinking
            seen["effort"] = personal_settings.deepseek_reasoning_effort

    monkeypatch.setattr(
        "app.worker.DeepSeekEditPreferenceExtractor",
        FakePersonalExtractor,
    )
    before = "她觉得这件事显然让人无比难过。"
    after = "她把信推回桌角，指尖一直压着折痕。"

    async def scenario():
        worker = AnalysisWorker(
            database,
            MockAnalyzer(),
            MockWriter(),
            settings.secret_key,
            settings,
            cipher,
            edit_preference_extractor=MockEditPreferenceExtractor(),
        )
        return await worker._extract_edit_preferences(
            {
                "user_id": user_id,
                "provider": "deepseek",
                "model": "deepseek-v4-pro",
                "credential_source": "personal",
                "title": "雾港来信",
                "genre": "悬疑",
                "point_of_view": "第三人称限知",
                "style_guide": "克制具体",
                "source_type": "chapter",
                "chapter_title": "第一章",
                "scene_goal": "",
                "author_change_summary": "",
                "change_sample": build_edit_sample(before, after),
            },
            before,
            after,
        )

    response = asyncio.run(scenario())
    assert response.provider == "deepseek"
    assert seen == {
        "api_key": raw_key,
        "model": "deepseek-v4-pro",
        "thinking": False,
        "effort": "high",
    }


def test_cross_edit_aggregate_replaces_sources_and_archive_restores_them(
    tmp_path,
):
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    user_id = database.create_user(
        "aggregate-owner", hash_password("password-123")
    )
    project_id = "aggregate-project"
    _create_project(database, user_id=user_id, project_id=project_id)
    _, first_after_id, _, _ = _manual_edit_pair(
        database,
        tmp_path,
        user_id=user_id,
        project_id=project_id,
        chapter_id="chapter-one",
    )
    _, second_after_id, _, _ = _manual_edit_pair(
        database,
        tmp_path,
        user_id=user_id,
        project_id=project_id,
        chapter_id="chapter-two",
    )
    _, third_after_id, _, _ = _manual_edit_pair(
        database,
        tmp_path,
        user_id=user_id,
        project_id=project_id,
        chapter_id="chapter-three",
    )
    first_id = _confirm_manual_edit_preference(
        database,
        user_id=user_id,
        project_id=project_id,
        chapter_id="chapter-one",
        after_version_id=first_after_id,
    )
    second_id = _confirm_manual_edit_preference(
        database,
        user_id=user_id,
        project_id=project_id,
        chapter_id="chapter-two",
        after_version_id=second_after_id,
        guidance="即时反应已有可见动作时，删掉解释性的情绪结论。",
        applicability="适用于受刺激后的动作段落，不限制必要推理。",
    )
    third_id = _confirm_manual_edit_preference(
        database,
        user_id=user_id,
        project_id=project_id,
        chapter_id="chapter-three",
        after_version_id=third_after_id,
        guidance="复杂迟疑场景允许保留一处内在判断，不必全部改成动作。",
        applicability="仅适用于人物需要权衡多个动机的迟疑段落。",
    )
    service = PreferenceService(database)
    candidates = service.list_aggregation_candidates(
        user_id=user_id, project_id=project_id
    )
    assert len(candidates) == 1
    assert candidates[0]["source_count"] == 3
    assert {item["id"] for item in candidates[0]["preferences"]} == {
        first_id,
        second_id,
        third_id,
    }

    aggregate_id = service.create_aggregate(
        user_id=user_id,
        project_id=project_id,
        category="emotional_expression",
        guidance="即时反应已有具体动作时，省略追加的抽象情绪结论。",
        applicability="适用于人物受刺激后的即时反应；复杂推理可保留必要解释。",
        author_note="对白场景仍按人物声音判断。",
        support_preference_ids=[first_id, second_id],
        conflict_preference_ids=[third_id],
    )
    aggregates = service.list_aggregates(
        user_id=user_id, project_id=project_id
    )
    assert aggregates[0]["id"] == aggregate_id
    assert aggregates[0]["support_count"] == 2
    assert (
        aggregates[0]["confidence_label"]
        == "重复出现 · 有冲突证据"
    )
    assert aggregates[0]["conflict_count"] == 1
    assert service.list_aggregation_candidates(
        user_id=user_id, project_id=project_id
    ) == []

    context = database.get_writing_context(user_id, "chapter-one")
    preferences = context["confirmed_editing_preferences"]
    assert len(preferences) == 1
    assert preferences[0]["id"] == aggregate_id
    assert preferences[0]["source_type"] == "stable_aggregate"
    assert preferences[0]["support_count"] == 2
    serialized = json.dumps(preferences, ensure_ascii=False)
    assert "她感到十分难过" not in serialized
    assert "她把湿信折了两次" not in serialized
    assert "对白场景仍按人物声音判断" not in serialized
    audit_preferences = StyleService(database).list_preferences(
        user_id=user_id, project_id=project_id
    )
    assert len(audit_preferences) == 1
    assert (
        audit_preferences[0]["source_type"]
        == "stable_editing_preference"
    )
    assert (
        service.archive_preference(
            user_id=user_id, preference_id=first_id
        )
        is None
    )

    assert (
        service.archive_aggregate(
            user_id=user_id, aggregate_id=aggregate_id
        )
        == project_id
    )
    restored = database.get_writing_context(
        user_id, "chapter-one"
    )["confirmed_editing_preferences"]
    assert {item["id"] for item in restored} == {
        first_id,
        second_id,
        third_id,
    }
    history = service.list_aggregates(
        user_id=user_id,
        project_id=project_id,
        include_archived=True,
    )
    assert history[0]["status"] == "archived"
    assert len(history[0]["supports"]) == 2


def test_aggregate_requires_two_distinct_manual_edits(tmp_path):
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    user_id = database.create_user(
        "aggregate-source-owner", hash_password("password-123")
    )
    project_id = "aggregate-source-project"
    _create_project(database, user_id=user_id, project_id=project_id)
    _, after_id, _, _ = _manual_edit_pair(
        database,
        tmp_path,
        user_id=user_id,
        project_id=project_id,
    )
    service = PreferenceService(database)
    suggestion_id = service.create_suggestion(
        user_id=user_id,
        project_id=project_id,
        chapter_id="chapter-one",
        source_type="chapter",
        after_version_id=after_id,
        provider="mock",
        model="mock-edit-preference-learner",
        credential_source="default",
    )
    claimed = service.claim_next_suggestion()
    preferences = [
        {
            "category": "omission",
            "guidance": "动作已经表达态度时，不再补充解释性的总结句。",
            "applicability": "适用于即时反应段落。",
            "before_quote": "她知道自己很难过",
            "after_quote": "她把信折起来",
            "rationale": "删去解释。",
        },
        {
            "category": "omission",
            "guidance": "段尾留在动作或物件上，不替读者概括人物感受。",
            "applicability": "适用于情绪段落结尾。",
            "before_quote": "这让她百感交集",
            "after_quote": "折痕压在她指下",
            "rationale": "保留留白。",
        },
    ]
    assert service.complete_suggestion(
        suggestion_id=suggestion_id,
        claim_token=str(claimed["claim_token"]),
        suggestion={
            "summary": "同一次改稿产生两条候选。",
            "preferences": preferences,
            "uncertainties": [],
        },
        raw_response="{}",
        provider="mock",
        model="mock-edit-preference-learner",
        valid_evidence_count=2,
        dropped_evidence_count=0,
        input_tokens=0,
        output_tokens=0,
    )
    service.apply_suggestion(
        user_id=user_id,
        suggestion_id=suggestion_id,
        selections=[
            {
                "index": index,
                "category": item["category"],
                "guidance": item["guidance"],
                "applicability": item["applicability"],
            }
            for index, item in enumerate(preferences)
        ],
    )
    active = service.list_active_preferences(
        user_id=user_id, project_id=project_id
    )
    assert len(active) == 2
    with pytest.raises(
        ValueError,
        match="至少两次不同的手工改稿",
    ):
        service.create_aggregate(
            user_id=user_id,
            project_id=project_id,
            category="omission",
            guidance="动作已经表达态度时，段尾不再追加解释总结。",
            applicability="适用于即时反应与情绪段落结尾。",
            author_note="",
            support_preference_ids=[item["id"] for item in active],
            conflict_preference_ids=[],
        )
    with database.connection() as connection:
        count = connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM author_editing_preference_aggregates
            """
        ).fetchone()
    assert count["count"] == 0


def test_stable_preference_reports_only_non_causal_audit_observation(
    tmp_path,
):
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    user_id = database.create_user(
        "effect-owner", hash_password("password-123")
    )
    project_id = "effect-project"
    _create_project(database, user_id=user_id, project_id=project_id)
    with database.connection() as connection:
        voice_profile = connection.execute(
            """
            SELECT id FROM novel_voice_profiles WHERE project_id=?
            """,
            (project_id,),
        ).fetchone()

    samples = [
        ("before", "2026-03-01T10:00:00+00:00", 1),
        ("before", "2026-03-08T10:00:00+00:00", 1),
        ("before", "2026-03-15T10:00:00+00:00", 1),
        ("after", "2026-05-01T10:00:00+00:00", 0),
        ("after", "2026-05-08T10:00:00+00:00", 0),
        ("after", "2026-05-15T10:00:00+00:00", 0),
    ]
    for index, (period, timestamp, issue_count) in enumerate(samples):
        chapter_id = f"effect-{period}-{index}"
        chapter_dir = tmp_path / project_id / chapter_id
        chapter_dir.mkdir(parents=True)
        content_path = chapter_dir / "content.txt"
        content_path.write_text("潮声压住窗缝。", encoding="utf-8")
        database.add_novel_chapter(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            title=f"观察章节 {index}",
            outline="观察审校趋势。",
            key_points="潮声",
            content_path=content_path,
        )
        version_path = chapter_dir / "version.txt"
        version_path.write_text("潮声压住窗缝。", encoding="utf-8")
        version_id = database.record_manual_chapter_version(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            version_path=version_path,
            char_count=2500,
            effective_char_count=2500,
            content_hash=f"effect-hash-{period}-{index}",
        )
        audit_id = f"effect-audit-{period}-{index}"
        with database.connection() as connection:
            connection.execute(
                """
                UPDATE novel_chapter_versions
                SET created_at=? WHERE id=?
                """,
                (timestamp, version_id),
            )
            connection.execute(
                """
                INSERT INTO chapter_style_audits(
                    id, project_id, chapter_id, version_id,
                    voice_profile_id, summary, issue_count,
                    dropped_issue_count, provider, model, result_json,
                    input_tokens, output_tokens, created_at
                ) VALUES (
                    ?, ?, ?, ?, ?, '同期观察', ?, 0,
                    'mock', 'same-auditor', '{}', 0, 0, ?
                )
                """,
                (
                    audit_id,
                    project_id,
                    chapter_id,
                    version_id,
                    voice_profile["id"],
                    issue_count,
                    timestamp,
                ),
            )
            if issue_count:
                connection.execute(
                    """
                    INSERT INTO chapter_style_issues(
                        id, audit_id, project_id, chapter_id, version_id,
                        position, paragraph_index, start_offset,
                        end_offset, quote, issue_type, severity, evidence,
                        reader_impact, rewrite_direction, status,
                        created_at, updated_at
                    ) VALUES (
                        ?, ?, ?, ?, ?, 1, 1, 0, 2, '难过',
                        'abstract_emotion', 'medium', '抽象概括',
                        '读者缺少可见证据', '改为动作', 'open', ?, ?
                    )
                    """,
                    (
                        f"effect-issue-{period}-{index}",
                        audit_id,
                        project_id,
                        chapter_id,
                        version_id,
                        timestamp,
                        timestamp,
                    ),
                )
            connection.commit()

    effect = PreferenceService(database).get_effect_observation(
        user_id=user_id,
        aggregate={
            "project_id": project_id,
            "category": "emotional_expression",
            "confirmed_at": "2026-04-01T10:00:00+00:00",
        },
    )
    assert effect["status"] == "observable"
    assert effect["comparability"] == "matched"
    assert effect["before"]["versions"] == 3
    assert effect["before"]["chapters"] == 3
    assert effect["before"]["chars"] == 7500
    assert effect["before"]["issues"] == 3
    assert effect["before"]["rate_per_10k"] == 4.0
    assert effect["after"]["rate_per_10k"] == 0.0
    assert effect["direction"] == "decreased"
    assert "不能证明" in effect["disclaimer"]


def test_v13_migration_preserves_existing_style_preferences(tmp_path):
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    user_id = database.create_user(
        "edit-migration", hash_password("password-123")
    )
    project_id = "edit-migration-project"
    _create_project(database, user_id=user_id, project_id=project_id)
    with database.connection() as connection:
        connection.execute(
            """
            INSERT INTO author_style_preferences(
                id, project_id, issue_id, issue_type, decision,
                original_text, replacement_text, guidance, created_at
            ) VALUES (
                'existing-pref', ?, NULL, 'repetition', 'ignored',
                '原文', '', '保留这次重复', '2026-01-01T00:00:00+00:00'
            )
            """,
            (project_id,),
        )
        connection.execute("DROP TABLE author_editing_preferences")
        connection.execute("DROP TABLE editing_preference_suggestions")
        connection.execute(
            "DELETE FROM schema_migrations WHERE version=13"
        )
        connection.commit()

    database.initialize()
    with database.connection() as connection:
        legacy = connection.execute(
            """
            SELECT guidance FROM author_style_preferences
            WHERE id='existing-pref'
            """
        ).fetchone()
        migration = connection.execute(
            "SELECT name FROM schema_migrations WHERE version=13"
        ).fetchone()
        tables = {
            row["name"]
            for row in connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name IN (
                    'editing_preference_suggestions',
                    'author_editing_preferences'
                )
                """
            ).fetchall()
        }
    assert legacy["guidance"] == "保留这次重复"
    assert migration["name"] == "manual_edit_preference_learning_v13"
    assert tables == {
        "editing_preference_suggestions",
        "author_editing_preferences",
    }


def test_v22_migration_preserves_existing_editing_preferences(tmp_path):
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    user_id = database.create_user(
        "aggregate-migration", hash_password("password-123")
    )
    project_id = "aggregate-migration-project"
    _create_project(database, user_id=user_id, project_id=project_id)
    with database.connection() as connection:
        connection.execute(
            """
            INSERT INTO author_editing_preferences(
                id, project_id, suggestion_id, category, guidance,
                applicability, status, created_at, updated_at
            ) VALUES (
                'legacy-edit-pref', ?, NULL, 'omission',
                '动作已经表达态度时，不再补充解释性总结。',
                '适用于即时反应段落。', 'active',
                '2026-01-01T00:00:00+00:00',
                '2026-01-01T00:00:00+00:00'
            )
            """,
            (project_id,),
        )
        connection.execute(
            """
            DROP TABLE author_editing_preference_aggregate_evidence
            """
        )
        connection.execute(
            "DROP TABLE author_editing_preference_aggregates"
        )
        connection.execute(
            "DELETE FROM schema_migrations WHERE version=22"
        )
        connection.commit()

    database.initialize()
    with database.connection() as connection:
        preference = connection.execute(
            """
            SELECT guidance FROM author_editing_preferences
            WHERE id='legacy-edit-pref'
            """
        ).fetchone()
        migration = connection.execute(
            "SELECT name FROM schema_migrations WHERE version=22"
        ).fetchone()
        tables = {
            row["name"]
            for row in connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name IN (
                  'author_editing_preference_aggregates',
                  'author_editing_preference_aggregate_evidence'
                )
                """
            ).fetchall()
        }
    assert preference["guidance"].startswith("动作已经表达")
    assert (
        migration["name"]
        == "editing_preference_aggregation_v22"
    )
    assert tables == {
        "author_editing_preference_aggregates",
        "author_editing_preference_aggregate_evidence",
    }


def test_edit_preference_web_flow_requires_author_confirmation(tmp_path):
    application = create_app(_settings(tmp_path))
    with TestClient(application) as client:
        page = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "改稿作者",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": _csrf(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        page = client.get("/novels/new")
        response = client.post(
            "/novels/new",
            data={
                "title": "改稿学习测试",
                "genre": "悬疑",
                "premise": "记者收到一封来自失踪父亲的旧信。",
                "world_setting": "当代港城。",
                "style_guide": "克制具体。",
                "point_of_view": "第三人称限知",
                "target_chapter_chars": "3000",
                "csrf": _csrf(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        workbench_url = response.headers["location"]
        project_id = workbench_url.split("/novels/", 1)[1].split("/", 1)[0]
        project_url = f"/novels/{project_id}"
        database = application.state.database
        user = database.get_user_by_username("改稿作者")
        _, after_id, _, _ = _manual_edit_pair(
            database,
            tmp_path,
            user_id=int(user["id"]),
            project_id=project_id,
        )

        chapter_url = f"{project_url}/chapters/chapter-one"
        page = client.get(
            f"{project_url}/workbench?chapter_id=chapter-one"
        )
        assert 'class="studio-manuscript-view"' in page.text
        assert "从这次改稿学习" not in page.text
        response = client.post(
            f"{chapter_url}/versions/{after_id}/learn-edit-preferences",
            data={"csrf": _csrf(page.text)},
            follow_redirects=False,
        )
        assert response.status_code == 303
        suggestion_url = response.headers["location"]
        assert suggestion_url.startswith("/editing-preference-suggestions/")
        suggestion_id = suggestion_url.rsplit("/", 1)[-1]

        terminal = None
        for _ in range(100):
            terminal = PreferenceService(database).get_suggestion(
                user_id=int(user["id"]),
                suggestion_id=suggestion_id,
            )
            if terminal and terminal["status"] in {
                "ready",
                "failed",
            }:
                break
            time.sleep(0.02)
        assert terminal["status"] == "ready"
        assert database.get_writing_context(
            int(user["id"]), "chapter-one"
        )["confirmed_editing_preferences"] == []

        page = client.get(suggestion_url)
        assert "修改前" in page.text
        assert "修改后" in page.text
        candidate = terminal["suggestion"]["preferences"][0]
        applied = client.post(
            f"{suggestion_url}/apply",
            data={
                "csrf": _csrf(page.text),
                "selected": "0",
                "category_0": candidate["category"],
                "guidance_0": "人物反应已由动作表达时，不再追加抽象总结。",
                "applicability_0": "适用于人物即时反应段落。",
            },
            follow_redirects=False,
        )
        assert applied.status_code == 303
        assert (
            "view=archive&archive_tab=creative&settings_tab=style&saved=true"
            in applied.headers["location"]
        )
        context = database.get_writing_context(
            int(user["id"]), "chapter-one"
        )
        assert context["confirmed_editing_preferences"][0][
            "guidance"
        ].startswith("人物反应")


def test_stable_preference_web_actions_use_workbench_return_path(
    tmp_path,
):
    application = create_app(_settings(tmp_path))
    with TestClient(application) as client:
        page = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "编辑记忆作者",
                "password": "password-123",
                "password_confirm": "password-123",
                "csrf": _csrf(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        page = client.get("/novels/new")
        response = client.post(
            "/novels/new",
            data={
                "title": "编辑记忆测试",
                "genre": "悬疑",
                "premise": "记者收到一封不可能存在的旧信。",
                "world_setting": "当代港城。",
                "style_guide": "克制具体。",
                "point_of_view": "第三人称限知",
                "target_chapter_chars": "3000",
                "csrf": _csrf(page.text),
            },
            follow_redirects=False,
        )
        workbench_url = response.headers["location"]
        project_id = workbench_url.split("/novels/", 1)[1].split("/", 1)[0]
        database = application.state.database
        user = database.get_user_by_username("编辑记忆作者")
        _, first_after_id, _, _ = _manual_edit_pair(
            database,
            tmp_path,
            user_id=int(user["id"]),
            project_id=project_id,
            chapter_id="memory-chapter-one",
        )
        _, second_after_id, _, _ = _manual_edit_pair(
            database,
            tmp_path,
            user_id=int(user["id"]),
            project_id=project_id,
            chapter_id="memory-chapter-two",
        )
        first_id = _confirm_manual_edit_preference(
            database,
            user_id=int(user["id"]),
            project_id=project_id,
            chapter_id="memory-chapter-one",
            after_version_id=first_after_id,
        )
        second_id = _confirm_manual_edit_preference(
            database,
            user_id=int(user["id"]),
            project_id=project_id,
            chapter_id="memory-chapter-two",
            after_version_id=second_after_id,
            guidance="人物动作已经显出情绪时，删除后接的解释总结。",
            applicability="适用于人物即时反应，不限制复杂推理。",
        )

        page = client.get(
            f"/novels/{project_id}/workbench"
            "?view=archive&archive_tab=creative&settings_tab=style"
        )
        assert page.status_code == 200
        response = client.post(
            f"/novels/{project_id}/editing-preference-aggregates",
            data={
                "csrf": _csrf(page.text),
                "category": "emotional_expression",
                f"role_{first_id}": "support",
                f"role_{second_id}": "support",
                "guidance": (
                    "人物即时反应已有具体动作时，省略追加的抽象情绪总结。"
                ),
                "applicability": (
                    "适用于受刺激后的即时反应；复杂推理保留必要解释。"
                ),
                "author_note": "这是作者确认的边界。",
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert (
            "view=archive&archive_tab=creative&settings_tab=style&saved=true"
            in response.headers["location"]
        )
        with database.connection() as connection:
            aggregate = connection.execute(
                """
                SELECT id, guidance
                FROM author_editing_preference_aggregates
                WHERE project_id=? AND status='active'
                """,
                (project_id,),
            ).fetchone()
        assert "人物即时反应已有具体动作时" in aggregate["guidance"]
        archived = client.post(
            f"/editing-preference-aggregates/{aggregate['id']}/archive",
            data={"csrf": _csrf(page.text)},
            follow_redirects=False,
        )
        assert archived.status_code == 303
        assert (
            "view=archive&archive_tab=creative&settings_tab=style&saved=true"
            in archived.headers["location"]
        )
        restored = database.get_writing_context(
            int(user["id"]), "memory-chapter-one"
        )["confirmed_editing_preferences"]
        assert {item["id"] for item in restored} == {
            first_id,
            second_id,
        }
