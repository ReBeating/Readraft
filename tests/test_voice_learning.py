import asyncio
import json
import re
import time
from dataclasses import replace
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from app.config import Settings
from app.credentials import CredentialCipher, key_hint
from app.db import Database
from app.model_client import MockAnalyzer
from app.main import create_app
from app.security import hash_password
from app.style_service import StyleService
from app.voice_extraction import (
    BaseVoiceProfileExtractor,
    ProviderVoiceProfileExtractor,
    MockVoiceProfileExtractor,
    VoiceExtractionResponse,
)
from app.voice_schema import VoiceProfileSuggestion
from app.worker import AnalysisWorker
from app.writing import build_writing_messages


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        app_name="Readraft 测试",
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
        model_api_key=None,
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


def _sample_text() -> str:
    paragraph = (
        "林岚没有回答。她把湿信封压在杯底，先看邮戳，再看父亲的署名。"
        "窗框被风推响了一次，她才把日期抄进本子，余下的话停在笔尖。"
    )
    return "\n\n".join(paragraph for _ in range(10))


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
        style_guide="原始文风要求。",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
    )


def _csrf(html: str) -> str:
    match = re.search(r'name="csrf" value="([^"]+)"', html)
    assert match
    return match.group(1)


def test_deepseek_voice_extractor_uses_json_schema_and_exact_evidence(
    tmp_path,
):
    sample = _sample_text()
    seen = {}
    result = {
        "summary": "样章以动作和具体物件承担判断，较少直接解释人物情绪。",
        "narration_rules": "紧贴视角人物可观察的信息，先写动作再写判断。",
        "sentence_rhythm": "推进使用短句，转折处允许较长句形成停顿。",
        "dialogue_voice": "",
        "sensory_palette": "使用人物当下接触的潮湿纸张、窗框和声音。",
        "metaphor_policy": "",
        "allowed_omissions": "可以让未说出口的话停在动作或物件上。",
        "preferred_patterns": ["动作先于情绪解释"],
        "banned_expressions": [],
        "evidence": [
            {
                "dimension": "narration",
                "quote": "她把湿信封压在杯底，先看邮戳，再看父亲的署名。",
                "observation": "用连续动作交代判断过程。",
            },
            {
                "dimension": "omission",
                "quote": "余下的话停在笔尖。",
                "observation": "没有把人物未说出口的情绪解释完。",
            },
        ],
        "uncertainties": ["样章没有足够对话，暂不固定对话声音。"],
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
                    "prompt_tokens": 321,
                    "completion_tokens": 123,
                },
            },
        )

    settings = replace(
        _settings(tmp_path), model_api_key="sk-voice-test"
    )

    async def scenario():
        extractor = ProviderVoiceProfileExtractor(
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
                sample_title="作者样章",
                sample_text=sample,
                author_intent="保留动作先于解释。",
                provider_user_id="stable-user",
            )
        finally:
            await extractor.close()

    response = asyncio.run(scenario())
    assert response.result.narration_rules.startswith("紧贴视角人物")
    assert response.input_tokens == 321
    assert response.output_tokens == 123
    assert seen["payload"]["response_format"] == {"type": "json_object"}
    assert seen["payload"]["user_id"] == "stable-user"
    assert "作者自己拥有使用权" in seen["payload"]["messages"][0]["content"]
    prompt_payload = json.loads(
        seen["payload"]["messages"][1]["content"].split("\n", 1)[1]
    )
    assert prompt_payload["author_owned_sample"] == sample


def test_voice_suggestion_is_evidence_gated_and_author_applied(
    tmp_path,
):
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    user_id = database.create_user(
        "voice-owner", hash_password("password-123")
    )
    other_user_id = database.create_user(
        "voice-other", hash_password("password-123")
    )
    project_id = "voice-project"
    _create_project(database, user_id=user_id, project_id=project_id)
    service = StyleService(database)
    sample = _sample_text()
    suggestion_id = service.create_voice_suggestion(
        user_id=user_id,
        project_id=project_id,
        sample_title="我写的开篇",
        sample_text=sample,
        author_intent="不要使用：命运的齿轮",
        provider="mock",
        model="mock-voice-profiler",
        credential_source="default",
    )
    before = service.get_voice_profile(
        user_id=user_id, project_id=project_id
    )
    assert before["status"] == "draft"
    assert before["narration_rules"] == "原始文风要求。"
    claimed = service.claim_next_voice_suggestion()
    assert claimed["id"] == suggestion_id

    async def scenario():
        worker = AnalysisWorker(
            database,
            MockAnalyzer(),
            settings.secret_key,
            settings,
            CredentialCipher(settings.credential_secret),
            voice_profile_extractor=MockVoiceProfileExtractor(),
            poll_seconds=0.01,
        )
        await worker._process_voice_suggestion(claimed)

    asyncio.run(scenario())
    ready = service.get_voice_suggestion(
        user_id=user_id, suggestion_id=suggestion_id
    )
    assert ready["status"] == "ready"
    assert ready["valid_evidence_count"] >= 2
    assert ready["suggestion"]["evidence"][0]["start_offset"] >= 0
    assert (
        service.get_voice_suggestion(
            user_id=other_user_id, suggestion_id=suggestion_id
        )
        is None
    )
    still_draft = service.get_voice_profile(
        user_id=user_id, project_id=project_id
    )
    assert still_draft["narration_rules"] == "原始文风要求。"

    proposed = ready["suggestion"]
    applied_project_id = service.apply_voice_suggestion(
        user_id=user_id,
        suggestion_id=suggestion_id,
        narration_rules=proposed["narration_rules"],
        sentence_rhythm=proposed["sentence_rhythm"],
        dialogue_voice=proposed["dialogue_voice"],
        sensory_palette=proposed["sensory_palette"],
        metaphor_policy=proposed["metaphor_policy"],
        allowed_omissions=proposed["allowed_omissions"],
        preferred_patterns=proposed["preferred_patterns"],
        banned_expressions=["命运的齿轮"],
        author_notes="作者已逐项审核。",
        confirm=True,
    )
    assert applied_project_id == project_id
    profile = service.get_voice_profile(
        user_id=user_id, project_id=project_id
    )
    assert profile["status"] == "confirmed"
    assert profile["narration_rules"] == proposed["narration_rules"]
    assert profile["banned_expressions"] == ["命运的齿轮"]
    assert service.get_voice_suggestion(
        user_id=user_id, suggestion_id=suggestion_id
    )["status"] == "applied"

    chapter_path = tmp_path / "chapter.txt"
    database.add_novel_chapter(
        user_id=user_id,
        project_id=project_id,
        chapter_id="chapter-one",
        title="第一章",
        outline="林岚拆开信。",
        key_points="北塔地址",
        content_path=chapter_path,
    )
    context = database.get_writing_context(user_id, "chapter-one")
    assert context["voice_profile"]["status"] == "confirmed"
    messages = build_writing_messages(
        context=context,
        operation="draft",
        instruction="",
        current_content="",
        previous_content="",
    )
    assert proposed["narration_rules"] in messages[1]["content"]
    assert "命运的齿轮" in messages[1]["content"]

    service.update_voice_profile(
        user_id=user_id,
        project_id=project_id,
        narration_rules=profile["narration_rules"],
        sentence_rhythm=profile["sentence_rhythm"],
        dialogue_voice=profile["dialogue_voice"],
        sensory_palette=profile["sensory_palette"],
        metaphor_policy=profile["metaphor_policy"],
        allowed_omissions=profile["allowed_omissions"],
        preferred_patterns=profile["preferred_patterns"],
        banned_expressions=profile["banned_expressions"],
        author_notes=profile["author_notes"],
        confirm=False,
    )
    assert database.get_writing_context(
        user_id, "chapter-one"
    )["voice_profile"] is None


def test_voice_suggestion_with_hallucinated_quotes_fails_without_mutation(
    tmp_path,
):
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    user_id = database.create_user(
        "voice-evidence", hash_password("password-123")
    )
    project_id = "evidence-project"
    _create_project(database, user_id=user_id, project_id=project_id)
    service = StyleService(database)
    suggestion_id = service.create_voice_suggestion(
        user_id=user_id,
        project_id=project_id,
        sample_title="证据测试",
        sample_text=_sample_text(),
        author_intent="",
        provider="mock",
        model="bad-evidence",
        credential_source="default",
    )
    claimed = service.claim_next_voice_suggestion()

    class BadEvidenceExtractor(BaseVoiceProfileExtractor):
        provider = "mock"
        model = "bad-evidence"

        async def extract(self, **kwargs):
            del kwargs
            result = VoiceProfileSuggestion.model_validate(
                {
                    "summary": "这份结果故意提供无法定位的引用，用于验证证据门禁。",
                    "narration_rules": "使用动作承担人物判断。",
                    "sentence_rhythm": "长短句交替。",
                    "evidence": [
                        {
                            "dimension": "narration",
                            "quote": "样章中不存在的第一条句子。",
                            "observation": "无法核对。",
                        },
                        {
                            "dimension": "rhythm",
                            "quote": "样章中不存在的第二条句子。",
                            "observation": "仍然无法核对。",
                        },
                    ],
                }
            )
            return VoiceExtractionResponse(
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
            settings.secret_key,
            settings,
            CredentialCipher(settings.credential_secret),
            voice_profile_extractor=BadEvidenceExtractor(),
        )
        await worker._process_voice_suggestion(claimed)

    asyncio.run(scenario())
    failed = service.get_voice_suggestion(
        user_id=user_id, suggestion_id=suggestion_id
    )
    assert failed["status"] == "failed"
    assert "至少两条" in failed["error"]
    profile = service.get_voice_profile(
        user_id=user_id, project_id=project_id
    )
    assert profile["status"] == "draft"
    assert profile["narration_rules"] == "原始文风要求。"


def test_expired_voice_suggestion_lease_is_requeued(tmp_path):
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    user_id = database.create_user(
        "voice-lease", hash_password("password-123")
    )
    project_id = "voice-lease-project"
    _create_project(database, user_id=user_id, project_id=project_id)
    service = StyleService(database)
    suggestion_id = service.create_voice_suggestion(
        user_id=user_id,
        project_id=project_id,
        sample_title="租约测试",
        sample_text=_sample_text(),
        author_intent="",
        provider="mock",
        model="mock-voice-profiler",
        credential_source="default",
    )
    first = service.claim_next_voice_suggestion()
    with database.connection() as connection:
        connection.execute(
            """
            UPDATE voice_profile_suggestions
            SET lease_expires_at='2000-01-01T00:00:00+00:00'
            WHERE id=?
            """,
            (suggestion_id,),
        )
        connection.commit()
    second = service.claim_next_voice_suggestion()
    assert second["id"] == suggestion_id
    assert second["claim_token"] != first["claim_token"]
    current = service.get_voice_suggestion(
        user_id=user_id, suggestion_id=suggestion_id
    )
    assert current["status"] == "running"


def test_voice_extractor_uses_owning_users_decrypted_api_key(
    tmp_path, monkeypatch
):
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    user_id = database.create_user(
        "voice-key-owner", hash_password("password-123")
    )
    cipher = CredentialCipher(settings.credential_secret)
    raw_key = "sk-voice-owner-2468"
    database.upsert_api_credential(
        user_id=user_id,
        encrypted_key=cipher.encrypt(raw_key),
        key_hint=key_hint(raw_key),
        model="deepseek-v4-pro",
    )
    seen = {}

    class FakePersonalVoiceExtractor(MockVoiceProfileExtractor):
        provider = "deepseek"

        def __init__(self, personal_settings):
            self.model = personal_settings.model_name
            seen["api_key"] = personal_settings.model_api_key
            seen["model"] = personal_settings.model_name
            seen["thinking"] = personal_settings.model_thinking
            seen["effort"] = personal_settings.model_reasoning_effort

    monkeypatch.setattr(
        "app.worker.ProviderVoiceProfileExtractor",
        FakePersonalVoiceExtractor,
    )

    async def scenario():
        worker = AnalysisWorker(
            database,
            MockAnalyzer(),
            settings.secret_key,
            settings,
            cipher,
            voice_profile_extractor=MockVoiceProfileExtractor(),
        )
        return await worker._extract_voice_profile(
            {
                "user_id": user_id,
                "provider": "deepseek",
                "model": "deepseek-v4-pro",
                "credential_source": "personal",
                "title": "雾港来信",
                "genre": "悬疑",
                "point_of_view": "第三人称限知",
                "style_guide": "克制具体",
                "target_audience": "",
                "sample_title": "我的样章",
                "sample_text": _sample_text(),
                "author_intent": "",
            }
        )

    response = asyncio.run(scenario())
    assert response.provider == "deepseek"
    assert seen == {
        "api_key": raw_key,
        "model": "deepseek-v4-pro",
        "thinking": False,
        "effort": "high",
    }


def test_v12_migration_preserves_existing_voice_profile(tmp_path):
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    user_id = database.create_user(
        "voice-migration", hash_password("password-123")
    )
    project_id = "voice-migration-project"
    _create_project(database, user_id=user_id, project_id=project_id)
    service = StyleService(database)
    service.update_voice_profile(
        user_id=user_id,
        project_id=project_id,
        narration_rules="第三人称紧贴林岚。",
        sentence_rhythm="调查段落短句推进。",
        dialogue_voice="林岚习惯用反问回避。",
        sensory_palette="盐雾与旧金属。",
        metaphor_policy="低密度。",
        allowed_omissions="动作已说明时不解释。",
        preferred_patterns=["动作先于判断"],
        banned_expressions=["不禁"],
        author_notes="旧数据必须保留。",
        confirm=True,
    )
    with database.connection() as connection:
        connection.execute("DROP TABLE voice_profile_suggestions")
        connection.execute(
            "DELETE FROM schema_migrations WHERE version=12"
        )
        connection.commit()

    database.initialize()
    profile = service.get_voice_profile(
        user_id=user_id, project_id=project_id
    )
    assert profile["status"] == "confirmed"
    assert profile["narration_rules"] == "第三人称紧贴林岚。"
    assert profile["preferred_patterns"] == ["动作先于判断"]
    with database.connection() as connection:
        migration = connection.execute(
            "SELECT name FROM schema_migrations WHERE version=12"
        ).fetchone()
        table = connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='voice_profile_suggestions'
            """
        ).fetchone()
    assert migration["name"] == "voice_profile_learning_v12"
    assert table["name"] == "voice_profile_suggestions"


def test_voice_learning_web_workflow_requires_rights_and_author_confirmation(
    tmp_path,
):
    application = create_app(_settings(tmp_path))
    with TestClient(application) as client:
        page = client.get("/register")
        response = client.post(
            "/register",
            data={
                "username": "声纹作者",
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
                "title": "声纹测试小说",
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
        page = client.get(workbench_url)
        rejected = client.post(
            f"{project_url}/voice/suggestions",
            data={
                "sample_title": "我的样章",
                "sample_text": _sample_text(),
                "author_intent": "",
                "rights_confirmed": "",
                "csrf": _csrf(page.text),
            },
            follow_redirects=False,
        )
        assert rejected.status_code == 303
        assert "error=" in rejected.headers["location"]

        page = client.get(workbench_url)
        response = client.post(
            f"{project_url}/voice/suggestions",
            data={
                "sample_title": "我的样章",
                "sample_text": _sample_text(),
                "author_intent": "不要使用：命运的齿轮",
                "rights_confirmed": "yes",
                "csrf": _csrf(page.text),
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        suggestion_url = response.headers["location"]
        assert suggestion_url.startswith("/voice-suggestions/")

        suggestion_page = None
        for _ in range(100):
            suggestion_page = client.get(suggestion_url)
            if "声纹提取结论" in suggestion_page.text:
                break
            time.sleep(0.02)
        assert suggestion_page is not None
        assert "声纹提取结论" in suggestion_page.text
        assert "条证据已逐字核对" in suggestion_page.text
        apply_response = client.post(
            f"{suggestion_url}/apply",
            data={
                "narration_rules": "紧贴人物当前能够观察的信息。",
                "sentence_rhythm": "动作推进用短句，转折处改变句长。",
                "dialogue_voice": "",
                "sensory_palette": "潮湿纸张、旧金属和港口声音。",
                "metaphor_policy": "低密度。",
                "allowed_omissions": "动作能够说明时不再解释情绪。",
                "preferred_patterns": "动作先于解释",
                "banned_expressions": "命运的齿轮",
                "author_notes": "已审核作者样章证据。",
                "action": "apply_confirm",
                "csrf": _csrf(suggestion_page.text),
            },
            follow_redirects=False,
        )
        assert apply_response.status_code == 303
        assert (
            "view=archive&archive_tab=creative&settings_tab=style&saved=true"
            in apply_response.headers["location"]
        )
        workspace = client.get(apply_response.headers["location"])
        assert "已保存" in workspace.text
        assert "叙事与文风" in workspace.text
        assert "动作先于解释" in workspace.text
        assert "已确认" in workspace.text
