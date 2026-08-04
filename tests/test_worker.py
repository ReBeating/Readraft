import asyncio
from pathlib import Path

from app.agent_model import MockAgentModel
from app.config import Settings
from app.credentials import CredentialCipher, key_hint
from app.db import Database
from app.model_client import MockAnalyzer
from app.memory_extraction import MockMemoryExtractor
from app.planning_ai import MockChapterPlanner
from app.reader_planner import MockReaderPlanner
from app.security import hash_password
from app.style_editor import MockStyleEditor
from app.worker import AnalysisWorker


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


def test_personal_model_strategy_routes_two_models_and_three_modes(tmp_path):
    settings = make_settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    user_id = database.create_user(
        "strategy-user", hash_password("password-123")
    )
    cipher = CredentialCipher(settings.credential_secret)
    database.upsert_api_credential(
        user_id=user_id,
        encrypted_key=cipher.encrypt("sk-strategy-user-2468"),
        key_hint="sk-••••2468",
        model="deepseek-v4-flash",
        models=["deepseek-v4-flash", "deepseek-v4-pro"],
    )
    database.upsert_model_routing_preferences(
        user_id=user_id,
        fast_provider="deepseek",
        fast_model="deepseek-v4-flash",
        quality_provider="deepseek",
        quality_model="deepseek-v4-pro",
        default_quality_mode="standard",
    )
    worker = AnalysisWorker(
        database,
        MockAnalyzer(),
        settings.secret_key,
        settings,
        cipher,
        poll_seconds=0.01,
    )
    base_item = {
        "user_id": user_id,
        "provider": "deepseek",
        "model": "deepseek-v4-flash",
        "credential_source": "personal",
    }

    async def settings_for(mode, task):
        return await worker._personal_model_settings(
            {**base_item, "quality_mode": mode},
            task,
        )

    low = asyncio.run(settings_for("low", "deep"))
    standard_discussion = asyncio.run(
        settings_for("standard", "discussion")
    )
    standard_writing = asyncio.run(
        settings_for("standard", "reasoning")
    )
    standard_planning = asyncio.run(
        settings_for("standard", "deep")
    )
    maximum = asyncio.run(settings_for("max", "fast"))

    assert (
        low.model_name,
        low.model_thinking,
    ) == ("deepseek-v4-flash", False)
    assert (
        standard_discussion.model_name,
        standard_discussion.model_thinking,
        standard_discussion.model_reasoning_effort,
    ) == ("deepseek-v4-flash", True, "high")
    assert (
        standard_writing.model_name,
        standard_writing.model_thinking,
        standard_writing.model_reasoning_effort,
    ) == ("deepseek-v4-pro", True, "high")
    assert (
        standard_planning.model_name,
        standard_planning.model_thinking,
        standard_planning.model_reasoning_effort,
    ) == ("deepseek-v4-pro", True, "max")
    assert (
        maximum.model_name,
        maximum.model_thinking,
        maximum.model_reasoning_effort,
    ) == ("deepseek-v4-pro", True, "max")


def test_worker_uses_owning_users_decrypted_api_key(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    user_id = database.create_user("worker-user", hash_password("password-123"))
    cipher = CredentialCipher(settings.credential_secret)
    raw_key = "sk-worker-personal-2468"
    database.upsert_model_adapter_prompt(
        user_id, "受限时保留事件因果并降低细节。"
    )
    database.upsert_api_credential(
        user_id=user_id,
        encrypted_key=cipher.encrypt(raw_key),
        key_hint=key_hint(raw_key),
        model="deepseek-v4-pro",
    )
    database.upsert_api_credential(
        user_id=user_id,
        provider="openai_compatible",
        base_url="https://gateway.example.com/v1",
        encrypted_key=cipher.encrypt("compatible-worker-key"),
        key_hint="••••-key",
        model="other-default-model",
    )
    seen = {}

    class FakePersonalAnalyzer(MockAnalyzer):
        provider = "deepseek"

        def __init__(self, personal_settings):
            self.model = personal_settings.model_name
            seen["api_key"] = personal_settings.model_api_key
            seen["model"] = personal_settings.model_name
            seen["thinking"] = personal_settings.model_thinking
            seen["effort"] = personal_settings.model_reasoning_effort
            seen["adapter"] = personal_settings.model_adapter_prompt

    monkeypatch.setattr("app.worker.ProviderAnalyzer", FakePersonalAnalyzer)

    async def scenario():
        worker = AnalysisWorker(
            database,
            MockAnalyzer(),
            settings.secret_key,
            settings,
            cipher,
            poll_seconds=0.01,
        )
        return await worker._analyze(
            {
                "user_id": user_id,
                "provider": "deepseek",
                "model": "deepseek-v4-pro",
                "credential_source": "personal",
                "chapter_title": "第一章",
            },
            "这是用于测试的章节正文。",
        )

    response = asyncio.run(scenario())
    assert response.result.chapter_title == "第一章"
    assert seen == {
        "api_key": raw_key,
        "model": "deepseek-v4-pro",
        "thinking": True,
        "effort": "high",
        "adapter": "受限时保留事件因果并降低细节。",
    }


def test_chat_worker_uses_owning_users_decrypted_api_key(
    tmp_path, monkeypatch
):
    settings = make_settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    user_id = database.create_user(
        "chat-worker", hash_password("password-123")
    )
    cipher = CredentialCipher(settings.credential_secret)
    raw_key = "sk-chat-worker-9753"
    database.upsert_api_credential(
        user_id=user_id,
        encrypted_key=cipher.encrypt(raw_key),
        key_hint=key_hint(raw_key),
        model="deepseek-v4-pro",
    )
    seen = []

    class FakePersonalChatModel(MockAgentModel):
        provider = "deepseek"

        def __init__(self, personal_settings):
            self.model = personal_settings.model_name
            seen.append(
                {
                    "api_key": personal_settings.model_api_key,
                    "model": personal_settings.model_name,
                    "thinking": personal_settings.model_thinking,
                    "effort": personal_settings.model_reasoning_effort,
                }
            )

    monkeypatch.setattr(
        "app.worker.ProviderAgentModel",
        FakePersonalChatModel,
    )

    async def scenario():
        worker = AnalysisWorker(
            database,
            MockAnalyzer(),
            settings.secret_key,
            settings,
            cipher,
            assistant_chat_model=MockAgentModel(),
            poll_seconds=0.01,
        )
        project_id = "chat-worker-project"
        database.create_novel_project(
            user_id=user_id,
            project_id=project_id,
            title="凭据路由测试",
            genre="悬疑",
            premise="测试对话任务只使用作品所有者的凭据。",
            world_setting="",
            style_guide="",
            point_of_view="第三人称限知",
            target_chapter_chars=3000,
        )
        conversation_id = worker.assistant_chat_service.create_conversation(
            user_id=user_id,
            scope_type="project",
            title="讨论开场",
            project_id=project_id,
        )
        message_id = worker.assistant_chat_service.queue_message(
            user_id=user_id,
            conversation_id=conversation_id,
            question="如何安排开场？",
            provider="deepseek",
            model="deepseek-v4-pro",
            credential_source="personal",
            agent_role="advisor",
        )
        item = worker.assistant_chat_service.claim_next_message()
        assert item and item["id"] == message_id
        return await worker._reply_assistant_chat(
            item=item,
            payload=worker.assistant_chat_service.build_job_payload(item),
        )

    response = asyncio.run(scenario())
    assert response.result.answer
    assert seen
    assert all(item["api_key"] == raw_key for item in seen)
    assert {item["model"] for item in seen} == {"deepseek-v4-pro"}
    assert any(item["thinking"] for item in seen)
    assert any(item["effort"] == "high" for item in seen)


def test_memory_worker_uses_owning_users_decrypted_api_key(
    tmp_path, monkeypatch
):
    settings = make_settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    user_id = database.create_user(
        "memory-worker", hash_password("password-123")
    )
    cipher = CredentialCipher(settings.credential_secret)
    raw_key = "sk-memory-worker-8642"
    database.upsert_api_credential(
        user_id=user_id,
        encrypted_key=cipher.encrypt(raw_key),
        key_hint=key_hint(raw_key),
        model="deepseek-v4-pro",
    )
    seen = {}

    class FakePersonalExtractor(MockMemoryExtractor):
        provider = "deepseek"

        def __init__(self, personal_settings):
            self.model = personal_settings.model_name
            seen["api_key"] = personal_settings.model_api_key
            seen["model"] = personal_settings.model_name
            seen["thinking"] = personal_settings.model_thinking
            seen["effort"] = personal_settings.model_reasoning_effort

    monkeypatch.setattr(
        "app.worker.ProviderMemoryExtractor", FakePersonalExtractor
    )

    async def scenario():
        worker = AnalysisWorker(
            database,
            MockAnalyzer(),
            settings.secret_key,
            settings,
            cipher,
            poll_seconds=0.01,
        )
        return await worker._extract_memory(
            item={
                "user_id": user_id,
                "provider": "deepseek",
                "model": "deepseek-v4-pro",
                "credential_source": "personal",
            },
            context={
                "chapter": {
                    "title": "第一章",
                }
            },
            chapter_text="用于提取故事记忆的正文。",
        )

    response = asyncio.run(scenario())
    assert response.result.chapter_summary
    assert seen == {
        "api_key": raw_key,
        "model": "deepseek-v4-pro",
        "thinking": False,
        "effort": "high",
    }


def test_planner_worker_uses_owning_users_decrypted_api_key(
    tmp_path, monkeypatch
):
    settings = make_settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    user_id = database.create_user(
        "planner-worker", hash_password("password-123")
    )
    cipher = CredentialCipher(settings.credential_secret)
    raw_key = "sk-planner-worker-9753"
    database.upsert_api_credential(
        user_id=user_id,
        encrypted_key=cipher.encrypt(raw_key),
        key_hint=key_hint(raw_key),
        model="deepseek-v4-pro",
    )
    seen = {}

    class FakePersonalPlanner(MockChapterPlanner):
        provider = "deepseek"

        def __init__(self, personal_settings):
            self.model = personal_settings.model_name
            seen["api_key"] = personal_settings.model_api_key
            seen["model"] = personal_settings.model_name
            seen["thinking"] = personal_settings.model_thinking
            seen["effort"] = personal_settings.model_reasoning_effort

    monkeypatch.setattr(
        "app.worker.ProviderChapterPlanner", FakePersonalPlanner
    )

    async def scenario():
        worker = AnalysisWorker(
            database,
            MockAnalyzer(),
            settings.secret_key,
            settings,
            cipher,
            poll_seconds=0.01,
        )
        return await worker._plan_chapter(
            item={
                "user_id": user_id,
                "provider": "deepseek",
                "model": "deepseek-v4-pro",
                "credential_source": "personal",
                "instruction": "",
            },
            context={
                "chapter": {
                    "title": "第一章",
                    "outline": "主角收到来信",
                    "key_points": "核对邮戳",
                    "target_chapter_chars": 3000,
                },
                "characters": [{"name": "林岚"}],
            },
        )

    response = asyncio.run(scenario())
    assert len(response.result.scenes) == 2
    assert seen == {
        "api_key": raw_key,
        "model": "deepseek-v4-pro",
        "thinking": True,
        "effort": "max",
    }


def test_style_editor_uses_owning_users_decrypted_api_key(
    tmp_path, monkeypatch
):
    settings = make_settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    user_id = database.create_user(
        "style-worker", hash_password("password-123")
    )
    cipher = CredentialCipher(settings.credential_secret)
    raw_key = "sk-style-worker-4680"
    database.upsert_api_credential(
        user_id=user_id,
        encrypted_key=cipher.encrypt(raw_key),
        key_hint=key_hint(raw_key),
        model="deepseek-v4-pro",
    )
    seen = {}

    class FakePersonalStyleEditor(MockStyleEditor):
        provider = "deepseek"

        def __init__(self, personal_settings):
            self.model = personal_settings.model_name
            seen["api_key"] = personal_settings.model_api_key
            seen["model"] = personal_settings.model_name
            seen["thinking"] = personal_settings.model_thinking
            seen["effort"] = personal_settings.model_reasoning_effort

    monkeypatch.setattr(
        "app.worker.ProviderStyleEditor", FakePersonalStyleEditor
    )

    async def scenario():
        worker = AnalysisWorker(
            database,
            MockAnalyzer(),
            settings.secret_key,
            settings,
            cipher,
            style_editor=MockStyleEditor(),
            poll_seconds=0.01,
        )
        editor, close_editor = await worker._style_editor_for_item(
            {
                "user_id": user_id,
                "provider": "deepseek",
                "model": "deepseek-v4-pro",
                "credential_source": "personal",
            }
        )
        try:
            return editor.provider
        finally:
            if close_editor:
                await editor.close()

    assert asyncio.run(scenario()) == "deepseek"
    assert seen == {
        "api_key": raw_key,
        "model": "deepseek-v4-pro",
        "thinking": True,
        "effort": "high",
    }


def test_reader_planner_uses_owning_users_decrypted_api_key(
    tmp_path, monkeypatch
):
    settings = make_settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    user_id = database.create_user(
        "reader-planner-worker", hash_password("password-123")
    )
    cipher = CredentialCipher(settings.credential_secret)
    raw_key = "sk-reader-planner-worker-8642"
    database.upsert_api_credential(
        user_id=user_id,
        encrypted_key=cipher.encrypt(raw_key),
        key_hint=key_hint(raw_key),
        model="deepseek-v4-pro",
    )
    seen = {}

    class FakePersonalReaderPlanner(MockReaderPlanner):
        provider = "deepseek"

        def __init__(self, personal_settings):
            self.model = personal_settings.model_name
            seen["api_key"] = personal_settings.model_api_key
            seen["model"] = personal_settings.model_name
            seen["thinking"] = personal_settings.model_thinking
            seen["effort"] = personal_settings.model_reasoning_effort

    monkeypatch.setattr(
        "app.worker.ProviderReaderPlanner",
        FakePersonalReaderPlanner,
    )

    async def scenario():
        worker = AnalysisWorker(
            database,
            MockAnalyzer(),
            settings.secret_key,
            settings,
            cipher,
            poll_seconds=0.01,
        )
        context = {
            "request": {
                "raw_text": "希望关系线慢一点。",
                "impact_scope": "next_three",
            },
            "current_position": 1,
            "planning_horizon": 20,
            "future_chapters": [],
        }
        return await worker._plan_reader_request(
            item={
                "user_id": user_id,
                "provider": "deepseek",
                "model": "deepseek-v4-pro",
                "credential_source": "personal",
            },
            context=context,
        )

    response = asyncio.run(scenario())
    assert len(response.result.alternatives) == 3
    assert seen == {
        "api_key": raw_key,
        "model": "deepseek-v4-pro",
        "thinking": True,
        "effort": "high",
    }
