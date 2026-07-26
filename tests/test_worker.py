import asyncio
from pathlib import Path

from app.assistant_chat import MockAssistantChatModel
from app.config import Settings
from app.credentials import CredentialCipher, key_hint
from app.db import Database
from app.deepseek import MockAnalyzer
from app.memory_extraction import MockMemoryExtractor
from app.planning_ai import MockChapterPlanner
from app.reader_planner import MockReaderPlanner
from app.security import hash_password
from app.style_editor import MockStyleEditor
from app.worker import AnalysisWorker
from app.writing import MockWriter


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


def test_worker_uses_owning_users_decrypted_api_key(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    user_id = database.create_user("worker-user", hash_password("password-123"))
    cipher = CredentialCipher(settings.credential_secret)
    raw_key = "sk-worker-personal-2468"
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
            self.model = personal_settings.deepseek_model
            seen["api_key"] = personal_settings.deepseek_api_key
            seen["model"] = personal_settings.deepseek_model
            seen["thinking"] = personal_settings.deepseek_thinking
            seen["effort"] = personal_settings.deepseek_reasoning_effort

    monkeypatch.setattr("app.worker.DeepSeekAnalyzer", FakePersonalAnalyzer)

    async def scenario():
        worker = AnalysisWorker(
            database,
            MockAnalyzer(),
            MockWriter(),
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
    }


def test_writing_worker_uses_owning_users_decrypted_api_key(
    tmp_path, monkeypatch
):
    settings = make_settings(tmp_path)
    database = Database(settings.database_path)
    database.initialize()
    user_id = database.create_user("novel-writer", hash_password("password-123"))
    cipher = CredentialCipher(settings.credential_secret)
    raw_key = "sk-novel-writer-1357"
    database.upsert_api_credential(
        user_id=user_id,
        encrypted_key=cipher.encrypt(raw_key),
        key_hint=key_hint(raw_key),
        model="deepseek-v4-flash",
    )
    seen = {}

    class FakePersonalWriter(MockWriter):
        provider = "deepseek"

        def __init__(self, personal_settings):
            self.model = personal_settings.deepseek_model
            seen["api_key"] = personal_settings.deepseek_api_key
            seen["model"] = personal_settings.deepseek_model

    monkeypatch.setattr("app.worker.DeepSeekWriter", FakePersonalWriter)

    async def scenario():
        worker = AnalysisWorker(
            database,
            MockAnalyzer(),
            MockWriter(),
            settings.secret_key,
            settings,
            cipher,
            poll_seconds=0.01,
        )
        return await worker._write(
            item={
                "user_id": user_id,
                "provider": "deepseek",
                "model": "deepseek-v4-flash",
                "credential_source": "personal",
                "operation": "draft",
                "instruction": "",
            },
            context={
                "chapter": {
                    "title": "第一章",
                    "outline": "主角收到来信",
                }
            },
            current_content="",
            previous_content="",
        )

    response = asyncio.run(scenario())
    assert response.content
    assert seen == {
        "api_key": raw_key,
        "model": "deepseek-v4-flash",
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
    seen = {}

    class FakePersonalChatModel(MockAssistantChatModel):
        provider = "deepseek"

        def __init__(self, personal_settings):
            self.model = personal_settings.deepseek_model
            seen["api_key"] = personal_settings.deepseek_api_key
            seen["model"] = personal_settings.deepseek_model
            seen["thinking"] = personal_settings.deepseek_thinking
            seen["effort"] = personal_settings.deepseek_reasoning_effort

    monkeypatch.setattr(
        "app.worker.DeepSeekAssistantChatModel",
        FakePersonalChatModel,
    )

    async def scenario():
        worker = AnalysisWorker(
            database,
            MockAnalyzer(),
            MockWriter(),
            settings.secret_key,
            settings,
            cipher,
            assistant_chat_model=MockAssistantChatModel(),
            poll_seconds=0.01,
        )
        return await worker._reply_assistant_chat(
            item={
                "user_id": user_id,
                "provider": "deepseek",
                "model": "deepseek-v4-pro",
                "credential_source": "personal",
            },
            payload={
                "context": {"scope": "novel_project"},
                "sources": [],
                "history": [],
                "question": "如何安排开场？",
                "selected_quote": "",
            },
        )

    response = asyncio.run(scenario())
    assert response.result.answer
    assert seen == {
        "api_key": raw_key,
        "model": "deepseek-v4-pro",
        "thinking": True,
        "effort": "high",
    }


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
            self.model = personal_settings.deepseek_model
            seen["api_key"] = personal_settings.deepseek_api_key
            seen["model"] = personal_settings.deepseek_model
            seen["thinking"] = personal_settings.deepseek_thinking
            seen["effort"] = personal_settings.deepseek_reasoning_effort

    monkeypatch.setattr(
        "app.worker.DeepSeekMemoryExtractor", FakePersonalExtractor
    )

    async def scenario():
        worker = AnalysisWorker(
            database,
            MockAnalyzer(),
            MockWriter(),
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
            self.model = personal_settings.deepseek_model
            seen["api_key"] = personal_settings.deepseek_api_key
            seen["model"] = personal_settings.deepseek_model
            seen["thinking"] = personal_settings.deepseek_thinking
            seen["effort"] = personal_settings.deepseek_reasoning_effort

    monkeypatch.setattr(
        "app.worker.DeepSeekChapterPlanner", FakePersonalPlanner
    )

    async def scenario():
        worker = AnalysisWorker(
            database,
            MockAnalyzer(),
            MockWriter(),
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
            self.model = personal_settings.deepseek_model
            seen["api_key"] = personal_settings.deepseek_api_key
            seen["model"] = personal_settings.deepseek_model
            seen["thinking"] = personal_settings.deepseek_thinking
            seen["effort"] = personal_settings.deepseek_reasoning_effort

    monkeypatch.setattr(
        "app.worker.DeepSeekStyleEditor", FakePersonalStyleEditor
    )

    async def scenario():
        worker = AnalysisWorker(
            database,
            MockAnalyzer(),
            MockWriter(),
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
            self.model = personal_settings.deepseek_model
            seen["api_key"] = personal_settings.deepseek_api_key
            seen["model"] = personal_settings.deepseek_model
            seen["thinking"] = personal_settings.deepseek_thinking
            seen["effort"] = personal_settings.deepseek_reasoning_effort

    monkeypatch.setattr(
        "app.worker.DeepSeekReaderPlanner",
        FakePersonalReaderPlanner,
    )

    async def scenario():
        worker = AnalysisWorker(
            database,
            MockAnalyzer(),
            MockWriter(),
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
