import json
import hashlib
from pathlib import Path

import pytest

from app.agent_capabilities import (
    WRITE_CHAPTER,
    CREATE_TECHNIQUE_CARD,
    MANAGE_CHAPTERS,
    MANAGE_NOTES,
    PROPOSE_SETTINGS_PATCH,
    PROPOSE_STORY_PLAN,
    READ_CHAPTER,
    READ_PROJECT,
    RUN_BOUNDED_TASK,
)
from app.agent_actions import ComposeArguments, available_agent_actions
from app.agent_workspace import AgentWorkspace
from app.chapter_splitter import split_chapters
from app.assistant_chat_service import AssistantChatService
from app.assistant_chat_schema import AssistantChatResponse, AssistantChatResult
from app.db import Database
from app.security import hash_password


def seed_workspace(tmp_path: Path) -> tuple[Database, int, dict]:
    database = Database(tmp_path / "app.db")
    database.initialize()
    user_id = database.create_user(
        "workspace-writer", hash_password("password-123")
    )
    project_id = "p" * 32
    chapter_id = "c" * 32
    database.create_novel_project(
        user_id=user_id,
        project_id=project_id,
        title="雾港来信",
        genre="悬疑",
        premise="记者收到不可能出现的旧信。",
        world_setting="冬季海港。",
        style_guide="克制，以动作承担情绪。",
        point_of_view="第三人称限知",
        target_chapter_chars=3000,
    )
    chapter_dir = tmp_path / "chapters" / chapter_id
    chapter_dir.mkdir(parents=True)
    content_path = chapter_dir / "content.txt"
    content_path.write_text("林岚拆开旧信。\n邮戳还是湿的。", encoding="utf-8")
    database.add_novel_chapter(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        title="第一章 迟到的信",
        outline="林岚核对旧信。",
        key_points="邮戳日期异常",
        content_path=content_path,
    )
    initial_content = content_path.read_text(encoding="utf-8")
    initial_version_path = chapter_dir / "versions" / "initial.txt"
    initial_version_path.parent.mkdir(parents=True)
    initial_version_path.write_text(initial_content, encoding="utf-8")
    initial_version_id = database.record_manual_chapter_version(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        version_path=initial_version_path,
        char_count=len(initial_content),
        content_hash=hashlib.sha256(initial_content.encode()).hexdigest(),
        change_summary="建立初始 main HEAD",
        expected_old_head_version_id="",
    )
    assert initial_version_id
    context = {
        "scope": "novel_chapter",
        "project": {
            "id": project_id,
            "title": "雾港来信",
            "genre": "悬疑",
            "premise": "记者收到不可能出现的旧信。",
            "world_setting": "冬季海港。",
            "style_guide": "克制，以动作承担情绪。",
            "point_of_view": "第三人称限知",
        },
        "chapter": {"id": chapter_id, "position": 1},
        "agent": {
            "role": "writer",
            "capabilities": [
                READ_PROJECT,
                READ_CHAPTER,
                WRITE_CHAPTER,
                PROPOSE_SETTINGS_PATCH,
                RUN_BOUNDED_TASK,
            ],
        },
        "structured_settings": {
            "schema_version": 1,
            "characters": [
                {
                    "id": "character-1",
                    "name": "林岚",
                    "role": "记者",
                    "traits": "冷静",
                }
            ],
        },
    }
    return database, user_id, context


def test_workspace_tools_and_agent_actions_have_separate_registries(tmp_path):
    database, user_id, context = seed_workspace(tmp_path)
    workspace = AgentWorkspace(
        database,
        user_id=user_id,
        context=context,
        sources=[],
    )

    workspace_tool_names = {
        item.name for item in workspace.available_workspace_tools()
    }
    action_names = {
        item.name
        for item in available_agent_actions(
            capabilities=workspace.capabilities,
            main_writable=workspace.main_writable,
            has_writable_chapter=workspace.has_writable_chapter,
        )
    }
    listing = workspace.execute_tool(
        "glob", {"pattern": "book/manuscript/*"}
    )
    reading = workspace.execute_tool(
        "read", {"path": "book/manuscript/chapters/001.md"}
    )

    assert {"glob", "read", "grep", "edit", "write", "patch"} <= (
        workspace_tool_names
    )
    assert "compose" not in workspace_tool_names
    assert action_names == {"compose", "task"}
    assert any(
        item["path"] == "book/manuscript/chapters/001.md"
        for item in listing.result["matches"]
    )
    assert "1: 林岚拆开旧信。" in reading.result["content"]
    assert reading.result["writable"] is True
    assert len(reading.result["revision"]) == 64


def test_compose_accepts_a_small_target_for_bounded_append():
    arguments = ComposeArguments.model_validate(
        {
            "path": "book/manuscript/chapters/001.md",
            "instruction": "在章末追加一小段环境描写",
            "expected_revision": "a" * 64,
            "mode": "append",
            "target_chars": 100,
        }
    )

    assert arguments.target_chars == 100


def test_workspace_reads_main_head_instead_of_mutable_chapter_cache(tmp_path):
    database, user_id, context = seed_workspace(tmp_path)
    project_id = context["project"]["id"]
    chapter_id = context["chapter"]["id"]
    chapter = database.get_novel_chapter(user_id, project_id, chapter_id)
    assert chapter
    Path(str(chapter["content_path"])).write_text(
        "这是故意制造的过期正文缓存。",
        encoding="utf-8",
    )

    workspace = AgentWorkspace(
        database,
        user_id=user_id,
        context=context,
        sources=[],
    )
    reading = workspace.execute_tool(
        "read", {"path": "book/manuscript/chapters/001.md"}
    )
    assert "林岚拆开旧信" in reading.result["content"]
    assert "过期正文缓存" not in reading.result["content"]


def test_specialist_task_packet_only_contains_explicit_bounded_resources(
    tmp_path,
):
    database, user_id, context = seed_workspace(tmp_path)
    workspace = AgentWorkspace(
        database,
        user_id=user_id,
        context=context,
        sources=[],
    )

    packet = workspace.build_specialist_task_packet(
        paths=[
            "book/manuscript/chapters/001.md",
            "book/settings/core.json",
        ]
    )

    assert packet.result["resource_count"] == 2
    assert [
        resource["path"] for resource in packet.result["resources"]
    ] == [
        "book/manuscript/chapters/001.md",
        "book/settings/core.json",
    ]
    assert "林岚拆开旧信" in packet.result["resources"][0]["content"]
    assert len(packet.accessed_sources) == 2
    with pytest.raises(ValueError, match="不存在"):
        workspace.build_specialist_task_packet(
            paths=["book/private/secret.md"]
        )


def test_concept_search_uses_model_supplied_semantic_expansion(tmp_path):
    database, user_id, context = seed_workspace(tmp_path)
    workspace = AgentWorkspace(
        database,
        user_id=user_id,
        context=context,
        sources=[],
    )

    result = workspace.execute_tool(
        "search",
        {
            "query": "信件出现了时间矛盾",
            "related_concepts": ["湿邮戳", "邮戳日期"],
            "path": "book/manuscript",
        },
    )

    assert result.result["engine"] == "model_expanded_sparse_cjk"
    assert result.result["matched_count"] >= 1
    prose_matches = [
        match
        for match in result.result["matches"]
        if match["path"] == "book/manuscript/chapters/001.md"
    ]
    assert prose_matches
    assert "邮戳还是湿的" in prose_matches[0]["excerpt"]


def test_workspace_write_cannot_bypass_compose_action(tmp_path):
    database, user_id, context = seed_workspace(tmp_path)
    workspace = AgentWorkspace(
        database,
        user_id=user_id,
        context=context,
        sources=[],
    )
    path = "book/manuscript/chapters/001.md"
    reading = workspace.execute_tool("read", {"path": path})

    with pytest.raises(PermissionError, match="compose"):
        workspace.execute_tool(
            "write",
            {
                "path": path,
                "content": "试图绕过章节写作动作。",
                "expected_revision": reading.result["revision"],
            },
        )


def test_chapter_metadata_is_a_revision_checked_writable_resource(tmp_path):
    database, user_id, context = seed_workspace(tmp_path)
    context["agent"]["capabilities"].append(MANAGE_CHAPTERS)
    workspace = AgentWorkspace(
        database,
        user_id=user_id,
        context=context,
        sources=[],
    )
    path = "book/manuscript/chapters/001.meta.json"
    reading = workspace.execute_tool("read", {"path": path})

    execution = workspace.execute_tool(
        "edit",
        {
            "path": path,
            "old_string": "第一章 迟到的信",
            "new_string": "第一章 湿邮戳",
            "expected_revision": reading.result["revision"],
        },
    )

    assert execution.chapter_patch is not None
    assert execution.chapter_patch.edits[0].title == "第一章 湿邮戳"
    assert execution.draft is None
    assert execution.settings_patch is None


def test_completed_chapter_patch_atomically_updates_metadata_and_order(
    tmp_path,
):
    database, user_id, context = seed_workspace(tmp_path)
    project_id = context["project"]["id"]
    second_id = "d" * 32
    second_dir = tmp_path / "chapters" / second_id
    second_dir.mkdir(parents=True)
    second_path = second_dir / "content.txt"
    second_path.write_text("林岚走向灯塔。", encoding="utf-8")
    database.add_novel_chapter(
        user_id=user_id,
        project_id=project_id,
        chapter_id=second_id,
        title="第二章 灯塔",
        outline="林岚抵达灯塔。",
        key_points="灯塔提前亮起",
        content_path=second_path,
    )
    service = AssistantChatService(
        database,
        tmp_path / "novels",
        tmp_path / "documents",
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="chapter",
        title="调整章节顺序",
        project_id=project_id,
        novel_chapter_id=context["chapter"]["id"],
    )
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="把第二章移动到第一章，并改名为开场。",
        provider="mock",
        model="mock",
        credential_source="default",
        agent_role="planner",
        auto_commit=True,
    )
    claimed = service.claim_next_message()
    assert claimed and claimed["id"] == message_id
    payload = service.build_job_payload(claimed)
    workspace = AgentWorkspace(
        database,
        user_id=user_id,
        context=payload["context"],
        sources=payload["sources"],
    )
    metadata_path = "book/manuscript/chapters/002.meta.json"
    reading = workspace.execute_tool("read", {"path": metadata_path})
    execution = workspace.execute_tool(
        "patch",
        {
            "targets": [
                {
                    "path": metadata_path,
                    "expected_revision": reading.result["revision"],
                    "replacements": [
                        {
                            "old_string": '"position": 2',
                            "new_string": '"position": 1',
                        },
                        {
                            "old_string": "第二章 灯塔",
                            "new_string": "第一章 开场",
                        },
                    ],
                }
            ]
        },
    )
    response = AssistantChatResponse(
        result=AssistantChatResult(
            answer="已调整章节顺序和标题。",
            chapter_patch=execution.chapter_patch,
        ),
        raw_response="",
        input_tokens=0,
        output_tokens=0,
        provider="mock",
        model="mock",
    )

    assert service.complete_message(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        response=response,
    )
    chapters = database.list_novel_chapters(user_id, project_id)
    assert [str(chapter["id"]) for chapter in chapters] == [
        second_id,
        context["chapter"]["id"],
    ]
    assert [int(chapter["position"]) for chapter in chapters] == [1, 2]
    assert chapters[0]["title"] == "第一章 开场"
    message = service.get_message_stream_state(
        user_id=user_id,
        message_id=message_id,
    )
    assert message
    stored = service.get_message(user_id=user_id, message_id=message_id)
    assert stored
    assert stored["response"]["chapter_patch_status"] == "applied"


def test_explicit_chapter_delete_rebinds_conversation_and_keeps_recovery(
    tmp_path,
):
    database, user_id, context = seed_workspace(tmp_path)
    project_id = context["project"]["id"]
    second_id = "e" * 32
    second_dir = tmp_path / "chapters" / second_id
    second_dir.mkdir(parents=True)
    second_path = second_dir / "content.txt"
    second_path.write_text("这一章将被删除。", encoding="utf-8")
    database.add_novel_chapter(
        user_id=user_id,
        project_id=project_id,
        chapter_id=second_id,
        title="第二章 待删除",
        outline="测试删除",
        key_points="",
        content_path=second_path,
    )
    novels_dir = tmp_path / "novels"
    service = AssistantChatService(
        database,
        novels_dir,
        tmp_path / "documents",
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="chapter",
        title="删除章节",
        project_id=project_id,
        novel_chapter_id=second_id,
    )
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="请删除第二章《待删除》。",
        provider="mock",
        model="mock",
        credential_source="default",
        agent_role="planner",
        auto_commit=True,
    )
    claimed = service.claim_next_message()
    assert claimed and claimed["id"] == message_id
    payload = service.build_job_payload(claimed)
    workspace = AgentWorkspace(
        database,
        user_id=user_id,
        context=payload["context"],
        sources=payload["sources"],
    )
    metadata_path = "book/manuscript/chapters/002.meta.json"
    reading = workspace.execute_tool("read", {"path": metadata_path})
    execution = workspace.execute_tool(
        "delete",
        {
            "path": metadata_path,
            "expected_revision": reading.result["revision"],
            "rationale": "作者明确要求删除第二章",
        },
    )
    response = AssistantChatResponse(
        result=AssistantChatResult(
            answer="已删除第二章，并保留恢复副本。",
            chapter_patch=execution.chapter_patch,
        ),
        raw_response="",
        input_tokens=0,
        output_tokens=0,
        provider="mock",
        model="mock",
    )

    assert service.complete_message(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        response=response,
    )
    chapters = database.list_novel_chapters(user_id, project_id)
    assert len(chapters) == 1
    assert int(chapters[0]["position"]) == 1
    assert not second_dir.exists()
    recovery = (
        novels_dir
        / str(user_id)
        / project_id
        / ".chapter-recovery"
        / f"{message_id}-{second_id}"
    )
    assert (recovery / "files" / "content.txt").read_text(
        encoding="utf-8"
    ) == "这一章将被删除。"
    conversation = service.get_conversation(
        user_id=user_id,
        conversation_id=conversation_id,
    )
    assert conversation
    assert conversation["scope_type"] == "project"
    assert conversation["novel_chapter_id"] is None
    stored = service.get_message(user_id=user_id, message_id=message_id)
    assert stored
    assert stored["response"]["chapter_patch_status"] == "applied"


def test_workspace_delete_builds_explicit_structured_setting_removal(tmp_path):
    database, user_id, context = seed_workspace(tmp_path)
    workspace = AgentWorkspace(
        database,
        user_id=user_id,
        context=context,
        sources=[],
    )
    path = "book/settings/characters/character-1.json"
    reading = workspace.execute_tool("read", {"path": path})

    execution = workspace.execute_tool(
        "delete",
        {
            "path": path,
            "expected_revision": reading.result["revision"],
            "rationale": "作者明确要求删除人物",
        },
    )

    assert execution.settings_patch is not None
    edits = execution.settings_patch.structured_edits or []
    assert len(edits) == 1
    assert edits[0].action == "delete"
    assert edits[0].target_id == "character-1"


def test_author_requested_note_is_persisted_and_returns_to_workspace(tmp_path):
    database, user_id, context = seed_workspace(tmp_path)
    project_id = context["project"]["id"]
    service = AssistantChatService(
        database,
        tmp_path / "novels",
        tmp_path / "documents",
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="project",
        title="保存构思",
        project_id=project_id,
    )
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="把刚才关于灯塔线索的专项分析保存成作者笔记。",
        provider="mock",
        model="mock",
        credential_source="default",
        agent_role="planner",
        auto_commit=True,
    )
    claimed = service.claim_next_message()
    assert claimed and claimed["id"] == message_id
    payload = service.build_job_payload(claimed)
    assert MANAGE_NOTES in payload["context"]["agent"]["capabilities"]
    workspace = AgentWorkspace(
        database,
        user_id=user_id,
        context=payload["context"],
        sources=payload["sources"],
    )
    execution = workspace.execute_tool(
        "write",
        {
            "path": "book/notes/author/lighthouse-clue.md",
            "content": "# 灯塔线索分析\n\n让灯塔提前亮起成为可验证的时间矛盾。",
            "expected_revision": "new",
            "rationale": "作者明确要求保存专项分析",
        },
    )
    response = AssistantChatResponse(
        result=AssistantChatResult(
            answer="已保存灯塔线索分析。",
            note_patch=execution.note_patch,
        ),
        raw_response="",
        input_tokens=0,
        output_tokens=0,
        provider="mock",
        model="mock",
    )

    assert service.complete_message(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        response=response,
    )
    stored = service.get_message(user_id=user_id, message_id=message_id)
    assert stored
    assert stored["response"]["note_patch_status"] == "applied"
    with database.connection() as connection:
        note = connection.execute(
            """
            SELECT note_key, title, content FROM novel_author_notes
            WHERE project_id=?
            """,
            (project_id,),
        ).fetchone()
    assert note
    assert note["note_key"] == "lighthouse-clue"
    assert note["title"] == "灯塔线索分析"

    refreshed = AgentWorkspace(
        database,
        user_id=user_id,
        context=payload["context"],
        sources=[],
    )
    reading = refreshed.execute_tool(
        "read", {"path": "book/notes/author/lighthouse-clue.md"}
    )
    assert reading.result["writable"] is True
    assert "时间矛盾" in reading.result["content"]


def test_history_diff_and_restore_create_new_working_version(tmp_path):
    database, user_id, context = seed_workspace(tmp_path)
    project_id = context["project"]["id"]
    chapter_id = context["chapter"]["id"]
    chapter = database.get_novel_chapter(user_id, project_id, chapter_id)
    assert chapter
    initial_versions = database.list_chapter_versions(
        user_id, project_id, chapter_id
    )
    assert len(initial_versions) == 1
    initial_version_id = str(initial_versions[0]["id"])
    before = database.get_novel_chapter(user_id, project_id, chapter_id)
    assert before
    head_version_id = str(before["head_version_id"] or "")
    revised_path = tmp_path / "chapters" / chapter_id / "revised.txt"
    revised_content = "林岚没有拆信。\n她把湿邮戳对准了灯光。"
    revised_path.write_text(revised_content, encoding="utf-8")
    revised_version_id = database.record_manual_chapter_version(
        user_id=user_id,
        project_id=project_id,
        chapter_id=chapter_id,
        version_path=revised_path,
        char_count=len(revised_content),
    )
    assert revised_version_id

    service = AssistantChatService(
        database,
        tmp_path / "novels",
        tmp_path / "documents",
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="chapter",
        title="恢复版本",
        project_id=project_id,
        novel_chapter_id=chapter_id,
    )
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="请比较当前稿和最初版本，然后恢复最初版本。",
        provider="mock",
        model="mock",
        credential_source="default",
        agent_role="editor",
        auto_commit=True,
    )
    claimed = service.claim_next_message()
    assert claimed and claimed["id"] == message_id
    payload = service.build_job_payload(claimed)
    workspace = AgentWorkspace(
        database,
        user_id=user_id,
        context=payload["context"],
        sources=payload["sources"],
    )
    current_path = "book/manuscript/chapters/001.md"
    history_path = (
        "book/history/chapters/001/" + initial_version_id + ".md"
    )
    current = workspace.execute_tool("read", {"path": current_path})
    historical = workspace.execute_tool("read", {"path": history_path})
    comparison = workspace.execute_tool(
        "diff",
        {
            "path_a": current_path,
            "revision_a": current.result["revision"],
            "path_b": history_path,
            "revision_b": historical.result["revision"],
        },
    )
    assert comparison.result["different"] is True
    execution = workspace.execute_tool(
        "restore",
        {
            "version_path": history_path,
            "version_revision": historical.result["revision"],
            "current_path": current_path,
            "current_revision": current.result["revision"],
            "rationale": "作者明确要求恢复最初版本",
        },
    )
    response = AssistantChatResponse(
        result=AssistantChatResult(
            answer="已恢复最初版本为新的 main HEAD。",
            version_restore=execution.version_restore,
        ),
        raw_response="",
        input_tokens=0,
        output_tokens=0,
        provider="mock",
        model="mock",
    )

    assert service.complete_message(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        response=response,
    )
    after = database.get_novel_chapter(user_id, project_id, chapter_id)
    assert after
    assert str(after["head_version_id"]) not in {
        initial_version_id,
        revised_version_id,
    }
    assert str(after["head_version_id"] or "") != head_version_id
    restored = database.get_chapter_version(
        user_id,
        project_id,
        chapter_id,
        str(after["head_version_id"]),
    )
    assert restored
    assert restored["kind"] == "assistant_restore"
    assert Path(restored["content_path"]).read_text(encoding="utf-8") == (
        "林岚拆开旧信。\n邮戳还是湿的。"
    )
    stored = service.get_message(user_id=user_id, message_id=message_id)
    assert stored
    assert stored["response"]["version_restore_status"] == "applied"


def test_history_tool_loads_versions_beyond_initial_workspace_window(tmp_path):
    database, user_id, context = seed_workspace(tmp_path)
    project_id = context["project"]["id"]
    chapter_id = context["chapter"]["id"]
    chapter_dir = tmp_path / "chapters" / chapter_id / "versions"
    chapter_dir.mkdir(parents=True, exist_ok=True)

    for index in range(25):
        content = f"第 {index + 1} 稿：邮戳上的日期是线索。"
        version_path = chapter_dir / f"history-{index + 1:02d}.txt"
        version_path.write_text(content, encoding="utf-8")
        version_id = database.record_manual_chapter_version(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            version_path=version_path,
            char_count=len(content),
            change_summary=f"第 {index + 1} 次修改",
        )
        assert version_id

    versions = database.list_chapter_versions(
        user_id,
        project_id,
        chapter_id,
        limit=None,
    )
    assert len(versions) == 26
    oldest = versions[-1]
    oldest_path = (
        "book/history/chapters/001/" + str(oldest["id"]) + ".md"
    )

    workspace = AgentWorkspace(
        database,
        user_id=user_id,
        context=context,
        sources=[],
    )
    index_path = "book/history/chapters/001/index.json"
    workspace.execute_tool("read", {"path": index_path})
    index_items = json.loads(workspace.resources[index_path].content)
    assert len(index_items) == 20
    assert all(item["loaded"] is True for item in index_items)
    assert all(item["total_versions"] == 26 for item in index_items)
    assert all(
        item["more_available_via_history_tool"] is True
        for item in index_items
    )
    with pytest.raises(ValueError, match="不存在"):
        workspace.execute_tool("read", {"path": oldest_path})

    page = workspace.execute_tool(
        "history",
        {"chapter_position": 1, "page": 3, "page_size": 10},
    )
    assert page.result["version_count"] == 26
    assert page.result["page_count"] == 3
    assert len(page.result["versions"]) == 6
    assert oldest_path in {
        item["path"] for item in page.result["versions"]
    }
    loaded_oldest = workspace.execute_tool("read", {"path": oldest_path})
    assert "林岚拆开旧信" in loaded_oldest.result["content"]


def test_reference_evidence_can_be_saved_as_a_technique_card(tmp_path):
    database = Database(tmp_path / "reference-agent.db")
    database.initialize()
    user_id = database.create_user(
        "reference-agent", hash_password("password-123")
    )
    text = "第一章 门后\n脚步声先停在门后，灯光随后才从门缝漏出来。"
    chunks = split_chapters(text)
    document_dir = tmp_path / "documents" / str(user_id) / ("r" * 32)
    chapter_dir = document_dir / "chapters"
    chapter_dir.mkdir(parents=True)
    source_path = document_dir / "source.txt"
    source_path.write_text(text, encoding="utf-8")
    chapter_paths = []
    for index, chunk in enumerate(chunks, start=1):
        path = chapter_dir / f"{index:05d}.txt"
        path.write_text(chunk.text, encoding="utf-8")
        chapter_paths.append(path)
    document_id = database.create_document(
        user_id=user_id,
        title="参考小说",
        original_filename="reference.txt",
        source_path=source_path,
        source_encoding="utf-8",
        text_length=len(text),
        chunks=chunks,
        chapter_paths=chapter_paths,
    )
    chapter = database.list_chapters(user_id, document_id)[0]
    service = AssistantChatService(
        database,
        tmp_path / "novels",
        tmp_path / "documents",
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="reference_chapter",
        title="提炼技法",
        document_id=document_id,
        reference_chapter_id=str(chapter["id"]),
    )
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="请分析这一段的信息释放方式，提炼成技法卡并保存。",
        provider="mock",
        model="mock",
        credential_source="default",
        agent_role="researcher",
        auto_commit=True,
    )
    claimed = service.claim_next_message()
    assert claimed and claimed["id"] == message_id
    payload = service.build_job_payload(claimed)
    assert CREATE_TECHNIQUE_CARD in (
        payload["context"]["agent"]["capabilities"]
    )
    workspace = AgentWorkspace(
        database,
        user_id=user_id,
        context=payload["context"],
        sources=payload["sources"],
    )
    reference_path = "book/references/chapters/001.md"
    reference = workspace.execute_tool("read", {"path": reference_path})
    execution = workspace.execute_tool(
        "write",
        {
            "path": "book/techniques/new/delayed-reveal.json",
            "expected_revision": "new",
            "content": json.dumps(
                {
                    "source_path": reference_path,
                    "source_revision": reference.result["revision"],
                    "name": "先给感官后果再解释来源",
                    "dimension": "information",
                    "source_location": "第一章门后脚步与灯光的连续句",
                    "observation": "文本先呈现门后脚步停止的可感后果，随后才用灯光补充空间信息。",
                    "effect": "读者先形成具体疑问，再获得一部分解释，因此会继续追踪门后人物。",
                    "suitable_for": ["悬念场景", "人物尚未掌握全貌时"],
                    "unsuitable_for": ["行动前必须先说明规则的场景"],
                    "execution_rule": "先让未知事物造成一个可观察后果，至少一个叙事节拍后再解释一项来源。",
                    "originality_boundary": "不得复用门、脚步、灯缝、人物身份、原句或参考作品的具体答案。",
                    "author_note": "只学习信息出现的顺序。",
                },
                ensure_ascii=False,
            ),
        },
    )
    response = AssistantChatResponse(
        result=AssistantChatResult(
            answer="已保存一张带来源证据的技法卡。",
            technique_patch=execution.technique_patch,
        ),
        raw_response="",
        input_tokens=0,
        output_tokens=0,
        provider="mock",
        model="mock",
    )

    assert service.complete_message(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        response=response,
    )
    with database.connection() as connection:
        card = connection.execute(
            """
            SELECT name, source_document_id, source_chapter_id,
                   originality_boundary
            FROM reference_technique_cards WHERE user_id=?
            """,
            (user_id,),
        ).fetchone()
    assert card
    assert card["name"] == "先给感官后果再解释来源"
    assert card["source_document_id"] == document_id
    assert card["source_chapter_id"] == chapter["id"]
    stored = service.get_message(user_id=user_id, message_id=message_id)
    assert stored
    assert stored["response"]["technique_patch_status"] == "applied"


def test_multi_chapter_workflow_resumes_without_rewriting_completed_items(
    tmp_path,
):
    database, user_id, context = seed_workspace(tmp_path)
    project_id = context["project"]["id"]
    service = AssistantChatService(
        database,
        tmp_path / "novels",
        tmp_path / "documents",
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="project",
        title="连续写作",
        project_id=project_id,
    )
    message_id = service.queue_message(
        user_id=user_id,
        conversation_id=conversation_id,
        question="请连续写接下来的两章，中途失败时保留已经完成的章节。",
        provider="mock",
        model="mock",
        credential_source="default",
        agent_role="writer",
        auto_commit=True,
    )
    claimed = service.claim_next_message()
    assert claimed and claimed["id"] == message_id
    service.build_job_payload(claimed)
    workflow = service.start_or_resume_chapter_workflow_for_agent(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        user_id=user_id,
        chapters=[
            {
                "title": "第二章 灯塔",
                "outline": "林岚抵达灯塔。",
                "key_points": "灯提前亮起",
                "instruction": "承接湿邮戳，写林岚抵达灯塔。",
                "target_chars": 1200,
            },
            {
                "title": "第三章 值班表",
                "outline": "林岚核对值班表。",
                "key_points": "签名时间矛盾",
                "instruction": "承接灯塔现场，写林岚发现值班表矛盾。",
                "target_chars": 1200,
            },
        ],
        resume_latest=False,
    )
    workflow_id = str(workflow["id"])
    first = service.prepare_next_chapter_workflow_item_for_agent(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        user_id=user_id,
        workflow_id=workflow_id,
    )
    assert first
    first_result = service.complete_chapter_workflow_item_for_agent(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        user_id=user_id,
        workflow_id=workflow_id,
        item_id=str(first["item"]["id"]),
        content="林岚在暮色里抵达灯塔。值班室的灯比潮汐表早亮了半小时。",
        change_summary="测试连续写作第一章",
    )
    second = service.prepare_next_chapter_workflow_item_for_agent(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        user_id=user_id,
        workflow_id=workflow_id,
    )
    assert second
    second_chapter_id = str(second["item"]["chapter_id"])
    paused = service.pause_chapter_workflow_for_agent(
        user_id=user_id,
        workflow_id=workflow_id,
        item_id=str(second["item"]["id"]),
        error="模拟模型暂时不可用",
    )
    assert paused["status"] == "paused"
    assert paused["completed_count"] == 1

    resumed = service.start_or_resume_chapter_workflow_for_agent(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        user_id=user_id,
        chapters=[],
        resume_latest=True,
    )
    assert resumed["status"] == "running"
    resumed_item = service.prepare_next_chapter_workflow_item_for_agent(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        user_id=user_id,
        workflow_id=workflow_id,
    )
    assert resumed_item
    assert resumed_item["item"]["chapter_id"] == second_chapter_id
    service.complete_chapter_workflow_item_for_agent(
        message_id=message_id,
        claim_token=claimed["claim_token"],
        user_id=user_id,
        workflow_id=workflow_id,
        item_id=str(resumed_item["item"]["id"]),
        content="值班表夹在玻璃下。林岚发现签名时间早于灯塔通电记录。",
        change_summary="测试断点恢复",
    )
    completed = service.get_chapter_workflow(
        user_id=user_id,
        workflow_id=workflow_id,
    )
    assert completed["status"] == "completed"
    assert completed["completed_count"] == 2
    assert [item["status"] for item in completed["items"]] == [
        "completed",
        "completed",
    ]
    assert completed["items"][0]["chapter_id"] == first_result["chapter_id"]
    chapters = database.list_novel_chapters(user_id, project_id)
    assert len(chapters) == 3
    assert [chapter["title"] for chapter in chapters[-2:]] == [
        "第二章 灯塔",
        "第三章 值班表",
    ]
    workflow_versions = {
        str(item["chapter_id"]): str(item["version_id"])
        for item in completed["items"]
    }
    for chapter in chapters[-2:]:
        assert chapter["head_version_id"] == workflow_versions[chapter["id"]]


def test_workspace_patch_atomically_updates_multiple_settings(tmp_path):
    database, user_id, context = seed_workspace(tmp_path)
    workspace = AgentWorkspace(
        database,
        user_id=user_id,
        context=context,
        sources=[],
    )
    core_path = "book/settings/core.json"
    character_path = "book/settings/characters/character-1.json"
    core = workspace.execute_tool("read", {"path": core_path})
    character = workspace.execute_tool("read", {"path": character_path})

    execution = workspace.execute_tool(
        "patch",
        {
            "targets": [
                {
                    "path": core_path,
                    "expected_revision": core.result["revision"],
                    "replacements": [
                        {
                            "old_string": "克制，以动作承担情绪。",
                            "new_string": "冷峻，以动作承担情绪。",
                        }
                    ],
                },
                {
                    "path": character_path,
                    "expected_revision": character.result["revision"],
                    "replacements": [
                        {"old_string": "冷静", "new_string": "警觉"}
                    ],
                },
            ],
            "rationale": "同步文风和人物状态",
        },
    )

    assert execution.result["target_count"] == 2
    assert execution.result["replacement_count"] == 2
    assert execution.draft is None
    assert execution.settings_patch is not None
    assert execution.settings_patch.style_guide.startswith("冷峻")
    structured = execution.settings_patch.structured_edits or []
    assert len(structured) == 1
    assert structured[0].changes["traits"] == "警觉"


def test_workspace_patch_rolls_back_every_target_when_one_is_invalid(tmp_path):
    database, user_id, context = seed_workspace(tmp_path)
    workspace = AgentWorkspace(
        database,
        user_id=user_id,
        context=context,
        sources=[],
    )
    core_path = "book/settings/core.json"
    character_path = "book/settings/characters/character-1.json"
    core = workspace.execute_tool("read", {"path": core_path})
    character = workspace.execute_tool("read", {"path": character_path})

    with pytest.raises(ValueError, match="old_string"):
        workspace.execute_tool(
            "patch",
            {
                "targets": [
                    {
                        "path": core_path,
                        "expected_revision": core.result["revision"],
                        "replacements": [
                            {
                                "old_string": "克制，以动作承担情绪。",
                                "new_string": "冷峻，以动作承担情绪。",
                            }
                        ],
                    },
                    {
                        "path": character_path,
                        "expected_revision": character.result["revision"],
                        "replacements": [
                            {
                                "old_string": "不存在的人物状态",
                                "new_string": "警觉",
                            }
                        ],
                    },
                ]
            },
        )

    reread = workspace.execute_tool("read", {"path": core_path})
    assert reread.result["revision"] == core.result["revision"]
    assert workspace.draft is None
    assert workspace.settings_patch is None


def test_workspace_patch_rejects_mixed_commit_boundaries(tmp_path):
    database, user_id, context = seed_workspace(tmp_path)
    workspace = AgentWorkspace(
        database,
        user_id=user_id,
        context=context,
        sources=[],
    )
    chapter_path = "book/manuscript/chapters/001.md"
    core_path = "book/settings/core.json"
    chapter = workspace.execute_tool("read", {"path": chapter_path})
    core = workspace.execute_tool("read", {"path": core_path})

    with pytest.raises(ValueError, match="提交边界"):
        workspace.execute_tool(
            "patch",
            {
                "targets": [
                    {
                        "path": chapter_path,
                        "expected_revision": chapter.result["revision"],
                        "replacements": [
                            {"old_string": "旧信", "new_string": "来信"}
                        ],
                    },
                    {
                        "path": core_path,
                        "expected_revision": core.result["revision"],
                        "replacements": [
                            {"old_string": "悬疑", "new_string": "推理"}
                        ],
                    },
                ]
            },
        )


def test_workspace_edit_uses_revision_and_returns_revertible_draft(tmp_path):
    database, user_id, context = seed_workspace(tmp_path)
    workspace = AgentWorkspace(
        database,
        user_id=user_id,
        context=context,
        sources=[],
    )
    path = "book/manuscript/chapters/001.md"
    reading = workspace.execute_tool("read", {"path": path})

    execution = workspace.execute_tool(
        "edit",
        {
            "path": path,
            "old_string": "旧信",
            "new_string": "来信",
            "expected_revision": reading.result["revision"],
            "rationale": "统一物件称呼",
        },
    )

    assert execution.draft is not None
    assert execution.draft.content.startswith("林岚拆开来信")
    with pytest.raises(ValueError, match="版本已经变化"):
        workspace.execute_tool(
            "edit",
            {
                "path": path,
                "old_string": "邮戳",
                "new_string": "日期戳",
                "expected_revision": reading.result["revision"],
            },
        )


def test_workspace_setting_edit_and_create_become_structured_patches(tmp_path):
    database, user_id, context = seed_workspace(tmp_path)
    workspace = AgentWorkspace(
        database,
        user_id=user_id,
        context=context,
        sources=[],
    )
    core_path = "book/settings/core.json"
    core = workspace.execute_tool("read", {"path": core_path})
    core_write = workspace.execute_tool(
        "edit",
        {
            "path": core_path,
            "old_string": "克制，以动作承担情绪。",
            "new_string": "冷峻，以动作承担情绪。",
            "expected_revision": core.result["revision"],
        },
    )
    created = workspace.execute_tool(
        "write",
        {
            "path": "book/settings/world/new-lighthouse.json",
            "content": (
                '{"entry_type":"location","name":"七号灯塔",'
                '"description":"停用多年的港口灯塔。"}'
            ),
            "expected_revision": "new",
        },
    )

    assert core_write.settings_patch is not None
    assert core_write.settings_patch.style_guide.startswith("冷峻")
    assert created.settings_patch is not None
    assert len(created.settings_patch.structured_edits or []) == 1
    edit = created.settings_patch.structured_edits[0]
    assert edit.action == "create"
    assert edit.entity_type == "world_entry"
    assert edit.changes["name"] == "七号灯塔"


def test_story_blueprint_requires_story_planning_capability(tmp_path):
    database, user_id, context = seed_workspace(tmp_path)
    workspace = AgentWorkspace(
        database,
        user_id=user_id,
        context=context,
        sources=[],
    )
    blueprint_path = "book/settings/structure/blueprint.json"
    content = (
        '{"central_question":"旧信是谁寄出的？",'
        '"protagonist_goal":"找到寄信人",'
        '"core_conflict":"真相会伤害仍然活着的人",'
        '"stakes":"林岚失去唯一证人",'
        '"opening_state":"林岚收到旧信",'
        '"ending_state":"林岚公开真相",'
        '"major_turns":["发现寄信时间不可能"],'
        '"must_payoffs":["揭示寄信人的身份"]}'
    )

    with pytest.raises(PermissionError, match="不能新建"):
        workspace.execute_tool(
            "write",
            {
                "path": blueprint_path,
                "content": content,
                "expected_revision": "new",
            },
        )

    context["agent"]["capabilities"].append(PROPOSE_STORY_PLAN)
    story_workspace = AgentWorkspace(
        database,
        user_id=user_id,
        context=context,
        sources=[],
    )
    execution = story_workspace.execute_tool(
        "write",
        {
            "path": blueprint_path,
            "content": content,
            "expected_revision": "new",
        },
    )

    assert execution.story_plan is not None
    assert execution.settings_patch is None
    assert execution.story_plan.blueprint.central_question == (
        "旧信是谁寄出的？"
    )


def test_workspace_rejects_paths_outside_virtual_book(tmp_path):
    database, user_id, context = seed_workspace(tmp_path)
    workspace = AgentWorkspace(
        database,
        user_id=user_id,
        context=context,
        sources=[],
    )

    with pytest.raises(ValueError, match="book/"):
        workspace.execute_tool("read", {"path": "../../etc/passwd"})


def test_workspace_can_search_full_conversation_history(tmp_path):
    database, user_id, context = seed_workspace(tmp_path)
    service = AssistantChatService(
        database, tmp_path / "novels", tmp_path / "documents"
    )
    conversation_id = service.create_conversation(
        user_id=user_id,
        scope_type="chapter",
        title="长期创作讨论",
        project_id=context["project"]["id"],
        novel_chapter_id=context["chapter"]["id"],
    )
    with database.connection() as connection:
        connection.executemany(
            """
            INSERT INTO assistant_messages(
                id, conversation_id, role, content, status, created_at
            ) VALUES (?, ?, 'user', ?, 'completed', ?)
            """,
            [
                (
                    "old-message",
                    conversation_id,
                    "以后灯塔钥匙必须一直由林岚保管。",
                    "2026-01-01T00:00:00+00:00",
                ),
                (
                    "current-message",
                    conversation_id,
                    "你还记得钥匙归谁吗？",
                    "2026-01-02T00:00:00+00:00",
                ),
            ],
        )
        connection.commit()
    context.update(
        {
            "conversation_id": conversation_id,
            "current_user_message_id": "current-message",
        }
    )
    workspace = AgentWorkspace(
        database,
        user_id=user_id,
        context=context,
        sources=[],
    )

    result = workspace.execute_tool(
        "grep",
        {
            "path": "book/notes",
            "pattern": "灯塔钥匙",
        },
    )

    assert result.result["matches"] == [
        {
            "path": "book/notes/conversation-history.jsonl",
            "line": 1,
            "text": result.result["matches"][0]["text"],
        }
    ]
    assert "林岚保管" in result.result["matches"][0]["text"]
    assert "current-message" not in result.result["matches"][0]["text"]


def test_chapter_writing_packet_compiles_scene_style_and_story_contracts(
    tmp_path,
):
    database, user_id, context = seed_workspace(tmp_path)
    context.update(
        {
            "confirmed_task_card": {
                "purpose": "林岚从试探守塔人转为决定亲自登塔",
                "start_state": "她只有一封异常旧信",
                "end_state": "她拿到潮汐时刻并决定冒险",
                "central_conflict": "守塔人拒绝承认见过寄信人",
                "must_happen": ["守塔人说漏潮汐时间"],
                "must_preserve": ["钥匙始终在林岚手中"],
                "forbidden": ["直接揭晓寄信人身份"],
                "ending_hook": "灯塔在错误时间亮起",
                "scenes": ["修理铺内的试探", "走向防波堤"],
            },
            "characters": [
                {
                    "name": "林岚",
                    "goal": "查明旧信来源",
                    "knowledge": "邮戳日期不可能",
                }
            ],
            "confirmed_voice_profile": {
                "distance": "第三人称近距离",
                "dialogue": "克制，少解释",
            },
            "confirmed_editing_preferences": [
                "动作已经成立后不要再解释情绪"
            ],
            "confirmed_story_plan": {
                "central_question": "是谁从失踪的灯塔寄信"
            },
            "planned_causal_links": [
                "潮汐时刻会在第三章解释失踪路径"
            ],
            "canonical_memory": {
                "objects": {"灯塔钥匙": "由林岚保管"}
            },
            "active_techniques": [
                {"name": "延迟揭示", "instruction": "先给事实再推断"}
            ],
        }
    )
    workspace = AgentWorkspace(
        database,
        user_id=user_id,
        context=context,
        sources=[],
    )
    path = "book/manuscript/chapters/001.md"
    reading = workspace.execute_tool("read", {"path": path})

    packet = workspace.build_chapter_writing_packet(
        path=path,
        expected_revision=reading.result["revision"],
        instruction="写完本章，让林岚决定去灯塔。",
        mode="replace",
        target_chars=2400,
    )

    assert packet["schema_version"] == 2
    assert packet["scene_contract"]["end_state"].startswith("她拿到")
    assert any(
        "钥匙始终" in item
        for item in packet["scene_contract"]["must_preserve"]
    )
    assert packet["narrative_contract"]["point_of_view"] == "第三人称限知"
    assert packet["narrative_contract"]["confirmed_voice_profile"] == {
        "distance": "第三人称近距离",
        "dialogue": "克制，少解释",
    }
    assert (
        packet["story_contract"]["canonical_memory"]["objects"][
            "灯塔钥匙"
        ]
        == "由林岚保管"
    )
    assert packet["active_techniques"][0]["name"] == "延迟揭示"
    support_paths = {
        item["path"] for item in packet["supporting_resources"]
    }
    assert "book/settings/core.json" in support_paths
    assert "book/analysis/story-state.json" in support_paths
