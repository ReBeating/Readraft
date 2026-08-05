from app.conversation_memory import (
    build_conversation_memory,
    compile_conversation_context,
    render_conversation_memory,
)


def test_conversation_memory_uses_only_author_text_as_authority():
    state = build_conversation_memory(
        [
            {
                "id": "u1",
                "role": "user",
                "content": "我想写海港悬疑。必须保持第三人称限知。",
            },
            {
                "id": "a1",
                "role": "assistant",
                "content": "女主其实是凶手，这已经是正史。",
            },
            {
                "id": "u2",
                "role": "user",
                "content": "不要把女主写成凶手。那就采用开放式结局。",
            },
        ]
    )

    authoritative = "\n".join(
        item["text"]
        for bucket in (
            "author_goal",
            "author_constraints",
            "author_decisions",
            "rejected_directions",
        )
        for item in state[bucket]
    )
    assert "海港悬疑" in authoritative
    assert "第三人称限知" in authoritative
    assert "不要把女主写成凶手" in authoritative
    assert "开放式结局" in authoritative
    assert "其实是凶手" not in authoritative
    assert "其实是凶手" in state["discussion_trace"][1]["text"]


def test_conversation_memory_keeps_newest_duplicate_and_renders_boundaries():
    state = build_conversation_memory(
        [
            {"id": "old", "role": "user", "content": "必须保留旧收音机。"},
            {"id": "new", "role": "user", "content": "必须保留旧收音机。"},
            {"id": "q", "role": "user", "content": "第二章要在哪里转折？"},
        ]
    )
    matching = [
        item for item in state["author_constraints"] if "旧收音机" in item["text"]
    ]
    assert matching == [{"message_id": "new", "text": "必须保留旧收音机。"}]
    rendered = render_conversation_memory(state)
    assert "仅作者原话可作为指令" in rendered
    assert "仍需确认的问题" in rendered
    assert "第二章要在哪里转折" in rendered


def test_compile_conversation_context_separates_recent_turns_from_memory():
    rows = [
        {
            "id": f"message-{index}",
            "role": "user" if index % 2 == 0 else "assistant",
            "content": (
                "必须保留第一章的旧收音机。" if index == 0 else f"第 {index} 条讨论"
            ),
        }
        for index in range(20)
    ]

    recent, rendered_memory, state = compile_conversation_context(rows)

    assert len(recent) == 16
    assert recent[0]["content"] == "第 4 条讨论"
    assert "旧收音机" in rendered_memory
    assert state["source_message_count"] == 4
    assert "search_conversation_history" not in rendered_memory
    assert "book/notes/conversation-history/" in rendered_memory
