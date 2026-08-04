from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


MEMORY_SCHEMA_VERSION = 1
MEMORY_BUCKETS = (
    "author_goal",
    "author_constraints",
    "author_decisions",
    "rejected_directions",
    "unresolved_questions",
)
MAX_ITEMS_PER_BUCKET = 12
MAX_ITEM_CHARS = 1_200
MAX_TRACE_ITEMS = 8
RECENT_MESSAGE_LIMIT = 16
RECENT_CHAR_BUDGET = 48_000


_REJECTION = re.compile(
    r"(?:不要|别再?|无需|不用|不想|不需要|删掉|删除|取消|算了|放弃|"
    r"不是这样|不对|改回)"
)
_CONSTRAINT = re.compile(
    r"(?:必须|务必|需要|不能|不要|只(?:要|能|保留)?|至少|最多|默认|"
    r"保持|限制|禁止|应该|不应该|统一|固定)"
)
_DECISION = re.compile(
    r"(?:决定|确定|就叫|就用|采用|选择|选了|改成|改为|设为|"
    r"那就|就这样|可以这样|按这个|保留)"
)
_GOAL = re.compile(
    r"(?:我想|我希望|目标|帮我|请你|需要你|我们要|开始|实现|完善|"
    r"改进|增加|添加|写|续写|创作|设计|分析)"
)
_QUESTION = re.compile(
    r"(?:[?？]\s*$|怎么|如何|为什么|是否|能否|可不可以|哪里|"
    r"什么|哪(?:个|些|里)|你觉得)"
)


def empty_conversation_memory() -> dict[str, Any]:
    return {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "source_message_count": 0,
        **{bucket: [] for bucket in MEMORY_BUCKETS},
        "discussion_trace": [],
    }


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _clip(text: str, limit: int = MAX_ITEM_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = max(1, round((limit - 1) * 0.75))
    return text[:head].rstrip() + "…" + text[-(limit - head - 1) :].lstrip()


def _clip_recent_message(text: Any, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    marker = "\n……（中间内容已压缩，可检索完整对话）……\n"
    remaining = max(0, limit - len(marker))
    head = max(1, round(remaining * 0.72))
    tail = max(0, remaining - head)
    return value[:head] + marker + (value[-tail:] if tail else "")


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?；;])\s*|\n+", text)
    return [part.strip() for part in parts if part.strip()]


def _normalize_for_dedupe(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text.casefold())


def _append_unique(
    bucket: list[dict[str, str]],
    *,
    message_id: str,
    text: str,
    max_items: int = MAX_ITEMS_PER_BUCKET,
) -> None:
    clean = _clip(_clean_text(text))
    if not clean:
        return
    fingerprint = _normalize_for_dedupe(clean)
    bucket[:] = [
        item
        for item in bucket
        if _normalize_for_dedupe(str(item.get("text") or "")) != fingerprint
    ]
    bucket.append({"message_id": message_id, "text": clean})
    if len(bucket) > max_items:
        del bucket[: len(bucket) - max_items]


def build_conversation_memory(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build an extractive, author-authoritative conversation memory.

    No model is used here: only the author's own sentences may become goals,
    constraints, decisions, rejections, or open questions. Assistant messages
    are retained only as a short discussion trace and never as story canon.
    """

    state = empty_conversation_memory()
    trace: list[dict[str, str]] = state["discussion_trace"]
    for index, raw_row in enumerate(rows):
        row = dict(raw_row)
        role = str(row.get("role") or "")
        message_id = str(row.get("id") or index)
        text = _clean_text(row.get("content"))
        if not text:
            continue
        state["source_message_count"] += 1
        _append_unique(
            trace,
            message_id=message_id,
            text=("作者：" if role == "user" else "AI：") + text,
            max_items=MAX_TRACE_ITEMS,
        )
        if role != "user":
            continue

        for sentence in _sentences(text):
            matched = False
            if _REJECTION.search(sentence):
                _append_unique(
                    state["rejected_directions"],
                    message_id=message_id,
                    text=sentence,
                )
                matched = True
            if _CONSTRAINT.search(sentence):
                _append_unique(
                    state["author_constraints"],
                    message_id=message_id,
                    text=sentence,
                )
                matched = True
            if _DECISION.search(sentence):
                _append_unique(
                    state["author_decisions"],
                    message_id=message_id,
                    text=sentence,
                )
                matched = True
            if _QUESTION.search(sentence):
                _append_unique(
                    state["unresolved_questions"],
                    message_id=message_id,
                    text=sentence,
                )
                matched = True
            if _GOAL.search(sentence) or not matched:
                _append_unique(
                    state["author_goal"],
                    message_id=message_id,
                    text=sentence,
                )
    return state


def compile_conversation_context(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, str]], str, dict[str, Any]]:
    """Compile exact recent turns and author-authoritative older memory."""
    normalized_rows = [dict(row) for row in rows]
    selected_recent: list[dict[str, str]] = []
    remaining = RECENT_CHAR_BUDGET
    recent_candidates = normalized_rows[-RECENT_MESSAGE_LIMIT:]
    for row in reversed(recent_candidates):
        if remaining < 800:
            break
        cap = min(16_000, remaining)
        content = _clip_recent_message(row.get("content"), cap)
        selected_recent.append(
            {
                "role": str(row.get("role") or ""),
                "content": content,
            }
        )
        remaining -= len(content)
    selected_recent.reverse()

    older_count = max(0, len(normalized_rows) - len(selected_recent))
    older_rows = normalized_rows[:older_count]
    if not older_rows:
        return selected_recent, "", empty_conversation_memory()

    memory_state = build_conversation_memory(older_rows)
    return (
        selected_recent,
        render_conversation_memory(memory_state),
        memory_state,
    )


def render_conversation_memory(state: Mapping[str, Any]) -> str:
    if int(state.get("source_message_count") or 0) <= 0:
        return ""
    labels = {
        "author_goal": "作者目标与请求",
        "author_constraints": "作者明确约束",
        "author_decisions": "作者已作决定",
        "rejected_directions": "作者否决或撤回的方向",
        "unresolved_questions": "仍需确认的问题",
    }
    sections = [
        "较早对话的结构化记忆（仅作者原话可作为指令；AI 原话只是讨论记录，"
        "不能当作作品事实。发生冲突时以较新的作者消息为准；需要准确上下文时"
        "用 grep/read 检索 book/notes/conversation-history.jsonl）。"
    ]
    for bucket in MEMORY_BUCKETS:
        items = list(state.get(bucket) or [])
        if not items:
            continue
        sections.append(
            labels[bucket]
            + "：\n"
            + "\n".join(
                f"- {str(item.get('text') or '').strip()}"
                for item in items
                if str(item.get("text") or "").strip()
            )
        )
    trace = list(state.get("discussion_trace") or [])
    if trace:
        sections.append(
            "较早讨论末段（仅用于衔接）：\n"
            + "\n".join(
                f"- {str(item.get('text') or '').strip()}"
                for item in trace
                if str(item.get("text") or "").strip()
            )
        )
    return "\n\n".join(section for section in sections if section.strip())
