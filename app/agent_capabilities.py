from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet


READ_PROJECT = "read_project"
READ_CHAPTER = "read_chapter"
SEARCH_PROJECT = "search_project"
READ_REFERENCE = "read_reference"
SEARCH_REFERENCE = "search_reference"
SEARCH_CONVERSATION = "search_conversation"
ANALYZE_REFERENCE = "analyze_reference"
WEB_SEARCH = "web_search"
PROPOSE_SETTINGS_PATCH = "propose_settings_patch"
PROPOSE_STORY_PLAN = "propose_story_plan"
WRITE_CHAPTER = "write_chapter"
CREATE_CHAPTER = "create_chapter"
RUN_BOUNDED_TASK = "run_bounded_task"
MANAGE_CHAPTERS = "manage_chapters"
MANAGE_NOTES = "manage_notes"
CREATE_TECHNIQUE_CARD = "create_technique_card"


AGENT_ROLES: Dict[str, Dict[str, Any]] = {
    "advisor": {
        "label": "讨论",
        "description": "只读分析与构思，不创建设定或正文变更。",
        "capabilities": frozenset(
            {
                READ_PROJECT,
                READ_CHAPTER,
                SEARCH_PROJECT,
                SEARCH_CONVERSATION,
                WEB_SEARCH,
                RUN_BOUNDED_TASK,
            }
        ),
    },
    "analyst": {
        "label": "分析",
        "description": "只读检查作品、正文和结构，输出有依据的诊断。",
        "capabilities": frozenset(
            {
                READ_PROJECT,
                READ_CHAPTER,
                SEARCH_PROJECT,
                SEARCH_CONVERSATION,
                WEB_SEARCH,
                RUN_BOUNDED_TASK,
            }
        ),
    },
    "planner": {
        "label": "设定",
        "description": "整理作品材料并提出结构化设定候选，不创作正文。",
        "capabilities": frozenset(
            {
                READ_PROJECT,
                READ_CHAPTER,
                SEARCH_PROJECT,
                SEARCH_CONVERSATION,
                WEB_SEARCH,
                PROPOSE_SETTINGS_PATCH,
                RUN_BOUNDED_TASK,
                MANAGE_CHAPTERS,
                MANAGE_NOTES,
            }
        ),
    },
    "story_planner": {
        "label": "规划",
        "description": "基于已有设定提出可版本化的全书故事规划。",
        "capabilities": frozenset(
            {
                READ_PROJECT,
                READ_CHAPTER,
                SEARCH_PROJECT,
                SEARCH_CONVERSATION,
                WEB_SEARCH,
                PROPOSE_STORY_PLAN,
                PROPOSE_SETTINGS_PATCH,
                RUN_BOUNDED_TASK,
                MANAGE_CHAPTERS,
                MANAGE_NOTES,
            }
        ),
    },
    "researcher": {
        "label": "研究",
        "description": "只读拆解参考作品，提炼带证据的可迁移方法。",
        "capabilities": frozenset(
            {
                READ_REFERENCE,
                SEARCH_REFERENCE,
                SEARCH_CONVERSATION,
                ANALYZE_REFERENCE,
                CREATE_TECHNIQUE_CARD,
                WEB_SEARCH,
                RUN_BOUNDED_TASK,
            }
        ),
    },
    "writer": {
        "label": "创作",
        "description": "读取设定并创作正文；写入推进 main HEAD，并保留历史。",
        "capabilities": frozenset(
            {
                READ_PROJECT,
                READ_CHAPTER,
                SEARCH_PROJECT,
                SEARCH_CONVERSATION,
                WEB_SEARCH,
                WRITE_CHAPTER,
                CREATE_CHAPTER,
                RUN_BOUNDED_TASK,
                MANAGE_CHAPTERS,
                MANAGE_NOTES,
            }
        ),
    },
    "editor": {
        "label": "修订",
        "description": (
            "检查现有文字并局部修订；有选区时精确替换，"
            "无选区时可提交保留其余内容的整章修订稿。"
        ),
        "capabilities": frozenset(
            {
                READ_PROJECT,
                READ_CHAPTER,
                SEARCH_PROJECT,
                SEARCH_CONVERSATION,
                WEB_SEARCH,
                WRITE_CHAPTER,
                RUN_BOUNDED_TASK,
                MANAGE_CHAPTERS,
                MANAGE_NOTES,
            }
        ),
    },
}

AUTO_AGENT_ROLE = "auto"
AUTO_INTENT_CONFIDENCE_THRESHOLD = 0.65
AGENT_INTENTS = frozenset(
    {
        "discuss",
        "analyze_work",
        "update_settings",
        "plan_story",
        "draft_prose",
        "draft_new_chapter",
        "revise_prose",
    }
)


@dataclass(frozen=True)
class ResolvedAgentDispatch:
    role: str
    intent: str
    reason: str
    goal: str
    settings_prerequisite: bool = False


def normalize_agent_intent(value: str) -> str:
    intent = str(value or "").strip().lower()
    if intent not in AGENT_INTENTS:
        raise ValueError("不支持的 AI 任务意图")
    return intent


def normalize_agent_role(value: str) -> str:
    role = str(value or "").strip().lower()
    if role not in AGENT_ROLES:
        raise ValueError("不支持的 AI 协作角色")
    return role


def normalize_requested_agent_role(value: str) -> str:
    role = str(value or "").strip().lower() or AUTO_AGENT_ROLE
    if role == AUTO_AGENT_ROLE:
        return role
    return normalize_agent_role(role)


def resolve_agent_dispatch(
    *,
    requested_role: str,
    scope_type: str,
    intent: str,
    has_quote: bool,
    settings_ready: bool = True,
    confidence: float = 1.0,
) -> ResolvedAgentDispatch:
    requested = normalize_requested_agent_role(requested_role)
    if scope_type in {"document", "reference_chapter"}:
        return ResolvedAgentDispatch(
            role="researcher",
            intent="analyze_work",
            reason="reference_scope",
            goal="answer",
        )

    if requested != AUTO_AGENT_ROLE:
        selected_intent = {
            "advisor": "discuss",
            "analyst": "analyze_work",
            "planner": "update_settings",
            "story_planner": "plan_story",
            "writer": "draft_prose",
            "editor": "revise_prose",
            "researcher": "analyze_work",
        }[requested]
        reason = "author_override"
    else:
        selected_intent = normalize_agent_intent(intent)
        if float(confidence) < AUTO_INTENT_CONFIDENCE_THRESHOLD:
            selected_intent = "discuss"
            reason = "low_confidence_fallback"
        else:
            reason = "model_intent"

    if selected_intent in {"draft_prose", "draft_new_chapter"}:
        if scope_type not in {"project", "chapter"}:
            return ResolvedAgentDispatch(
                role="advisor",
                intent="discuss",
                reason="invalid_scope_fallback",
                goal="answer",
            )
        return ResolvedAgentDispatch(
            role="writer",
            intent=selected_intent,
            reason=reason,
            goal="create_chapter_draft",
        )

    if selected_intent == "revise_prose":
        if scope_type != "chapter":
            return ResolvedAgentDispatch(
                role="advisor",
                intent="discuss",
                reason="invalid_scope_fallback",
                goal="answer",
            )
        return ResolvedAgentDispatch(
            role="editor",
            intent=selected_intent,
            reason=(reason if has_quote else "whole_chapter_revision"),
            goal=(
                "replace_selected_text"
                if has_quote
                else "create_chapter_draft"
            ),
        )

    if selected_intent == "update_settings":
        if scope_type not in {"project", "chapter"}:
            return ResolvedAgentDispatch(
                role="advisor",
                intent="discuss",
                reason="invalid_scope_fallback",
                goal="answer",
            )
        return ResolvedAgentDispatch(
            role="planner",
            intent=selected_intent,
            reason=reason,
            goal="propose_settings_patch",
        )

    if selected_intent == "plan_story":
        if scope_type not in {"project", "chapter"}:
            return ResolvedAgentDispatch(
                role="advisor",
                intent="discuss",
                reason="invalid_scope_fallback",
                goal="answer",
            )
        return ResolvedAgentDispatch(
            role="story_planner",
            intent=selected_intent,
            reason=reason,
            goal="propose_story_plan",
        )

    if selected_intent == "analyze_work":
        return ResolvedAgentDispatch(
            role="analyst",
            intent=selected_intent,
            reason=reason,
            goal="answer",
        )

    return ResolvedAgentDispatch(
        role="advisor",
        intent="discuss",
        reason=reason,
        goal="answer",
    )


def agent_capabilities(role: str) -> FrozenSet[str]:
    normalized = normalize_agent_role(role)
    return AGENT_ROLES[normalized]["capabilities"]


def agent_manifest(role: str) -> Dict[str, Any]:
    normalized = normalize_agent_role(role)
    profile = AGENT_ROLES[normalized]
    return {
        "role": normalized,
        "label": str(profile["label"]),
        "description": str(profile["description"]),
        "capabilities": sorted(profile["capabilities"]),
        "write_policy": (
            "模型只能提出结构化变更。服务器按本轮权限原子写入；正文写入"
            "会创建不可变历史并推进 main HEAD，Tag 始终只读。"
        ),
    }


def native_agent_role_prompt(role: str) -> str:
    """Role guidance for the native resource/tool runtime."""

    manifest = agent_manifest(role)
    instructions = {
        "advisor": (
            "只读讨论：可以检索作品并帮助作者收敛构思，不修改任何资源。"
        ),
        "analyst": (
            "只读分析：先读取与问题直接相关的证据，再诊断结构、人物、"
            "情节、文风或连续性，不修改任何资源。"
        ),
        "planner": (
            "设定编辑：把已有依据的想法写入 settings/ 下准确的结构化"
            "资源。微调已有对象时优先 edit，只改变作者要求的字段。作者明确"
            "要求保存构思或专项结论时，可写入 notes/author/；普通讨论不归档。"
        ),
        "story_planner": (
            "故事规划：读取已有设定后完善全书蓝图与剧情线；不要写正文。"
            "作者明确要求沉淀规划讨论时，可保存为作者笔记。"
        ),
        "researcher": (
            "只读研究：用参考原文或分析中的证据提炼可迁移方法，不续写"
            "参考作品，也不修改其资源。作者明确要求保存或提炼技法卡时，"
            "可以把有原文证据和原创边界的方法写入 techniques/new/。"
        ),
        "writer": (
            "章节创作：写作前读取上一章结尾、当前章节目标和必要设定。"
            "另起一章时先用 create 建立章节，再读取新路径并用 compose 写正文。"
            "不要让计划、解释或检查清单混入正文。作者明确要求保存非正文"
            "材料时，写入作者笔记而不是正文。"
        ),
        "editor": (
            "正文修订：先定位问题，再用 edit 做最小范围修改。未被要求"
            "改变的文字尽量保持原样；大范围重写才使用 compose 动作。作者"
            "明确要求保留诊断时，可将诊断保存为作者笔记。"
        ),
    }[manifest["role"]]
    return (
        f"当前协作角色：{manifest['label']}（{manifest['role']}）。\n"
        f"{instructions}"
    )
