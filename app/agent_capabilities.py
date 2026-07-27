from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet


READ_PROJECT = "read_project"
READ_CHAPTER = "read_chapter"
SEARCH_PROJECT = "search_project"
READ_REFERENCE = "read_reference"
SEARCH_REFERENCE = "search_reference"
ANALYZE_REFERENCE = "analyze_reference"
WEB_SEARCH = "web_search"
PROPOSE_SETTINGS_PATCH = "propose_settings_patch"
PROPOSE_STORY_PLAN = "propose_story_plan"
CREATE_CANDIDATE_DRAFT = "create_candidate_draft"
PROPOSE_TEXT_PATCH = "propose_text_patch"


AGENT_ROLES: Dict[str, Dict[str, Any]] = {
    "advisor": {
        "label": "讨论",
        "description": "只读分析与构思，不创建设定或正文变更。",
        "capabilities": frozenset(
            {
                READ_PROJECT,
                READ_CHAPTER,
                SEARCH_PROJECT,
                WEB_SEARCH,
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
                WEB_SEARCH,
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
                WEB_SEARCH,
                PROPOSE_SETTINGS_PATCH,
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
                WEB_SEARCH,
                PROPOSE_STORY_PLAN,
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
                ANALYZE_REFERENCE,
                WEB_SEARCH,
            }
        ),
    },
    "writer": {
        "label": "创作",
        "description": "读取设定并创作正文，自动提交为可撤回工作稿。",
        "capabilities": frozenset(
            {
                READ_PROJECT,
                READ_CHAPTER,
                SEARCH_PROJECT,
                WEB_SEARCH,
                CREATE_CANDIDATE_DRAFT,
            }
        ),
    },
    "editor": {
        "label": "修订",
        "description": "检查现有文字并局部修订，自动提交且可以撤回。",
        "capabilities": frozenset(
            {
                READ_PROJECT,
                READ_CHAPTER,
                SEARCH_PROJECT,
                WEB_SEARCH,
                PROPOSE_TEXT_PATCH,
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

    if selected_intent == "draft_prose":
        if scope_type not in {"project", "chapter"}:
            return ResolvedAgentDispatch(
                role="advisor",
                intent="discuss",
                reason="invalid_scope_fallback",
                goal="answer",
            )
        if not settings_ready:
            return ResolvedAgentDispatch(
                role="planner",
                intent="update_settings",
                reason="settings_prerequisite",
                goal="propose_settings_patch",
                settings_prerequisite=True,
            )
        return ResolvedAgentDispatch(
            role="writer",
            intent=selected_intent,
            reason=reason,
            goal="create_chapter_draft",
        )

    if selected_intent == "revise_prose":
        if scope_type != "chapter" or not has_quote:
            fallback_role = (
                "editor"
                if requested == "editor" and scope_type == "chapter"
                else "advisor"
            )
            return ResolvedAgentDispatch(
                role=fallback_role,
                intent="discuss",
                reason="selection_required",
                goal="answer",
            )
        return ResolvedAgentDispatch(
            role="editor",
            intent=selected_intent,
            reason=reason,
            goal="replace_selected_text",
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
        if not settings_ready:
            return ResolvedAgentDispatch(
                role="planner",
                intent="update_settings",
                reason="settings_prerequisite",
                goal="propose_settings_patch",
                settings_prerequisite=True,
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
            "模型只能提出结构化变更。服务器可按作者开启的策略自动提交到"
            "可撤回工作稿；进入正史仍必须由作者触发。"
        ),
    }


def agent_role_prompt(role: str) -> str:
    manifest = agent_manifest(role)
    capabilities = "、".join(manifest["capabilities"])
    role_instructions = {
        "advisor": (
            "你处于只读讨论模式。可以分析、提问和搜索已有上下文，但不得"
            "创建设定候选或正文变更；settings_patch、draft 和 rewrite "
            "必须为 null。"
        ),
        "analyst": (
            "你处于作品分析模式。应先读取与问题直接相关的材料，再给出"
            "有依据的结构、人物、情节、文风或连续性诊断。不得创建设定、"
            "规划或正文候选；settings_patch、story_plan、draft 和 rewrite "
            "必须为 null。"
        ),
        "planner": (
            "你处于设定策划模式。应先读取作品材料，再把已经有依据的内容"
            "整理为结构化候选设定。只能通过 propose_settings_patch 提交"
            "候选，不得创作或修订正文；story_plan、draft 和 rewrite 必须为 null。"
        ),
        "story_planner": (
            "你处于故事规划模式。应读取已有设定，把全书核心悬问、主角目标、"
            "冲突引擎、终局状态、关键转折和必须兑现项整理为一个完整规划。"
            "只能通过 propose_story_plan 提交候选，不得创作或修订正文；"
            "settings_patch、draft 和 rewrite 必须为 null。"
        ),
        "researcher": (
            "你处于只读研究模式。必须以参考原文或拆书分析中的具体证据"
            "说明结构、节奏、视角和信息释放方法，只提炼可迁移的抽象规则。"
            "不得续写参考作品、复刻独特措辞或生成可直接应用的正文；"
            "rewrite、draft、settings_patch 和 story_plan 必须为 null。"
        ),
        "writer": (
            "你处于创作模式。可以创作新的候选文字，但不能直接覆盖现有"
            "正文，也不能把草稿称为已经保存或生效。服务器会依据当前策略"
            "提交工作稿；局部修订应交给修订模式，settings_patch 和 story_plan "
            "必须为 null。"
        ),
        "editor": (
            "你处于修订模式。应优先诊断现有文本并做最小范围修改。只有"
            "存在经过校验的正文选区且作者明确要求修改时，才可返回 rewrite；"
            "服务器会依据当前策略提交可撤回工作稿，settings_patch 和 draft "
            "和 story_plan 必须为 null。"
        ),
    }[manifest["role"]]
    return (
        f"当前协作角色：{manifest['label']}（{manifest['role']}）。\n"
        f"服务器授予的能力：{capabilities}。\n"
        f"{role_instructions}\n"
        f"{manifest['write_policy']}"
    )
