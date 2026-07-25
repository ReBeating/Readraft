from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import secrets
import shutil
import sqlite3
import tempfile
import time
from collections import deque
from contextlib import asynccontextmanager
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from urllib.parse import quote

from fastapi import FastAPI, File, Form, Request, UploadFile, status
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from starlette.background import BackgroundTask

from .assistant_chat import build_assistant_chat_model
from .assistant_chat_service import (
    SETTING_FIELD_LABELS,
    AssistantChatService,
)
from .canon_impact_service import CanonImpactService
from .causal_branch_adoption_schema import CausalBranchTaskPatch
from .causal_branch_adoption_service import CausalBranchAdoptionService
from .causal_branch_planner import build_causal_branch_planner
from .causal_branch_service import CausalBranchSimulationService
from .causal_suggestion_planner import build_causal_suggestion_planner
from .causal_suggestion_service import CausalSuggestionService
from .chapter_splitter import decode_upload, split_chapters
from .config import Settings
from .continuity import ContinuityService
from .context_compiler import (
    compile_active_techniques,
    compile_planned_causal_links,
    compile_story_plan_context,
)
from .credentials import (
    CredentialCipher,
    CredentialError,
    DEFAULT_SYSTEM_PROMPT,
    key_hint,
    validate_api_key,
    validate_model,
)
from .db import Database, utc_now
from .deepseek import build_analyzer
from .memory_extraction import build_memory_extractor
from .memory_identity import (
    IDENTITY_TYPE_LABELS,
    IDENTITY_TYPES,
    MemoryIdentityService,
)
from .memory_schema import StoryDelta
from .memory_service import MemoryService
from .model_catalog import ModelCatalogError, fetch_models
from .model_provider import (
    ProviderConfigError,
    get_provider,
    list_providers,
)
from .planning_schema import ChapterTaskCard, SceneBeat
from .planning_ai import build_chapter_planner
from .planning_service import PlanningService
from .preference_extraction import build_edit_preference_extractor
from .preference_service import (
    EDIT_PREFERENCE_CATEGORIES,
    PreferenceService,
)
from .process_lock import ProcessLock
from .project_archive import (
    ProjectArchiveError,
    create_project_archive,
    import_project_archive,
)
from .quality_audit import build_quality_auditor, effective_char_count
from .reader_planner import build_reader_planner
from .reader_service import ReaderDecisionService
from .security import (
    csrf_token,
    hash_password,
    validate_password,
    validate_username,
    verify_csrf,
    verify_password,
)
from .scene_service import SceneService
from .story_planning_schema import PlannedStoryArc, StoryBlueprint
from .story_planning_service import StoryPlanningService
from .story_plan_suggestion_service import StoryPlanSuggestionService
from .story_planner import build_story_planner
from .story_structure_planner import build_story_structure_planner
from .story_structure_schema import AuthorChapterSkeleton
from .story_structure_service import StoryStructureSuggestionService
from .structure_health import StructureHealthService
from .structure_link_service import StructureLinkService
from .worker import AnalysisWorker
from .style_editor import build_style_editor
from .style_service import StyleService
from .technique_schema import TechniqueObservation
from .technique_service import TechniqueService
from .voice_extraction import build_voice_profile_extractor
from .version_diff import build_version_diff
from .workflow import ChapterWorkflowService
from .writing import build_default_writer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"
ALLOWED_EXTENSIONS = {".txt", ".md", ".text"}
POV_OPTIONS = (
    "第三人称限知",
    "第一人称",
    "第三人称全知",
    "多视角",
)
WORKBENCH_SETTING_TABS = (
    ("core", "作品核心"),
    ("world", "世界规则"),
    ("characters", "人物关系"),
    ("structure", "故事结构"),
    ("style", "文风约束"),
    ("parameters", "创作参数"),
)
WORKBENCH_SETTING_TAB_KEYS = frozenset(
    key for key, _label in WORKBENCH_SETTING_TABS
)
EDIT_PREFERENCE_CATEGORY_OPTIONS = (
    "diction",
    "sentence_rhythm",
    "narration_distance",
    "dialogue",
    "emotional_expression",
    "sensory_detail",
    "metaphor",
    "omission",
    "paragraph_structure",
    "other",
)
STORY_ARC_TYPE_OPTIONS = (
    "main",
    "subplot",
    "character",
    "relationship",
    "mystery",
    "world",
)
STORY_ARC_LIFECYCLE_OPTIONS = (
    "planned",
    "active",
    "paused",
    "resolved",
    "abandoned",
)
CHAPTER_STRUCTURE_ROLE_OPTIONS = (
    "setup",
    "escalation",
    "reversal",
    "payoff",
    "transition",
)


class SlidingWindowLimiter:
    def __init__(self):
        self._events: dict[str, deque[float]] = {}

    def allow(self, key: str, *, limit: int, window_seconds: int) -> bool:
        now = time.monotonic()
        events = self._events.setdefault(key, deque())
        cutoff = now - window_seconds
        while events and events[0] <= cutoff:
            events.popleft()
        if len(events) >= limit:
            return False
        events.append(now)
        if len(self._events) > 10_000:
            self._events = {
                item_key: item_events
                for item_key, item_events in self._events.items()
                if item_events and item_events[-1] > cutoff
            }
        return True


def _client_address(request: Request) -> str:
    direct = request.client.host if request.client else "unknown"
    forwarded = request.headers.get("x-forwarded-for", "")
    if direct in {"127.0.0.1", "::1"} and forwarded:
        return forwarded.split(",", 1)[0].strip()[:128]
    return direct[:128]


def _current_user(request: Request) -> Optional[Dict[str, Any]]:
    user_id = request.session.get("user_id")
    if not isinstance(user_id, int):
        return None
    database: Database = request.app.state.database
    user = database.get_user(user_id)
    if not user:
        request.session.clear()
    return user


def _login_redirect(request: Request) -> RedirectResponse:
    next_path = request.url.path
    return RedirectResponse(
        f"/login?next={quote(next_path, safe='/')}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


def _safe_next(value: str) -> str:
    if value.startswith("/") and not value.startswith("//"):
        return value
    return "/"


def _template_context(
    request: Request,
    *,
    user: Optional[Dict[str, Any]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    settings: Settings = request.app.state.settings
    personal_api_configured = False
    if user:
        database: Database = request.app.state.database
        personal_api_configured = database.has_api_credential(int(user["id"]))
    using_mock = (
        not personal_api_configured
        and not settings.deepseek_api_key
        and settings.uses_test_models
    )
    return {
        "request": request,
        "app_name": settings.app_name,
        "user": user,
        "csrf_token": csrf_token(request),
        "using_mock": using_mock,
        "api_missing": (
            not personal_api_configured
            and not settings.deepseek_api_key
            and not using_mock
        ),
        "personal_api_configured": personal_api_configured,
        **extra,
    }


def _human_size(value: int) -> str:
    if value < 1_000:
        return f"{value} 字"
    if value < 1_000_000:
        return f"{value / 1_000:.1f} 千字"
    return f"{value / 1_000_000:.2f} 百万字"


def _status_label(value: str) -> str:
    return {
        "ready": "等待分析",
        "planned": "待创作",
        "draft": "草稿",
        "canonical": "正史",
        "final": "已定稿",
        "queued": "排队中",
        "running": "分析中",
        "completed": "已完成",
        "partial": "部分完成",
        "failed": "失败",
    }.get(value, value)


def _writing_status_label(value: str) -> str:
    return {
        "queued": "排队中",
        "running": "写作中",
        "completed": "已完成",
        "failed": "失败",
    }.get(value, value)


def _writing_operation_label(value: str) -> str:
    return {
        "draft": "生成初稿",
        "continue": "续写",
        "rewrite": "整章重写",
        "polish": "润色",
        "manual": "手动保存",
        "extract_story_delta": "提取故事记忆",
        "plan_chapter": "规划章节任务卡",
        "plan_scene_beats": "只拆分场景节拍",
        "audit_chapter": "执行正文硬审计",
        "audit_ai_style": "定位 AI 味问题",
        "rewrite_style_issue": "定点改写",
        "targeted_rewrite": "定点改写候选",
        "propose_reader_branches": "评估读者意见",
        "generate_scene": "生成场景",
        "rewrite_scene": "重写场景",
        "audit_scene": "检查场景",
        "scene_assembly": "场景组装",
    }.get(value, value)


def _style_issue_label(value: str) -> str:
    return {
        "abstract_emotion": "抽象概括情绪",
        "over_explanation": "过度解释",
        "uniform_rhythm": "句段节奏过齐",
        "generic_atmosphere": "通用氛围",
        "cliche": "陈词滥调",
        "dialogue_convergence": "人物对话趋同",
        "over_complete_paragraph": "段落过度完整",
        "unnecessary_summary": "无必要总结",
        "repetition": "重复信息",
        "non_specific_detail": "伪具体细节",
    }.get(value, value)


def _voice_suggestion_status_label(value: str) -> str:
    return {
        "queued": "排队中",
        "running": "正在分析",
        "ready": "等待作者审核",
        "applied": "已应用",
        "rejected": "已放弃",
        "failed": "提取失败",
    }.get(value, value)


def _story_plan_status_label(value: str) -> str:
    return {
        "queued": "排队中",
        "running": "正在规划",
        "completed": "可比较与采纳",
        "failed": "生成失败",
    }.get(value, value)


def _story_planning_mode_label(value: str) -> str:
    return {
        "create": "从项目资料建立结构",
        "refine": "优化已确认方向",
        "rethink": "重想未来结构",
    }.get(value, value)


def _chapter_structure_role_label(value: str) -> str:
    return {
        "setup": "建立",
        "escalation": "升级",
        "reversal": "反转",
        "payoff": "兑现",
        "transition": "转场",
    }.get(value, value or "待定义")


def _voice_dimension_label(value: str) -> str:
    return {
        "narration": "叙述距离",
        "rhythm": "句段节奏",
        "dialogue": "对话声音",
        "sensory": "感官与意象",
        "metaphor": "比喻策略",
        "omission": "省略与留白",
    }.get(value, value)


def _edit_preference_category_label(value: str) -> str:
    return {
        "diction": "用词",
        "sentence_rhythm": "句段节奏",
        "narration_distance": "叙述距离",
        "dialogue": "对话",
        "emotional_expression": "情绪表达",
        "sensory_detail": "感官细节",
        "metaphor": "比喻",
        "omission": "留白",
        "paragraph_structure": "段落结构",
        "other": "其他",
    }.get(value, value)


def _edit_preference_status_label(value: str) -> str:
    return {
        "queued": "排队中",
        "running": "正在分析",
        "ready": "等待作者审核",
        "applied": "已确认偏好",
        "rejected": "已放弃",
        "failed": "提取失败",
    }.get(value, value)


def _story_arc_type_label(value: str) -> str:
    return {
        "main": "主线",
        "subplot": "支线",
        "character": "人物弧光",
        "relationship": "关系线",
        "mystery": "谜团线",
        "world": "世界线",
    }.get(value, value)


def _story_arc_lifecycle_label(value: str) -> str:
    return {
        "planned": "计划中",
        "active": "正在推进",
        "paused": "暂缓推进",
        "resolved": "计划收束",
        "abandoned": "已放弃",
    }.get(value, value)


def _reader_request_type_label(value: str) -> str:
    return {
        "pace": "节奏",
        "character": "人物",
        "relationship": "关系",
        "plot": "剧情",
        "world": "世界设定",
        "payoff": "回报 / 爽点",
        "other": "其他",
    }.get(value, value)


def _reader_scope_label(value: str) -> str:
    return {
        "next_chapter": "下一章",
        "next_three": "未来三章",
        "current_volume": "当前分卷",
        "long_term": "长期主线",
    }.get(value, value)


def _reader_status_label(value: str) -> str:
    return {
        "draft": "待评估",
        "proposing": "正在生成方案",
        "reviewing": "等待作者选择",
        "adopted": "已采纳",
        "dismissed": "已归档",
        "failed": "生成失败",
    }.get(value, value)


def _impact_item_type_label(value: str) -> str:
    return {
        "chapter": "后续正史章节",
        "fact": "后续事实",
        "knowledge": "人物知情",
        "plot_thread": "剧情线",
        "foreshadowing": "伏笔",
    }.get(value, value)


def _technique_dimension_label(value: str) -> str:
    return {
        "plot": "剧情",
        "structure": "结构",
        "scene": "场景",
        "pacing": "节奏",
        "information": "信息释放",
        "character": "人物",
        "dialogue": "对话",
        "language": "语言",
        "suspense": "悬念",
    }.get(value, value)


def _technique_scope_label(value: str) -> str:
    return {
        "project": "全书",
        "volume": "分卷",
        "chapter": "章节",
        "scene": "场景",
    }.get(value, value)


def _technique_usage_label(value: str) -> str:
    return {
        "plan": "规划",
        "write": "正文",
        "audit": "审校",
    }.get(value, value)


def _scene_draft_status_label(value: str) -> str:
    return {
        "empty": "尚未写作",
        "draft": "场景草稿",
        "stale": "任务卡变化，待重写",
        "assembled": "已组装进候选章",
    }.get(value, value)


def _continuity_issue_label(value: str) -> str:
    return {
        "state_before_mismatch": "人物状态前后不一致",
        "relationship_before_mismatch": "人物关系前后不一致",
        "location_before_mismatch": "地点连续性冲突",
        "item_holder_mismatch": "物品持有者冲突",
        "item_after_destroyed": "已毁物品再次出现",
        "story_time_mismatch": "故事时间衔接冲突",
        "missing_baseline": "缺少可核对的前置状态",
        "knowledge_without_baseline": "遗忘缺少知情基线",
        "plot_thread_without_setup": "剧情线缺少建立记录",
        "plot_thread_duplicate_open": "剧情线重复建立",
        "plot_thread_reopened": "已关闭剧情线重新开启",
        "plot_thread_after_closed": "已关闭剧情线继续推进",
        "foreshadow_without_setup": "伏笔缺少埋设记录",
        "foreshadow_duplicate_setup": "伏笔重复埋设",
        "foreshadow_reopened": "已关闭伏笔重新埋设",
        "foreshadow_after_closed": "已关闭伏笔继续推进",
        "duplicate_event_identity": "事件身份重复",
        "causal_self_reference": "事件因果自指",
        "causal_reference_missing": "直接原因事件缺失",
    }.get(value, value)


def _memory_identity_type_label(value: str) -> str:
    return IDENTITY_TYPE_LABELS.get(value, value)


def _knowledge_state_label(value: str) -> str:
    return {
        "knows": "知道",
        "suspects": "怀疑",
        "believes_false": "相信错误信息",
        "forgets": "已经遗忘",
    }.get(value, value)


def _plot_status_label(value: str) -> str:
    return {
        "open": "已建立",
        "active": "推进中",
        "paused": "暂缓",
        "resolved": "已解决",
        "abandoned": "已放弃",
    }.get(value, value)


def _foreshadow_status_label(value: str) -> str:
    return {
        "setup": "已埋设",
        "advanced": "已推进",
        "payoff": "已回收",
        "abandoned": "已放弃",
    }.get(value, value)


def _clean_field(
    value: str,
    label: str,
    *,
    max_length: int,
    required: bool = False,
    min_length: int = 1,
) -> str:
    cleaned = value.strip()
    if required and len(cleaned) < min_length:
        raise ValueError(f"{label}至少需要 {min_length} 个字符")
    if len(cleaned) > max_length:
        raise ValueError(f"{label}不能超过 {max_length:,} 个字符")
    return cleaned


def _split_lines(value: str, *, limit: int = 30) -> list[str]:
    items = [line.strip() for line in value.splitlines() if line.strip()]
    if len(items) > limit:
        raise ValueError(f"逐行条目不能超过 {limit} 条")
    return items


def _story_plan_lines(
    value: str,
    label: str,
    *,
    limit: int,
    item_max_length: int = 1200,
) -> list[str]:
    items = _split_lines(value, limit=limit)
    for item in items:
        if len(item) > item_max_length:
            raise ValueError(
                f"{label}中每条不能超过 {item_max_length:,} 个字符"
            )
    return items


def _story_blueprint_from_form(
    *,
    central_question: str,
    protagonist_goal: str,
    core_conflict: str,
    stakes: str,
    opening_state: str,
    ending_state: str,
    major_turns: str,
    must_payoffs: str,
    forbidden_shortcuts: str,
    author_notes: str,
) -> StoryBlueprint:
    return StoryBlueprint.model_validate(
        {
            "central_question": _clean_field(
                central_question, "核心悬问", max_length=2000
            ),
            "protagonist_goal": _clean_field(
                protagonist_goal, "主角长期目标", max_length=2000
            ),
            "core_conflict": _clean_field(
                core_conflict, "全书冲突引擎", max_length=3000
            ),
            "stakes": _clean_field(
                stakes, "长期代价与风险", max_length=3000
            ),
            "opening_state": _clean_field(
                opening_state, "开篇状态", max_length=3000
            ),
            "ending_state": _clean_field(
                ending_state, "终局状态", max_length=3000
            ),
            "major_turns": _story_plan_lines(
                major_turns, "全书转折", limit=20
            ),
            "must_payoffs": _story_plan_lines(
                must_payoffs, "必须兑现项", limit=30
            ),
            "forbidden_shortcuts": _story_plan_lines(
                forbidden_shortcuts, "禁止捷径", limit=30
            ),
            "author_notes": _clean_field(
                author_notes, "蓝图作者备注", max_length=6000
            ),
        }
    )


def _planned_story_arc_from_form(
    *,
    arc_type: str,
    title: str,
    dramatic_question: str,
    promise: str,
    start_state: str,
    target_payoff: str,
    involved_characters: str,
    planned_turns: str,
    lifecycle_status: str,
    priority: int,
    author_notes: str,
) -> PlannedStoryArc:
    if arc_type not in STORY_ARC_TYPE_OPTIONS:
        raise ValueError("请选择有效的剧情线类型")
    if lifecycle_status not in STORY_ARC_LIFECYCLE_OPTIONS:
        raise ValueError("请选择有效的剧情线阶段")
    return PlannedStoryArc.model_validate(
        {
            "arc_type": arc_type,
            "title": _clean_field(
                title, "剧情线名称", max_length=160, required=True
            ),
            "dramatic_question": _clean_field(
                dramatic_question, "剧情线悬问", max_length=2000
            ),
            "promise": _clean_field(
                promise, "剧情线读者承诺", max_length=2000
            ),
            "start_state": _clean_field(
                start_state, "剧情线起始状态", max_length=2000
            ),
            "target_payoff": _clean_field(
                target_payoff, "剧情线目标回报", max_length=3000
            ),
            "involved_characters": _story_plan_lines(
                involved_characters, "涉及人物", limit=30, item_max_length=120
            ),
            "planned_turns": _story_plan_lines(
                planned_turns, "剧情线转折", limit=20
            ),
            "lifecycle_status": lifecycle_status,
            "priority": priority,
            "author_notes": _clean_field(
                author_notes, "剧情线作者备注", max_length=4000
            ),
        }
    )


def _technique_observation_from_form(
    *,
    name: str,
    dimension: str,
    source_location: str,
    observation: str,
    effect: str,
    suitable_for: str,
    unsuitable_for: str,
    execution_rule: str,
    originality_boundary: str,
) -> TechniqueObservation:
    return TechniqueObservation.model_validate(
        {
            "name": _clean_field(
                name, "技法名称", max_length=80, required=True, min_length=2
            ),
            "dimension": dimension,
            "source_location": _clean_field(
                source_location,
                "来源位置",
                max_length=200,
                required=True,
            ),
            "observation": _clean_field(
                observation,
                "文本观察",
                max_length=600,
                required=True,
                min_length=10,
            ),
            "effect": _clean_field(
                effect,
                "读者效果",
                max_length=600,
                required=True,
                min_length=10,
            ),
            "suitable_for": _split_lines(suitable_for, limit=8),
            "unsuitable_for": _split_lines(unsuitable_for, limit=8),
            "execution_rule": _clean_field(
                execution_rule,
                "执行规则",
                max_length=600,
                required=True,
                min_length=10,
            ),
            "originality_boundary": _clean_field(
                originality_boundary,
                "原创性边界",
                max_length=600,
                required=True,
                min_length=10,
            ),
        }
    )


def _read_optional_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _read_utf8_with_file_hash(path: Path) -> tuple[str, str]:
    raw_content = path.read_bytes()
    content = raw_content.decode("utf-8")
    normalized_content = content.replace("\r\n", "\n").replace("\r", "\n")
    return normalized_content, hashlib.sha256(raw_content).hexdigest()


def _atomic_write_text(path: Path, content: str, token: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{token}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)
    path.chmod(0o600)


def _diff_segments(original: str, replacement: str) -> dict[str, list[dict]]:
    before: list[dict[str, str]] = []
    after: list[dict[str, str]] = []
    matcher = SequenceMatcher(None, original, replacement, autojunk=False)
    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        old_text = original[old_start:old_end]
        new_text = replacement[new_start:new_end]
        if tag == "equal":
            if old_text:
                before.append({"kind": "same", "text": old_text})
                after.append({"kind": "same", "text": new_text})
        elif tag == "delete":
            before.append({"kind": "removed", "text": old_text})
        elif tag == "insert":
            after.append({"kind": "added", "text": new_text})
        else:
            before.append({"kind": "removed", "text": old_text})
            after.append({"kind": "added", "text": new_text})
    return {"before": before, "after": after}


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    app_settings = settings or Settings.from_env()
    app_settings.validate()
    app_settings.ensure_directories()
    database = Database(app_settings.database_path)
    memory_service = MemoryService(database)
    planning_service = PlanningService(database)
    story_planning_service = StoryPlanningService(database)
    story_plan_suggestion_service = StoryPlanSuggestionService(database)
    story_structure_suggestion_service = StoryStructureSuggestionService(
        database, app_settings.novels_dir
    )
    structure_health_service = StructureHealthService(database)
    structure_link_service = StructureLinkService(database)
    causal_suggestion_service = CausalSuggestionService(database)
    causal_branch_service = CausalBranchSimulationService(database)
    causal_branch_adoption_service = CausalBranchAdoptionService(database)
    style_service = StyleService(database)
    preference_service = PreferenceService(database)
    reader_service = ReaderDecisionService(
        database, app_settings.novels_dir
    )
    assistant_chat_service = AssistantChatService(
        database,
        app_settings.novels_dir,
        app_settings.documents_dir,
    )
    impact_service = CanonImpactService(database)
    technique_service = TechniqueService(database)
    scene_service = SceneService(database)
    workflow_service = ChapterWorkflowService(
        database,
        planning_service=planning_service,
        scene_service=scene_service,
        memory_service=memory_service,
        style_service=style_service,
    )
    continuity_service = ContinuityService(database)
    identity_service = MemoryIdentityService(database)
    credential_cipher = CredentialCipher(app_settings.credential_secret)
    auth_limiter = SlidingWindowLimiter()
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.filters["human_size"] = _human_size
    templates.env.filters["status_label"] = _status_label
    templates.env.filters["writing_status_label"] = _writing_status_label
    templates.env.filters["writing_operation_label"] = _writing_operation_label
    templates.env.filters["style_issue_label"] = _style_issue_label
    templates.env.filters[
        "voice_suggestion_status_label"
    ] = _voice_suggestion_status_label
    templates.env.filters[
        "voice_dimension_label"
    ] = _voice_dimension_label
    templates.env.filters[
        "edit_preference_category_label"
    ] = _edit_preference_category_label
    templates.env.filters[
        "edit_preference_status_label"
    ] = _edit_preference_status_label
    templates.env.filters["story_arc_type_label"] = _story_arc_type_label
    templates.env.filters[
        "story_arc_lifecycle_label"
    ] = _story_arc_lifecycle_label
    templates.env.filters[
        "story_plan_status_label"
    ] = _story_plan_status_label
    templates.env.filters[
        "story_planning_mode_label"
    ] = _story_planning_mode_label
    templates.env.filters[
        "chapter_structure_role_label"
    ] = _chapter_structure_role_label
    templates.env.filters[
        "reader_request_type_label"
    ] = _reader_request_type_label
    templates.env.filters["reader_scope_label"] = _reader_scope_label
    templates.env.filters["reader_status_label"] = _reader_status_label
    templates.env.filters[
        "impact_item_type_label"
    ] = _impact_item_type_label
    templates.env.filters[
        "technique_dimension_label"
    ] = _technique_dimension_label
    templates.env.filters["technique_scope_label"] = _technique_scope_label
    templates.env.filters["technique_usage_label"] = _technique_usage_label
    templates.env.filters[
        "scene_draft_status_label"
    ] = _scene_draft_status_label
    templates.env.filters[
        "continuity_issue_label"
    ] = _continuity_issue_label
    templates.env.filters[
        "knowledge_state_label"
    ] = _knowledge_state_label
    templates.env.filters["plot_status_label"] = _plot_status_label
    templates.env.filters[
        "foreshadow_status_label"
    ] = _foreshadow_status_label
    templates.env.filters[
        "memory_identity_type_label"
    ] = _memory_identity_type_label

    def render_template(
        name: str, context: Dict[str, Any], status_code: int = status.HTTP_200_OK
    ):
        return templates.TemplateResponse(
            context["request"], name, context, status_code=status_code
        )

    def api_profile(user_id: int) -> Optional[Dict[str, str]]:
        credential = database.get_api_credential_summary(user_id)
        if credential:
            return {
                "provider": str(credential["provider"]),
                "model": str(credential["model"]),
                "credential_source": "personal",
            }
        if app_settings.deepseek_api_key:
            return {
                "provider": "deepseek",
                "model": app_settings.deepseek_model,
                "credential_source": "default",
            }
        if app_settings.uses_test_models:
            return {
                "provider": "mock",
                "model": "mock-novel-writer",
                "credential_source": "default",
            }
        return None

    async def after_canon_acceptance(
        request: Request,
        *,
        user_id: int,
        project_id: str,
        chapter_id: str,
        version_id: str,
    ) -> Response:
        deltas = memory_service.list_chapter_deltas(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
        )
        active_delta = next(
            (
                item
                for item in deltas
                if str(item["version_id"]) == version_id
                and str(item["status"])
                in {"proposed", "author_edited", "projected"}
            ),
            None,
        )
        if active_delta:
            if str(active_delta["status"]) in {"proposed", "author_edited"}:
                return RedirectResponse(
                    f"/story-deltas/{active_delta['id']}",
                    status_code=status.HTTP_303_SEE_OTHER,
                )
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}?canonical=true",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        profile = api_profile(user_id)
        if not profile:
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}"
                "?canonical=true&error="
                + quote(
                    "正文已成为正史；配置模型服务后可提取故事记忆"
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        try:
            job_id = database.create_memory_extraction_job(
                user_id=user_id,
                project_id=project_id,
                chapter_id=chapter_id,
                version_id=version_id,
                provider=profile["provider"],
                model=profile["model"],
                credential_source=profile["credential_source"],
                max_jobs_per_day=app_settings.max_jobs_per_day,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}"
                f"?canonical=true&error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        request.app.state.worker.wake()
        return RedirectResponse(
            f"/writing-jobs/{job_id}", status_code=status.HTTP_303_SEE_OTHER
        )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        previous_umask = os.umask(0o077)
        process_lock = ProcessLock(app_settings.data_dir / ".worker.lock")
        worker = None
        worker_task = None
        try:
            process_lock.acquire()
            database.initialize()
            app_settings.ensure_directories()
            default_provider_configured = bool(
                app_settings.deepseek_api_key
            ) or app_settings.uses_test_models
            if default_provider_configured:
                analyzer = build_analyzer(app_settings)
                writer = build_default_writer(app_settings)
                memory_extractor = build_memory_extractor(app_settings)
                chapter_planner = build_chapter_planner(app_settings)
                quality_auditor = build_quality_auditor(app_settings)
                style_editor = build_style_editor(app_settings)
                reader_planner = build_reader_planner(app_settings)
                story_planner = build_story_planner(app_settings)
                story_structure_planner = (
                    build_story_structure_planner(app_settings)
                )
                causal_suggestion_planner = (
                    build_causal_suggestion_planner(app_settings)
                )
                causal_branch_planner = build_causal_branch_planner(
                    app_settings
                )
                voice_profile_extractor = (
                    build_voice_profile_extractor(app_settings)
                )
                edit_preference_extractor = (
                    build_edit_preference_extractor(app_settings)
                )
                assistant_chat_model = build_assistant_chat_model(
                    app_settings
                )
            else:
                analyzer = None
                writer = None
                memory_extractor = None
                chapter_planner = None
                quality_auditor = None
                style_editor = None
                reader_planner = None
                story_planner = None
                story_structure_planner = None
                causal_suggestion_planner = None
                causal_branch_planner = None
                voice_profile_extractor = None
                edit_preference_extractor = None
                assistant_chat_model = None
            worker = AnalysisWorker(
                database,
                analyzer,
                writer,
                app_settings.secret_key,
                app_settings,
                credential_cipher,
                memory_extractor=memory_extractor,
                chapter_planner=chapter_planner,
                quality_auditor=quality_auditor,
                style_editor=style_editor,
                reader_planner=reader_planner,
                story_planner=story_planner,
                story_structure_planner=story_structure_planner,
                causal_suggestion_planner=causal_suggestion_planner,
                causal_branch_planner=causal_branch_planner,
                voice_profile_extractor=voice_profile_extractor,
                edit_preference_extractor=edit_preference_extractor,
                assistant_chat_model=assistant_chat_model,
                poll_seconds=app_settings.worker_poll_seconds,
            )
            application.state.analyzer = analyzer
            application.state.writer = writer
            application.state.memory_extractor = memory_extractor
            application.state.chapter_planner = chapter_planner
            application.state.quality_auditor = quality_auditor
            application.state.style_editor = style_editor
            application.state.reader_planner = reader_planner
            application.state.story_planner = story_planner
            application.state.story_structure_planner = (
                story_structure_planner
            )
            application.state.causal_suggestion_planner = (
                causal_suggestion_planner
            )
            application.state.causal_branch_planner = (
                causal_branch_planner
            )
            application.state.voice_profile_extractor = (
                voice_profile_extractor
            )
            application.state.edit_preference_extractor = (
                edit_preference_extractor
            )
            application.state.assistant_chat_model = assistant_chat_model
            application.state.worker = worker
            application.state.credential_cipher = credential_cipher
            application.state.password_slots = asyncio.Semaphore(2)
            worker_task = asyncio.create_task(worker.run(), name="analysis-worker")
            application.state.worker_task = worker_task
            yield
        finally:
            if worker_task is not None:
                worker_task.cancel()
                try:
                    await worker_task
                except asyncio.CancelledError:
                    pass
            if worker is not None:
                await worker.stop()
            process_lock.release()
            os.umask(previous_umask)

    application = FastAPI(
        title=app_settings.app_name,
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    application.state.settings = app_settings
    application.state.database = database
    application.state.workflow_service = workflow_service
    application.state.structure_health_service = structure_health_service
    application.state.structure_link_service = structure_link_service
    application.state.causal_suggestion_service = causal_suggestion_service
    application.state.causal_branch_service = causal_branch_service
    application.state.causal_branch_adoption_service = (
        causal_branch_adoption_service
    )
    application.state.assistant_chat_service = assistant_chat_service
    application.state.templates = templates
    application.add_middleware(
        SessionMiddleware,
        secret_key=app_settings.secret_key,
        session_cookie="story_session",
        max_age=60 * 60 * 24 * 30,
        same_site="lax",
        https_only=app_settings.cookie_secure,
    )
    application.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @application.get("/healthz", include_in_schema=False)
    async def healthz():
        worker = getattr(application.state, "worker", None)
        worker_task = getattr(application.state, "worker_task", None)
        try:
            database_ok = await asyncio.to_thread(database.ping)
        except Exception:
            logger.exception("health check database failure")
            database_ok = False
        worker_ok = bool(
            worker
            and worker.healthy
            and worker_task
            and not worker_task.done()
        )
        healthy = database_ok and worker_ok
        return JSONResponse(
            {
                "status": "ok" if healthy else "unhealthy",
                "analyzer": (
                    "mock"
                    if app_settings.uses_test_models
                    else (
                        "deepseek"
                        if app_settings.deepseek_api_key
                        else "personal-key-only"
                    )
                ),
                "database": "ok" if database_ok else "unavailable",
                "worker": "ok" if worker_ok else "unavailable",
            },
            status_code=(
                status.HTTP_200_OK
                if healthy
                else status.HTTP_503_SERVICE_UNAVAILABLE
            ),
        )

    @application.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        if _current_user(request):
            return RedirectResponse("/dashboard", status_code=status.HTTP_303_SEE_OTHER)
        return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)

    @application.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request, next: str = "/"):
        if _current_user(request):
            return RedirectResponse(
                _safe_next(next), status_code=status.HTTP_303_SEE_OTHER
            )
        return render_template(
            "login.html",
            _template_context(
                request,
                next_path=_safe_next(next),
                can_register=app_settings.allow_registration,
            ),
        )

    @application.post("/login", response_class=HTMLResponse)
    async def login(
        request: Request,
        username: str = Form(...),
        password: str = Form(...),
        csrf: str = Form(...),
        next: str = Form("/"),
    ):
        verify_csrf(request, csrf)
        client_address = _client_address(request)
        clean_login = username.strip()
        if not auth_limiter.allow(
            f"login-ip:{client_address}", limit=20, window_seconds=300
        ) or not auth_limiter.allow(
            f"login-user:{client_address}:{clean_login.casefold()}",
            limit=10,
            window_seconds=300,
        ):
            return render_template(
                "login.html",
                _template_context(
                    request,
                    error="尝试次数过多，请稍后再试",
                    next_path=_safe_next(next),
                    can_register=app_settings.allow_registration,
                ),
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            )
        user = database.get_user_by_username(clean_login)
        password_ok = False
        if user:
            async with request.app.state.password_slots:
                password_ok = await asyncio.to_thread(
                    verify_password, password, str(user["password_hash"])
                )
        if not user or not password_ok:
            return render_template(
                "login.html",
                _template_context(
                    request,
                    error="用户名或密码不正确",
                    next_path=_safe_next(next),
                    can_register=app_settings.allow_registration,
                ),
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        request.session.clear()
        request.session["user_id"] = int(user["id"])
        csrf_token(request)
        return RedirectResponse(
            _safe_next(next), status_code=status.HTTP_303_SEE_OTHER
        )

    @application.get("/register", response_class=HTMLResponse)
    async def register_page(request: Request):
        if _current_user(request):
            return RedirectResponse("/dashboard", status_code=status.HTTP_303_SEE_OTHER)
        if not app_settings.allow_registration:
            return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)
        return render_template(
            "register.html", _template_context(request)
        )

    @application.post("/register", response_class=HTMLResponse)
    async def register(
        request: Request,
        username: str = Form(...),
        password: str = Form(...),
        password_confirm: str = Form(...),
        csrf: str = Form(...),
    ):
        verify_csrf(request, csrf)
        if not app_settings.allow_registration:
            return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)
        if not auth_limiter.allow(
            f"register:{_client_address(request)}",
            limit=5,
            window_seconds=60 * 60,
        ):
            return render_template(
                "register.html",
                _template_context(
                    request,
                    error="注册尝试次数过多，请稍后再试",
                    username=username,
                ),
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            )
        try:
            clean_username = validate_username(username)
            validate_password(password)
            if password != password_confirm:
                raise ValueError("两次输入的密码不一致")
            async with request.app.state.password_slots:
                password_hash = await asyncio.to_thread(hash_password, password)
            user_id = database.create_user(clean_username, password_hash)
        except ValueError as exc:
            return render_template(
                "register.html",
                _template_context(request, error=str(exc), username=username),
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        except Exception as exc:
            duplicate = "UNIQUE constraint failed" in str(exc)
            if not duplicate:
                logger.exception("failed to register user")
            return render_template(
                "register.html",
                _template_context(
                    request,
                    error="该用户名已存在" if duplicate else "创建账号失败，请稍后重试",
                    username=username,
                ),
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        request.session.clear()
        request.session["user_id"] = user_id
        csrf_token(request)
        return RedirectResponse("/dashboard", status_code=status.HTTP_303_SEE_OTHER)

    @application.post("/logout")
    async def logout(request: Request, csrf: str = Form(...)):
        verify_csrf(request, csrf)
        request.session.clear()
        return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)

    def render_api_settings(
        request: Request,
        user: Dict[str, Any],
        *,
        error: Optional[str] = None,
        saved: bool = False,
        removed: bool = False,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        system_prompt: Optional[str] = None,
        status_code: int = status.HTTP_200_OK,
    ):
        credential = database.get_api_credential_summary(int(user["id"]))
        current_provider = provider or (
            str(credential["provider"]) if credential else "deepseek"
        )
        try:
            current_provider_spec = get_provider(current_provider)
        except ProviderConfigError:
            current_provider_spec = get_provider("deepseek")
            current_provider = current_provider_spec.id
        current_model = model or (
            str(credential["model"])
            if credential and str(credential["provider"]) == current_provider
            else app_settings.deepseek_model
        )
        current_thinking = (
            thinking
            if thinking is not None
            else (
                bool(credential["thinking"])
                if credential
                else app_settings.deepseek_thinking
            )
        )
        current_effort = reasoning_effort or (
            str(credential["reasoning_effort"])
            if credential
            else app_settings.deepseek_reasoning_effort
        )
        current_system_prompt = (
            system_prompt
            if system_prompt is not None
            else (
                str(credential.get("system_prompt") or "")
                if credential
                else (
                    app_settings.deepseek_system_prompt
                    or DEFAULT_SYSTEM_PROMPT
                )
            )
        )
        return render_template(
            "api_settings.html",
            _template_context(
                request,
                user=user,
                credential=credential,
                providers=[
                    item.public_payload() for item in list_providers()
                ],
                selected_provider=current_provider,
                selected_provider_spec=current_provider_spec.public_payload(),
                error=error,
                saved=saved,
                removed=removed,
                selected_model=current_model,
                selected_thinking=current_thinking,
                selected_effort=current_effort,
                selected_system_prompt=current_system_prompt,
                default_system_prompt=DEFAULT_SYSTEM_PROMPT,
                server_api_available=bool(app_settings.deepseek_api_key),
            ),
            status_code=status_code,
        )

    @application.get("/settings/api", response_class=HTMLResponse)
    async def api_settings_page(
        request: Request,
        saved: bool = False,
        removed: bool = False,
        error: Optional[str] = None,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        return render_api_settings(
            request, user, error=error, saved=saved, removed=removed
        )

    @application.post("/settings/api", response_class=HTMLResponse)
    async def save_api_settings(
        request: Request,
        api_key: str = Form(""),
        provider: str = Form("deepseek"),
        model: str = Form(...),
        thinking: Optional[str] = Form(None),
        reasoning_effort: str = Form("high"),
        system_prompt: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        thinking_enabled = thinking == "enabled"
        try:
            provider_spec = get_provider(provider)
            clean_model = validate_model(model)
            clean_system_prompt = _clean_field(
                system_prompt,
                "全局系统提示词",
                max_length=20_000,
            )
            if reasoning_effort not in {"high", "max"}:
                raise CredentialError("思考强度只能选择 high 或 max")
            if (
                thinking_enabled
                and not provider_spec.capabilities.thinking
            ):
                raise CredentialError(
                    f"{provider_spec.label} 暂不支持 novelAI 的思考模式"
                )
            existing = database.get_api_credential(int(user["id"]))
            if api_key:
                clean_key = validate_api_key(api_key)
                encrypted_key = credential_cipher.encrypt(clean_key)
                masked_key = key_hint(clean_key)
            elif (
                existing
                and str(existing["provider"]) == provider_spec.id
            ):
                encrypted_key = str(existing["encrypted_key"])
                masked_key = str(existing["key_hint"])
            elif not provider_spec.capabilities.api_key_required:
                encrypted_key = credential_cipher.encrypt("")
                masked_key = "无需 Key"
            else:
                raise CredentialError(
                    f"请填写 {provider_spec.label} API Key"
                )
            database.upsert_api_credential(
                user_id=int(user["id"]),
                provider=provider_spec.id,
                encrypted_key=encrypted_key,
                key_hint=masked_key,
                model=clean_model,
                thinking=thinking_enabled,
                reasoning_effort=reasoning_effort,
                system_prompt=clean_system_prompt,
            )
        except ValueError as exc:
            return render_api_settings(
                request,
                user,
                error=str(exc),
                provider=provider,
                model=model,
                thinking=thinking_enabled,
                reasoning_effort=reasoning_effort,
                system_prompt=system_prompt,
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        return RedirectResponse(
            "/settings/api?saved=true", status_code=status.HTTP_303_SEE_OTHER
        )

    @application.post("/api/settings/models")
    async def api_model_catalog(
        request: Request,
        api_key: str = Form(""),
        provider: str = Form("deepseek"),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return JSONResponse(
                {"error": "unauthorized"},
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        verify_csrf(request, csrf)
        try:
            provider_spec = get_provider(provider)
            if api_key:
                clean_key = validate_api_key(api_key)
            else:
                existing = database.get_api_credential(int(user["id"]))
                if (
                    existing
                    and str(existing["provider"]) == provider_spec.id
                ):
                    clean_key = credential_cipher.decrypt(
                        str(existing["encrypted_key"])
                    )
                elif (
                    provider_spec.id == "deepseek"
                    and app_settings.deepseek_api_key
                ):
                    clean_key = app_settings.deepseek_api_key
                elif not provider_spec.capabilities.api_key_required:
                    clean_key = None
                else:
                    raise ModelCatalogError(
                        f"请先输入 {provider_spec.label} API Key"
                    )
            models = await fetch_models(
                provider_id=provider_spec.id,
                api_key=clean_key,
                timeout_seconds=app_settings.deepseek_connect_timeout_seconds,
            )
        except (
            CredentialError,
            ModelCatalogError,
            ProviderConfigError,
        ) as exc:
            return JSONResponse(
                {"error": str(exc)},
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        return JSONResponse(
            {"models": models},
            headers={"Cache-Control": "no-store"},
        )

    @application.post("/settings/api/delete")
    async def delete_api_settings(request: Request, csrf: str = Form(...)):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            database.delete_api_credential(int(user["id"]))
        except ValueError as exc:
            return RedirectResponse(
                "/settings/api?error=" + quote(str(exc)),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            "/settings/api?removed=true", status_code=status.HTTP_303_SEE_OTHER
        )

    @application.get("/dashboard", response_class=HTMLResponse)
    async def dashboard(
        request: Request,
        error: Optional[str] = None,
        deleted: bool = False,
        imported: bool = False,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        projects = database.list_novel_projects(int(user["id"]))
        documents = database.list_documents(int(user["id"]))
        technique_count = technique_service.count_cards(
            user_id=int(user["id"])
        )
        return render_template(
            "dashboard.html",
            _template_context(
                request,
                user=user,
                projects=projects,
                documents=documents,
                technique_count=technique_count,
                error=error,
                deleted=deleted,
                imported=imported,
            ),
        )

    @application.get(
        "/novels/{project_id}/workbench",
        response_class=HTMLResponse,
    )
    async def unified_novel_workbench(
        request: Request,
        project_id: str,
        chapter_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        view: str = "body",
        settings_tab: str = "core",
        onboarding: bool = False,
        error: Optional[str] = None,
        saved: bool = False,
        sent: bool = False,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        user_id = int(user["id"])
        project = database.get_novel_project(user_id, project_id)
        if not project:
            return render_template(
                "not_found.html",
                _template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        chapters = database.list_novel_chapters(user_id, project_id)
        effective_view = view if view in {"body", "settings"} else "body"
        if not chapters:
            effective_view = "settings"
        active_settings_tab = (
            settings_tab
            if settings_tab in WORKBENCH_SETTING_TAB_KEYS
            else "core"
        )
        display_title = str(project.get("title") or "").strip()
        display_title = display_title or "未命名作品"
        selected_chapter = None
        if effective_view == "body" and chapter_id:
            selected_chapter = next(
                (
                    item
                    for item in chapters
                    if str(item["id"]) == chapter_id
                ),
                None,
            )
            if not selected_chapter:
                return render_template(
                    "not_found.html",
                    _template_context(request, user=user),
                    status_code=status.HTTP_404_NOT_FOUND,
                )
        elif effective_view == "body" and chapters:
            selected_chapter = chapters[0]

        content = ""
        selected_index = -1
        previous_chapter = None
        next_chapter = None
        working_version = None
        working_version_hash = ""
        chapter_effective_chars = 0
        chapter_target_chars = int(
            project.get("target_chapter_chars") or 3000
        )
        chapter_workflow = None
        if selected_chapter:
            selected_index = next(
                index
                for index, item in enumerate(chapters)
                if str(item["id"]) == str(selected_chapter["id"])
            )
            previous_chapter = (
                chapters[selected_index - 1]
                if selected_index > 0
                else None
            )
            next_chapter = (
                chapters[selected_index + 1]
                if selected_index + 1 < len(chapters)
                else None
            )
            content = _read_optional_text(
                Path(str(selected_chapter["content_path"]))
            )
            chapter_effective_chars = effective_char_count(content)
            task_card = planning_service.get_task_card(
                user_id=user_id,
                project_id=project_id,
                chapter_id=str(selected_chapter["id"]),
            )
            if task_card:
                chapter_target_chars = int(
                    task_card.get("target_chars")
                    or chapter_target_chars
                )
            versions = database.list_chapter_versions(
                user_id, project_id, str(selected_chapter["id"])
            )
            working_version = next(
                (
                    version
                    for version in versions
                    if str(version["id"])
                    == str(
                        selected_chapter.get("working_version_id")
                        or ""
                    )
                ),
                None,
            )
            if working_version:
                version_content = _read_optional_text(
                    Path(str(working_version["content_path"]))
                )
                if version_content == content:
                    working_version_hash = hashlib.sha256(
                        content.encode("utf-8")
                    ).hexdigest()
                else:
                    working_version = None
            chapter_workflow = workflow_service.get_state(
                user_id=user_id,
                project_id=project_id,
                chapter_id=str(selected_chapter["id"]),
            )

        conversations = assistant_chat_service.list_project_conversations(
            user_id=user_id,
            project_id=project_id,
        )
        active_conversation = None
        if conversation_id:
            active_conversation = (
                assistant_chat_service.get_conversation(
                    user_id=user_id,
                    conversation_id=conversation_id,
                )
            )
            if (
                not active_conversation
                or str(active_conversation.get("project_id") or "")
                != project_id
            ):
                return Response(
                    status_code=status.HTTP_404_NOT_FOUND
                )
        elif effective_view == "settings":
            latest = next(
                (
                    item
                    for item in conversations
                    if str(item.get("scope_type") or "") == "project"
                ),
                None,
            )
            if latest:
                active_conversation = (
                    assistant_chat_service.get_conversation(
                        user_id=user_id,
                        conversation_id=str(latest["id"]),
                    )
                )
        elif selected_chapter:
            latest = next(
                (
                    item
                    for item in conversations
                    if str(item.get("novel_chapter_id") or "")
                    == str(selected_chapter["id"])
                ),
                None,
            )
            if latest:
                active_conversation = (
                    assistant_chat_service.get_conversation(
                        user_id=user_id,
                        conversation_id=str(latest["id"]),
                    )
                )

        setting_characters: list[dict[str, Any]] = []
        setting_story_blueprint = None
        setting_story_arcs: list[dict[str, Any]] = []
        setting_voice_profile = None
        if effective_view == "settings":
            setting_characters = database.list_novel_characters(
                user_id, project_id
            )
            setting_story_blueprint = story_planning_service.get_blueprint(
                user_id=user_id, project_id=project_id
            )
            setting_story_arcs = story_planning_service.list_arcs(
                user_id=user_id, project_id=project_id
            )
            setting_voice_profile = style_service.get_voice_profile(
                user_id=user_id, project_id=project_id
            )

        return render_template(
            "novel_workbench.html",
            _template_context(
                request,
                user=user,
                project=project,
                display_title=display_title,
                chapters=chapters,
                chapter=selected_chapter,
                chapter_content=content,
                chapter_index=selected_index,
                previous_chapter=previous_chapter,
                next_chapter=next_chapter,
                conversations=conversations,
                active_conversation=active_conversation,
                working_version=working_version,
                working_version_hash=working_version_hash,
                chapter_effective_chars=chapter_effective_chars,
                chapter_target_chars=chapter_target_chars,
                chapter_workflow=chapter_workflow,
                view=effective_view,
                setting_tabs=WORKBENCH_SETTING_TABS,
                active_settings_tab=active_settings_tab,
                setting_characters=setting_characters,
                setting_story_blueprint=setting_story_blueprint,
                setting_story_arcs=setting_story_arcs,
                setting_voice_profile=setting_voice_profile,
                pov_options=POV_OPTIONS,
                setting_field_labels=SETTING_FIELD_LABELS,
                onboarding=onboarding,
                error=error,
                saved=saved,
                sent=sent,
            ),
        )

    @application.get("/techniques", response_class=HTMLResponse)
    async def technique_library(
        request: Request,
        error: Optional[str] = None,
        created: bool = False,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        cards = technique_service.list_cards(user_id=int(user["id"]))
        return render_template(
            "techniques.html",
            _template_context(
                request,
                user=user,
                cards=cards,
                error=error,
                created=created,
            ),
        )

    @application.post("/techniques")
    async def create_manual_technique(
        request: Request,
        name: str = Form(...),
        dimension: str = Form(...),
        source_location: str = Form("作者手动总结"),
        observation: str = Form(...),
        effect: str = Form(...),
        suitable_for: str = Form(""),
        unsuitable_for: str = Form(""),
        execution_rule: str = Form(...),
        originality_boundary: str = Form(...),
        author_note: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            card_id = technique_service.create_manual(
                user_id=int(user["id"]),
                observation=_technique_observation_from_form(
                    name=name,
                    dimension=dimension,
                    source_location=source_location,
                    observation=observation,
                    effect=effect,
                    suitable_for=suitable_for,
                    unsuitable_for=unsuitable_for,
                    execution_rule=execution_rule,
                    originality_boundary=originality_boundary,
                ),
                author_note=_clean_field(
                    author_note, "作者备注", max_length=2000
                ),
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/techniques?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/techniques/{card_id}?created=true",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.get(
        "/techniques/{technique_id}", response_class=HTMLResponse
    )
    async def technique_card_page(
        request: Request,
        technique_id: str,
        error: Optional[str] = None,
        saved: bool = False,
        created: bool = False,
        bound: bool = False,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        card = technique_service.get_card(
            user_id=int(user["id"]), technique_id=technique_id
        )
        if not card:
            return render_template(
                "not_found.html",
                _template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        targets = technique_service.binding_targets(user_id=int(user["id"]))
        return render_template(
            "technique_card.html",
            _template_context(
                request,
                user=user,
                card=card,
                targets=targets,
                error=error,
                saved=saved,
                created=created,
                bound=bound,
            ),
        )

    @application.post("/techniques/{technique_id}")
    async def update_technique_card(
        request: Request,
        technique_id: str,
        name: str = Form(...),
        dimension: str = Form(...),
        source_location: str = Form(...),
        observation: str = Form(...),
        effect: str = Form(...),
        suitable_for: str = Form(""),
        unsuitable_for: str = Form(""),
        execution_rule: str = Form(...),
        originality_boundary: str = Form(...),
        author_note: str = Form(""),
        card_status: str = Form("active"),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            updated = technique_service.update_card(
                user_id=int(user["id"]),
                technique_id=technique_id,
                observation=_technique_observation_from_form(
                    name=name,
                    dimension=dimension,
                    source_location=source_location,
                    observation=observation,
                    effect=effect,
                    suitable_for=suitable_for,
                    unsuitable_for=unsuitable_for,
                    execution_rule=execution_rule,
                    originality_boundary=originality_boundary,
                ),
                author_note=_clean_field(
                    author_note, "作者备注", max_length=2000
                ),
                status=card_status,
            )
            if not updated:
                return Response(status_code=status.HTTP_404_NOT_FOUND)
        except ValueError as exc:
            return RedirectResponse(
                f"/techniques/{technique_id}?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/techniques/{technique_id}?saved=true",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/techniques/{technique_id}/bindings")
    async def bind_technique_card(
        request: Request,
        technique_id: str,
        target: str = Form(...),
        usage_modes: list[str] = Form(...),
        author_adaptation: str = Form(""),
        priority: int = Form(50),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            technique_service.bind(
                user_id=int(user["id"]),
                technique_id=technique_id,
                target=target,
                usage_modes=usage_modes,
                author_adaptation=_clean_field(
                    author_adaptation,
                    "针对本书的改造",
                    max_length=1000,
                ),
                priority=priority,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/techniques/{technique_id}?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/techniques/{technique_id}?bound=true",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/technique-bindings/{binding_id}/status")
    async def set_technique_binding_status(
        request: Request,
        binding_id: str,
        binding_status: str = Form(...),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            technique_id = technique_service.set_binding_status(
                user_id=int(user["id"]),
                binding_id=binding_id,
                status=binding_status,
            )
            if not technique_id:
                return Response(status_code=status.HTTP_404_NOT_FOUND)
        except ValueError as exc:
            return RedirectResponse(
                f"/techniques?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/techniques/{technique_id}?saved=true",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/novels/new/blank")
    async def create_blank_novel(
        request: Request,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        project_id = secrets.token_hex(16)
        project_dir = (
            app_settings.novels_dir / str(user["id"]) / project_id
        )
        try:
            (project_dir / "chapters").mkdir(
                parents=True, exist_ok=False, mode=0o700
            )
            os.chmod(app_settings.novels_dir / str(user["id"]), 0o700)
            os.chmod(project_dir, 0o700)
            os.chmod(project_dir / "chapters", 0o700)
            database.create_novel_project(
                user_id=int(user["id"]),
                project_id=project_id,
                title="",
                genre="",
                premise="",
                world_setting="",
                style_guide="",
                point_of_view="第三人称限知",
                target_chapter_chars=3000,
            )
        except Exception:
            shutil.rmtree(project_dir, ignore_errors=True)
            logger.exception("failed to create blank novel project")
            return RedirectResponse(
                "/dashboard?error="
                + quote("创建空白作品失败，请稍后重试"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{project_id}/workbench"
            "?view=settings&onboarding=true",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/novels/{project_id}/delete")
    async def delete_novel_project(
        request: Request,
        project_id: str,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        project_dir = (
            app_settings.novels_dir / str(user_id) / project_id
        )
        try:
            deleted = database.delete_novel_project(user_id, project_id)
        except ValueError as exc:
            return RedirectResponse(
                "/dashboard?error=" + quote(str(exc)),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        if not deleted:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        try:
            if project_dir.exists():
                shutil.rmtree(project_dir)
        except OSError:
            logger.exception(
                "novel project database row deleted but files could not be "
                "removed project_id=%s user_id=%s",
                project_id,
                user_id,
            )
            return RedirectResponse(
                "/dashboard?error="
                + quote("作品记录已删除，但本地文件清理失败，请联系管理员"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            "/dashboard?deleted=true", status_code=status.HTTP_303_SEE_OTHER
        )

    @application.get("/novels/new")
    async def new_novel_page(request: Request):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        return RedirectResponse(
            "/dashboard", status_code=status.HTTP_303_SEE_OTHER
        )

    @application.post("/novels/new")
    async def create_novel(
        request: Request,
        title: str = Form(...),
        genre: str = Form(""),
        premise: str = Form(...),
        story_promise: str = Form(""),
        target_audience: str = Form(""),
        core_appeal: str = Form(""),
        ending_constraint: str = Form(""),
        world_setting: str = Form(""),
        style_guide: str = Form(""),
        ai_instructions: str = Form(""),
        point_of_view: str = Form("第三人称限知"),
        target_chapter_chars: int = Form(3000),
        planning_horizon: int = Form(20),
        return_to_workbench: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            clean_title = _clean_field(
                title, "书名", max_length=120, required=True
            )
            clean_genre = _clean_field(
                genre or "未分类", "题材", max_length=80, required=True
            )
            clean_premise = _clean_field(
                premise,
                "故事梗概",
                max_length=4000,
                required=True,
                min_length=10,
            )
            clean_world = _clean_field(
                world_setting, "世界设定", max_length=20_000
            )
            clean_style = _clean_field(
                style_guide, "文风要求", max_length=10_000
            )
            clean_ai_instructions = _clean_field(
                ai_instructions,
                "本书 AI 协作补充指令",
                max_length=10_000,
            )
            clean_promise = _clean_field(
                story_promise, "作品承诺", max_length=4000
            )
            clean_audience = _clean_field(
                target_audience, "目标读者", max_length=1000
            )
            clean_appeal = _clean_field(
                core_appeal, "核心吸引力", max_length=4000
            )
            clean_ending = _clean_field(
                ending_constraint, "结局约束", max_length=4000
            )
            if point_of_view not in POV_OPTIONS:
                raise ValueError("请选择有效的叙事视角")
            if not 2_000 <= target_chapter_chars <= 12_000:
                raise ValueError("单章目标字数必须在 2000–12000 之间")
            if not 3 <= planning_horizon <= 50:
                raise ValueError("滚动规划窗口必须在 3–50 章之间")
        except ValueError as exc:
            return RedirectResponse(
                "/dashboard?error=" + quote(str(exc)),
                status_code=status.HTTP_303_SEE_OTHER,
            )

        project_id = secrets.token_hex(16)
        project_dir = (
            app_settings.novels_dir / str(user["id"]) / project_id
        )
        try:
            (project_dir / "chapters").mkdir(
                parents=True, exist_ok=False, mode=0o700
            )
            os.chmod(app_settings.novels_dir / str(user["id"]), 0o700)
            os.chmod(project_dir, 0o700)
            os.chmod(project_dir / "chapters", 0o700)
            database.create_novel_project(
                user_id=int(user["id"]),
                project_id=project_id,
                title=clean_title,
                genre=clean_genre,
                premise=clean_premise,
                world_setting=clean_world,
                style_guide=clean_style,
                point_of_view=point_of_view,
                target_chapter_chars=target_chapter_chars,
                story_promise=clean_promise,
                target_audience=clean_audience,
                core_appeal=clean_appeal,
                ending_constraint=clean_ending,
                planning_horizon=planning_horizon,
                ai_instructions=clean_ai_instructions,
            )
        except Exception:
            shutil.rmtree(project_dir, ignore_errors=True)
            logger.exception("failed to create novel project")
            return RedirectResponse(
                "/dashboard?error="
                + quote("创建小说项目失败，请稍后重试"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        destination = (
            f"/novels/{project_id}/workbench"
            if return_to_workbench
            else f"/novels/{project_id}"
        )
        return RedirectResponse(
            destination, status_code=status.HTTP_303_SEE_OTHER
        )

    @application.get("/novels/{project_id}", response_class=HTMLResponse)
    async def novel_workspace(
        request: Request,
        project_id: str,
        error: Optional[str] = None,
        saved: bool = False,
        voice_learned: bool = False,
        preference_learned: bool = False,
        blueprint_saved: bool = False,
        arc_saved: bool = False,
        story_plan_applied: bool = False,
        story_plan_baseline_changed: bool = False,
        structure_applied: bool = False,
        structure_baseline_changed: bool = False,
        structure_reverted: bool = False,
        volume_saved: bool = False,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        project = database.get_novel_project(int(user["id"]), project_id)
        if not project:
            return render_template(
                "not_found.html",
                _template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        characters = database.list_novel_characters(
            int(user["id"]), project_id
        )
        chapters = database.list_novel_chapters(int(user["id"]), project_id)
        volumes = planning_service.list_volumes(
            user_id=int(user["id"]), project_id=project_id
        )
        rolling_plan = planning_service.get_rolling_plan(
            user_id=int(user["id"]), project_id=project_id
        )
        story_blueprint = story_planning_service.get_blueprint(
            user_id=int(user["id"]), project_id=project_id
        )
        story_blueprint_versions = (
            story_planning_service.list_blueprint_versions(
                user_id=int(user["id"]), project_id=project_id, limit=6
            )
        )
        planned_plot_arcs = story_planning_service.list_arcs(
            user_id=int(user["id"]), project_id=project_id
        )
        story_plan_suggestions = (
            story_plan_suggestion_service.list_suggestions(
                user_id=int(user["id"]),
                project_id=project_id,
                limit=6,
            )
        )
        story_structure_suggestions = (
            story_structure_suggestion_service.list_suggestions(
                user_id=int(user["id"]),
                project_id=project_id,
                limit=6,
            )
        )
        structure_ready = bool(
            story_blueprint
            and story_blueprint.get("confirmed")
            and any(
                arc.get("confirmed")
                and str(
                    arc["confirmed"].get("arc_type") or ""
                )
                == "main"
                and str(
                    arc["confirmed"].get("lifecycle_status") or ""
                )
                in {"planned", "active"}
                for arc in planned_plot_arcs
            )
        )
        voice_profile = style_service.get_voice_profile(
            user_id=int(user["id"]), project_id=project_id
        )
        style_preferences = style_service.list_preferences(
            user_id=int(user["id"]), project_id=project_id, limit=5
        )
        voice_suggestions = style_service.list_voice_suggestions(
            user_id=int(user["id"]), project_id=project_id, limit=6
        )
        editing_preference_suggestions = (
            preference_service.list_suggestions(
                user_id=int(user["id"]), project_id=project_id, limit=6
            )
        )
        editing_preferences = preference_service.list_active_preferences(
            user_id=int(user["id"]),
            project_id=project_id,
            limit=12,
            exclude_aggregated=True,
        )
        editing_memory_summary = preference_service.get_memory_summary(
            user_id=int(user["id"]), project_id=project_id
        )
        reader_requests = reader_service.list_requests(
            user_id=int(user["id"]), project_id=project_id
        )
        impact_reports = impact_service.list_reports(
            user_id=int(user["id"]), project_id=project_id
        )
        project_techniques = technique_service.list_project_bindings(
            user_id=int(user["id"]), project_id=project_id
        )
        continuity = continuity_service.get_dashboard(
            user_id=int(user["id"]), project_id=project_id
        )
        structure_health = structure_health_service.get_report(
            user_id=int(user["id"]), project_id=project_id
        )
        next_chapter_workflow = workflow_service.get_next_project_state(
            user_id=int(user["id"]), project_id=project_id
        )
        return render_template(
            "novel_workspace.html",
            _template_context(
                request,
                user=user,
                project=project,
                characters=characters,
                chapters=chapters,
                volumes=volumes,
                rolling_plan=rolling_plan,
                story_blueprint=story_blueprint,
                story_blueprint_form=(
                    story_blueprint["current"]
                    if story_blueprint
                    and story_blueprint["current"]
                    else {}
                ),
                story_blueprint_versions=story_blueprint_versions,
                planned_plot_arcs=planned_plot_arcs,
                story_plan_suggestions=story_plan_suggestions,
                story_structure_suggestions=(
                    story_structure_suggestions
                ),
                structure_ready=structure_ready,
                structure_default_chapter_count=max(
                    10,
                    min(30, int(project.get("planning_horizon") or 20)),
                ),
                story_arc_type_options=STORY_ARC_TYPE_OPTIONS,
                story_arc_lifecycle_options=(
                    STORY_ARC_LIFECYCLE_OPTIONS
                ),
                voice_profile=voice_profile,
                style_preferences=style_preferences,
                voice_suggestions=voice_suggestions,
                editing_preference_suggestions=(
                    editing_preference_suggestions
                ),
                editing_preferences=editing_preferences,
                editing_memory_summary=editing_memory_summary,
                reader_requests=reader_requests,
                impact_reports=impact_reports,
                project_techniques=project_techniques,
                continuity=continuity,
                structure_health=structure_health,
                next_chapter_workflow=next_chapter_workflow,
                pov_options=POV_OPTIONS,
                error=error,
                saved=saved,
                voice_learned=voice_learned,
                preference_learned=preference_learned,
                blueprint_saved=blueprint_saved,
                arc_saved=arc_saved,
                story_plan_applied=story_plan_applied,
                story_plan_baseline_changed=(
                    story_plan_baseline_changed
                ),
                structure_applied=structure_applied,
                structure_baseline_changed=(
                    structure_baseline_changed
                ),
                structure_reverted=structure_reverted,
                volume_saved=volume_saved,
            ),
        )

    @application.post("/novels/{project_id}/story-plan-suggestions")
    async def create_story_plan_suggestion(
        request: Request,
        project_id: str,
        planning_mode: str = Form("create"),
        instruction: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            if planning_mode not in {"create", "refine", "rethink"}:
                raise ValueError("请选择有效的全书规划模式")
            clean_instruction = _clean_field(
                instruction,
                "本次规划重点",
                max_length=4000,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}?error={quote(str(exc))}"
                "#story-planner",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        profile = api_profile(int(user["id"]))
        if not profile:
            return RedirectResponse(
                "/settings/api?error="
                + quote("生成全书方案前，请先配置模型服务"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        try:
            suggestion_id = (
                story_plan_suggestion_service.create_suggestion(
                    user_id=int(user["id"]),
                    project_id=project_id,
                    planning_mode=planning_mode,
                    instruction=clean_instruction,
                    provider=profile["provider"],
                    model=profile["model"],
                    credential_source=profile["credential_source"],
                    max_jobs_per_day=app_settings.max_jobs_per_day,
                )
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}?error={quote(str(exc))}"
                "#story-planner",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        request.app.state.worker.wake()
        return RedirectResponse(
            f"/story-plan-suggestions/{suggestion_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.get(
        "/story-plan-suggestions/{suggestion_id}",
        response_class=HTMLResponse,
    )
    async def story_plan_suggestion_page(
        request: Request,
        suggestion_id: str,
        error: Optional[str] = None,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        suggestion = story_plan_suggestion_service.get_suggestion(
            user_id=int(user["id"]),
            suggestion_id=suggestion_id,
        )
        if not suggestion:
            return render_template(
                "not_found.html",
                _template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        return render_template(
            "story_plan_suggestion.html",
            _template_context(
                request,
                user=user,
                suggestion=suggestion,
                error=error,
            ),
        )

    @application.get("/api/story-plan-suggestions/{suggestion_id}")
    async def story_plan_suggestion_status(
        request: Request, suggestion_id: str
    ):
        user = _current_user(request)
        if not user:
            return JSONResponse(
                {"detail": "未登录"},
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        current_status = story_plan_suggestion_service.get_status(
            user_id=int(user["id"]),
            suggestion_id=suggestion_id,
        )
        if current_status is None:
            return JSONResponse(
                {"detail": "任务不存在"},
                status_code=status.HTTP_404_NOT_FOUND,
            )
        return JSONResponse(
            {
                "status": current_status,
                "terminal": current_status in {"completed", "failed"},
            }
        )

    @application.post(
        "/story-plan-suggestions/{suggestion_id}/apply"
    )
    async def apply_story_plan_suggestion(
        request: Request,
        suggestion_id: str,
        option_index: int = Form(...),
        apply_blueprint: str = Form(""),
        arc_indices: list[int] = Form(default=[]),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            applied = story_plan_suggestion_service.apply_suggestion(
                user_id=int(user["id"]),
                suggestion_id=suggestion_id,
                option_index=option_index,
                apply_blueprint=apply_blueprint == "yes",
                selected_arc_indices=arc_indices,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/story-plan-suggestions/{suggestion_id}"
                f"?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        baseline_query = (
            "&story_plan_baseline_changed=true"
            if applied["baseline_changed"]
            else ""
        )
        return RedirectResponse(
            f"/novels/{applied['project_id']}"
            f"?story_plan_applied=true{baseline_query}"
            "#story-blueprint",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post(
        "/novels/{project_id}/story-structure-suggestions"
    )
    async def create_story_structure_suggestion(
        request: Request,
        project_id: str,
        chapter_count: int = Form(20),
        instruction: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            if not 10 <= chapter_count <= 30:
                raise ValueError("请选择未来 10–30 章")
            clean_instruction = _clean_field(
                instruction,
                "本次结构重点",
                max_length=4000,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}?error={quote(str(exc))}"
                "#rolling-structure",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        profile = api_profile(int(user["id"]))
        if not profile:
            return RedirectResponse(
                "/settings/api?error="
                + quote(
                    "生成分卷与滚动章节骨架前，请先配置模型服务"
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        try:
            suggestion_id = (
                story_structure_suggestion_service.create_suggestion(
                    user_id=int(user["id"]),
                    project_id=project_id,
                    chapter_count=chapter_count,
                    instruction=clean_instruction,
                    provider=profile["provider"],
                    model=profile["model"],
                    credential_source=profile["credential_source"],
                    max_jobs_per_day=app_settings.max_jobs_per_day,
                )
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}?error={quote(str(exc))}"
                "#rolling-structure",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        request.app.state.worker.wake()
        return RedirectResponse(
            f"/story-structure-suggestions/{suggestion_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.get(
        "/story-structure-suggestions/{suggestion_id}",
        response_class=HTMLResponse,
    )
    async def story_structure_suggestion_page(
        request: Request,
        suggestion_id: str,
        error: Optional[str] = None,
        applied: bool = False,
        reverted: bool = False,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        suggestion = story_structure_suggestion_service.get_suggestion(
            user_id=int(user["id"]),
            suggestion_id=suggestion_id,
        )
        if not suggestion:
            return render_template(
                "not_found.html",
                _template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        return render_template(
            "story_structure_suggestion.html",
            _template_context(
                request,
                user=user,
                suggestion=suggestion,
                error=error,
                applied=applied,
                reverted=reverted,
            ),
        )

    @application.get(
        "/api/story-structure-suggestions/{suggestion_id}"
    )
    async def story_structure_suggestion_status(
        request: Request, suggestion_id: str
    ):
        user = _current_user(request)
        if not user:
            return JSONResponse(
                {"detail": "未登录"},
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        current_status = (
            story_structure_suggestion_service.get_status(
                user_id=int(user["id"]),
                suggestion_id=suggestion_id,
            )
        )
        if current_status is None:
            return JSONResponse(
                {"detail": "任务不存在"},
                status_code=status.HTTP_404_NOT_FOUND,
            )
        return JSONResponse(
            {
                "status": current_status,
                "terminal": current_status in {"completed", "failed"},
            }
        )

    @application.post(
        "/story-structure-suggestions/{suggestion_id}/apply"
    )
    async def apply_story_structure_suggestion(
        request: Request,
        suggestion_id: str,
        option_index: int = Form(...),
        preview_fingerprint: str = Form(...),
        confirm_changes: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            if confirm_changes != "yes":
                raise ValueError("请先确认已核对本次精确变更")
            applied_result = (
                story_structure_suggestion_service.apply_suggestion(
                    user_id=int(user["id"]),
                    suggestion_id=suggestion_id,
                    option_index=option_index,
                    preview_fingerprint=preview_fingerprint,
                )
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/story-structure-suggestions/{suggestion_id}"
                f"?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        baseline_query = (
            "&structure_baseline_changed=true"
            if applied_result["baseline_changed"]
            else ""
        )
        return RedirectResponse(
            f"/novels/{applied_result['project_id']}"
            f"?structure_applied=true{baseline_query}"
            "#rolling-structure",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post(
        "/story-structure-applications/{application_id}/revert"
    )
    async def revert_story_structure_application(
        request: Request,
        application_id: str,
        suggestion_id: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            reverted_result = (
                story_structure_suggestion_service.revert_application(
                    user_id=int(user["id"]),
                    application_id=application_id,
                )
            )
        except ValueError as exc:
            error_target = (
                f"/story-structure-suggestions/{quote(suggestion_id)}"
                if suggestion_id
                else "/"
            )
            return RedirectResponse(
                f"{error_target}?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            "/story-structure-suggestions/"
            f"{reverted_result['suggestion_id']}?reverted=true",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.get(
        "/novels/{project_id}/structure-health",
        response_class=HTMLResponse,
    )
    async def structure_health_page(
        request: Request,
        project_id: str,
        error: Optional[str] = None,
        causal_saved: bool = False,
        causal_archived: bool = False,
        reset_task_cards: int = 0,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        report = structure_health_service.get_report(
            user_id=int(user["id"]), project_id=project_id
        )
        if not report:
            return render_template(
                "not_found.html",
                _template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        causal_suggestions = causal_suggestion_service.list_suggestions(
            user_id=int(user["id"]),
            project_id=project_id,
            limit=6,
        )
        return render_template(
            "structure_health.html",
            _template_context(
                request,
                user=user,
                report=report,
                error=error,
                causal_saved=causal_saved,
                causal_archived=causal_archived,
                reset_task_cards=reset_task_cards,
                causal_suggestions=causal_suggestions,
                causal_suggestion_default_limit=min(
                    80,
                    max(
                        20,
                        int(
                            report["project"].get(
                                "planning_horizon", 20
                            )
                            or 20
                        )
                        * 2,
                    ),
                ),
            ),
        )

    @application.post(
        "/novels/{project_id}/causal-link-suggestions"
    )
    async def create_causal_link_suggestion(
        request: Request,
        project_id: str,
        chapter_limit: int = Form(40),
        instruction: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            if not 2 <= chapter_limit <= 80:
                raise ValueError("请选择未来 2–80 章")
            clean_instruction = _clean_field(
                instruction,
                "本次因果审查重点",
                max_length=4000,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}/structure-health"
                f"?error={quote(str(exc))}#causal-suggestions",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        profile = api_profile(int(user["id"]))
        if not profile:
            return RedirectResponse(
                "/settings/api?error="
                + quote("生成因果建议前，请先配置模型服务"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        try:
            suggestion_id = causal_suggestion_service.create_suggestion(
                user_id=int(user["id"]),
                project_id=project_id,
                chapter_limit=chapter_limit,
                instruction=clean_instruction,
                provider=profile["provider"],
                model=profile["model"],
                credential_source=profile["credential_source"],
                max_jobs_per_day=app_settings.max_jobs_per_day,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}/structure-health"
                f"?error={quote(str(exc))}#causal-suggestions",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        request.app.state.worker.wake()
        return RedirectResponse(
            f"/causal-link-suggestions/{suggestion_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.get(
        "/causal-link-suggestions/{suggestion_id}",
        response_class=HTMLResponse,
    )
    async def causal_link_suggestion_page(
        request: Request,
        suggestion_id: str,
        error: Optional[str] = None,
        accepted: bool = False,
        dismissed: bool = False,
        reset_task_cards: int = 0,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        suggestion = causal_suggestion_service.get_suggestion(
            user_id=int(user["id"]),
            suggestion_id=suggestion_id,
        )
        if not suggestion:
            return render_template(
                "not_found.html",
                _template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        branch_simulations = causal_branch_service.list_for_suggestion(
            user_id=int(user["id"]),
            suggestion_id=suggestion_id,
        )
        simulations_by_proposal: Dict[int, list[Dict[str, Any]]] = {}
        for simulation in branch_simulations:
            simulations_by_proposal.setdefault(
                int(simulation["proposal_index"]),
                [],
            ).append(simulation)
        if suggestion.get("result"):
            for proposal in suggestion["result"].get("proposals") or []:
                proposal["branch_simulations"] = (
                    simulations_by_proposal.get(
                        int(proposal["proposal_index"]),
                        [],
                    )
                )
        simulation_max_horizon = min(
            30,
            len(
                suggestion.get("context_snapshot", {}).get(
                    "future_chapters",
                    [],
                )
            ),
        )
        return render_template(
            "causal_link_suggestion.html",
            _template_context(
                request,
                user=user,
                suggestion=suggestion,
                error=error,
                accepted=accepted,
                dismissed=dismissed,
                reset_task_cards=reset_task_cards,
                simulation_max_horizon=simulation_max_horizon,
                simulation_default_horizon=min(
                    20,
                    simulation_max_horizon,
                ),
            ),
        )

    @application.get(
        "/api/causal-link-suggestions/{suggestion_id}"
    )
    async def causal_link_suggestion_status(
        request: Request,
        suggestion_id: str,
    ):
        user = _current_user(request)
        if not user:
            return JSONResponse(
                {"detail": "未登录"},
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        current_status = causal_suggestion_service.get_status(
            user_id=int(user["id"]),
            suggestion_id=suggestion_id,
        )
        if current_status is None:
            return JSONResponse(
                {"detail": "任务不存在"},
                status_code=status.HTTP_404_NOT_FOUND,
            )
        return JSONResponse(
            {
                "status": current_status,
                "terminal": current_status in {"completed", "failed"},
            }
        )

    @application.post(
        "/causal-link-suggestions/{suggestion_id}/proposals/"
        "{proposal_index}/branch-simulations"
    )
    async def create_causal_branch_simulation(
        request: Request,
        suggestion_id: str,
        proposal_index: int,
        horizon_chapter_count: int = Form(20),
        instruction: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            if not 10 <= horizon_chapter_count <= 30:
                raise ValueError("请选择未来 10–30 章")
            clean_instruction = _clean_field(
                instruction,
                "本次长期推演重点",
                max_length=4000,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/causal-link-suggestions/{suggestion_id}"
                f"?error={quote(str(exc))}#proposal-{proposal_index}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        profile = api_profile(int(user["id"]))
        if not profile:
            return RedirectResponse(
                "/settings/api?error="
                + quote("生成长期因果分支前，请先配置模型服务"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        try:
            simulation_id = causal_branch_service.create_simulation(
                user_id=int(user["id"]),
                suggestion_id=suggestion_id,
                proposal_index=proposal_index,
                horizon_chapter_count=horizon_chapter_count,
                instruction=clean_instruction,
                provider=profile["provider"],
                model=profile["model"],
                credential_source=profile["credential_source"],
                max_jobs_per_day=app_settings.max_jobs_per_day,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/causal-link-suggestions/{suggestion_id}"
                f"?error={quote(str(exc)[:1000])}"
                f"#proposal-{proposal_index}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        request.app.state.worker.wake()
        return RedirectResponse(
            f"/causal-branch-simulations/{simulation_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.get(
        "/causal-branch-simulations/{simulation_id}",
        response_class=HTMLResponse,
    )
    async def causal_branch_simulation_page(
        request: Request,
        simulation_id: str,
        error: Optional[str] = None,
        adoption_created: bool = False,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        simulation = causal_branch_service.get_simulation(
            user_id=int(user["id"]),
            simulation_id=simulation_id,
        )
        if not simulation:
            return render_template(
                "not_found.html",
                _template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        adoptions = causal_branch_adoption_service.list_for_simulation(
            user_id=int(user["id"]),
            simulation_id=simulation_id,
        )
        adoption_by_branch: Dict[str, Dict[str, Any]] = {}
        for adoption in adoptions:
            adoption_by_branch.setdefault(
                str(adoption["branch_key"]),
                adoption,
            )
        if simulation.get("result"):
            for branch in simulation["result"].get("branches") or []:
                branch["adoption"] = adoption_by_branch.get(
                    str(branch.get("branch_key") or "")
                )
        return render_template(
            "causal_branch_simulation.html",
            _template_context(
                request,
                user=user,
                simulation=simulation,
                error=error,
                adoption_created=adoption_created,
            ),
        )

    @application.get(
        "/api/causal-branch-simulations/{simulation_id}"
    )
    async def causal_branch_simulation_status(
        request: Request,
        simulation_id: str,
    ):
        user = _current_user(request)
        if not user:
            return JSONResponse(
                {"detail": "未登录"},
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        current_status = causal_branch_service.get_status(
            user_id=int(user["id"]),
            simulation_id=simulation_id,
        )
        if current_status is None:
            return JSONResponse(
                {"detail": "任务不存在"},
                status_code=status.HTTP_404_NOT_FOUND,
            )
        return JSONResponse(
            {
                "status": current_status,
                "terminal": current_status in {"completed", "failed"},
            }
        )

    @application.post(
        "/causal-branch-simulations/{simulation_id}/branches/"
        "{branch_key}/adoptions"
    )
    async def create_causal_branch_adoption(
        request: Request,
        simulation_id: str,
        branch_key: str,
        meaning_confirmed: bool = Form(False),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            adoption_id = causal_branch_adoption_service.create_adoption(
                user_id=int(user["id"]),
                simulation_id=simulation_id,
                branch_key=branch_key,
                meaning_confirmed=meaning_confirmed,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/causal-branch-simulations/{simulation_id}"
                f"?error={quote(str(exc)[:1000])}"
                f"#{branch_key}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/causal-branch-adoptions/{adoption_id}"
            "?adoption_created=true",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.get(
        "/causal-branch-adoptions/{adoption_id}",
        response_class=HTMLResponse,
    )
    async def causal_branch_adoption_page(
        request: Request,
        adoption_id: str,
        error: Optional[str] = None,
        adoption_created: bool = False,
        saved: bool = False,
        applied: bool = False,
        reverted: bool = False,
        applied_items: int = 0,
        reset_task_cards: int = 0,
        stale_scenes: int = 0,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        adoption = causal_branch_adoption_service.get_adoption(
            user_id=int(user["id"]),
            adoption_id=adoption_id,
        )
        if not adoption:
            return render_template(
                "not_found.html",
                _template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        return render_template(
            "causal_branch_adoption.html",
            _template_context(
                request,
                user=user,
                adoption=adoption,
                error=error,
                adoption_created=adoption_created,
                saved=saved,
                applied=applied,
                reverted=reverted,
                applied_items=applied_items,
                reset_task_cards=reset_task_cards,
                stale_scenes=stale_scenes,
            ),
        )

    @application.post(
        "/causal-branch-adoptions/{adoption_id}/items/{item_id}"
    )
    async def review_causal_branch_adoption_item(
        request: Request,
        adoption_id: str,
        item_id: str,
        decision: str = Form(...),
        must_happen: str = Form(""),
        plot_threads: list[str] = Form([]),
        foreshadow_setup: str = Form(""),
        foreshadow_payoff: str = Form(""),
        author_note: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            patch = CausalBranchTaskPatch(
                must_happen=_story_plan_lines(
                    must_happen,
                    "本章必须落实",
                    limit=12,
                ),
                plot_threads=list(dict.fromkeys(plot_threads)),
                foreshadow_setup=_story_plan_lines(
                    foreshadow_setup,
                    "伏笔铺设",
                    limit=10,
                ),
                foreshadow_payoff=_story_plan_lines(
                    foreshadow_payoff,
                    "伏笔回收",
                    limit=10,
                ),
            )
            causal_branch_adoption_service.review_item(
                user_id=int(user["id"]),
                adoption_id=adoption_id,
                item_id=item_id,
                decision=decision,
                patch=patch,
                author_note=_clean_field(
                    author_note,
                    "单项作者备注",
                    max_length=1200,
                ),
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/causal-branch-adoptions/{adoption_id}"
                f"?error={quote(str(exc)[:1000])}#item-{item_id}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/causal-branch-adoptions/{adoption_id}"
            f"?saved=true#item-{item_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post(
        "/causal-branch-adoptions/{adoption_id}/apply"
    )
    async def apply_causal_branch_adoption(
        request: Request,
        adoption_id: str,
        author_confirmed: bool = Form(False),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            result = causal_branch_adoption_service.apply_adoption(
                user_id=int(user["id"]),
                adoption_id=adoption_id,
                author_confirmed=author_confirmed,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/causal-branch-adoptions/{adoption_id}"
                f"?error={quote(str(exc)[:1000])}#apply",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/causal-branch-adoptions/{adoption_id}?applied=true"
            f"&applied_items={result['applied_item_count']}"
            f"&reset_task_cards={result['reset_task_card_count']}"
            f"&stale_scenes={result['stale_scene_count']}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post(
        "/causal-branch-adoptions/{adoption_id}/abandon"
    )
    async def abandon_causal_branch_adoption(
        request: Request,
        adoption_id: str,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        adoption = causal_branch_adoption_service.get_adoption(
            user_id=int(user["id"]),
            adoption_id=adoption_id,
        )
        if not adoption:
            return render_template(
                "not_found.html",
                _template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        try:
            causal_branch_adoption_service.abandon_adoption(
                user_id=int(user["id"]),
                adoption_id=adoption_id,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/causal-branch-adoptions/{adoption_id}"
                f"?error={quote(str(exc)[:1000])}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            "/causal-branch-simulations/"
            + str(adoption["simulation_id"]),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post(
        "/causal-branch-adoptions/{adoption_id}/revert"
    )
    async def revert_causal_branch_adoption(
        request: Request,
        adoption_id: str,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            causal_branch_adoption_service.revert_adoption(
                user_id=int(user["id"]),
                adoption_id=adoption_id,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/causal-branch-adoptions/{adoption_id}"
                f"?error={quote(str(exc)[:1000])}#revert",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/causal-branch-adoptions/{adoption_id}?reverted=true",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post(
        "/causal-link-suggestions/{suggestion_id}/proposals/"
        "{proposal_index}/accept"
    )
    async def accept_causal_link_proposal(
        request: Request,
        suggestion_id: str,
        proposal_index: int,
        cause_text: str = Form(...),
        effect_text: str = Form(...),
        author_note: str = Form(""),
        confirm_changes: str = Form(""),
        confirm_comparison: str = Form(""),
        confirm_semantic_review: str = Form(""),
        semantic_override_reason: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            if confirm_changes != "yes":
                raise ValueError(
                    "请先确认已核对起因、结果和正史边界"
                )
            result = causal_suggestion_service.accept_proposal(
                user_id=int(user["id"]),
                suggestion_id=suggestion_id,
                proposal_index=proposal_index,
                cause_text=cause_text,
                effect_text=effect_text,
                author_note=author_note,
                comparison_confirmed=confirm_comparison == "yes",
                semantic_review_confirmed=(
                    confirm_semantic_review == "yes"
                ),
                semantic_override_reason=semantic_override_reason,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/causal-link-suggestions/{suggestion_id}"
                f"?error={quote(str(exc)[:1000])}"
                f"#proposal-{proposal_index}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/causal-link-suggestions/{suggestion_id}"
            "?accepted=true&reset_task_cards="
            f"{int(result['reset_task_card_count'])}"
            f"#proposal-{proposal_index}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post(
        "/causal-link-suggestions/{suggestion_id}/proposals/"
        "{proposal_index}/dismiss"
    )
    async def dismiss_causal_link_proposal(
        request: Request,
        suggestion_id: str,
        proposal_index: int,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            causal_suggestion_service.dismiss_proposal(
                user_id=int(user["id"]),
                suggestion_id=suggestion_id,
                proposal_index=proposal_index,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/causal-link-suggestions/{suggestion_id}"
                f"?error={quote(str(exc)[:1000])}"
                f"#proposal-{proposal_index}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/causal-link-suggestions/{suggestion_id}"
            f"?dismissed=true#proposal-{proposal_index}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/novels/{project_id}/structure-links")
    async def create_structure_link(
        request: Request,
        project_id: str,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        form = await request.form()
        verify_csrf(request, str(form.get("csrf") or ""))
        try:
            result = structure_link_service.create_link(
                user_id=int(user["id"]),
                project_id=project_id,
                source_chapter_id=str(
                    form.get("source_chapter_id") or ""
                ).strip(),
                target_chapter_id=str(
                    form.get("target_chapter_id") or ""
                ).strip(),
                relation_type=str(
                    form.get("relation_type") or ""
                ).strip(),
                cause_text=str(form.get("cause_text") or ""),
                effect_text=str(form.get("effect_text") or ""),
                author_note=str(form.get("author_note") or ""),
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}/structure-health"
                f"?error={quote(str(exc)[:1000])}#causal-links",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{project_id}/structure-health"
            "?causal_saved=true&reset_task_cards="
            f"{int(result['reset_task_card_count'])}#causal-links",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post(
        "/novels/{project_id}/structure-links/{link_id}/archive"
    )
    async def archive_structure_link(
        request: Request,
        project_id: str,
        link_id: str,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        form = await request.form()
        verify_csrf(request, str(form.get("csrf") or ""))
        try:
            result = structure_link_service.archive_link(
                user_id=int(user["id"]),
                project_id=project_id,
                link_id=link_id,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}/structure-health"
                f"?error={quote(str(exc)[:1000])}#causal-links",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{project_id}/structure-health"
            "?causal_archived=true&reset_task_cards="
            f"{int(result['reset_task_card_count'])}#causal-links",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.get(
        "/novels/{project_id}/continuity", response_class=HTMLResponse
    )
    async def continuity_dashboard(
        request: Request,
        project_id: str,
        error: Optional[str] = None,
        saved: bool = False,
        identity_saved: bool = False,
        identity_removed: bool = False,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        dashboard = continuity_service.get_dashboard(
            user_id=int(user["id"]), project_id=project_id
        )
        if not dashboard:
            return render_template(
                "not_found.html",
                _template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        return render_template(
            "continuity.html",
            _template_context(
                request,
                user=user,
                dashboard=dashboard,
                error=error,
                saved=saved,
                identity_saved=identity_saved,
                identity_removed=identity_removed,
            ),
        )

    @application.post("/novels/{project_id}/memory-identities")
    async def save_memory_identity(
        request: Request,
        project_id: str,
        identity_type: str = Form(...),
        canonical_text: str = Form(...),
        aliases: str = Form(...),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            if identity_type not in IDENTITY_TYPES:
                raise ValueError("不支持的归一类型")
            identity_service.save_rule(
                user_id=int(user["id"]),
                project_id=project_id,
                identity_type=identity_type,
                canonical_text=_clean_field(
                    canonical_text,
                    "标准称呼",
                    max_length=1500,
                    required=True,
                ),
                aliases=_split_lines(aliases, limit=30),
                updated_at=utc_now(),
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}/continuity?error="
                f"{quote(str(exc))}#identity-rules",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{project_id}/continuity?identity_saved=true"
            "#identity-rules",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post(
        "/novels/{project_id}/memory-identities/{identity_id}/delete"
    )
    async def remove_memory_identity(
        request: Request,
        project_id: str,
        identity_id: str,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        removed = identity_service.remove_rule(
            user_id=int(user["id"]),
            project_id=project_id,
            identity_id=identity_id,
            updated_at=utc_now(),
        )
        if not removed:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        return RedirectResponse(
            f"/novels/{project_id}/continuity?identity_removed=true"
            "#identity-rules",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post(
        "/novels/{project_id}/continuity/issues/{issue_id}"
    )
    async def update_continuity_issue(
        request: Request,
        project_id: str,
        issue_id: str,
        action: str = Form(...),
        author_note: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            updated = continuity_service.set_issue_status(
                user_id=int(user["id"]),
                project_id=project_id,
                issue_id=issue_id,
                action=action,
                author_note=author_note,
                updated_at=utc_now(),
            )
            if not updated:
                return Response(status_code=status.HTTP_404_NOT_FOUND)
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}/continuity?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{project_id}/continuity?saved=true",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/novels/{project_id}/settings")
    async def update_novel_settings(
        request: Request,
        project_id: str,
        title: str = Form(""),
        genre: str = Form(""),
        premise: str = Form(""),
        story_promise: str = Form(""),
        target_audience: str = Form(""),
        core_appeal: str = Form(""),
        ending_constraint: str = Form(""),
        world_setting: str = Form(""),
        style_guide: str = Form(""),
        ai_instructions: str = Form(""),
        point_of_view: str = Form("第三人称限知"),
        target_chapter_chars: int = Form(3000),
        planning_horizon: int = Form(20),
        settings_tab: str = Form("core"),
        return_to_workbench: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        clean_settings_tab = (
            settings_tab
            if settings_tab in WORKBENCH_SETTING_TAB_KEYS
            else "core"
        )
        try:
            clean_title = _clean_field(
                title, "书名", max_length=120
            )
            clean_genre = _clean_field(
                genre, "题材", max_length=80
            )
            clean_premise = _clean_field(
                premise,
                "故事梗概",
                max_length=4000,
            )
            clean_world = _clean_field(
                world_setting, "世界设定", max_length=20_000
            )
            clean_style = _clean_field(
                style_guide, "文风要求", max_length=10_000
            )
            clean_ai_instructions = _clean_field(
                ai_instructions,
                "本书 AI 协作补充指令",
                max_length=10_000,
            )
            clean_promise = _clean_field(
                story_promise, "作品承诺", max_length=4000
            )
            clean_audience = _clean_field(
                target_audience, "目标读者", max_length=1000
            )
            clean_appeal = _clean_field(
                core_appeal, "核心吸引力", max_length=4000
            )
            clean_ending = _clean_field(
                ending_constraint, "结局约束", max_length=4000
            )
            if point_of_view not in POV_OPTIONS:
                raise ValueError("请选择有效的叙事视角")
            if not 2_000 <= target_chapter_chars <= 12_000:
                raise ValueError("单章目标字数必须在 2000–12000 之间")
            if not 3 <= planning_horizon <= 50:
                raise ValueError("滚动规划窗口必须在 3–50 章之间")
            updated = database.update_novel_project(
                user_id=int(user["id"]),
                project_id=project_id,
                title=clean_title,
                genre=clean_genre,
                premise=clean_premise,
                world_setting=clean_world,
                style_guide=clean_style,
                point_of_view=point_of_view,
                target_chapter_chars=target_chapter_chars,
                story_promise=clean_promise,
                target_audience=clean_audience,
                core_appeal=clean_appeal,
                ending_constraint=clean_ending,
                planning_horizon=planning_horizon,
                ai_instructions=clean_ai_instructions,
            )
            if not updated:
                return Response(status_code=status.HTTP_404_NOT_FOUND)
        except ValueError as exc:
            if return_to_workbench:
                return RedirectResponse(
                    f"/novels/{project_id}/workbench"
                    "?view=settings"
                    f"&settings_tab={clean_settings_tab}"
                    f"&error={quote(str(exc))}",
                    status_code=status.HTTP_303_SEE_OTHER,
                )
            return RedirectResponse(
                f"/novels/{project_id}?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        if return_to_workbench:
            return RedirectResponse(
                f"/novels/{project_id}/workbench"
                "?view=settings"
                f"&settings_tab={clean_settings_tab}"
                "&saved=true",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{project_id}?saved=true",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/novels/{project_id}/story-blueprint")
    async def save_story_blueprint(
        request: Request,
        project_id: str,
        central_question: str = Form(""),
        protagonist_goal: str = Form(""),
        core_conflict: str = Form(""),
        stakes: str = Form(""),
        opening_state: str = Form(""),
        ending_state: str = Form(""),
        major_turns: str = Form(""),
        must_payoffs: str = Form(""),
        forbidden_shortcuts: str = Form(""),
        author_notes: str = Form(""),
        action: str = Form("save_draft"),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            if action not in {"save_draft", "confirm"}:
                raise ValueError("不支持的全书蓝图操作")
            blueprint = _story_blueprint_from_form(
                central_question=central_question,
                protagonist_goal=protagonist_goal,
                core_conflict=core_conflict,
                stakes=stakes,
                opening_state=opening_state,
                ending_state=ending_state,
                major_turns=major_turns,
                must_payoffs=must_payoffs,
                forbidden_shortcuts=forbidden_shortcuts,
                author_notes=author_notes,
            )
            story_planning_service.save_blueprint(
                user_id=int(user["id"]),
                project_id=project_id,
                blueprint=blueprint,
                confirm=action == "confirm",
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}?error={quote(str(exc))}"
                "#story-blueprint",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{project_id}?blueprint_saved=true"
            "#story-blueprint",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post(
        "/novels/{project_id}/story-blueprint/versions/"
        "{version_id}/restore"
    )
    async def restore_story_blueprint(
        request: Request,
        project_id: str,
        version_id: str,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            story_planning_service.restore_blueprint_version(
                user_id=int(user["id"]),
                project_id=project_id,
                version_id=version_id,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}?error={quote(str(exc))}"
                "#story-blueprint",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{project_id}?blueprint_saved=true"
            "#story-blueprint",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/novels/{project_id}/plot-arcs")
    async def create_planned_plot_arc(
        request: Request,
        project_id: str,
        arc_type: str = Form("subplot"),
        title: str = Form(...),
        dramatic_question: str = Form(""),
        promise: str = Form(""),
        start_state: str = Form(""),
        target_payoff: str = Form(""),
        involved_characters: str = Form(""),
        planned_turns: str = Form(""),
        lifecycle_status: str = Form("planned"),
        priority: int = Form(3),
        author_notes: str = Form(""),
        action: str = Form("save_draft"),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            if action not in {"save_draft", "confirm"}:
                raise ValueError("不支持的剧情线操作")
            arc = _planned_story_arc_from_form(
                arc_type=arc_type,
                title=title,
                dramatic_question=dramatic_question,
                promise=promise,
                start_state=start_state,
                target_payoff=target_payoff,
                involved_characters=involved_characters,
                planned_turns=planned_turns,
                lifecycle_status=lifecycle_status,
                priority=priority,
                author_notes=author_notes,
            )
            story_planning_service.create_arc(
                user_id=int(user["id"]),
                project_id=project_id,
                arc=arc,
                confirm=action == "confirm",
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}?error={quote(str(exc))}"
                "#plot-arcs",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{project_id}?arc_saved=true#plot-arcs",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/novels/{project_id}/plot-arcs/{arc_id}")
    async def update_planned_plot_arc(
        request: Request,
        project_id: str,
        arc_id: str,
        arc_type: str = Form("subplot"),
        title: str = Form(...),
        dramatic_question: str = Form(""),
        promise: str = Form(""),
        start_state: str = Form(""),
        target_payoff: str = Form(""),
        involved_characters: str = Form(""),
        planned_turns: str = Form(""),
        lifecycle_status: str = Form("planned"),
        priority: int = Form(3),
        author_notes: str = Form(""),
        action: str = Form("save_draft"),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            if action not in {"save_draft", "confirm"}:
                raise ValueError("不支持的剧情线操作")
            arc = _planned_story_arc_from_form(
                arc_type=arc_type,
                title=title,
                dramatic_question=dramatic_question,
                promise=promise,
                start_state=start_state,
                target_payoff=target_payoff,
                involved_characters=involved_characters,
                planned_turns=planned_turns,
                lifecycle_status=lifecycle_status,
                priority=priority,
                author_notes=author_notes,
            )
            story_planning_service.update_arc(
                user_id=int(user["id"]),
                project_id=project_id,
                arc_id=arc_id,
                arc=arc,
                confirm=action == "confirm",
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}?error={quote(str(exc))}"
                "#plot-arcs",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{project_id}?arc_saved=true#plot-arcs",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post(
        "/novels/{project_id}/plot-arcs/{arc_id}/versions/"
        "{version_id}/restore"
    )
    async def restore_planned_plot_arc(
        request: Request,
        project_id: str,
        arc_id: str,
        version_id: str,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            story_planning_service.restore_arc_version(
                user_id=int(user["id"]),
                project_id=project_id,
                arc_id=arc_id,
                version_id=version_id,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}?error={quote(str(exc))}"
                "#plot-arcs",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{project_id}?arc_saved=true#plot-arcs",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post(
        "/novels/{project_id}/plot-arcs/{arc_id}/archive"
    )
    async def archive_planned_plot_arc(
        request: Request,
        project_id: str,
        arc_id: str,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            story_planning_service.archive_arc(
                user_id=int(user["id"]),
                project_id=project_id,
                arc_id=arc_id,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}?error={quote(str(exc))}"
                "#plot-arcs",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{project_id}?arc_saved=true#plot-arcs",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/novels/{project_id}/reader-requests")
    async def create_reader_request(
        request: Request,
        project_id: str,
        raw_text: str = Form(...),
        request_type: str = Form("plot"),
        impact_scope: str = Form("next_three"),
        priority: str = Form("soft"),
        constraints: str = Form(""),
        author_note: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            request_id = reader_service.create_request(
                user_id=int(user["id"]),
                project_id=project_id,
                raw_text=_clean_field(
                    raw_text,
                    "读者原始意见",
                    max_length=8000,
                    required=True,
                    min_length=2,
                ),
                request_type=request_type,
                impact_scope=impact_scope,
                priority=priority,
                constraints=_split_lines(constraints, limit=30),
                author_note=_clean_field(
                    author_note, "作者备注", max_length=4000
                ),
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}?error={quote(str(exc))}"
                "#reader-decisions",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/reader-requests/{request_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.get(
        "/reader-requests/{request_id}", response_class=HTMLResponse
    )
    async def reader_request_page(
        request: Request,
        request_id: str,
        error: Optional[str] = None,
        adopted: bool = False,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        reader_request = reader_service.get_request(
            user_id=int(user["id"]), request_id=request_id
        )
        if not reader_request:
            return render_template(
                "not_found.html",
                _template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        project = database.get_novel_project(
            int(user["id"]), str(reader_request["project_id"])
        )
        rolling_plan = planning_service.get_rolling_plan(
            user_id=int(user["id"]),
            project_id=str(reader_request["project_id"]),
        )
        return render_template(
            "reader_request.html",
            _template_context(
                request,
                user=user,
                reader_request=reader_request,
                project=project,
                rolling_plan=rolling_plan,
                error=error,
                adopted=adopted,
            ),
        )

    @application.post("/reader-requests/{request_id}/propose")
    async def propose_reader_branches(
        request: Request,
        request_id: str,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        reader_request = reader_service.get_request(
            user_id=int(user["id"]), request_id=request_id
        )
        if not reader_request:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        if str(reader_request["status"]) in {"adopted", "dismissed"}:
            return RedirectResponse(
                f"/reader-requests/{request_id}?error="
                + quote("这条读者意见已经处理"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        profile = api_profile(int(user["id"]))
        if not profile:
            return RedirectResponse(
                "/settings/api?error="
                + quote("生成剧情方案前，请先配置模型服务"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        try:
            job_id = database.create_reader_planning_job(
                user_id=int(user["id"]),
                project_id=str(reader_request["project_id"]),
                request_id=request_id,
                provider=profile["provider"],
                model=profile["model"],
                credential_source=profile["credential_source"],
                max_jobs_per_day=app_settings.max_jobs_per_day,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/reader-requests/{request_id}?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        request.app.state.worker.wake()
        return RedirectResponse(
            f"/writing-jobs/{job_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/reader-proposals/{proposal_id}/accept")
    async def accept_reader_proposal(
        request: Request,
        proposal_id: str,
        request_id: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            result = reader_service.accept_proposal(
                user_id=int(user["id"]), proposal_id=proposal_id
            )
            if not result:
                return Response(status_code=status.HTTP_404_NOT_FOUND)
        except ValueError as exc:
            reader_request = reader_service.get_request(
                user_id=int(user["id"]),
                request_id=request_id,
            )
            if not reader_request:
                return Response(status_code=status.HTTP_404_NOT_FOUND)
            return RedirectResponse(
                f"/reader-requests/{request_id}?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/reader-requests/{result['request_id']}?adopted=true",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/reader-requests/{request_id}/dismiss")
    async def dismiss_reader_request(
        request: Request,
        request_id: str,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            project_id = reader_service.dismiss_request(
                user_id=int(user["id"]), request_id=request_id
            )
            if not project_id:
                return Response(status_code=status.HTTP_404_NOT_FOUND)
        except ValueError as exc:
            return RedirectResponse(
                f"/reader-requests/{request_id}?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{project_id}#reader-decisions",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/novels/{project_id}/voice/suggestions")
    async def create_voice_profile_suggestion(
        request: Request,
        project_id: str,
        sample_title: str = Form(...),
        sample_text: str = Form(...),
        author_intent: str = Form(""),
        rights_confirmed: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            clean_title = _clean_field(
                sample_title,
                "样章名称",
                max_length=120,
                required=True,
                min_length=2,
            )
            clean_sample = _clean_field(
                sample_text,
                "作者样章",
                max_length=30_000,
                required=True,
                min_length=500,
            )
            clean_intent = _clean_field(
                author_intent,
                "希望保留或避免的方向",
                max_length=2000,
            )
            if rights_confirmed != "yes":
                raise ValueError("请确认样章是你有权用于分析的文本")
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}?error={quote(str(exc))}#voice-learning",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        profile = api_profile(int(user["id"]))
        if not profile:
            return RedirectResponse(
                "/settings/api?error="
                + quote("提取作品声纹前，请先配置模型服务"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        try:
            suggestion_id = style_service.create_voice_suggestion(
                user_id=int(user["id"]),
                project_id=project_id,
                sample_title=clean_title,
                sample_text=clean_sample,
                author_intent=clean_intent,
                provider=profile["provider"],
                model=profile["model"],
                credential_source=profile["credential_source"],
                max_jobs_per_day=app_settings.max_jobs_per_day,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}?error={quote(str(exc))}#voice-learning",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        request.app.state.worker.wake()
        return RedirectResponse(
            f"/voice-suggestions/{suggestion_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.get(
        "/voice-suggestions/{suggestion_id}", response_class=HTMLResponse
    )
    async def voice_profile_suggestion_page(
        request: Request,
        suggestion_id: str,
        error: Optional[str] = None,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        suggestion = style_service.get_voice_suggestion(
            user_id=int(user["id"]), suggestion_id=suggestion_id
        )
        if not suggestion:
            return render_template(
                "not_found.html",
                _template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        proposed = dict(suggestion.get("suggestion") or {})

        def _prefer(proposed_key: str, current_key: str) -> str:
            value = str(proposed.get(proposed_key) or "").strip()
            if value:
                return value
            return str(suggestion.get(current_key) or "")

        def _merge_items(
            current: list[Any], proposed_items: Any
        ) -> list[str]:
            merged: list[str] = []
            for raw in [*current, *(proposed_items or [])]:
                item = str(raw).strip()
                if item and item not in merged:
                    merged.append(item)
            return merged

        review_profile = {
            "narration_rules": _prefer(
                "narration_rules", "current_narration_rules"
            ),
            "sentence_rhythm": _prefer(
                "sentence_rhythm", "current_sentence_rhythm"
            ),
            "dialogue_voice": _prefer(
                "dialogue_voice", "current_dialogue_voice"
            ),
            "sensory_palette": _prefer(
                "sensory_palette", "current_sensory_palette"
            ),
            "metaphor_policy": _prefer(
                "metaphor_policy", "current_metaphor_policy"
            ),
            "allowed_omissions": _prefer(
                "allowed_omissions", "current_allowed_omissions"
            ),
            "preferred_patterns": _merge_items(
                list(suggestion.get("current_preferred_patterns") or []),
                proposed.get("preferred_patterns") or [],
            ),
            "banned_expressions": _merge_items(
                list(suggestion.get("current_banned_expressions") or []),
                proposed.get("banned_expressions") or [],
            ),
            "author_notes": str(
                suggestion.get("current_author_notes") or ""
            ),
        }
        return render_template(
            "voice_suggestion.html",
            _template_context(
                request,
                user=user,
                suggestion=suggestion,
                review_profile=review_profile,
                error=error,
            ),
        )

    @application.get("/api/voice-suggestions/{suggestion_id}")
    async def voice_profile_suggestion_status(
        request: Request, suggestion_id: str
    ):
        user = _current_user(request)
        if not user:
            return JSONResponse(
                {"detail": "未登录"}, status_code=status.HTTP_401_UNAUTHORIZED
            )
        suggestion = style_service.get_voice_suggestion(
            user_id=int(user["id"]), suggestion_id=suggestion_id
        )
        if not suggestion:
            return JSONResponse(
                {"detail": "任务不存在"},
                status_code=status.HTTP_404_NOT_FOUND,
            )
        current_status = str(suggestion["status"])
        return JSONResponse(
            {
                "status": current_status,
                "terminal": current_status
                in {"ready", "applied", "rejected", "failed"},
            }
        )

    @application.post("/voice-suggestions/{suggestion_id}/apply")
    async def apply_voice_profile_suggestion(
        request: Request,
        suggestion_id: str,
        narration_rules: str = Form(""),
        sentence_rhythm: str = Form(""),
        dialogue_voice: str = Form(""),
        sensory_palette: str = Form(""),
        metaphor_policy: str = Form(""),
        allowed_omissions: str = Form(""),
        preferred_patterns: str = Form(""),
        banned_expressions: str = Form(""),
        author_notes: str = Form(""),
        action: str = Form("apply_draft"),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        suggestion = style_service.get_voice_suggestion(
            user_id=int(user["id"]), suggestion_id=suggestion_id
        )
        if not suggestion:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        try:
            if action not in {"apply_draft", "apply_confirm"}:
                raise ValueError("不支持的声纹建议操作")
            fields = {
                "narration_rules": _clean_field(
                    narration_rules, "叙述规则", max_length=6000
                ),
                "sentence_rhythm": _clean_field(
                    sentence_rhythm, "句段节奏", max_length=4000
                ),
                "dialogue_voice": _clean_field(
                    dialogue_voice, "对话声音", max_length=6000
                ),
                "sensory_palette": _clean_field(
                    sensory_palette, "感官与意象", max_length=4000
                ),
                "metaphor_policy": _clean_field(
                    metaphor_policy, "比喻策略", max_length=4000
                ),
                "allowed_omissions": _clean_field(
                    allowed_omissions, "留白规则", max_length=4000
                ),
                "preferred_patterns": _split_lines(
                    preferred_patterns, limit=50
                ),
                "banned_expressions": _split_lines(
                    banned_expressions, limit=100
                ),
                "author_notes": _clean_field(
                    author_notes, "作者补充", max_length=6000
                ),
            }
            if action == "apply_confirm" and not any(
                (
                    fields["narration_rules"],
                    fields["sentence_rhythm"],
                    fields["dialogue_voice"],
                    fields["preferred_patterns"],
                    fields["banned_expressions"],
                )
            ):
                raise ValueError("至少保留一项可执行声纹规则后才能确认")
            project_id = style_service.apply_voice_suggestion(
                user_id=int(user["id"]),
                suggestion_id=suggestion_id,
                confirm=action == "apply_confirm",
                **fields,
            )
            if not project_id:
                raise ValueError("这份声纹建议已经处理或不再可用")
        except ValueError as exc:
            return RedirectResponse(
                f"/voice-suggestions/{suggestion_id}"
                f"?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{project_id}?voice_learned=true#voice",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/voice-suggestions/{suggestion_id}/reject")
    async def reject_voice_profile_suggestion(
        request: Request,
        suggestion_id: str,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        project_id = style_service.reject_voice_suggestion(
            user_id=int(user["id"]), suggestion_id=suggestion_id
        )
        if not project_id:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        return RedirectResponse(
            f"/novels/{project_id}#voice-learning",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    def _queue_edit_preference_suggestion(
        *,
        request: Request,
        user_id: int,
        project_id: str,
        chapter_id: str,
        source_type: str,
        after_version_id: str,
        scene_beat_id: Optional[str],
        error_path: str,
    ) -> RedirectResponse:
        profile = api_profile(user_id)
        if not profile:
            return RedirectResponse(
                "/settings/api?error="
                + quote("从手工改稿学习前，请先配置模型服务"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        try:
            suggestion_id = preference_service.create_suggestion(
                user_id=user_id,
                project_id=project_id,
                chapter_id=chapter_id,
                source_type=source_type,
                after_version_id=after_version_id,
                expected_scene_beat_id=scene_beat_id,
                provider=profile["provider"],
                model=profile["model"],
                credential_source=profile["credential_source"],
                max_jobs_per_day=app_settings.max_jobs_per_day,
            )
        except ValueError as exc:
            separator = "&" if "?" in error_path else "?"
            return RedirectResponse(
                f"{error_path}{separator}error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        request.app.state.worker.wake()
        return RedirectResponse(
            f"/editing-preference-suggestions/{suggestion_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post(
        "/novels/{project_id}/chapters/{chapter_id}"
        "/versions/{version_id}/learn-edit-preferences"
    )
    async def learn_chapter_edit_preferences(
        request: Request,
        project_id: str,
        chapter_id: str,
        version_id: str,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        return _queue_edit_preference_suggestion(
            request=request,
            user_id=int(user["id"]),
            project_id=project_id,
            chapter_id=chapter_id,
            source_type="chapter",
            after_version_id=version_id,
            scene_beat_id=None,
            error_path=f"/novels/{project_id}/chapters/{chapter_id}",
        )

    @application.post(
        "/novels/{project_id}/chapters/{chapter_id}"
        "/scenes/{scene_beat_id}/versions/{version_id}"
        "/learn-edit-preferences"
    )
    async def learn_scene_edit_preferences(
        request: Request,
        project_id: str,
        chapter_id: str,
        scene_beat_id: str,
        version_id: str,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        return _queue_edit_preference_suggestion(
            request=request,
            user_id=int(user["id"]),
            project_id=project_id,
            chapter_id=chapter_id,
            source_type="scene",
            after_version_id=version_id,
            scene_beat_id=scene_beat_id,
            error_path=(
                f"/novels/{project_id}/chapters/{chapter_id}/scenes"
            ),
        )

    @application.get(
        "/novels/{project_id}/editing-memory",
        response_class=HTMLResponse,
    )
    async def editing_memory_page(
        request: Request,
        project_id: str,
        error: Optional[str] = None,
        preference_learned: bool = False,
        aggregate_created: bool = False,
        aggregate_archived: bool = False,
        preference_archived: bool = False,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        project = database.get_novel_project(int(user["id"]), project_id)
        if not project:
            return render_template(
                "not_found.html",
                _template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        aggregates = preference_service.list_aggregates(
            user_id=int(user["id"]),
            project_id=project_id,
            include_archived=True,
            limit=40,
        )
        active_aggregates = [
            item for item in aggregates if item["status"] == "active"
        ]
        archived_aggregates = [
            item for item in aggregates if item["status"] == "archived"
        ]
        candidates = preference_service.list_aggregation_candidates(
            user_id=int(user["id"]), project_id=project_id
        )
        single_preferences = (
            preference_service.list_active_preferences(
                user_id=int(user["id"]),
                project_id=project_id,
                limit=80,
                exclude_aggregated=True,
            )
        )
        suggestions = preference_service.list_suggestions(
            user_id=int(user["id"]), project_id=project_id, limit=12
        )
        summary = preference_service.get_memory_summary(
            user_id=int(user["id"]), project_id=project_id
        ) or {
            "stable_count": 0,
            "single_count": 0,
            "conflict_count": 0,
        }
        summary["candidate_count"] = len(candidates)
        summary["awaiting_effect_count"] = sum(
            1
            for item in active_aggregates
            if (
                item.get("effect_observation")
                and item["effect_observation"]["status"]
                != "observable"
            )
        )
        return render_template(
            "editing_memory.html",
            _template_context(
                request,
                user=user,
                project=project,
                active_aggregates=active_aggregates,
                archived_aggregates=archived_aggregates,
                aggregation_candidates=candidates,
                single_preferences=single_preferences,
                editing_preference_suggestions=suggestions,
                editing_memory_summary=summary,
                error=error,
                preference_learned=preference_learned,
                aggregate_created=aggregate_created,
                aggregate_archived=aggregate_archived,
                preference_archived=preference_archived,
            ),
        )

    @application.post(
        "/novels/{project_id}/editing-memory/aggregates"
    )
    async def create_editing_preference_aggregate(
        request: Request, project_id: str
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        form = await request.form()
        verify_csrf(request, str(form.get("csrf") or ""))
        try:
            category = _clean_field(
                str(form.get("category") or ""),
                "偏好类别",
                max_length=40,
                required=True,
            )
            if category not in EDIT_PREFERENCE_CATEGORIES:
                raise ValueError("不支持的编辑偏好类别")
            support_ids = []
            conflict_ids = []
            for key, raw_role in form.multi_items():
                if not str(key).startswith("role_"):
                    continue
                preference_id = str(key)[5:].strip()
                if not preference_id or len(preference_id) > 128:
                    raise ValueError("改稿观察选择无效")
                role = str(raw_role or "")
                if role == "support":
                    support_ids.append(preference_id)
                elif role == "conflict":
                    conflict_ids.append(preference_id)
                elif role != "ignore":
                    raise ValueError("不支持的证据关系")
            preference_service.create_aggregate(
                user_id=int(user["id"]),
                project_id=project_id,
                category=category,
                guidance=_clean_field(
                    str(form.get("guidance") or ""),
                    "稳定偏好规则",
                    max_length=600,
                    required=True,
                    min_length=8,
                ),
                applicability=_clean_field(
                    str(form.get("applicability") or ""),
                    "适用范围",
                    max_length=500,
                    required=True,
                    min_length=4,
                ),
                author_note=_clean_field(
                    str(form.get("author_note") or ""),
                    "作者说明",
                    max_length=1000,
                    required=False,
                ),
                support_preference_ids=support_ids,
                conflict_preference_ids=conflict_ids,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}/editing-memory"
                f"?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{project_id}/editing-memory"
            "?aggregate_created=true",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.get(
        "/editing-preference-suggestions/{suggestion_id}",
        response_class=HTMLResponse,
    )
    async def editing_preference_suggestion_page(
        request: Request,
        suggestion_id: str,
        error: Optional[str] = None,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        suggestion = preference_service.get_suggestion(
            user_id=int(user["id"]), suggestion_id=suggestion_id
        )
        if not suggestion:
            return render_template(
                "not_found.html",
                _template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        return render_template(
            "editing_preference_suggestion.html",
            _template_context(
                request,
                user=user,
                suggestion=suggestion,
                category_options=EDIT_PREFERENCE_CATEGORY_OPTIONS,
                error=error,
            ),
        )

    @application.get(
        "/api/editing-preference-suggestions/{suggestion_id}"
    )
    async def editing_preference_suggestion_status(
        request: Request, suggestion_id: str
    ):
        user = _current_user(request)
        if not user:
            return JSONResponse(
                {"detail": "未登录"}, status_code=status.HTTP_401_UNAUTHORIZED
            )
        suggestion = preference_service.get_suggestion(
            user_id=int(user["id"]), suggestion_id=suggestion_id
        )
        if not suggestion:
            return JSONResponse(
                {"detail": "任务不存在"},
                status_code=status.HTTP_404_NOT_FOUND,
            )
        current_status = str(suggestion["status"])
        return JSONResponse(
            {
                "status": current_status,
                "terminal": current_status
                in {"ready", "applied", "rejected", "failed"},
            }
        )

    @application.post(
        "/editing-preference-suggestions/{suggestion_id}/apply"
    )
    async def apply_editing_preference_suggestion(
        request: Request, suggestion_id: str
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        form = await request.form()
        verify_csrf(request, str(form.get("csrf") or ""))
        try:
            raw_indices = list(form.getlist("selected"))
            if not raw_indices:
                raise ValueError("至少选择一条你确认的编辑偏好")
            if len(raw_indices) > 6:
                raise ValueError("一次最多确认 6 条编辑偏好")
            selections = []
            seen: set[int] = set()
            for raw_index in raw_indices:
                try:
                    index = int(str(raw_index))
                except ValueError as exc:
                    raise ValueError("编辑偏好选择无效") from exc
                if index in seen:
                    raise ValueError("编辑偏好选择重复")
                seen.add(index)
                category = _clean_field(
                    str(form.get(f"category_{index}") or ""),
                    "偏好类别",
                    max_length=40,
                    required=True,
                )
                if category not in EDIT_PREFERENCE_CATEGORIES:
                    raise ValueError("不支持的编辑偏好类别")
                selections.append(
                    {
                        "index": index,
                        "category": category,
                        "guidance": _clean_field(
                            str(form.get(f"guidance_{index}") or ""),
                            "偏好规则",
                            max_length=600,
                            required=True,
                            min_length=8,
                        ),
                        "applicability": _clean_field(
                            str(
                                form.get(f"applicability_{index}") or ""
                            ),
                            "适用范围",
                            max_length=500,
                            required=True,
                            min_length=4,
                        ),
                    }
                )
            project_id = preference_service.apply_suggestion(
                user_id=int(user["id"]),
                suggestion_id=suggestion_id,
                selections=selections,
            )
            if not project_id:
                raise ValueError("这份编辑偏好建议已经处理或不再可用")
        except ValueError as exc:
            return RedirectResponse(
                f"/editing-preference-suggestions/{suggestion_id}"
                f"?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{project_id}/editing-memory"
            "?preference_learned=true",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post(
        "/editing-preference-suggestions/{suggestion_id}/reject"
    )
    async def reject_editing_preference_suggestion(
        request: Request,
        suggestion_id: str,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        project_id = preference_service.reject_suggestion(
            user_id=int(user["id"]), suggestion_id=suggestion_id
        )
        if not project_id:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        return RedirectResponse(
            f"/editing-preference-suggestions/{suggestion_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/editing-preferences/{preference_id}/archive")
    async def archive_editing_preference(
        request: Request,
        preference_id: str,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        project_id = preference_service.archive_preference(
            user_id=int(user["id"]), preference_id=preference_id
        )
        if not project_id:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        return RedirectResponse(
            f"/novels/{project_id}/editing-memory"
            "?preference_archived=true",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post(
        "/editing-preference-aggregates/{aggregate_id}/archive"
    )
    async def archive_editing_preference_aggregate(
        request: Request,
        aggregate_id: str,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        project_id = preference_service.archive_aggregate(
            user_id=int(user["id"]), aggregate_id=aggregate_id
        )
        if not project_id:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        return RedirectResponse(
            f"/novels/{project_id}/editing-memory"
            "?aggregate_archived=true",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/novels/{project_id}/voice")
    async def update_novel_voice_profile(
        request: Request,
        project_id: str,
        narration_rules: str = Form(""),
        sentence_rhythm: str = Form(""),
        dialogue_voice: str = Form(""),
        sensory_palette: str = Form(""),
        metaphor_policy: str = Form(""),
        allowed_omissions: str = Form(""),
        preferred_patterns: str = Form(""),
        banned_expressions: str = Form(""),
        author_notes: str = Form(""),
        action: str = Form("save_draft"),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            if action not in {"save_draft", "confirm"}:
                raise ValueError("不支持的声纹操作")
            fields = {
                "narration_rules": _clean_field(
                    narration_rules, "叙述规则", max_length=6000
                ),
                "sentence_rhythm": _clean_field(
                    sentence_rhythm, "句段节奏", max_length=4000
                ),
                "dialogue_voice": _clean_field(
                    dialogue_voice, "对话声音", max_length=6000
                ),
                "sensory_palette": _clean_field(
                    sensory_palette, "感官与意象", max_length=4000
                ),
                "metaphor_policy": _clean_field(
                    metaphor_policy, "比喻策略", max_length=4000
                ),
                "allowed_omissions": _clean_field(
                    allowed_omissions, "留白规则", max_length=4000
                ),
                "preferred_patterns": _split_lines(
                    preferred_patterns, limit=50
                ),
                "banned_expressions": _split_lines(
                    banned_expressions, limit=100
                ),
                "author_notes": _clean_field(
                    author_notes, "作者补充", max_length=6000
                ),
            }
            if action == "confirm" and not any(
                (
                    fields["narration_rules"],
                    fields["sentence_rhythm"],
                    fields["dialogue_voice"],
                    fields["preferred_patterns"],
                    fields["banned_expressions"],
                )
            ):
                raise ValueError("至少填写一项可执行声纹规则后才能确认")
            updated = style_service.update_voice_profile(
                user_id=int(user["id"]),
                project_id=project_id,
                confirm=action == "confirm",
                **fields,
            )
            if not updated:
                return Response(status_code=status.HTTP_404_NOT_FOUND)
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}?error={quote(str(exc))}#voice",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{project_id}?saved=true#voice",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/novels/{project_id}/volumes")
    async def add_novel_volume(
        request: Request,
        project_id: str,
        title: str = Form(...),
        goal: str = Form(""),
        start_state: str = Form(""),
        end_state: str = Form(""),
        major_conflict: str = Form(""),
        payoff: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            planning_service.create_volume(
                user_id=int(user["id"]),
                project_id=project_id,
                title=_clean_field(
                    title, "分卷名", max_length=120, required=True
                ),
                goal=_clean_field(goal, "分卷目标", max_length=4000),
                start_state=_clean_field(
                    start_state, "开卷状态", max_length=4000
                ),
                end_state=_clean_field(
                    end_state, "收卷状态", max_length=4000
                ),
                major_conflict=_clean_field(
                    major_conflict, "主要冲突", max_length=4000
                ),
                payoff=_clean_field(
                    payoff, "本卷回报", max_length=4000
                ),
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}?error={quote(str(exc))}#volumes",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{project_id}#volumes",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/novels/{project_id}/volumes/{volume_id}/edit")
    async def edit_novel_volume(
        request: Request,
        project_id: str,
        volume_id: str,
        title: str = Form(...),
        goal: str = Form(""),
        start_state: str = Form(""),
        end_state: str = Form(""),
        major_conflict: str = Form(""),
        payoff: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            planning_service.update_volume(
                user_id=int(user["id"]),
                project_id=project_id,
                volume_id=volume_id,
                title=_clean_field(
                    title, "分卷名", max_length=120, required=True
                ),
                goal=_clean_field(goal, "分卷目标", max_length=4000),
                start_state=_clean_field(
                    start_state, "开卷状态", max_length=4000
                ),
                end_state=_clean_field(
                    end_state, "收卷状态", max_length=4000
                ),
                major_conflict=_clean_field(
                    major_conflict, "主要冲突", max_length=4000
                ),
                payoff=_clean_field(
                    payoff, "本卷回报", max_length=4000
                ),
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}?error={quote(str(exc))}#volumes",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{project_id}?volume_saved=true#volumes",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/novels/{project_id}/characters")
    async def add_novel_character(
        request: Request,
        project_id: str,
        name: str = Form(...),
        role: str = Form(""),
        traits: str = Form(""),
        background: str = Form(""),
        character_arc: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            database.add_novel_character(
                user_id=int(user["id"]),
                project_id=project_id,
                name=_clean_field(
                    name, "人物名", max_length=60, required=True
                ),
                role=_clean_field(role, "人物定位", max_length=300),
                traits=_clean_field(traits, "性格特征", max_length=1000),
                background=_clean_field(background, "人物背景", max_length=4000),
                character_arc=_clean_field(
                    character_arc, "人物弧光", max_length=2000
                ),
            )
        except ValueError as exc:
            message = str(exc)
            return RedirectResponse(
                f"/novels/{project_id}?error={quote(message)}#characters",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        except Exception as exc:
            duplicate = "UNIQUE constraint failed" in str(exc)
            if not duplicate:
                logger.exception("failed to add novel character")
            message = "该人物名已经存在" if duplicate else "添加人物失败"
            return RedirectResponse(
                f"/novels/{project_id}?error={quote(message)}#characters",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{project_id}#characters",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post(
        "/novels/{project_id}/characters/{character_id}/delete"
    )
    async def delete_novel_character(
        request: Request,
        project_id: str,
        character_id: str,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        database.delete_novel_character(
            int(user["id"]), project_id, character_id
        )
        return RedirectResponse(
            f"/novels/{project_id}#characters",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/novels/{project_id}/chapters")
    async def add_novel_chapter(
        request: Request,
        project_id: str,
        title: str = Form(""),
        outline: str = Form(""),
        key_points: str = Form(""),
        volume_id: str = Form(""),
        return_to_workbench: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        chapter_id = secrets.token_hex(16)
        chapter_dir = (
            app_settings.novels_dir
            / str(user["id"])
            / project_id
            / "chapters"
            / chapter_id
        )
        try:
            clean_title = _clean_field(
                title, "章节名", max_length=120
            )
            clean_outline = _clean_field(
                outline, "章节大纲", max_length=6000
            )
            clean_key_points = _clean_field(
                key_points, "关键情节点", max_length=4000
            )
            (chapter_dir / "versions").mkdir(
                parents=True, exist_ok=False, mode=0o700
            )
            os.chmod(chapter_dir, 0o700)
            os.chmod(chapter_dir / "versions", 0o700)
            content_path = chapter_dir / "content.txt"
            content_path.write_text("", encoding="utf-8")
            content_path.chmod(0o600)
            database.add_novel_chapter(
                user_id=int(user["id"]),
                project_id=project_id,
                chapter_id=chapter_id,
                title=clean_title,
                outline=clean_outline,
                key_points=clean_key_points,
                content_path=content_path,
                volume_id=volume_id or None,
            )
        except ValueError as exc:
            shutil.rmtree(chapter_dir, ignore_errors=True)
            if return_to_workbench:
                return RedirectResponse(
                    f"/novels/{project_id}/workbench"
                    f"?error={quote(str(exc))}",
                    status_code=status.HTTP_303_SEE_OTHER,
                )
            return RedirectResponse(
                f"/novels/{project_id}?error={quote(str(exc))}#chapters",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        except Exception:
            shutil.rmtree(chapter_dir, ignore_errors=True)
            logger.exception("failed to add novel chapter")
            if return_to_workbench:
                return RedirectResponse(
                    f"/novels/{project_id}/workbench"
                    f"?error={quote('添加章节失败')}",
                    status_code=status.HTTP_303_SEE_OTHER,
                )
            return RedirectResponse(
                f"/novels/{project_id}?error={quote('添加章节失败')}#chapters",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        destination = (
            f"/novels/{project_id}/workbench?chapter_id={chapter_id}"
            if return_to_workbench
            else f"/novels/{project_id}/chapters/{chapter_id}"
        )
        return RedirectResponse(
            destination, status_code=status.HTTP_303_SEE_OTHER
        )

    @application.get(
        "/novels/{project_id}/chapters/{chapter_id}",
        response_class=HTMLResponse,
    )
    async def novel_chapter_editor(
        request: Request,
        project_id: str,
        chapter_id: str,
        error: Optional[str] = None,
        saved: bool = False,
        canonical: bool = False,
        memory_saved: bool = False,
        scene_assembled: bool = False,
        assistant_rewrite: bool = False,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        chapter = database.get_novel_chapter(
            int(user["id"]), project_id, chapter_id
        )
        if not chapter:
            return render_template(
                "not_found.html",
                _template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        content = _read_optional_text(Path(str(chapter["content_path"])))
        versions = database.list_chapter_versions(
            int(user["id"]), project_id, chapter_id
        )
        deltas = memory_service.list_chapter_deltas(
            user_id=int(user["id"]),
            project_id=project_id,
            chapter_id=chapter_id,
        )
        task_card = planning_service.get_task_card(
            user_id=int(user["id"]),
            project_id=project_id,
            chapter_id=chapter_id,
        )
        voice_profile = style_service.get_voice_profile(
            user_id=int(user["id"]), project_id=project_id
        )
        working_version = next(
            (
                version
                for version in versions
                if str(version["id"])
                == str(chapter.get("working_version_id") or "")
            ),
            None,
        )
        working_version_hash = ""
        working_version_matches_editor = False
        if working_version:
            working_version_content = _read_optional_text(
                Path(str(working_version["content_path"]))
            )
            working_version_matches_editor = (
                working_version_content == content
            )
            if working_version_matches_editor:
                working_version_hash = hashlib.sha256(
                    working_version_content.encode("utf-8")
                ).hexdigest()
        writing_context = database.get_writing_context(
            int(user["id"]), chapter_id
        )
        technique_cards = (
            writing_context.get("technique_cards") or []
            if writing_context
            else []
        )
        writing_techniques = compile_active_techniques(
            technique_cards, usage="write"
        )
        audit_techniques = compile_active_techniques(
            technique_cards, usage="audit"
        )
        chapter_workflow = workflow_service.get_state(
            user_id=int(user["id"]),
            project_id=project_id,
            chapter_id=chapter_id,
        )
        return render_template(
            "novel_chapter.html",
            _template_context(
                request,
                user=user,
                chapter=chapter,
                content=content,
                versions=versions,
                deltas=deltas,
                task_card=task_card,
                voice_profile=voice_profile,
                working_version=working_version,
                working_version_hash=working_version_hash,
                working_version_matches_editor=(
                    working_version_matches_editor
                ),
                writing_techniques=writing_techniques,
                audit_techniques=audit_techniques,
                chapter_workflow=chapter_workflow,
                error=error,
                saved=saved,
                canonical=canonical,
                memory_saved=memory_saved,
                scene_assembled=scene_assembled,
                assistant_rewrite=assistant_rewrite,
            ),
        )

    @application.get(
        "/novels/{project_id}/chapters/{chapter_id}/scenes",
        response_class=HTMLResponse,
    )
    async def scene_workbench_page(
        request: Request,
        project_id: str,
        chapter_id: str,
        error: Optional[str] = None,
        saved: bool = False,
        overridden: bool = False,
        restored: bool = False,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        user_id = int(user["id"])
        workbench = scene_service.get_workbench(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
        )
        if not workbench:
            return render_template(
                "not_found.html",
                _template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        for scene in workbench["scenes"]:
            writing_context = database.get_writing_context(
                user_id, chapter_id, str(scene["id"])
            )
            scene["writing_techniques"] = compile_active_techniques(
                (
                    writing_context.get("technique_cards") or []
                    if writing_context
                    else []
                ),
                usage="write",
            )
        chapter_workflow = workflow_service.get_state(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
        )
        return render_template(
            "scene_workbench.html",
            _template_context(
                request,
                user=user,
                workbench=workbench,
                chapter=workbench["chapter"],
                scenes=workbench["scenes"],
                chapter_workflow=chapter_workflow,
                error=error,
                saved=saved,
                overridden=overridden,
                restored=restored,
            ),
        )

    @application.post(
        "/novels/{project_id}/chapters/{chapter_id}/scenes/assemble"
    )
    async def assemble_scene_drafts(
        request: Request,
        project_id: str,
        chapter_id: str,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        try:
            assembly = scene_service.build_assembly(
                user_id=user_id,
                project_id=project_id,
                chapter_id=chapter_id,
            )
            workbench = scene_service.get_workbench(
                user_id=user_id,
                project_id=project_id,
                chapter_id=chapter_id,
            )
            if not workbench:
                raise ValueError("章节不存在")
            content = str(assembly["content"])
            chapter_content_path = Path(
                str(workbench["chapter"]["content_path"])
            )
            token = secrets.token_hex(16)
            version_path = (
                chapter_content_path.parent
                / "versions"
                / f"scene-assembly-{token}.txt"
            )
            previous_content = _read_optional_text(chapter_content_path)
            _atomic_write_text(version_path, content, token)
            _atomic_write_text(chapter_content_path, content, token)
            try:
                scene_service.record_assembly(
                    user_id=user_id,
                    project_id=project_id,
                    chapter_id=chapter_id,
                    version_path=version_path,
                    content=content,
                    scene_versions=assembly["scene_versions"],
                )
            except Exception:
                _atomic_write_text(
                    chapter_content_path, previous_content, token
                )
                raise
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}/scenes"
                f"?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        except Exception:
            logger.exception("failed to assemble scene drafts")
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}/scenes"
                f"?error={quote('组装场景失败')}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{project_id}/chapters/{chapter_id}"
            "?scene_assembled=true",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post(
        "/novels/{project_id}/chapters/{chapter_id}"
        "/scenes/{scene_beat_id}/save"
    )
    async def save_scene_draft(
        request: Request,
        project_id: str,
        chapter_id: str,
        scene_beat_id: str,
        content: str = Form(...),
        change_summary: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            clean_content = _clean_field(
                content,
                "场景正文",
                max_length=60_000,
                required=True,
                min_length=1,
            )
            clean_change_summary = _clean_field(
                change_summary,
                "这次主要修改了什么",
                max_length=1000,
            )
            workbench = scene_service.get_workbench(
                user_id=int(user["id"]),
                project_id=project_id,
                chapter_id=chapter_id,
            )
            if not workbench:
                raise ValueError("章节不存在")
            if workbench["active_job"]:
                raise ValueError("AI 任务正在运行，请完成后再保存场景")
            if not any(
                str(item["id"]) == scene_beat_id
                for item in workbench["scenes"]
            ):
                raise ValueError("场景节拍不存在")
            token = secrets.token_hex(16)
            version_path = (
                Path(str(workbench["chapter"]["content_path"])).parent
                / "scenes"
                / scene_beat_id
                / "versions"
                / f"manual-{token}.txt"
            )
            _atomic_write_text(version_path, clean_content, token)
            scene_service.record_manual_version(
                user_id=int(user["id"]),
                project_id=project_id,
                chapter_id=chapter_id,
                scene_beat_id=scene_beat_id,
                version_path=version_path,
                content=clean_content,
                change_summary=clean_change_summary,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}/scenes"
                f"?error={quote(str(exc))}#scene-{scene_beat_id}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        except Exception:
            logger.exception("failed to save scene draft")
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}/scenes"
                f"?error={quote('保存场景失败')}#scene-{scene_beat_id}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{project_id}/chapters/{chapter_id}/scenes"
            f"?saved=true#scene-{scene_beat_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post(
        "/novels/{project_id}/chapters/{chapter_id}"
        "/scenes/{scene_beat_id}/generate"
    )
    async def generate_scene_draft(
        request: Request,
        project_id: str,
        chapter_id: str,
        scene_beat_id: str,
        operation: str = Form("generate_scene"),
        instruction: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        if operation not in {"generate_scene", "rewrite_scene"}:
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}/scenes"
                f"?error={quote('不支持的场景写作操作')}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        try:
            clean_instruction = _clean_field(
                instruction, "额外要求", max_length=4000
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}/scenes"
                f"?error={quote(str(exc))}#scene-{scene_beat_id}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        profile = api_profile(int(user["id"]))
        if not profile:
            return RedirectResponse(
                "/settings/api?error="
                + quote("开始写作前，请先配置模型服务"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        try:
            job_id = database.create_generation_job(
                user_id=int(user["id"]),
                project_id=project_id,
                chapter_id=chapter_id,
                operation=operation,
                instruction=clean_instruction,
                provider=profile["provider"],
                model=profile["model"],
                credential_source=profile["credential_source"],
                subject_id=scene_beat_id,
                max_jobs_per_day=app_settings.max_jobs_per_day,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}/scenes"
                f"?error={quote(str(exc))}#scene-{scene_beat_id}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        request.app.state.worker.wake()
        return RedirectResponse(
            f"/writing-jobs/{job_id}", status_code=status.HTTP_303_SEE_OTHER
        )

    @application.post(
        "/novels/{project_id}/chapters/{chapter_id}"
        "/scenes/{scene_beat_id}/versions/{scene_version_id}/audit"
    )
    async def audit_scene_draft(
        request: Request,
        project_id: str,
        chapter_id: str,
        scene_beat_id: str,
        scene_version_id: str,
        csrf: str = Form(...),
    ):
        del scene_beat_id
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        profile = api_profile(int(user["id"]))
        if not profile:
            return RedirectResponse(
                "/settings/api?error="
                + quote("执行场景检查前，请先配置模型服务"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        try:
            job_id = database.create_generation_job(
                user_id=int(user["id"]),
                project_id=project_id,
                chapter_id=chapter_id,
                operation="audit_scene",
                instruction="",
                provider=profile["provider"],
                model=profile["model"],
                credential_source=profile["credential_source"],
                subject_id=scene_version_id,
                max_jobs_per_day=app_settings.max_jobs_per_day,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}/scenes"
                f"?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        request.app.state.worker.wake()
        return RedirectResponse(
            f"/writing-jobs/{job_id}", status_code=status.HTTP_303_SEE_OTHER
        )

    @application.post(
        "/novels/{project_id}/chapters/{chapter_id}"
        "/scenes/{scene_beat_id}/versions/{scene_version_id}/override"
    )
    async def override_scene_audit(
        request: Request,
        project_id: str,
        chapter_id: str,
        scene_beat_id: str,
        scene_version_id: str,
        reason: str = Form(...),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            changed = scene_service.override_audit(
                user_id=int(user["id"]),
                project_id=project_id,
                chapter_id=chapter_id,
                scene_beat_id=scene_beat_id,
                scene_version_id=scene_version_id,
                reason=reason,
            )
            if not changed:
                raise ValueError("场景版本已变化或不需要覆盖")
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}/scenes"
                f"?error={quote(str(exc))}#scene-{scene_beat_id}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{project_id}/chapters/{chapter_id}/scenes"
            f"?overridden=true#scene-{scene_beat_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post(
        "/novels/{project_id}/chapters/{chapter_id}"
        "/scenes/{scene_beat_id}/versions/{scene_version_id}/restore"
    )
    async def restore_scene_version(
        request: Request,
        project_id: str,
        chapter_id: str,
        scene_beat_id: str,
        scene_version_id: str,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            workbench = scene_service.get_workbench(
                user_id=int(user["id"]),
                project_id=project_id,
                chapter_id=chapter_id,
            )
            if not workbench:
                raise ValueError("章节不存在")
            if workbench["active_job"]:
                raise ValueError(
                    "AI 任务正在运行，请完成后再恢复场景版本"
                )
            scene = next(
                (
                    item
                    for item in workbench["scenes"]
                    if str(item["id"]) == scene_beat_id
                ),
                None,
            )
            if not scene:
                raise ValueError("场景节拍不存在")
            version = next(
                (
                    item
                    for item in scene["versions"]
                    if str(item["id"]) == scene_version_id
                ),
                None,
            )
            if not version:
                raise ValueError("场景历史版本不存在")
            restored_content = _read_optional_text(
                Path(str(version["content_path"]))
            )
            if not restored_content.strip():
                raise ValueError("场景历史版本内容为空")
            token = secrets.token_hex(16)
            restored_path = (
                Path(str(workbench["chapter"]["content_path"])).parent
                / "scenes"
                / scene_beat_id
                / "versions"
                / f"restored-{token}.txt"
            )
            _atomic_write_text(restored_path, restored_content, token)
            scene_service.record_manual_version(
                user_id=int(user["id"]),
                project_id=project_id,
                chapter_id=chapter_id,
                scene_beat_id=scene_beat_id,
                version_path=restored_path,
                content=restored_content,
                source="restored",
                kind="scene_restore",
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}/scenes"
                f"?error={quote(str(exc))}#scene-{scene_beat_id}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        except Exception:
            logger.exception("failed to restore scene version")
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}/scenes"
                f"?error={quote('恢复场景版本失败')}#scene-{scene_beat_id}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{project_id}/chapters/{chapter_id}/scenes"
            f"?restored=true#scene-{scene_beat_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.get(
        "/novels/{project_id}/chapters/{chapter_id}/task-card",
        response_class=HTMLResponse,
    )
    async def chapter_task_card_page(
        request: Request,
        project_id: str,
        chapter_id: str,
        error: Optional[str] = None,
        saved: bool = False,
        confirmed: bool = False,
        skeleton_saved: bool = False,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        user_id = int(user["id"])
        task_card = planning_service.get_task_card(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
        )
        if not task_card:
            return render_template(
                "not_found.html",
                _template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        volumes = planning_service.list_volumes(
            user_id=user_id, project_id=project_id
        )
        characters = database.list_novel_characters(user_id, project_id)
        scene_slots = [dict(scene) for scene in task_card["scenes"]]
        while len(scene_slots) < 5:
            scene_slots.append({})
        writing_context = database.get_writing_context(user_id, chapter_id)
        planning_techniques = compile_active_techniques(
            (
                writing_context.get("technique_cards") or []
                if writing_context
                else []
            ),
            usage="plan",
        )
        planning_story_plan = compile_story_plan_context(
            writing_context or {}, usage="plan"
        )
        planning_causal_links = compile_planned_causal_links(
            writing_context or {}, usage="plan"
        )
        structure_arc_options = [
            item
            for item in planning_story_plan["plot_arcs"]
            if str(item.get("lifecycle_status") or "")
            in {"planned", "active"}
        ]
        confirmed_arc_titles = {
            str(item.get("title") or "")
            for item in planning_story_plan["plot_arcs"]
        }
        custom_plot_threads = [
            item
            for item in task_card["plot_threads"]
            if item not in confirmed_arc_titles
        ]
        form_plot_threads = [
            str(item.get("title") or "")
            for item in planning_story_plan["plot_arcs"]
            if str(item.get("title") or "") in task_card["plot_threads"]
        ] + custom_plot_threads
        chapter_workflow = workflow_service.get_state(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
        )
        return render_template(
            "chapter_task_card.html",
            _template_context(
                request,
                user=user,
                task_card=task_card,
                volumes=volumes,
                characters=characters,
                scene_slots=scene_slots,
                planning_techniques=planning_techniques,
                planning_story_plan=planning_story_plan,
                planning_causal_links=planning_causal_links,
                structure_arc_options=structure_arc_options,
                chapter_structure_role_options=(
                    CHAPTER_STRUCTURE_ROLE_OPTIONS
                ),
                custom_plot_threads=custom_plot_threads,
                form_plot_threads=form_plot_threads,
                chapter_workflow=chapter_workflow,
                error=error,
                saved=saved,
                confirmed=confirmed,
                skeleton_saved=skeleton_saved,
            ),
        )

    @application.post(
        "/novels/{project_id}/chapters/{chapter_id}/skeleton"
    )
    async def save_future_chapter_skeleton(
        request: Request,
        project_id: str,
        chapter_id: str,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        form = await request.form()
        verify_csrf(request, str(form.get("csrf") or ""))

        def value(name: str) -> str:
            return str(form.get(name) or "").strip()

        try:
            arc_titles = [
                str(item).strip()
                for item in form.getlist("arc_titles")
                if str(item).strip()
            ]
            skeleton = AuthorChapterSkeleton.model_validate(
                {
                    "title": value("title"),
                    "structural_role": value("structural_role"),
                    "purpose": value("purpose"),
                    "key_points": _split_lines(
                        value("key_points"), limit=5
                    ),
                    "arc_titles": arc_titles,
                    "ending_hook": value("ending_hook"),
                }
            )
            planning_service.update_future_chapter_skeleton(
                user_id=int(user["id"]),
                project_id=project_id,
                chapter_id=chapter_id,
                volume_id=value("volume_id") or None,
                skeleton=skeleton,
            )
        except (TypeError, ValueError) as exc:
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}/task-card"
                f"?error={quote(str(exc)[:1000])}#rolling-skeleton",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{project_id}/chapters/{chapter_id}/task-card"
            "?skeleton_saved=true#rolling-skeleton",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post(
        "/novels/{project_id}/chapters/{chapter_id}/task-card"
    )
    async def save_chapter_task_card(
        request: Request,
        project_id: str,
        chapter_id: str,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        form = await request.form()
        verify_csrf(request, str(form.get("csrf") or ""))

        def value(name: str) -> str:
            return str(form.get(name) or "").strip()

        try:
            action = value("action") or "save_draft"
            if action not in {"save_draft", "confirm"}:
                raise ValueError("不支持的任务卡操作")
            selected_plot_threads = [
                str(item).strip()
                for item in form.getlist("selected_plot_threads")
                if str(item).strip()
            ]
            custom_plot_threads = _split_lines(
                value("custom_plot_threads") or value("plot_threads"),
                limit=20,
            )
            combined_plot_threads = list(
                dict.fromkeys(
                    [*selected_plot_threads, *custom_plot_threads]
                )
            )
            if len(combined_plot_threads) > 20:
                raise ValueError("推进的剧情线不能超过 20 条")
            must_happen = _split_lines(
                value("must_happen"), limit=30
            )
            must_preserve = _split_lines(
                value("must_preserve"), limit=30
            )
            forbidden = _split_lines(
                value("forbidden"), limit=30
            )
            foreshadow_setup = _split_lines(
                value("foreshadow_setup"), limit=20
            )
            foreshadow_payoff = _split_lines(
                value("foreshadow_payoff"), limit=20
            )
            ending_hook = value("ending_hook")
            requirement_sources = {
                "plot_thread": combined_plot_threads,
                "must_happen": must_happen,
                "foreshadow_setup": foreshadow_setup,
                "foreshadow_payoff": foreshadow_payoff,
                "ending_hook": [ending_hook] if ending_hook else [],
            }
            scenes = []
            for position in range(1, 6):
                requirement_refs = []
                seen_requirement_tokens = set()
                for raw_token in form.getlist(
                    f"scene_requirement_{position}"
                ):
                    token = str(raw_token)
                    if token in seen_requirement_tokens:
                        continue
                    seen_requirement_tokens.add(token)
                    try:
                        kind, raw_index = token.rsplit(":", 1)
                        index = int(raw_index)
                        text = requirement_sources[kind][index]
                    except (
                        KeyError,
                        IndexError,
                        TypeError,
                        ValueError,
                    ) as exc:
                        raise ValueError(
                            "场景要求映射已经失效，请重新打开任务卡"
                        ) from exc
                    requirement_refs.append(
                        {"kind": kind, "text": text}
                    )
                scene_values = {
                    "pov_character": value(
                        f"scene_pov_character_{position}"
                    ),
                    "goal": value(f"scene_goal_{position}"),
                    "obstacle": value(f"scene_obstacle_{position}"),
                    "action": value(f"scene_action_{position}"),
                    "reveal": value(f"scene_reveal_{position}"),
                    "conceal": value(f"scene_conceal_{position}"),
                    "subtext": value(f"scene_subtext_{position}"),
                    "location": value(f"scene_location_{position}"),
                    "key_items": _split_lines(
                        value(f"scene_key_items_{position}"), limit=20
                    ),
                    "end_state": value(f"scene_end_state_{position}"),
                    "transition": value(f"scene_transition_{position}"),
                    "requirement_refs": requirement_refs,
                }
                if any(
                    item
                    for key, item in scene_values.items()
                    if key not in {"key_items", "requirement_refs"}
                ) or scene_values["key_items"] or requirement_refs:
                    scenes.append(SceneBeat.model_validate(scene_values))
            card = ChapterTaskCard.model_validate(
                {
                    "purpose": value("purpose"),
                    "start_state": value("start_state"),
                    "end_state": value("end_state"),
                    "central_conflict": value("central_conflict"),
                    "emotional_value": value("emotional_value"),
                    "plot_threads": combined_plot_threads,
                    "must_happen": must_happen,
                    "must_preserve": must_preserve,
                    "forbidden": forbidden,
                    "foreshadow_setup": foreshadow_setup,
                    "foreshadow_payoff": foreshadow_payoff,
                    "ending_hook": ending_hook,
                    "target_chars": int(value("target_chars") or "3000"),
                    "scenes": scenes,
                }
            )
            planning_service.upsert_task_card(
                user_id=int(user["id"]),
                project_id=project_id,
                chapter_id=chapter_id,
                volume_id=value("volume_id") or None,
                card=card,
                confirm=action == "confirm",
            )
        except (TypeError, ValueError) as exc:
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}/task-card"
                f"?error={quote(str(exc)[:1000])}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        query = "confirmed=true" if action == "confirm" else "saved=true"
        return RedirectResponse(
            f"/novels/{project_id}/chapters/{chapter_id}/task-card?{query}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post(
        "/novels/{project_id}/chapters/{chapter_id}/task-card/generate"
    )
    async def generate_chapter_task_card(
        request: Request,
        project_id: str,
        chapter_id: str,
        instruction: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        task_card = planning_service.get_task_card(
            user_id=int(user["id"]),
            project_id=project_id,
            chapter_id=chapter_id,
        )
        if not task_card:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        try:
            clean_instruction = _clean_field(
                instruction, "Planner 额外要求", max_length=4000
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}/task-card"
                f"?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        profile = api_profile(int(user["id"]))
        if not profile:
            return RedirectResponse(
                "/settings/api?error="
                + quote("使用 Planner 前，请先配置模型服务"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        try:
            job_id = database.create_chapter_planning_job(
                user_id=int(user["id"]),
                project_id=project_id,
                chapter_id=chapter_id,
                instruction=clean_instruction,
                provider=profile["provider"],
                model=profile["model"],
                credential_source=profile["credential_source"],
                max_jobs_per_day=app_settings.max_jobs_per_day,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}/task-card"
                f"?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        request.app.state.worker.wake()
        return RedirectResponse(
            f"/writing-jobs/{job_id}", status_code=status.HTTP_303_SEE_OTHER
        )

    @application.post(
        "/novels/{project_id}/chapters/{chapter_id}/task-card/"
        "generate-scenes"
    )
    async def generate_chapter_scene_beats(
        request: Request,
        project_id: str,
        chapter_id: str,
        instruction: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        task_card = planning_service.get_task_card(
            user_id=int(user["id"]),
            project_id=project_id,
            chapter_id=chapter_id,
        )
        if not task_card:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        if not task_card.get("plan_id"):
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}/task-card"
                "?error="
                + quote("请先保存章节任务卡，再让 Planner 只拆分场景"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        try:
            clean_instruction = _clean_field(
                instruction, "场景拆解额外要求", max_length=4000
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}/task-card"
                f"?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        profile = api_profile(int(user["id"]))
        if not profile:
            return RedirectResponse(
                "/settings/api?error="
                + quote("使用 Scene Planner 前，请先配置模型服务"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        try:
            job_id = database.create_chapter_planning_job(
                user_id=int(user["id"]),
                project_id=project_id,
                chapter_id=chapter_id,
                instruction=clean_instruction,
                provider=profile["provider"],
                model=profile["model"],
                credential_source=profile["credential_source"],
                max_jobs_per_day=app_settings.max_jobs_per_day,
                operation="plan_scene_beats",
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}/task-card"
                f"?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        request.app.state.worker.wake()
        return RedirectResponse(
            f"/writing-jobs/{job_id}", status_code=status.HTTP_303_SEE_OTHER
        )

    @application.post("/novels/{project_id}/chapters/{chapter_id}/plan")
    async def update_novel_chapter_plan(
        request: Request,
        project_id: str,
        chapter_id: str,
        title: str = Form(...),
        outline: str = Form(""),
        key_points: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            updated = database.update_novel_chapter_plan(
                user_id=int(user["id"]),
                project_id=project_id,
                chapter_id=chapter_id,
                title=_clean_field(
                    title, "章节名", max_length=120, required=True
                ),
                outline=_clean_field(
                    outline, "章节大纲", max_length=6000
                ),
                key_points=_clean_field(
                    key_points, "关键情节点", max_length=4000
                ),
            )
            if not updated:
                return Response(status_code=status.HTTP_404_NOT_FOUND)
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}"
                f"?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{project_id}/chapters/{chapter_id}?saved=true",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.get(
        "/novels/{project_id}/chapters/{chapter_id}"
        "/versions/{version_id}/compare",
        response_class=HTMLResponse,
    )
    async def compare_chapter_versions(
        request: Request,
        project_id: str,
        chapter_id: str,
        version_id: str,
        base_id: Optional[str] = None,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        user_id = int(user["id"])
        chapter = database.get_novel_chapter(
            user_id, project_id, chapter_id
        )
        target_version = database.get_chapter_version(
            user_id, project_id, chapter_id, version_id
        )
        if not chapter or not target_version:
            return render_template(
                "not_found.html",
                _template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )

        versions = database.list_chapter_versions(
            user_id, project_id, chapter_id
        )
        versions_by_id = {
            str(item["id"]): item for item in versions
        }
        requested_base_id = str(base_id or "").strip()
        if requested_base_id:
            base_version = database.get_chapter_version(
                user_id, project_id, chapter_id, requested_base_id
            )
            if not base_version or requested_base_id == version_id:
                return render_template(
                    "not_found.html",
                    _template_context(request, user=user),
                    status_code=status.HTTP_404_NOT_FOUND,
                )
        else:
            base_version = None
            preferred_ids = [
                str(target_version.get("parent_version_id") or ""),
                (
                    str(chapter.get("canonical_version_id") or "")
                    if str(chapter.get("canonical_version_id") or "")
                    != version_id
                    else ""
                ),
            ]
            for preferred_id in preferred_ids:
                if not preferred_id:
                    continue
                base_version = versions_by_id.get(preferred_id)
                if not base_version:
                    base_version = database.get_chapter_version(
                        user_id, project_id, chapter_id, preferred_id
                    )
                if base_version:
                    break
            if not base_version:
                for item in versions:
                    if str(item["id"]) != version_id:
                        base_version = item
                        break

        comparison = None
        comparison_error = ""
        if base_version:
            base_path = Path(str(base_version["content_path"]))
            target_path = Path(str(target_version["content_path"]))
            if not base_path.is_file() or not target_path.is_file():
                comparison_error = "正文版本文件不存在，暂时无法比较。"
            else:
                comparison = build_version_diff(
                    _read_optional_text(base_path),
                    _read_optional_text(target_path),
                )

        return render_template(
            "chapter_version_compare.html",
            _template_context(
                request,
                user=user,
                chapter=chapter,
                target_version=target_version,
                base_version=base_version,
                versions=versions,
                comparison=comparison,
                comparison_error=comparison_error,
            ),
        )

    @application.get(
        "/novels/{project_id}/chapters/{chapter_id}"
        "/versions/{version_id}/quality",
        response_class=HTMLResponse,
    )
    async def chapter_version_quality_page(
        request: Request,
        project_id: str,
        chapter_id: str,
        version_id: str,
        error: Optional[str] = None,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        user_id = int(user["id"])
        chapter = database.get_novel_chapter(
            user_id, project_id, chapter_id
        )
        version = database.get_chapter_version(
            user_id, project_id, chapter_id, version_id
        )
        if not chapter or not version:
            return render_template(
                "not_found.html",
                _template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        audit = database.get_latest_quality_audit(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            version_id=version_id,
        )
        chapter_workflow = workflow_service.get_state(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
        )
        return render_template(
            "chapter_quality.html",
            _template_context(
                request,
                user=user,
                chapter=chapter,
                version=version,
                audit=audit,
                report=(audit or {}).get("report") or {},
                chapter_workflow=chapter_workflow,
                error=error,
            ),
        )

    @application.post(
        "/novels/{project_id}/chapters/{chapter_id}"
        "/versions/{version_id}/quality"
    )
    async def run_chapter_version_quality_audit(
        request: Request,
        project_id: str,
        chapter_id: str,
        version_id: str,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        version = database.get_chapter_version(
            user_id, project_id, chapter_id, version_id
        )
        if not version:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        profile = api_profile(user_id)
        if not profile:
            return RedirectResponse(
                "/settings/api?error="
                + quote("执行正文硬审计前，请先配置模型服务"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        try:
            job_id = database.create_quality_audit_job(
                user_id=user_id,
                project_id=project_id,
                chapter_id=chapter_id,
                version_id=version_id,
                provider=profile["provider"],
                model=profile["model"],
                credential_source=profile["credential_source"],
                max_jobs_per_day=app_settings.max_jobs_per_day,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}"
                f"/versions/{version_id}/quality"
                f"?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        request.app.state.worker.wake()
        return RedirectResponse(
            f"/writing-jobs/{job_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.get(
        "/novels/{project_id}/chapters/{chapter_id}"
        "/versions/{version_id}/style",
        response_class=HTMLResponse,
    )
    async def chapter_version_style_page(
        request: Request,
        project_id: str,
        chapter_id: str,
        version_id: str,
        error: Optional[str] = None,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        user_id = int(user["id"])
        chapter = database.get_novel_chapter(
            user_id, project_id, chapter_id
        )
        version = database.get_chapter_version(
            user_id, project_id, chapter_id, version_id
        )
        if not chapter or not version:
            return render_template(
                "not_found.html",
                _template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        profile = style_service.get_voice_profile(
            user_id=user_id, project_id=project_id
        )
        audit = style_service.get_latest_audit(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            version_id=version_id,
        )
        chapter_workflow = workflow_service.get_state(
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
        )
        return render_template(
            "chapter_style.html",
            _template_context(
                request,
                user=user,
                chapter=chapter,
                version=version,
                voice_profile=profile,
                audit=audit,
                issues=(audit or {}).get("issues") or [],
                chapter_workflow=chapter_workflow,
                error=error,
            ),
        )

    @application.post(
        "/novels/{project_id}/chapters/{chapter_id}"
        "/versions/{version_id}/style"
    )
    async def run_chapter_version_style_audit(
        request: Request,
        project_id: str,
        chapter_id: str,
        version_id: str,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        profile = style_service.get_voice_profile(
            user_id=user_id, project_id=project_id
        )
        if not profile:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        if profile["status"] != "confirmed":
            return RedirectResponse(
                f"/novels/{project_id}?error="
                + quote("请先填写并确认作品声纹")
                + "#voice",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        api = api_profile(user_id)
        if not api:
            return RedirectResponse(
                "/settings/api?error="
                + quote("执行 AI 味审校前，请先配置模型服务"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        try:
            job_id = database.create_style_job(
                user_id=user_id,
                project_id=project_id,
                chapter_id=chapter_id,
                version_id=version_id,
                operation="audit_ai_style",
                provider=api["provider"],
                model=api["model"],
                credential_source=api["credential_source"],
                max_jobs_per_day=app_settings.max_jobs_per_day,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}"
                f"/versions/{version_id}/style?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        request.app.state.worker.wake()
        return RedirectResponse(
            f"/writing-jobs/{job_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/style-issues/{issue_id}/rewrite")
    async def request_style_issue_rewrite(
        request: Request,
        issue_id: str,
        instruction: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        issue = style_service.get_issue(user_id=user_id, issue_id=issue_id)
        if not issue:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        try:
            clean_instruction = _clean_field(
                instruction, "定点改写补充要求", max_length=2000
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/style-issues/{issue_id}?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        api = api_profile(user_id)
        if not api:
            return RedirectResponse(
                "/settings/api?error="
                + quote("生成定点改写前，请先配置模型服务"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        try:
            job_id = database.create_style_job(
                user_id=user_id,
                project_id=str(issue["project_id"]),
                chapter_id=str(issue["chapter_id"]),
                version_id=str(issue["version_id"]),
                operation="rewrite_style_issue",
                subject_id=issue_id,
                instruction=clean_instruction,
                provider=api["provider"],
                model=api["model"],
                credential_source=api["credential_source"],
                max_jobs_per_day=app_settings.max_jobs_per_day,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/style-issues/{issue_id}?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        request.app.state.worker.wake()
        return RedirectResponse(
            f"/writing-jobs/{job_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.get(
        "/style-issues/{issue_id}", response_class=HTMLResponse
    )
    async def style_issue_review(
        request: Request,
        issue_id: str,
        error: Optional[str] = None,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        issue = style_service.get_issue_review(
            user_id=int(user["id"]), issue_id=issue_id
        )
        if not issue:
            return render_template(
                "not_found.html",
                _template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        candidates = []
        for candidate in issue["candidates"]:
            candidate = dict(candidate)
            candidate["diff"] = _diff_segments(
                str(issue["quote"]), str(candidate["replacement_text"])
            )
            candidates.append(candidate)
        return render_template(
            "style_issue.html",
            _template_context(
                request,
                user=user,
                issue=issue,
                candidates=candidates,
                error=error,
            ),
        )

    @application.post("/style-issues/{issue_id}/ignore")
    async def ignore_style_issue(
        request: Request,
        issue_id: str,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        result = style_service.ignore_issue(
            user_id=int(user["id"]), issue_id=issue_id
        )
        if not result:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        return RedirectResponse(
            f"/novels/{result['project_id']}/chapters/"
            f"{result['chapter_id']}/versions/{result['version_id']}/style",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post(
        "/style-rewrite-candidates/{candidate_id}/accept"
    )
    async def accept_style_rewrite_candidate(
        request: Request,
        candidate_id: str,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        candidate = style_service.get_candidate(
            user_id=user_id, candidate_id=candidate_id
        )
        if not candidate:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        if (
            str(candidate["status"]) != "candidate"
            or str(candidate["issue_status"]) != "open"
        ):
            return RedirectResponse(
                f"/style-issues/{candidate['issue_id']}?error="
                + quote("这个改写候选已处理或失效"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        source_path = Path(str(candidate["source_content_path"]))
        working_path = Path(str(candidate["working_content_path"]))
        try:
            source_content, actual_hash = _read_utf8_with_file_hash(
                source_path
            )
        except (OSError, UnicodeError):
            return RedirectResponse(
                f"/style-issues/{candidate['issue_id']}?error="
                + quote("原版本正文文件无法读取"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        expected_hash = str(candidate["source_content_hash"] or "")
        if expected_hash and expected_hash != actual_hash:
            return RedirectResponse(
                f"/style-issues/{candidate['issue_id']}?error="
                + quote("原版本文件校验失败，请重新执行审校"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        start_offset = int(candidate["start_offset"])
        end_offset = int(candidate["end_offset"])
        if source_content[start_offset:end_offset] != str(candidate["quote"]):
            return RedirectResponse(
                f"/style-issues/{candidate['issue_id']}?error="
                + quote("问题定位与原文不一致，请重新执行审校"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        replacement = str(candidate["replacement_text"])
        revised_content = (
            source_content[:start_offset]
            + replacement
            + source_content[end_offset:]
        )
        if len(revised_content) > 200_000:
            return RedirectResponse(
                f"/style-issues/{candidate['issue_id']}?error="
                + quote("定点改写后的单章正文超过 200000 字"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        previous_working = _read_optional_text(working_path)
        token = secrets.token_hex(16)
        version_path = (
            source_path.parent / f"style-{token}.txt"
        )
        try:
            _atomic_write_text(version_path, revised_content, token)
            _atomic_write_text(working_path, revised_content, token)
            result = style_service.accept_rewrite_candidate(
                user_id=user_id,
                candidate_id=candidate_id,
                version_path=version_path,
                char_count=len(revised_content),
                effective_char_count=effective_char_count(revised_content),
                content_hash=hashlib.sha256(
                    revised_content.encode("utf-8")
                ).hexdigest(),
            )
            if not result:
                raise ValueError("改写候选已处理或失效")
        except ValueError as exc:
            try:
                _atomic_write_text(
                    working_path, previous_working, secrets.token_hex(16)
                )
            except Exception:
                logger.exception(
                    "failed to restore working chapter after rewrite rejection"
                )
            return RedirectResponse(
                f"/style-issues/{candidate['issue_id']}?error="
                + quote(str(exc)),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        except Exception:
            try:
                _atomic_write_text(
                    working_path, previous_working, secrets.token_hex(16)
                )
            except Exception:
                logger.exception(
                    "failed to restore working chapter after rewrite failure"
                )
            logger.exception("failed to accept targeted style rewrite")
            return RedirectResponse(
                f"/style-issues/{candidate['issue_id']}?error="
                + quote("接受定点改写失败"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{result['project_id']}/chapters/"
            f"{result['chapter_id']}?saved=true",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.get(
        "/canon-impact-reports/{report_id}",
        response_class=HTMLResponse,
    )
    async def canon_impact_report_page(
        request: Request,
        report_id: str,
        error: Optional[str] = None,
        saved: bool = False,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        report = impact_service.get_report(
            user_id=int(user["id"]), report_id=report_id
        )
        if not report:
            return render_template(
                "not_found.html",
                _template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        return render_template(
            "canon_impact_report.html",
            _template_context(
                request,
                user=user,
                report=report,
                error=error,
                saved=saved,
            ),
        )

    @application.post("/canon-impact-reports/{report_id}")
    async def decide_canon_impact_report(
        request: Request,
        report_id: str,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        form = await request.form()
        verify_csrf(request, str(form.get("csrf") or ""))
        report = impact_service.get_report(
            user_id=int(user["id"]), report_id=report_id
        )
        if not report:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        if str(report["status"]) != "pending":
            return RedirectResponse(
                f"/canon-impact-reports/{report_id}?error="
                + quote("这份影响报告已经处理"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        decisions = {
            str(item["id"]): {
                "decision": str(
                    form.get(f"decision_{item['id']}") or "recheck"
                ),
                "note": str(form.get(f"note_{item['id']}") or ""),
            }
            for item in report["items"]
        }
        try:
            updated = impact_service.update_decisions(
                user_id=int(user["id"]),
                report_id=report_id,
                decisions=decisions,
            )
            if not updated:
                return Response(status_code=status.HTTP_404_NOT_FOUND)
        except ValueError as exc:
            return RedirectResponse(
                f"/canon-impact-reports/{report_id}?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        if str(form.get("action") or "save") != "confirm":
            return RedirectResponse(
                f"/canon-impact-reports/{report_id}?saved=true",
                status_code=status.HTTP_303_SEE_OTHER,
            )

        report = impact_service.get_report(
            user_id=int(user["id"]), report_id=report_id
        )
        if not report:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        if (
            str(report["current_canonical_version_id"] or "")
            != str(report["old_version_id"])
        ):
            impact_service.mark_stale(
                user_id=int(user["id"]), report_id=report_id
            )
            return RedirectResponse(
                f"/canon-impact-reports/{report_id}?error="
                + quote("正史版本已经变化，这份报告已失效"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        version_path = Path(str(report["proposed_content_path"]))
        content_path = Path(str(report["working_content_path"]))
        try:
            candidate_content, actual_hash = _read_utf8_with_file_hash(
                version_path
            )
        except (OSError, UnicodeError):
            return RedirectResponse(
                f"/canon-impact-reports/{report_id}?error="
                + quote("候选版本正文文件无法读取"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        expected_hash = str(report["proposed_content_hash"] or "")
        if expected_hash and expected_hash != actual_hash:
            impact_service.mark_stale(
                user_id=int(user["id"]), report_id=report_id
            )
            return RedirectResponse(
                f"/canon-impact-reports/{report_id}?error="
                + quote("候选版本文件校验失败，未切换正史"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        previous_content = _read_optional_text(content_path)
        try:
            _atomic_write_text(
                content_path, candidate_content, secrets.token_hex(16)
            )
            accepted = database.accept_chapter_version(
                user_id=int(user["id"]),
                project_id=str(report["project_id"]),
                chapter_id=str(report["chapter_id"]),
                version_id=str(report["proposed_version_id"]),
                override_reason=str(report["override_reason"] or ""),
                expected_old_canonical_version_id=str(
                    report["old_version_id"]
                ),
            )
            if not accepted:
                raise ValueError("章节版本不存在")
        except ValueError as exc:
            try:
                _atomic_write_text(
                    content_path,
                    previous_content,
                    secrets.token_hex(16),
                )
            except Exception:
                logger.exception(
                    "failed to restore content after impact confirmation"
                )
            impact_service.mark_stale(
                user_id=int(user["id"]), report_id=report_id
            )
            return RedirectResponse(
                f"/canon-impact-reports/{report_id}?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        except Exception:
            try:
                _atomic_write_text(
                    content_path,
                    previous_content,
                    secrets.token_hex(16),
                )
            except Exception:
                logger.exception(
                    "failed to restore content after impact failure"
                )
            logger.exception("failed to confirm canon impact report")
            return RedirectResponse(
                f"/canon-impact-reports/{report_id}?error="
                + quote("切换正史版本失败"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        if not impact_service.mark_applied(
            user_id=int(user["id"]), report_id=report_id
        ):
            logger.error(
                "canon changed but impact report was not marked applied id=%s",
                report_id,
            )
        return await after_canon_acceptance(
            request,
            user_id=int(user["id"]),
            project_id=str(report["project_id"]),
            chapter_id=str(report["chapter_id"]),
            version_id=str(report["proposed_version_id"]),
        )

    @application.post("/canon-impact-reports/{report_id}/cancel")
    async def cancel_canon_impact_report(
        request: Request,
        report_id: str,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        result = impact_service.cancel_report(
            user_id=int(user["id"]), report_id=report_id
        )
        if not result:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        return RedirectResponse(
            f"/novels/{result['project_id']}/chapters/"
            f"{result['chapter_id']}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post(
        "/novels/{project_id}/chapters/{chapter_id}"
        "/versions/{version_id}/accept"
    )
    async def accept_novel_chapter_version(
        request: Request,
        project_id: str,
        chapter_id: str,
        version_id: str,
        csrf: str = Form(...),
        override_reason: str = Form(""),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        chapter = database.get_novel_chapter(
            user_id, project_id, chapter_id
        )
        version = database.get_chapter_version(
            user_id, project_id, chapter_id, version_id
        )
        if not chapter or not version:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        current_canonical = str(
            chapter.get("canonical_version_id") or ""
        )
        if current_canonical and current_canonical != version_id:
            try:
                report_id = impact_service.prepare_report(
                    user_id=user_id,
                    project_id=project_id,
                    chapter_id=chapter_id,
                    proposed_version_id=version_id,
                    override_reason=override_reason,
                )
            except ValueError as exc:
                return RedirectResponse(
                    f"/novels/{project_id}/chapters/{chapter_id}"
                    f"?error={quote(str(exc))}",
                    status_code=status.HTTP_303_SEE_OTHER,
                )
            return RedirectResponse(
                f"/canon-impact-reports/{report_id}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        version_path = Path(str(version["content_path"]))
        content_path = Path(str(chapter["content_path"]))
        try:
            candidate_content, actual_hash = _read_utf8_with_file_hash(
                version_path
            )
        except (OSError, UnicodeError):
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}"
                f"?error={quote('这个版本的正文文件无法读取')}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        expected_hash = str(version["content_hash"] or "")
        if expected_hash and expected_hash != actual_hash:
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}"
                f"?error={quote('版本文件校验失败，未切换正史')}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        previous_content = _read_optional_text(content_path)
        write_token = secrets.token_hex(16)
        try:
            _atomic_write_text(content_path, candidate_content, write_token)
            accepted = database.accept_chapter_version(
                user_id=user_id,
                project_id=project_id,
                chapter_id=chapter_id,
                version_id=version_id,
                override_reason=override_reason,
            )
            if not accepted:
                raise ValueError("章节版本不存在")
        except ValueError as exc:
            try:
                _atomic_write_text(
                    content_path, previous_content, secrets.token_hex(16)
                )
            except Exception:
                logger.exception(
                    "failed to restore chapter content after canon rejection"
                )
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}"
                f"?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        except Exception:
            try:
                _atomic_write_text(
                    content_path, previous_content, secrets.token_hex(16)
                )
            except Exception:
                logger.exception(
                    "failed to restore chapter content after canon failure"
                )
            logger.exception("failed to accept canonical chapter version")
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}"
                f"?error={quote('确认正史版本失败')}",
                status_code=status.HTTP_303_SEE_OTHER,
            )

        return await after_canon_acceptance(
            request,
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            version_id=version_id,
        )

    @application.post(
        "/novels/{project_id}/chapters/{chapter_id}"
        "/versions/{version_id}/extract-memory"
    )
    async def extract_canonical_chapter_memory(
        request: Request,
        project_id: str,
        chapter_id: str,
        version_id: str,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        chapter = database.get_novel_chapter(
            user_id, project_id, chapter_id
        )
        version = database.get_chapter_version(
            user_id, project_id, chapter_id, version_id
        )
        if not chapter or not version:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        if str(chapter.get("canonical_version_id") or "") != version_id:
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}"
                "?error="
                + quote("只能从当前正史版本提取故事记忆")
                + "#chapter-workflow",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        profile = api_profile(user_id)
        if not profile:
            return RedirectResponse(
                "/settings/api?error="
                + quote("提取故事记忆前，请先配置模型服务"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        try:
            job_id = database.create_memory_extraction_job(
                user_id=user_id,
                project_id=project_id,
                chapter_id=chapter_id,
                version_id=version_id,
                provider=profile["provider"],
                model=profile["model"],
                credential_source=profile["credential_source"],
                max_jobs_per_day=app_settings.max_jobs_per_day,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}"
                f"?error={quote(str(exc))}#chapter-workflow",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        request.app.state.worker.wake()
        return RedirectResponse(
            f"/writing-jobs/{job_id}", status_code=status.HTTP_303_SEE_OTHER
        )

    @application.post("/novels/{project_id}/chapters/{chapter_id}/save")
    async def save_novel_chapter(
        request: Request,
        project_id: str,
        chapter_id: str,
        content: str = Form(""),
        change_summary: str = Form(""),
        return_to_workbench: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        chapter = database.get_novel_chapter(
            int(user["id"]), project_id, chapter_id
        )
        if not chapter:
            return Response(status_code=status.HTTP_404_NOT_FOUND)

        def save_error_redirect(message: str) -> RedirectResponse:
            destination = (
                f"/novels/{project_id}/workbench"
                f"?chapter_id={chapter_id}&error={quote(message)}"
                if return_to_workbench
                else (
                    f"/novels/{project_id}/chapters/{chapter_id}"
                    f"?error={quote(message)}"
                )
            )
            return RedirectResponse(
                destination,
                status_code=status.HTTP_303_SEE_OTHER,
            )

        if len(content) > 200_000:
            return save_error_redirect("单章正文不能超过 200000 字")
        try:
            clean_change_summary = _clean_field(
                change_summary,
                "这次主要修改了什么",
                max_length=1000,
            )
        except ValueError as exc:
            return save_error_redirect(str(exc))
        if database.chapter_has_active_generation(
            int(user["id"]), project_id, chapter_id
        ):
            return save_error_redirect("AI 正在生成本章，请完成后再保存")
        content_path = Path(str(chapter["content_path"]))
        version_token = secrets.token_hex(16)
        version_path = (
            content_path.parent / "versions" / f"manual-{version_token}.txt"
        )
        try:
            _atomic_write_text(version_path, content, version_token)
            _atomic_write_text(content_path, content, version_token)
            version_id = database.record_manual_chapter_version(
                user_id=int(user["id"]),
                project_id=project_id,
                chapter_id=chapter_id,
                version_path=version_path,
                char_count=len(content),
                effective_char_count=effective_char_count(content),
                content_hash=hashlib.sha256(
                    content.encode("utf-8")
                ).hexdigest(),
                change_summary=clean_change_summary,
            )
            if not version_id:
                raise ValueError("章节不存在")
        except ValueError as exc:
            return save_error_redirect(str(exc))
        except Exception:
            logger.exception("failed to save novel chapter")
            return save_error_redirect("保存正文失败")
        destination = (
            f"/novels/{project_id}/workbench"
            f"?chapter_id={chapter_id}&saved=true"
            if return_to_workbench
            else (
                f"/novels/{project_id}/chapters/"
                f"{chapter_id}?saved=true"
            )
        )
        return RedirectResponse(
            destination, status_code=status.HTTP_303_SEE_OTHER
        )

    @application.post("/novels/{project_id}/chapters/{chapter_id}/generate")
    async def generate_novel_chapter(
        request: Request,
        project_id: str,
        chapter_id: str,
        operation: str = Form("draft"),
        instruction: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        chapter = database.get_novel_chapter(
            int(user["id"]), project_id, chapter_id
        )
        if not chapter:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        task_card = planning_service.get_task_card(
            user_id=int(user["id"]),
            project_id=project_id,
            chapter_id=chapter_id,
        )
        if not task_card or task_card["status"] != "confirmed":
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}/task-card"
                "?error="
                + quote("先确认章节任务卡和至少两个场景节拍，再生成正文"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        current_content = _read_optional_text(Path(str(chapter["content_path"])))
        if operation in {"continue", "rewrite", "polish"} and not current_content.strip():
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}"
                f"?error={quote('本章还没有正文，请先生成初稿')}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        if operation == "draft" and current_content.strip():
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}"
                f"?error={quote('本章已有正文，请选择重写或继续写')}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        try:
            clean_instruction = _clean_field(
                instruction, "额外要求", max_length=4000
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}"
                f"?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        profile = api_profile(int(user["id"]))
        if not profile:
            return RedirectResponse(
                "/settings/api?error="
                + quote("开始写作前，请先配置模型服务"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        try:
            job_id = database.create_generation_job(
                user_id=int(user["id"]),
                project_id=project_id,
                chapter_id=chapter_id,
                operation=operation,
                instruction=clean_instruction,
                provider=profile["provider"],
                model=profile["model"],
                credential_source=profile["credential_source"],
                max_jobs_per_day=app_settings.max_jobs_per_day,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}/chapters/{chapter_id}"
                f"?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        request.app.state.worker.wake()
        return RedirectResponse(
            f"/writing-jobs/{job_id}", status_code=status.HTTP_303_SEE_OTHER
        )

    @application.get("/writing-jobs/{job_id}", response_class=HTMLResponse)
    async def writing_job_page(request: Request, job_id: str):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        job = database.get_generation_job(int(user["id"]), job_id)
        if not job:
            return render_template(
                "not_found.html",
                _template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        memory_delta_id = None
        memory_retrieval = None
        if (
            str(job["operation"]) == "extract_story_delta"
            and str(job["status"]) == "completed"
            and job.get("result_json")
        ):
            try:
                memory_delta_id = json.loads(
                    str(job["result_json"])
                ).get("delta_id")
            except (AttributeError, TypeError, ValueError):
                logger.warning(
                    "memory extraction job has invalid result_json id=%s",
                    job_id,
                )
        if job.get("context_snapshot_json"):
            try:
                snapshot = json.loads(str(job["context_snapshot_json"]))
                canonical_memory = dict(
                    snapshot.get("canonical_memory") or {}
                )
                retrieval = dict(canonical_memory.get("retrieval") or {})
                if retrieval.get("engine") not in {None, "", "not_available"}:
                    memory_retrieval = {
                        "engine": str(retrieval.get("engine")),
                        "scope": str(retrieval.get("scope") or ""),
                        "query_terms": [
                            str(term)
                            for term in (
                                retrieval.get("query_concepts")
                                or retrieval.get("query_terms")
                                or []
                            )[:24]
                        ],
                        "expanded_term_count": len(
                            retrieval.get("query_terms") or []
                        ),
                        "matched_count": int(
                            retrieval.get("matched_count") or 0
                        ),
                        "items": [
                            dict(item)
                            for item in (
                                canonical_memory.get(
                                    "retrieved_memory"
                                )
                                or []
                            )[:8]
                        ],
                    }
            except (AttributeError, TypeError, ValueError):
                logger.warning(
                    "writing job has invalid context snapshot id=%s",
                    job_id,
                )
        return render_template(
            "writing_job.html",
            _template_context(
                request,
                user=user,
                job=job,
                memory_delta_id=memory_delta_id,
                memory_retrieval=memory_retrieval,
            ),
        )

    @application.get("/api/writing-jobs/{job_id}")
    async def writing_job_status(request: Request, job_id: str):
        user = _current_user(request)
        if not user:
            return JSONResponse(
                {"detail": "未登录"}, status_code=status.HTTP_401_UNAUTHORIZED
            )
        job = database.get_generation_job(int(user["id"]), job_id)
        if not job:
            return JSONResponse(
                {"detail": "任务不存在"}, status_code=status.HTTP_404_NOT_FOUND
            )
        redirect_url = None
        if job["status"] == "completed":
            if str(job["operation"]) in {
                "plan_chapter",
                "plan_scene_beats",
            }:
                redirect_url = (
                    f"/novels/{job['project_id']}/chapters/"
                    f"{job['chapter_id']}/task-card"
                    + (
                        "#scene-beats"
                        if str(job["operation"]) == "plan_scene_beats"
                        else ""
                    )
                )
            elif (
                str(job["operation"]) == "propose_reader_branches"
                and job.get("subject_id")
            ):
                redirect_url = f"/reader-requests/{job['subject_id']}"
            elif (
                str(job["operation"]) == "audit_chapter"
                and job.get("version_id")
            ):
                redirect_url = (
                    f"/novels/{job['project_id']}/chapters/"
                    f"{job['chapter_id']}/versions/{job['version_id']}/quality"
                )
            elif (
                str(job["operation"]) == "audit_ai_style"
                and job.get("version_id")
            ):
                redirect_url = (
                    f"/novels/{job['project_id']}/chapters/"
                    f"{job['chapter_id']}/versions/{job['version_id']}/style"
                )
            elif (
                str(job["operation"]) == "rewrite_style_issue"
                and job.get("subject_id")
            ):
                redirect_url = f"/style-issues/{job['subject_id']}"
            elif str(job["operation"]) in {
                "generate_scene",
                "rewrite_scene",
                "audit_scene",
            }:
                redirect_url = (
                    f"/novels/{job['project_id']}/chapters/"
                    f"{job['chapter_id']}/scenes"
                )
            elif (
                str(job["operation"]) == "extract_story_delta"
                and job.get("result_json")
            ):
                try:
                    delta_id = json.loads(str(job["result_json"])).get(
                        "delta_id"
                    )
                    if delta_id:
                        redirect_url = f"/story-deltas/{delta_id}"
                except (AttributeError, TypeError, ValueError):
                    pass
            if not redirect_url:
                redirect_url = (
                    f"/novels/{job['project_id']}/chapters/"
                    f"{job['chapter_id']}"
                )
        return {
            "id": job_id,
            "status": job["status"],
            "terminal": job["status"] in {"completed", "failed"},
            "redirect_url": redirect_url,
            "error": job["error"] if job["status"] == "failed" else None,
        }

    @application.get(
        "/story-deltas/{delta_id}", response_class=HTMLResponse
    )
    async def story_delta_review(
        request: Request,
        delta_id: str,
        error: Optional[str] = None,
        saved: bool = False,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        delta = memory_service.get_delta(
            user_id=int(user["id"]), delta_id=delta_id
        )
        if not delta:
            return render_template(
                "not_found.html",
                _template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        return render_template(
            "story_delta.html",
            _template_context(
                request,
                user=user,
                delta=delta,
                payload_json=json.dumps(
                    delta["payload"], ensure_ascii=False, indent=2
                ),
                error=error,
                saved=saved,
            ),
        )

    @application.post("/story-deltas/{delta_id}/save")
    async def save_story_delta(
        request: Request,
        delta_id: str,
        payload_json: str = Form(...),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        if len(payload_json) > 200_000:
            return RedirectResponse(
                f"/story-deltas/{delta_id}?error="
                + quote("Story Delta 不能超过 200000 个字符"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        try:
            payload = json.loads(payload_json)
            delta = StoryDelta.model_validate(payload)
        except (TypeError, ValueError) as exc:
            return RedirectResponse(
                f"/story-deltas/{delta_id}?error="
                + quote(f"结构校验失败：{str(exc)[:800]}"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        updated = memory_service.update_proposal(
            user_id=int(user["id"]),
            delta_id=delta_id,
            payload=delta,
        )
        if not updated:
            return RedirectResponse(
                f"/story-deltas/{delta_id}?error="
                + quote("这份提案已确认、拒绝或不存在，不能继续编辑"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/story-deltas/{delta_id}?saved=true",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/story-deltas/{delta_id}/accept")
    async def accept_story_delta(
        request: Request,
        delta_id: str,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        delta = memory_service.get_delta(
            user_id=int(user["id"]), delta_id=delta_id
        )
        if not delta:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        try:
            projected = memory_service.accept_delta(
                user_id=int(user["id"]), delta_id=delta_id
            )
            if not projected:
                return Response(status_code=status.HTTP_404_NOT_FOUND)
        except ValueError as exc:
            return RedirectResponse(
                f"/story-deltas/{delta_id}?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{delta['project_id']}/chapters/"
            f"{delta['chapter_id']}?memory_saved=true",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/story-deltas/{delta_id}/reject")
    async def reject_story_delta(
        request: Request,
        delta_id: str,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        delta = memory_service.get_delta(
            user_id=int(user["id"]), delta_id=delta_id
        )
        if not delta:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        rejected = memory_service.reject_delta(
            user_id=int(user["id"]), delta_id=delta_id
        )
        if not rejected:
            return RedirectResponse(
                f"/story-deltas/{delta_id}?error="
                + quote("这份提案已不能拒绝"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{delta['project_id']}/chapters/"
            f"{delta['chapter_id']}?canonical=true",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.get("/projects/import", response_class=HTMLResponse)
    async def project_import_page(request: Request):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        return render_template(
            "project_import.html",
            _template_context(
                request,
                user=user,
                max_archive_mb=(
                    app_settings.max_project_archive_bytes
                    // 1024
                    // 1024
                ),
            ),
        )

    @application.post("/projects/import", response_class=HTMLResponse)
    async def import_novel_project(
        request: Request,
        archive_file: UploadFile = File(...),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        temporary = tempfile.NamedTemporaryFile(
            prefix="novelai-project-upload-",
            suffix=".zip",
            delete=False,
        )
        temporary_path = Path(temporary.name)
        total_bytes = 0
        try:
            while True:
                chunk = await archive_file.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > app_settings.max_project_archive_bytes:
                    raise ProjectArchiveError(
                        "作品归档文件超过允许大小"
                    )
                temporary.write(chunk)
            temporary.close()
            if total_bytes == 0:
                raise ProjectArchiveError("请选择非空的作品归档")
            imported = import_project_archive(
                database=database,
                novels_dir=app_settings.novels_dir,
                user_id=int(user["id"]),
                archive_path=temporary_path,
                max_uncompressed_bytes=(
                    app_settings.max_project_archive_bytes
                ),
            )
        except (ProjectArchiveError, OSError, sqlite3.Error) as exc:
            logger.warning("project archive import rejected: %s", exc)
            return render_template(
                "project_import.html",
                _template_context(
                    request,
                    user=user,
                    error=str(exc),
                    max_archive_mb=(
                        app_settings.max_project_archive_bytes
                        // 1024
                        // 1024
                    ),
                ),
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        finally:
            temporary.close()
            temporary_path.unlink(missing_ok=True)
            await archive_file.close()
        return RedirectResponse(
            f"/dashboard?imported=true&project={imported.project_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.get("/novels/{project_id}/export.novelai.zip")
    async def export_novel_project_archive(
        request: Request, project_id: str
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        project = database.get_novel_project(int(user["id"]), project_id)
        if not project:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        file_descriptor, raw_path = tempfile.mkstemp(
            prefix=f"novelai-project-{project_id[:8]}-",
            suffix=".novelai.zip",
        )
        os.close(file_descriptor)
        archive_path = Path(raw_path)
        try:
            create_project_archive(
                database=database,
                novels_dir=app_settings.novels_dir,
                user_id=int(user["id"]),
                project_id=project_id,
                destination=archive_path,
                max_uncompressed_bytes=(
                    app_settings.max_project_archive_bytes
                ),
            )
        except (ProjectArchiveError, OSError, sqlite3.Error) as exc:
            archive_path.unlink(missing_ok=True)
            return RedirectResponse(
                f"/novels/{project_id}/workbench?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return FileResponse(
            archive_path,
            media_type="application/zip",
            filename=f"novel-{project_id[:8]}.novelai.zip",
            headers={"Cache-Control": "no-store"},
            background=BackgroundTask(
                archive_path.unlink, missing_ok=True
            ),
        )

    @application.get("/novels/{project_id}/export.txt")
    async def export_novel(request: Request, project_id: str):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        project = database.get_novel_project(int(user["id"]), project_id)
        if not project:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        chapters = database.list_novel_chapters(int(user["id"]), project_id)
        parts = [str(project["title"])]
        for chapter in chapters:
            content = _read_optional_text(Path(str(chapter["content_path"])))
            parts.extend(["", str(chapter["title"]), "", content])
        payload = "\n".join(parts).strip() + "\n"
        return Response(
            payload,
            media_type="text/plain; charset=utf-8",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="novel-{project_id[:8]}.txt"'
                )
            },
        )

    @application.get("/novels/{project_id}/assistant")
    async def novel_assistant_page(
        request: Request,
        project_id: str,
        conversation_id: Optional[str] = None,
        chapter_id: Optional[str] = None,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        user_id = int(user["id"])
        project = database.get_novel_project(user_id, project_id)
        if not project:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        if conversation_id:
            conversation = assistant_chat_service.get_conversation(
                user_id=user_id,
                conversation_id=conversation_id,
            )
            if (
                not conversation
                or str(conversation.get("project_id") or "")
                != project_id
            ):
                return Response(status_code=status.HTTP_404_NOT_FOUND)
            if conversation.get("novel_chapter_id"):
                chapter_id = str(
                    conversation["novel_chapter_id"]
                )
        query = []
        if conversation_id:
            query.append(
                "conversation_id=" + quote(conversation_id)
            )
        if chapter_id:
            query.append("chapter_id=" + quote(chapter_id))
        else:
            query.append("view=settings")
        destination = f"/novels/{project_id}/workbench"
        if query:
            destination += "?" + "&".join(query)
        return RedirectResponse(
            destination, status_code=status.HTTP_303_SEE_OTHER
        )

    @application.post("/novels/{project_id}/assistant/new")
    async def new_novel_assistant_conversation(
        request: Request,
        project_id: str,
        chapter_id: str = Form(""),
        settings_tab: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        project = database.get_novel_project(user_id, project_id)
        if not project:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        clean_chapter_id = chapter_id.strip()
        try:
            conversation_id = assistant_chat_service.create_conversation(
                user_id=user_id,
                scope_type="chapter" if clean_chapter_id else "project",
                title="新对话",
                project_id=project_id,
                novel_chapter_id=clean_chapter_id or None,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/novels/{project_id}/workbench"
                f"?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        query = f"?conversation_id={quote(conversation_id)}"
        if clean_chapter_id:
            query += f"&chapter_id={quote(clean_chapter_id)}"
        else:
            clean_settings_tab = (
                settings_tab
                if settings_tab in WORKBENCH_SETTING_TAB_KEYS
                else "core"
            )
            query += (
                "&view=settings"
                f"&settings_tab={clean_settings_tab}"
            )
        return RedirectResponse(
            f"/novels/{project_id}/workbench" + query,
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post(
        "/novels/{project_id}/assistant/conversations/"
        "{conversation_id}/delete"
    )
    async def delete_novel_assistant_conversation(
        request: Request,
        project_id: str,
        conversation_id: str,
        current_conversation_id: str = Form(""),
        chapter_id: str = Form(""),
        return_view: str = Form(""),
        settings_tab: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        target = assistant_chat_service.get_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if (
            not target
            or str(target.get("project_id") or "") != project_id
        ):
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        clean_settings_tab = (
            settings_tab
            if settings_tab in WORKBENCH_SETTING_TAB_KEYS
            else "core"
        )
        destination = f"/novels/{project_id}/workbench"
        try:
            deleted = assistant_chat_service.delete_conversation(
                user_id=user_id,
                conversation_id=conversation_id,
            )
            if not deleted:
                return Response(status_code=status.HTTP_404_NOT_FOUND)
        except ValueError as exc:
            return RedirectResponse(
                destination
                + "?conversation_id="
                + quote(conversation_id)
                + "&error="
                + quote(str(exc)),
                status_code=status.HTTP_303_SEE_OTHER,
            )

        return_conversation = None
        if (
            current_conversation_id
            and current_conversation_id != conversation_id
        ):
            candidate = assistant_chat_service.get_conversation(
                user_id=user_id,
                conversation_id=current_conversation_id,
            )
            if (
                candidate
                and str(candidate.get("project_id") or "") == project_id
            ):
                return_conversation = candidate
        query: list[str] = []
        if return_conversation:
            query.append(
                "conversation_id="
                + quote(str(return_conversation["id"]))
            )
            if return_conversation.get("novel_chapter_id"):
                query.append(
                    "chapter_id="
                    + quote(
                        str(return_conversation["novel_chapter_id"])
                    )
                )
            else:
                query.extend(
                    [
                        "view=settings",
                        f"settings_tab={clean_settings_tab}",
                    ]
                )
        elif return_view == "settings":
            query.extend(
                [
                    "view=settings",
                    f"settings_tab={clean_settings_tab}",
                ]
            )
        elif chapter_id:
            query.append("chapter_id=" + quote(chapter_id))
        else:
            query.append("view=settings")
        return RedirectResponse(
            destination + ("?" + "&".join(query) if query else ""),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/novels/{project_id}/assistant/messages")
    async def send_novel_assistant_message(
        request: Request,
        project_id: str,
        question: str = Form(...),
        conversation_id: str = Form(""),
        novel_chapter_id: str = Form(""),
        source_type: str = Form(""),
        source_version_id: str = Form(""),
        quote_start: str = Form(""),
        quote_end: str = Form(""),
        quote_text: str = Form(""),
        source_hash: str = Form(""),
        return_view: str = Form(""),
        return_settings_tab: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        project = database.get_novel_project(user_id, project_id)
        if not project:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        return_path = f"/novels/{project_id}/workbench"
        clean_return_settings_tab = (
            return_settings_tab
            if return_settings_tab in WORKBENCH_SETTING_TAB_KEYS
            else "core"
        )
        try:
            if conversation_id:
                conversation = assistant_chat_service.get_conversation(
                    user_id=user_id,
                    conversation_id=conversation_id,
                )
                if (
                    not conversation
                    or str(conversation.get("project_id") or "")
                    != project_id
                ):
                    raise ValueError("对话不存在")
            else:
                scope_type = (
                    "chapter" if novel_chapter_id else "project"
                )
                conversation_id = (
                    assistant_chat_service.create_conversation(
                        user_id=user_id,
                        scope_type=scope_type,
                        title=question,
                        project_id=project_id,
                        novel_chapter_id=(
                            novel_chapter_id or None
                        ),
                    )
                )
            profile = api_profile(user_id)
            if not profile:
                return RedirectResponse(
                    "/settings/api?error="
                    + quote(
                        "开始创作对话前，请先配置模型服务"
                    ),
                    status_code=status.HTTP_303_SEE_OTHER,
                )
            quote_payload = None
            if quote_text:
                quote_payload = {
                    "source_type": source_type,
                    "project_id": project_id,
                    "novel_chapter_id": novel_chapter_id,
                    "version_id": source_version_id,
                    "start_offset": quote_start,
                    "end_offset": quote_end,
                    "quote_text": quote_text,
                    "content_hash": source_hash,
                }
            assistant_chat_service.queue_message(
                user_id=user_id,
                conversation_id=conversation_id,
                question=question,
                provider=profile["provider"],
                model=profile["model"],
                credential_source=profile["credential_source"],
                quote=quote_payload,
                agent_role="auto",
                ui_surface=(
                    "settings"
                    if return_view == "settings"
                    else (
                        "chapter"
                        if novel_chapter_id
                        else "project"
                    )
                ),
                auto_commit=True,
                max_jobs_per_day=app_settings.max_jobs_per_day,
            )
            prepared_conversation = (
                assistant_chat_service.get_conversation(
                    user_id=user_id,
                    conversation_id=conversation_id,
                )
            )
            prepared_chapter_id = str(
                (prepared_conversation or {}).get(
                    "novel_chapter_id"
                )
                or ""
            )
            if not novel_chapter_id and prepared_chapter_id:
                novel_chapter_id = prepared_chapter_id
                return_view = "body"
        except ValueError as exc:
            suffix = (
                f"?conversation_id={quote(conversation_id)}"
                if conversation_id
                else ""
            )
            if novel_chapter_id:
                suffix += (
                    ("&" if suffix else "?")
                    + f"chapter_id={quote(novel_chapter_id)}"
                )
            if return_view == "settings":
                suffix += (
                    ("&" if suffix else "?")
                    + "view=settings"
                    + f"&settings_tab={clean_return_settings_tab}"
                )
            separator = "&" if suffix else "?"
            return RedirectResponse(
                return_path
                + suffix
                + separator
                + "error="
                + quote(str(exc)),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        request.app.state.worker.wake()
        query = (
            f"?conversation_id={quote(conversation_id)}&sent=true"
        )
        if novel_chapter_id:
            query += f"&chapter_id={quote(novel_chapter_id)}"
        if return_view == "settings":
            query += (
                "&view=settings"
                f"&settings_tab={clean_return_settings_tab}"
            )
        return RedirectResponse(
            return_path + query,
            status_code=status.HTTP_303_SEE_OTHER,
        )

    def _novel_branch_destination(
        branch: Mapping[str, Any],
        *,
        settings_tab: str,
        sent: bool = False,
    ) -> str:
        destination = (
            f"/novels/{branch['project_id']}/workbench"
            f"?conversation_id={quote(str(branch['conversation_id']))}"
        )
        if sent:
            destination += "&sent=true"
        if branch.get("novel_chapter_id"):
            destination += (
                "&chapter_id="
                + quote(str(branch["novel_chapter_id"]))
            )
        else:
            clean_settings_tab = (
                settings_tab
                if settings_tab in WORKBENCH_SETTING_TAB_KEYS
                else "core"
            )
            destination += (
                "&view=settings"
                f"&settings_tab={clean_settings_tab}"
            )
        return destination

    def _original_novel_message_destination(
        conversation: Mapping[str, Any],
        *,
        settings_tab: str,
        error: str = "",
    ) -> str:
        branch = {
            "project_id": conversation["project_id"],
            "conversation_id": conversation["id"],
            "novel_chapter_id": conversation.get(
                "novel_chapter_id"
            ),
        }
        destination = _novel_branch_destination(
            branch, settings_tab=settings_tab
        )
        if error:
            destination += "&error=" + quote(error)
        return destination

    def _queue_novel_message_branch(
        *,
        user_id: int,
        message_id: str,
        replacement_question: Optional[str],
        profile: Mapping[str, Any],
        settings_tab: str,
    ) -> Dict[str, Any]:
        branch = None
        try:
            branch = (
                assistant_chat_service.branch_conversation_from_message(
                    user_id=user_id,
                    message_id=message_id,
                    replacement_question=replacement_question,
                )
            )
            if not branch.get("project_id"):
                raise ValueError("这条消息不属于小说创作对话")
            assistant_chat_service.queue_message(
                user_id=user_id,
                conversation_id=str(branch["conversation_id"]),
                question=str(branch["question"]),
                provider=str(profile["provider"]),
                model=str(profile["model"]),
                credential_source=str(
                    profile["credential_source"]
                ),
                quote=branch.get("quote"),
                agent_role="auto",
                ui_surface=(
                    "chapter"
                    if branch.get("novel_chapter_id")
                    else "settings"
                ),
                auto_commit=True,
                max_jobs_per_day=app_settings.max_jobs_per_day,
            )
        except Exception:
            if branch and branch.get("conversation_id"):
                try:
                    assistant_chat_service.delete_conversation(
                        user_id=user_id,
                        conversation_id=str(
                            branch["conversation_id"]
                        ),
                    )
                except (TypeError, ValueError):
                    logger.exception(
                        "failed to clean up assistant branch"
                    )
            raise
        branch["destination"] = _novel_branch_destination(
            branch, settings_tab=settings_tab, sent=True
        )
        return branch

    @application.post("/assistant/messages/{message_id}/branch")
    async def branch_novel_assistant_message(
        request: Request,
        message_id: str,
        question: str = Form(...),
        settings_tab: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        source = assistant_chat_service.get_message(
            user_id=user_id, message_id=message_id
        )
        if not source or str(source.get("role") or "") != "user":
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        conversation = assistant_chat_service.get_conversation(
            user_id=user_id,
            conversation_id=str(source["conversation_id"]),
        )
        if not conversation or not conversation.get("project_id"):
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        profile = api_profile(user_id)
        if not profile:
            return RedirectResponse(
                "/settings/api?error="
                + quote(
                    "重新发送消息前，请先配置模型服务"
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        try:
            branch = _queue_novel_message_branch(
                user_id=user_id,
                message_id=message_id,
                replacement_question=question,
                profile=profile,
                settings_tab=settings_tab,
            )
        except ValueError as exc:
            return RedirectResponse(
                _original_novel_message_destination(
                    conversation,
                    settings_tab=settings_tab,
                    error=str(exc),
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        request.app.state.worker.wake()
        return RedirectResponse(
            str(branch["destination"]),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/assistant/messages/{message_id}/regenerate")
    async def regenerate_novel_assistant_message(
        request: Request,
        message_id: str,
        settings_tab: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        source = assistant_chat_service.get_message(
            user_id=user_id, message_id=message_id
        )
        if (
            not source
            or str(source.get("role") or "") != "assistant"
        ):
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        conversation = assistant_chat_service.get_conversation(
            user_id=user_id,
            conversation_id=str(source["conversation_id"]),
        )
        if not conversation or not conversation.get("project_id"):
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        profile = api_profile(user_id)
        if not profile:
            return RedirectResponse(
                "/settings/api?error="
                + quote(
                    "重新生成回复前，请先配置模型服务"
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        try:
            branch = _queue_novel_message_branch(
                user_id=user_id,
                message_id=message_id,
                replacement_question=None,
                profile=profile,
                settings_tab=settings_tab,
            )
        except ValueError as exc:
            return RedirectResponse(
                _original_novel_message_destination(
                    conversation,
                    settings_tab=settings_tab,
                    error=str(exc),
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        request.app.state.worker.wake()
        return RedirectResponse(
            str(branch["destination"]),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.get(
        "/documents/{document_id}/assistant",
        response_class=HTMLResponse,
    )
    async def document_assistant_page(
        request: Request,
        document_id: str,
        conversation_id: Optional[str] = None,
        reference_chapter_id: Optional[str] = None,
        new: bool = False,
        error: Optional[str] = None,
        sent: bool = False,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        user_id = int(user["id"])
        document = database.get_document(user_id, document_id)
        if not document:
            return render_template(
                "not_found.html",
                _template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        conversations = (
            assistant_chat_service.list_document_conversations(
                user_id=user_id, document_id=document_id
            )
        )
        active = None
        if conversation_id:
            active = assistant_chat_service.get_conversation(
                user_id=user_id,
                conversation_id=conversation_id,
            )
            if (
                not active
                or str(active.get("document_id") or "")
                != document_id
            ):
                return Response(
                    status_code=status.HTTP_404_NOT_FOUND
                )
        elif not new:
            selected = None
            if reference_chapter_id:
                selected = next(
                    (
                        item
                        for item in conversations
                        if str(
                            item.get("reference_chapter_id") or ""
                        )
                        == reference_chapter_id
                    ),
                    None,
                )
            selected = selected or (
                conversations[0] if conversations else None
            )
            if selected:
                active = assistant_chat_service.get_conversation(
                    user_id=user_id,
                    conversation_id=str(selected["id"]),
                )
        chapters = database.list_chapters(user_id, document_id)
        source_chapter_id = reference_chapter_id or ""
        if active and active.get("reference_chapter_id"):
            source_chapter_id = str(
                active["reference_chapter_id"]
            )
        source = None
        if source_chapter_id:
            source = assistant_chat_service.get_reference_source(
                user_id=user_id,
                document_id=document_id,
                reference_chapter_id=source_chapter_id,
            )
            if source:
                source["content_hash"] = hashlib.sha256(
                    str(source["content"]).encode("utf-8")
                ).hexdigest()
                source["source_label"] = (
                    f"参考书第 {source['position']} 章"
                    f"《{source['title']}》"
                )
        return render_template(
            "assistant_chat.html",
            _template_context(
                request,
                user=user,
                root=document,
                conversations=conversations,
                active_conversation=active,
                scope_chapters=chapters,
                selected_scope_chapter_id=source_chapter_id,
                source=source,
                assistant_base_url=(
                    f"/documents/{document_id}/assistant"
                ),
                assistant_new_url=(
                    f"/documents/{document_id}/assistant/new"
                ),
                assistant_post_url=(
                    f"/documents/{document_id}/assistant/messages"
                ),
                back_url=f"/documents/{document_id}",
                back_label="返回拆书文档",
                error=error,
                sent=sent,
            ),
        )

    @application.post("/documents/{document_id}/assistant/new")
    async def new_document_assistant_conversation(
        request: Request,
        document_id: str,
        reference_chapter_id: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        document = database.get_document(user_id, document_id)
        if not document:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        clean_chapter_id = reference_chapter_id.strip()
        try:
            conversation_id = assistant_chat_service.create_conversation(
                user_id=user_id,
                scope_type=(
                    "reference_chapter"
                    if clean_chapter_id
                    else "document"
                ),
                title="新对话",
                document_id=document_id,
                reference_chapter_id=clean_chapter_id or None,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/documents/{document_id}/assistant?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/documents/{document_id}/assistant?conversation_id={quote(conversation_id)}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/documents/{document_id}/assistant/messages")
    async def send_document_assistant_message(
        request: Request,
        document_id: str,
        question: str = Form(...),
        conversation_id: str = Form(""),
        reference_chapter_id: str = Form(""),
        source_type: str = Form(""),
        quote_start: str = Form(""),
        quote_end: str = Form(""),
        quote_text: str = Form(""),
        source_hash: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        document = database.get_document(user_id, document_id)
        if not document:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        return_path = f"/documents/{document_id}/assistant"
        try:
            if conversation_id:
                conversation = assistant_chat_service.get_conversation(
                    user_id=user_id,
                    conversation_id=conversation_id,
                )
                if (
                    not conversation
                    or str(conversation.get("document_id") or "")
                    != document_id
                ):
                    raise ValueError("对话不存在")
            else:
                scope_type = (
                    "reference_chapter"
                    if reference_chapter_id
                    else "document"
                )
                conversation_id = (
                    assistant_chat_service.create_conversation(
                        user_id=user_id,
                        scope_type=scope_type,
                        title=question,
                        document_id=document_id,
                        reference_chapter_id=(
                            reference_chapter_id or None
                        ),
                    )
                )
            profile = api_profile(user_id)
            if not profile:
                return RedirectResponse(
                    "/settings/api?error="
                    + quote(
                        "开始拆书对话前，请先配置模型服务"
                    ),
                    status_code=status.HTTP_303_SEE_OTHER,
                )
            quote_payload = None
            if quote_text:
                quote_payload = {
                    "source_type": source_type,
                    "document_id": document_id,
                    "reference_chapter_id": reference_chapter_id,
                    "start_offset": quote_start,
                    "end_offset": quote_end,
                    "quote_text": quote_text,
                    "content_hash": source_hash,
                }
            assistant_chat_service.queue_message(
                user_id=user_id,
                conversation_id=conversation_id,
                question=question,
                provider=profile["provider"],
                model=profile["model"],
                credential_source=profile["credential_source"],
                quote=quote_payload,
                agent_role="researcher",
                max_jobs_per_day=app_settings.max_jobs_per_day,
            )
        except ValueError as exc:
            suffix = (
                f"?conversation_id={quote(conversation_id)}"
                if conversation_id
                else ""
            )
            separator = "&" if suffix else "?"
            return RedirectResponse(
                return_path
                + suffix
                + separator
                + "error="
                + quote(str(exc)),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        request.app.state.worker.wake()
        return RedirectResponse(
            return_path
            + f"?conversation_id={quote(conversation_id)}&sent=true",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.get("/api/assistant/messages/{message_id}")
    async def assistant_message_status(
        request: Request, message_id: str
    ):
        user = _current_user(request)
        if not user:
            return JSONResponse(
                {"error": "unauthorized"},
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        message = assistant_chat_service.get_message(
            user_id=int(user["id"]), message_id=message_id
        )
        if not message:
            return JSONResponse(
                {"error": "not_found"},
                status_code=status.HTTP_404_NOT_FOUND,
            )
        message_status = str(message["status"])
        auto_commit_status = str(
            (
                (message.get("response") or {}).get("auto_commit")
                or {}
            ).get("status", "")
        )
        return JSONResponse(
            {
                "id": message["id"],
                "conversation_id": message["conversation_id"],
                "status": message_status,
                "terminal": (
                    message_status == "failed"
                    or (
                        message_status == "completed"
                        and auto_commit_status != "pending"
                    )
                ),
                "error": message.get("error"),
            },
            headers={"Cache-Control": "no-store"},
        )

    @application.post(
        "/assistant/messages/{message_id}/save-rewrite"
    )
    async def save_assistant_rewrite(
        request: Request,
        message_id: str,
        return_to_workbench: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            result = assistant_chat_service.save_rewrite_candidate(
                user_id=int(user["id"]),
                assistant_message_id=message_id,
            )
        except ValueError as exc:
            message = assistant_chat_service.get_message(
                user_id=int(user["id"]), message_id=message_id
            )
            if not message:
                return Response(
                    status_code=status.HTTP_404_NOT_FOUND
                )
            conversation = assistant_chat_service.get_conversation(
                user_id=int(user["id"]),
                conversation_id=str(message["conversation_id"]),
            )
            if conversation and conversation.get("project_id"):
                if (
                    return_to_workbench
                    and conversation.get("novel_chapter_id")
                ):
                    return RedirectResponse(
                        f"/novels/{conversation['project_id']}/workbench"
                        f"?chapter_id={conversation['novel_chapter_id']}"
                        f"&conversation_id={conversation['id']}"
                        f"&error={quote(str(exc))}",
                        status_code=status.HTTP_303_SEE_OTHER,
                    )
                return RedirectResponse(
                    f"/novels/{conversation['project_id']}/assistant"
                    f"?conversation_id={conversation['id']}"
                    f"&error={quote(str(exc))}",
                    status_code=status.HTTP_303_SEE_OTHER,
                )
            return Response(
                status_code=status.HTTP_400_BAD_REQUEST
            )
        if return_to_workbench:
            return RedirectResponse(
                f"/novels/{result['project_id']}/workbench"
                f"?chapter_id={result['chapter_id']}"
                f"&conversation_id={result['conversation_id']}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{result['project_id']}/chapters/"
            f"{result['chapter_id']}?assistant_rewrite=true",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post(
        "/assistant/messages/{message_id}/apply-settings"
    )
    async def apply_assistant_settings(
        request: Request,
        message_id: str,
        return_to_workbench: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        try:
            result = assistant_chat_service.apply_settings_candidate(
                user_id=user_id,
                assistant_message_id=message_id,
            )
        except ValueError as exc:
            message = assistant_chat_service.get_message(
                user_id=user_id, message_id=message_id
            )
            if not message:
                return Response(
                    status_code=status.HTTP_404_NOT_FOUND
                )
            conversation = assistant_chat_service.get_conversation(
                user_id=user_id,
                conversation_id=str(message["conversation_id"]),
            )
            if not conversation or not conversation.get("project_id"):
                return Response(
                    status_code=status.HTTP_400_BAD_REQUEST
                )
            return RedirectResponse(
                f"/novels/{conversation['project_id']}/workbench"
                f"?view=settings&conversation_id={conversation['id']}"
                f"&error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        destination = (
            f"/novels/{result['project_id']}/workbench"
            f"?view=settings&conversation_id={result['conversation_id']}"
        )
        if not return_to_workbench:
            destination = (
                f"/novels/{result['project_id']}/assistant"
                f"?conversation_id={result['conversation_id']}"
            )
        return RedirectResponse(
            destination, status_code=status.HTTP_303_SEE_OTHER
        )

    @application.post(
        "/assistant/messages/{message_id}/apply-story-plan"
    )
    async def apply_assistant_story_plan(
        request: Request,
        message_id: str,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        try:
            result = assistant_chat_service.apply_story_plan_candidate(
                user_id=user_id,
                assistant_message_id=message_id,
            )
        except ValueError as exc:
            message = assistant_chat_service.get_message(
                user_id=user_id, message_id=message_id
            )
            if not message:
                return Response(
                    status_code=status.HTTP_404_NOT_FOUND
                )
            conversation = assistant_chat_service.get_conversation(
                user_id=user_id,
                conversation_id=str(message["conversation_id"]),
            )
            if not conversation or not conversation.get("project_id"):
                return Response(
                    status_code=status.HTTP_400_BAD_REQUEST
                )
            return RedirectResponse(
                f"/novels/{conversation['project_id']}/workbench"
                f"?view=settings&settings_tab=structure"
                f"&conversation_id={conversation['id']}"
                f"&error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{result['project_id']}/workbench"
            "?view=settings&settings_tab=structure"
            f"&conversation_id={result['conversation_id']}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post(
        "/assistant/messages/{message_id}/save-draft"
    )
    async def save_assistant_draft(
        request: Request,
        message_id: str,
        return_to_workbench: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        try:
            result = assistant_chat_service.save_draft_candidate(
                user_id=user_id,
                assistant_message_id=message_id,
            )
        except ValueError as exc:
            message = assistant_chat_service.get_message(
                user_id=user_id, message_id=message_id
            )
            if not message:
                return Response(
                    status_code=status.HTTP_404_NOT_FOUND
                )
            conversation = assistant_chat_service.get_conversation(
                user_id=user_id,
                conversation_id=str(message["conversation_id"]),
            )
            if not conversation or not conversation.get("project_id"):
                return Response(
                    status_code=status.HTTP_400_BAD_REQUEST
                )
            return RedirectResponse(
                f"/novels/{conversation['project_id']}/workbench"
                f"?chapter_id={conversation['novel_chapter_id']}"
                f"&conversation_id={conversation['id']}"
                f"&error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        destination = (
            f"/novels/{result['project_id']}/workbench"
            f"?chapter_id={result['chapter_id']}"
            f"&conversation_id={result['conversation_id']}"
        )
        if not return_to_workbench:
            destination = (
                f"/novels/{result['project_id']}/assistant"
                f"?conversation_id={result['conversation_id']}"
            )
        return RedirectResponse(
            destination, status_code=status.HTTP_303_SEE_OTHER
        )

    @application.post(
        "/assistant/messages/{message_id}/revert-auto-commit"
    )
    async def revert_assistant_auto_commit(
        request: Request,
        message_id: str,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        try:
            result = assistant_chat_service.revert_auto_commit(
                user_id=user_id,
                assistant_message_id=message_id,
            )
        except ValueError as exc:
            message = assistant_chat_service.get_message(
                user_id=user_id, message_id=message_id
            )
            if not message:
                return Response(
                    status_code=status.HTTP_404_NOT_FOUND
                )
            conversation = assistant_chat_service.get_conversation(
                user_id=user_id,
                conversation_id=str(message["conversation_id"]),
            )
            if not conversation or not conversation.get("project_id"):
                return Response(
                    status_code=status.HTTP_400_BAD_REQUEST
                )
            return RedirectResponse(
                f"/novels/{conversation['project_id']}/workbench"
                f"?chapter_id={conversation['novel_chapter_id']}"
                f"&conversation_id={conversation['id']}"
                f"&error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{result['project_id']}/workbench"
            f"?chapter_id={result['chapter_id']}"
            f"&conversation_id={result['conversation_id']}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.get(
        "/novels/{project_id}/chapters/{chapter_id}/versions/"
        "{version_id}/source",
        response_class=HTMLResponse,
    )
    async def novel_version_source_page(
        request: Request,
        project_id: str,
        chapter_id: str,
        version_id: str,
        start: int = 0,
        end: int = 0,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        source = assistant_chat_service.get_novel_source(
            user_id=int(user["id"]),
            project_id=project_id,
            chapter_id=chapter_id,
            version_id=version_id,
        )
        if not source:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        content = str(source["content"])
        safe_start = max(0, min(start, len(content)))
        safe_end = max(safe_start, min(end, len(content)))
        return render_template(
            "assistant_source.html",
            _template_context(
                request,
                user=user,
                source=source,
                source_title=(
                    f"第 {source['position']} 章"
                    f"《{source['chapter_title'] or '未命名章节'}》"
                ),
                source_meta=(
                    "不可变正文版本 · "
                    + (
                        "正史"
                        if str(source["id"])
                        == str(
                            source.get("canonical_version_id") or ""
                        )
                        else "候选 / 历史版本"
                    )
                ),
                before=content[:safe_start],
                selection=content[safe_start:safe_end],
                after=content[safe_end:],
                back_url=(
                    f"/novels/{project_id}/chapters/{chapter_id}"
                ),
            ),
        )

    @application.get(
        "/documents/{document_id}/chapters/{reference_chapter_id}/source",
        response_class=HTMLResponse,
    )
    async def reference_chapter_source_page(
        request: Request,
        document_id: str,
        reference_chapter_id: str,
        start: int = 0,
        end: int = 0,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        source = assistant_chat_service.get_reference_source(
            user_id=int(user["id"]),
            document_id=document_id,
            reference_chapter_id=reference_chapter_id,
        )
        if not source:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        content = str(source["content"])
        safe_start = max(0, min(start, len(content)))
        safe_end = max(safe_start, min(end, len(content)))
        return render_template(
            "assistant_source.html",
            _template_context(
                request,
                user=user,
                source=source,
                source_title=(
                    f"参考书第 {source['position']} 章"
                    f"《{source['title']}》"
                ),
                source_meta=(
                    f"拆书原文 · {source['document_title']}"
                ),
                before=content[:safe_start],
                selection=content[safe_start:safe_end],
                after=content[safe_end:],
                back_url=f"/documents/{document_id}",
            ),
        )

    @application.get("/upload", response_class=HTMLResponse)
    async def upload_page(request: Request):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        return render_template(
            "upload.html",
            _template_context(
                request,
                user=user,
                max_upload_mb=app_settings.max_upload_bytes // 1024 // 1024,
            ),
        )

    @application.post("/upload", response_class=HTMLResponse)
    async def upload(
        request: Request,
        source_file: UploadFile = File(...),
        title: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        filename = Path(source_file.filename or "untitled.txt").name
        extension = Path(filename).suffix.lower()
        error: Optional[str] = None
        if extension not in ALLOWED_EXTENSIONS:
            error = "目前仅支持 .txt、.text 和 .md 文本文件"

        raw = b""
        if not error:
            raw = await source_file.read(app_settings.max_upload_bytes + 1)
            if len(raw) > app_settings.max_upload_bytes:
                error = (
                    f"文件超过 {app_settings.max_upload_bytes // 1024 // 1024} MB 限制"
                )
            elif not raw:
                error = "文件为空"

        text = ""
        encoding = ""
        chunks = []
        if not error:
            try:
                text, encoding = decode_upload(raw)
                if not text.strip():
                    raise ValueError("文件中没有可分析的正文")
                if len(text) > app_settings.max_text_chars:
                    raise ValueError(
                        f"正文超过 {app_settings.max_text_chars:,} 字限制"
                    )
                chunks = split_chapters(
                    text,
                    target_chars=app_settings.target_chapter_chars,
                    max_chars=app_settings.max_chapter_chars,
                )
                if not chunks:
                    raise ValueError("没有识别到可分析内容")
                usage = database.get_user_usage(int(user["id"]))
                if (
                    usage["document_count"]
                    >= app_settings.max_documents_per_user
                ):
                    raise ValueError(
                        f"每个账号最多保存 "
                        f"{app_settings.max_documents_per_user} 本文档"
                    )
                if (
                    usage["stored_chars"] + len(text)
                    > app_settings.max_stored_chars_per_user
                ):
                    raise ValueError(
                        f"账号累计正文不能超过 "
                        f"{app_settings.max_stored_chars_per_user:,} 字"
                    )
            except ValueError as exc:
                error = str(exc)

        clean_title = title.strip() or Path(filename).stem
        clean_title = clean_title[:120]
        if error:
            return render_template(
                "upload.html",
                _template_context(
                    request,
                    user=user,
                    error=error,
                    title=clean_title,
                    max_upload_mb=app_settings.max_upload_bytes // 1024 // 1024,
                ),
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        document_id = secrets.token_hex(16)
        document_dir = (
            app_settings.documents_dir / str(user["id"]) / document_id
        )
        chapters_dir = document_dir / "chapters"
        source_path = document_dir / "source.txt"
        try:
            chapters_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
            os.chmod(app_settings.documents_dir / str(user["id"]), 0o700)
            os.chmod(document_dir, 0o700)
            os.chmod(chapters_dir, 0o700)
            source_path.write_text(text, encoding="utf-8")
            source_path.chmod(0o600)
            chapter_paths = []
            for position, chunk in enumerate(chunks, start=1):
                chapter_path = chapters_dir / f"{position:05d}.txt"
                chapter_path.write_text(chunk.text, encoding="utf-8")
                chapter_path.chmod(0o600)
                chapter_paths.append(chapter_path)
            database.create_document(
                user_id=int(user["id"]),
                title=clean_title,
                original_filename=filename,
                source_path=source_path,
                source_encoding=encoding,
                text_length=len(text),
                chunks=chunks,
                chapter_paths=chapter_paths,
                max_documents=app_settings.max_documents_per_user,
                max_stored_chars=app_settings.max_stored_chars_per_user,
            )
        except ValueError as exc:
            shutil.rmtree(document_dir, ignore_errors=True)
            return render_template(
                "upload.html",
                _template_context(
                    request,
                    user=user,
                    error=str(exc),
                    title=clean_title,
                    max_upload_mb=app_settings.max_upload_bytes // 1024 // 1024,
                ),
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        except Exception:
            shutil.rmtree(document_dir, ignore_errors=True)
            logger.exception("failed to persist uploaded document")
            return render_template(
                "upload.html",
                _template_context(
                    request,
                    user=user,
                    error="保存文件失败，请稍后重试",
                    title=clean_title,
                    max_upload_mb=app_settings.max_upload_bytes // 1024 // 1024,
                ),
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        return RedirectResponse(
            f"/documents/{document_id}", status_code=status.HTTP_303_SEE_OTHER
        )

    @application.get("/documents/{document_id}", response_class=HTMLResponse)
    async def document_page(
        request: Request, document_id: str, error: Optional[str] = None
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        document = database.get_document(int(user["id"]), document_id)
        if not document:
            return render_template(
                "not_found.html",
                _template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        chapters = database.list_chapters(
            int(user["id"]), document_id, document.get("latest_job_id")
        )
        return render_template(
            "document.html",
            _template_context(
                request,
                user=user,
                document=document,
                chapters=chapters,
                error=error,
            ),
        )

    @application.post("/documents/{document_id}/analyze")
    async def analyze_document(
        request: Request, document_id: str, csrf: str = Form(...)
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        document = database.get_document(int(user["id"]), document_id)
        if not document:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        credential = database.get_api_credential_summary(int(user["id"]))
        analyzer = request.app.state.analyzer
        if credential:
            provider = str(credential["provider"])
            model = str(credential["model"])
            credential_source = "personal"
        elif app_settings.deepseek_api_key:
            provider = analyzer.provider
            model = analyzer.model
            credential_source = "default"
        elif app_settings.uses_test_models:
            provider = analyzer.provider
            model = analyzer.model
            credential_source = "default"
        else:
            return RedirectResponse(
                "/settings/api?error="
                + quote("开始分析前，请先配置你的模型服务"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        try:
            job_id = database.create_job(
                user_id=int(user["id"]),
                document_id=document_id,
                provider=provider,
                model=model,
                credential_source=credential_source,
                max_jobs_per_day=app_settings.max_jobs_per_day,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/documents/{document_id}?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        request.app.state.worker.wake()
        return RedirectResponse(
            f"/jobs/{job_id}", status_code=status.HTTP_303_SEE_OTHER
        )

    @application.get("/jobs/{job_id}", response_class=HTMLResponse)
    async def job_page(
        request: Request, job_id: str, error: Optional[str] = None
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        job = database.get_job(int(user["id"]), job_id)
        if not job:
            return render_template(
                "not_found.html",
                _template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        chapters = database.list_chapters(
            int(user["id"]), str(job["document_id"]), job_id
        )
        return render_template(
            "job.html",
            _template_context(
                request, user=user, job=job, chapters=chapters, error=error
            ),
        )

    @application.get("/api/jobs/{job_id}")
    async def job_status(request: Request, job_id: str):
        user = _current_user(request)
        if not user:
            return JSONResponse(
                {"detail": "未登录"}, status_code=status.HTTP_401_UNAUTHORIZED
            )
        job = database.get_job(int(user["id"]), job_id)
        if not job:
            return JSONResponse(
                {"detail": "任务不存在"}, status_code=status.HTTP_404_NOT_FOUND
            )
        completed = int(job["completed_chapters"])
        failed = int(job["failed_chapters"])
        total = int(job["total_chapters"])
        return {
            "id": job_id,
            "status": job["status"],
            "completed": completed,
            "failed": failed,
            "total": total,
            "processed": completed + failed,
            "percent": round((completed + failed) / total * 100, 1) if total else 0,
            "terminal": job["status"] in {"completed", "partial", "failed"},
        }

    @application.post("/jobs/{job_id}/retry")
    async def retry_job(request: Request, job_id: str, csrf: str = Form(...)):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        job = database.get_job(int(user["id"]), job_id)
        if not job:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        if (
            job.get("credential_source") == "personal"
            and not database.has_api_credential(int(user["id"]))
        ):
            return RedirectResponse(
                "/settings/api?error="
                + quote("重试这个任务前，请先重新配置个人模型凭据"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        retried = database.retry_failed(int(user["id"]), job_id)
        if not retried:
            return RedirectResponse(
                f"/jobs/{job_id}?error={quote('当前无法重试；请确认任务已结束且没有其他任务正在运行')}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        request.app.state.worker.wake()
        return RedirectResponse(
            f"/jobs/{job_id}", status_code=status.HTTP_303_SEE_OTHER
        )

    @application.get("/analyses/{analysis_id}", response_class=HTMLResponse)
    async def analysis_page(
        request: Request,
        analysis_id: str,
        error: Optional[str] = None,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        analysis = database.get_analysis(int(user["id"]), analysis_id)
        if not analysis:
            return render_template(
                "not_found.html",
                _template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        result = (
            json.loads(analysis["result_json"])
            if analysis.get("result_json")
            else None
        )
        saved_technique_names = technique_service.list_saved_names_for_analysis(
            user_id=int(user["id"]), analysis_id=analysis_id
        )
        source_text = Path(str(analysis["content_path"])).read_text(encoding="utf-8")
        return render_template(
            "analysis.html",
            _template_context(
                request,
                user=user,
                analysis=analysis,
                result=result,
                saved_technique_names=saved_technique_names,
                source_text=source_text,
                error=error,
            ),
        )

    @application.post(
        "/analyses/{analysis_id}/techniques/{technique_index}"
    )
    async def save_analysis_technique(
        request: Request,
        analysis_id: str,
        technique_index: int,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            technique_id, created = technique_service.create_from_analysis(
                user_id=int(user["id"]),
                analysis_id=analysis_id,
                technique_index=technique_index,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/analyses/{analysis_id}?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/techniques/{technique_id}?"
            + ("created=true" if created else "saved=true"),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.get("/jobs/{job_id}/export.json")
    async def export_job(request: Request, job_id: str):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        payload = database.export_job(int(user["id"]), job_id)
        if not payload:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        filename = f"story-analysis-{job_id[:8]}.json"
        return Response(
            json.dumps(payload, ensure_ascii=False, indent=2),
            media_type="application/json; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    return application


app = create_app()
