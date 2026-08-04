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
from contextlib import asynccontextmanager
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
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from starlette.background import BackgroundTask

from .agent_model import build_agent_model
from .assistant_chat_service import (
    SETTING_FIELD_LABELS,
    AssistantChatService,
)
from .causal_branch_adoption_schema import CausalBranchTaskPatch
from .causal_branch_adoption_service import CausalBranchAdoptionService
from .causal_branch_planner import build_causal_branch_planner
from .causal_branch_service import CausalBranchSimulationService
from .causal_suggestion_planner import build_causal_suggestion_planner
from .causal_suggestion_service import CausalSuggestionService
from .chapter_splitter import decode_upload, split_chapters
from .config import Settings
from .continuity import ContinuityService
from .credentials import (
    CredentialCipher,
    CredentialError,
    key_hint,
    validate_api_key,
    validate_model,
)
from .db import ChapterHeadConflict, Database, utc_now
from .model_client import build_analyzer
from .memory_extraction import build_memory_extractor
from .memory_identity import (
    IDENTITY_TYPES,
    MemoryIdentityService,
)
from .memory_service import MemoryService
from .model_catalog import ModelCatalogError, fetch_models
from .model_provider import (
    ProviderConfigError,
    get_provider,
    list_providers,
    normalize_provider_base_url,
    settings_for_reasoning_policy,
)
from .model_routing import (
    ModelTaskPolicy,
    normalize_quality_mode,
    route_model_task,
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
from .text_metrics import effective_char_count
from .reader_planner import build_reader_planner
from .reader_service import ReaderDecisionService
from .security import (
    csrf_token,
    verify_csrf,
)
from .story_planning_service import StoryPlanningService
from .story_plan_suggestion_service import StoryPlanSuggestionService
from .story_planner import build_story_planner
from .story_structure_planner import build_story_structure_planner
from .story_structure_schema import AuthorChapterSkeleton
from .story_structure_service import StoryStructureSuggestionService
from .structure_link_service import StructureLinkService
from .worker import AnalysisWorker
from .work_archive import (
    WORK_ARCHIVE_FORMAT,
    WorkArchiveError,
    create_work_archive,
    detect_archive_format,
    import_work_archive,
)
from .style_editor import build_style_editor
from .style_service import StyleService
from .technique_service import TechniqueService
from .template_filters import (
    _chapter_structure_role_label,
    _continuity_issue_label,
    _edit_preference_category_label,
    _edit_preference_status_label,
    _foreshadow_status_label,
    _human_size,
    _impact_item_type_label,
    _knowledge_state_label,
    _memory_identity_type_label,
    _plot_status_label,
    _reader_request_type_label,
    _reader_scope_label,
    _reader_status_label,
    _status_label,
    _story_arc_lifecycle_label,
    _story_arc_type_label,
    _story_memory_label,
    _story_plan_status_label,
    _story_planning_mode_label,
    _style_issue_label,
    _technique_dimension_label,
    _technique_scope_label,
    _technique_usage_label,
    _voice_dimension_label,
    _voice_suggestion_status_label,
    _writing_operation_label,
    _writing_status_label,
)
from .voice_extraction import build_voice_profile_extractor
from .version_diff import build_version_diff
from .work_library import (
    create_main_from_version,
    create_reading_document_from_chunks,
    create_version_tag,
)
from .web_paths import (
    api_settings_path as _api_settings_path,
    append_query as _append_query,
    document_workbench_path as _document_workbench_path,
    request_relative_url as _request_relative_url,
    safe_next as _safe_next,
    wants_json as _wants_json,
    work_archive_destination as _work_archive_destination,
    workbench_path as _workbench_path,
)
from .web_security import (
    current_user as _current_user,
    login_redirect as _login_redirect,
)
from .web_forms import (
    _clean_field,
    _planned_story_arc_from_form,
    _split_lines,
    _story_blueprint_from_form,
    _story_plan_lines,
    _technique_observation_from_form,
)
from .workbench_view import (
    POV_OPTIONS,
    STORY_ARC_TYPE_OPTIONS,
    WORK_ANALYSIS_CATEGORIES,
    WORK_ARCHIVE_CATEGORIES,
    WORK_ARCHIVE_TAB_KEYS,
    WORKBENCH_SETTING_TAB_KEYS,
    WORKBENCH_SETTING_TABS,
    WORLD_ENTRY_TYPE_OPTIONS,
    WorkbenchNotFound,
    WorkbenchUnavailable,
    WorkbenchViewBuilder,
)
from .web_auth import build_auth_router
from .web_system import build_system_router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"
ALLOWED_EXTENSIONS = {".txt", ".md", ".text"}
WORK_ARCHIVE_CATEGORY_KEYS = frozenset(key for key, _label in WORK_ARCHIVE_CATEGORIES)
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
def _template_context(
    request: Request,
    *,
    user: Optional[Dict[str, Any]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    settings: Settings = request.app.state.settings
    current_url = _request_relative_url(request)
    personal_api_configured = False
    if user:
        database: Database = request.app.state.database
        personal_api_configured = database.has_api_credential(int(user["id"]))
    using_mock = (
        not personal_api_configured
        and not settings.model_api_key
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
            and not settings.model_api_key
            and not using_mock
        ),
        "personal_api_configured": personal_api_configured,
        "model_settings_url": _api_settings_path(return_to=current_url),
        "model_settings_panel_url": _api_settings_path(
            embedded="true",
            return_to=current_url,
        ),
        "suppress_model_settings_dialog": False,
        **extra,
    }


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
    structure_link_service = StructureLinkService(database)
    causal_suggestion_service = CausalSuggestionService(database)
    causal_branch_service = CausalBranchSimulationService(database)
    causal_branch_adoption_service = CausalBranchAdoptionService(database)
    style_service = StyleService(database)
    preference_service = PreferenceService(database)
    reader_service = ReaderDecisionService(database, app_settings.novels_dir)
    assistant_chat_service = AssistantChatService(
        database,
        app_settings.novels_dir,
        app_settings.documents_dir,
    )
    technique_service = TechniqueService(database)
    continuity_service = ContinuityService(database)
    identity_service = MemoryIdentityService(database)
    workbench_view_builder = WorkbenchViewBuilder(
        database=database,
        memory_service=memory_service,
        assistant_chat_service=assistant_chat_service,
        planning_service=planning_service,
        story_planning_service=story_planning_service,
        style_service=style_service,
        continuity_service=continuity_service,
    )
    credential_cipher = CredentialCipher(app_settings.credential_secret)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.filters["human_size"] = _human_size
    templates.env.filters["status_label"] = _status_label
    templates.env.filters["writing_status_label"] = _writing_status_label
    templates.env.filters["writing_operation_label"] = _writing_operation_label
    templates.env.filters["style_issue_label"] = _style_issue_label
    templates.env.filters["voice_suggestion_status_label"] = (
        _voice_suggestion_status_label
    )
    templates.env.filters["voice_dimension_label"] = _voice_dimension_label
    templates.env.filters["edit_preference_category_label"] = (
        _edit_preference_category_label
    )
    templates.env.filters["edit_preference_status_label"] = (
        _edit_preference_status_label
    )
    templates.env.filters["story_arc_type_label"] = _story_arc_type_label
    templates.env.filters["story_arc_lifecycle_label"] = _story_arc_lifecycle_label
    templates.env.filters["story_plan_status_label"] = _story_plan_status_label
    templates.env.filters["story_planning_mode_label"] = _story_planning_mode_label
    templates.env.filters["chapter_structure_role_label"] = (
        _chapter_structure_role_label
    )
    templates.env.filters["reader_request_type_label"] = _reader_request_type_label
    templates.env.filters["reader_scope_label"] = _reader_scope_label
    templates.env.filters["reader_status_label"] = _reader_status_label
    templates.env.filters["impact_item_type_label"] = _impact_item_type_label
    templates.env.filters["technique_dimension_label"] = _technique_dimension_label
    templates.env.filters["technique_scope_label"] = _technique_scope_label
    templates.env.filters["technique_usage_label"] = _technique_usage_label
    templates.env.filters["continuity_issue_label"] = _continuity_issue_label
    templates.env.filters["knowledge_state_label"] = _knowledge_state_label
    templates.env.filters["plot_status_label"] = _plot_status_label
    templates.env.filters["foreshadow_status_label"] = _foreshadow_status_label
    templates.env.filters["memory_identity_type_label"] = _memory_identity_type_label
    templates.env.filters["story_memory_label"] = _story_memory_label

    def render_template(
        name: str, context: Dict[str, Any], status_code: int = status.HTTP_200_OK
    ):
        return templates.TemplateResponse(
            context["request"], name, context, status_code=status_code
        )

    def api_profile(user_id: int, model_choice: str = "") -> Optional[Dict[str, str]]:
        clean_choice = str(model_choice or "").strip()
        if clean_choice:
            try:
                provider, model = clean_choice.split("|", 1)
            except ValueError as exc:
                raise ValueError("所选模型无效，请刷新页面后重试") from exc
            credential = database.get_api_credential_summary(user_id, provider)
            if credential:
                allowed_models = set(database.list_api_models(user_id, provider))
                allowed_models.add(str(credential["model"]))
                if model not in allowed_models:
                    raise ValueError("所选模型不在“我的模型”中，请先到模型设置添加")
                return {
                    "provider": provider,
                    "model": model,
                    "credential_source": "personal",
                }
            default_profile = {
                "provider": app_settings.model_provider,
                "model": app_settings.model_name,
                "credential_source": "default",
            }
            if app_settings.uses_test_models:
                default_profile = {
                    "provider": "mock",
                    "model": "mock-novel-writer",
                    "credential_source": "default",
                }
            if (
                app_settings.model_api_key or app_settings.uses_test_models
            ) and clean_choice == (
                f"{default_profile['provider']}|{default_profile['model']}"
            ):
                return default_profile
            raise ValueError("所选模型配置不存在，请重新选择")
        credential = database.get_api_credential_summary(user_id)
        if credential:
            return {
                "provider": str(credential["provider"]),
                "model": str(credential["model"]),
                "credential_source": "personal",
            }
        if app_settings.model_api_key:
            return {
                "provider": app_settings.model_provider,
                "model": app_settings.model_name,
                "credential_source": "default",
            }
        if app_settings.uses_test_models:
            return {
                "provider": "mock",
                "model": "mock-novel-writer",
                "credential_source": "default",
            }
        return None

    def chat_model_groups(user_id: int) -> list[Dict[str, Any]]:
        groups: list[Dict[str, Any]] = []
        for credential in database.list_api_credentials(user_id):
            provider_id = str(credential["provider"])
            try:
                provider_label = get_provider(provider_id).label
            except ProviderConfigError:
                provider_label = provider_id
            models = database.list_api_models(user_id, provider_id)
            default_model = str(credential["model"])
            if default_model not in models:
                models.insert(0, default_model)
            groups.append(
                {
                    "provider": provider_id,
                    "label": provider_label,
                    "models": [
                        {
                            "id": model,
                            "value": f"{provider_id}|{model}",
                            "is_default": (
                                bool(credential["is_default"])
                                and model == default_model
                            ),
                        }
                        for model in models
                    ],
                }
            )
        if groups:
            return groups
        profile = api_profile(user_id)
        if not profile:
            return []
        provider_id = profile["provider"]
        try:
            provider_label = get_provider(provider_id).label
        except ProviderConfigError:
            provider_label = "测试模型" if provider_id == "mock" else provider_id
        return [
            {
                "provider": provider_id,
                "label": provider_label,
                "models": [
                    {
                        "id": profile["model"],
                        "value": f"{provider_id}|{profile['model']}",
                        "is_default": True,
                    }
                ],
            }
        ]

    def selected_chat_model(
        groups: list[Dict[str, Any]],
        conversation: Optional[Mapping[str, Any]] = None,
    ) -> str:
        available = {
            str(model["value"]) for group in groups for model in group["models"]
        }
        if conversation:
            for message in reversed(conversation.get("messages") or []):
                if (
                    str(message.get("role") or "") == "assistant"
                    and message.get("provider")
                    and message.get("model")
                ):
                    previous = f"{message['provider']}|{message['model']}"
                    if previous in available:
                        return previous
        for group in groups:
            for model in group["models"]:
                if model["is_default"]:
                    return str(model["value"])
        for group in groups:
            if group["models"]:
                return str(group["models"][0]["value"])
        return ""

    def model_routing_configuration(
        user_id: int,
    ) -> Dict[str, Any]:
        groups = chat_model_groups(user_id)
        available = {
            str(model["value"]) for group in groups for model in group["models"]
        }
        preferences = database.get_model_routing_preferences(user_id)
        fallback = selected_chat_model(groups)

        def configured_choice(role: str) -> str:
            value = f"{preferences[f'{role}_provider']}|{preferences[f'{role}_model']}"
            return value if value in available else fallback

        return {
            "groups": groups,
            "available": available,
            "fast_model_choice": configured_choice("fast"),
            "quality_model_choice": configured_choice("quality"),
            "default_quality_mode": normalize_quality_mode(
                preferences["default_quality_mode"]
            ),
        }

    def routed_api_profile(
        user_id: int,
        *,
        quality_mode: str,
        task_policy: ModelTaskPolicy,
        model_choice: str = "",
    ) -> Optional[Dict[str, str]]:
        if str(model_choice or "").strip():
            return api_profile(user_id, model_choice)
        configuration = model_routing_configuration(user_id)
        decision = route_model_task(
            normalize_quality_mode(quality_mode),
            task_policy,
        )
        choice = str(configuration[f"{decision.model_role}_model_choice"])
        return api_profile(user_id, choice) if choice else api_profile(user_id)

    def selected_quality_mode(
        user_id: int,
        conversation: Optional[Mapping[str, Any]] = None,
    ) -> str:
        if conversation and conversation.get("quality_mode"):
            return normalize_quality_mode(conversation["quality_mode"])
        return str(model_routing_configuration(user_id)["default_quality_mode"])

    def quality_mode_options(user_id: int) -> list[Dict[str, str]]:
        configuration = model_routing_configuration(user_id)
        fast_model = str(configuration["fast_model_choice"]).partition("|")[2]
        quality_model = str(configuration["quality_model_choice"]).partition("|")[2]
        return [
            {
                "value": "low",
                "label": "Low",
                "detail": fast_model or "快速模型",
                "description": ("全部使用快速模型；服务商支持时关闭思考。"),
            },
            {
                "value": "standard",
                "label": "Standard",
                "detail": "自动",
                "description": ("轻量步骤使用快速模型，写作与分析使用高质量模型。"),
            },
            {
                "value": "max",
                "label": "Max",
                "detail": quality_model or "高质量模型",
                "description": (
                    "全部使用高质量模型，并在服务商支持时使用最高推理强度。"
                ),
            },
        ]

    def queue_background_memory(
        request: Request,
        *,
        user_id: int,
        project_id: str,
        chapter_id: str,
        version_id: str,
    ) -> None:
        try:
            targets = database.list_memory_refresh_targets(
                user_id=user_id,
                project_id=project_id,
                chapter_id=chapter_id,
            )
            if not targets:
                targets = [
                    {
                        "chapter_id": chapter_id,
                        "head_version_id": version_id,
                    }
                ]
            profile: Optional[dict[str, str]] = None
            queued = False
            for target in targets:
                target_chapter_id = str(target["chapter_id"])
                target_version_id = str(target["head_version_id"])
                deltas = memory_service.list_chapter_deltas(
                    user_id=user_id,
                    project_id=project_id,
                    chapter_id=target_chapter_id,
                )
                active_delta = next(
                    (
                        item
                        for item in deltas
                        if str(item["version_id"]) == target_version_id
                        and str(item["status"])
                        in {"proposed", "author_edited", "projected"}
                    ),
                    None,
                )
                if active_delta:
                    if str(active_delta["status"]) in {
                        "proposed",
                        "author_edited",
                    }:
                        memory_service.accept_delta(
                            user_id=user_id,
                            delta_id=str(active_delta["id"]),
                        )
                    continue
                if profile is None:
                    profile = api_profile(user_id)
                if not profile:
                    continue
                database.create_memory_extraction_job(
                    user_id=user_id,
                    project_id=project_id,
                    chapter_id=target_chapter_id,
                    version_id=target_version_id,
                    provider=profile["provider"],
                    model=profile["model"],
                    credential_source=profile["credential_source"],
                )
                queued = True
            if queued:
                request.app.state.worker.wake()
        except Exception:
            logger.exception(
                "background story memory was not queued chapter=%s",
                chapter_id,
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
            database.prune_chapter_edit_buffers(
                retention_days=app_settings.edit_buffer_retention_days
            )
            app_settings.ensure_directories()
            default_provider_configured = (
                bool(app_settings.model_api_key) or app_settings.uses_test_models
            )
            if default_provider_configured:
                fast_settings = settings_for_reasoning_policy(app_settings, "fast")
                reasoning_settings = settings_for_reasoning_policy(
                    app_settings, "reasoning"
                )
                deep_reasoning_settings = settings_for_reasoning_policy(
                    app_settings, "deep"
                )
                analyzer = build_analyzer(reasoning_settings)
                memory_extractor = build_memory_extractor(fast_settings)
                chapter_planner = build_chapter_planner(deep_reasoning_settings)
                style_editor = build_style_editor(reasoning_settings)
                reader_planner = build_reader_planner(reasoning_settings)
                story_planner = build_story_planner(deep_reasoning_settings)
                story_structure_planner = build_story_structure_planner(
                    deep_reasoning_settings
                )
                causal_suggestion_planner = build_causal_suggestion_planner(
                    deep_reasoning_settings
                )
                causal_branch_planner = build_causal_branch_planner(
                    deep_reasoning_settings
                )
                voice_profile_extractor = build_voice_profile_extractor(fast_settings)
                edit_preference_extractor = build_edit_preference_extractor(
                    fast_settings
                )
                assistant_chat_model = build_agent_model(reasoning_settings)
            else:
                analyzer = None
                memory_extractor = None
                chapter_planner = None
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
                app_settings.secret_key,
                app_settings,
                credential_cipher,
                memory_extractor=memory_extractor,
                chapter_planner=chapter_planner,
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
            application.state.memory_extractor = memory_extractor
            application.state.chapter_planner = chapter_planner
            application.state.style_editor = style_editor
            application.state.reader_planner = reader_planner
            application.state.story_planner = story_planner
            application.state.story_structure_planner = story_structure_planner
            application.state.causal_suggestion_planner = causal_suggestion_planner
            application.state.causal_branch_planner = causal_branch_planner
            application.state.voice_profile_extractor = voice_profile_extractor
            application.state.edit_preference_extractor = edit_preference_extractor
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
    application.state.structure_link_service = structure_link_service
    application.state.causal_suggestion_service = causal_suggestion_service
    application.state.causal_branch_service = causal_branch_service
    application.state.causal_branch_adoption_service = causal_branch_adoption_service
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

    application.include_router(
        build_system_router(database=database, settings=app_settings)
    )
    application.include_router(
        build_auth_router(
            database=database,
            settings=app_settings,
            template_context=_template_context,
            render_template=render_template,
        )
    )

    def render_api_settings(
        request: Request,
        user: Dict[str, Any],
        *,
        error: Optional[str] = None,
        saved: bool = False,
        removed: bool = False,
        adapter_saved: bool = False,
        adapter_error: Optional[str] = None,
        routing_saved: bool = False,
        routing_error: Optional[str] = None,
        search_saved: bool = False,
        search_error: Optional[str] = None,
        model_adapter_prompt: Optional[str] = None,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        models: Optional[list[str]] = None,
        return_to: str = "/dashboard",
        embedded: bool = False,
        settings_tab: str = "providers",
        status_code: int = status.HTTP_200_OK,
    ):
        user_id = int(user["id"])
        safe_return_to = _safe_next(return_to, "/dashboard")
        active_settings_tab = (
            settings_tab
            if settings_tab in {"providers", "routing", "search", "prompts"}
            else "providers"
        )
        default_credential = database.get_api_credential_summary(user_id)
        current_provider = provider or (
            str(default_credential["provider"]) if default_credential else "deepseek"
        )
        try:
            current_provider_spec = get_provider(current_provider)
        except ProviderConfigError:
            current_provider_spec = get_provider("deepseek")
            current_provider = current_provider_spec.id
        credential = database.get_api_credential_summary(user_id, current_provider)
        credential_summaries = []
        for item in database.list_api_credentials(user_id):
            try:
                item_provider = get_provider(str(item["provider"])).public_payload()
            except ProviderConfigError:
                item_provider = {
                    "id": str(item["provider"]),
                    "label": str(item["provider"]),
                    "capabilities": {
                        "configurable_base_url": False,
                    },
                }
            summary = dict(item)
            summary["provider_spec"] = item_provider
            summary["models"] = database.list_api_models(user_id, str(item["provider"]))
            summary["settings_url"] = _api_settings_path(
                provider=str(item["provider"]),
                embedded="true" if embedded else None,
                return_to=safe_return_to,
            )
            credential_summaries.append(summary)
        current_base_url = (
            str(base_url).strip()
            if base_url is not None
            else (
                str(credential.get("base_url") or "").strip()
                if credential
                else current_provider_spec.base_url
            )
        )
        if not current_base_url:
            current_base_url = current_provider_spec.base_url
        current_model = (
            str(model).strip()
            if model is not None
            else (str(credential["model"]) if credential else "")
        )
        current_models = (
            list(dict.fromkeys(str(item) for item in models))
            if models is not None
            else (
                database.list_api_models(user_id, current_provider)
                if credential
                else []
            )
        )
        if current_model and current_model not in current_models:
            current_models.insert(0, current_model)
        stored_adapter_prompt = database.get_model_adapter_prompt(user_id)
        current_adapter_prompt = (
            model_adapter_prompt
            if model_adapter_prompt is not None
            else (
                stored_adapter_prompt
                if stored_adapter_prompt is not None
                else app_settings.model_adapter_prompt
            )
        )
        routing_configuration = model_routing_configuration(user_id)
        web_search_settings = database.get_web_search_summary(user_id)
        return render_template(
            "api_settings.html",
            _template_context(
                request,
                user=user,
                credential=credential,
                credentials=credential_summaries,
                providers=[item.public_payload() for item in list_providers()],
                selected_provider=current_provider,
                selected_provider_spec=current_provider_spec.public_payload(),
                selected_base_url=current_base_url,
                error=error,
                saved=saved,
                removed=removed,
                adapter_saved=adapter_saved,
                adapter_error=adapter_error,
                routing_saved=routing_saved,
                routing_error=routing_error,
                search_saved=search_saved,
                search_error=search_error,
                web_search_settings=web_search_settings,
                selected_model=current_model,
                selected_models=current_models,
                model_adapter_prompt=current_adapter_prompt,
                routing_model_groups=routing_configuration["groups"],
                fast_model_choice=routing_configuration["fast_model_choice"],
                quality_model_choice=routing_configuration["quality_model_choice"],
                default_quality_mode=routing_configuration["default_quality_mode"],
                server_api_available=bool(app_settings.model_api_key),
                settings_back_url=safe_return_to,
                return_to=safe_return_to,
                embedded=embedded,
                active_settings_tab=active_settings_tab,
                provider_tab_url=_api_settings_path(
                    provider=current_provider,
                    embedded="true" if embedded else None,
                    return_to=safe_return_to,
                ),
                prompt_tab_url=_api_settings_path(
                    provider=current_provider,
                    tab="prompts",
                    embedded="true" if embedded else None,
                    return_to=safe_return_to,
                ),
                routing_tab_url=_api_settings_path(
                    provider=current_provider,
                    tab="routing",
                    embedded="true" if embedded else None,
                    return_to=safe_return_to,
                ),
                search_tab_url=_api_settings_path(
                    provider=current_provider,
                    tab="search",
                    embedded="true" if embedded else None,
                    return_to=safe_return_to,
                ),
                suppress_model_settings_dialog=True,
            ),
            status_code=status_code,
        )

    @application.get("/settings/api", response_class=HTMLResponse)
    async def api_settings_page(
        request: Request,
        saved: bool = False,
        removed: bool = False,
        adapter_saved: bool = False,
        routing_saved: bool = False,
        search_saved: bool = False,
        error: Optional[str] = None,
        provider: Optional[str] = None,
        return_to: str = "/dashboard",
        embedded: bool = False,
        tab: str = "providers",
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        return render_api_settings(
            request,
            user,
            error=error,
            saved=saved,
            removed=removed,
            adapter_saved=adapter_saved,
            routing_saved=routing_saved,
            search_saved=search_saved,
            provider=provider,
            return_to=return_to,
            embedded=embedded,
            settings_tab=tab,
        )

    @application.post("/settings/api", response_class=HTMLResponse)
    async def save_api_settings(
        request: Request,
        api_key: str = Form(""),
        provider: str = Form("deepseek"),
        base_url: str = Form(""),
        model: str = Form(""),
        models: list[str] = Form([]),
        return_to: str = Form("/dashboard"),
        embedded: bool = Form(False),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            provider_spec = get_provider(provider)
            clean_base_url = normalize_provider_base_url(
                provider_spec,
                base_url,
                allow_private=(app_settings.permits_private_model_base_urls),
                production=app_settings.app_env.lower() == "production",
            )
            clean_models = list(dict.fromkeys(validate_model(item) for item in models))
            clean_model = validate_model(model)
            if clean_model not in clean_models:
                clean_models.insert(0, clean_model)
            if len(clean_models) > 100:
                raise CredentialError("每个服务商最多保存 100 个模型")
            existing = database.get_api_credential(int(user["id"]), provider_spec.id)
            if api_key:
                clean_key = validate_api_key(api_key)
                encrypted_key = credential_cipher.encrypt(clean_key)
                masked_key = key_hint(clean_key)
            elif existing:
                encrypted_key = str(existing["encrypted_key"])
                masked_key = str(existing["key_hint"])
            elif not provider_spec.capabilities.api_key_required:
                encrypted_key = credential_cipher.encrypt("")
                masked_key = (
                    "无需 Key" if provider_spec.id == "ollama" else "未设置 Key"
                )
            else:
                raise CredentialError(f"请填写 {provider_spec.label} API Key")
            database.upsert_api_credential(
                user_id=int(user["id"]),
                provider=provider_spec.id,
                base_url=clean_base_url,
                encrypted_key=encrypted_key,
                key_hint=masked_key,
                model=clean_model,
                models=clean_models,
            )
        except ValueError as exc:
            return render_api_settings(
                request,
                user,
                error=str(exc),
                provider=provider,
                base_url=base_url,
                model=model,
                models=models,
                return_to=return_to,
                embedded=embedded,
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        return RedirectResponse(
            _api_settings_path(
                saved="true",
                embedded="true" if embedded else None,
                return_to=return_to,
            ),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post(
        "/settings/model-adapter",
        response_class=HTMLResponse,
    )
    async def save_model_adapter(
        request: Request,
        model_adapter_prompt: str = Form(""),
        provider: str = Form("deepseek"),
        return_to: str = Form("/dashboard"),
        embedded: bool = Form(False),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            current_provider = get_provider(provider).id
            clean_prompt = _clean_field(
                model_adapter_prompt,
                "通用模型适配策略",
                max_length=20_000,
            )
            database.upsert_model_adapter_prompt(int(user["id"]), clean_prompt)
        except ValueError as exc:
            return render_api_settings(
                request,
                user,
                adapter_error=str(exc),
                model_adapter_prompt=model_adapter_prompt,
                provider=provider,
                return_to=return_to,
                embedded=embedded,
                settings_tab="prompts",
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        return RedirectResponse(
            _api_settings_path(
                provider=current_provider,
                tab="prompts",
                adapter_saved="true",
                embedded="true" if embedded else None,
                return_to=return_to,
            ),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post(
        "/settings/model-routing",
        response_class=HTMLResponse,
    )
    async def save_model_routing(
        request: Request,
        fast_model_choice: str = Form(""),
        quality_model_choice: str = Form(""),
        default_quality_mode: str = Form("standard"),
        provider: str = Form("deepseek"),
        return_to: str = Form("/dashboard"),
        embedded: bool = Form(False),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        try:
            configuration = model_routing_configuration(user_id)
            available = set(configuration["available"])
            if (
                fast_model_choice not in available
                or quality_model_choice not in available
            ):
                raise ValueError("快速模型和高质量模型都必须来自“我的模型”")
            fast_provider, fast_model = fast_model_choice.split("|", 1)
            quality_provider, quality_model = quality_model_choice.split("|", 1)
            clean_mode = normalize_quality_mode(default_quality_mode)
            if clean_mode != str(default_quality_mode).strip().lower():
                raise ValueError("不支持的默认模型强度")
            database.upsert_model_routing_preferences(
                user_id=user_id,
                fast_provider=fast_provider,
                fast_model=fast_model,
                quality_provider=quality_provider,
                quality_model=quality_model,
                default_quality_mode=clean_mode,
            )
            current_provider = get_provider(provider).id
        except ValueError as exc:
            return render_api_settings(
                request,
                user,
                routing_error=str(exc),
                provider=provider,
                return_to=return_to,
                embedded=embedded,
                settings_tab="routing",
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        return RedirectResponse(
            _api_settings_path(
                provider=current_provider,
                tab="routing",
                routing_saved="true",
                embedded="true" if embedded else None,
                return_to=return_to,
            ),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post(
        "/settings/web-search",
        response_class=HTMLResponse,
    )
    async def save_web_search_settings(
        request: Request,
        enabled: bool = Form(False),
        provider: str = Form("deepseek"),
        return_to: str = Form("/dashboard"),
        embedded: bool = Form(False),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        try:
            current_provider = get_provider(provider).id
            database.upsert_web_search_settings(
                user_id=user_id,
                enabled=enabled,
            )
        except ValueError as exc:
            return render_api_settings(
                request,
                user,
                search_error=str(exc),
                provider=provider,
                return_to=return_to,
                embedded=embedded,
                settings_tab="search",
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        return RedirectResponse(
            _api_settings_path(
                provider=current_provider,
                tab="search",
                search_saved="true",
                embedded="true" if embedded else None,
                return_to=return_to,
            ),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/api/assistant/conversations/{conversation_id}/quality-mode")
    async def set_assistant_conversation_quality_mode(
        request: Request,
        conversation_id: str,
        quality_mode: str = Form(...),
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
            selected = assistant_chat_service.set_conversation_quality_mode(
                user_id=int(user["id"]),
                conversation_id=conversation_id,
                quality_mode=quality_mode,
            )
        except ValueError as exc:
            return JSONResponse(
                {"error": str(exc)},
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        return JSONResponse({"quality_mode": selected})

    @application.post("/api/settings/quality-mode")
    async def remember_default_quality_mode(
        request: Request,
        quality_mode: str = Form(...),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return JSONResponse(
                {"error": "unauthorized"},
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        verify_csrf(request, csrf)
        selected = normalize_quality_mode(quality_mode)
        if str(quality_mode or "").strip().lower() != selected:
            return JSONResponse(
                {"error": "不支持的模型强度"},
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        database.remember_quality_mode(int(user["id"]), selected)
        return JSONResponse({"quality_mode": selected})

    @application.post("/api/settings/models")
    async def api_model_catalog(
        request: Request,
        api_key: str = Form(""),
        provider: str = Form("deepseek"),
        base_url: str = Form(""),
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
            existing = database.get_api_credential(int(user["id"]), provider_spec.id)
            submitted_base_url = base_url
            if not submitted_base_url.strip() and existing:
                submitted_base_url = str(existing.get("base_url") or "")
            clean_base_url = normalize_provider_base_url(
                provider_spec,
                submitted_base_url,
                allow_private=(app_settings.permits_private_model_base_urls),
                production=app_settings.app_env.lower() == "production",
            )
            if api_key:
                clean_key = validate_api_key(api_key)
            else:
                if existing:
                    clean_key = credential_cipher.decrypt(
                        str(existing["encrypted_key"])
                    )
                elif provider_spec.id == "deepseek" and app_settings.model_api_key:
                    clean_key = app_settings.model_api_key
                elif not provider_spec.capabilities.api_key_required:
                    clean_key = None
                else:
                    raise ModelCatalogError(f"请先输入 {provider_spec.label} API Key")
            models = await fetch_models(
                provider_id=provider_spec.id,
                api_key=clean_key,
                base_url=clean_base_url,
                timeout_seconds=app_settings.model_connect_timeout_seconds,
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

    @application.post("/api/settings/api-key")
    async def reveal_api_key(
        request: Request,
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
            credential = database.get_api_credential(int(user["id"]), provider_spec.id)
            if not credential:
                return JSONResponse(
                    {"error": (f"尚未保存 {provider_spec.label} API Key")},
                    status_code=status.HTTP_404_NOT_FOUND,
                    headers={
                        "Cache-Control": "no-store, private",
                        "Pragma": "no-cache",
                    },
                )
            api_key = credential_cipher.decrypt(str(credential["encrypted_key"]))
        except (CredentialError, ProviderConfigError) as exc:
            return JSONResponse(
                {"error": str(exc)},
                status_code=status.HTTP_400_BAD_REQUEST,
                headers={
                    "Cache-Control": "no-store, private",
                    "Pragma": "no-cache",
                },
            )
        return JSONResponse(
            {"api_key": api_key},
            headers={
                "Cache-Control": "no-store, private",
                "Pragma": "no-cache",
            },
        )

    @application.post("/settings/api/delete")
    async def delete_api_settings(
        request: Request,
        provider: str = Form(""),
        return_to: str = Form("/dashboard"),
        embedded: bool = Form(False),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            provider_id = get_provider(provider).id if provider else None
            database.delete_api_credential(int(user["id"]), provider_id)
        except ValueError as exc:
            return RedirectResponse(
                _api_settings_path(
                    error=str(exc),
                    embedded="true" if embedded else None,
                    return_to=return_to,
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            _api_settings_path(
                removed="true",
                embedded="true" if embedded else None,
                return_to=return_to,
            ),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.get("/api/settings/chat-models")
    async def api_chat_models(request: Request):
        user = _current_user(request)
        if not user:
            return JSONResponse(
                {"error": "unauthorized"},
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        user_id = int(user["id"])
        groups = chat_model_groups(user_id)
        return JSONResponse(
            {
                "groups": groups,
                "default": selected_chat_model(groups),
                "quality_modes": quality_mode_options(user_id),
                "default_quality_mode": selected_quality_mode(user_id),
            }
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
        works = database.list_works(int(user["id"]))
        technique_count = technique_service.count_cards(user_id=int(user["id"]))
        return render_template(
            "dashboard.html",
            _template_context(
                request,
                user=user,
                works=works,
                technique_count=technique_count,
                error=error,
                deleted=deleted,
                imported=imported,
            ),
        )

    @application.get("/import", response_class=HTMLResponse)
    async def unified_import_page(
        request: Request,
        error: Optional[str] = None,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        return render_template(
            "work_import.html",
            _template_context(
                request,
                user=user,
                error=error,
                max_upload_mb=app_settings.max_upload_bytes // 1024 // 1024,
                max_archive_mb=(app_settings.max_work_archive_bytes // 1024 // 1024),
            ),
        )

    @application.post("/import", response_class=HTMLResponse)
    async def unified_import(
        request: Request,
        work_file: UploadFile = File(...),
        title: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        filename = Path(work_file.filename or "untitled.txt").name
        lower_filename = filename.lower()

        def render_import_error(message: str, status_code: int = 400):
            return render_template(
                "work_import.html",
                _template_context(
                    request,
                    user=user,
                    error=message,
                    title=title.strip()[:120],
                    max_upload_mb=(app_settings.max_upload_bytes // 1024 // 1024),
                    max_archive_mb=(
                        app_settings.max_work_archive_bytes // 1024 // 1024
                    ),
                ),
                status_code=status_code,
            )

        if lower_filename.endswith(".zip"):
            temporary = tempfile.NamedTemporaryFile(
                prefix="readraft-unified-import-",
                suffix=".zip",
                delete=False,
            )
            temporary_path = Path(temporary.name)
            total_bytes = 0
            imported_work_id = ""
            try:
                while True:
                    chunk = await work_file.read(1024 * 1024)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > app_settings.max_work_archive_bytes:
                        raise WorkArchiveError("作品归档文件超过允许大小")
                    temporary.write(chunk)
                temporary.close()
                if total_bytes == 0:
                    raise WorkArchiveError("请选择非空的作品归档")
                archive_format = detect_archive_format(temporary_path)
                if archive_format != WORK_ARCHIVE_FORMAT:
                    raise WorkArchiveError("只支持当前版本导出的完整作品归档")
                imported_work = import_work_archive(
                    database=database,
                    novels_dir=app_settings.novels_dir,
                    documents_dir=app_settings.documents_dir,
                    user_id=user_id,
                    archive_path=temporary_path,
                    max_uncompressed_bytes=(app_settings.max_work_archive_bytes),
                    max_documents=app_settings.max_documents_per_user,
                    max_stored_chars=(app_settings.max_stored_chars_per_user),
                )
                imported_work_id = imported_work.work_id
            except (WorkArchiveError, OSError, sqlite3.Error) as exc:
                logger.warning("unified archive import rejected: %s", exc)
                return render_import_error(str(exc))
            finally:
                temporary.close()
                temporary_path.unlink(missing_ok=True)
                await work_file.close()
            return RedirectResponse(
                f"/dashboard?imported=true&work={imported_work_id}",
                status_code=status.HTTP_303_SEE_OTHER,
            )

        extension = Path(filename).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            await work_file.close()
            return render_import_error("支持 TXT、Markdown 和 .readraft.zip 作品归档")
        raw = await work_file.read(app_settings.max_upload_bytes + 1)
        await work_file.close()
        if len(raw) > app_settings.max_upload_bytes:
            return render_import_error(
                f"文件超过 {app_settings.max_upload_bytes // 1024 // 1024} MB 限制"
            )
        if not raw:
            return render_import_error("文件为空")
        try:
            text, encoding = decode_upload(raw)
            if not text.strip():
                raise ValueError("文件中没有可导入的正文")
            if len(text) > app_settings.max_text_chars:
                raise ValueError(f"正文超过 {app_settings.max_text_chars:,} 字限制")
            chunks = split_chapters(
                text,
                target_chars=app_settings.target_chapter_chars,
                max_chars=app_settings.max_chapter_chars,
            )
            if not chunks:
                raise ValueError("没有识别到可导入内容")
            clean_title = (title.strip() or Path(filename).stem)[:120]
            document_id = create_reading_document_from_chunks(
                database=database,
                documents_dir=app_settings.documents_dir,
                user_id=user_id,
                title=clean_title,
                original_filename=filename,
                source_encoding=encoding,
                source_text=text,
                chunks=chunks,
                max_documents=app_settings.max_documents_per_user,
                max_stored_chars=app_settings.max_stored_chars_per_user,
            )
            return RedirectResponse(
                f"/documents/{document_id}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        except ValueError as exc:
            return render_import_error(str(exc))
        except Exception:
            logger.exception("failed to import work")
            return render_import_error("导入失败，请稍后重试", 500)

    @application.get("/works/{work_id}")
    async def open_work(request: Request, work_id: str):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        work = database.get_work(int(user["id"]), work_id)
        if not work:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        return RedirectResponse(
            str(work["resume_url"]),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.get("/works/{work_id}/versions/{version_id}")
    async def open_work_version(request: Request, work_id: str, version_id: str):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        user_id = int(user["id"])
        work = database.get_work(user_id, work_id)
        target_version = (
            next(
                (item for item in work["versions"] if str(item["id"]) == version_id),
                None,
            )
            if work
            else None
        )
        if not work or not target_version:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        database.set_work_version(
            user_id=user_id,
            work_id=work_id,
            version_id=version_id,
        )
        return RedirectResponse(
            str(target_version["open_url"]),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.get("/works/{work_id}/archive", response_class=HTMLResponse)
    async def work_archive_page(
        request: Request,
        work_id: str,
        saved: bool = False,
        error: Optional[str] = None,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        user_id = int(user["id"])
        work = database.get_work(user_id, work_id)
        if not work:
            return render_template(
                "not_found.html",
                _template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        return RedirectResponse(
            _work_archive_destination(
                work,
                saved=saved,
                error=error,
            ),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/works/{work_id}/archive")
    async def add_work_archive_note(
        request: Request,
        work_id: str,
        entry_type: str = Form("analysis_note"),
        category: str = Form("uncategorized"),
        content_version_id: str = Form(""),
        title: str = Form(""),
        content: str = Form(...),
        evidence: str = Form(""),
        return_to: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        work = database.get_work(user_id, work_id)
        if not work:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        fallback = _work_archive_destination(work)
        destination = _safe_next(return_to, fallback)
        try:
            clean_title = _clean_field(title, "标题", max_length=120)
            clean_content = _clean_field(
                content,
                "档案内容",
                max_length=20_000,
                required=True,
            )
            clean_evidence = _clean_field(evidence, "依据", max_length=2_000)
            database.add_work_archive_entry(
                user_id=user_id,
                work_id=work_id,
                entry_type=entry_type,
                title=clean_title,
                content=clean_content,
                evidence=clean_evidence,
                content_version_id=content_version_id.strip() or None,
                category=category,
            )
        except ValueError as exc:
            return RedirectResponse(
                _append_query(destination, error=str(exc)),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            _append_query(destination, saved="true"),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/works/{work_id}/archive/entries/{entry_id}/adopt")
    async def adopt_work_archive_note(
        request: Request,
        work_id: str,
        entry_id: str,
        category: str = Form(...),
        title: str = Form(""),
        content: str = Form(""),
        return_to: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        work = database.get_work(user_id, work_id)
        if not work:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        fallback = _work_archive_destination(work)
        destination = _safe_next(return_to, fallback)
        try:
            clean_title = _clean_field(title, "标题", max_length=120)
            clean_content = _clean_field(content, "创作设定", max_length=20_000)
            database.adopt_work_archive_entry(
                user_id=user_id,
                work_id=work_id,
                entry_id=entry_id,
                category=category,
                title=clean_title,
                content=clean_content,
            )
        except ValueError as exc:
            return RedirectResponse(
                _append_query(destination, error=str(exc)),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            _append_query(destination, adopted="true"),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/works/{work_id}/archive/analyses/{analysis_id}/adopt")
    async def adopt_work_chapter_analysis(
        request: Request,
        work_id: str,
        analysis_id: str,
        category: str = Form(...),
        title: str = Form(""),
        content: str = Form(...),
        return_to: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        work = database.get_work(user_id, work_id)
        if not work:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        fallback = _work_archive_destination(work)
        destination = _safe_next(return_to, fallback)
        try:
            clean_title = _clean_field(title, "标题", max_length=120)
            clean_content = _clean_field(
                content,
                "创作设定",
                max_length=20_000,
                required=True,
            )
            database.adopt_work_analysis(
                user_id=user_id,
                work_id=work_id,
                analysis_id=analysis_id,
                category=category,
                title=clean_title,
                content=clean_content,
            )
        except ValueError as exc:
            return RedirectResponse(
                _append_query(destination, error=str(exc)),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            _append_query(destination, adopted="true"),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/works/{work_id}/archive/entries/{entry_id}/delete")
    async def delete_work_archive_note(
        request: Request,
        work_id: str,
        entry_id: str,
        return_to: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        work = database.get_work(user_id, work_id)
        if not work:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        fallback = _work_archive_destination(work)
        destination = _safe_next(return_to, fallback)
        if not database.delete_work_archive_entry(
            user_id=user_id,
            work_id=work_id,
            entry_id=entry_id,
        ):
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        return RedirectResponse(
            _append_query(destination, removed="true"),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/works/{work_id}/delete")
    async def delete_work(
        request: Request,
        work_id: str,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        try:
            deleted = database.delete_work(user_id=user_id, work_id=work_id)
        except ValueError as exc:
            return RedirectResponse(
                "/dashboard?error=" + quote(str(exc)),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        if deleted is None:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        cleanup_paths = [
            app_settings.novels_dir / str(user_id) / project_id
            for project_id in deleted["project_ids"]
        ]
        cleanup_paths.extend(Path(path) for path in deleted["document_paths"])
        cleanup_failed = False
        for path in cleanup_paths:
            try:
                if path.exists():
                    shutil.rmtree(path)
            except OSError:
                cleanup_failed = True
                logger.exception(
                    "work database rows deleted but files remain path=%s",
                    path,
                )
        if cleanup_failed:
            return RedirectResponse(
                "/dashboard?error=" + quote("作品已删除，但部分本地文件清理失败"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            "/dashboard?deleted=true", status_code=status.HTTP_303_SEE_OTHER
        )

    @application.post("/works/{work_id}/main")
    async def create_work_main(
        request: Request,
        work_id: str,
        base_version_id: str = Form(...),
        intent: str = Form(...),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        work = database.get_work(user_id, work_id)
        version = database.get_work_version(user_id, base_version_id)
        if (
            not work
            or not version
            or str(version["work_id"]) != work_id
            or not version.get("document_id")
        ):
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        try:
            project_id = create_main_from_version(
                database=database,
                novels_dir=app_settings.novels_dir,
                user_id=user_id,
                document_id=str(version["document_id"]),
                intent=intent,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/documents/{version['document_id']}?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        except Exception:
            logger.exception("failed to create main branch")
            return RedirectResponse(
                f"/documents/{version['document_id']}"
                f"?error={quote('创建 main 分支失败')}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{project_id}/workbench",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/works/{work_id}/tags")
    async def create_work_tag(
        request: Request,
        work_id: str,
        label: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        work = database.get_work(user_id, work_id)
        if not work or not work.get("main_version"):
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        project_id = str(work["main_version"]["project_id"])
        try:
            document_id = create_version_tag(
                database=database,
                documents_dir=app_settings.documents_dir,
                user_id=user_id,
                project_id=project_id,
                label=label,
                max_documents=app_settings.max_documents_per_user,
                max_stored_chars=app_settings.max_stored_chars_per_user,
            )
        except ValueError as exc:
            return RedirectResponse(
                _workbench_path(
                    project_id,
                    archive_tab="versions",
                    error=str(exc),
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        except Exception:
            logger.exception("failed to create version tag")
            return RedirectResponse(
                _workbench_path(
                    project_id,
                    archive_tab="versions",
                    error="创建固定版本失败",
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/documents/{document_id}?view=archive&archive_tab=versions",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/works/{work_id}/versions/{version_id}/delete")
    async def delete_work_tag(
        request: Request,
        work_id: str,
        version_id: str,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        work = database.get_work(user_id, work_id)
        if not work:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        main_version = work.get("main_version")
        error_destination = (
            _workbench_path(
                str(main_version["project_id"]),
                archive_tab="versions",
            )
            if main_version
            else _work_archive_destination(work)
        )
        try:
            deleted = database.delete_work_tag(
                user_id=user_id,
                work_id=work_id,
                version_id=version_id,
            )
        except ValueError as exc:
            return RedirectResponse(
                _append_query(error_destination, error=str(exc)),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        if deleted is None:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        document_path = Path(str(deleted["document_path"]))
        cleanup_error = None
        try:
            if document_path.exists():
                shutil.rmtree(document_path)
        except OSError:
            logger.exception(
                "tag database rows deleted but files remain path=%s",
                document_path,
            )
            cleanup_error = "Tag 已删除，但部分本地文件清理失败"
        if deleted["fallback_project_id"]:
            destination = _workbench_path(
                str(deleted["fallback_project_id"]),
                archive_tab="versions",
                removed="true" if cleanup_error is None else None,
                error=cleanup_error,
            )
        else:
            destination = _document_workbench_path(
                str(deleted["fallback_document_id"]),
                view="archive",
                archive_tab="versions",
                removed="true" if cleanup_error is None else None,
                error=cleanup_error,
            )
        return RedirectResponse(
            destination,
            status_code=status.HTTP_303_SEE_OTHER,
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
        archive_tab: str = "creative",
        settings_tab: str = "core",
        onboarding: bool = False,
        error: Optional[str] = None,
        saved: bool = False,
        adopted: bool = False,
        removed: bool = False,
        sent: bool = False,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        user_id = int(user["id"])
        try:
            context = workbench_view_builder.build_novel(
                user_id=user_id,
                project_id=project_id,
                chapter_id=chapter_id,
                conversation_id=conversation_id,
                view=view,
                archive_tab=archive_tab,
                settings_tab=settings_tab,
            )
        except WorkbenchNotFound:
            return render_template(
                "not_found.html",
                _template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        except WorkbenchUnavailable as exc:
            return Response(
                str(exc),
                status_code=status.HTTP_409_CONFLICT,
            )

        active_conversation = context["active_conversation"]
        available_chat_models = chat_model_groups(user_id)
        context.update(
            archive_categories=WORK_ARCHIVE_CATEGORIES,
            analysis_categories=WORK_ANALYSIS_CATEGORIES,
            setting_tabs=WORKBENCH_SETTING_TABS,
            model_groups=available_chat_models,
            selected_model_choice=selected_chat_model(
                available_chat_models,
                active_conversation,
            ),
            quality_modes=quality_mode_options(user_id),
            selected_quality_mode=selected_quality_mode(
                user_id,
                active_conversation,
            ),
            pov_options=POV_OPTIONS,
            world_entry_type_options=WORLD_ENTRY_TYPE_OPTIONS,
            story_arc_type_options=STORY_ARC_TYPE_OPTIONS,
            setting_field_labels=SETTING_FIELD_LABELS,
            onboarding=onboarding,
            archive_saved=saved,
            archive_adopted=adopted,
            archive_removed=removed,
            archive_error=error,
            error=error,
            saved=saved,
            sent=sent,
        )
        return render_template(
            "novel_workbench.html",
            _template_context(request, user=user, **context),
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
                author_note=_clean_field(author_note, "作者备注", max_length=2000),
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

    @application.get("/techniques/{technique_id}", response_class=HTMLResponse)
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
                author_note=_clean_field(author_note, "作者备注", max_length=2000),
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
        project_dir = app_settings.novels_dir / str(user["id"]) / project_id
        try:
            (project_dir / "chapters").mkdir(parents=True, exist_ok=False, mode=0o700)
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
                "/dashboard?error=" + quote("创建空白作品失败，请稍后重试"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{project_id}/workbench"
            "?view=archive&archive_tab=creative&onboarding=true",
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
        project_dir = app_settings.novels_dir / str(user_id) / project_id
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
        return RedirectResponse("/dashboard", status_code=status.HTTP_303_SEE_OTHER)

    @application.post("/novels/new")
    async def create_novel(
        request: Request,
        title: str = Form(...),
        genre: str = Form(""),
        premise: str = Form(...),
        theme: str = Form(""),
        story_promise: str = Form(""),
        target_audience: str = Form(""),
        core_appeal: str = Form(""),
        ending_constraint: str = Form(""),
        world_setting: str = Form(""),
        style_guide: str = Form(""),
        point_of_view: str = Form("第三人称限知"),
        target_chapter_chars: int = Form(3000),
        planning_horizon: int = Form(20),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            clean_title = _clean_field(title, "书名", max_length=120, required=True)
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
            clean_world = _clean_field(world_setting, "世界设定", max_length=20_000)
            clean_style = _clean_field(style_guide, "文风要求", max_length=10_000)
            clean_theme = _clean_field(theme, "主题", max_length=2000)
            clean_promise = _clean_field(story_promise, "作品承诺", max_length=4000)
            clean_audience = _clean_field(target_audience, "目标读者", max_length=1000)
            clean_appeal = _clean_field(core_appeal, "核心吸引力", max_length=4000)
            clean_ending = _clean_field(ending_constraint, "结局约束", max_length=4000)
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
        project_dir = app_settings.novels_dir / str(user["id"]) / project_id
        try:
            (project_dir / "chapters").mkdir(parents=True, exist_ok=False, mode=0o700)
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
                theme=clean_theme,
                story_promise=clean_promise,
                target_audience=clean_audience,
                core_appeal=clean_appeal,
                ending_constraint=clean_ending,
                planning_horizon=planning_horizon,
                ai_instructions="",
            )
        except Exception:
            shutil.rmtree(project_dir, ignore_errors=True)
            logger.exception("failed to create novel project")
            return RedirectResponse(
                "/dashboard?error=" + quote("创建小说项目失败，请稍后重试"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{project_id}/workbench",
            status_code=status.HTTP_303_SEE_OTHER,
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
                _workbench_path(
                    project_id,
                    settings_tab="structure",
                    error=str(exc),
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        profile = api_profile(int(user["id"]))
        if not profile:
            return RedirectResponse(
                "/settings/api?error=" + quote("生成全书方案前，请先配置模型服务"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        try:
            suggestion_id = story_plan_suggestion_service.create_suggestion(
                user_id=int(user["id"]),
                project_id=project_id,
                planning_mode=planning_mode,
                instruction=clean_instruction,
                provider=profile["provider"],
                model=profile["model"],
                credential_source=profile["credential_source"],
            )
        except ValueError as exc:
            return RedirectResponse(
                _workbench_path(
                    project_id,
                    settings_tab="structure",
                    error=str(exc),
                ),
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
    async def story_plan_suggestion_status(request: Request, suggestion_id: str):
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

    @application.post("/story-plan-suggestions/{suggestion_id}/apply")
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
                f"/story-plan-suggestions/{suggestion_id}?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            _workbench_path(
                str(applied["project_id"]),
                settings_tab="structure",
                saved="true",
            ),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/novels/{project_id}/story-structure-suggestions")
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
                _workbench_path(
                    project_id,
                    settings_tab="structure",
                    error=str(exc),
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        profile = api_profile(int(user["id"]))
        if not profile:
            return RedirectResponse(
                "/settings/api?error="
                + quote("生成分卷与滚动章节骨架前，请先配置模型服务"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        try:
            suggestion_id = story_structure_suggestion_service.create_suggestion(
                user_id=int(user["id"]),
                project_id=project_id,
                chapter_count=chapter_count,
                instruction=clean_instruction,
                provider=profile["provider"],
                model=profile["model"],
                credential_source=profile["credential_source"],
            )
        except ValueError as exc:
            return RedirectResponse(
                _workbench_path(
                    project_id,
                    settings_tab="structure",
                    error=str(exc),
                ),
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

    @application.get("/api/story-structure-suggestions/{suggestion_id}")
    async def story_structure_suggestion_status(request: Request, suggestion_id: str):
        user = _current_user(request)
        if not user:
            return JSONResponse(
                {"detail": "未登录"},
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        current_status = story_structure_suggestion_service.get_status(
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

    @application.post("/story-structure-suggestions/{suggestion_id}/apply")
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
            applied_result = story_structure_suggestion_service.apply_suggestion(
                user_id=int(user["id"]),
                suggestion_id=suggestion_id,
                option_index=option_index,
                preview_fingerprint=preview_fingerprint,
            )
        except ValueError as exc:
            return RedirectResponse(
                f"/story-structure-suggestions/{suggestion_id}?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            _workbench_path(
                str(applied_result["project_id"]),
                settings_tab="structure",
                saved="true",
            ),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/story-structure-applications/{application_id}/revert")
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
            reverted_result = story_structure_suggestion_service.revert_application(
                user_id=int(user["id"]),
                application_id=application_id,
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

    @application.post("/novels/{project_id}/causal-link-suggestions")
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
                _workbench_path(
                    project_id,
                    settings_tab="structure",
                    error=str(exc),
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        profile = api_profile(int(user["id"]))
        if not profile:
            return RedirectResponse(
                "/settings/api?error=" + quote("生成因果建议前，请先配置模型服务"),
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
            )
        except ValueError as exc:
            return RedirectResponse(
                _workbench_path(
                    project_id,
                    settings_tab="structure",
                    error=str(exc),
                ),
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
                proposal["branch_simulations"] = simulations_by_proposal.get(
                    int(proposal["proposal_index"]),
                    [],
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

    @application.get("/api/causal-link-suggestions/{suggestion_id}")
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
                "/settings/api?error=" + quote("生成长期因果分支前，请先配置模型服务"),
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

    @application.get("/api/causal-branch-simulations/{simulation_id}")
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
        "/causal-branch-simulations/{simulation_id}/branches/{branch_key}/adoptions"
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
            f"/causal-branch-adoptions/{adoption_id}?adoption_created=true",
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
            ),
        )

    @application.post("/causal-branch-adoptions/{adoption_id}/items/{item_id}")
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
            f"/causal-branch-adoptions/{adoption_id}?saved=true#item-{item_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/causal-branch-adoptions/{adoption_id}/apply")
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
            f"&reset_task_cards={result['reset_task_card_count']}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/causal-branch-adoptions/{adoption_id}/abandon")
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
            "/causal-branch-simulations/" + str(adoption["simulation_id"]),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/causal-branch-adoptions/{adoption_id}/revert")
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
        "/causal-link-suggestions/{suggestion_id}/proposals/{proposal_index}/accept"
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
                raise ValueError("请先确认已核对起因、结果和正史边界")
            result = causal_suggestion_service.accept_proposal(
                user_id=int(user["id"]),
                suggestion_id=suggestion_id,
                proposal_index=proposal_index,
                cause_text=cause_text,
                effect_text=effect_text,
                author_note=author_note,
                comparison_confirmed=confirm_comparison == "yes",
                semantic_review_confirmed=(confirm_semantic_review == "yes"),
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
        "/causal-link-suggestions/{suggestion_id}/proposals/{proposal_index}/dismiss"
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
            structure_link_service.create_link(
                user_id=int(user["id"]),
                project_id=project_id,
                source_chapter_id=str(form.get("source_chapter_id") or "").strip(),
                target_chapter_id=str(form.get("target_chapter_id") or "").strip(),
                relation_type=str(form.get("relation_type") or "").strip(),
                cause_text=str(form.get("cause_text") or ""),
                effect_text=str(form.get("effect_text") or ""),
                author_note=str(form.get("author_note") or ""),
            )
        except ValueError as exc:
            return RedirectResponse(
                _workbench_path(
                    project_id,
                    settings_tab="structure",
                    error=str(exc)[:1000],
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            _workbench_path(
                project_id,
                settings_tab="structure",
                saved="true",
            ),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/novels/{project_id}/structure-links/{link_id}/archive")
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
            structure_link_service.archive_link(
                user_id=int(user["id"]),
                project_id=project_id,
                link_id=link_id,
            )
        except ValueError as exc:
            return RedirectResponse(
                _workbench_path(
                    project_id,
                    settings_tab="structure",
                    error=str(exc)[:1000],
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            _workbench_path(
                project_id,
                settings_tab="structure",
                saved="true",
            ),
            status_code=status.HTTP_303_SEE_OTHER,
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
                _workbench_path(
                    project_id,
                    settings_tab="world",
                    error=str(exc),
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            _workbench_path(
                project_id,
                settings_tab="world",
                saved="true",
            ),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/novels/{project_id}/memory-identities/{identity_id}/delete")
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
            _workbench_path(
                project_id,
                settings_tab="world",
                saved="true",
            ),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/novels/{project_id}/continuity/issues/{issue_id}")
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
                _workbench_path(
                    project_id,
                    settings_tab="world",
                    error=str(exc),
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            _workbench_path(
                project_id,
                settings_tab="world",
                saved="true",
            ),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/novels/{project_id}/settings")
    async def update_novel_settings(
        request: Request,
        project_id: str,
        title: str = Form(""),
        genre: str = Form(""),
        premise: str = Form(""),
        theme: str = Form(""),
        story_promise: str = Form(""),
        target_audience: str = Form(""),
        core_appeal: str = Form(""),
        ending_constraint: str = Form(""),
        world_setting: str = Form(""),
        style_guide: str = Form(""),
        point_of_view: str = Form("第三人称限知"),
        settings_tab: str = Form("core"),
        return_to: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        clean_settings_tab = (
            settings_tab if settings_tab in WORKBENCH_SETTING_TAB_KEYS else "core"
        )
        destination = (
            _append_query(
                _safe_next(return_to, "/dashboard"),
                settings_tab=clean_settings_tab,
            )
            if return_to.strip()
            else _workbench_path(
                project_id,
                settings_tab=clean_settings_tab,
            )
        )
        try:
            current = database.get_novel_project(int(user["id"]), project_id)
            if not current:
                return Response(status_code=status.HTTP_404_NOT_FOUND)
            values = {
                "title": str(current.get("title") or ""),
                "genre": str(current.get("genre") or ""),
                "premise": str(current.get("premise") or ""),
                "theme": str(current.get("theme") or ""),
                "story_promise": str(current.get("story_promise") or ""),
                "target_audience": str(current.get("target_audience") or ""),
                "core_appeal": str(current.get("core_appeal") or ""),
                "ending_constraint": str(current.get("ending_constraint") or ""),
                "world_setting": str(current.get("world_setting") or ""),
                "style_guide": str(current.get("style_guide") or ""),
                "point_of_view": str(current.get("point_of_view") or "第三人称限知"),
            }
            if clean_settings_tab == "core":
                values.update(
                    title=_clean_field(title, "书名", max_length=120),
                    genre=_clean_field(genre, "题材", max_length=80),
                    premise=_clean_field(premise, "一句话故事", max_length=4000),
                    theme=_clean_field(theme, "主题", max_length=2000),
                    story_promise=_clean_field(
                        story_promise, "读者体验", max_length=4000
                    ),
                    target_audience=_clean_field(
                        target_audience, "目标读者", max_length=1000
                    ),
                    core_appeal=_clean_field(
                        core_appeal, "核心吸引力", max_length=4000
                    ),
                )
            elif clean_settings_tab == "world":
                values["world_setting"] = _clean_field(
                    world_setting, "世界概述", max_length=20_000
                )
            elif clean_settings_tab == "structure":
                values["ending_constraint"] = _clean_field(
                    ending_constraint, "结局约束", max_length=4000
                )
            elif clean_settings_tab == "style":
                if point_of_view not in POV_OPTIONS:
                    raise ValueError("请选择有效的叙事视角")
                values["point_of_view"] = point_of_view
                values["style_guide"] = _clean_field(
                    style_guide, "叙事风格规范", max_length=10_000
                )
            updated = database.update_novel_project(
                user_id=int(user["id"]),
                project_id=project_id,
                title=values["title"],
                genre=values["genre"],
                premise=values["premise"],
                theme=values["theme"],
                world_setting=values["world_setting"],
                style_guide=values["style_guide"],
                point_of_view=values["point_of_view"],
                target_chapter_chars=int(current.get("target_chapter_chars") or 3000),
                story_promise=values["story_promise"],
                target_audience=values["target_audience"],
                core_appeal=values["core_appeal"],
                ending_constraint=values["ending_constraint"],
                planning_horizon=int(current.get("planning_horizon") or 20),
                ai_instructions="",
            )
            if not updated:
                return Response(status_code=status.HTTP_404_NOT_FOUND)
        except ValueError as exc:
            return RedirectResponse(
                _append_query(destination, error=str(exc)),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            _append_query(destination, saved="true"),
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
                _workbench_path(
                    project_id,
                    settings_tab="structure",
                    error=str(exc),
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            _workbench_path(
                project_id,
                settings_tab="structure",
                saved="true",
            ),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post(
        "/novels/{project_id}/story-blueprint/versions/{version_id}/restore"
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
                _workbench_path(
                    project_id,
                    settings_tab="structure",
                    error=str(exc),
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            _workbench_path(
                project_id,
                settings_tab="structure",
                saved="true",
            ),
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
                _workbench_path(
                    project_id,
                    settings_tab="structure",
                    error=str(exc),
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            _workbench_path(
                project_id,
                settings_tab="structure",
                saved="true",
            ),
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
                _workbench_path(
                    project_id,
                    settings_tab="structure",
                    error=str(exc),
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            _workbench_path(
                project_id,
                settings_tab="structure",
                saved="true",
            ),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post(
        "/novels/{project_id}/plot-arcs/{arc_id}/versions/{version_id}/restore"
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
                _workbench_path(
                    project_id,
                    settings_tab="structure",
                    error=str(exc),
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            _workbench_path(
                project_id,
                settings_tab="structure",
                saved="true",
            ),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/novels/{project_id}/plot-arcs/{arc_id}/archive")
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
                _workbench_path(
                    project_id,
                    settings_tab="structure",
                    error=str(exc),
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            _workbench_path(
                project_id,
                settings_tab="structure",
                saved="true",
            ),
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
                author_note=_clean_field(author_note, "作者备注", max_length=4000),
            )
        except ValueError as exc:
            return RedirectResponse(
                _workbench_path(
                    project_id,
                    settings_tab="structure",
                    error=str(exc),
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/reader-requests/{request_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.get("/reader-requests/{request_id}", response_class=HTMLResponse)
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
                f"/reader-requests/{request_id}?error=" + quote("这条读者意见已经处理"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        profile = api_profile(int(user["id"]))
        if not profile:
            return RedirectResponse(
                "/settings/api?error=" + quote("生成剧情方案前，请先配置模型服务"),
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
            _workbench_path(
                project_id,
                settings_tab="structure",
            ),
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
                _workbench_path(
                    project_id,
                    settings_tab="style",
                    error=str(exc),
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        profile = api_profile(int(user["id"]))
        if not profile:
            return RedirectResponse(
                "/settings/api?error=" + quote("提取作品声纹前，请先配置模型服务"),
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
            )
        except ValueError as exc:
            return RedirectResponse(
                _workbench_path(
                    project_id,
                    settings_tab="style",
                    error=str(exc),
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        request.app.state.worker.wake()
        return RedirectResponse(
            f"/voice-suggestions/{suggestion_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.get("/voice-suggestions/{suggestion_id}", response_class=HTMLResponse)
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

        def _merge_items(current: list[Any], proposed_items: Any) -> list[str]:
            merged: list[str] = []
            for raw in [*current, *(proposed_items or [])]:
                item = str(raw).strip()
                if item and item not in merged:
                    merged.append(item)
            return merged

        review_profile = {
            "narration_rules": _prefer("narration_rules", "current_narration_rules"),
            "sentence_rhythm": _prefer("sentence_rhythm", "current_sentence_rhythm"),
            "dialogue_voice": _prefer("dialogue_voice", "current_dialogue_voice"),
            "sensory_palette": _prefer("sensory_palette", "current_sensory_palette"),
            "metaphor_policy": _prefer("metaphor_policy", "current_metaphor_policy"),
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
            "author_notes": str(suggestion.get("current_author_notes") or ""),
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
    async def voice_profile_suggestion_status(request: Request, suggestion_id: str):
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
                "preferred_patterns": _split_lines(preferred_patterns, limit=50),
                "banned_expressions": _split_lines(banned_expressions, limit=100),
                "author_notes": _clean_field(author_notes, "作者补充", max_length=6000),
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
                f"/voice-suggestions/{suggestion_id}?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            _workbench_path(
                project_id,
                settings_tab="style",
                saved="true",
            ),
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
            _workbench_path(
                project_id,
                settings_tab="style",
            ),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    def _queue_edit_preference_suggestion(
        *,
        request: Request,
        user_id: int,
        project_id: str,
        chapter_id: str,
        after_version_id: str,
        error_path: str,
    ) -> RedirectResponse:
        profile = api_profile(user_id)
        if not profile:
            return RedirectResponse(
                "/settings/api?error=" + quote("从手工改稿学习前，请先配置模型服务"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        try:
            suggestion_id = preference_service.create_suggestion(
                user_id=user_id,
                project_id=project_id,
                chapter_id=chapter_id,
                after_version_id=after_version_id,
                provider=profile["provider"],
                model=profile["model"],
                credential_source=profile["credential_source"],
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
            after_version_id=version_id,
            error_path=_workbench_path(project_id, chapter_id=chapter_id),
        )

    @application.post("/novels/{project_id}/editing-preference-aggregates")
    async def create_editing_preference_aggregate(request: Request, project_id: str):
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
                _workbench_path(
                    project_id,
                    settings_tab="style",
                    error=str(exc),
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            _workbench_path(
                project_id,
                settings_tab="style",
                saved="true",
            ),
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

    @application.get("/api/editing-preference-suggestions/{suggestion_id}")
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

    @application.post("/editing-preference-suggestions/{suggestion_id}/apply")
    async def apply_editing_preference_suggestion(request: Request, suggestion_id: str):
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
                            str(form.get(f"applicability_{index}") or ""),
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
            _workbench_path(
                project_id,
                settings_tab="style",
                saved="true",
            ),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/editing-preference-suggestions/{suggestion_id}/reject")
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
            _workbench_path(
                project_id,
                settings_tab="style",
                saved="true",
            ),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/editing-preference-aggregates/{aggregate_id}/archive")
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
            _workbench_path(
                project_id,
                settings_tab="style",
                saved="true",
            ),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/novels/{project_id}/voice")
    async def update_novel_voice_profile(
        request: Request,
        project_id: str,
        point_of_view: str = Form("第三人称限知"),
        style_guide: str = Form(""),
        narrative_tense: str = Form(""),
        narrative_distance: str = Form(""),
        tone: str = Form(""),
        narration_rules: str = Form(""),
        sentence_rhythm: str = Form(""),
        dialogue_voice: str = Form(""),
        sensory_palette: str = Form(""),
        metaphor_policy: str = Form(""),
        allowed_omissions: str = Form(""),
        preferred_patterns: str = Form(""),
        banned_expressions: str = Form(""),
        style_examples: str = Form(""),
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
            if point_of_view not in POV_OPTIONS:
                raise ValueError("请选择有效的叙事视角")
            fields = {
                "narrative_tense": _clean_field(
                    narrative_tense, "叙事时态", max_length=200
                ),
                "narrative_distance": _clean_field(
                    narrative_distance, "叙事距离", max_length=1000
                ),
                "tone": _clean_field(tone, "整体基调", max_length=1000),
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
                "preferred_patterns": _split_lines(preferred_patterns, limit=50),
                "banned_expressions": _split_lines(banned_expressions, limit=100),
                "style_examples": _split_lines(style_examples, limit=30),
                "author_notes": _clean_field(author_notes, "作者补充", max_length=6000),
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
            project = database.get_novel_project(int(user["id"]), project_id)
            if not project:
                return Response(status_code=status.HTTP_404_NOT_FOUND)
            database.update_novel_project(
                user_id=int(user["id"]),
                project_id=project_id,
                title=str(project.get("title") or ""),
                genre=str(project.get("genre") or ""),
                premise=str(project.get("premise") or ""),
                theme=str(project.get("theme") or ""),
                world_setting=str(project.get("world_setting") or ""),
                style_guide=_clean_field(
                    style_guide, "叙事风格规范", max_length=10_000
                ),
                point_of_view=point_of_view,
                target_chapter_chars=int(project.get("target_chapter_chars") or 3000),
                story_promise=str(project.get("story_promise") or ""),
                target_audience=str(project.get("target_audience") or ""),
                core_appeal=str(project.get("core_appeal") or ""),
                ending_constraint=str(project.get("ending_constraint") or ""),
                planning_horizon=int(project.get("planning_horizon") or 20),
                ai_instructions="",
            )
        except ValueError as exc:
            return RedirectResponse(
                _workbench_path(
                    project_id,
                    settings_tab="style",
                    error=str(exc),
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            _workbench_path(
                project_id,
                settings_tab="style",
                saved="true",
            ),
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
                title=_clean_field(title, "分卷名", max_length=120, required=True),
                goal=_clean_field(goal, "分卷目标", max_length=4000),
                start_state=_clean_field(start_state, "开卷状态", max_length=4000),
                end_state=_clean_field(end_state, "收卷状态", max_length=4000),
                major_conflict=_clean_field(
                    major_conflict, "主要冲突", max_length=4000
                ),
                payoff=_clean_field(payoff, "本卷回报", max_length=4000),
            )
        except ValueError as exc:
            return RedirectResponse(
                _workbench_path(
                    project_id,
                    settings_tab="structure",
                    error=str(exc),
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            _workbench_path(
                project_id,
                settings_tab="structure",
                saved="true",
            ),
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
                title=_clean_field(title, "分卷名", max_length=120, required=True),
                goal=_clean_field(goal, "分卷目标", max_length=4000),
                start_state=_clean_field(start_state, "开卷状态", max_length=4000),
                end_state=_clean_field(end_state, "收卷状态", max_length=4000),
                major_conflict=_clean_field(
                    major_conflict, "主要冲突", max_length=4000
                ),
                payoff=_clean_field(payoff, "本卷回报", max_length=4000),
            )
        except ValueError as exc:
            return RedirectResponse(
                _workbench_path(
                    project_id,
                    settings_tab="structure",
                    error=str(exc),
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            _workbench_path(
                project_id,
                settings_tab="structure",
                saved="true",
            ),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/novels/{project_id}/world-entries")
    async def add_world_entry(
        request: Request,
        project_id: str,
        entry_type: str = Form("background"),
        name: str = Form(...),
        description: str = Form(""),
        constraints: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            database.add_world_entry(
                user_id=int(user["id"]),
                project_id=project_id,
                entry_type=entry_type,
                name=_clean_field(name, "资料名称", max_length=120, required=True),
                description=_clean_field(description, "资料内容", max_length=6000),
                constraints=_clean_field(constraints, "规则与边界", max_length=4000),
            )
        except Exception as exc:
            duplicate = "UNIQUE constraint failed" in str(exc)
            if not isinstance(exc, ValueError) and not duplicate:
                logger.exception("failed to add world entry")
            message = (
                "同类世界资料中已经有这个名称"
                if duplicate
                else str(exc)
                if isinstance(exc, ValueError)
                else "添加世界资料失败"
            )
            return RedirectResponse(
                _workbench_path(project_id, settings_tab="world", error=message),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            _workbench_path(project_id, settings_tab="world", saved="true"),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/novels/{project_id}/world-entries/{entry_id}/edit")
    async def edit_world_entry(
        request: Request,
        project_id: str,
        entry_id: str,
        entry_type: str = Form("background"),
        name: str = Form(...),
        description: str = Form(""),
        constraints: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            updated = database.update_world_entry(
                user_id=int(user["id"]),
                project_id=project_id,
                entry_id=entry_id,
                entry_type=entry_type,
                name=_clean_field(name, "资料名称", max_length=120, required=True),
                description=_clean_field(description, "资料内容", max_length=6000),
                constraints=_clean_field(constraints, "规则与边界", max_length=4000),
            )
            if not updated:
                return Response(status_code=status.HTTP_404_NOT_FOUND)
        except Exception as exc:
            duplicate = "UNIQUE constraint failed" in str(exc)
            message = (
                "同类世界资料中已经有这个名称"
                if duplicate
                else str(exc)
                if isinstance(exc, ValueError)
                else "保存世界资料失败"
            )
            return RedirectResponse(
                _workbench_path(project_id, settings_tab="world", error=message),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            _workbench_path(project_id, settings_tab="world", saved="true"),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/novels/{project_id}/world-entries/{entry_id}/delete")
    async def remove_world_entry(
        request: Request,
        project_id: str,
        entry_id: str,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        removed = database.delete_world_entry(int(user["id"]), project_id, entry_id)
        if not removed:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        return RedirectResponse(
            _workbench_path(project_id, settings_tab="world", saved="true"),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/novels/{project_id}/relationships")
    async def add_character_relationship(
        request: Request,
        project_id: str,
        character_a_id: str = Form(...),
        character_b_id: str = Form(...),
        relationship: str = Form(""),
        tension: str = Form(""),
        change_direction: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            database.add_character_relationship(
                user_id=int(user["id"]),
                project_id=project_id,
                character_a_id=character_a_id,
                character_b_id=character_b_id,
                relationship=_clean_field(relationship, "人物关系", max_length=3000),
                tension=_clean_field(tension, "关系张力", max_length=3000),
                change_direction=_clean_field(
                    change_direction, "变化方向", max_length=3000
                ),
            )
        except Exception as exc:
            duplicate = "UNIQUE constraint failed" in str(exc)
            message = (
                "这两个人物之间已经有一张关系卡"
                if duplicate
                else str(exc)
                if isinstance(exc, ValueError)
                else "添加人物关系失败"
            )
            return RedirectResponse(
                _workbench_path(project_id, settings_tab="characters", error=message),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            _workbench_path(project_id, settings_tab="characters", saved="true"),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/novels/{project_id}/relationships/{relationship_id}/edit")
    async def edit_character_relationship(
        request: Request,
        project_id: str,
        relationship_id: str,
        character_a_id: str = Form(...),
        character_b_id: str = Form(...),
        relationship: str = Form(""),
        tension: str = Form(""),
        change_direction: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            updated = database.update_character_relationship(
                user_id=int(user["id"]),
                project_id=project_id,
                relationship_id=relationship_id,
                character_a_id=character_a_id,
                character_b_id=character_b_id,
                relationship=_clean_field(relationship, "人物关系", max_length=3000),
                tension=_clean_field(tension, "关系张力", max_length=3000),
                change_direction=_clean_field(
                    change_direction, "变化方向", max_length=3000
                ),
            )
            if not updated:
                return Response(status_code=status.HTTP_404_NOT_FOUND)
        except Exception as exc:
            duplicate = "UNIQUE constraint failed" in str(exc)
            message = (
                "这两个人物之间已经有一张关系卡"
                if duplicate
                else str(exc)
                if isinstance(exc, ValueError)
                else "保存人物关系失败"
            )
            return RedirectResponse(
                _workbench_path(project_id, settings_tab="characters", error=message),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            _workbench_path(project_id, settings_tab="characters", saved="true"),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/novels/{project_id}/relationships/{relationship_id}/delete")
    async def remove_character_relationship(
        request: Request,
        project_id: str,
        relationship_id: str,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        removed = database.delete_character_relationship(
            int(user["id"]), project_id, relationship_id
        )
        if not removed:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        return RedirectResponse(
            _workbench_path(project_id, settings_tab="characters", saved="true"),
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
        external_goal: str = Form(""),
        internal_need: str = Form(""),
        central_conflict: str = Form(""),
        secret: str = Form(""),
        speech_style: str = Form(""),
        initial_state: str = Form(""),
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
                name=_clean_field(name, "人物名", max_length=60, required=True),
                role=_clean_field(role, "人物定位", max_length=300),
                traits=_clean_field(traits, "性格特征", max_length=1000),
                background=_clean_field(background, "人物背景", max_length=4000),
                character_arc=_clean_field(character_arc, "人物弧光", max_length=2000),
                external_goal=_clean_field(external_goal, "外在目标", max_length=2000),
                internal_need=_clean_field(internal_need, "内在需求", max_length=2000),
                central_conflict=_clean_field(
                    central_conflict, "人物矛盾", max_length=2000
                ),
                secret=_clean_field(secret, "秘密", max_length=2000),
                speech_style=_clean_field(speech_style, "说话方式", max_length=2000),
                initial_state=_clean_field(initial_state, "初始状态", max_length=2000),
            )
        except ValueError as exc:
            message = str(exc)
            return RedirectResponse(
                _workbench_path(
                    project_id,
                    settings_tab="characters",
                    error=message,
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        except Exception as exc:
            duplicate = "UNIQUE constraint failed" in str(exc)
            if not duplicate:
                logger.exception("failed to add novel character")
            message = "该人物名已经存在" if duplicate else "添加人物失败"
            return RedirectResponse(
                _workbench_path(
                    project_id,
                    settings_tab="characters",
                    error=message,
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            _workbench_path(
                project_id,
                settings_tab="characters",
                saved="true",
            ),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/novels/{project_id}/characters/{character_id}/edit")
    async def edit_novel_character(
        request: Request,
        project_id: str,
        character_id: str,
        name: str = Form(...),
        role: str = Form(""),
        traits: str = Form(""),
        background: str = Form(""),
        external_goal: str = Form(""),
        internal_need: str = Form(""),
        central_conflict: str = Form(""),
        secret: str = Form(""),
        speech_style: str = Form(""),
        initial_state: str = Form(""),
        character_arc: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        try:
            updated = database.update_novel_character(
                user_id=int(user["id"]),
                project_id=project_id,
                character_id=character_id,
                name=_clean_field(name, "人物名", max_length=60, required=True),
                role=_clean_field(role, "人物定位", max_length=300),
                traits=_clean_field(traits, "性格特征", max_length=1000),
                background=_clean_field(background, "人物背景", max_length=4000),
                external_goal=_clean_field(external_goal, "外在目标", max_length=2000),
                internal_need=_clean_field(internal_need, "内在需求", max_length=2000),
                central_conflict=_clean_field(
                    central_conflict, "人物矛盾", max_length=2000
                ),
                secret=_clean_field(secret, "秘密", max_length=2000),
                speech_style=_clean_field(speech_style, "说话方式", max_length=2000),
                initial_state=_clean_field(initial_state, "初始状态", max_length=2000),
                character_arc=_clean_field(character_arc, "人物弧光", max_length=2000),
            )
            if not updated:
                return Response(status_code=status.HTTP_404_NOT_FOUND)
        except Exception as exc:
            duplicate = "UNIQUE constraint failed" in str(exc)
            if not isinstance(exc, ValueError) and not duplicate:
                logger.exception("failed to update novel character")
            message = (
                "该人物名已经存在"
                if duplicate
                else str(exc)
                if isinstance(exc, ValueError)
                else "保存人物失败"
            )
            return RedirectResponse(
                _workbench_path(
                    project_id,
                    settings_tab="characters",
                    error=message,
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            _workbench_path(project_id, settings_tab="characters", saved="true"),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/novels/{project_id}/characters/{character_id}/delete")
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
        database.delete_novel_character(int(user["id"]), project_id, character_id)
        return RedirectResponse(
            _workbench_path(
                project_id,
                settings_tab="characters",
                saved="true",
            ),
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
            clean_title = _clean_field(title, "章节名", max_length=120)
            clean_outline = _clean_field(outline, "章节大纲", max_length=6000)
            clean_key_points = _clean_field(key_points, "关键情节点", max_length=4000)
            (chapter_dir / "versions").mkdir(parents=True, exist_ok=False, mode=0o700)
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
            return RedirectResponse(
                _workbench_path(project_id, error=str(exc)),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        except Exception:
            shutil.rmtree(chapter_dir, ignore_errors=True)
            logger.exception("failed to add novel chapter")
            return RedirectResponse(
                _workbench_path(project_id, error="添加章节失败"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            _workbench_path(project_id, chapter_id=chapter_id),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.get(
        "/novels/{project_id}/chapters/{chapter_id}/history",
        response_class=HTMLResponse,
    )
    async def chapter_version_history(
        request: Request,
        project_id: str,
        chapter_id: str,
        page: int = 1,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        user_id = int(user["id"])
        chapter = database.get_novel_chapter(
            user_id, project_id, chapter_id
        )
        if not chapter:
            return render_template(
                "not_found.html",
                _template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        page_size = 30
        version_count = database.count_chapter_versions(
            user_id, project_id, chapter_id
        )
        page_count = max(1, (version_count + page_size - 1) // page_size)
        current_page = min(max(1, int(page)), page_count)
        versions = database.list_chapter_versions(
            user_id,
            project_id,
            chapter_id,
            limit=page_size,
            offset=(current_page - 1) * page_size,
        )
        return render_template(
            "chapter_version_history.html",
            _template_context(
                request,
                user=user,
                chapter=chapter,
                versions=versions,
                version_count=version_count,
                current_page=current_page,
                page_count=page_count,
            ),
        )

    @application.get(
        "/novels/{project_id}/chapters/{chapter_id}/versions/{version_id}/compare",
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
        chapter = database.get_novel_chapter(user_id, project_id, chapter_id)
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
            user_id, project_id, chapter_id, limit=30
        )
        versions_by_id = {str(item["id"]): item for item in versions}
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
                    str(chapter.get("head_version_id") or "")
                    if str(chapter.get("head_version_id") or "") != version_id
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

        for required_version in (target_version, base_version):
            if not required_version:
                continue
            required_id = str(required_version["id"])
            if required_id not in versions_by_id:
                versions.append(required_version)
                versions_by_id[required_id] = required_version

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
        "/novels/{project_id}/chapters/{chapter_id}/versions/{version_id}/style",
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
        chapter = database.get_novel_chapter(user_id, project_id, chapter_id)
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
                error=error,
            ),
        )

    @application.post(
        "/novels/{project_id}/chapters/{chapter_id}/versions/{version_id}/style"
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
                _workbench_path(
                    project_id,
                    settings_tab="style",
                    error="请先填写并确认作品声纹",
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        api = api_profile(user_id)
        if not api:
            return RedirectResponse(
                "/settings/api?error=" + quote("执行 AI 味审校前，请先配置模型服务"),
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
                "/settings/api?error=" + quote("生成定点改写前，请先配置模型服务"),
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

    @application.get("/style-issues/{issue_id}", response_class=HTMLResponse)
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
            candidate["diff"] = build_version_diff(
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
        result = style_service.ignore_issue(user_id=int(user["id"]), issue_id=issue_id)
        if not result:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        return RedirectResponse(
            f"/novels/{result['project_id']}/chapters/"
            f"{result['chapter_id']}/versions/{result['version_id']}/style",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/style-rewrite-candidates/{candidate_id}/accept")
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
            source_content, actual_hash = _read_utf8_with_file_hash(source_path)
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
            source_content[:start_offset] + replacement + source_content[end_offset:]
        )
        if len(revised_content) > 200_000:
            return RedirectResponse(
                f"/style-issues/{candidate['issue_id']}?error="
                + quote("定点改写后的单章正文超过 200000 字"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        token = secrets.token_hex(16)
        version_path = source_path.parent / f"style-{token}.txt"
        try:
            _atomic_write_text(version_path, revised_content, token)
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
            try:
                _atomic_write_text(
                    working_path,
                    revised_content,
                    secrets.token_hex(16),
                )
            except Exception:
                logger.warning(
                    "failed to refresh non-authoritative chapter cache",
                    exc_info=True,
                )
            queue_background_memory(
                request,
                user_id=user_id,
                project_id=str(result["project_id"]),
                chapter_id=str(result["chapter_id"]),
                version_id=str(result["version_id"]),
            )
        except ValueError as exc:
            version_path.unlink(missing_ok=True)
            return RedirectResponse(
                f"/style-issues/{candidate['issue_id']}?error=" + quote(str(exc)),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        except Exception:
            version_path.unlink(missing_ok=True)
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

    @application.post(
        "/novels/{project_id}/chapters/{chapter_id}/versions/{version_id}/restore"
    )
    async def restore_novel_chapter_version(
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
        chapter = database.get_novel_chapter(user_id, project_id, chapter_id)
        version = database.get_chapter_version(
            user_id, project_id, chapter_id, version_id
        )
        if not chapter or not version:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        if str(chapter.get("head_version_id") or "") == version_id:
            return RedirectResponse(
                _workbench_path(
                    project_id,
                    chapter_id=chapter_id,
                    error="这个版本已经是 main HEAD",
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        version_path = Path(str(version["content_path"]))
        content_path = Path(str(chapter["content_path"]))
        try:
            candidate_content, actual_hash = _read_utf8_with_file_hash(version_path)
        except (OSError, UnicodeError):
            return RedirectResponse(
                _workbench_path(
                    project_id,
                    chapter_id=chapter_id,
                    error="这个版本的正文文件无法读取",
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        expected_hash = str(version["content_hash"] or "")
        if expected_hash and expected_hash != actual_hash:
            return RedirectResponse(
                _workbench_path(
                    project_id,
                    chapter_id=chapter_id,
                    error="历史版本文件校验失败，未恢复",
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        write_token = secrets.token_hex(16)
        restored_path = (
            content_path.parent / "versions" / f"restore-{write_token}.txt"
        )
        try:
            _atomic_write_text(restored_path, candidate_content, write_token)
            restored_version_id = database.record_manual_chapter_version(
                user_id=user_id,
                project_id=project_id,
                chapter_id=chapter_id,
                version_path=restored_path,
                char_count=len(candidate_content),
                effective_char_count=effective_char_count(candidate_content),
                content_hash=hashlib.sha256(
                    candidate_content.encode("utf-8")
                ).hexdigest(),
                change_summary=(
                    f"从历史版本 {version_id} 恢复为新的 main HEAD"
                ),
                kind="history_restore",
                expected_old_head_version_id=str(
                    chapter.get("head_version_id") or ""
                ),
            )
            if not restored_version_id:
                raise ValueError("章节版本不存在")
            try:
                _atomic_write_text(
                    content_path,
                    candidate_content,
                    secrets.token_hex(16),
                )
            except Exception:
                logger.warning(
                    "failed to refresh non-authoritative chapter cache",
                    exc_info=True,
                )
        except ValueError as exc:
            restored_path.unlink(missing_ok=True)
            return RedirectResponse(
                _workbench_path(
                    project_id,
                    chapter_id=chapter_id,
                    error=str(exc),
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        except Exception:
            restored_path.unlink(missing_ok=True)
            logger.exception("failed to restore historical chapter version")
            return RedirectResponse(
                _workbench_path(
                    project_id,
                    chapter_id=chapter_id,
                    error="恢复历史版本失败",
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )

        queue_background_memory(
            request,
            user_id=user_id,
            project_id=project_id,
            chapter_id=chapter_id,
            version_id=str(restored_version_id),
        )
        return RedirectResponse(
            _workbench_path(
                project_id,
                chapter_id=chapter_id,
                restored="true",
            ),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/novels/{project_id}/chapters/{chapter_id}/buffer")
    async def buffer_novel_chapter(
        request: Request,
        project_id: str,
        chapter_id: str,
        content: str = Form(""),
        expected_head_version_id: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return JSONResponse(
                {"error": "登录状态已失效"},
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        verify_csrf(request, csrf)
        if len(content) > app_settings.chapter_edit_buffer_max_chars:
            return JSONResponse(
                {
                    "error": "单章暂存稿不能超过 "
                    f"{app_settings.chapter_edit_buffer_max_chars:,} 字"
                },
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        try:
            result = database.save_chapter_edit_buffer(
                user_id=int(user["id"]),
                project_id=project_id,
                chapter_id=chapter_id,
                base_version_id=expected_head_version_id,
                content=content,
                content_hash=hashlib.sha256(
                    content.encode("utf-8")
                ).hexdigest(),
                max_chapter_chars=app_settings.chapter_edit_buffer_max_chars,
                max_user_chars=app_settings.max_edit_buffer_chars_per_user,
                retention_days=app_settings.edit_buffer_retention_days,
            )
        except ChapterHeadConflict as exc:
            return JSONResponse(
                {"error": str(exc), "conflict": True},
                status_code=status.HTTP_409_CONFLICT,
            )
        except ValueError as exc:
            return JSONResponse(
                {"error": str(exc)},
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        return JSONResponse(result, headers={"Cache-Control": "no-store"})

    @application.post(
        "/novels/{project_id}/chapters/{chapter_id}/buffer/rebase"
    )
    async def rebase_novel_chapter_buffer(
        request: Request,
        project_id: str,
        chapter_id: str,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        if not database.rebase_chapter_edit_buffer(
            user_id=int(user["id"]),
            project_id=project_id,
            chapter_id=chapter_id,
        ):
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        return RedirectResponse(
            _workbench_path(project_id, chapter_id=chapter_id),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post(
        "/novels/{project_id}/chapters/{chapter_id}/buffer/discard"
    )
    async def discard_novel_chapter_buffer(
        request: Request,
        project_id: str,
        chapter_id: str,
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        if not database.delete_chapter_edit_buffer(
            user_id=int(user["id"]),
            project_id=project_id,
            chapter_id=chapter_id,
        ):
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        return RedirectResponse(
            _workbench_path(project_id, chapter_id=chapter_id),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/novels/{project_id}/chapters/{chapter_id}/save")
    async def save_novel_chapter(
        request: Request,
        project_id: str,
        chapter_id: str,
        content: str = Form(""),
        change_summary: str = Form(""),
        expected_head_version_id: Optional[str] = Form(None),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        verify_csrf(request, csrf)
        chapter = database.get_novel_chapter(int(user["id"]), project_id, chapter_id)
        if not chapter:
            return Response(status_code=status.HTTP_404_NOT_FOUND)

        wants_json = (
            request.headers.get("X-Requested-With", "").lower()
            == "xmlhttprequest"
        )

        def save_error_response(
            message: str,
            *,
            conflict: bool = False,
        ) -> Response:
            if wants_json:
                return JSONResponse(
                    {"error": message, "conflict": conflict},
                    status_code=(
                        status.HTTP_409_CONFLICT
                        if conflict
                        else status.HTTP_400_BAD_REQUEST
                    ),
                )
            return RedirectResponse(
                _workbench_path(
                    project_id,
                    chapter_id=chapter_id,
                    error=message,
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )

        if len(content) > 200_000:
            return save_error_response("单章正文不能超过 200000 字")
        try:
            clean_change_summary = _clean_field(
                change_summary,
                "这次主要修改了什么",
                max_length=1000,
            )
        except ValueError as exc:
            return save_error_response(str(exc))
        if database.chapter_has_active_generation(
            int(user["id"]), project_id, chapter_id
        ):
            return save_error_response("AI 正在生成本章，请完成后再保存")
        content_path = Path(str(chapter["content_path"]))
        expected_head = (
            str(chapter.get("head_version_id") or "")
            if expected_head_version_id is None
            else str(expected_head_version_id)
        )
        version_token = secrets.token_hex(16)
        version_path = content_path.parent / "versions" / f"manual-{version_token}.txt"
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        version_id = ""
        created_new_version = False
        try:
            _atomic_write_text(version_path, content, version_token)
            version_id = database.record_manual_chapter_version(
                user_id=int(user["id"]),
                project_id=project_id,
                chapter_id=chapter_id,
                version_path=version_path,
                char_count=len(content),
                effective_char_count=effective_char_count(content),
                content_hash=content_hash,
                change_summary=clean_change_summary,
                expected_old_head_version_id=expected_head,
            )
            if not version_id:
                raise ValueError("章节不存在")
            created_new_version = version_id != expected_head
            if not created_new_version:
                version_path.unlink(missing_ok=True)
            try:
                _atomic_write_text(content_path, content, secrets.token_hex(16))
            except Exception:
                logger.warning(
                    "failed to refresh non-authoritative chapter cache",
                    exc_info=True,
                )
            if created_new_version and content.strip():
                queue_background_memory(
                    request,
                    user_id=int(user["id"]),
                    project_id=project_id,
                    chapter_id=chapter_id,
                    version_id=version_id,
                )
        except ChapterHeadConflict as exc:
            version_path.unlink(missing_ok=True)
            return save_error_response(str(exc), conflict=True)
        except ValueError as exc:
            version_path.unlink(missing_ok=True)
            return save_error_response(str(exc))
        except Exception:
            version_path.unlink(missing_ok=True)
            logger.exception("failed to save novel chapter")
            return save_error_response("保存正文失败")
        if wants_json:
            return JSONResponse(
                {
                    "saved": True,
                    "version_id": version_id,
                    "content_hash": content_hash,
                    "created_new_version": created_new_version,
                },
                headers={"Cache-Control": "no-store"},
            )
        return RedirectResponse(
            _workbench_path(
                project_id,
                chapter_id=chapter_id,
                saved="true",
            ),
            status_code=status.HTTP_303_SEE_OTHER,
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
        memory_retrieval = None
        if job.get("context_snapshot_json"):
            try:
                snapshot = json.loads(str(job["context_snapshot_json"]))
                canonical_memory = dict(snapshot.get("canonical_memory") or {})
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
                        "expanded_term_count": len(retrieval.get("query_terms") or []),
                        "matched_count": int(retrieval.get("matched_count") or 0),
                        "items": [
                            dict(item)
                            for item in (
                                canonical_memory.get("retrieved_memory") or []
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
            if str(job["operation"]) == "propose_reader_branches" and job.get(
                "subject_id"
            ):
                redirect_url = f"/reader-requests/{job['subject_id']}"
            elif str(job["operation"]) == "audit_ai_style" and job.get("version_id"):
                redirect_url = (
                    f"/novels/{job['project_id']}/chapters/"
                    f"{job['chapter_id']}/versions/{job['version_id']}/style"
                )
            elif str(job["operation"]) == "rewrite_style_issue" and job.get(
                "subject_id"
            ):
                redirect_url = f"/style-issues/{job['subject_id']}"
            if not redirect_url:
                redirect_url = _workbench_path(
                    str(job["project_id"]),
                    chapter_id=str(job["chapter_id"]),
                )
        return {
            "id": job_id,
            "status": job["status"],
            "terminal": job["status"] in {"completed", "failed"},
            "redirect_url": redirect_url,
            "error": job["error"] if job["status"] == "failed" else None,
        }

    @application.get("/works/{work_id}/export.readraft.zip")
    async def export_complete_work_archive(request: Request, work_id: str):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        work = database.get_work(int(user["id"]), work_id)
        if not work:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        file_descriptor, raw_path = tempfile.mkstemp(
            prefix=f"readraft-work-{work_id[:8]}-",
            suffix=".readraft.zip",
        )
        os.close(file_descriptor)
        archive_path = Path(raw_path)
        try:
            create_work_archive(
                database=database,
                novels_dir=app_settings.novels_dir,
                documents_dir=app_settings.documents_dir,
                user_id=int(user["id"]),
                work_id=work_id,
                destination=archive_path,
                max_uncompressed_bytes=(app_settings.max_work_archive_bytes),
            )
        except (WorkArchiveError, OSError, sqlite3.Error) as exc:
            archive_path.unlink(missing_ok=True)
            return RedirectResponse(
                f"/dashboard?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return FileResponse(
            archive_path,
            media_type="application/zip",
            filename=f"work-{work_id[:8]}.readraft.zip",
            headers={"Cache-Control": "no-store"},
            background=BackgroundTask(archive_path.unlink, missing_ok=True),
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
            head_path = str(chapter.get("head_content_path") or "")
            content = (
                _read_optional_text(Path(head_path)) if head_path else ""
            )
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

    @application.post("/novels/{project_id}/assistant/new")
    async def new_novel_assistant_conversation(
        request: Request,
        project_id: str,
        chapter_id: str = Form(""),
        settings_tab: str = Form(""),
        archive_tab: str = Form("creative"),
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
                f"/novels/{project_id}/workbench?error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        query = f"?conversation_id={quote(conversation_id)}"
        if clean_chapter_id:
            query += f"&chapter_id={quote(clean_chapter_id)}"
        else:
            clean_settings_tab = (
                settings_tab if settings_tab in WORKBENCH_SETTING_TAB_KEYS else "core"
            )
            query += (
                "&view=archive&archive_tab="
                + quote(
                    archive_tab if archive_tab in WORK_ARCHIVE_TAB_KEYS else "creative"
                )
                + f"&settings_tab={clean_settings_tab}"
            )
        return RedirectResponse(
            f"/novels/{project_id}/workbench" + query,
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post(
        "/novels/{project_id}/assistant/conversations/{conversation_id}/delete"
    )
    async def delete_novel_assistant_conversation(
        request: Request,
        project_id: str,
        conversation_id: str,
        current_conversation_id: str = Form(""),
        chapter_id: str = Form(""),
        return_view: str = Form(""),
        return_archive_tab: str = Form("creative"),
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
        if not target or str(target.get("project_id") or "") != project_id:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        clean_settings_tab = (
            settings_tab if settings_tab in WORKBENCH_SETTING_TAB_KEYS else "core"
        )
        clean_archive_tab = (
            return_archive_tab
            if return_archive_tab in WORK_ARCHIVE_TAB_KEYS
            else "creative"
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
        if current_conversation_id and current_conversation_id != conversation_id:
            candidate = assistant_chat_service.get_conversation(
                user_id=user_id,
                conversation_id=current_conversation_id,
            )
            if candidate and str(candidate.get("project_id") or "") == project_id:
                return_conversation = candidate
        query: list[str] = []
        if return_conversation:
            query.append("conversation_id=" + quote(str(return_conversation["id"])))
            if return_conversation.get("novel_chapter_id"):
                query.append(
                    "chapter_id=" + quote(str(return_conversation["novel_chapter_id"]))
                )
            else:
                query.extend(
                    [
                        "view=archive",
                        f"archive_tab={clean_archive_tab}",
                        f"settings_tab={clean_settings_tab}",
                    ]
                )
        elif return_view == "archive":
            query.extend(
                [
                    "view=archive",
                    f"archive_tab={clean_archive_tab}",
                    f"settings_tab={clean_settings_tab}",
                ]
            )
        elif chapter_id:
            query.append("chapter_id=" + quote(chapter_id))
        else:
            query.extend(["view=archive", "archive_tab=creative"])
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
        model_choice: str = Form(""),
        quality_mode: str = Form("standard"),
        return_view: str = Form(""),
        return_archive_tab: str = Form("creative"),
        return_settings_tab: str = Form(""),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            if _wants_json(request):
                return JSONResponse(
                    {"error": "请先登录"},
                    status_code=status.HTTP_401_UNAUTHORIZED,
                )
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
        clean_return_archive_tab = (
            return_archive_tab
            if return_archive_tab in WORK_ARCHIVE_TAB_KEYS
            else "creative"
        )
        try:
            if conversation_id:
                conversation = assistant_chat_service.get_conversation(
                    user_id=user_id,
                    conversation_id=conversation_id,
                )
                if (
                    not conversation
                    or str(conversation.get("project_id") or "") != project_id
                ):
                    raise ValueError("对话不存在")
            else:
                scope_type = "chapter" if novel_chapter_id else "project"
                conversation_id = assistant_chat_service.create_conversation(
                    user_id=user_id,
                    scope_type=scope_type,
                    title=question,
                    project_id=project_id,
                    novel_chapter_id=(novel_chapter_id or None),
                )
            profile = routed_api_profile(
                user_id,
                quality_mode=quality_mode,
                task_policy="discussion",
                model_choice=model_choice,
            )
            if not profile:
                if _wants_json(request):
                    return JSONResponse(
                        {"error": "开始创作对话前，请先配置模型服务"},
                        status_code=status.HTTP_400_BAD_REQUEST,
                    )
                return RedirectResponse(
                    "/settings/api?error=" + quote("开始创作对话前，请先配置模型服务"),
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
            message_id = assistant_chat_service.queue_message(
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
                    if return_view == "archive"
                    and clean_return_archive_tab == "creative"
                    else ("chapter" if novel_chapter_id else "project")
                ),
                auto_commit=True,
                quality_mode=quality_mode,
            )
            prepared_conversation = assistant_chat_service.get_conversation(
                user_id=user_id,
                conversation_id=conversation_id,
            )
            prepared_chapter_id = str(
                (prepared_conversation or {}).get("novel_chapter_id") or ""
            )
            if not novel_chapter_id and prepared_chapter_id:
                novel_chapter_id = prepared_chapter_id
                return_view = "body"
        except ValueError as exc:
            if _wants_json(request):
                return JSONResponse(
                    {"error": str(exc)},
                    status_code=status.HTTP_400_BAD_REQUEST,
                )
            suffix = (
                f"?conversation_id={quote(conversation_id)}" if conversation_id else ""
            )
            if novel_chapter_id:
                suffix += (
                    "&" if suffix else "?"
                ) + f"chapter_id={quote(novel_chapter_id)}"
            if return_view == "archive":
                suffix += (
                    ("&" if suffix else "?")
                    + "view=archive"
                    + f"&archive_tab={clean_return_archive_tab}"
                    + f"&settings_tab={clean_return_settings_tab}"
                )
            separator = "&" if suffix else "?"
            return RedirectResponse(
                return_path + suffix + separator + "error=" + quote(str(exc)),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        request.app.state.worker.wake()
        query = f"?conversation_id={quote(conversation_id)}&sent=true"
        if novel_chapter_id:
            query += f"&chapter_id={quote(novel_chapter_id)}"
        if return_view == "archive":
            query += (
                "&view=archive"
                f"&archive_tab={clean_return_archive_tab}"
                f"&settings_tab={clean_return_settings_tab}"
            )
        destination = return_path + query
        if _wants_json(request):
            return JSONResponse(
                {
                    "message_id": message_id,
                    "conversation_id": conversation_id,
                    "stream_url": (f"/api/assistant/messages/{message_id}/stream"),
                    "redirect_url": destination,
                },
                status_code=status.HTTP_202_ACCEPTED,
            )
        return RedirectResponse(
            destination,
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
            destination += "&chapter_id=" + quote(str(branch["novel_chapter_id"]))
        else:
            clean_settings_tab = (
                settings_tab if settings_tab in WORKBENCH_SETTING_TAB_KEYS else "core"
            )
            destination += (
                f"&view=archive&archive_tab=creative&settings_tab={clean_settings_tab}"
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
            "novel_chapter_id": conversation.get("novel_chapter_id"),
        }
        destination = _novel_branch_destination(branch, settings_tab=settings_tab)
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
            branch = assistant_chat_service.branch_conversation_from_message(
                user_id=user_id,
                message_id=message_id,
                replacement_question=replacement_question,
            )
            if not branch.get("project_id"):
                raise ValueError("这条消息不属于小说创作对话")
            assistant_chat_service.queue_message(
                user_id=user_id,
                conversation_id=str(branch["conversation_id"]),
                question=str(branch["question"]),
                provider=str(profile["provider"]),
                model=str(profile["model"]),
                credential_source=str(profile["credential_source"]),
                quote=branch.get("quote"),
                agent_role="auto",
                ui_surface=(
                    "chapter" if branch.get("novel_chapter_id") else "settings"
                ),
                auto_commit=True,
            )
        except Exception:
            if branch and branch.get("conversation_id"):
                try:
                    assistant_chat_service.delete_conversation(
                        user_id=user_id,
                        conversation_id=str(branch["conversation_id"]),
                    )
                except (TypeError, ValueError):
                    logger.exception("failed to clean up assistant branch")
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
                "/settings/api?error=" + quote("重新发送消息前，请先配置模型服务"),
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
        if not source or str(source.get("role") or "") != "assistant":
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
                "/settings/api?error=" + quote("重新生成回复前，请先配置模型服务"),
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

    @application.post("/documents/{document_id}/assistant/new")
    async def new_document_assistant_conversation(
        request: Request,
        document_id: str,
        reference_chapter_id: str = Form(""),
        return_view: str = Form(""),
        return_archive_tab: str = Form("analysis"),
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
        clean_archive_tab = (
            return_archive_tab
            if return_archive_tab in WORK_ARCHIVE_TAB_KEYS
            else "analysis"
        )
        return_archive = return_view == "archive" and not clean_chapter_id
        try:
            conversation_id = assistant_chat_service.create_conversation(
                user_id=user_id,
                scope_type=("reference_chapter" if clean_chapter_id else "document"),
                title="新对话",
                document_id=document_id,
                reference_chapter_id=clean_chapter_id or None,
            )
        except ValueError as exc:
            return RedirectResponse(
                _document_workbench_path(
                    document_id,
                    chapter_id=clean_chapter_id or None,
                    view="archive" if return_archive else None,
                    archive_tab=clean_archive_tab if return_archive else None,
                    error=str(exc),
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            _document_workbench_path(
                document_id,
                chapter_id=clean_chapter_id or None,
                conversation_id=conversation_id,
                view="archive" if return_archive else None,
                archive_tab=clean_archive_tab if return_archive else None,
            ),
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
        model_choice: str = Form(""),
        quality_mode: str = Form("standard"),
        return_view: str = Form(""),
        return_archive_tab: str = Form("analysis"),
        csrf: str = Form(...),
    ):
        user = _current_user(request)
        if not user:
            if _wants_json(request):
                return JSONResponse(
                    {"error": "请先登录"},
                    status_code=status.HTTP_401_UNAUTHORIZED,
                )
            return _login_redirect(request)
        verify_csrf(request, csrf)
        user_id = int(user["id"])
        document = database.get_document(user_id, document_id)
        if not document:
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        clean_archive_tab = (
            return_archive_tab
            if return_archive_tab in WORK_ARCHIVE_TAB_KEYS
            else "analysis"
        )
        return_archive = return_view == "archive" and not reference_chapter_id
        try:
            if conversation_id:
                conversation = assistant_chat_service.get_conversation(
                    user_id=user_id,
                    conversation_id=conversation_id,
                )
                if (
                    not conversation
                    or str(conversation.get("document_id") or "") != document_id
                ):
                    raise ValueError("对话不存在")
            else:
                scope_type = "reference_chapter" if reference_chapter_id else "document"
                conversation_id = assistant_chat_service.create_conversation(
                    user_id=user_id,
                    scope_type=scope_type,
                    title=question,
                    document_id=document_id,
                    reference_chapter_id=(reference_chapter_id or None),
                )
            profile = routed_api_profile(
                user_id,
                quality_mode=quality_mode,
                task_policy="discussion",
                model_choice=model_choice,
            )
            if not profile:
                if _wants_json(request):
                    return JSONResponse(
                        {"error": "开始拆书对话前，请先配置模型服务"},
                        status_code=status.HTTP_400_BAD_REQUEST,
                    )
                return RedirectResponse(
                    "/settings/api?error=" + quote("开始拆书对话前，请先配置模型服务"),
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
            message_id = assistant_chat_service.queue_message(
                user_id=user_id,
                conversation_id=conversation_id,
                question=question,
                provider=profile["provider"],
                model=profile["model"],
                credential_source=profile["credential_source"],
                quote=quote_payload,
                agent_role="researcher",
                quality_mode=quality_mode,
            )
        except ValueError as exc:
            if _wants_json(request):
                return JSONResponse(
                    {"error": str(exc)},
                    status_code=status.HTTP_400_BAD_REQUEST,
                )
            return RedirectResponse(
                _document_workbench_path(
                    document_id,
                    chapter_id=reference_chapter_id or None,
                    conversation_id=conversation_id or None,
                    view="archive" if return_archive else None,
                    archive_tab=clean_archive_tab if return_archive else None,
                    error=str(exc),
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        request.app.state.worker.wake()
        destination = _document_workbench_path(
            document_id,
            chapter_id=reference_chapter_id or None,
            conversation_id=conversation_id,
            view="archive" if return_archive else None,
            archive_tab=clean_archive_tab if return_archive else None,
            sent="true",
        )
        if _wants_json(request):
            return JSONResponse(
                {
                    "message_id": message_id,
                    "conversation_id": conversation_id,
                    "stream_url": (f"/api/assistant/messages/{message_id}/stream"),
                    "redirect_url": destination,
                },
                status_code=status.HTTP_202_ACCEPTED,
            )
        return RedirectResponse(
            destination,
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/api/assistant/messages/{message_id}/cancel")
    async def cancel_assistant_message(
        request: Request,
        message_id: str,
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
            result = assistant_chat_service.request_message_cancellation(
                user_id=int(user["id"]),
                message_id=message_id,
            )
        except ValueError as exc:
            return JSONResponse(
                {"error": str(exc)},
                status_code=status.HTTP_404_NOT_FOUND,
            )
        worker = getattr(request.app.state, "worker", None)
        interrupted = bool(
            worker
            and result.get("status") == "running"
            and worker.cancel_assistant_message(message_id)
        )
        if worker:
            worker.wake()
        return JSONResponse(
            {**result, "interrupted": interrupted},
            status_code=status.HTTP_202_ACCEPTED,
            headers={"Cache-Control": "no-store"},
        )

    @application.get("/api/assistant/messages/{message_id}/stream")
    async def assistant_message_stream(
        request: Request,
        message_id: str,
    ):
        user = _current_user(request)
        if not user:
            return JSONResponse(
                {"error": "unauthorized"},
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        user_id = int(user["id"])
        initial = assistant_chat_service.get_message_stream_state(
            user_id=user_id,
            message_id=message_id,
        )
        if not initial:
            return JSONResponse(
                {"error": "not_found"},
                status_code=status.HTTP_404_NOT_FOUND,
            )

        async def events():
            previous_sequence = -1
            previous_status = ""
            previous_event_sequence = 0
            last_keepalive = time.monotonic()
            for _attempt in range(7_200):
                if await request.is_disconnected():
                    return
                state = await asyncio.to_thread(
                    assistant_chat_service.get_message_stream_state,
                    user_id=user_id,
                    message_id=message_id,
                    after_event_sequence=previous_event_sequence,
                )
                if not state:
                    yield ('event: failed\ndata: {"error":"消息不存在"}\n\n')
                    return
                sequence = int(state.get("stream_sequence") or 0)
                message_status = str(state.get("status") or "")
                for agent_event in state.get("events") or []:
                    yield (
                        "event: agent\n"
                        "data: " + json.dumps(agent_event, ensure_ascii=False) + "\n\n"
                    )
                    previous_event_sequence = max(
                        previous_event_sequence,
                        int(agent_event.get("sequence") or 0),
                    )
                if sequence != previous_sequence or message_status != previous_status:
                    payload = {
                        "id": state["id"],
                        "conversation_id": state["conversation_id"],
                        "status": message_status,
                        "content": state.get("content") or "",
                        "sequence": sequence,
                        "model": state.get("model") or "",
                        "error": state.get("error"),
                        "run_state": state.get("run_state") or "",
                        "run_state_label": (state.get("run_state_label") or ""),
                        "run_sequence": int(state.get("run_sequence") or 0),
                        "cancel_requested": bool(state.get("cancel_requested")),
                        "cancelled": bool(state.get("cancelled")),
                        "terminal": bool(state.get("terminal")),
                    }
                    yield (
                        "event: snapshot\n"
                        "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
                    )
                    previous_sequence = sequence
                    previous_status = message_status
                if state.get("terminal"):
                    redirect_url = ""
                    if state.get("project_id"):
                        redirect_url = _workbench_path(
                            str(state["project_id"]),
                            chapter_id=(
                                str(state["novel_chapter_id"])
                                if state.get("novel_chapter_id")
                                else None
                            ),
                            conversation_id=str(state["conversation_id"]),
                        )
                    elif state.get("document_id"):
                        redirect_url = _document_workbench_path(
                            str(state["document_id"]),
                            chapter_id=(
                                str(state["reference_chapter_id"])
                                if state.get("reference_chapter_id")
                                else None
                            ),
                            conversation_id=str(state["conversation_id"]),
                        )
                    yield (
                        "event: done\n"
                        "data: "
                        + json.dumps(
                            {
                                "status": message_status,
                                "error": state.get("error"),
                                "run_state": (state.get("run_state") or ""),
                                "cancelled": bool(state.get("cancelled")),
                                "redirect_url": redirect_url,
                            },
                            ensure_ascii=False,
                        )
                        + "\n\n"
                    )
                    return
                if time.monotonic() - last_keepalive >= 12:
                    yield ": keepalive\n\n"
                    last_keepalive = time.monotonic()
                await asyncio.sleep(0.25)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-store",
                "X-Accel-Buffering": "no",
            },
        )

    @application.get("/api/assistant/messages/{message_id}")
    async def assistant_message_status(request: Request, message_id: str):
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
            ((message.get("response") or {}).get("auto_commit") or {}).get("status", "")
        )
        return JSONResponse(
            {
                "id": message["id"],
                "conversation_id": message["conversation_id"],
                "status": message_status,
                "run_state": message.get("run_state") or "",
                "run_state_label": message.get("run_state_label") or "",
                "cancelled": message.get("run_state") == "cancelled",
                "terminal": (
                    message_status == "failed"
                    or (
                        message_status == "completed"
                        and auto_commit_status != "pending"
                    )
                ),
                "error": message.get("error"),
                "content": message.get("stream_content")
                or message.get("content")
                or "",
                "sequence": int(message.get("stream_sequence") or 0),
            },
            headers={"Cache-Control": "no-store"},
        )

    @application.post("/assistant/messages/{message_id}/apply-settings")
    async def apply_assistant_settings(
        request: Request,
        message_id: str,
        selection_present: str = Form(""),
        selected_path: list[str] = Form(default=[]),
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
                selected_paths=(
                    {str(value) for value in selected_path}
                    if selection_present
                    else None
                ),
            )
        except ValueError as exc:
            message = assistant_chat_service.get_message(
                user_id=user_id, message_id=message_id
            )
            if not message:
                return Response(status_code=status.HTTP_404_NOT_FOUND)
            conversation = assistant_chat_service.get_conversation(
                user_id=user_id,
                conversation_id=str(message["conversation_id"]),
            )
            if not conversation or not conversation.get("project_id"):
                return Response(status_code=status.HTTP_400_BAD_REQUEST)
            return RedirectResponse(
                f"/novels/{conversation['project_id']}/workbench"
                "?view=archive&archive_tab=creative"
                f"&conversation_id={conversation['id']}"
                f"&error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            _workbench_path(
                str(result["project_id"]),
                archive_tab="creative",
                conversation_id=str(result["conversation_id"]),
            ),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/assistant/messages/{message_id}/apply-story-plan")
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
                return Response(status_code=status.HTTP_404_NOT_FOUND)
            conversation = assistant_chat_service.get_conversation(
                user_id=user_id,
                conversation_id=str(message["conversation_id"]),
            )
            if not conversation or not conversation.get("project_id"):
                return Response(status_code=status.HTTP_400_BAD_REQUEST)
            return RedirectResponse(
                f"/novels/{conversation['project_id']}/workbench"
                "?view=archive&archive_tab=creative&settings_tab=structure"
                f"&conversation_id={conversation['id']}"
                f"&error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            f"/novels/{result['project_id']}/workbench"
            "?view=archive&archive_tab=creative&settings_tab=structure"
            f"&conversation_id={result['conversation_id']}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/assistant/messages/{message_id}/commit")
    async def commit_assistant_draft(
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
            result = assistant_chat_service.commit_draft_to_head(
                user_id=user_id,
                assistant_message_id=message_id,
            )
        except ValueError as exc:
            message = assistant_chat_service.get_message(
                user_id=user_id, message_id=message_id
            )
            if not message:
                return Response(status_code=status.HTTP_404_NOT_FOUND)
            conversation = assistant_chat_service.get_conversation(
                user_id=user_id,
                conversation_id=str(message["conversation_id"]),
            )
            if not conversation or not conversation.get("project_id"):
                return Response(status_code=status.HTTP_400_BAD_REQUEST)
            return RedirectResponse(
                f"/novels/{conversation['project_id']}/workbench"
                f"?chapter_id={conversation['novel_chapter_id']}"
                f"&conversation_id={conversation['id']}"
                f"&error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        return RedirectResponse(
            _workbench_path(
                str(result["project_id"]),
                chapter_id=str(result["chapter_id"]),
                conversation_id=str(result["conversation_id"]),
            ),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.post("/assistant/messages/{message_id}/revert-auto-commit")
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
                return Response(status_code=status.HTTP_404_NOT_FOUND)
            conversation = assistant_chat_service.get_conversation(
                user_id=user_id,
                conversation_id=str(message["conversation_id"]),
            )
            if not conversation or not conversation.get("project_id"):
                return Response(status_code=status.HTTP_400_BAD_REQUEST)
            return RedirectResponse(
                f"/novels/{conversation['project_id']}/workbench"
                f"?chapter_id={conversation['novel_chapter_id']}"
                f"&conversation_id={conversation['id']}"
                f"&error={quote(str(exc))}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        queue_background_memory(
            request,
            user_id=user_id,
            project_id=str(result["project_id"]),
            chapter_id=str(result["chapter_id"]),
            version_id=str(result["version_id"]),
        )
        return RedirectResponse(
            f"/novels/{result['project_id']}/workbench"
            f"?chapter_id={result['chapter_id']}"
            f"&conversation_id={result['conversation_id']}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @application.get(
        "/novels/{project_id}/chapters/{chapter_id}/versions/{version_id}/source",
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
                        == str(source.get("head_version_id") or "")
                        else "候选 / 历史版本"
                    )
                ),
                before=content[:safe_start],
                selection=content[safe_start:safe_end],
                after=content[safe_end:],
                back_url=_workbench_path(project_id, chapter_id=chapter_id),
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
                source_title=(f"参考书第 {source['position']} 章《{source['title']}》"),
                source_meta=(f"拆书原文 · {source['document_title']}"),
                before=content[:safe_start],
                selection=content[safe_start:safe_end],
                after=content[safe_end:],
                back_url=f"/documents/{document_id}",
            ),
        )

    @application.get("/documents/{document_id}", response_class=HTMLResponse)
    async def document_page(
        request: Request,
        document_id: str,
        chapter_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        view: str = "body",
        archive_tab: str = "analysis",
        settings_tab: str = "core",
        error: Optional[str] = None,
        saved: bool = False,
        adopted: bool = False,
        removed: bool = False,
        sent: bool = False,
    ):
        user = _current_user(request)
        if not user:
            return _login_redirect(request)
        user_id = int(user["id"])
        try:
            context = workbench_view_builder.build_document(
                user_id=user_id,
                document_id=document_id,
                chapter_id=chapter_id,
                conversation_id=conversation_id,
                view=view,
                archive_tab=archive_tab,
                settings_tab=settings_tab,
            )
        except WorkbenchNotFound:
            return render_template(
                "not_found.html",
                _template_context(request, user=user),
                status_code=status.HTTP_404_NOT_FOUND,
            )
        except WorkbenchUnavailable as exc:
            return Response(
                str(exc),
                status_code=status.HTTP_409_CONFLICT,
            )

        active_conversation = context["active_conversation"]
        available_chat_models = chat_model_groups(user_id)
        context.update(
            archive_categories=WORK_ARCHIVE_CATEGORIES,
            analysis_categories=WORK_ANALYSIS_CATEGORIES,
            setting_tabs=WORKBENCH_SETTING_TABS,
            model_groups=available_chat_models,
            selected_model_choice=selected_chat_model(
                available_chat_models,
                active_conversation,
            ),
            quality_modes=quality_mode_options(user_id),
            selected_quality_mode=selected_quality_mode(
                user_id,
                active_conversation,
            ),
            pov_options=POV_OPTIONS,
            world_entry_type_options=WORLD_ENTRY_TYPE_OPTIONS,
            story_arc_type_options=STORY_ARC_TYPE_OPTIONS,
            archive_saved=saved,
            archive_adopted=adopted,
            archive_removed=removed,
            archive_error=error,
            error=error,
            sent=sent,
        )
        return render_template(
            "document.html",
            _template_context(request, user=user, **context),
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
        elif app_settings.model_api_key:
            provider = analyzer.provider
            model = analyzer.model
            credential_source = "default"
        elif app_settings.uses_test_models:
            provider = analyzer.provider
            model = analyzer.model
            credential_source = "default"
        else:
            return RedirectResponse(
                "/settings/api?error=" + quote("开始分析前，请先配置你的模型服务"),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        try:
            job_id = database.create_job(
                user_id=int(user["id"]),
                document_id=document_id,
                provider=provider,
                model=model,
                credential_source=credential_source,
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
    async def job_page(request: Request, job_id: str, error: Optional[str] = None):
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
        if job.get(
            "credential_source"
        ) == "personal" and not database.has_api_credential(int(user["id"])):
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
            json.loads(analysis["result_json"]) if analysis.get("result_json") else None
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

    @application.post("/analyses/{analysis_id}/techniques/{technique_index}")
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
