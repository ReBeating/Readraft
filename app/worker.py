from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from pathlib import Path

from .assistant_chat import (
    AnswerUpdateCallback,
    BaseAssistantChatModel,
    DeepSeekAssistantChatModel,
)
from .agent_orchestrator import AssistantAgentOrchestrator
from .assistant_chat_service import AssistantChatService
from .causal_branch_planner import (
    BaseCausalBranchPlanner,
    DeepSeekCausalBranchPlanner,
)
from .causal_branch_service import CausalBranchSimulationService
from .causal_suggestion_planner import (
    BaseCausalSuggestionPlanner,
    DeepSeekCausalSuggestionPlanner,
)
from .causal_suggestion_service import CausalSuggestionService
from .config import Settings
from .context_compiler import (
    build_scene_context_snapshot,
    build_writing_context_snapshot,
    compile_active_techniques,
    compile_canonical_memory,
    compile_planned_causal_links,
    compile_story_plan_context,
)
from .credentials import CredentialCipher, CredentialError
from .db import Database
from .deepseek import AnalyzerError, BaseAnalyzer, DeepSeekAnalyzer
from .memory_extraction import (
    BaseMemoryExtractor,
    DeepSeekMemoryExtractor,
)
from .memory_service import MemoryService
from .model_provider import (
    ProviderConfigError,
    settings_for_credential,
    settings_for_reasoning_policy,
)
from .model_routing import (
    ModelTaskPolicy,
    normalize_quality_mode,
    route_model_task,
)
from .planning_ai import (
    BaseChapterPlanner,
    DeepSeekChapterPlanner,
)
from .planning_schema import (
    ChapterTaskCard,
    chapter_task_card_fingerprint,
    chapter_task_card_payload,
)
from .planning_service import PlanningService
from .preference_extraction import (
    BaseEditPreferenceExtractor,
    DeepSeekEditPreferenceExtractor,
    locate_edit_preference_evidence,
)
from .preference_service import PreferenceService
from .reader_planner import (
    BaseReaderPlanner,
    DeepSeekReaderPlanner,
)
from .reader_service import ReaderDecisionService
from .security import stable_provider_user_id
from .scene_service import SceneService
from .style_editor import (
    BaseStyleEditor,
    DeepSeekStyleEditor,
    locate_style_issues,
    surrounding_excerpt,
)
from .style_service import StyleService
from .story_plan_suggestion_service import StoryPlanSuggestionService
from .story_planner import (
    BaseStoryPlanner,
    DeepSeekStoryPlanner,
)
from .story_structure_planner import (
    BaseStoryStructurePlanner,
    DeepSeekStoryStructurePlanner,
)
from .story_structure_service import StoryStructureSuggestionService
from .voice_extraction import (
    BaseVoiceProfileExtractor,
    DeepSeekVoiceProfileExtractor,
    locate_voice_evidence,
)
from .writing import BaseWriter, DeepSeekWriter
from .web_search import ExaWebSearch, WebSearchError


logger = logging.getLogger(__name__)


class AnalysisWorker:
    def __init__(
        self,
        database: Database,
        analyzer: BaseAnalyzer | None,
        writer: BaseWriter | None,
        provider_user_secret: str,
        settings: Settings,
        credential_cipher: CredentialCipher,
        memory_extractor: BaseMemoryExtractor | None = None,
        chapter_planner: BaseChapterPlanner | None = None,
        style_editor: BaseStyleEditor | None = None,
        reader_planner: BaseReaderPlanner | None = None,
        voice_profile_extractor: BaseVoiceProfileExtractor | None = None,
        edit_preference_extractor: BaseEditPreferenceExtractor | None = None,
        story_planner: BaseStoryPlanner | None = None,
        story_structure_planner: BaseStoryStructurePlanner | None = None,
        causal_suggestion_planner: (
            BaseCausalSuggestionPlanner | None
        ) = None,
        causal_branch_planner: BaseCausalBranchPlanner | None = None,
        assistant_chat_model: BaseAssistantChatModel | None = None,
        poll_seconds: float = 1.0,
    ):
        self.database = database
        self.analyzer = analyzer
        self.writer = writer
        self.provider_user_secret = provider_user_secret
        self.settings = settings
        self.credential_cipher = credential_cipher
        self.memory_extractor = memory_extractor
        self.memory_service = MemoryService(database)
        self.chapter_planner = chapter_planner
        self.style_editor = style_editor
        self.reader_planner = reader_planner
        self.voice_profile_extractor = voice_profile_extractor
        self.edit_preference_extractor = edit_preference_extractor
        self.story_planner = story_planner
        self.story_structure_planner = story_structure_planner
        self.causal_suggestion_planner = causal_suggestion_planner
        self.causal_branch_planner = causal_branch_planner
        self.assistant_chat_model = assistant_chat_model
        self.assistant_chat_service = AssistantChatService(
            database,
            settings.novels_dir,
            settings.documents_dir,
        )
        self.assistant_agent_orchestrator = AssistantAgentOrchestrator(
            self.assistant_chat_service,
            web_search=self._search_web,
        )
        self.planning_service = PlanningService(database)
        self.scene_service = SceneService(database)
        self.style_service = StyleService(database)
        self.preference_service = PreferenceService(database)
        self.story_plan_suggestion_service = StoryPlanSuggestionService(
            database
        )
        self.story_structure_suggestion_service = (
            StoryStructureSuggestionService(
                database, settings.novels_dir
            )
        )
        self.causal_suggestion_service = CausalSuggestionService(database)
        self.causal_branch_service = CausalBranchSimulationService(
            database
        )
        self.reader_service = ReaderDecisionService(
            database, settings.novels_dir
        )
        self.poll_seconds = poll_seconds
        self._wake = asyncio.Event()
        self._stopping = False
        self.last_heartbeat = time.monotonic()
        self.last_error: str | None = None
        self.consecutive_loop_failures = 0

    def _search_web(
        self,
        user_id: int,
        query: str,
        max_results: int,
    ) -> list[dict[str, str]]:
        settings = self.database.get_web_search_settings(user_id)
        if settings is not None and not settings.get("enabled"):
            raise ValueError("联网搜索尚未启用")
        try:
            results = ExaWebSearch(
                api_key=self.settings.exa_api_key,
            ).search(
                query,
                max_results=max_results,
            )
        except WebSearchError as exc:
            raise ValueError(str(exc)) from exc
        return [
            {
                "title": item.title,
                "url": item.url,
                "snippet": item.snippet,
            }
            for item in results
        ]

    @property
    def healthy(self) -> bool:
        return not self._stopping and self.consecutive_loop_failures < 3

    def wake(self) -> None:
        self._wake.set()

    async def _personal_model_settings(
        self, item: dict, task_policy: ModelTaskPolicy
    ) -> Settings:
        user_id = int(item["user_id"])
        routing_preferences, stored_adapter_prompt = await asyncio.gather(
            asyncio.to_thread(
                self.database.get_model_routing_preferences, user_id
            ),
            asyncio.to_thread(
                self.database.get_model_adapter_prompt, user_id
            ),
        )
        quality_mode = normalize_quality_mode(
            item.get("quality_mode")
            or routing_preferences["default_quality_mode"]
        )
        routing = route_model_task(quality_mode, task_policy)
        preferred_provider = str(
            routing_preferences[
                f"{routing.model_role}_provider"
            ]
            or ""
        )
        preferred_model = str(
            routing_preferences[f"{routing.model_role}_model"]
            or ""
        )
        provider = preferred_provider or str(item["provider"])
        model = preferred_model or str(item["model"])
        credential, allowed_models = await asyncio.gather(
            asyncio.to_thread(
                self.database.get_api_credential, user_id, provider
            ),
            asyncio.to_thread(
                self.database.list_api_models, user_id, provider
            ),
        )
        if (
            not credential
            or (
                model != str(credential["model"])
                and model not in allowed_models
            )
        ):
            provider = str(item["provider"])
            model = str(item["model"])
            credential = await asyncio.to_thread(
                self.database.get_api_credential, user_id, provider
            )
        if not credential:
            raise AnalyzerError(
                "个人模型凭据已被删除，请重新配置后重试"
            )
        try:
            api_key = self.credential_cipher.decrypt(
                str(credential["encrypted_key"])
            )
            return settings_for_reasoning_policy(
                settings_for_credential(
                    self.settings,
                    credential=credential,
                    api_key=api_key,
                    model=model,
                    model_adapter_prompt=(
                        self.settings.model_adapter_prompt
                        if stored_adapter_prompt is None
                        else stored_adapter_prompt
                    ),
                ),
                routing.reasoning_policy,
            )
        except (CredentialError, ProviderConfigError) as exc:
            raise AnalyzerError(str(exc)) from exc

    async def run(self) -> None:
        logger.info(
            "AI worker started analyzer=%s/%s writer=%s/%s",
            self.analyzer.provider if self.analyzer else "unconfigured",
            self.analyzer.model if self.analyzer else "unconfigured",
            self.writer.provider if self.writer else "unconfigured",
            self.writer.model if self.writer else "unconfigured",
        )
        while not self._stopping:
            item = None
            generation_item = None
            story_plan_item = None
            story_structure_item = None
            causal_suggestion_item = None
            causal_branch_item = None
            voice_item = None
            preference_item = None
            chat_item = None
            try:
                self.last_heartbeat = time.monotonic()
                generation_item = await asyncio.to_thread(
                    self.database.claim_next_generation
                )
                if generation_item:
                    self.consecutive_loop_failures = 0
                    self.last_error = None
                    await self._process_generation(generation_item)
                    continue
                story_plan_item = await asyncio.to_thread(
                    self.story_plan_suggestion_service.claim_next_suggestion
                )
                if story_plan_item:
                    self.consecutive_loop_failures = 0
                    self.last_error = None
                    await self._process_story_plan_suggestion(
                        story_plan_item
                    )
                    continue
                story_structure_item = await asyncio.to_thread(
                    self.story_structure_suggestion_service.claim_next_suggestion
                )
                if story_structure_item:
                    self.consecutive_loop_failures = 0
                    self.last_error = None
                    await self._process_story_structure_suggestion(
                        story_structure_item
                    )
                    continue
                causal_suggestion_item = await asyncio.to_thread(
                    self.causal_suggestion_service.claim_next_suggestion
                )
                if causal_suggestion_item:
                    self.consecutive_loop_failures = 0
                    self.last_error = None
                    await self._process_causal_suggestion(
                        causal_suggestion_item
                    )
                    continue
                causal_branch_item = await asyncio.to_thread(
                    self.causal_branch_service.claim_next_simulation
                )
                if causal_branch_item:
                    self.consecutive_loop_failures = 0
                    self.last_error = None
                    await self._process_causal_branch_simulation(
                        causal_branch_item
                    )
                    continue
                voice_item = await asyncio.to_thread(
                    self.style_service.claim_next_voice_suggestion
                )
                if voice_item:
                    self.consecutive_loop_failures = 0
                    self.last_error = None
                    await self._process_voice_suggestion(voice_item)
                    continue
                preference_item = await asyncio.to_thread(
                    self.preference_service.claim_next_suggestion
                )
                if preference_item:
                    self.consecutive_loop_failures = 0
                    self.last_error = None
                    await self._process_edit_preference_suggestion(
                        preference_item
                    )
                    continue
                chat_item = await asyncio.to_thread(
                    self.assistant_chat_service.claim_next_message
                )
                if chat_item:
                    self.consecutive_loop_failures = 0
                    self.last_error = None
                    await self._process_assistant_chat(chat_item)
                    continue
                item = await asyncio.to_thread(self.database.claim_next_analysis)
                self.consecutive_loop_failures = 0
                self.last_error = None
                if item:
                    await self._process(item)
                    continue
                self._wake.clear()
                try:
                    await asyncio.wait_for(
                        self._wake.wait(), timeout=self.poll_seconds
                    )
                except asyncio.TimeoutError:
                    pass
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.consecutive_loop_failures += 1
                self.last_error = str(exc)
                logger.exception(
                    "AI worker loop failure count=%s",
                    self.consecutive_loop_failures,
                )
                if generation_item and generation_item.get("claim_token"):
                    try:
                        await asyncio.to_thread(
                            self.database.release_generation_claim,
                            str(generation_item["id"]),
                            str(generation_item["claim_token"]),
                            "工作进程发生临时错误，已重新排队",
                        )
                    except Exception:
                        logger.exception(
                            "failed to release generation claim id=%s",
                            generation_item.get("id"),
                        )
                elif story_plan_item and story_plan_item.get("claim_token"):
                    try:
                        await asyncio.to_thread(
                            self.story_plan_suggestion_service.release_claim,
                            str(story_plan_item["id"]),
                            str(story_plan_item["claim_token"]),
                            "工作进程发生临时错误，已重新排队",
                        )
                    except Exception:
                        logger.exception(
                            "failed to release story plan suggestion "
                            "claim id=%s",
                            story_plan_item.get("id"),
                        )
                elif (
                    story_structure_item
                    and story_structure_item.get("claim_token")
                ):
                    try:
                        await asyncio.to_thread(
                            self.story_structure_suggestion_service.release_claim,
                            str(story_structure_item["id"]),
                            str(story_structure_item["claim_token"]),
                            "工作进程发生临时错误，已重新排队",
                        )
                    except Exception:
                        logger.exception(
                            "failed to release story structure "
                            "suggestion claim id=%s",
                            story_structure_item.get("id"),
                        )
                elif (
                    causal_suggestion_item
                    and causal_suggestion_item.get("claim_token")
                ):
                    try:
                        await asyncio.to_thread(
                            self.causal_suggestion_service.release_claim,
                            str(causal_suggestion_item["id"]),
                            str(causal_suggestion_item["claim_token"]),
                            "工作进程发生临时错误，已重新排队",
                        )
                    except Exception:
                        logger.exception(
                            "failed to release causal suggestion "
                            "claim id=%s",
                            causal_suggestion_item.get("id"),
                        )
                elif (
                    causal_branch_item
                    and causal_branch_item.get("claim_token")
                ):
                    try:
                        await asyncio.to_thread(
                            self.causal_branch_service.release_claim,
                            str(causal_branch_item["id"]),
                            str(causal_branch_item["claim_token"]),
                            "工作进程发生临时错误，已重新排队",
                        )
                    except Exception:
                        logger.exception(
                            "failed to release causal branch simulation "
                            "claim id=%s",
                            causal_branch_item.get("id"),
                        )
                elif voice_item and voice_item.get("claim_token"):
                    try:
                        await asyncio.to_thread(
                            self.style_service.release_voice_suggestion_claim,
                            str(voice_item["id"]),
                            str(voice_item["claim_token"]),
                            "工作进程发生临时错误，已重新排队",
                        )
                    except Exception:
                        logger.exception(
                            "failed to release voice suggestion claim id=%s",
                            voice_item.get("id"),
                        )
                elif preference_item and preference_item.get("claim_token"):
                    try:
                        await asyncio.to_thread(
                            self.preference_service.release_claim,
                            str(preference_item["id"]),
                            str(preference_item["claim_token"]),
                            "工作进程发生临时错误，已重新排队",
                        )
                    except Exception:
                        logger.exception(
                            "failed to release editing preference claim id=%s",
                            preference_item.get("id"),
                        )
                elif chat_item and chat_item.get("claim_token"):
                    try:
                        await asyncio.to_thread(
                            self.assistant_chat_service.release_claim,
                            str(chat_item["id"]),
                            str(chat_item["claim_token"]),
                            "工作进程发生临时错误，已重新排队",
                        )
                    except Exception:
                        logger.exception(
                            "failed to release assistant chat claim id=%s",
                            chat_item.get("id"),
                        )
                elif item and item.get("claim_token"):
                    try:
                        await asyncio.to_thread(
                            self.database.release_claim,
                            str(item["analysis_id"]),
                            str(item["job_id"]),
                            str(item["claim_token"]),
                            "工作进程发生临时错误，已重新排队",
                        )
                    except Exception:
                        logger.exception(
                            "failed to release chapter claim id=%s",
                            item.get("analysis_id"),
                        )
                self._wake.clear()
                try:
                    await asyncio.wait_for(
                        self._wake.wait(),
                        timeout=min(10.0, max(self.poll_seconds, 2.0)),
                    )
                except asyncio.TimeoutError:
                    pass

    async def _process_assistant_chat(self, item: dict) -> None:
        message_id = str(item["id"])
        claim_token = str(item["claim_token"])
        pending_stream = ""
        flushed_stream = ""
        last_stream_write = 0.0

        async def update_stream(content: str) -> None:
            nonlocal pending_stream, flushed_stream, last_stream_write
            pending_stream = content
            now = time.monotonic()
            if (
                len(pending_stream) - len(flushed_stream) < 48
                and now - last_stream_write < 0.08
            ):
                return
            accepted = await asyncio.to_thread(
                self.assistant_chat_service.set_message_stream,
                message_id=message_id,
                claim_token=claim_token,
                content=pending_stream,
            )
            if accepted:
                flushed_stream = pending_stream
                last_stream_write = now

        async def flush_stream() -> None:
            nonlocal flushed_stream
            if pending_stream == flushed_stream:
                return
            accepted = await asyncio.to_thread(
                self.assistant_chat_service.set_message_stream,
                message_id=message_id,
                claim_token=claim_token,
                content=pending_stream,
            )
            if accepted:
                flushed_stream = pending_stream

        try:
            payload = await asyncio.to_thread(
                self.assistant_chat_service.build_job_payload, item
            )
            response = await self._reply_assistant_chat(
                item=item,
                payload=payload,
                on_answer_update=update_stream,
            )
            await flush_stream()
            accepted = await asyncio.to_thread(
                self.assistant_chat_service.complete_message,
                message_id=message_id,
                claim_token=claim_token,
                response=response,
            )
            if not accepted:
                logger.warning(
                    "discarded stale assistant chat result id=%s",
                    message_id,
                )
            else:
                await self._queue_assistant_memory(item, message_id)
        except AnalyzerError as exc:
            await flush_stream()
            logger.warning(
                "assistant chat failed id=%s: %s", message_id, exc
            )
            await asyncio.to_thread(
                self.assistant_chat_service.fail_message,
                message_id,
                claim_token,
                str(exc),
                input_tokens=exc.input_tokens,
                output_tokens=exc.output_tokens,
            )
        except (OSError, UnicodeError, ValueError) as exc:
            await flush_stream()
            logger.warning(
                "assistant chat input failed id=%s: %s",
                message_id,
                exc,
            )
            await asyncio.to_thread(
                self.assistant_chat_service.fail_message,
                message_id,
                claim_token,
                str(exc),
            )

    async def _queue_assistant_memory(
        self, item: dict, message_id: str
    ) -> None:
        if not item.get("project_id") or not item.get("novel_chapter_id"):
            return
        try:
            message = await asyncio.to_thread(
                self.assistant_chat_service.get_message,
                user_id=int(item["user_id"]),
                message_id=message_id,
            )
            auto_commit = dict(
                ((message or {}).get("response") or {}).get(
                    "auto_commit"
                )
                or {}
            )
            version_id = str(auto_commit.get("version_id") or "")
            if auto_commit.get("status") != "applied" or not version_id:
                return
            await asyncio.to_thread(
                self.database.create_memory_extraction_job,
                user_id=int(item["user_id"]),
                project_id=str(item["project_id"]),
                chapter_id=str(item["novel_chapter_id"]),
                version_id=version_id,
                provider=str(item["provider"]),
                model=str(item["model"]),
                credential_source=str(item["credential_source"]),
            )
        except Exception:
            logger.exception(
                "failed to queue background memory for assistant message id=%s",
                message_id,
            )

    async def _process(self, item: dict) -> None:
        analysis_id = str(item["analysis_id"])
        job_id = str(item["job_id"])
        claim_token = str(item["claim_token"])
        try:
            content = await asyncio.to_thread(
                Path(str(item["content_path"])).read_text, encoding="utf-8"
            )
            response = await self._analyze(
                item,
                content,
            )
            accepted = await asyncio.to_thread(
                self.database.complete_analysis,
                analysis_id=analysis_id,
                job_id=job_id,
                result=response.result.model_dump(mode="json"),
                raw_response=response.raw_response,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                claim_token=claim_token,
            )
            if not accepted:
                logger.warning(
                    "discarded stale chapter analysis result id=%s", analysis_id
                )
        except AnalyzerError as exc:
            logger.warning("chapter analysis failed id=%s: %s", analysis_id, exc)
            await asyncio.to_thread(
                self.database.fail_analysis,
                analysis_id,
                job_id,
                str(exc),
                claim_token,
                exc.input_tokens,
                exc.output_tokens,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("unexpected chapter analysis failure id=%s", analysis_id)
            await asyncio.to_thread(
                self.database.fail_analysis,
                analysis_id,
                job_id,
                "处理章节时发生内部错误，请重试",
                claim_token,
            )

    async def _process_voice_suggestion(self, item: dict) -> None:
        suggestion_id = str(item["id"])
        claim_token = str(item["claim_token"])
        try:
            response = await self._extract_voice_profile(item)
            located, dropped = locate_voice_evidence(
                str(item["sample_text"]), response.result
            )
            if len(located) < 2:
                raise AnalyzerError(
                    "作品声纹建议缺少至少两条可在样章中逐字核对的证据",
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                )
            suggestion = response.result.model_dump(mode="json")
            suggestion["evidence"] = located
            accepted = await asyncio.to_thread(
                self.style_service.complete_voice_suggestion,
                suggestion_id=suggestion_id,
                claim_token=claim_token,
                suggestion=suggestion,
                raw_response=response.raw_response,
                provider=response.provider,
                model=response.model,
                valid_evidence_count=len(located),
                dropped_evidence_count=dropped,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
            if not accepted:
                logger.warning(
                    "discarded stale voice suggestion result id=%s",
                    suggestion_id,
                )
        except AnalyzerError as exc:
            logger.warning(
                "voice profile extraction failed id=%s: %s",
                suggestion_id,
                exc,
            )
            await asyncio.to_thread(
                self.style_service.fail_voice_suggestion,
                suggestion_id=suggestion_id,
                claim_token=claim_token,
                error=str(exc),
                input_tokens=exc.input_tokens,
                output_tokens=exc.output_tokens,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "unexpected voice profile extraction failure id=%s",
                suggestion_id,
            )
            await asyncio.to_thread(
                self.style_service.fail_voice_suggestion,
                suggestion_id=suggestion_id,
                claim_token=claim_token,
                error="处理作者样章时发生内部错误，请重试",
            )

    async def _process_story_plan_suggestion(self, item: dict) -> None:
        suggestion_id = str(item["id"])
        claim_token = str(item["claim_token"])
        try:
            response = await self._plan_story(
                item=item,
                context=dict(item.get("context_snapshot") or {}),
            )
            accepted = await asyncio.to_thread(
                self.story_plan_suggestion_service.complete_suggestion,
                suggestion_id=suggestion_id,
                claim_token=claim_token,
                result=response.result,
                raw_response=response.raw_response,
                provider=response.provider,
                model=response.model,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
            if not accepted:
                logger.warning(
                    "discarded stale story plan suggestion result id=%s",
                    suggestion_id,
                )
        except AnalyzerError as exc:
            logger.warning(
                "story plan suggestion failed id=%s: %s",
                suggestion_id,
                exc,
            )
            await asyncio.to_thread(
                self.story_plan_suggestion_service.fail_suggestion,
                suggestion_id=suggestion_id,
                claim_token=claim_token,
                error=str(exc),
                input_tokens=exc.input_tokens,
                output_tokens=exc.output_tokens,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "unexpected story plan suggestion failure id=%s",
                suggestion_id,
            )
            await asyncio.to_thread(
                self.story_plan_suggestion_service.fail_suggestion,
                suggestion_id=suggestion_id,
                claim_token=claim_token,
                error="生成全书方案时发生内部错误，请重试",
            )

    async def _process_story_structure_suggestion(
        self, item: dict
    ) -> None:
        suggestion_id = str(item["id"])
        claim_token = str(item["claim_token"])
        try:
            response = await self._plan_story_structure(
                item=item,
                context=dict(item.get("context_snapshot") or {}),
            )
            accepted = await asyncio.to_thread(
                self.story_structure_suggestion_service.complete_suggestion,
                suggestion_id=suggestion_id,
                claim_token=claim_token,
                result=response.result,
                raw_response=response.raw_response,
                provider=response.provider,
                model=response.model,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
            if not accepted:
                logger.warning(
                    "discarded stale story structure suggestion result id=%s",
                    suggestion_id,
                )
        except AnalyzerError as exc:
            logger.warning(
                "story structure suggestion failed id=%s: %s",
                suggestion_id,
                exc,
            )
            await asyncio.to_thread(
                self.story_structure_suggestion_service.fail_suggestion,
                suggestion_id=suggestion_id,
                claim_token=claim_token,
                error=str(exc),
                input_tokens=exc.input_tokens,
                output_tokens=exc.output_tokens,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "unexpected story structure suggestion failure id=%s",
                suggestion_id,
            )
            await asyncio.to_thread(
                self.story_structure_suggestion_service.fail_suggestion,
                suggestion_id=suggestion_id,
                claim_token=claim_token,
                error="生成分卷与滚动章节骨架时发生内部错误，请重试",
            )

    async def _process_causal_suggestion(self, item: dict) -> None:
        suggestion_id = str(item["id"])
        claim_token = str(item["claim_token"])
        try:
            response = await self._plan_causal_suggestions(
                item=item,
                context=dict(item.get("context_snapshot") or {}),
            )
            accepted = await asyncio.to_thread(
                self.causal_suggestion_service.complete_suggestion,
                suggestion_id=suggestion_id,
                claim_token=claim_token,
                result=response.result,
                raw_response=response.raw_response,
                provider=response.provider,
                model=response.model,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
            if not accepted:
                logger.warning(
                    "discarded stale causal suggestion result id=%s",
                    suggestion_id,
                )
        except AnalyzerError as exc:
            logger.warning(
                "causal suggestion failed id=%s: %s",
                suggestion_id,
                exc,
            )
            await asyncio.to_thread(
                self.causal_suggestion_service.fail_suggestion,
                suggestion_id=suggestion_id,
                claim_token=claim_token,
                error=str(exc),
                input_tokens=exc.input_tokens,
                output_tokens=exc.output_tokens,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "unexpected causal suggestion failure id=%s",
                suggestion_id,
            )
            await asyncio.to_thread(
                self.causal_suggestion_service.fail_suggestion,
                suggestion_id=suggestion_id,
                claim_token=claim_token,
                error="生成跨章节因果建议时发生内部错误，请重试",
            )

    async def _process_causal_branch_simulation(
        self,
        item: dict,
    ) -> None:
        simulation_id = str(item["id"])
        claim_token = str(item["claim_token"])
        try:
            response = await self._simulate_causal_branches(
                item=item,
                context=dict(item.get("context_snapshot") or {}),
            )
            accepted = await asyncio.to_thread(
                self.causal_branch_service.complete_simulation,
                simulation_id=simulation_id,
                claim_token=claim_token,
                result=response.result,
                raw_response=response.raw_response,
                provider=response.provider,
                model=response.model,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
            if not accepted:
                logger.warning(
                    "discarded stale causal branch simulation result id=%s",
                    simulation_id,
                )
        except AnalyzerError as exc:
            logger.warning(
                "causal branch simulation failed id=%s: %s",
                simulation_id,
                exc,
            )
            await asyncio.to_thread(
                self.causal_branch_service.fail_simulation,
                simulation_id=simulation_id,
                claim_token=claim_token,
                error=str(exc),
                input_tokens=exc.input_tokens,
                output_tokens=exc.output_tokens,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "unexpected causal branch simulation failure id=%s",
                simulation_id,
            )
            await asyncio.to_thread(
                self.causal_branch_service.fail_simulation,
                simulation_id=simulation_id,
                claim_token=claim_token,
                error="生成长期因果分支时发生内部错误，请重试",
            )

    async def _process_edit_preference_suggestion(
        self, item: dict
    ) -> None:
        suggestion_id = str(item["id"])
        claim_token = str(item["claim_token"])
        try:
            try:
                before_text, after_text = await asyncio.to_thread(
                    self.preference_service.load_running_source_pair,
                    suggestion_id=suggestion_id,
                    claim_token=claim_token,
                )
            except ValueError as exc:
                raise AnalyzerError(str(exc)) from exc
            response = await self._extract_edit_preferences(
                item, before_text, after_text
            )
            located, dropped = locate_edit_preference_evidence(
                before_text,
                after_text,
                item["change_sample"],
                response.result,
            )
            if not located:
                raise AnalyzerError(
                    "编辑偏好建议缺少可在实际改稿中逐字核对的变化证据",
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                )
            suggestion = response.result.model_dump(mode="json")
            suggestion["preferences"] = located
            accepted = await asyncio.to_thread(
                self.preference_service.complete_suggestion,
                suggestion_id=suggestion_id,
                claim_token=claim_token,
                suggestion=suggestion,
                raw_response=response.raw_response,
                provider=response.provider,
                model=response.model,
                valid_evidence_count=len(located),
                dropped_evidence_count=dropped,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
            if not accepted:
                logger.warning(
                    "discarded stale editing preference result id=%s",
                    suggestion_id,
                )
        except AnalyzerError as exc:
            logger.warning(
                "editing preference extraction failed id=%s: %s",
                suggestion_id,
                exc,
            )
            await asyncio.to_thread(
                self.preference_service.fail_suggestion,
                suggestion_id=suggestion_id,
                claim_token=claim_token,
                error=str(exc),
                input_tokens=exc.input_tokens,
                output_tokens=exc.output_tokens,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "unexpected editing preference extraction failure id=%s",
                suggestion_id,
            )
            await asyncio.to_thread(
                self.preference_service.fail_suggestion,
                suggestion_id=suggestion_id,
                claim_token=claim_token,
                error="处理作者手工改稿时发生内部错误，请重试",
            )

    async def _process_generation(self, item: dict) -> None:
        job_id = str(item["id"])
        claim_token = str(item["claim_token"])
        try:
            if str(item["operation"]) == "extract_story_delta":
                await self._process_memory_extraction(item)
                return
            if str(item["operation"]) == "plan_chapter":
                await self._process_chapter_planning(item)
                return
            if str(item["operation"]) == "plan_scene_beats":
                await self._process_scene_beat_planning(item)
                return
            if str(item["operation"]) == "propose_reader_branches":
                await self._process_reader_planning(item)
                return
            if str(item["operation"]) == "audit_ai_style":
                await self._process_style_audit(item)
                return
            if str(item["operation"]) == "rewrite_style_issue":
                await self._process_style_rewrite(item)
                return
            if str(item["operation"]) in {
                "generate_scene",
                "rewrite_scene",
            }:
                await self._process_scene_generation(item)
                return
            context = await asyncio.to_thread(
                self.database.get_writing_context,
                int(item["user_id"]),
                str(item["chapter_id"]),
                None,
                str(item["instruction"] or ""),
            )
            if not context:
                raise AnalyzerError("写作项目或章节不存在")
            if not context.get("task_card"):
                raise AnalyzerError(
                    "章节任务卡尚未确认，已阻止向 DeepSeek 发送写作请求"
                )
            content_path = Path(str(item["content_path"]))
            current_content = await asyncio.to_thread(
                self._read_optional_text, content_path
            )
            previous = context.get("previous_chapter")
            previous_content = ""
            if previous and previous.get("content_path"):
                previous_content = await asyncio.to_thread(
                    self._read_optional_text,
                    Path(str(previous["content_path"])),
                )
            context = {
                **context,
                "canonical_memory": compile_canonical_memory(
                    context.get("canonical_memory") or {}
                ),
                "active_techniques": compile_active_techniques(
                    context.get("technique_cards") or [], usage="write"
                ),
            }
            context_recorded = await asyncio.to_thread(
                self.database.record_generation_context_snapshot,
                job_id=job_id,
                claim_token=claim_token,
                snapshot=build_writing_context_snapshot(
                    context=context,
                    operation=str(item["operation"]),
                    instruction=str(item["instruction"] or ""),
                    current_content=current_content,
                    previous_content=previous_content,
                ),
            )
            if not context_recorded:
                raise AnalyzerError(
                    "写作任务已失效，未向 DeepSeek 发送正文"
                )
            response = await self._write(
                item=item,
                context=context,
                current_content=current_content,
                previous_content=previous_content,
            )
            total_input_tokens = response.input_tokens
            total_output_tokens = response.output_tokens
            warnings: list[str] = []
            if response.truncated:
                warnings.append("本次写作输出达到 token 上限")
            if str(item["operation"]) == "continue" and current_content.strip():
                final_content = (
                    current_content.rstrip() + "\n\n" + response.content.lstrip()
                )
            else:
                final_content = response.content.strip()

            version_path = await asyncio.to_thread(
                self._persist_generated_content,
                content_path,
                job_id,
                final_content,
            )
            version_id = await asyncio.to_thread(
                self.database.complete_generation,
                job_id=job_id,
                claim_token=claim_token,
                version_path=version_path,
                result_char_count=len(final_content),
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                content_hash=hashlib.sha256(
                    final_content.encode("utf-8")
                ).hexdigest(),
                warning="；".join(warnings),
                accept_as_canonical=True,
            )
            if not version_id:
                logger.warning("discarded stale generation result id=%s", job_id)
                return
            try:
                await asyncio.to_thread(
                    self.database.create_memory_extraction_job,
                    user_id=int(item["user_id"]),
                    project_id=str(item["project_id"]),
                    chapter_id=str(item["chapter_id"]),
                    version_id=version_id,
                    provider=str(item["provider"]),
                    model=str(item["model"]),
                    credential_source=str(item["credential_source"]),
                )
            except Exception:
                logger.exception(
                    "background memory extraction was not queued "
                    "chapter=%s",
                    item["chapter_id"],
                )
        except AnalyzerError as exc:
            logger.warning("writing task failed id=%s: %s", job_id, exc)
            await asyncio.to_thread(
                self.database.fail_generation,
                job_id,
                claim_token,
                str(exc),
                exc.input_tokens,
                exc.output_tokens,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("unexpected writing task failure id=%s", job_id)
            await asyncio.to_thread(
                self.database.fail_generation,
                job_id,
                claim_token,
                (
                    "提取故事记忆时发生内部错误，请重试"
                    if str(item["operation"]) == "extract_story_delta"
                    else "生成章节时发生内部错误，请重试"
                ),
            )

    async def _process_scene_generation(self, item: dict) -> None:
        job_id = str(item["id"])
        scene_beat_id = str(item["subject_id"] or "")
        if not scene_beat_id:
            raise AnalyzerError("场景写作任务没有指定场景节拍")
        state = await asyncio.to_thread(
            self.scene_service.get_generation_state,
            user_id=int(item["user_id"]),
            project_id=str(item["project_id"]),
            chapter_id=str(item["chapter_id"]),
            scene_beat_id=scene_beat_id,
        )
        context = await asyncio.to_thread(
            self.database.get_writing_context,
            int(item["user_id"]),
            str(item["chapter_id"]),
            scene_beat_id,
            str(item["instruction"] or ""),
        )
        if not context or not context.get("task_card"):
            raise AnalyzerError("章节任务卡尚未确认")
        previous_chapter = context.get("previous_chapter")
        previous_chapter_content = ""
        if previous_chapter and previous_chapter.get("content_path"):
            previous_chapter_content = await asyncio.to_thread(
                self._read_optional_text,
                Path(str(previous_chapter["content_path"])),
            )
        context = {
            **context,
            "canonical_memory": compile_canonical_memory(
                context.get("canonical_memory") or {}
            ),
            "active_techniques": compile_active_techniques(
                context.get("technique_cards") or [], usage="write"
            ),
            "focused_scene": state["focused_scene"],
            "scene_sequence": state["scene_sequence"],
            "previous_scene": state["previous_scene"],
            "next_scene": state["next_scene"],
            "previous_scene_content": state["previous_scene_content"],
            "previous_chapter_content": previous_chapter_content,
            "scene_target_chars": state["target_chars"],
            "scene_minimum_chars": state["minimum_chars"],
        }
        context_recorded = await asyncio.to_thread(
            self.database.record_generation_context_snapshot,
            job_id=job_id,
            claim_token=str(item["claim_token"]),
            snapshot=build_scene_context_snapshot(
                context=context,
                operation=str(item["operation"]),
                instruction=str(item["instruction"] or ""),
                current_scene_content=state["current_content"],
                previous_scene_content=state["previous_scene_content"],
                previous_chapter_content=previous_chapter_content,
            ),
        )
        if not context_recorded:
            raise AnalyzerError(
                "场景写作任务已失效，未向 DeepSeek 发送正文"
            )
        response = await self._write(
            item=item,
            context=context,
            current_content=state["current_content"],
            previous_content=state["previous_scene_content"],
        )
        final_content = response.content.strip()
        if not final_content:
            raise AnalyzerError(
                "DeepSeek 没有返回场景正文",
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
        total_input_tokens = response.input_tokens
        total_output_tokens = response.output_tokens
        warnings: list[str] = []
        if response.truncated:
            warnings.append("本次场景写作输出达到 token 上限")

        version_path = await asyncio.to_thread(
            self._persist_generated_scene,
            Path(str(state["chapter_content_path"])),
            scene_beat_id,
            job_id,
            final_content,
        )
        version_id = await asyncio.to_thread(
            self.scene_service.complete_generation,
            job_id=job_id,
            claim_token=str(item["claim_token"]),
            version_path=version_path,
            content=final_content,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            warning="；".join(warnings),
        )
        if not version_id:
            logger.warning("discarded stale scene generation id=%s", job_id)

    async def _process_memory_extraction(self, item: dict) -> None:
        job_id = str(item["id"])
        version_id = str(item["version_id"] or "")
        version = await asyncio.to_thread(
            self.database.get_chapter_version,
            int(item["user_id"]),
            str(item["project_id"]),
            str(item["chapter_id"]),
            version_id,
        )
        if not version:
            raise AnalyzerError("要提取故事记忆的正文版本不存在")
        if str(version["canonical_version_id"] or "") != version_id:
            raise AnalyzerError("正文正史已经变化，请重新提取故事记忆")
        context = await asyncio.to_thread(
            self.database.get_writing_context,
            int(item["user_id"]),
            str(item["chapter_id"]),
        )
        if not context:
            raise AnalyzerError("写作项目或章节不存在")
        chapter_text = await asyncio.to_thread(
            self._read_optional_text, Path(str(version["content_path"]))
        )
        response = await self._extract_memory(
            item=item,
            context=context,
            chapter_text=chapter_text,
        )
        delta_id = await asyncio.to_thread(
            self.memory_service.create_proposal,
            user_id=int(item["user_id"]),
            project_id=str(item["project_id"]),
            chapter_id=str(item["chapter_id"]),
            version_id=version_id,
            payload=response.result,
        )
        projected = await asyncio.to_thread(
            self.memory_service.accept_delta,
            user_id=int(item["user_id"]),
            delta_id=delta_id,
        )
        if not projected:
            raise AnalyzerError("故事记忆自动写入失败")
        accepted = await asyncio.to_thread(
            self.database.complete_memory_extraction,
            job_id=job_id,
            claim_token=str(item["claim_token"]),
            result={
                "delta_id": delta_id,
                "projected": True,
                "story_delta": response.result.model_dump(mode="json"),
            },
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
        if not accepted:
            logger.warning(
                "discarded stale memory extraction result id=%s", job_id
            )

    async def _process_chapter_planning(self, item: dict) -> None:
        job_id = str(item["id"])
        context = await asyncio.to_thread(
            self.database.get_writing_context,
            int(item["user_id"]),
            str(item["chapter_id"]),
            None,
            str(item["instruction"] or ""),
        )
        if not context:
            raise AnalyzerError("写作项目或章节不存在")
        context = {
            **context,
            "canonical_memory": compile_canonical_memory(
                context.get("canonical_memory") or {}
            ),
            "active_techniques": compile_active_techniques(
                context.get("technique_cards") or [], usage="plan"
            ),
        }
        context_recorded = await asyncio.to_thread(
            self.database.record_generation_context_snapshot,
            job_id=job_id,
            claim_token=str(item["claim_token"]),
            snapshot={
                "schema_version": 1,
                "operation": "plan_chapter",
                "instruction": str(item["instruction"] or ""),
                "chapter": dict(context["chapter"]),
                "characters": list(context.get("characters") or []),
                "canonical_memory": context["canonical_memory"],
                "active_techniques": context["active_techniques"],
                "confirmed_story_plan": compile_story_plan_context(
                    context, usage="plan"
                ),
                "planned_causal_links": compile_planned_causal_links(
                    context, usage="plan"
                ),
            },
        )
        if not context_recorded:
            raise AnalyzerError(
                "章节规划任务已失效，未向 DeepSeek 发送资料"
            )
        response = await self._plan_chapter(
            item=item,
            context=context,
        )
        plan_id = await asyncio.to_thread(
            self.planning_service.upsert_task_card,
            user_id=int(item["user_id"]),
            project_id=str(item["project_id"]),
            chapter_id=str(item["chapter_id"]),
            volume_id=(
                str(context["chapter"]["volume_id"])
                if context["chapter"].get("volume_id")
                else None
            ),
            card=response.result,
            confirm=False,
            source="ai",
            allow_active_plan_job=True,
        )
        accepted = await asyncio.to_thread(
            self.database.complete_chapter_planning,
            job_id=job_id,
            claim_token=str(item["claim_token"]),
            result={
                "plan_id": plan_id,
                "task_card": response.result.model_dump(mode="json"),
            },
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
        if not accepted:
            logger.warning(
                "discarded stale chapter planning result id=%s", job_id
            )

    async def _process_scene_beat_planning(self, item: dict) -> None:
        job_id = str(item["id"])
        task_card_data = await asyncio.to_thread(
            self.planning_service.get_task_card,
            user_id=int(item["user_id"]),
            project_id=str(item["project_id"]),
            chapter_id=str(item["chapter_id"]),
        )
        if not task_card_data or not task_card_data.get("plan_id"):
            raise AnalyzerError(
                "请先保存章节任务卡，再让 Planner 只拆分场景"
            )
        task_card = ChapterTaskCard.model_validate(
            chapter_task_card_payload(task_card_data)
        )
        baseline_fingerprint = chapter_task_card_fingerprint(task_card)
        retrieval_hint = "\n".join(
            [
                task_card.purpose,
                task_card.central_conflict,
                *task_card.plot_threads,
                *task_card.must_happen,
                *task_card.foreshadow_setup,
                *task_card.foreshadow_payoff,
                task_card.ending_hook,
            ]
        )
        context = await asyncio.to_thread(
            self.database.get_writing_context,
            int(item["user_id"]),
            str(item["chapter_id"]),
            None,
            retrieval_hint,
        )
        if not context:
            raise AnalyzerError("写作项目或章节不存在")
        context = {
            **context,
            "canonical_memory": compile_canonical_memory(
                context.get("canonical_memory") or {}
            ),
            "active_techniques": compile_active_techniques(
                context.get("technique_cards") or [], usage="plan"
            ),
        }
        context_recorded = await asyncio.to_thread(
            self.database.record_generation_context_snapshot,
            job_id=job_id,
            claim_token=str(item["claim_token"]),
            snapshot={
                "schema_version": 2,
                "operation": "plan_scene_beats",
                "instruction": str(item["instruction"] or ""),
                "task_card_fingerprint": baseline_fingerprint,
                "locked_task_card": task_card.model_dump(mode="json"),
                "chapter": dict(context["chapter"]),
                "characters": list(context.get("characters") or []),
                "canonical_memory": context["canonical_memory"],
                "active_techniques": context["active_techniques"],
                "confirmed_story_plan": compile_story_plan_context(
                    context, usage="plan"
                ),
                "planned_causal_links": compile_planned_causal_links(
                    context, usage="plan"
                ),
            },
        )
        if not context_recorded:
            raise AnalyzerError(
                "场景拆解任务已失效，未向 DeepSeek 发送资料"
            )
        response = await self._plan_scene_beats(
            item=item,
            context=context,
            task_card=task_card,
        )
        response.result.ensure_covers(task_card)
        updated_card = ChapterTaskCard.model_validate(
            {
                **task_card.model_dump(mode="json"),
                "scenes": [
                    scene.model_dump(mode="json")
                    for scene in response.result.scenes
                ],
            }
        )
        try:
            plan_id = await asyncio.to_thread(
                self.planning_service.upsert_task_card,
                user_id=int(item["user_id"]),
                project_id=str(item["project_id"]),
                chapter_id=str(item["chapter_id"]),
                volume_id=(
                    str(task_card_data["volume_id"])
                    if task_card_data.get("volume_id")
                    else None
                ),
                card=updated_card,
                confirm=False,
                source="ai",
                allow_active_plan_job=True,
                expected_card_fingerprint=baseline_fingerprint,
            )
        except ValueError as exc:
            raise AnalyzerError(str(exc)) from exc
        accepted = await asyncio.to_thread(
            self.database.complete_chapter_planning,
            job_id=job_id,
            claim_token=str(item["claim_token"]),
            operation="plan_scene_beats",
            result={
                "plan_id": plan_id,
                "task_card_fingerprint": baseline_fingerprint,
                "scene_beat_plan": response.result.model_dump(mode="json"),
                "preserved_task_card": {
                    name: value
                    for name, value in task_card.model_dump(
                        mode="json"
                    ).items()
                    if name != "scenes"
                },
            },
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
        if not accepted:
            logger.warning(
                "discarded stale scene beat planning result id=%s",
                job_id,
            )

    async def _process_reader_planning(self, item: dict) -> None:
        job_id = str(item["id"])
        request_id = str(item["subject_id"] or "")
        context = await asyncio.to_thread(
            self.reader_service.build_planning_context,
            user_id=int(item["user_id"]),
            request_id=request_id,
        )
        if not context:
            raise AnalyzerError("读者意见或小说项目不存在")
        context = {
            **context,
            "canonical_memory": compile_canonical_memory(
                context.get("canonical_memory") or {}
            ),
        }
        context_recorded = await asyncio.to_thread(
            self.database.record_generation_context_snapshot,
            job_id=job_id,
            claim_token=str(item["claim_token"]),
            snapshot={
                "schema_version": 1,
                "operation": "propose_reader_branches",
                "request": dict(context["request"]),
                "project": dict(context["project"]),
                "current_position": int(context["current_position"]),
                "planning_horizon": int(context["planning_horizon"]),
                "future_chapters": list(
                    context.get("future_chapters") or []
                ),
                "characters": list(context.get("characters") or []),
                "volumes": list(context.get("volumes") or []),
                "confirmed_story_blueprint": dict(
                    context.get("confirmed_story_blueprint") or {}
                ),
                "planned_plot_arcs": list(
                    context.get("planned_plot_arcs") or []
                ),
                "canonical_memory": context["canonical_memory"],
            },
        )
        if not context_recorded:
            raise AnalyzerError(
                "读者意见规划任务已失效，未向 DeepSeek 发送资料"
            )
        response = await self._plan_reader_request(
            item=item, context=context
        )
        proposal_ids = await asyncio.to_thread(
            self.reader_service.save_proposals,
            user_id=int(item["user_id"]),
            request_id=request_id,
            job_id=job_id,
            result=response.result,
            provider=response.provider,
            model=response.model,
        )
        accepted = await asyncio.to_thread(
            self.database.complete_reader_planning,
            job_id=job_id,
            claim_token=str(item["claim_token"]),
            result={
                "request_id": request_id,
                "proposal_ids": proposal_ids,
                "analysis_summary": response.result.analysis_summary,
            },
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
        if not accepted:
            logger.warning(
                "discarded stale reader planning result id=%s", job_id
            )

    async def _process_style_audit(self, item: dict) -> None:
        job_id = str(item["id"])
        version_id = str(item["version_id"] or "")
        version = await asyncio.to_thread(
            self.database.get_chapter_version,
            int(item["user_id"]),
            str(item["project_id"]),
            str(item["chapter_id"]),
            version_id,
        )
        if not version:
            raise AnalyzerError("要审校的正文版本不存在")
        context = await asyncio.to_thread(
            self.database.get_writing_context,
            int(item["user_id"]),
            str(item["chapter_id"]),
        )
        if not context:
            raise AnalyzerError("写作项目或章节不存在")
        context = {
            **context,
            "canonical_memory": compile_canonical_memory(
                context.get("canonical_memory") or {}
            ),
            "active_techniques": compile_active_techniques(
                context.get("technique_cards") or [], usage="audit"
            ),
        }
        voice_profile = await asyncio.to_thread(
            self.style_service.get_voice_profile,
            user_id=int(item["user_id"]),
            project_id=str(item["project_id"]),
        )
        if not voice_profile or voice_profile["status"] != "confirmed":
            raise AnalyzerError("请先确认作品声纹，再执行 AI 味审校")
        preferences = await asyncio.to_thread(
            self.style_service.list_preferences,
            user_id=int(item["user_id"]),
            project_id=str(item["project_id"]),
            limit=20,
        )
        chapter_text = await asyncio.to_thread(
            self._read_optional_text, Path(str(version["content_path"]))
        )
        context_recorded = await asyncio.to_thread(
            self.database.record_generation_context_snapshot,
            job_id=job_id,
            claim_token=str(item["claim_token"]),
            snapshot={
                "schema_version": 1,
                "operation": "audit_ai_style",
                "version_id": version_id,
                "chapter": dict(context["chapter"]),
                "voice_profile": dict(voice_profile),
                "recent_author_preferences": list(preferences),
                "active_techniques": context["active_techniques"],
                "candidate_content_hash": hashlib.sha256(
                    chapter_text.encode("utf-8")
                ).hexdigest(),
            },
        )
        if not context_recorded:
            raise AnalyzerError(
                "AI 味审校任务已失效，未向 DeepSeek 发送正文"
            )
        response = await self._audit_style(
            item=item,
            context=context,
            chapter_text=chapter_text,
            voice_profile=voice_profile,
            preferences=preferences,
        )
        located, dropped = locate_style_issues(
            chapter_text, response.result
        )
        audit_id = await asyncio.to_thread(
            self.style_service.create_audit,
            user_id=int(item["user_id"]),
            project_id=str(item["project_id"]),
            chapter_id=str(item["chapter_id"]),
            version_id=version_id,
            result=response.result,
            located_issues=located,
            dropped_issue_count=dropped,
            provider=response.provider,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
        accepted = await asyncio.to_thread(
            self.database.complete_style_job,
            job_id=job_id,
            claim_token=str(item["claim_token"]),
            result={
                "audit_id": audit_id,
                "version_id": version_id,
                "issue_count": len(located),
                "dropped_issue_count": dropped,
            },
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
        if not accepted:
            logger.warning("discarded stale style audit result id=%s", job_id)

    async def _process_style_rewrite(self, item: dict) -> None:
        job_id = str(item["id"])
        issue_id = str(item["subject_id"] or "")
        issue = await asyncio.to_thread(
            self.style_service.get_issue,
            user_id=int(item["user_id"]),
            issue_id=issue_id,
        )
        if not issue or str(issue["status"]) != "open":
            raise AnalyzerError("要改写的问题不存在或已经处理")
        context = await asyncio.to_thread(
            self.database.get_writing_context,
            int(item["user_id"]),
            str(item["chapter_id"]),
        )
        if not context:
            raise AnalyzerError("写作项目或章节不存在")
        context = {
            **context,
            "active_techniques": compile_active_techniques(
                context.get("technique_cards") or [], usage="audit"
            ),
        }
        voice_profile = await asyncio.to_thread(
            self.style_service.get_voice_profile,
            user_id=int(item["user_id"]),
            project_id=str(item["project_id"]),
        )
        if not voice_profile or voice_profile["status"] != "confirmed":
            raise AnalyzerError("作品声纹尚未确认")
        chapter_text = await asyncio.to_thread(
            self._read_optional_text,
            Path(str(issue["version_content_path"])),
        )
        start_offset = int(issue["start_offset"])
        end_offset = int(issue["end_offset"])
        if chapter_text[start_offset:end_offset] != str(issue["quote"]):
            raise AnalyzerError(
                "原版本正文与问题定位不再一致，请重新执行 AI 味审校"
            )
        excerpt = surrounding_excerpt(
            chapter_text, start_offset, end_offset
        )
        context_recorded = await asyncio.to_thread(
            self.database.record_generation_context_snapshot,
            job_id=job_id,
            claim_token=str(item["claim_token"]),
            snapshot={
                "schema_version": 1,
                "operation": "rewrite_style_issue",
                "version_id": str(issue["version_id"]),
                "issue_id": issue_id,
                "chapter": dict(context["chapter"]),
                "voice_profile": dict(voice_profile),
                "active_techniques": context["active_techniques"],
                "issue": {
                    key: issue[key]
                    for key in (
                        "quote",
                        "issue_type",
                        "evidence",
                        "reader_impact",
                        "rewrite_direction",
                        "start_offset",
                        "end_offset",
                    )
                },
                "surrounding_text": excerpt,
            },
        )
        if not context_recorded:
            raise AnalyzerError(
                "定点改写任务已失效，未向 DeepSeek 发送正文"
            )
        response = await self._rewrite_style(
            item=item,
            context=context,
            issue=issue,
            surrounding_text=excerpt,
            voice_profile=voice_profile,
        )
        candidate_ids = await asyncio.to_thread(
            self.style_service.save_rewrite_candidates,
            user_id=int(item["user_id"]),
            issue_id=issue_id,
            result=response.result,
            provider=response.provider,
            model=response.model,
        )
        accepted = await asyncio.to_thread(
            self.database.complete_style_job,
            job_id=job_id,
            claim_token=str(item["claim_token"]),
            result={
                "issue_id": issue_id,
                "candidate_ids": candidate_ids,
            },
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
        if not accepted:
            logger.warning("discarded stale style rewrite result id=%s", job_id)

    @staticmethod
    def _read_optional_text(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    @staticmethod
    def _persist_generated_content(
        content_path: Path, job_id: str, content: str
    ) -> Path:
        content_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        versions_dir = content_path.parent / "versions"
        versions_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(content_path.parent, 0o700)
        os.chmod(versions_dir, 0o700)
        version_path = versions_dir / f"{job_id}.txt"
        version_path.write_text(content, encoding="utf-8")
        version_path.chmod(0o600)
        temporary_path = content_path.with_name(f".{content_path.name}.{job_id}.tmp")
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.chmod(0o600)
        os.replace(temporary_path, content_path)
        content_path.chmod(0o600)
        return version_path

    @staticmethod
    def _persist_generated_scene(
        chapter_content_path: Path,
        scene_beat_id: str,
        job_id: str,
        content: str,
    ) -> Path:
        versions_dir = (
            chapter_content_path.parent
            / "scenes"
            / scene_beat_id
            / "versions"
        )
        versions_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(versions_dir.parent, 0o700)
        os.chmod(versions_dir, 0o700)
        version_path = versions_dir / f"{job_id}.txt"
        temporary_path = versions_dir / f".{job_id}.tmp"
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.chmod(0o600)
        os.replace(temporary_path, version_path)
        version_path.chmod(0o600)
        return version_path

    async def _write(
        self,
        *,
        item: dict,
        context: dict,
        current_content: str,
        previous_content: str,
    ):
        user_id = int(item["user_id"])
        provider_user_id = stable_provider_user_id(
            user_id, self.provider_user_secret
        )
        writer = self.writer
        close_writer = False
        if item.get("credential_source") == "personal":
            personal_settings = await self._personal_model_settings(
                item, "reasoning"
            )
            writer = DeepSeekWriter(personal_settings)
            close_writer = True
        elif self.writer is None or (
            str(item["provider"]) != self.writer.provider
        ):
            raise AnalyzerError("任务所需的服务器 DeepSeek API Key 当前未配置")

        try:
            return await writer.write(
                context=context,
                operation=str(item["operation"]),
                instruction=str(item["instruction"] or ""),
                current_content=current_content,
                previous_content=previous_content,
                provider_user_id=provider_user_id,
            )
        finally:
            if close_writer:
                await writer.close()

    async def _extract_memory(
        self,
        *,
        item: dict,
        context: dict,
        chapter_text: str,
    ):
        user_id = int(item["user_id"])
        provider_user_id = stable_provider_user_id(
            user_id, self.provider_user_secret
        )
        extractor = self.memory_extractor
        close_extractor = False
        if item.get("credential_source") == "personal":
            personal_settings = await self._personal_model_settings(
                item, "fast"
            )
            extractor = DeepSeekMemoryExtractor(personal_settings)
            close_extractor = True
        elif self.memory_extractor is None or (
            str(item["provider"]) != self.memory_extractor.provider
        ):
            raise AnalyzerError(
                "任务所需的服务器 DeepSeek API Key 当前未配置"
            )
        try:
            return await extractor.extract(
                context=context,
                chapter_text=chapter_text,
                provider_user_id=provider_user_id,
            )
        finally:
            if close_extractor:
                await extractor.close()

    async def _plan_chapter(
        self,
        *,
        item: dict,
        context: dict,
    ):
        user_id = int(item["user_id"])
        provider_user_id = stable_provider_user_id(
            user_id, self.provider_user_secret
        )
        planner = self.chapter_planner
        close_planner = False
        if item.get("credential_source") == "personal":
            personal_settings = await self._personal_model_settings(
                item, "deep"
            )
            planner = DeepSeekChapterPlanner(personal_settings)
            close_planner = True
        elif self.chapter_planner is None or (
            str(item["provider"]) != self.chapter_planner.provider
        ):
            raise AnalyzerError(
                "任务所需的服务器 DeepSeek API Key 当前未配置"
            )
        try:
            return await planner.propose(
                context=context,
                instruction=str(item["instruction"] or ""),
                provider_user_id=provider_user_id,
            )
        finally:
            if close_planner:
                await planner.close()

    async def _plan_scene_beats(
        self,
        *,
        item: dict,
        context: dict,
        task_card: ChapterTaskCard,
    ):
        user_id = int(item["user_id"])
        provider_user_id = stable_provider_user_id(
            user_id, self.provider_user_secret
        )
        planner = self.chapter_planner
        close_planner = False
        if item.get("credential_source") == "personal":
            personal_settings = await self._personal_model_settings(
                item, "deep"
            )
            planner = DeepSeekChapterPlanner(personal_settings)
            close_planner = True
        elif self.chapter_planner is None or (
            str(item["provider"]) != self.chapter_planner.provider
        ):
            raise AnalyzerError(
                "任务所需的服务器 DeepSeek API Key 当前未配置"
            )
        try:
            return await planner.propose_scene_beats(
                context=context,
                task_card=task_card,
                instruction=str(item["instruction"] or ""),
                provider_user_id=provider_user_id,
            )
        finally:
            if close_planner:
                await planner.close()

    async def _plan_story(
        self,
        *,
        item: dict,
        context: dict,
    ):
        user_id = int(item["user_id"])
        provider_user_id = stable_provider_user_id(
            user_id, self.provider_user_secret
        )
        planner = self.story_planner
        close_planner = False
        if item.get("credential_source") == "personal":
            personal_settings = await self._personal_model_settings(
                item, "deep"
            )
            planner = DeepSeekStoryPlanner(personal_settings)
            close_planner = True
        elif self.story_planner is None or (
            str(item["provider"]) != self.story_planner.provider
        ):
            raise AnalyzerError(
                "任务所需的服务器 DeepSeek API Key 当前未配置"
            )
        try:
            return await planner.propose(
                context=context,
                mode=str(item["planning_mode"]),
                instruction=str(item["instruction"] or ""),
                provider_user_id=provider_user_id,
            )
        finally:
            if close_planner:
                await planner.close()

    async def _plan_story_structure(
        self,
        *,
        item: dict,
        context: dict,
    ):
        user_id = int(item["user_id"])
        provider_user_id = stable_provider_user_id(
            user_id, self.provider_user_secret
        )
        planner = self.story_structure_planner
        close_planner = False
        if item.get("credential_source") == "personal":
            personal_settings = await self._personal_model_settings(
                item, "deep"
            )
            planner = DeepSeekStoryStructurePlanner(personal_settings)
            close_planner = True
        elif self.story_structure_planner is None or (
            str(item["provider"])
            != self.story_structure_planner.provider
        ):
            raise AnalyzerError(
                "任务所需的服务器 DeepSeek API Key 当前未配置"
            )
        try:
            return await planner.propose(
                context=context,
                instruction=str(item["instruction"] or ""),
                provider_user_id=provider_user_id,
            )
        finally:
            if close_planner:
                await planner.close()

    async def _plan_causal_suggestions(
        self,
        *,
        item: dict,
        context: dict,
    ):
        user_id = int(item["user_id"])
        provider_user_id = stable_provider_user_id(
            user_id, self.provider_user_secret
        )
        planner = self.causal_suggestion_planner
        close_planner = False
        if item.get("credential_source") == "personal":
            personal_settings = await self._personal_model_settings(
                item, "deep"
            )
            planner = DeepSeekCausalSuggestionPlanner(personal_settings)
            close_planner = True
        elif self.causal_suggestion_planner is None or (
            str(item["provider"])
            != self.causal_suggestion_planner.provider
        ):
            raise AnalyzerError(
                "任务所需的服务器 DeepSeek API Key 当前未配置"
            )
        try:
            return await planner.propose(
                context=context,
                instruction=str(item["instruction"] or ""),
                provider_user_id=provider_user_id,
            )
        finally:
            if close_planner:
                await planner.close()

    async def _simulate_causal_branches(
        self,
        *,
        item: dict,
        context: dict,
    ):
        user_id = int(item["user_id"])
        provider_user_id = stable_provider_user_id(
            user_id,
            self.provider_user_secret,
        )
        planner = self.causal_branch_planner
        close_planner = False
        if item.get("credential_source") == "personal":
            personal_settings = await self._personal_model_settings(
                item, "deep"
            )
            planner = DeepSeekCausalBranchPlanner(personal_settings)
            close_planner = True
        elif self.causal_branch_planner is None or (
            str(item["provider"]) != self.causal_branch_planner.provider
        ):
            raise AnalyzerError(
                "任务所需的服务器 DeepSeek API Key 当前未配置"
            )
        try:
            return await planner.simulate(
                context=context,
                instruction=str(item["instruction"] or ""),
                provider_user_id=provider_user_id,
            )
        finally:
            if close_planner:
                await planner.close()

    async def _plan_reader_request(
        self,
        *,
        item: dict,
        context: dict,
    ):
        user_id = int(item["user_id"])
        provider_user_id = stable_provider_user_id(
            user_id, self.provider_user_secret
        )
        planner = self.reader_planner
        close_planner = False
        if item.get("credential_source") == "personal":
            personal_settings = await self._personal_model_settings(
                item, "reasoning"
            )
            planner = DeepSeekReaderPlanner(personal_settings)
            close_planner = True
        elif self.reader_planner is None or (
            str(item["provider"]) != self.reader_planner.provider
        ):
            raise AnalyzerError(
                "任务所需的服务器 DeepSeek API Key 当前未配置"
            )
        try:
            return await planner.propose(
                context=context,
                request_data=context["request"],
                provider_user_id=provider_user_id,
            )
        finally:
            if close_planner:
                await planner.close()

    async def _audit_style(
        self,
        *,
        item: dict,
        context: dict,
        chapter_text: str,
        voice_profile: dict,
        preferences: list[dict],
    ):
        editor, close_editor = await self._style_editor_for_item(item)
        provider_user_id = stable_provider_user_id(
            int(item["user_id"]), self.provider_user_secret
        )
        try:
            return await editor.audit(
                context=context,
                chapter_text=chapter_text,
                voice_profile=voice_profile,
                preferences=preferences,
                provider_user_id=provider_user_id,
            )
        finally:
            if close_editor:
                await editor.close()

    async def _rewrite_style(
        self,
        *,
        item: dict,
        context: dict,
        issue: dict,
        surrounding_text: str,
        voice_profile: dict,
    ):
        editor, close_editor = await self._style_editor_for_item(item)
        provider_user_id = stable_provider_user_id(
            int(item["user_id"]), self.provider_user_secret
        )
        try:
            return await editor.rewrite(
                context=context,
                issue=issue,
                surrounding_text=surrounding_text,
                voice_profile=voice_profile,
                instruction=str(item["instruction"] or ""),
                provider_user_id=provider_user_id,
            )
        finally:
            if close_editor:
                await editor.close()

    async def _extract_voice_profile(self, item: dict):
        user_id = int(item["user_id"])
        provider_user_id = stable_provider_user_id(
            user_id, self.provider_user_secret
        )
        extractor = self.voice_profile_extractor
        close_extractor = False
        if item.get("credential_source") == "personal":
            personal_settings = await self._personal_model_settings(
                item, "fast"
            )
            extractor = DeepSeekVoiceProfileExtractor(personal_settings)
            close_extractor = True
        elif self.voice_profile_extractor is None or (
            str(item["provider"])
            != self.voice_profile_extractor.provider
        ):
            raise AnalyzerError(
                "任务所需的服务器 DeepSeek API Key 当前未配置"
            )
        try:
            return await extractor.extract(
                project={
                    "title": item.get("title"),
                    "genre": item.get("genre"),
                    "point_of_view": item.get("point_of_view"),
                    "style_guide": item.get("style_guide"),
                    "target_audience": item.get("target_audience"),
                },
                sample_title=str(item["sample_title"]),
                sample_text=str(item["sample_text"]),
                author_intent=str(item["author_intent"] or ""),
                provider_user_id=provider_user_id,
            )
        finally:
            if close_extractor:
                await extractor.close()

    async def _extract_edit_preferences(
        self, item: dict, before_text: str, after_text: str
    ):
        user_id = int(item["user_id"])
        provider_user_id = stable_provider_user_id(
            user_id, self.provider_user_secret
        )
        extractor = self.edit_preference_extractor
        close_extractor = False
        if item.get("credential_source") == "personal":
            personal_settings = await self._personal_model_settings(
                item, "fast"
            )
            extractor = DeepSeekEditPreferenceExtractor(personal_settings)
            close_extractor = True
        elif self.edit_preference_extractor is None or (
            str(item["provider"])
            != self.edit_preference_extractor.provider
        ):
            raise AnalyzerError(
                "任务所需的服务器 DeepSeek API Key 当前未配置"
            )
        try:
            return await extractor.extract(
                project={
                    "title": item.get("title"),
                    "genre": item.get("genre"),
                    "point_of_view": item.get("point_of_view"),
                    "style_guide": item.get("style_guide"),
                },
                source={
                    "source_type": item.get("source_type"),
                    "chapter_title": item.get("chapter_title"),
                    "scene_goal": item.get("scene_goal"),
                    "author_change_summary": item.get(
                        "author_change_summary"
                    ),
                },
                change_sample=item["change_sample"],
                before_text=before_text,
                after_text=after_text,
                provider_user_id=provider_user_id,
            )
        finally:
            if close_extractor:
                await extractor.close()

    async def _style_editor_for_item(
        self, item: dict
    ) -> tuple[BaseStyleEditor, bool]:
        user_id = int(item["user_id"])
        if item.get("credential_source") == "personal":
            personal_settings = await self._personal_model_settings(
                item, "reasoning"
            )
            return DeepSeekStyleEditor(personal_settings), True
        if self.style_editor is None or (
            str(item["provider"]) != self.style_editor.provider
        ):
            raise AnalyzerError(
                "任务所需的服务器 DeepSeek API Key 当前未配置"
            )
        return self.style_editor, False

    async def _analyze(self, item: dict, content: str):
        user_id = int(item["user_id"])
        provider_user_id = stable_provider_user_id(
            user_id, self.provider_user_secret
        )
        if item.get("credential_source") != "personal":
            if self.analyzer is None or (
                str(item["provider"]) != self.analyzer.provider
            ):
                raise AnalyzerError(
                    "任务所需的服务器 DeepSeek API Key 当前未配置"
                )
            return await self.analyzer.analyze(
                str(item["chapter_title"]), content, provider_user_id
            )

        personal_settings = await self._personal_model_settings(
            item, "reasoning"
        )
        analyzer = DeepSeekAnalyzer(personal_settings)
        try:
            return await analyzer.analyze(
                str(item["chapter_title"]), content, provider_user_id
            )
        finally:
            await analyzer.close()

    async def _reply_assistant_chat(
        self,
        *,
        item: dict,
        payload: dict,
        on_answer_update: AnswerUpdateCallback | None = None,
    ):
        user_id = int(item["user_id"])
        provider_user_id = stable_provider_user_id(
            user_id, self.provider_user_secret
        )
        model = self.assistant_chat_model
        routing_model = model
        role_models: dict[str, BaseAssistantChatModel] = {}
        owned_models: list[BaseAssistantChatModel] = []
        if item.get("credential_source") == "personal":
            if not (item.get("id") and item.get("claim_token")):
                personal_settings = await self._personal_model_settings(
                    item, "discussion"
                )
                model = DeepSeekAssistantChatModel(personal_settings)
                routing_model = model
                owned_models.append(model)
            else:
                (
                    routing_settings,
                    discussion_settings,
                    reasoning_settings,
                    deep_settings,
                ) = await asyncio.gather(
                    self._personal_model_settings(item, "fast"),
                    self._personal_model_settings(item, "discussion"),
                    self._personal_model_settings(item, "reasoning"),
                    self._personal_model_settings(item, "deep"),
                )

                models_by_settings: dict[
                    Settings, BaseAssistantChatModel
                ] = {}

                def model_for(
                    settings: Settings,
                ) -> BaseAssistantChatModel:
                    selected = models_by_settings.get(settings)
                    if selected is None:
                        selected = DeepSeekAssistantChatModel(settings)
                        models_by_settings[settings] = selected
                        owned_models.append(selected)
                    return selected

                routing_model = model_for(routing_settings)
                model = model_for(discussion_settings)
                reasoning_model = model_for(reasoning_settings)
                deep_model = model_for(deep_settings)
                role_models = {
                    "advisor": model,
                    "analyst": reasoning_model,
                    "planner": reasoning_model,
                    "researcher": reasoning_model,
                    "writer": reasoning_model,
                    "editor": reasoning_model,
                    "story_planner": deep_model,
                }
        elif self.assistant_chat_model is None or (
            str(item["provider"]) != self.assistant_chat_model.provider
        ):
            raise AnalyzerError(
                "任务所需的服务器 DeepSeek API Key 当前未配置"
            )
        try:
            if item.get("id") and item.get("claim_token"):
                return await self.assistant_agent_orchestrator.run(
                    model=model,
                    routing_model=routing_model,
                    role_models=role_models,
                    item=item,
                    payload=payload,
                    provider_user_id=provider_user_id,
                    on_answer_update=on_answer_update,
                )
            return await model.reply(
                context=payload["context"],
                sources=payload["sources"],
                history=payload["history"],
                question=payload["question"],
                selected_quote=payload["selected_quote"],
                provider_user_id=provider_user_id,
            )
        finally:
            for owned_model in owned_models:
                await owned_model.close()

    async def stop(self) -> None:
        self._stopping = True
        self._wake.set()
        providers = (
            self.analyzer,
            self.writer,
            self.memory_extractor,
            self.chapter_planner,
            self.style_editor,
            self.reader_planner,
            self.story_planner,
            self.story_structure_planner,
            self.causal_suggestion_planner,
            self.causal_branch_planner,
            self.voice_profile_extractor,
            self.edit_preference_extractor,
            self.assistant_chat_model,
        )
        for provider in providers:
            if provider is not None:
                await provider.close()
