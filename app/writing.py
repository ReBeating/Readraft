from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional

import httpx

from .config import Settings
from .context_compiler import (
    compile_active_techniques,
    compile_canonical_memory,
    compile_planned_causal_links,
    compile_story_plan_context,
)
from .model_client import AnalyzerError
from .model_provider import (
    ProviderConfigError,
    build_chat_payload,
    build_provider_headers,
    get_provider,
)
from .model_protocol import (
    ModelProtocolError,
    normalize_model_response,
    prepare_model_request,
)
from .prose_craft import (
    PROSE_WRITING_SYSTEM_PROMPT,
    compose_craft_brief,
    select_prose_craft_modules,
)


WRITING_SYSTEM_PROMPT = (
    PROSE_WRITING_SYSTEM_PROMPT
    + """

作品级补充约束：
1. 参考技法只提供抽象规则和作者改造要求；不得复用参考作品的人名、专有物件、
   具体情节、独特意象或措辞，也不能推翻正史、任务卡和作品声纹。
2. 编辑偏好只在 applicability 所述场景应用，不能推翻正史、人物知情或视角。
3. 全书蓝图与规划剧情线是长期方向，不是已经发生的事实；当前章不得提前实现
   后续转折、终局或回报。
4. 未来因果链接是创作约束，不是正史。起因章只落实 cause，不提前写目标章的
   effect；结果章要让 effect 从指定 cause 自然发生。
5. 使用简体中文。
"""
).strip()


def compose_writing_system_prompt(
    *,
    book_prompt: str = "",
    craft_context: Mapping[str, Any] | None = None,
) -> str:
    sections = [WRITING_SYSTEM_PROMPT]
    if craft_context:
        modules = select_prose_craft_modules(craft_context)
        sections.append(compose_craft_brief(modules))
    clean_book = book_prompt.strip()
    if clean_book:
        sections.append(
            "以下是当前作品的补充指令。它只适用于本书，优先于全局"
            "协作原则，但不能覆盖服务端权限、已确认正史和任务卡：\n"
            "<book_specific_preferences>\n"
            f"{clean_book}\n"
            "</book_specific_preferences>"
        )
    return "\n\n".join(sections)


@dataclass(frozen=True)
class WritingResponse:
    content: str
    input_tokens: int
    output_tokens: int
    truncated: bool = False


class BaseWriter:
    provider = "unknown"
    model = "unknown"

    async def write(
        self,
        *,
        context: Mapping[str, Any],
        operation: str,
        instruction: str,
        current_content: str,
        previous_content: str,
        provider_user_id: str,
    ) -> WritingResponse:
        raise NotImplementedError

    async def close(self) -> None:
        return None


def _operation_instruction(operation: str, target_chars: int) -> str:
    if operation == "draft":
        return (
            f"从头创作本章完整初稿，目标约 {target_chars} 个中文字符。"
            "完整落实章节大纲，并在章末保留自然的阅读推动力。"
        )
    if operation == "continue":
        return (
            f"紧接现有正文继续写作约 {target_chars} 个中文字符。"
            "只输出新增段落，不要重复现有正文。"
        )
    if operation == "rewrite":
        return (
            f"重写本章完整正文，目标约 {target_chars} 个中文字符。"
            "保留大纲要求的关键事实，但改善节奏、场景与表达。"
        )
    if operation == "polish":
        return (
            "润色本章完整正文。保留原有剧情、事实和段落顺序，"
            "改善语言、对话、节奏与重复表达；输出润色后的完整正文。"
        )
    raise AnalyzerError("不支持的写作操作")


def build_writing_messages(
    *,
    context: Mapping[str, Any],
    operation: str,
    instruction: str,
    current_content: str,
    previous_content: str,
) -> List[Mapping[str, str]]:
    chapter = context["chapter"]
    characters = context.get("characters") or []
    task_card = context.get("task_card") or {
        "purpose": chapter.get("outline") or "",
        "must_happen": [
            line.strip()
            for line in str(chapter.get("key_points") or "").splitlines()
            if line.strip()
        ],
        "target_chars": chapter["target_chapter_chars"],
        "scenes": [],
    }
    character_text = json.dumps(characters, ensure_ascii=False, indent=2)
    raw_memory = context.get("canonical_memory") or {}
    canonical_memory = (
        dict(raw_memory)
        if raw_memory.get("source") == "author_confirmed_canon_only"
        else compile_canonical_memory(raw_memory)
    )
    memory_text = json.dumps(
        canonical_memory, ensure_ascii=False, indent=2
    )
    active_techniques = context.get("active_techniques") or (
        compile_active_techniques(
            context.get("technique_cards") or [], usage="write"
        )
    )
    technique_text = json.dumps(
        active_techniques, ensure_ascii=False, indent=2
    )
    voice_profile_text = json.dumps(
        context.get("voice_profile") or {},
        ensure_ascii=False,
        indent=2,
    )
    editing_preferences_text = json.dumps(
        context.get("confirmed_editing_preferences") or [],
        ensure_ascii=False,
        indent=2,
    )
    archive_settings_text = json.dumps(
        context.get("confirmed_archive_rules") or [],
        ensure_ascii=False,
        indent=2,
    )
    story_plan_text = json.dumps(
        compile_story_plan_context(context, usage="write"),
        ensure_ascii=False,
        indent=2,
    )
    causal_links_text = json.dumps(
        compile_planned_causal_links(context, usage="write"),
        ensure_ascii=False,
        indent=2,
    )
    task_card_text = json.dumps(task_card, ensure_ascii=False, indent=2)
    volume_text = json.dumps(
        {
            "title": chapter.get("volume_title") or "",
            "goal": chapter.get("volume_goal") or "",
            "start_state": chapter.get("volume_start_state") or "",
            "end_state": chapter.get("volume_end_state") or "",
            "major_conflict": chapter.get("volume_major_conflict") or "",
            "payoff": chapter.get("volume_payoff") or "",
        },
        ensure_ascii=False,
        indent=2,
    )
    target_chars = int(
        task_card.get("target_chars") or chapter["target_chapter_chars"]
    )
    task = _operation_instruction(operation, target_chars)
    user_prompt = f"""
请完成下面的小说写作任务。

<project>
书名：{chapter["project_title"]}
类型：{chapter["genre"]}
核心梗概：{chapter["premise"]}
作品承诺：{chapter.get("story_promise") or "未补充"}
目标读者：{chapter.get("target_audience") or "未补充"}
核心吸引力：{chapter.get("core_appeal") or "未补充"}
结局约束：{chapter.get("ending_constraint") or "未补充"}
叙事视角：{chapter["point_of_view"]}
世界设定：
{chapter["world_setting"] or "未补充"}
文风要求：
{chapter["style_guide"] or "自然、具体、符合题材"}
</project>

<volume_plan>
{volume_text}
</volume_plan>

<characters>
{character_text or "[]"}
</characters>

<confirmed_work_archive_settings>
这些是作者手动确认，或从阅读分析中明确采纳的创作设定。它们可以补充项目
字段，但任何未经采纳的分析与笔记都不会出现在这里。
{archive_settings_text}
</confirmed_work_archive_settings>

<confirmed_voice_profile>
这是作者逐项确认的可执行声纹。它约束叙述距离、句段节奏、对话、意象、
比喻、留白与禁用表达；若为空，则只使用项目文风要求。
{voice_profile_text}
</confirmed_voice_profile>

<confirmed_editing_preferences>
这些规则来自作者手工改稿，并由作者逐项确认。只在 applicability 指定的
情境执行 guidance；不要把规则机械套到所有段落。列表不含改稿原文证据。
{editing_preferences_text}
</confirmed_editing_preferences>

<confirmed_story_plan>
这是作者确认的全书长期方向，以及任务卡本章明确选择的规划剧情线。
只用于维持方向、承诺与禁止捷径；不得提前实现后续转折或终局，也不得把
尚未发生的规划当作正史。
{story_plan_text}
</confirmed_story_plan>

<planned_future_causality>
这些是作者明确建立、且只与本章有关的未来因果约束。incoming 表示本章
必须承接前章 cause 并落实 effect；outgoing 表示本章必须具体建立 cause，
但不能提前写出目标章 effect。它们不是已经发生的正史。
{causal_links_text}
</planned_future_causality>

<canonical_story_memory>
以下内容只包含作者已经确认的正史摘要、事实、角色知情、剧情线和伏笔。
它的约束优先于模型猜测；不得把角色不知道的事实写成其已知信息。
{memory_text}
</canonical_story_memory>

<previous_chapter_excerpt>
{previous_content[-6000:] if previous_content else "无前一章正文"}
</previous_chapter_excerpt>

<chapter_plan>
章节序号：{chapter["position"]}
章节名：{chapter["title"] or "未命名章节"}
章节大纲：{chapter["outline"] or "未补充"}
必须出现的关键点：{chapter["key_points"] or "未补充"}
</chapter_plan>

<confirmed_chapter_task_card>
这是作者确认后的本章执行契约。必须落实 must_happen 和每个 scene beat，
不得违反 must_preserve、forbidden、角色知情边界与正史事实。
{task_card_text}
</confirmed_chapter_task_card>

<active_writing_techniques>
这些是作者主动启用的抽象写作方法。只执行 execution_rule 和
author_adaptation，并严格遵守 originality_boundary；不得把参考作品的
具体表达或情节当作素材。
{technique_text}
</active_writing_techniques>

<current_chapter>
{current_content[-30000:] if current_content else "尚无正文"}
</current_chapter>

<extra_instruction>
{instruction or "无额外要求"}
</extra_instruction>

<task>
{task}
</task>
""".strip()
    return [
        {
            "role": "system",
            "content": compose_writing_system_prompt(
                book_prompt=str(chapter.get("ai_instructions") or ""),
                craft_context={
                    "instruction": instruction,
                    "genre": chapter.get("genre"),
                    "scene_contract": task_card,
                    "chapter": chapter,
                },
            ),
        },
        {"role": "user", "content": user_prompt},
    ]


class MockWriter(BaseWriter):
    provider = "mock"
    model = "mock-novel-writer"

    async def write(
        self,
        *,
        context: Mapping[str, Any],
        operation: str,
        instruction: str,
        current_content: str,
        previous_content: str,
        provider_user_id: str,
    ) -> WritingResponse:
        del provider_user_id
        chapter = context["chapter"]
        title = str(chapter["title"] or "未命名章节")
        outline = str(chapter["outline"] or "人物踏入新的场景，故事继续向前")
        del instruction, previous_content
        generated = (
            f"风从远处推来一阵潮湿的气息。\n\n"
            f"围绕“{outline[:120]}”，人物终于迈出了不能收回的一步。"
            "眼前的细节逐渐清晰，原本平静的局面也出现了细小却危险的裂缝。\n\n"
            f"这是《{title}》的本地演示草稿。配置个人模型 API Key 后，"
            "系统会根据项目设定、人物卡、章节大纲和前文生成正式正文。"
        )
        if operation == "continue" and current_content:
            generated = (
                "门外忽然传来第二次敲击，比刚才更轻，也更坚定。\n\n"
                + generated
            )
        await asyncio.sleep(0)
        return WritingResponse(
            content=generated,
            input_tokens=0,
            output_tokens=0,
        )


class ProviderWriter(BaseWriter):
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(
        self,
        settings: Settings,
        *,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.settings = settings
        self.provider = settings.model_provider
        self._provider_spec = get_provider(self.provider)
        self.model = settings.model_name
        self._sleep = sleep
        timeout = httpx.Timeout(
            connect=settings.model_connect_timeout_seconds,
            read=settings.model_read_timeout_seconds,
            write=30,
            pool=10,
        )
        self._client = httpx.AsyncClient(
            base_url=settings.model_base_url.rstrip("/") + "/",
            headers=build_provider_headers(
                self._provider_spec,
                settings.model_api_key,
                model=self.model,
            ),
            timeout=timeout,
            limits=httpx.Limits(max_connections=2, max_keepalive_connections=1),
            transport=transport,
        )

    def _payload(
        self, messages: List[Mapping[str, str]], provider_user_id: str
    ) -> Dict[str, Any]:
        try:
            return build_chat_payload(
                settings=self.settings,
                messages=messages,
                provider_user_id=provider_user_id,
                max_tokens=self.settings.model_max_tokens,
                json_object=False,
                temperature=0.85,
            )
        except ProviderConfigError as exc:
            raise AnalyzerError(str(exc)) from exc

    async def _post(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        total_attempts = self.settings.model_max_retries + 1
        last_error: Optional[Exception] = None
        for attempt in range(total_attempts):
            try:
                prepared = prepare_model_request(self.settings, payload)
                response = await self._client.post(
                    prepared.endpoint, json=prepared.payload
                )
                if response.status_code in self.RETRYABLE_STATUS_CODES:
                    raise httpx.HTTPStatusError(
                        f"{self._provider_spec.label} 暂时不可用"
                        f"（HTTP {response.status_code}）",
                        request=response.request,
                        response=response,
                    )
                if response.status_code >= 400:
                    label = self._provider_spec.label
                    messages = {
                        400: f"{label} 写作请求格式不正确",
                        401: f"{label} API Key 无效",
                        402: f"{label} 账户余额不足",
                        403: f"{label} API Key 无权执行此请求",
                        422: f"{label} 写作请求参数无效",
                    }
                    raise AnalyzerError(
                        messages.get(
                            response.status_code,
                            f"{label} 写作请求失败（HTTP {response.status_code}）",
                        )
                    )
                body = response.json()
                if not isinstance(body, dict):
                    raise AnalyzerError(
                        f"{self._provider_spec.label} 返回结构不正确"
                    )
                return normalize_model_response(
                    prepared.protocol, body
                )
            except ModelProtocolError as exc:
                raise AnalyzerError(str(exc)) from exc
            except AnalyzerError:
                raise
            except ValueError as exc:
                raise AnalyzerError(
                    f"{self._provider_spec.label} 返回了无法解析的响应"
                ) from exc
            except (
                httpx.TimeoutException,
                httpx.RequestError,
                httpx.HTTPStatusError,
            ) as exc:
                last_error = exc
                if attempt + 1 >= total_attempts:
                    break
                delay = min(8.0, 2**attempt) + random.uniform(0, 0.25)
                await self._sleep(delay)
        raise AnalyzerError(
            f"{self._provider_spec.label} 连接失败，已重试 "
            f"{self.settings.model_max_retries} 次"
        ) from last_error

    @staticmethod
    def _extract(
        body: Mapping[str, Any], provider_label: str = "模型服务"
    ) -> tuple[str, str, int, int]:
        usage = body.get("usage") or {}
        try:
            input_tokens = int(usage.get("prompt_tokens") or 0)
            output_tokens = int(usage.get("completion_tokens") or 0)
        except (AttributeError, TypeError, ValueError):
            input_tokens = 0
            output_tokens = 0
        try:
            choice = body["choices"][0]
            content = choice["message"]["content"]
            finish_reason = str(choice.get("finish_reason") or "")
        except (AttributeError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise AnalyzerError(
                f"{provider_label} 写作响应缺少必要字段",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ) from exc
        if not isinstance(content, str):
            if finish_reason in {
                "content_filter",
                "insufficient_system_resource",
            }:
                content = ""
            else:
                raise AnalyzerError(
                    f"{provider_label} 写作响应的正文类型不正确",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
        if (
            not content.strip()
            and finish_reason
            not in {"content_filter", "insufficient_system_resource"}
        ):
            raise AnalyzerError(
                f"{provider_label} 没有返回正文",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        return content.strip(), finish_reason, input_tokens, output_tokens

    async def write(
        self,
        *,
        context: Mapping[str, Any],
        operation: str,
        instruction: str,
        current_content: str,
        previous_content: str,
        provider_user_id: str,
    ) -> WritingResponse:
        messages = build_writing_messages(
            context=context,
            operation=operation,
            instruction=instruction,
            current_content=current_content,
            previous_content=previous_content,
        )
        total_input_tokens = 0
        total_output_tokens = 0
        for resource_attempt in range(2):
            body = await self._post(self._payload(messages, provider_user_id))
            content, reason, input_tokens, output_tokens = self._extract(
                body, self._provider_spec.label
            )
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            if reason == "insufficient_system_resource":
                if resource_attempt == 0:
                    await self._sleep(1.0)
                    continue
                raise AnalyzerError(
                    f"{self._provider_spec.label} 当前系统资源不足",
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                )
            if reason == "content_filter":
                raise AnalyzerError(
                    f"{self._provider_spec.label} 内容安全策略拒绝了"
                    "本次写作输出",
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                )
            if reason not in {"stop", "length"}:
                raise AnalyzerError(
                    f"{self._provider_spec.label} 返回了未支持的结束原因："
                    f"{reason or 'empty'}",
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                )
            return WritingResponse(
                content=content,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                truncated=reason == "length",
            )
        raise AnalyzerError(
            f"{self._provider_spec.label} 当前系统资源不足",
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )

    async def close(self) -> None:
        await self._client.aclose()


def build_default_writer(settings: Settings) -> BaseWriter:
    if settings.uses_test_models:
        return MockWriter()
    return ProviderWriter(settings)
