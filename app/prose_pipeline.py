from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping

from .agent_model import BaseAgentModel
from .model_client import AnalyzerError
from .prose_craft import (
    PROSE_WRITING_SYSTEM_PROMPT,
    compose_craft_brief,
    select_prose_craft_modules,
)

ProseDeltaCallback = Callable[[str], Awaitable[None] | None]


@dataclass(frozen=True)
class ProseGenerationResult:
    content: str
    input_tokens: int
    output_tokens: int
    provider: str
    model: str
    craft_modules: tuple[str, ...] = ()


class ProseDraftPipeline:
    async def generate(
        self,
        *,
        model: BaseAgentModel,
        packet: Mapping[str, Any],
        provider_user_id: str,
        on_text_delta: ProseDeltaCallback | None = None,
    ) -> ProseGenerationResult:
        prepared_packet = dict(packet)
        raw_modules = prepared_packet.get("craft_modules")
        if not isinstance(raw_modules, list) or not raw_modules:
            raw_modules = select_prose_craft_modules(prepared_packet)
            prepared_packet["craft_modules"] = raw_modules
        craft_brief = compose_craft_brief(raw_modules)
        target_chars = max(
            80, int(prepared_packet.get("target_chars") or 3000)
        )
        max_tokens = min(20_000, max(4_000, target_chars * 2 + 2_000))
        turn = await model.native_turn(
            messages=[
                {
                    "role": "system",
                    "content": (
                        PROSE_WRITING_SYSTEM_PROMPT
                        + "\n\n"
                        + craft_brief
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "请按照下面的 writing_packet 创作正文。只返回正文：\n"
                        "<writing_packet>\n"
                        + json.dumps(
                            prepared_packet,
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n</writing_packet>"
                    ),
                },
            ],
            tools=[],
            provider_user_id=provider_user_id,
            max_tokens=max_tokens,
            on_text_delta=on_text_delta,
        )
        if turn.tool_calls:
            raise AnalyzerError(
                "正文模型返回了不应出现的工具调用",
                input_tokens=turn.input_tokens,
                output_tokens=turn.output_tokens,
            )
        content = self._clean_content(turn.content)
        if not content:
            raise AnalyzerError(
                "正文模型没有返回可保存的正文",
                input_tokens=turn.input_tokens,
                output_tokens=turn.output_tokens,
            )
        return ProseGenerationResult(
            content=content,
            input_tokens=turn.input_tokens,
            output_tokens=turn.output_tokens,
            provider=model.provider,
            model=model.model,
            craft_modules=tuple(
                str(item.get("key") or "")
                for item in raw_modules
                if isinstance(item, Mapping)
                and str(item.get("key") or "")
            ),
        )

    @staticmethod
    def _clean_content(value: str) -> str:
        content = str(value or "").strip()
        if content.startswith("```") and content.endswith("```"):
            lines = content.splitlines()
            if len(lines) >= 3:
                content = "\n".join(lines[1:-1]).strip()
        return content
