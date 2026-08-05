"""Execution state machine for one frozen reference chapter."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

from pydantic import ValidationError

from .analysis_repository import AnalysisRepository
from .model_client import AnalyzerError, BaseAnalyzer
from .reference_analysis_metrics import deterministic_structure
from .reference_analysis_schema import (
    LAYER_MODELS,
    MODEL_LAYERS,
    combine_layers,
    content_hash,
    validate_evidence,
)
from .security import stable_provider_user_id


logger = logging.getLogger(__name__)
AnalyzerFactory = Callable[[dict[str, Any]], Awaitable[tuple[BaseAnalyzer, bool]]]


class ReferenceAnalysisPipeline:
    def __init__(
        self,
        repository: AnalysisRepository,
        *,
        analyzer_factory: AnalyzerFactory,
        provider_user_secret: str,
    ):
        self.repository = repository
        self.analyzer_factory = analyzer_factory
        self.provider_user_secret = provider_user_secret

    @staticmethod
    def _validated_model_layer(
        layer: str, value: Mapping[str, Any], source: str
    ) -> dict[str, Any]:
        try:
            result = LAYER_MODELS[layer].model_validate(value).model_dump(
                mode="json"
            )
            validate_evidence(result, source)
            return result
        except (ValidationError, ValueError, TypeError) as exc:
            raise AnalyzerError(f"{layer} 层保存结果无效：{exc}") from exc

    async def process(self, item: dict[str, Any]) -> None:
        analysis_id = str(item["analysis_id"])
        job_id = str(item["job_id"])
        claim_token = str(item["claim_token"])
        active_layer = "structure"
        analyzer: BaseAnalyzer | None = None
        close_analyzer = False
        try:
            content = await asyncio.to_thread(
                Path(str(item["content_path"])).read_text,
                encoding="utf-8",
            )
            digest = content_hash(content)
            frozen_hash = str(item.get("content_hash") or "")
            if frozen_hash and digest != frozen_hash:
                raise AnalyzerError("章节正文已变化，已停止使用旧分析任务")
            layer_rows = await asyncio.to_thread(
                self.repository.layers, analysis_id
            )
            results: dict[str, dict[str, Any]] = {}
            provider_user_id = stable_provider_user_id(
                int(item["user_id"]), self.provider_user_secret
            )

            for layer in ("structure", *MODEL_LAYERS):
                active_layer = layer
                current = layer_rows.get(layer) or {}
                if current.get("status") == "completed" and current.get(
                    "result_json"
                ):
                    parsed = json.loads(str(current["result_json"]))
                    if not isinstance(parsed, dict):
                        raise AnalyzerError(f"{layer} 层保存结果损坏")
                    results[layer] = (
                        parsed
                        if layer == "structure"
                        else self._validated_model_layer(layer, parsed, content)
                    )
                    continue

                cache_provider = (
                    "local" if layer == "structure" else str(item["provider"])
                )
                cache_model = (
                    "deterministic-v2"
                    if layer == "structure"
                    else str(item["model"])
                )
                cached = await asyncio.to_thread(
                    self.repository.cached_layer,
                    content_hash=digest,
                    layer=layer,
                    provider=cache_provider,
                    model=cache_model,
                )
                if cached is not None:
                    cached = (
                        cached
                        if layer == "structure"
                        else self._validated_model_layer(layer, cached, content)
                    )
                    if not await asyncio.to_thread(
                        self.repository.start_layer,
                        analysis_id=analysis_id,
                        layer=layer,
                        claim_token=claim_token,
                    ):
                        raise AnalyzerError(f"{layer} 层状态已经变化")
                    accepted = await asyncio.to_thread(
                        self.repository.complete_layer,
                        analysis_id=analysis_id,
                        content_hash=digest,
                        layer=layer,
                        provider=cache_provider,
                        model=cache_model,
                        result=cached,
                        raw_response="",
                        input_tokens=0,
                        output_tokens=0,
                        claim_token=claim_token,
                        cache_hit=True,
                    )
                    if not accepted:
                        raise AnalyzerError(f"{layer} 层缓存结果已过期")
                    results[layer] = cached
                    continue

                if not await asyncio.to_thread(
                    self.repository.start_layer,
                    analysis_id=analysis_id,
                    layer=layer,
                    claim_token=claim_token,
                ):
                    raise AnalyzerError(f"{layer} 层当前无法开始")
                if layer == "structure":
                    result = deterministic_structure(content)
                    raw_response = json.dumps(result, ensure_ascii=False)
                    input_tokens = output_tokens = 0
                else:
                    if analyzer is None:
                        analyzer, close_analyzer = await self.analyzer_factory(item)
                    response = await analyzer.analyze_layer(
                        layer,
                        str(item["chapter_title"]),
                        content,
                        provider_user_id,
                        results,
                    )
                    result = self._validated_model_layer(
                        layer, response.result, content
                    )
                    raw_response = response.raw_response
                    input_tokens = response.input_tokens
                    output_tokens = response.output_tokens
                accepted = await asyncio.to_thread(
                    self.repository.complete_layer,
                    analysis_id=analysis_id,
                    content_hash=digest,
                    layer=layer,
                    provider=cache_provider,
                    model=cache_model,
                    result=result,
                    raw_response=raw_response,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    claim_token=claim_token,
                )
                if not accepted:
                    raise AnalyzerError(f"{layer} 层结果已经过期")
                results[layer] = result

            combined = combine_layers(
                chapter_title=str(item["chapter_title"]),
                chapter_position=int(item.get("position") or 0) or None,
                text=content,
                layers=results,
            )
            accepted = await asyncio.to_thread(
                self.repository.complete,
                analysis_id=analysis_id,
                job_id=job_id,
                result=combined,
                claim_token=claim_token,
            )
            if not accepted:
                logger.warning(
                    "discarded stale chapter analysis result id=%s", analysis_id
                )
        except AnalyzerError as exc:
            logger.warning("chapter analysis failed id=%s: %s", analysis_id, exc)
            await asyncio.to_thread(
                self.repository.fail_layer,
                analysis_id=analysis_id,
                layer=active_layer,
                error=str(exc),
                input_tokens=exc.input_tokens,
                output_tokens=exc.output_tokens,
            )
            await asyncio.to_thread(
                self.repository.fail,
                analysis_id=analysis_id,
                job_id=job_id,
                error=str(exc),
                claim_token=claim_token,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "unexpected chapter analysis failure id=%s", analysis_id
            )
            await asyncio.to_thread(
                self.repository.fail_layer,
                analysis_id=analysis_id,
                layer=active_layer,
                error="处理分析层时发生内部错误",
            )
            await asyncio.to_thread(
                self.repository.fail,
                analysis_id=analysis_id,
                job_id=job_id,
                error="处理章节时发生内部错误，请重试",
                claim_token=claim_token,
            )
        finally:
            if close_analyzer and analyzer is not None:
                await analyzer.close()
