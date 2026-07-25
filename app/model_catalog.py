from __future__ import annotations

from typing import List, Optional

import httpx

from .credentials import CredentialError, validate_model
from .model_provider import (
    ProviderConfigError,
    build_provider_headers,
    get_provider,
)


class ModelCatalogError(ValueError):
    pass


async def fetch_models(
    *,
    provider_id: str,
    api_key: Optional[str],
    base_url: Optional[str] = None,
    timeout_seconds: int = 15,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> List[str]:
    try:
        provider = get_provider(provider_id)
        headers = build_provider_headers(provider, api_key)
    except ProviderConfigError as exc:
        raise ModelCatalogError(str(exc)) from exc
    timeout = httpx.Timeout(
        connect=timeout_seconds,
        read=timeout_seconds,
        write=timeout_seconds,
        pool=timeout_seconds,
    )
    request_base_url = str(base_url or provider.base_url).strip()
    if not request_base_url:
        raise ModelCatalogError(f"请填写 {provider.label} Base URL")
    try:
        async with httpx.AsyncClient(
            base_url=request_base_url.rstrip("/") + "/",
            headers={**headers, "Accept": "application/json"},
            timeout=timeout,
            transport=transport,
        ) as client:
            response = await client.get("models")
    except (httpx.TimeoutException, httpx.RequestError) as exc:
        raise ModelCatalogError(
            f"连接 {provider.label} 读取模型列表失败"
        ) from exc

    if response.status_code in {401, 403}:
        raise ModelCatalogError("API Key 无效或无权读取模型列表")
    if response.status_code >= 400:
        raise ModelCatalogError(
            f"读取模型列表失败（HTTP {response.status_code}）"
        )
    try:
        payload = response.json()
        items = payload["data"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelCatalogError(
            f"{provider.label} 返回的模型列表格式不正确"
        ) from exc
    if not isinstance(items, list):
        raise ModelCatalogError(
            f"{provider.label} 返回的模型列表格式不正确"
        )

    models = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            model_id = validate_model(str(item.get("id") or ""))
        except CredentialError:
            continue
        if model_id not in models:
            models.append(model_id)
    if not models:
        raise ModelCatalogError("当前 API Key 没有可用模型")
    return models


async def fetch_deepseek_models(
    *,
    api_key: str,
    base_url: str,
    timeout_seconds: int = 15,
    transport: Optional[httpx.AsyncBaseTransport] = None,
) -> List[str]:
    """Backward-compatible DeepSeek catalog helper."""
    return await fetch_models(
        provider_id="deepseek",
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        transport=transport,
    )
