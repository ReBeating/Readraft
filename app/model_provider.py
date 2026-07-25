from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional

if TYPE_CHECKING:
    from .config import Settings


class ProviderConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ProviderCapabilities:
    json_object: bool
    thinking: bool
    model_catalog: bool
    api_key_required: bool


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    label: str
    base_url: str
    capabilities: ProviderCapabilities
    max_tokens_field: str = "max_tokens"
    user_field: Optional[str] = None
    notes: str = ""

    def public_payload(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "base_url": self.base_url,
            "capabilities": {
                "json_object": self.capabilities.json_object,
                "thinking": self.capabilities.thinking,
                "model_catalog": self.capabilities.model_catalog,
                "api_key_required": self.capabilities.api_key_required,
            },
            "notes": self.notes,
        }


_PROVIDERS = (
    ProviderSpec(
        id="deepseek",
        label="DeepSeek",
        base_url="https://api.deepseek.com",
        capabilities=ProviderCapabilities(
            json_object=True,
            thinking=True,
            model_catalog=True,
            api_key_required=True,
        ),
        user_field="user_id",
        notes="完整支持当前工作流、JSON 输出与思考模式。",
    ),
    ProviderSpec(
        id="openai",
        label="OpenAI",
        base_url="https://api.openai.com/v1",
        capabilities=ProviderCapabilities(
            json_object=True,
            thinking=False,
            model_catalog=True,
            api_key_required=True,
        ),
        max_tokens_field="max_completion_tokens",
        user_field="user",
        notes="通过 Chat Completions 接口接入；思考参数暂不跨厂商映射。",
    ),
    ProviderSpec(
        id="gemini",
        label="Google Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        capabilities=ProviderCapabilities(
            json_object=True,
            thinking=False,
            model_catalog=True,
            api_key_required=True,
        ),
        notes="使用 Google 官方 OpenAI 兼容端点。",
    ),
    ProviderSpec(
        id="ollama",
        label="Ollama（本机）",
        base_url="http://127.0.0.1:11434/v1",
        capabilities=ProviderCapabilities(
            json_object=True,
            thinking=False,
            model_catalog=True,
            api_key_required=False,
        ),
        notes="只连接本机 Ollama；无需 API Key。",
    ),
)
PROVIDERS = {provider.id: provider for provider in _PROVIDERS}


def list_providers() -> List[ProviderSpec]:
    return list(_PROVIDERS)


def get_provider(provider_id: str) -> ProviderSpec:
    normalized = str(provider_id or "").strip().lower()
    try:
        return PROVIDERS[normalized]
    except KeyError as exc:
        raise ProviderConfigError("不支持的模型服务商") from exc


def build_provider_headers(
    provider: ProviderSpec, api_key: Optional[str]
) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    clean_key = str(api_key or "").strip()
    if clean_key:
        headers["Authorization"] = f"Bearer {clean_key}"
    elif provider.capabilities.api_key_required:
        raise ProviderConfigError(f"{provider.label} API Key 未配置")
    return headers


def build_chat_payload(
    *,
    settings: "Settings",
    messages: Iterable[Mapping[str, str]],
    provider_user_id: str,
    max_tokens: int,
    json_object: bool,
    temperature: Optional[float],
) -> Dict[str, Any]:
    provider = get_provider(settings.model_provider)
    if json_object and not provider.capabilities.json_object:
        raise ProviderConfigError(
            f"{provider.label} 不支持当前任务所需的 JSON 输出"
        )
    payload: Dict[str, Any] = {
        "model": settings.deepseek_model,
        "messages": list(messages),
        provider.max_tokens_field: max_tokens,
        "stream": False,
    }
    if json_object:
        payload["response_format"] = {"type": "json_object"}
    if provider.user_field and provider_user_id:
        payload[provider.user_field] = provider_user_id

    if provider.id == "deepseek":
        payload["thinking"] = {
            "type": (
                "enabled" if settings.deepseek_thinking else "disabled"
            )
        }
        if settings.deepseek_thinking:
            payload["reasoning_effort"] = (
                settings.deepseek_reasoning_effort
            )
        elif temperature is not None:
            payload["temperature"] = temperature
    elif temperature is not None:
        payload["temperature"] = temperature
    return payload


def settings_for_credential(
    settings: "Settings",
    *,
    credential: Mapping[str, Any],
    api_key: str,
    model: Optional[str] = None,
) -> "Settings":
    provider = get_provider(str(credential.get("provider") or "deepseek"))
    thinking = bool(credential.get("thinking"))
    if thinking and not provider.capabilities.thinking:
        raise ProviderConfigError(
            f"{provider.label} 暂不支持 novelAI 的思考模式参数"
        )
    return replace(
        settings,
        model_provider=provider.id,
        deepseek_api_key=api_key or None,
        deepseek_base_url=provider.base_url,
        deepseek_model=str(model or credential.get("model") or "").strip(),
        deepseek_thinking=thinking,
        deepseek_reasoning_effort=str(
            credential.get("reasoning_effort") or "high"
        ),
        deepseek_system_prompt=str(
            credential.get("system_prompt") or ""
        ),
    )
