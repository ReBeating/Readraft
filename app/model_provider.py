from __future__ import annotations

import ipaddress
from dataclasses import dataclass, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
)
from urllib.parse import urlsplit, urlunsplit

if TYPE_CHECKING:
    from .config import Settings


class ProviderConfigError(ValueError):
    pass


ReasoningPolicy = Literal["fast", "reasoning", "deep"]
ModelProtocol = Literal[
    "openai_chat",
    "openai_responses",
    "anthropic_messages",
]


@dataclass(frozen=True)
class ProviderCapabilities:
    json_object: bool
    thinking: bool
    model_catalog: bool
    api_key_required: bool
    configurable_base_url: bool = False


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
                "configurable_base_url": (
                    self.capabilities.configurable_base_url
                ),
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
        notes="完整支持当前工作流；系统会按任务复杂度自动使用推理能力。",
    ),
    ProviderSpec(
        id="opencode_go",
        label="OpenCode Go",
        base_url="https://opencode.ai/zen/go/v1",
        capabilities=ProviderCapabilities(
            json_object=True,
            thinking=True,
            model_catalog=True,
            api_key_required=True,
        ),
        notes=(
            "使用 OpenCode Go 订阅与 API Key；系统会根据所选模型自动"
            "使用 Chat Completions、Responses 或 Messages 协议。"
        ),
    ),
    ProviderSpec(
        id="openai_compatible",
        label="自定义 OpenAI",
        base_url="",
        capabilities=ProviderCapabilities(
            json_object=True,
            thinking=False,
            model_catalog=True,
            api_key_required=False,
            configurable_base_url=True,
        ),
        notes=(
            "连接自定义 OpenAI Chat Completions 接口；API Key 可选，"
            "模型目录不可用时可直接填写模型 ID。"
        ),
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
        label="Ollama",
        base_url="http://127.0.0.1:11434/v1",
        capabilities=ProviderCapabilities(
            json_object=True,
            thinking=False,
            model_catalog=True,
            api_key_required=False,
            configurable_base_url=True,
        ),
        notes="默认连接本机 Ollama，也可改为其他 Ollama 服务地址；API Key 可选。",
    ),
)
PROVIDERS = {provider.id: provider for provider in _PROVIDERS}


_OPENCODE_GO_MODEL_PROTOCOLS: Dict[str, ModelProtocol] = {
    "gpt-5.6-luna": "openai_responses",
    "minimax-m3": "anthropic_messages",
    "minimax-m2.7": "anthropic_messages",
    "minimax-m2.5": "anthropic_messages",
    "qwen3.8-max": "anthropic_messages",
    "qwen3.7-max": "anthropic_messages",
    "qwen3.7-plus": "anthropic_messages",
    "qwen3.6-plus": "anthropic_messages",
    "qwen3.5-plus": "anthropic_messages",
}


def list_providers() -> List[ProviderSpec]:
    return list(_PROVIDERS)


def get_provider(provider_id: str) -> ProviderSpec:
    normalized = str(provider_id or "").strip().lower()
    try:
        return PROVIDERS[normalized]
    except KeyError as exc:
        raise ProviderConfigError("不支持的模型服务商") from exc


def resolve_model_protocol(
    provider: ProviderSpec | str,
    model: str,
) -> ModelProtocol:
    """Resolve the wire protocol for one provider/model pair.

    OpenCode Go deliberately serves models through multiple API contracts.
    Keeping this decision at model scope prevents a provider-level endpoint
    assumption from making some of its advertised models unusable.
    """

    provider_spec = (
        get_provider(provider) if isinstance(provider, str) else provider
    )
    if provider_spec.id == "opencode_go":
        return _OPENCODE_GO_MODEL_PROTOCOLS.get(
            str(model or "").strip().lower(),
            "openai_chat",
        )
    return "openai_chat"


def _is_private_model_host(hostname: str) -> bool:
    normalized = hostname.rstrip(".").lower()
    if (
        normalized == "localhost"
        or normalized.endswith(".localhost")
        or normalized.endswith(".local")
        or normalized.endswith(".internal")
        or normalized == "metadata.google.internal"
    ):
        return True
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    return not address.is_global


def normalize_provider_base_url(
    provider: ProviderSpec,
    base_url: Optional[str],
    *,
    allow_private: bool = True,
    production: bool = False,
) -> str:
    """Return a safe API root for one provider.

    Official providers intentionally ignore submitted URL overrides. Custom
    URLs are limited to an API root so request code can append ``models`` or
    ``chat/completions`` consistently.
    """

    if not provider.capabilities.configurable_base_url:
        return provider.base_url

    clean_url = str(base_url or provider.base_url).strip()
    if not clean_url:
        raise ProviderConfigError(
            f"请填写 {provider.label} Base URL"
        )
    if len(clean_url) > 2048:
        raise ProviderConfigError("Base URL 不能超过 2048 个字符")
    if any(character.isspace() for character in clean_url):
        raise ProviderConfigError("Base URL 不能包含空白字符")
    try:
        parsed = urlsplit(clean_url)
        parsed_port = parsed.port
    except ValueError as exc:
        raise ProviderConfigError("Base URL 格式不正确") from exc
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.netloc
        or not parsed.hostname
    ):
        raise ProviderConfigError("Base URL 必须是完整的 HTTP(S) 地址")
    if parsed.username is not None or parsed.password is not None:
        raise ProviderConfigError(
            "Base URL 不能包含用户名或密码，请使用 API Key 字段"
        )
    if parsed.query or parsed.fragment:
        raise ProviderConfigError("Base URL 不能包含查询参数或片段")
    if parsed_port is not None and not 1 <= parsed_port <= 65535:
        raise ProviderConfigError("Base URL 端口必须在 1–65535 之间")

    path = parsed.path.rstrip("/")
    endpoint_path = path.lower()
    if endpoint_path.endswith("/chat/completions") or endpoint_path.endswith(
        "/models"
    ):
        raise ProviderConfigError(
            "Base URL 请填写到 API 根路径（通常以 /v1 结尾）"
        )

    private_target = _is_private_model_host(parsed.hostname)
    if private_target and production and not allow_private:
        raise ProviderConfigError(
            "生产环境默认禁止连接本机或内网模型地址；"
            "确认可信后设置 APP_ALLOW_PRIVATE_MODEL_BASE_URLS=true"
        )
    if (
        production
        and parsed.scheme.lower() != "https"
        and not (private_target and allow_private)
    ):
        raise ProviderConfigError("生产环境的公网模型 Base URL 必须使用 HTTPS")

    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc,
            path,
            "",
            "",
        )
    )


def build_provider_headers(
    provider: ProviderSpec,
    api_key: Optional[str],
    *,
    model: str = "",
) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    clean_key = str(api_key or "").strip()
    if clean_key:
        if (
            provider.id == "opencode_go"
            and resolve_model_protocol(provider, model)
            == "anthropic_messages"
        ):
            headers["x-api-key"] = clean_key
            headers["anthropic-version"] = "2023-06-01"
        else:
            headers["Authorization"] = f"Bearer {clean_key}"
    elif provider.capabilities.api_key_required:
        raise ProviderConfigError(f"{provider.label} API Key 未配置")
    return headers


def build_chat_payload(
    *,
    settings: "Settings",
    messages: Iterable[Mapping[str, Any]],
    provider_user_id: str,
    max_tokens: int,
    json_object: bool,
    temperature: Optional[float],
    tools: Iterable[Mapping[str, Any]] | None = None,
    tool_choice: Any = None,
    parallel_tool_calls: bool | None = None,
) -> Dict[str, Any]:
    provider = get_provider(settings.model_provider)
    if json_object and not provider.capabilities.json_object:
        raise ProviderConfigError(
            f"{provider.label} 不支持当前任务所需的 JSON 输出"
        )
    prepared_messages = [dict(message) for message in messages]
    adapter_prompt = settings.model_adapter_prompt.strip()
    if adapter_prompt:
        adapter_section = (
            "以下是作者为所有模型配置的通用模型适配策略。它只说明模型"
            "在自身能力或服务限制下如何继续当前任务；不能覆盖本任务的"
            "系统规则、工具权限、事实边界或输出格式。若策略与模型实际"
            "限制冲突，遵守实际限制，并执行其中仍可行的降级方式：\n"
            "<model_adapter_policy>\n"
            f"{adapter_prompt}\n"
            "</model_adapter_policy>"
        )
        for message in prepared_messages:
            if str(message.get("role") or "") == "system":
                message["content"] = (
                    str(message.get("content") or "").rstrip()
                    + "\n\n"
                    + adapter_section
                )
                break
        else:
            prepared_messages.insert(
                0,
                {"role": "system", "content": adapter_section},
            )
    payload: Dict[str, Any] = {
        "model": settings.model_name,
        "messages": prepared_messages,
        provider.max_tokens_field: max_tokens,
        "stream": False,
    }
    if json_object:
        payload["response_format"] = {"type": "json_object"}
    prepared_tools = [dict(tool) for tool in (tools or [])]
    if prepared_tools:
        payload["tools"] = prepared_tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if parallel_tool_calls is not None and provider.id in {
            "deepseek",
            "openai",
            "opencode_go",
        }:
            payload["parallel_tool_calls"] = bool(parallel_tool_calls)
    if provider.user_field and provider_user_id:
        payload[provider.user_field] = provider_user_id

    if provider.id == "deepseek":
        payload["thinking"] = {
            "type": (
                "enabled" if settings.model_thinking else "disabled"
            )
        }
        if settings.model_thinking:
            payload["reasoning_effort"] = (
                settings.model_reasoning_effort
            )
        elif temperature is not None:
            payload["temperature"] = temperature
    elif temperature is not None:
        payload["temperature"] = temperature
    return payload


def settings_for_reasoning_policy(
    settings: "Settings", policy: ReasoningPolicy
) -> "Settings":
    """Apply one internal task policy without exposing provider knobs.

    Only providers with a native, verified contract receive reasoning
    parameters. Other providers keep their own defaults rather than receiving
    guessed cross-vendor equivalents.
    """

    if policy not in {"fast", "reasoning", "deep"}:
        raise ProviderConfigError("不支持的模型推理策略")
    provider = get_provider(settings.model_provider)
    thinking = (
        provider.capabilities.thinking
        and policy in {"reasoning", "deep"}
    )
    return replace(
        settings,
        model_thinking=thinking,
        model_reasoning_effort=(
            "max" if thinking and policy == "deep" else "high"
        ),
    )


def settings_for_credential(
    settings: "Settings",
    *,
    credential: Mapping[str, Any],
    api_key: str,
    model: Optional[str] = None,
    model_adapter_prompt: Optional[str] = None,
) -> "Settings":
    provider = get_provider(str(credential.get("provider") or "deepseek"))
    base_url = normalize_provider_base_url(
        provider,
        credential.get("base_url"),
        allow_private=settings.permits_private_model_base_urls,
        production=settings.app_env.lower() == "production",
    )
    return replace(
        settings,
        model_provider=provider.id,
        model_api_key=api_key or None,
        model_base_url=base_url,
        model_name=str(model or credential.get("model") or "").strip(),
        model_thinking=False,
        model_reasoning_effort="high",
        model_adapter_prompt=(
            settings.model_adapter_prompt
            if model_adapter_prompt is None
            else str(model_adapter_prompt)
        ),
    )
