"""Central registry and constructors for every supported LLM provider.

Provider capabilities belong here; selectable model identifiers and pricing stay
in ``model_registry`` so admins can change them without a deploy.  Keeping the
credential/base-url mapping in code avoids leaking secrets into the database
while still giving the admin UI enough metadata to render providers dynamically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from app.config import settings


Driver = Literal["ollama", "openai", "anthropic"]


@dataclass(frozen=True)
class ProviderDefinition:
    id: str
    display_name: str
    description: str
    driver: Driver
    local: bool
    admin_enabled: bool = True
    api_key_setting: str | None = None
    api_key_env: str | None = None
    base_url_setting: str | None = None
    base_url_env: str | None = None
    preferred_chat_models: tuple[str, ...] = ()
    preferred_small_models: tuple[str, ...] = ()
    stream_usage: bool = True
    supports_embeddings: bool = False


_PROVIDERS: dict[str, ProviderDefinition] = {
    "ollama": ProviderDefinition(
        id="ollama",
        display_name="Ollama（本地）",
        description="本机或内网自托管模型",
        driver="ollama",
        local=True,
        base_url_setting="ollama_base_url",
        base_url_env="OLLAMA_BASE_URL",
        preferred_chat_models=("qwen2.5:14b",),
        preferred_small_models=("qwen2.5:7b",),
        supports_embeddings=True,
    ),
    "dashscope": ProviderDefinition(
        id="dashscope",
        display_name="阿里云百炼 / DashScope",
        description="通义千问与 Qwen-Character，兼容 OpenAI Chat Completions",
        driver="openai",
        local=False,
        api_key_setting="dashscope_api_key",
        api_key_env="DASHSCOPE_API_KEY",
        base_url_setting="dashscope_base_url",
        base_url_env="DASHSCOPE_BASE_URL",
        preferred_chat_models=("qwen3.5-plus", "qwen-plus-character", "qwen3.5-flash"),
        preferred_small_models=("qwen3.5-flash", "qwen-flash-character", "qwen3.5-plus"),
        supports_embeddings=True,
    ),
    "deepseek": ProviderDefinition(
        id="deepseek",
        display_name="DeepSeek",
        description="DeepSeek 官方 OpenAI-compatible API",
        driver="openai",
        local=False,
        api_key_setting="deepseek_api_key",
        api_key_env="DEEPSEEK_API_KEY",
        base_url_setting="deepseek_base_url",
        base_url_env="DEEPSEEK_BASE_URL",
        preferred_chat_models=("deepseek-v4-pro", "deepseek-v4-flash"),
        preferred_small_models=("deepseek-v4-flash", "deepseek-v4-pro"),
    ),
    "qianfan": ProviderDefinition(
        id="qianfan",
        display_name="百度千帆",
        description="百度千帆 ModelBuilder OpenAI-compatible API",
        driver="openai",
        local=False,
        api_key_setting="qianfan_api_key",
        api_key_env="QIANFAN_API_KEY",
        base_url_setting="qianfan_base_url",
        base_url_env="QIANFAN_BASE_URL",
        # ERNIE Character 预置服务已退役；账号自定义接入点由模型库动态注册。
    ),
    "ark": ProviderDefinition(
        id="ark",
        display_name="火山方舟 / 豆包",
        description="豆包角色扮演模型与方舟自定义推理接入点",
        driver="openai",
        local=False,
        api_key_setting="ark_api_key",
        api_key_env="ARK_API_KEY",
        base_url_setting="ark_base_url",
        base_url_env="ARK_BASE_URL",
        # Character 的 model/endpoint id 以账号控制台实际值为准。
    ),
    "minimax": ProviderDefinition(
        id="minimax",
        display_name="MiniMax",
        description="MiniMax 开放平台，含角色扮演对话模型 M2-her",
        driver="openai",
        local=False,
        api_key_setting="minimax_api_key",
        api_key_env="MINIMAX_API_KEY",
        base_url_setting="minimax_base_url",
        base_url_env="MINIMAX_BASE_URL",
        preferred_chat_models=("M2-her",),
        preferred_small_models=("M2-her",),
        # MiniMax 的角色模型文档未承诺 OpenAI stream_options usage 扩展。
        stream_usage=False,
    ),
    # 保留历史 env-only Claude 支持，不把它暴露到 admin 模型库。
    "claude": ProviderDefinition(
        id="claude",
        display_name="Anthropic Claude",
        description="Anthropic Messages API",
        driver="anthropic",
        local=False,
        admin_enabled=False,
        api_key_setting="anthropic_api_key",
        api_key_env="ANTHROPIC_API_KEY",
        preferred_chat_models=("claude-sonnet-4-20250514",),
        preferred_small_models=("claude-sonnet-4-20250514",),
    ),
}


_OLLAMA_CLIENT_KWARGS = {"timeout": 300.0}


def get_provider(provider_id: str) -> ProviderDefinition:
    normalized = provider_id.strip().lower()
    try:
        return _PROVIDERS[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported LLM provider: {provider_id}") from exc


def provider_ids(*, admin_only: bool = False, remote_only: bool = False) -> set[str]:
    return {
        spec.id
        for spec in _PROVIDERS.values()
        if (not admin_only or spec.admin_enabled)
        and (not remote_only or not spec.local)
    }


def public_provider_options(*, include_local: bool = True) -> list[dict]:
    """Return secret-safe provider metadata for authenticated admin clients."""
    result: list[dict] = []
    for spec in _PROVIDERS.values():
        if not spec.admin_enabled or (spec.local and not include_local):
            continue
        api_key = getattr(settings, spec.api_key_setting, "") if spec.api_key_setting else ""
        base_url = getattr(settings, spec.base_url_setting, "") if spec.base_url_setting else ""
        result.append({
            "id": spec.id,
            "display_name": spec.display_name,
            "description": spec.description,
            "local": spec.local,
            "configured": spec.local or bool(str(api_key).strip()),
            "credential_env": spec.api_key_env,
            "base_url_env": spec.base_url_env,
            "base_url": str(base_url),
            "preferred_chat_models": list(spec.preferred_chat_models),
            "preferred_small_models": list(spec.preferred_small_models),
        })
    return result


def _is_dashscope_character(model_name: str) -> bool:
    return "character" in model_name.lower()


def _model_runtime_options(provider_id: str, model_name: str) -> dict:
    """Provider/model-specific request limits not suitable for DB pricing metadata."""
    if provider_id == "minimax" and model_name.lower() == "m2-her":
        return {"temperature": 1.0, "top_p": 0.95, "max_tokens": 2048}
    if provider_id == "dashscope":
        character_options = {
            "qwen-plus-character": {"temperature": 0.8, "max_tokens": 4096},
            "qwen-flash-character": {"temperature": 0.8, "max_tokens": 4096},
            "qwen-flash-character-2026-02-26": {
                "temperature": 0.8,
                "max_tokens": 32768,
            },
        }
        if model_name in character_options:
            return character_options[model_name]
    return {"temperature": 0.7, "max_tokens": 8192}


def _base_url_for(spec: ProviderDefinition, model_name: str) -> str:
    if spec.id == "dashscope" and _is_dashscope_character(model_name):
        character_url = settings.dashscope_character_base_url.strip()
        if character_url:
            return character_url
    if not spec.base_url_setting:
        return ""
    return str(getattr(settings, spec.base_url_setting, "")).strip()


def build_chat_model(provider_id: str, model_name: str) -> BaseChatModel:
    """Build a LangChain chat model from the central provider definition."""
    spec = get_provider(provider_id)
    if spec.driver == "ollama":
        return ChatOllama(
            model=model_name,
            base_url=_base_url_for(spec, model_name),
            client_kwargs=_OLLAMA_CLIENT_KWARGS,
            async_client_kwargs=_OLLAMA_CLIENT_KWARGS,
        )

    if not spec.api_key_setting:
        raise ValueError(f"Provider {spec.id} does not define an API credential")
    api_key = str(getattr(settings, spec.api_key_setting, "")).strip()
    if not api_key:
        raise ValueError(f"{spec.api_key_env} is required when provider is {spec.id}")

    if spec.driver == "anthropic":
        resolved_name = model_name if model_name.startswith("claude-") else spec.preferred_chat_models[0]
        return ChatAnthropic(model=resolved_name, api_key=api_key, max_tokens=8192)

    base_url = _base_url_for(spec, model_name)
    if not base_url:
        raise ValueError(f"{spec.base_url_env} is required when provider is {spec.id}")
    kwargs = _model_runtime_options(spec.id, model_name)
    kwargs["stream_usage"] = spec.stream_usage
    # Character models do not advertise DashScope's enable_thinking extension.
    if spec.id == "dashscope" and not _is_dashscope_character(model_name):
        kwargs["extra_body"] = {"enable_thinking": settings.dashscope_enable_thinking}
    model = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )
    object.__setattr__(model, "_companion_provider", spec.id)
    return model
