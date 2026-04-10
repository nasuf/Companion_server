"""LangChain model factory and helper functions."""

import json
import logging
import re
from functools import lru_cache
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import settings

logger = logging.getLogger(__name__)

Provider = str

# Ollama httpx 客户端超时
# 默认 5s 太短 — 大模型(qwen2.5:14b)首次加载 + 长 prompt 推理可能耗时几十秒
# 注意：不设 trust_env=False，让生产环境可以通过 HTTP_PROXY 环境变量配置代理。
# 本地开发时通过 start.sh 设置 NO_PROXY=localhost 来绕过代理（见 start.sh）。
_OLLAMA_CLIENT_KWARGS = {
    "timeout": 300.0,
}


def _default_provider() -> Provider:
    return "dashscope" if settings.online_model else "ollama"


def _utility_model_name() -> str:
    return (
        settings.utility_model
        or settings.ollama_model
        or (settings.remote_small_model if settings.online_model else settings.local_small_model)
    )


def _chat_model_name() -> str:
    return settings.chat_model or (
        settings.remote_chat_model if settings.online_model else settings.local_chat_model
    )


def _summarizer_model_name() -> str:
    return settings.summarizer_model or (
        settings.remote_small_model if settings.online_model else settings.local_small_model
    )


def _embedding_model_name() -> str:
    return settings.embedding_model or (
        settings.remote_embedding_model
        if settings.online_model
        else settings.local_embedding_model
    )


def _provider_for(role: str) -> Provider:
    override = getattr(settings, f"{role}_provider", "") or ""
    provider = (override or settings.llm_provider or _default_provider()).strip().lower()
    if provider not in {"ollama", "dashscope", "claude"}:
        raise ValueError(f"Unsupported provider for {role}: {provider}")
    return provider


def _dashscope_chat_model(model_name: str) -> ChatOpenAI:
    if not settings.dashscope_api_key:
        raise ValueError("DASHSCOPE_API_KEY is required when provider is dashscope")
    return ChatOpenAI(
        model=model_name,
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
        temperature=0.7,
        max_tokens=4096,
        extra_body={"enable_thinking": settings.dashscope_enable_thinking},
    )


def _claude_model() -> ChatAnthropic:
    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY is required when provider is claude")
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=settings.anthropic_api_key,
        max_tokens=4096,
    )


@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    """Return the large model used for final chat responses."""
    provider = _provider_for("chat")
    if provider == "claude":
        return _claude_model()
    if provider == "dashscope":
        return _dashscope_chat_model(_chat_model_name())
    return ChatOllama(
        model=_chat_model_name(),
        base_url=settings.ollama_base_url,
        client_kwargs=_OLLAMA_CLIENT_KWARGS,
        async_client_kwargs=_OLLAMA_CLIENT_KWARGS,
    )


@lru_cache(maxsize=1)
def get_summarizer_model() -> BaseChatModel:
    """Return the small model used by the 3-layer summarizer."""
    provider = _provider_for("summarizer")
    if provider == "claude":
        return _claude_model()
    if provider == "dashscope":
        return _dashscope_chat_model(_summarizer_model_name())
    return ChatOllama(
        model=_summarizer_model_name(),
        base_url=settings.ollama_base_url,
        client_kwargs=_OLLAMA_CLIENT_KWARGS,
        async_client_kwargs=_OLLAMA_CLIENT_KWARGS,
    )


@lru_cache(maxsize=1)
def get_utility_model() -> BaseChatModel:
    """Return the model used for tool / utility tasks."""
    provider = _provider_for("utility")
    if provider == "claude":
        return _claude_model()
    if provider == "dashscope":
        return _dashscope_chat_model(_utility_model_name())
    return ChatOllama(
        model=_utility_model_name(),
        base_url=settings.ollama_base_url,
        client_kwargs=_OLLAMA_CLIENT_KWARGS,
        async_client_kwargs=_OLLAMA_CLIENT_KWARGS,
    )


@lru_cache(maxsize=1)
def get_embedding_model() -> Embeddings:
    """Return the embedding model for vector operations."""
    provider = _provider_for("embedding")
    if provider == "dashscope":
        if not settings.dashscope_api_key:
            raise ValueError("DASHSCOPE_API_KEY is required when embedding_provider is dashscope")
        return OpenAIEmbeddings(
            model=_embedding_model_name(),
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
            dimensions=settings.embedding_dimensions,
            check_embedding_ctx_length=False,
        )
    if provider == "claude":
        raise ValueError("Claude does not provide embeddings; use ollama or dashscope")
    return OllamaEmbeddings(
        model=_embedding_model_name(),
        base_url=settings.ollama_base_url,
    )


def convert_messages(messages: list[dict]) -> list[BaseMessage]:
    """Convert a list of message dicts to LangChain BaseMessage objects."""
    role_map: dict[str, type[BaseMessage]] = {
        "system": SystemMessage,
        "user": HumanMessage,
        "assistant": AIMessage,
    }
    result: list[BaseMessage] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        cls = role_map.get(role, HumanMessage)
        result.append(cls(content=content))
    return result


def _extract_json(text: str) -> Any:
    """Extract and parse JSON from model output text."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not extract JSON from model output: {text[:200]}")


async def invoke_json(
    model: BaseChatModel,
    prompt: str | list[BaseMessage],
    **kwargs: Any,
) -> Any:
    """Invoke the model and parse the response as JSON."""
    if isinstance(prompt, str):
        messages = [HumanMessage(content=prompt)]
    else:
        messages = prompt

    if isinstance(model, ChatOllama):
        kwargs.setdefault("format", "json")

    response = await model.ainvoke(messages, **kwargs)
    return _extract_json(response.content)


async def invoke_text(
    model: BaseChatModel,
    prompt: str | list[BaseMessage],
    **kwargs: Any,
) -> str:
    """Invoke the model and return the response as plain text."""
    if isinstance(prompt, str):
        messages = [HumanMessage(content=prompt)]
    else:
        messages = prompt

    response = await model.ainvoke(messages, **kwargs)
    return response.content
