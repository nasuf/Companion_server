"""LangChain LLM model factory and helper functions."""

import json
import logging
import re
from functools import lru_cache
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_chat_model() -> ChatOllama | ChatAnthropic:
    """Return the large model used for final chat responses.

    Uses ``settings.chat_model`` (default qwen2.5:14b).
    When ``settings.llm_provider == "claude"``, returns ChatAnthropic instead.
    """
    if settings.llm_provider == "claude":
        return ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=settings.anthropic_api_key,
            max_tokens=4096,
        )
    return ChatOllama(
        model=settings.chat_model,
        base_url=settings.ollama_base_url,
    )


@lru_cache(maxsize=1)
def get_summarizer_model() -> ChatOllama:
    """Return the small model used by the 3-layer summarizer.

    Uses ``settings.summarizer_model`` (default qwen2.5:7b).
    Always Ollama regardless of llm_provider.
    """
    return ChatOllama(
        model=settings.summarizer_model,
        base_url=settings.ollama_base_url,
    )


@lru_cache(maxsize=1)
def get_utility_model() -> ChatOllama | ChatAnthropic:
    """Return the model used for tool / utility tasks.

    Handles memory extraction, conflict detection, deletion detection,
    and user portrait updates.  Uses ``settings.ollama_model``.
    When ``settings.llm_provider == "claude"``, returns ChatAnthropic instead.
    """
    if settings.llm_provider == "claude":
        return ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=settings.anthropic_api_key,
            max_tokens=4096,
        )
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
    )


@lru_cache(maxsize=1)
def get_embedding_model() -> OllamaEmbeddings:
    """Return the embedding model for vector operations."""
    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )


def convert_messages(messages: list[dict]) -> list[BaseMessage]:
    """Convert a list of message dicts to LangChain BaseMessage objects.

    Each dict must have ``role`` (``"system"`` | ``"user"`` | ``"assistant"``)
    and ``content`` keys.
    """
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
    """Extract and parse JSON from model output text.

    Handles both raw JSON and markdown-fenced code blocks.
    """
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } or [ ... ]
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
    model: ChatOllama | ChatAnthropic,
    prompt: str | list[BaseMessage],
    **kwargs: Any,
) -> Any:
    """Invoke the model and parse the response as JSON.

    Accepts either a plain text prompt (sent as a HumanMessage) or a list of
    BaseMessage objects.  Extra *kwargs* are forwarded to ``model.ainvoke``.

    If the model supports ``format="json"`` (ChatOllama), it is set automatically.
    """
    if isinstance(prompt, str):
        messages = [HumanMessage(content=prompt)]
    else:
        messages = prompt

    # Request JSON format from Ollama models
    if isinstance(model, ChatOllama):
        kwargs.setdefault("format", "json")

    response = await model.ainvoke(messages, **kwargs)
    return _extract_json(response.content)


async def invoke_text(
    model: ChatOllama | ChatAnthropic,
    prompt: str | list[BaseMessage],
    **kwargs: Any,
) -> str:
    """Invoke the model and return the response as plain text.

    Accepts either a plain text prompt (sent as a HumanMessage) or a list of
    BaseMessage objects.  Extra *kwargs* are forwarded to ``model.ainvoke``.
    """
    if isinstance(prompt, str):
        messages = [HumanMessage(content=prompt)]
    else:
        messages = prompt

    response = await model.ainvoke(messages, **kwargs)
    return response.content
