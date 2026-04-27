"""LangChain model factory and helper functions."""

import json
import logging
import re
from functools import lru_cache
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.llm.resilience import CallProfile  # type: ignore[import-not-found]

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


def _embedding_model_name() -> str:
    return settings.embedding_model


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
        # 8192 是 qwen-plus 输出硬上限. character schema v2 (26 项过去事件 ×
        # 3-5 场景 + 偏好/价值观/能力等) 满输出约 6-7K tokens, 4K 会截断丢失
        # 末尾字段 (life_events.special / emotion_events.relieved 等).
        # 普通聊天回复 <500 tokens, 不受影响.
        max_tokens=8192,
        extra_body={"enable_thinking": settings.dashscope_enable_thinking},
    )


def _claude_model() -> ChatAnthropic:
    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY is required when provider is claude")
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=settings.anthropic_api_key,
        max_tokens=8192,
    )


def _ollama_chat_model(model_name: str) -> ChatOllama:
    """统一构造 ChatOllama, 复用 base_url + httpx 超时等 kwargs."""
    return ChatOllama(
        model=model_name,
        base_url=settings.ollama_base_url,
        client_kwargs=_OLLAMA_CLIENT_KWARGS,
        async_client_kwargs=_OLLAMA_CLIENT_KWARGS,
    )


@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    """Return the large model used for final chat responses."""
    provider = _provider_for("chat")
    if provider == "claude":
        return _claude_model()
    if provider == "dashscope":
        return _dashscope_chat_model(_chat_model_name())
    return _ollama_chat_model(_chat_model_name())


@lru_cache(maxsize=1)
def get_utility_model() -> BaseChatModel:
    """Return the model used for tool / utility tasks."""
    provider = _provider_for("utility")
    if provider == "claude":
        return _claude_model()
    if provider == "dashscope":
        return _dashscope_chat_model(_utility_model_name())
    return _ollama_chat_model(_utility_model_name())


@lru_cache(maxsize=1)
def get_fallback_chat_model() -> ChatOllama:
    """LOCAL_CHAT_MODEL 大模型作为全局 LLM 失败兜底 (resilience.py 消费).

    不论 online_model 开关, 这里永远返回本地 Ollama 实例, 用于 primary
    (Dashscope/Claude) 不可用时的降级. 与 get_chat_model 区分: get_chat_model
    根据 online_model 可能返回 Dashscope; 本函数始终返回 Ollama.
    """
    return _ollama_chat_model(settings.local_chat_model)


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


def _salvage_truncated_json_object(text: str) -> dict | None:
    """Recover the longest complete prefix of a top-level JSON object.

    LLM 满 max_tokens 时输出会从中间 (字符串/数字/嵌套对象内部) 突然截断,
    json.loads 整体失败. 本函数走一遍状态机, 找到最后一次顶层逗号 (或开括号)
    的位置, 截断到那并补上闭括号, 至少把已完整的 top-level key:value 保住.

    返回 None 表示没救活. 调用方应让 repair pass 来填空缺.
    """
    text = text.strip()
    # 剥掉 ```json ... ``` 围栏 (可能闭围栏因截断缺失)
    fence = re.match(r"^```(?:json)?\s*\n?", text)
    if fence:
        text = text[fence.end():]
        text = text.rstrip("`").rstrip()

    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    last_safe_end = -1  # 顶层逗号或开括号后的位置, 在此截断 + 补 "}" 必合法

    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            if depth == 1:
                last_safe_end = i  # 顶层 `{` 后, 空对象也合法
        elif ch == "}":
            depth -= 1
            if depth == 0:
                # 完整结束 — 但 json.loads 已经在外层失败了, 说明内部某处坏了
                # 这种情况罕见, 直接返回 None 让外层处理
                return None
        elif ch == "," and depth == 1:
            last_safe_end = i  # 顶层逗号位, 截到此 (含逗号前) 必能闭合

    if last_safe_end < 0:
        return None

    # 截断到 last_safe_end (不含此位置的 `,`/`{`), 补 "}" 闭顶层
    salvaged = text[start:last_safe_end].rstrip().rstrip(",") + "}"
    try:
        result = json.loads(salvaged)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        return None


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

    # 截断救援: LLM 满 max_tokens 时会从中间突然停, 标准解析全失败.
    # 走状态机救出已完整的顶层字段, 残缺字段交给上层 repair pass.
    salvaged = _salvage_truncated_json_object(text)
    if salvaged is not None:
        return salvaged

    raise ValueError(f"Could not extract JSON from model output: {text[:200]}")


def _provider_and_profile(
    model: BaseChatModel, override: str | None,
):
    """自动从 model 识别 provider 和默认 profile. override 可强制指定 profile.

    默认按"是否是 get_chat_model() 返回的 lru_cache 单例"区分: chat_model →
    chat_extract (45s, 1 retry); utility_model 或自建 model → utility_fast
    (8s, 2 retry). 调用方可传 profile="background" 等 override.
    """
    from app.services.llm.resilience import get_profile, provider_name

    provider = provider_name(model)
    if override:
        return provider, get_profile(override)
    if model is get_chat_model():
        return provider, get_profile("chat_extract")
    return provider, get_profile("utility_fast")


async def _invoke_with_resilience(
    model: BaseChatModel,
    prompt: str | list[BaseMessage],
    *,
    op: str,
    force_json: bool = False,
    **kwargs: Any,
) -> Any:
    """invoke_text / invoke_json_with_usage 的共享骨架.

    force_json=True 时强制设 format='json' (ChatOllama JSON 模式),
    也会给 Ollama fallback 同步设上. 非 Ollama 的 primary (Dashscope/Claude)
    忽略 format, 回 OpenAI / Anthropic 的 JSON 约束要 prompt 层保证.

    返回原始 LangChain response 对象 (未解析), 让调用方自行处理
    .content / _extract_json / usage_metadata 等.

    Streaming 分支: profile.stream_mode=True 时, 内部 astream + 累积 chunks,
    用 AIMessage 包装让上层 `.content` 路径无感. 享受 idle_timeout 保护
    (相邻 chunk 间最大停顿), 不受总 timeout 误杀. 用于 character.generation
    这类 6-8K tokens 长输出场景.
    """
    from app.services.llm.resilience import call_with_resilience

    profile_override = kwargs.pop("profile", None)
    provider, profile = _provider_and_profile(model, profile_override)

    if isinstance(prompt, str):
        messages = [HumanMessage(content=prompt)]
    else:
        messages = prompt

    # Primary 是 Ollama 才加 format='json' (ChatOpenAI/Anthropic 忽略此 kwarg)
    if force_json and isinstance(model, ChatOllama):
        kwargs.setdefault("format", "json")

    if profile.stream_mode:
        return await _invoke_via_stream(
            model, messages, provider=provider, profile=profile,
            op=op, force_json=force_json, **kwargs,
        )

    async def _primary():
        return await model.ainvoke(messages, **kwargs)

    fallback_factory = None
    if provider != "ollama":
        _fb_model = get_fallback_chat_model()
        # Fallback 是 Ollama, 需要 format='json' 走 JSON 模式.
        # 用 setdefault 与 primary 路径对称 — 尊重调用方显式传入的 format.
        fb_kwargs = {**kwargs}
        if force_json:
            fb_kwargs.setdefault("format", "json")

        async def _fallback():
            return await _fb_model.ainvoke(messages, **fb_kwargs)
        fallback_factory = _fallback

    return await call_with_resilience(
        _primary,
        primary_provider=provider,
        profile=profile,
        op=op,
        fallback_factory=fallback_factory,
    )


async def _invoke_via_stream(
    model: BaseChatModel,
    messages: list[BaseMessage],
    *,
    provider: str,
    profile: "CallProfile",
    op: str,
    force_json: bool,
    **kwargs: Any,
) -> Any:
    """Streaming 适配器: 与 ainvoke 等价的接口, 内部走 astream + 累积 chunks.

    返回 AIMessage(content=joined), 上层 `_extract_json(response.content)` /
    `response.content` 路径无感.

    TODO(usage_metadata): LangChain stream chunk 的 usage_metadata 暴露不一致
    (dashscope 在末 chunk; ollama 不暴露). 当前所有 background profile 调用
    走 `invoke_json` 直接丢 usage, 无外部 caller 受影响. 若日后需要计费/观测,
    可在 collect_stream 内 fold 末 chunk 的 usage_metadata 到 AIMessage 上.
    """
    from app.services.llm.resilience import collect_stream
    from langchain_core.messages import AIMessage

    def _primary_stream():
        return model.astream(messages, **kwargs)

    fallback_factory = None
    if provider != "ollama" and profile.allow_ollama_fallback:
        _fb_model = get_fallback_chat_model()
        fb_kwargs = {**kwargs}
        if force_json:
            fb_kwargs.setdefault("format", "json")

        def _fallback_stream():
            return _fb_model.astream(messages, **fb_kwargs)
        fallback_factory = _fallback_stream

    text = await collect_stream(
        _primary_stream,
        primary_provider=provider,
        profile=profile,
        op=op,
        fallback_factory=fallback_factory,
    )
    return AIMessage(content=text)


async def invoke_json(
    model: BaseChatModel,
    prompt: str | list[BaseMessage],
    **kwargs: Any,
) -> Any:
    """Invoke the model and parse the response as JSON."""
    parsed, _ = await invoke_json_with_usage(model, prompt, **kwargs)
    return parsed


async def invoke_json_with_usage(
    model: BaseChatModel,
    prompt: str | list[BaseMessage],
    **kwargs: Any,
) -> tuple[Any, dict]:
    """Same as invoke_json but also returns `{input_tokens, output_tokens}` when
    available from the model. Empty dict if the provider doesn't expose usage.

    Resilience: primary 经 timeout+retry+CB, 失败后切本地 Ollama. 两级都挂抛
    LLMFailedError (调用方自己兜底, 多数已有 try/except 返 None/空 dict).
    """
    response = await _invoke_with_resilience(
        model, prompt, op="invoke_json", force_json=True, **kwargs,
    )
    usage: dict = {}
    meta = getattr(response, "usage_metadata", None)
    if isinstance(meta, dict):
        usage = {
            "input_tokens": int(meta.get("input_tokens", 0) or 0),
            "output_tokens": int(meta.get("output_tokens", 0) or 0),
        }
    return _extract_json(response.content), usage


async def invoke_text(
    model: BaseChatModel,
    prompt: str | list[BaseMessage],
    **kwargs: Any,
) -> str:
    """Invoke the model and return the response as plain text.

    Resilience: primary 经 timeout+retry+CB, 失败后切本地 Ollama. 两级都挂抛
    LLMFailedError.
    """
    response = await _invoke_with_resilience(
        model, prompt, op="invoke_text", force_json=False, **kwargs,
    )
    return response.content
