"""LangChain model factory and helper functions."""

import json
import logging
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.services.llm.resilience import CallProfile  # type: ignore[import-not-found]

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from app.config import settings
from app.services.llm.providers import build_chat_model, get_provider

logger = logging.getLogger(__name__)

Provider = str


def _default_provider(online: bool) -> Provider:
    return "dashscope" if online else "ollama"


def _resolved():
    """Read runtime resolved config (agent override → system → env). 不引循环 import."""
    from app.services.runtime_config import resolve_for_current
    return resolve_for_current()


def _utility_model_name() -> str:
    cfg = _resolved()
    # legacy override fields (settings.utility_model / ollama_model) 仍保留兼容,
    # 它们没在 SystemConfig 表里所以不能动态改; 改不到时就用动态 small_model.
    return (
        settings.utility_model
        or settings.ollama_model
        or (cfg.remote_small_model if cfg.online_model else cfg.local_small_model)
    )


def _chat_model_name() -> str:
    cfg = _resolved()
    return settings.chat_model or (
        cfg.remote_chat_model if cfg.online_model else cfg.local_chat_model
    )


def _embedding_model_name() -> str:
    return settings.embedding_model  # spec: embedding 不动态


def _provider_for(role: str) -> Provider:
    override = getattr(settings, f"{role}_provider", "") or ""
    if role == "embedding":
        # spec: embedding 不动态. settings.online_model 切换不应让 embedding
        # provider 翻 (ollama bge-m3=1024 维 ↔ dashscope text-embedding-* 维度不同),
        # 否则 pgvector 老向量跟新查询向量混在一起立刻失真. 用 env 默认 ollama.
        provider = (override or settings.llm_provider or "ollama").strip().lower()
    else:
        cfg = _resolved()
        remote_provider = (
            cfg.remote_chat_provider if role == "chat" else cfg.remote_small_provider
        )
        provider = (
            override
            or settings.llm_provider
            or (remote_provider if cfg.online_model else _default_provider(False))
        ).strip().lower()
    spec = get_provider(provider)
    if role == "embedding" and not spec.supports_embeddings:
        raise ValueError(
            f"Provider {provider} does not provide supported embeddings; "
            "use ollama or dashscope"
        )
    return spec.id


@lru_cache(maxsize=256)
def _build_chat_model(provider: str, model: str) -> BaseChatModel:
    return build_chat_model(provider, model)


@lru_cache(maxsize=256)
def _build_utility_model(provider: str, model: str) -> BaseChatModel:
    value = build_chat_model(provider, model)
    object.__setattr__(value, "_companion_role", "utility")
    return value


@lru_cache(maxsize=256)
def _build_grounding_model(provider: str, model: str) -> BaseChatModel:
    # The calibrated verifier is not a small-model task, even on DashScope.
    value = build_chat_model(provider, model)
    object.__setattr__(value, "_companion_role", "grounding")
    return value


@lru_cache(maxsize=256)
def _build_fallback_chat_model(provider: str, model: str) -> BaseChatModel:
    return build_chat_model(provider, model)


@lru_cache(maxsize=256)
def _build_fallback_utility_model(provider: str, model: str) -> BaseChatModel:
    return build_chat_model(provider, model)


# 调用方按 OpenAI 习惯传的采样参数 → ChatOllama 上的同义字段。
_OLLAMA_SAMPLING_FIELDS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "seed": "seed",
    "max_tokens": "num_predict",
}


def _for_ollama(model: BaseChatModel, kwargs: dict[str, Any]) -> tuple[BaseChatModel, dict[str, Any]]:
    """把采样参数从调用 kwargs 挪到 ChatOllama 实例上。

    OpenAI 兼容的模型把 ainvoke 的额外 kwargs 当请求参数, temperature=0 正常生效;
    ChatOllama 却把它们原样平铺进 AsyncClient.chat(**params), 直接 TypeError ——
    生产上地点核验 (invoke_json(..., temperature=0)) 主 provider 一挂, 兜底就三次
    全报错, 兜底形同虚设。采样参数在 Ollama 里属于 options, 这里换成同名字段的
    副本, 由 ChatOllama 自己拼 options, 模型原有的 num_ctx 等配置也不会被冲掉。
    """
    if not isinstance(model, ChatOllama):
        return model, {**kwargs}
    fields = {
        _OLLAMA_SAMPLING_FIELDS[key]: value
        for key, value in kwargs.items()
        if key in _OLLAMA_SAMPLING_FIELDS and value is not None
    }
    rest = {key: value for key, value in kwargs.items() if key not in _OLLAMA_SAMPLING_FIELDS}
    return (model.model_copy(update=fields) if fields else model), rest


def get_chat_model() -> BaseChatModel:
    # Content-addressed cache: an old in-flight turn cannot poison a new turn's bucket.
    return _build_chat_model(_provider_for("chat"), _chat_model_name())


def get_utility_model() -> BaseChatModel:
    return _build_utility_model(_provider_for("utility"), _utility_model_name())


def get_grounding_model() -> BaseChatModel:
    """Use the calibrated verifier on DashScope; keep offline/provider routing."""
    provider = _provider_for("utility")
    model = settings.persona_grounding_model if provider == "dashscope" else _utility_model_name()
    return _build_grounding_model(provider, model)


def get_fallback_chat_model() -> BaseChatModel:
    return _build_fallback_chat_model(
        settings.fallback_chat_provider, settings.fallback_chat_model,
    )


def get_fallback_utility_model() -> BaseChatModel:
    return _build_fallback_utility_model(
        settings.fallback_utility_provider, settings.fallback_utility_model,
    )


def get_fallback_for(primary: BaseChatModel) -> BaseChatModel | None:
    """Keep offline mode offline and never retry the identical model as a fallback.

    A missing fallback credential must not prevent a healthy primary request.
    Do not include the constructor exception (which may contain config) in logs.
    """
    from app.services.llm.resilience import provider_name

    if provider_name(primary) == "ollama":
        return None
    try:
        fallback = (get_fallback_utility_model()
                    if getattr(primary, "_companion_role", None) == "utility"
                    else get_fallback_chat_model())
    except ValueError:
        logger.error("[LLM-FALLBACK] backup model configuration unavailable")
        return None
    return None if _resolve_usage_model_key(primary) == _resolve_usage_model_key(fallback) else fallback


# 公开 cache_clear hook 给 runtime_config 调用 (PUT 配置后清缓存)
get_chat_model.cache_clear = _build_chat_model.cache_clear  # type: ignore[attr-defined]
get_utility_model.cache_clear = _build_utility_model.cache_clear  # type: ignore[attr-defined]
get_fallback_chat_model.cache_clear = _build_fallback_chat_model.cache_clear  # type: ignore[attr-defined]
get_grounding_model.cache_clear = _build_grounding_model.cache_clear  # type: ignore[attr-defined]
get_fallback_utility_model.cache_clear = _build_fallback_utility_model.cache_clear  # type: ignore[attr-defined]


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
    if provider != "ollama":
        raise ValueError(f"{provider} does not provide supported embeddings; use ollama or dashscope")
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
    model: BaseChatModel, override: "str | CallProfile | None",
):
    """自动从 model 识别 provider 和默认 profile. override 可强制指定 profile.

    默认按"是否是 get_chat_model() 返回的 lru_cache 单例"区分: chat_model →
    chat_extract (45s, 1 retry); utility_model 或自建 model → utility_fast
    (8s, 2 retry). 调用方可传 profile="background" 等 override.

    override 接受两种形式 (防 foot-gun):
    - str: profile 名字, 内部 get_profile 查 registry (主流用法)
    - CallProfile: 已构造好的对象直接用 (用于动态构造 / 测试注入)
    误传非 str 非 CallProfile 的对象 (e.g. get_profile(...) 嵌套调用产生的旧
    bug) 会被 isinstance 兜住直接抛 TypeError, 不再静默 KeyError 吞异常 →
    extract_memories 永远返空 dict 这类 silent failure (trace 019e0004).
    """
    from app.services.llm.resilience import CallProfile, get_profile, provider_name

    provider = provider_name(model)
    if override is None:
        resolved = get_profile("chat_extract") if model is get_chat_model() else get_profile("utility_fast")
        return provider, resolved
    if isinstance(override, CallProfile):
        return provider, override
    if isinstance(override, str):
        return provider, get_profile(override)
    raise TypeError(  # type: ignore[unreachable]
        f"profile override must be str | CallProfile | None, got {type(override).__name__}"
    )


async def _invoke_with_resilience(
    model: BaseChatModel,
    prompt: str | list[BaseMessage],
    *,
    op: str,
    force_json: bool = False,
    **kwargs: Any,
) -> Any:
    """invoke_text / invoke_json_with_usage 的共享骨架.

    force_json=True 仅对 ChatOllama 添加 format='json'。云端备选沿用原
    JSON 提示词约束，不把 Ollama 专属参数发送给云端 API。

    返回原始 LangChain response 对象 (未解析), 让调用方自行处理
    .content / _extract_json / usage_metadata 等.

    Streaming 分支: profile.stream_mode=True 时, 内部 astream + 累积 chunks,
    用 AIMessage 包装让上层 `.content` 路径无感. 享受 idle_timeout 保护
    (相邻 chunk 间最大停顿), 不受总 timeout 误杀. 用于 character.generation
    这类 6-8K tokens 长输出场景.
    """
    from app.services.llm.resilience import call_with_resilience, provider_name

    profile_override = kwargs.pop("profile", None)
    # Optional caller-owned audit output. Never forward it to a provider or
    # infer a successful route from the requested model after a fallback.
    invocation_metadata = kwargs.pop("invocation_metadata", None)
    if invocation_metadata is not None:
        invocation_metadata.clear()
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
            invocation_metadata=invocation_metadata,
        )

    primary_model, primary_kwargs = _for_ollama(model, kwargs)

    async def _primary():
        response = await primary_model.ainvoke(messages, **primary_kwargs)
        if invocation_metadata is not None:
            invocation_metadata["actual_model"] = _resolve_usage_model_key(primary_model)
        return response

    fallback_factory = None
    fallback = get_fallback_for(model) if profile.allow_ollama_fallback else None
    if fallback is not None:
        _fb_model, fb_kwargs = _for_ollama(fallback, kwargs)
        if force_json and isinstance(_fb_model, ChatOllama):
            fb_kwargs.setdefault("format", "json")

        async def _fallback():
            response = await _fb_model.ainvoke(messages, **fb_kwargs)
            if invocation_metadata is not None:
                invocation_metadata["actual_model"] = _resolve_usage_model_key(_fb_model)
            return response
        fallback_factory = _fallback

    return await call_with_resilience(
        _primary,
        primary_provider=provider,
        profile=profile,
        op=op,
        fallback_factory=fallback_factory,
        fallback_provider=provider_name(fallback) if fallback is not None else "",
        primary_model_name=_resolve_usage_model_key(model),
        role=getattr(model, "_companion_role", "chat"),
        fallback_model_name=_resolve_usage_model_key(fallback) if fallback is not None else "",
    )


async def _invoke_via_stream(
    model: BaseChatModel,
    messages: list[BaseMessage],
    *,
    provider: str,
    profile: "CallProfile",
    op: str,
    force_json: bool,
    invocation_metadata: dict | None = None,
    **kwargs: Any,
) -> Any:
    """Streaming 适配器: 与 ainvoke 等价的接口, 内部走 astream + 累积 chunks.

    返回 AIMessage(content=joined), 上层 `_extract_json(response.content)` /
    `response.content` 路径无感.

    用量在 collect_stream 内按实际供应商/模型记录，返回对象不重复携带 usage，
    防止外层再次计费；invocation_metadata 只在整个调用成功后写入。
    """
    from app.services.llm.resilience import collect_stream, provider_name
    from langchain_core.messages import AIMessage

    primary_model, primary_kwargs = _for_ollama(model, kwargs)
    selected_model = None

    def _primary_stream():
        nonlocal selected_model
        selected_model = primary_model
        return primary_model.astream(messages, **primary_kwargs)

    fallback_factory = None
    fb_model_name = ""
    fallback = get_fallback_for(model) if profile.allow_ollama_fallback else None
    if fallback is not None:
        _fb_model, fb_kwargs = _for_ollama(fallback, kwargs)
        fb_model_name = _resolve_usage_model_key(_fb_model)
        if force_json and isinstance(_fb_model, ChatOllama):
            fb_kwargs.setdefault("format", "json")

        def _fallback_stream():
            nonlocal selected_model
            selected_model = _fb_model
            return _fb_model.astream(messages, **fb_kwargs)
        fallback_factory = _fallback_stream

    text = await collect_stream(
        _primary_stream,
        primary_provider=provider,
        profile=profile,
        op=op,
        fallback_factory=fallback_factory,
        primary_model_name=_resolve_usage_model_key(model),
        role=getattr(model, "_companion_role", "chat"),
        fallback_model_name=fb_model_name,
        fallback_provider=provider_name(fallback) if fallback is not None else "",
    )
    if invocation_metadata is not None and selected_model is not None:
        invocation_metadata["actual_model"] = _resolve_usage_model_key(selected_model)
    return AIMessage(content=text)


async def invoke_json(
    model: BaseChatModel,
    prompt: str | list[BaseMessage],
    **kwargs: Any,
) -> Any:
    """Invoke the model and parse the response as JSON."""
    parsed, _ = await invoke_json_with_usage(model, prompt, **kwargs)
    return parsed


def _resolve_model_name(model: BaseChatModel) -> str:
    """Best-effort model name from a langchain BaseChatModel; e.g. "qwen3.5-flash"."""
    for attr in ("model_name", "model", "model_id"):
        v = getattr(model, attr, None)
        if isinstance(v, str) and v:
            return v
    return ""


def _resolve_usage_model_key(model: BaseChatModel) -> str:
    """Return provider-qualified model key for usage/pricing disambiguation."""
    from app.services.llm.resilience import provider_name

    name = _resolve_model_name(model)
    provider = provider_name(model)
    if not name:
        return provider or "unknown"
    if provider and provider != "unknown":
        return f"{provider}/{name}"
    return name


def _extract_cached_input_tokens(meta: dict) -> int:
    """从 usage_metadata 提取 prefix cache 命中的 input tokens.

    LangChain 把各 provider 的缓存字段归一到 input_token_details.cache_read
    (DeepSeek prompt_cache_hit_tokens / OpenAI+qwen prompt_tokens_details.
    cached_tokens); 兼容旧版 langchain 直接透传 cached_tokens 的情况.
    """
    details = meta.get("input_token_details")
    if isinstance(details, dict):
        for key in ("cache_read", "cached_tokens"):
            value = details.get(key)
            if value:
                return int(value)
    return 0


def _record_usage_from_response(model: BaseChatModel, response: Any, *, actual_model: str | None = None) -> dict:
    """提取 response.usage_metadata 并 record 到当前 chat session, 返回 usage dict."""
    from app.services.llm import usage_tracker

    meta = getattr(response, "usage_metadata", None)
    if not isinstance(meta, dict):
        return {}
    input_tok = int(meta.get("input_tokens", 0) or 0)
    output_tok = int(meta.get("output_tokens", 0) or 0)
    cached_tok = _extract_cached_input_tokens(meta)
    usage_tracker.record(
        actual_model or _resolve_usage_model_key(model), input_tok, output_tok,
        cached_input_tokens=cached_tok,
    )
    return {"input_tokens": input_tok, "output_tokens": output_tok}


async def invoke_json_with_usage(
    model: BaseChatModel,
    prompt: str | list[BaseMessage],
    **kwargs: Any,
) -> tuple[Any, dict]:
    """Same as invoke_json but also returns `{input_tokens, output_tokens}` when
    available from the model. Empty dict if the provider doesn't expose usage.

    Resilience: primary 经 timeout+retry+CB, 失败后切配置的备选模型. 两级都挂抛
    LLMFailedError (调用方自己兜底, 多数已有 try/except 返 None/空 dict).
    """
    metadata = kwargs.get("invocation_metadata")
    if metadata is None:
        metadata = kwargs["invocation_metadata"] = {}
    response = await _invoke_with_resilience(
        model, prompt, op="invoke_json", force_json=True, **kwargs,
    )
    usage = _record_usage_from_response(model, response, actual_model=metadata.get("actual_model"))
    return _extract_json(response.content), usage


async def invoke_text(
    model: BaseChatModel,
    prompt: str | list[BaseMessage],
    **kwargs: Any,
) -> str:
    """Invoke the model and return the response as plain text.

    Resilience: primary 经 timeout+retry+CB, 失败后切配置的备选模型. 两级都挂抛
    LLMFailedError.
    """
    metadata = kwargs.get("invocation_metadata")
    if metadata is None:
        metadata = kwargs["invocation_metadata"] = {}
    response = await _invoke_with_resilience(
        model, prompt, op="invoke_text", force_json=False, **kwargs,
    )
    _record_usage_from_response(model, response, actual_model=metadata.get("actual_model"))
    return response.content
