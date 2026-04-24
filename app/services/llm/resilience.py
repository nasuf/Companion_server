"""LLM resilience layer: circuit breaker + retry + timeout + Ollama fallback.

设计意图: 每条用户消息会 await 多次 LLM (意图识别 + 违禁检测 + 主回复
astream + 情绪分析 + 记忆抽取 …), 一旦 Dashscope 抽风 5 分钟, 每条请求
都会等满 300s timeout, asyncio event loop 被阻塞, 全站卡死.

本模块集中保护三个维度:
1. timeout: 每个 call profile 有显式上限 (utility_fast 8s / chat_extract
   45s / chat_stream 90s), 禁止长时间阻塞 event loop
2. retry + exponential backoff: 瞬时网络抖动不打扰用户; 流式调用**不 retry**
   (stream 语义不允许)
3. circuit breaker: 滑动窗口失败率超阈值 → fast-fail 快速降级到本地 Ollama,
   event loop 不被拖慢
4. Ollama fallback: primary provider (Dashscope 等) 彻底失败后, 自动切到本地
   LOCAL_CHAT_MODEL 再试一次, 保证"最坏情况下还有 AI 能回话"

入口通过 `call_with_resilience` (unary) 和 `astream_with_resilience` (stream);
models.py 里的 invoke_text / invoke_json / 主回复 _run_main_llm 用这两个入口,
31 个 LLM caller 不需要改动即获得保护.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable

from app.config import settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Exceptions
# ═══════════════════════════════════════════════════════════════════

class LLMFailedError(RuntimeError):
    """所有 retry + fallback 都失败后的终极异常, 调用方需要自己兜底 (静态文本等)."""


class LLMCircuitOpenError(LLMFailedError):
    """CB 是 open 状态, call 未 await LLM 直接被拒绝."""


# ═══════════════════════════════════════════════════════════════════
# Call profiles (policy presets per call class)
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CallProfile:
    timeout_s: float
    max_retries: int
    retry_backoff_s: tuple[float, ...]
    first_chunk_timeout_s: float = 30.0  # 流式专用

    def __post_init__(self) -> None:
        # retry_backoff_s 长度须覆盖 max_retries 次退避
        if len(self.retry_backoff_s) < self.max_retries:
            raise ValueError(
                f"retry_backoff_s needs at least {self.max_retries} entries, "
                f"got {len(self.retry_backoff_s)}",
            )


_PROFILES_CACHE: dict[str, CallProfile] | None = None


def _profiles() -> dict[str, CallProfile]:
    """从 settings 构建 profiles, 热路径级缓存避免每次 get_profile 重建 dict.

    settings 运行时不变 (部署级配置), cache 安全. 测试需动态改 settings 时
    先调 reset_profiles_cache_for_testing().
    """
    global _PROFILES_CACHE
    if _PROFILES_CACHE is None:
        _PROFILES_CACHE = {
            "utility_fast": CallProfile(
                timeout_s=settings.llm_utility_timeout_s,
                max_retries=2,
                retry_backoff_s=(0.5, 2.0),
            ),
            "chat_extract": CallProfile(
                timeout_s=settings.llm_chat_extract_timeout_s,
                max_retries=1,
                retry_backoff_s=(1.0,),
            ),
            "chat_stream": CallProfile(
                timeout_s=settings.llm_chat_stream_timeout_s,
                max_retries=0,
                retry_backoff_s=(),
                first_chunk_timeout_s=settings.llm_chat_stream_first_chunk_timeout_s,
            ),
            "background": CallProfile(
                timeout_s=120.0,
                max_retries=2,
                retry_backoff_s=(1.0, 5.0),
            ),
        }
    return _PROFILES_CACHE


def get_profile(name: str) -> CallProfile:
    """按名字取 profile. 未知名字抛 KeyError (编程错误, 不应静默默认)."""
    return _profiles()[name]


def reset_profiles_cache_for_testing() -> None:
    """清 profile cache (测试动态改 settings 前调用)."""
    global _PROFILES_CACHE
    _PROFILES_CACHE = None


# ═══════════════════════════════════════════════════════════════════
# Circuit Breaker (per provider)
# ═══════════════════════════════════════════════════════════════════

class CircuitBreaker:
    """三态熔断: closed / open / half_open.

    - closed: 正常放行, 滑动窗口累计失败次数, 达阈值 → open
    - open: 冷却期内 try_acquire 直接返回 False, 不 await LLM
    - half_open: 冷却期过后允许恰好 1 个 probe; 成功 → closed, 失败 → 重新 open

    单 asyncio loop, 不需要 threading.Lock.
    """

    def __init__(
        self,
        *,
        failure_threshold: int,
        window_sec: float,
        cooldown_sec: float,
    ) -> None:
        self._threshold = failure_threshold
        self._window_sec = window_sec
        self._cooldown_sec = cooldown_sec
        self._failures: deque[float] = deque()
        self._opened_at: float | None = None
        self._half_open_inflight: bool = False

    def _prune_old(self) -> None:
        now = time.monotonic()
        while self._failures and now - self._failures[0] > self._window_sec:
            self._failures.popleft()

    def state(self) -> str:
        """实时状态 (open 会根据 cooldown 自动变 half_open)."""
        if self._opened_at is None:
            return "closed"
        if time.monotonic() - self._opened_at < self._cooldown_sec:
            return "open"
        return "half_open"

    def try_acquire(self) -> bool:
        """返回 True 表示可以 await LLM; False 表示应 fast-fail.

        half_open 状态只放行 1 个并发 probe.
        """
        s = self.state()
        if s == "closed":
            return True
        if s == "open":
            return False
        # half_open
        if self._half_open_inflight:
            return False
        self._half_open_inflight = True
        return True

    def record_success(self) -> None:
        self._failures.clear()
        self._opened_at = None
        self._half_open_inflight = False

    def record_failure(self) -> None:
        self._half_open_inflight = False
        self._failures.append(time.monotonic())
        self._prune_old()
        if self._opened_at is None and len(self._failures) >= self._threshold:
            self._opened_at = time.monotonic()
        elif self._opened_at is not None and self.state() == "half_open":
            # probe 失败 → 重置 cooldown
            self._opened_at = time.monotonic()


_breakers: dict[str, CircuitBreaker] = {}


def _get_breaker(provider: str) -> CircuitBreaker:
    if provider not in _breakers:
        _breakers[provider] = CircuitBreaker(
            failure_threshold=settings.llm_cb_failure_threshold,
            window_sec=settings.llm_cb_window_sec,
            cooldown_sec=settings.llm_cb_cooldown_sec,
        )
    return _breakers[provider]


def reset_breakers_for_testing() -> None:
    _breakers.clear()


def provider_name(model: Any) -> str:
    """通过 LangChain model class 识别 provider. Primary / fallback 都走此函数,
    避免在 models.py 和 reply_generate.py 里重复 isinstance 判断.

    未知类型返回 "unknown"; CB 会为其单独建一个 breaker, 不影响 Dashscope/Ollama.
    """
    # 懒导入: resilience.py 作为 pure utility 不直接 import LangChain,
    # 避免循环依赖风险 + 允许纯 Python 的 CB/retry 单元测试免 LangChain.
    from langchain_anthropic import ChatAnthropic
    from langchain_ollama import ChatOllama
    from langchain_openai import ChatOpenAI

    if isinstance(model, ChatOllama):
        return "ollama"
    if isinstance(model, ChatOpenAI):
        return "dashscope"
    if isinstance(model, ChatAnthropic):
        return "claude"
    return "unknown"


# ═══════════════════════════════════════════════════════════════════
# Core unary call with retry + CB (single provider)
# ═══════════════════════════════════════════════════════════════════

def _log_attempt(
    *,
    provider: str,
    op: str,
    result: str,
    started: float,
    attempt: int | None = None,
    exc: Exception | None = None,
) -> None:
    """统一 [LLM] 结构化日志. result in {ok, timeout, error, mid_timeout,
    mid_error, first_chunk_fail}. 成功用 info, 其他 warning."""
    elapsed_ms = int((time.monotonic() - started) * 1000)
    parts = [f"provider={provider}", f"op={op}", f"result={result}", f"latency_ms={elapsed_ms}"]
    if attempt is not None:
        parts.append(f"attempt={attempt}")
    if exc is not None:
        parts.append(f"exc={type(exc).__name__}: {exc}")
    line = "[LLM] " + " ".join(parts)
    (logger.info if result == "ok" else logger.warning)(line)


async def _run_with_retry(
    factory: Callable[[], Awaitable[Any]],
    *,
    provider: str,
    profile: CallProfile,
    op: str,
) -> Any:
    if not settings.llm_resilience_enabled:
        # Kill switch: 只保留 timeout, 跳过 CB + retry
        return await asyncio.wait_for(factory(), timeout=profile.timeout_s)

    breaker = _get_breaker(provider)
    if not breaker.try_acquire():
        raise LLMCircuitOpenError(f"circuit open for {provider} on {op}")

    last_exc: Exception | None = None
    for attempt in range(profile.max_retries + 1):
        started = time.monotonic()
        try:
            result = await asyncio.wait_for(factory(), timeout=profile.timeout_s)
            breaker.record_success()
            _log_attempt(provider=provider, op=op, result="ok",
                         started=started, attempt=attempt)
            return result
        except asyncio.TimeoutError as e:
            last_exc = e
            _log_attempt(provider=provider, op=op, result="timeout",
                         started=started, attempt=attempt)
        except Exception as e:
            last_exc = e
            _log_attempt(provider=provider, op=op, result="error",
                         started=started, attempt=attempt, exc=e)

        breaker.record_failure()
        if attempt >= profile.max_retries:
            break
        await asyncio.sleep(profile.retry_backoff_s[attempt])

    raise LLMFailedError(
        f"{op} on {provider} failed after {profile.max_retries + 1} attempts: "
        f"{type(last_exc).__name__}: {last_exc}"
    )


# ═══════════════════════════════════════════════════════════════════
# Entry: call_with_resilience (unary + auto Ollama fallback)
# ═══════════════════════════════════════════════════════════════════

async def call_with_resilience(
    primary_factory: Callable[[], Awaitable[Any]],
    *,
    primary_provider: str,
    profile: CallProfile,
    op: str,
    fallback_factory: Callable[[], Awaitable[Any]] | None = None,
) -> Any:
    """Unary LLM call with timeout + retry + CB + 可选 Ollama fallback.

    primary_provider 决定使用哪个 CB (各 provider 独立计数).
    fallback_factory 非 None 时, primary 最终失败 (retry 耗尽 / CB open) 会
    再走一次 Ollama, Ollama 也有独立 CB + retry.
    primary 本就是 Ollama 时, 调用方应传 fallback_factory=None (本地重试意义不大).

    同一级别 fallback 也可能失败; 最终抛 LLMFailedError, 调用方自己静态兜底.
    """
    try:
        return await _run_with_retry(
            primary_factory, provider=primary_provider, profile=profile, op=op,
        )
    except LLMFailedError as primary_exc:
        if fallback_factory is None:
            raise
        logger.warning(
            f"[LLM-FALLBACK] op={op} primary={primary_provider} failed: "
            f"{type(primary_exc).__name__}: {primary_exc}; trying ollama",
        )

    try:
        return await _run_with_retry(
            fallback_factory, provider="ollama", profile=profile, op=f"{op}:fallback",
        )
    except LLMFailedError as fallback_exc:
        raise LLMFailedError(
            f"{op}: both primary={primary_provider} and Ollama fallback failed; "
            f"last error: {fallback_exc}"
        ) from fallback_exc


# ═══════════════════════════════════════════════════════════════════
# Entry: astream_with_resilience (stream + auto Ollama fallback)
# ═══════════════════════════════════════════════════════════════════

async def astream_with_resilience(
    primary_factory: Callable[[], AsyncIterator[Any]],
    *,
    primary_provider: str,
    profile: CallProfile,
    op: str,
    fallback_factory: Callable[[], AsyncIterator[Any]] | None = None,
) -> AsyncIterator[str]:
    """Stream LLM output, yield text tokens.

    策略 (stream 语义不允许中途 retry, 所以和 unary 不同):
    - 若 first chunk 在 `first_chunk_timeout_s` 内未到 → 视为 primary 失败,
      尝试 Ollama fallback 流
    - First chunk 到达后 commit 到 primary, 中途若超总 timeout 或抛异常, 不再
      fallback (用户已经在接收流, 再切 Ollama 重头再来会看到两段拼接文本),
      直接抛 LLMFailedError, 调用方自己静态兜底
    - Circuit breaker: pre-first-chunk 失败计入 CB; first chunk 到达即视为成功
    - Kill switch (llm_resilience_enabled=False): 纯原始 stream + 首 chunk 超时
    """
    got_first_chunk = False

    try:
        async for token in _stream_provider(
            primary_factory,
            provider=primary_provider,
            profile=profile,
            op=op,
        ):
            got_first_chunk = True
            yield token
        return
    except LLMFailedError as primary_exc:
        # 已经推过 token 或无 fallback → 直接让异常穿出, 调用方自己兜底
        if got_first_chunk or fallback_factory is None:
            raise
        logger.warning(
            f"[LLM-FALLBACK] op={op} primary={primary_provider} stream failed "
            f"pre-first-chunk: {type(primary_exc).__name__}: {primary_exc}; trying ollama",
        )

    # Fallback to Ollama
    try:
        async for token in _stream_provider(
            fallback_factory, provider="ollama", profile=profile, op=f"{op}:fallback",
        ):
            yield token
    except LLMFailedError as fallback_exc:
        raise LLMFailedError(
            f"{op}: primary={primary_provider} and Ollama fallback both failed: "
            f"{fallback_exc}"
        ) from fallback_exc


async def _stream_provider(
    factory: Callable[[], AsyncIterator[Any]],
    *,
    provider: str,
    profile: CallProfile,
    op: str,
) -> AsyncIterator[str]:
    """在单一 provider 上跑 stream; 管理 first_chunk_timeout 和总 timeout + CB."""
    if not settings.llm_resilience_enabled:
        started = time.monotonic()
        async for chunk in factory():
            # 仅尊重总 timeout (killswitch 场景保留基本防卡)
            if time.monotonic() - started > profile.timeout_s:
                raise LLMFailedError(f"{op} on {provider}: exceeded overall timeout (killswitch mode)")
            yield _chunk_text(chunk)
        return

    breaker = _get_breaker(provider)
    if not breaker.try_acquire():
        raise LLMCircuitOpenError(f"circuit open for {provider} on {op}")

    started = time.monotonic()
    stream = factory()
    aiter = stream.__aiter__()

    # First chunk with tight timeout (timeout / empty stream / upstream error 统一
    # 视为 primary 失败, 触发 fallback. 捕获后必须 aclose 释放底层 SSE/HTTP 连接,
    # 不能依赖 GC — Dashscope 抽风时 CB 会 open, 多次 timeout 堆积 fd 可能打爆.)
    try:
        first = await asyncio.wait_for(
            aiter.__anext__(), timeout=profile.first_chunk_timeout_s,
        )
    except Exception as e:
        breaker.record_failure()
        _log_attempt(provider=provider, op=op, result="first_chunk_fail",
                     started=started, exc=e)
        _close = getattr(stream, "aclose", None)
        if _close is not None:
            try:
                await _close()
            except Exception:
                pass  # aclose 本身失败不该覆盖原始异常
        raise LLMFailedError(f"stream no first chunk on {provider}: {e}") from e

    yield _chunk_text(first)
    deadline = started + profile.timeout_s

    # Remaining chunks with overall budget
    try:
        async for chunk in aiter:
            if time.monotonic() > deadline:
                breaker.record_failure()
                _log_attempt(provider=provider, op=op, result="mid_timeout",
                             started=started)
                raise LLMFailedError(f"stream exceeded overall {profile.timeout_s}s on {provider}")
            yield _chunk_text(chunk)
        breaker.record_success()
        _log_attempt(provider=provider, op=op, result="ok", started=started)
    except LLMFailedError:
        raise
    except Exception as e:
        breaker.record_failure()
        _log_attempt(provider=provider, op=op, result="mid_error",
                     started=started, exc=e)
        raise LLMFailedError(f"stream mid error on {provider}: {e}") from e


def _chunk_text(chunk: Any) -> str:
    """LangChain chunk 有 .content 字段 (AIMessageChunk), 其他 provider 也可能直接是 str."""
    if hasattr(chunk, "content"):
        return str(chunk.content)
    return str(chunk)
