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
5. 并发上限: 每个 provider 的在途请求数封顶, 且后台任务只能占其中一小部分,
   保证突发时前台聊天仍有槽位 (见 _llm_slot)

入口通过 `call_with_resilience` (unary) 和 `astream_with_resilience` (stream);
models.py 里的 invoke_text / invoke_json / 主回复 _run_main_llm 用这两个入口,
31 个 LLM caller 不需要改动即获得保护.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
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
    first_chunk_timeout_s: float = 30.0  # 流式专用: 首 chunk 等待
    # primary 失败后是否回退到 ollama. 大输出场景 (character.generation 6-8K
    # tokens) 主云超时往往是"输出量大", 此时 ollama 14B 必然更慢, fallback 反
    # 而拖慢用户 6+ 分钟. 设 False 则 fail-fast.
    allow_ollama_fallback: bool = True
    # 流式相邻 chunk 间最大允许停顿. None → 沿用 first_chunk_timeout_s.
    # 解决 dashscope qwen-plus 偶尔 30-40s 静默(GPU 排队/限流)导致总 timeout
    # 误判. 只要 LLM 在持续吐字, 就不会触发.
    idle_timeout_s: float | None = None
    # True 时 unary API (invoke_json/invoke_text) 内部走 streaming 累积, 享受
    # idle_timeout 保护. 用于 character.generation 这类 6-8K tokens 长输出.
    # API 不变, 调用方零改动. 默认 False, 不影响其他 profile.
    stream_mode: bool = False

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
                # character.generation 输出 ~6-8K tokens (87 字段), qwen-plus 稳态
                # ~50-80 tok/s → 80-160s. timeout_s 是 safety 网, 真正决定生死的
                # 是 first_chunk_timeout_s + idle_timeout_s. 600s 留给最坏情况.
                timeout_s=600.0,
                # timeout 后不重试: 重试同 prompt 必然再超时, 只浪费 4-8 分钟用户
                # 等待. 真网络抖动是少数, 让前端 failed UI 引导重建更快.
                max_retries=0,
                retry_backoff_s=(),
                # 主云超时 = LLM 输出量大, 本地 14B 必然更慢. 不 fallback,
                # 立即 fail-fast 由前端引导用户重建.
                allow_ollama_fallback=False,
                # dashscope cold-start 偶尔 30-40s 才返首字节, 给 45s 留余量.
                first_chunk_timeout_s=45.0,
                # 相邻 chunk 间隔上限. dashscope qwen-plus 偶尔 30-40s 静默
                # (GPU 排队/限流), 60s 不误杀;但持续断流可在 60s 内识别失败.
                idle_timeout_s=60.0,
                # ★ 启用 streaming. invoke_json 内部走 astream + 累积, 同样的
                # API 但享受 idle_timeout 保护. 见 models.py:_invoke_via_stream.
                stream_mode=True,
            ),
            "memory_extract": CallProfile(
                # 用户一次性发"画像 dump" (e.g. 血型/体重/体型/穿搭/忌口/亲戚...
                # 20+ 条事实) 时 LLM 输出 1500-2500 tokens JSON, qwen-plus 稳态
                # ~50-80 tok/s → 25-50s. chat_extract 45s 配置经常踩边界 →
                # asyncio.wait_for cancel 时 langchain callback 跳过 →
                # langsmith trace 永留 status=pending → silent failure 用户
                # 20+ 条事实全丢. 生产 trace 复现 (2026-04-29 019dd808).
                #
                # 修: 走 streaming + idle_timeout 保护. 对大输出 idle_timeout
                # 决定生死 (持续吐字就不挂, 只在持续断流 60s 才砍), 总 180s 留
                # safety. 跟 background 同范式但允许 ollama fallback (extraction
                # 失败比丢 silent 还差, 14B 兜底有价值; character_generation 是
                # 用户阻塞热路径, ollama 慢 6 分钟反而坏体感, extraction 是后台
                # fire-and-forget, ollama 慢点无感).
                timeout_s=180.0,
                max_retries=0,
                retry_backoff_s=(),
                allow_ollama_fallback=True,
                first_chunk_timeout_s=30.0,
                idle_timeout_s=60.0,
                stream_mode=True,
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


def set_profiles_for_testing(mapping: dict[str, CallProfile]) -> None:
    """直接注入一组 profile (仅测试用), 绕过 settings 构造逻辑.

    比 monkeypatch.setattr(_PROFILES_CACHE, ...) 更稳定: 不依赖私有名字,
    将来 cache 实现变化 (比如改成多层 dict / TTL) 也只需改一处.
    """
    global _PROFILES_CACHE
    _PROFILES_CACHE = dict(mapping)


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
        tagged = getattr(model, "_companion_provider", "")
        if isinstance(tagged, str) and tagged:
            return tagged
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
    _record_runtime_event(result=result, latency_ms=elapsed_ms)
    parts = [f"provider={provider}", f"op={op}", f"result={result}", f"latency_ms={elapsed_ms}"]
    if attempt is not None:
        parts.append(f"attempt={attempt}")
    if exc is not None:
        parts.append(f"exc={type(exc).__name__}: {exc}")
    line = "[LLM] " + " ".join(parts)
    (logger.info if result == "ok" else logger.warning)(line)


def _record_runtime_event(*, result: str, latency_ms: int | None = None) -> None:
    """Best-effort bridge from resilience decisions to llm_usage metrics."""
    try:
        from app.services.llm import usage_tracker
        usage_tracker.record_runtime_event(result=result, latency_ms=latency_ms)
    except Exception:
        return


# ═══════════════════════════════════════════════════════════════════
# 并发上限: 按 provider 分, 后台任务受更紧的配额
# ═══════════════════════════════════════════════════════════════════

# 后台 scope。这些调用不在用户等待的链路上, 突发时应该给前台让路。
# post_process 虽然由用户消息触发, 但发生在回复推送之后, 慢一点用户感知不到。
_BACKGROUND_SCOPES = frozenset({
    "schedule_cron", "post_process", "proactive", "agent_creation",
    "offline", "music",
})

_slots: dict[str, asyncio.Semaphore] = {}
_bg_slots: dict[str, asyncio.Semaphore] = {}


def _get_semaphore(pool: dict[str, asyncio.Semaphore], key: str, size: int) -> asyncio.Semaphore:
    """Lazy init 让 Semaphore 绑到当前 event loop.

    模块级初始化会在测试里出 "attached to a different loop" —— 每个 test 有自己
    的 loop。跟 proactive/triggers.py 里那个 trigger semaphore 同样的处理。

    首次创建后大小就固定了。上限来自环境变量, 运行时本就不变; 要改必须重启, 这跟
    改这个值的方式 (调 GH 变量后重新部署) 是一致的。
    """
    sem = pool.get(key)
    if sem is None:
        sem = asyncio.Semaphore(size)
        pool[key] = sem
    return sem


def reset_slots_for_testing() -> None:
    _slots.clear()
    _bg_slots.clear()


def _per_worker_share(total: int | None) -> int:
    """把全局上限摊到本 worker.

    信号量是进程内对象, 多 worker 下每个进程各持一份 —— 直接用配置值的话, 2 个
    worker 就变成全局 128 而不是 64, 而 provider 的 rate limit 是按全局算的。配置
    项的语义应当始终是"整个服务的在途上限", 不随部署形态漂移。

    向上取整并保底 1: 宁可略微超出也不要出现某个 worker 拿不到任何槽位 —— 那等于
    这个 worker 上的聊天全部卡死, 比稍微宽松的限流严重得多。
    """
    configured = int(total or 0)
    if configured <= 0:
        return 0
    workers = max(1, int(getattr(settings, "web_concurrency", 1) or 1))
    return max(1, -(-configured // workers))     # ceil division


@asynccontextmanager
async def _llm_slot(provider: str):
    """占一个 provider 的在途槽位.

    为什么需要: retry 会放大故障。provider 抖一下 → 所有在途调用重试 → 瞬时请求
    量翻几倍 → 真的触发 rate limit → 熔断器打开 → 全员降级。熔断器是事后止损,
    并发上限是事前不让它发生。提醒触发那条线早就用 Semaphore(8) 解决过同一个问题,
    聊天路径一直没有。

    后台调用要**先**拿后台配额再拿总配额, 于是: 总量 ≤ llm_max_concurrency,
    后台 ≤ llm_background_max_concurrency, 前台任何时候至少有两者之差个槽位。
    单一全局信号量做不到这点 —— 夜间任务可以把槽位占满, 白天聊天的人排在后面。

    槽位只在真正发请求时持有: retry 的 backoff sleep 不占槽, 否则一次重试链会把
    槽位按住十几秒, 限流反而成了瓶颈。
    """
    total = _per_worker_share(getattr(settings, "llm_max_concurrency", 0))
    if total <= 0:                       # 配 0 或负数 = 关闭限流
        yield
        return

    from app.services.llm.usage_tracker import current_scope

    is_background = current_scope() in _BACKGROUND_SCOPES
    if not is_background:
        async with _get_semaphore(_slots, provider, total):
            yield
        return

    # 下限 1: 配成 0 会让后台任务永远拿不到槽位, 夜间 cron 直接卡死 —— 那是比不
    # 限流更严重的故障, 不该由一个配错的数值造成。上限收到 total, 配得比总量还大
    # 等于没有前台保底。
    bg = max(1, min(total, _per_worker_share(
        getattr(settings, "llm_background_max_concurrency", 0),
    )))
    async with _get_semaphore(_bg_slots, provider, bg):
        async with _get_semaphore(_slots, provider, total):
            yield


async def _run_with_retry(
    factory: Callable[[], Awaitable[Any]],
    *,
    provider: str,
    profile: CallProfile,
    op: str,
) -> Any:
    if not settings.llm_resilience_enabled:
        # Kill switch: 只保留 timeout, 跳过 CB + retry。并发上限仍然生效 ——
        # 这个开关是为了排查时少一层重试/熔断的干扰, 而故障期恰恰最需要防止请求
        # 洪峰, 把限流一起关掉会让情况更糟。
        async with _llm_slot(provider):
            return await asyncio.wait_for(factory(), timeout=profile.timeout_s)

    breaker = _get_breaker(provider)
    if not breaker.try_acquire():
        _record_runtime_event(result="circuit_open")
        raise LLMCircuitOpenError(f"circuit open for {provider} on {op}")

    last_exc: Exception | None = None
    for attempt in range(profile.max_retries + 1):
        started = time.monotonic()
        try:
            # 槽位只包住这一次请求, 不包外层的 backoff sleep。
            async with _llm_slot(provider):
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
        if fallback_factory is None or not profile.allow_ollama_fallback:
            raise
        _record_runtime_event(result="fallback")
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
    primary_model_name: str = "",
    fallback_model_name: str = "",
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
            model_name=primary_model_name,
        ):
            got_first_chunk = True
            yield token
        return
    except LLMFailedError as primary_exc:
        # 已经推过 token 或无 fallback → 直接让异常穿出, 调用方自己兜底
        if got_first_chunk or fallback_factory is None:
            raise
        _record_runtime_event(result="fallback")
        logger.warning(
            f"[LLM-FALLBACK] op={op} primary={primary_provider} stream failed "
            f"pre-first-chunk: {type(primary_exc).__name__}: {primary_exc}; trying ollama",
        )

    # Fallback to Ollama
    try:
        async for token in _stream_provider(
            fallback_factory, provider="ollama", profile=profile, op=f"{op}:fallback",
            model_name=fallback_model_name,
        ):
            yield token
    except LLMFailedError as fallback_exc:
        raise LLMFailedError(
            f"{op}: primary={primary_provider} and Ollama fallback both failed: "
            f"{fallback_exc}"
        ) from fallback_exc


async def collect_stream(
    primary_factory: Callable[[], AsyncIterator[Any]],
    *,
    primary_provider: str,
    profile: CallProfile,
    op: str,
    fallback_factory: Callable[[], AsyncIterator[Any]] | None = None,
    primary_model_name: str = "",
    fallback_model_name: str = "",
) -> str:
    """累积 astream_with_resilience 的全部 token 成单一字符串.

    供 unary 接口 (e.g. invoke_json on background profile) 复用流式管线时
    使用; 直接的流式消费方仍应迭代 astream_with_resilience.
    LLMFailedError 透传到调用方.
    """
    chunks: list[str] = []
    async for token in astream_with_resilience(
        primary_factory,
        primary_provider=primary_provider,
        profile=profile,
        op=op,
        fallback_factory=fallback_factory,
        primary_model_name=primary_model_name,
        fallback_model_name=fallback_model_name,
    ):
        if token:
            chunks.append(token)
    return "".join(chunks)


def _capture_chunk_usage(chunk: Any, last_usage: dict) -> None:
    """langchain stream chunks 末尾 chunk 通常带 usage_metadata; 在迭代时
    每见到一次就覆盖 last_usage, 流结束后取最后一个非空值. 调用方把这个
    dict 在 finally 里 record 给 usage_tracker."""
    meta = getattr(chunk, "usage_metadata", None)
    if isinstance(meta, dict) and (meta.get("input_tokens") or meta.get("output_tokens")):
        last_usage["input_tokens"] = int(meta.get("input_tokens", 0) or 0)
        last_usage["output_tokens"] = int(meta.get("output_tokens", 0) or 0)
        details = meta.get("input_token_details")
        if isinstance(details, dict):
            cached = details.get("cache_read") or details.get("cached_tokens") or 0
            last_usage["cached_input_tokens"] = int(cached or 0)


def _flush_stream_usage(last_usage: dict, model_name: str) -> None:
    """流结束/异常时记一次 usage_tracker. 没拿到 usage 或没 model_name 则跳过."""
    if not last_usage or not model_name:
        return
    from app.services.llm import usage_tracker
    usage_tracker.record(
        model_name,
        last_usage.get("input_tokens", 0),
        last_usage.get("output_tokens", 0),
        cached_input_tokens=last_usage.get("cached_input_tokens", 0),
    )


async def _stream_provider(
    factory: Callable[[], AsyncIterator[Any]],
    *,
    provider: str,
    profile: CallProfile,
    op: str,
    model_name: str = "",
) -> AsyncIterator[str]:
    """在单一 provider 上跑 stream; 管理 first_chunk_timeout 和总 timeout + CB."""
    last_usage: dict = {}

    if not settings.llm_resilience_enabled:
        started = time.monotonic()
        try:
            async for chunk in factory():
                # 仅尊重总 timeout (killswitch 场景保留基本防卡)
                if time.monotonic() - started > profile.timeout_s:
                    raise LLMFailedError(f"{op} on {provider}: exceeded overall timeout (killswitch mode)")
                _capture_chunk_usage(chunk, last_usage)
                yield _chunk_text(chunk)
        finally:
            _flush_stream_usage(last_usage, model_name)
        return

    breaker = _get_breaker(provider)
    if not breaker.try_acquire():
        _record_runtime_event(result="circuit_open")
        raise LLMCircuitOpenError(f"circuit open for {provider} on {op}")

    started = time.monotonic()
    stream = factory()
    aiter = stream.__aiter__()

    # First chunk with tight timeout (timeout / empty stream / upstream error 统一
    # 视为 primary 失败, 触发 fallback. 捕获后必须 aclose 释放底层 SSE/HTTP 连接,
    # 不能依赖 GC — Dashscope 抽风时 CB 会 open, 多次 timeout 堆积 fd 可能打爆.)
    async def _safe_close() -> None:
        """关底层流, 释放 SSE/HTTP fd. aclose 失败不该覆盖原始异常."""
        _close = getattr(stream, "aclose", None)
        if _close is None:
            return
        try:
            await _close()
        except Exception:
            pass

    try:
        # 槽位只覆盖建连到首 chunk 这一段, 不覆盖整个流。两个理由:
        #
        # 其一, 要防的是"请求发起"的洪峰 (重试风暴、突发到达), 那正好发生在这一
        # 段 —— factory() 只是造生成器, 真正发请求是第一次 __anext__。
        #
        # 其二, 按住整个流会让上限失去意义: 流可以跑 30-90 秒, 64 个槽位就只剩每
        # 秒两条流的吞吐。而且异步生成器被消费方中途丢弃时, finally 要等 GC 才跑,
        # 槽位泄漏是永久的容量损失 —— 比没有限流更糟。这一段的持有时长有
        # first_chunk_timeout_s 兜底, 泄漏窗口是有界的。
        async with _llm_slot(provider):
            first = await asyncio.wait_for(
                aiter.__anext__(), timeout=profile.first_chunk_timeout_s,
            )
    except Exception as e:
        breaker.record_failure()
        _log_attempt(provider=provider, op=op, result="first_chunk_fail",
                     started=started, exc=e)
        await _safe_close()
        raise LLMFailedError(f"stream no first chunk on {provider}: {e}") from e

    _capture_chunk_usage(first, last_usage)
    yield _chunk_text(first)
    deadline = started + profile.timeout_s
    # idle 超时: chunk 间最大允许停顿. None → 沿用 first_chunk_timeout_s
    # (与改动前等价 — 老 chat_stream profile 行为不变).
    idle = profile.idle_timeout_s or profile.first_chunk_timeout_s

    # Remaining chunks with idle + overall budget. mid-stream 失败必须 aclose
    # 释放底层 SSE 连接 (dashscope 抽风时 CB open + fd 堆积可能打爆).
    try:
        while True:
            try:
                chunk = await asyncio.wait_for(aiter.__anext__(), timeout=idle)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                breaker.record_failure()
                _log_attempt(provider=provider, op=op, result="idle_timeout",
                             started=started)
                await _safe_close()
                raise LLMFailedError(
                    f"stream idle > {idle}s on {provider} (no chunk between yields)"
                )
            if time.monotonic() > deadline:
                breaker.record_failure()
                _log_attempt(provider=provider, op=op, result="mid_timeout",
                             started=started)
                await _safe_close()
                raise LLMFailedError(f"stream exceeded overall {profile.timeout_s}s on {provider}")
            _capture_chunk_usage(chunk, last_usage)
            yield _chunk_text(chunk)
        breaker.record_success()
        _log_attempt(provider=provider, op=op, result="ok", started=started)
    except LLMFailedError:
        raise
    except Exception as e:
        breaker.record_failure()
        _log_attempt(provider=provider, op=op, result="mid_error",
                     started=started, exc=e)
        await _safe_close()
        raise LLMFailedError(f"stream mid error on {provider}: {e}") from e
    finally:
        _flush_stream_usage(last_usage, model_name)


def _chunk_text(chunk: Any) -> str:
    """LangChain chunk 有 .content 字段 (AIMessageChunk), 其他 provider 也可能直接是 str."""
    if hasattr(chunk, "content"):
        return str(chunk.content)
    return str(chunk)
