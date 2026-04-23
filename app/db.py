import asyncio
import logging
import os
import time

import httpx
from prisma import Prisma

logger = logging.getLogger(__name__)

# Prisma Python client 通过 httpx 连接本地 Prisma engine 子进程。
# 默认超时很短（~10s），当 LLM 生成长时间占用 event loop 时，
# 其他并发的 DB 查询可能被饿死超时。延长到 120s 避免误杀。
db = Prisma(http={"timeout": httpx.Timeout(120.0)})

# ── 启动时连接重试参数 (可通过环境变量覆盖) ──
# Supabase pooler 本地启动时偶有抖动, 需要比较宽松的重试:
#   - 每次尝试给 engine 30s 完成 TLS 握手 + 首次查询 (默认 10s 太短)
#   - 指数退避 2/4/8/16/30s (cap), 共 8 次, 最差 ~4 min 后放弃
#   - 放弃阈值大于典型 pooler 冷启动 (30-90s)
_CONNECT_MAX_ATTEMPTS = int(os.getenv("DB_CONNECT_MAX_ATTEMPTS", "8"))
_CONNECT_TIMEOUT_S = int(os.getenv("DB_CONNECT_TIMEOUT_S", "30"))
_CONNECT_BACKOFF_BASE = float(os.getenv("DB_CONNECT_BACKOFF_BASE", "2.0"))
_CONNECT_BACKOFF_CAP = float(os.getenv("DB_CONNECT_BACKOFF_CAP", "30.0"))


def _backoff_seconds(attempt: int) -> float:
    """指数退避: 2, 4, 8, 16 → cap. attempt 从 1 开始。"""
    return min(_CONNECT_BACKOFF_BASE**attempt, _CONNECT_BACKOFF_CAP)


async def _ping() -> None:
    """主动验证连接确实可用。Supabase pooler 可能在 db.connect() 后立即关闭连接，
    必须发一个真实查询才能确认。"""
    await db.execute_raw("SELECT 1")


async def connect_db():
    """带重试的数据库连接。

    Supabase session pooler 偶尔会在 connect 后立刻关闭连接（idle timeout / network blip），
    导致首次查询触发 "Error { kind: Closed }"。这里通过 connect → ping 校验 → 失败重连
    的循环来保证启动时拿到的是真正可用的连接。
    """
    last_error: Exception | None = None
    t0 = time.monotonic()

    for attempt in range(1, _CONNECT_MAX_ATTEMPTS + 1):
        try:
            if not db.is_connected():
                logger.info(
                    f"  DB connecting... (attempt {attempt}/{_CONNECT_MAX_ATTEMPTS}, "
                    f"timeout={_CONNECT_TIMEOUT_S}s)"
                )
                await db.connect(timeout=_CONNECT_TIMEOUT_S)
            await _ping()
            elapsed = time.monotonic() - t0
            logger.info(f"  DB connected (attempt {attempt}, {elapsed:.1f}s total)")
            return
        except Exception as e:
            last_error = e
            logger.warning(
                f"DB connect/ping failed (attempt {attempt}/{_CONNECT_MAX_ATTEMPTS}): "
                f"{type(e).__name__}: {str(e)[:200]}"
            )
            # 强制 disconnect 然后重连，避免使用半死状态的连接
            try:
                if db.is_connected():
                    await db.disconnect()
            except Exception:
                pass
            if attempt < _CONNECT_MAX_ATTEMPTS:
                delay = _backoff_seconds(attempt)
                logger.info(f"  retrying in {delay:.1f}s (exponential backoff)...")
                await asyncio.sleep(delay)

    total = time.monotonic() - t0
    raise RuntimeError(
        f"Failed to connect to database after {_CONNECT_MAX_ATTEMPTS} attempts "
        f"({total:.1f}s total). Last error: {last_error}"
    )


async def ensure_connected() -> None:
    """Verify DB connection is alive; reconnect if stale.

    Supabase pooler may close idle connections during long-running tasks
    (e.g. batch embedding). Call this before write-heavy phases.
    """
    try:
        await _ping()
    except Exception:
        logger.warning("DB connection stale, reconnecting...")
        try:
            if db.is_connected():
                await db.disconnect()
        except Exception:
            pass
        await db.connect(timeout=_CONNECT_TIMEOUT_S)
        await _ping()
        logger.info("DB reconnected")


async def disconnect_db():
    if db.is_connected():
        await db.disconnect()
