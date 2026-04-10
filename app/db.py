import asyncio
import logging

import httpx
from prisma import Prisma

logger = logging.getLogger(__name__)

# Prisma Python client 通过 httpx 连接本地 Prisma engine 子进程。
# 默认超时很短（~10s），当 LLM 生成长时间占用 event loop 时，
# 其他并发的 DB 查询可能被饿死超时。延长到 120s 避免误杀。
db = Prisma(http={"timeout": httpx.Timeout(120.0)})

# 启动时连接重试参数
_CONNECT_MAX_ATTEMPTS = 5
_CONNECT_RETRY_DELAY = 2.0  # seconds


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

    for attempt in range(1, _CONNECT_MAX_ATTEMPTS + 1):
        try:
            if not db.is_connected():
                await db.connect()
            await _ping()
            if attempt > 1:
                logger.info(f"Database connected after {attempt} attempts")
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
                await asyncio.sleep(_CONNECT_RETRY_DELAY)

    raise RuntimeError(
        f"Failed to connect to database after {_CONNECT_MAX_ATTEMPTS} attempts. "
        f"Last error: {last_error}"
    )


async def disconnect_db():
    if db.is_connected():
        await db.disconnect()
