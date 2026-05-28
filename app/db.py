import asyncio
import logging
import os
import time
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
from dotenv import load_dotenv
from prisma import Prisma

logger = logging.getLogger(__name__)

load_dotenv()

# Keep Prisma's own pool conservative and throttle app-side bursts so one API
# process cannot exhaust all database sessions. The hard cap is intentional:
# local reloads, stale Prisma query-engine children, and deploy restarts can
# briefly overlap, so one process should not be allowed to reserve the whole
# database connection budget.
_DB_CONNECTION_LIMIT_DEFAULT = 3
_DB_CONNECTION_LIMIT_MAX_DEFAULT = 5
_DB_POOL_TIMEOUT_DEFAULT = 30
_DB_CONNECT_TIMEOUT_DEFAULT = 30


def _parse_positive_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


_DB_QUERY_MAX_RETRIES = _parse_positive_int(os.getenv("DB_QUERY_MAX_RETRIES")) or 2


def _env_flag_enabled(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _runtime_connection_limit_cap() -> int:
    return (
        _parse_positive_int(os.getenv("DB_CONNECTION_LIMIT_MAX"))
        or _DB_CONNECTION_LIMIT_MAX_DEFAULT
    )


def _safe_runtime_connection_limit(requested: int | None = None) -> int:
    requested_limit = requested or _DB_CONNECTION_LIMIT_DEFAULT
    if _env_flag_enabled("DB_ALLOW_UNSAFE_CONNECTION_LIMIT"):
        return requested_limit
    return min(requested_limit, _runtime_connection_limit_cap())


def _with_safe_database_params(
    url: str,
    *,
    default_connection_limit: int = _DB_CONNECTION_LIMIT_DEFAULT,
    default_pool_timeout: int = _DB_POOL_TIMEOUT_DEFAULT,
    default_connect_timeout: int = _DB_CONNECT_TIMEOUT_DEFAULT,
    forced_connection_limit: int | None = None,
) -> str:
    """Return DATABASE_URL with conservative runtime pool settings.

    Prisma reads connection parameters from DATABASE_URL when its query engine
    starts. If local/prod env accidentally sets `connection_limit` above the
    configured database connection budget, a single process can consume the
    whole pool and cause connection exhaustion. We cap only the runtime URL;
    migration URLs are handled separately by Prisma commands.
    """
    if not url.startswith(("postgres://", "postgresql://")):
        return url

    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))

    existing_limit = _parse_positive_int(query.get("connection_limit"))
    requested_limit = forced_connection_limit or default_connection_limit
    safe_limit = _safe_runtime_connection_limit(requested_limit)
    if existing_limit is None or existing_limit > safe_limit:
        query["connection_limit"] = str(safe_limit)

    query.setdefault("pool_timeout", str(default_pool_timeout))
    query.setdefault("connect_timeout", str(default_connect_timeout))

    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )


def _install_safe_database_url() -> None:
    raw_url = os.getenv("DATABASE_URL")
    if not raw_url:
        return

    forced_limit = _parse_positive_int(os.getenv("DB_CONNECTION_LIMIT"))
    effective_limit = _safe_runtime_connection_limit(forced_limit)
    if (
        forced_limit is not None
        and forced_limit > effective_limit
        and not _env_flag_enabled("DB_ALLOW_UNSAFE_CONNECTION_LIMIT")
    ):
        logger.warning(
            "DB_CONNECTION_LIMIT=%s exceeds safe cap; using %s instead "
            "(set DB_CONNECTION_LIMIT_MAX or DB_ALLOW_UNSAFE_CONNECTION_LIMIT=1 "
            "only if the database pool size has been increased)",
            forced_limit,
            effective_limit,
        )
    safe_url = _with_safe_database_params(raw_url, forced_connection_limit=forced_limit)
    if safe_url == raw_url:
        return

    os.environ["DATABASE_URL"] = safe_url
    logger.info(
        "Adjusted DATABASE_URL runtime pool params "
        "(connection_limit<=%s, pool_timeout=%s, connect_timeout=%s)",
        effective_limit,
        _DB_POOL_TIMEOUT_DEFAULT,
        _DB_CONNECT_TIMEOUT_DEFAULT,
    )


def _is_db_pool_exhaustion_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "emaxconnsession" in text
        or "max clients reached" in text
        or "too many clients already" in text
        or "remaining connection slots are reserved" in text
    )


def _connection_limit_from_database_url(
    url: str | None,
    *,
    default: int = _DB_CONNECTION_LIMIT_DEFAULT,
) -> int:
    if not url:
        return default
    query = dict(parse_qsl(urlsplit(url).query, keep_blank_values=True))
    return _parse_positive_int(query.get("connection_limit")) or default


_install_safe_database_url()

_RUNTIME_CONNECTION_LIMIT = _connection_limit_from_database_url(os.getenv("DATABASE_URL"))
_CONFIGURED_MAX_CONCURRENT_QUERIES = (
    _parse_positive_int(os.getenv("DB_MAX_CONCURRENT_QUERIES"))
    or _RUNTIME_CONNECTION_LIMIT
)
_DB_MAX_CONCURRENT_QUERIES = min(
    _CONFIGURED_MAX_CONCURRENT_QUERIES, _RUNTIME_CONNECTION_LIMIT
)
_query_semaphore = asyncio.Semaphore(_DB_MAX_CONCURRENT_QUERIES)
logger.info(
    "DB runtime pool guard enabled (pid=%s, connection_limit=%s, "
    "max_concurrent_queries=%s, retries=%s)",
    os.getpid(),
    _RUNTIME_CONNECTION_LIMIT,
    _DB_MAX_CONCURRENT_QUERIES,
    _DB_QUERY_MAX_RETRIES,
)


class ThrottledPrisma(Prisma):
    async def _execute(self, *args: Any, **kwargs: Any) -> Any:
        async with _query_semaphore:
            for attempt in range(_DB_QUERY_MAX_RETRIES + 1):
                try:
                    return await super()._execute(*args, **kwargs)
                except Exception as exc:
                    if (
                        attempt >= _DB_QUERY_MAX_RETRIES
                        or not _is_db_pool_exhaustion_error(exc)
                    ):
                        raise
                    delay = min(0.2 * (2**attempt), 1.0)
                    logger.warning(
                        "DB pool exhausted, retrying query in %.1fs "
                        "(attempt %s/%s): %s",
                        delay,
                        attempt + 1,
                        _DB_QUERY_MAX_RETRIES,
                        str(exc)[:200],
                    )
                    await asyncio.sleep(delay)

# Prisma Python client 通过 httpx 连接本地 Prisma engine 子进程。
# 默认超时很短（~10s），当 LLM 生成长时间占用 event loop 时，
# 其他并发的 DB 查询可能被饿死超时。延长到 120s 避免误杀。
#
# 这里必须禁用 trust_env: Prisma client 访问的是本机 query-engine
# (http://127.0.0.1:<port>/status)。如果系统或 shell 设置了 HTTP_PROXY /
# ALL_PROXY，httpx 会把这个本地请求转发给代理，导致启动时 DB connect 卡死。
db = ThrottledPrisma(
    http={
        "timeout": httpx.Timeout(120.0),
        "trust_env": False,
    }
)

# ── 启动时连接重试参数 (可通过环境变量覆盖) ──
# 数据库启动或网络偶有抖动, 需要比较宽松的重试:
#   - 每次尝试给 engine 30s 完成 TLS 握手 + 首次查询 (默认 10s 太短)
#   - 指数退避 2/4/8/16/30s (cap), 共 8 次, 最差 ~4 min 后放弃
#   - 放弃阈值大于典型数据库冷启动 / 网络抖动窗口 (30-90s)
_CONNECT_MAX_ATTEMPTS = int(os.getenv("DB_CONNECT_MAX_ATTEMPTS", "8"))
_CONNECT_TIMEOUT_S = int(os.getenv("DB_CONNECT_TIMEOUT_S", "30"))
_CONNECT_BACKOFF_BASE = float(os.getenv("DB_CONNECT_BACKOFF_BASE", "2.0"))
_CONNECT_BACKOFF_CAP = float(os.getenv("DB_CONNECT_BACKOFF_CAP", "30.0"))


def _backoff_seconds(attempt: int) -> float:
    """指数退避: 2, 4, 8, 16 → cap. attempt 从 1 开始。"""
    return min(_CONNECT_BACKOFF_BASE**attempt, _CONNECT_BACKOFF_CAP)


async def _ping() -> None:
    """主动验证连接确实可用。数据库连接可能在 db.connect() 后立即关闭，
    必须发一个真实查询才能确认。"""
    await db.execute_raw("SELECT 1")


async def connect_db():
    """带重试的数据库连接。

    Postgres 或连接池偶尔会在 connect 后立刻关闭连接（idle timeout / network blip），
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

    Postgres or its connection pool may close idle connections during long-running tasks
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
