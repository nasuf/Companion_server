"""Monitoring & observability middleware.

Structured logging, request timing, and LangSmith tracing configuration.
"""

import logging
import os
import time
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.config import settings
from app.observability import bind_context, setup_axiom
from app.observability.events import EVT_HTTP_REQUEST
from app.observability.log_filter import ContextInjectionFilter
from app.observability.log_formatter import HumanContextFormatter

logger = logging.getLogger(__name__)


def configure_logging():
    """Configure structured logging — console + Axiom + ContextVar 注入."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level, logging.INFO))

    # 防重入: 单元测试 / dev reload 可能多次调用. 清掉既有 handler/filter
    # 避免日志重复输出 + Filter 重复注入.
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for f in list(root.filters):
        root.removeFilter(f)

    # Console handler — 人类可读 + 内联高频 ID
    # Filter 必须挂 handler 而非 logger — Python logging 的 propagate 只把
    # record 传给祖先 logger 的 handler, 不再过祖先 logger 的 Filter 链.
    # 挂 handler 才能拦截所有从 child logger 冒泡上来的 record.
    ctx_filter = ContextInjectionFilter()
    console = logging.StreamHandler()
    console.setFormatter(
        HumanContextFormatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    console.addFilter(ctx_filter)
    root.addHandler(console)

    # Axiom handler (env 没配/lib 没装时跳过, 不影响 console).
    # setup_axiom() 内会再 attach 同样的 filter 到 axiom handler — 见 axiom_setup.py
    setup_axiom()

    # Reduce noise from chatty libraries
    for noisy in ("httpcore", "httpx", "apscheduler"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def configure_langsmith():
    """Configure LangSmith env tracing according to the trace backend.

    langchain reads LANGSMITH_TRACING / LANGCHAIN_TRACING_V2 straight from the
    process env on every call. Only the "langsmith" backend may leave them on;
    local/off must force them off, otherwise every LLM call still uploads runs
    to LangSmith and burns the monthly quota the local backend exists to avoid.
    """
    if (
        settings.trace_backend == "langsmith"
        and settings.langsmith_tracing
        and settings.langsmith_api_key
    ):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = "ai-companion"
        logger.info("LangSmith tracing enabled (project: ai-companion)")
        return
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    if settings.trace_backend == "local":
        logger.info("Trace backend: local (trace_runs collection); LangSmith env tracing off")


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """Log request timing + 绑定 request_id 到 ContextVar 让链路日志可追溯."""

    async def dispatch(self, request: Request, call_next):
        rid = uuid4().hex[:12]
        with bind_context(request_id=rid):
            start = time.perf_counter()
            response = await call_next(request)
            elapsed_ms = (time.perf_counter() - start) * 1000

            # /health 太频繁 (LB probe), 跳过避免污染 — 失败状态码仍打 (排错信号)
            if request.url.path != "/health" or response.status_code >= 400:
                logger.info(
                    f"{request.method} {request.url.path} "
                    f"-> {response.status_code} ({elapsed_ms:.0f}ms)",
                    extra={
                        "event": EVT_HTTP_REQUEST,
                        "method": request.method,
                        "path": request.url.path,
                        "status": response.status_code,
                        "elapsed_ms": round(elapsed_ms, 1),
                    },
                )

            response.headers["X-Response-Time"] = f"{elapsed_ms:.0f}ms"
            response.headers["X-Request-ID"] = rid
            return response
