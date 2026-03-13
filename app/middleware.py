"""Monitoring & observability middleware.

Structured logging, request timing, and LangSmith tracing configuration.
"""

import logging
import os
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.config import settings

logger = logging.getLogger(__name__)


def configure_logging():
    """Configure structured logging for the application."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Reduce noise from libraries
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)


def configure_langsmith():
    """Configure LangSmith tracing if enabled."""
    if settings.langsmith_tracing and settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = "ai-companion"
        logger.info("LangSmith tracing enabled (project: ai-companion)")
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """Log request timing and status for all API calls."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Skip health checks from verbose logging
        if request.url.path != "/health":
            logger.info(
                f"{request.method} {request.url.path} "
                f"-> {response.status_code} ({elapsed_ms:.0f}ms)"
            )

        response.headers["X-Response-Time"] = f"{elapsed_ms:.0f}ms"
        return response
