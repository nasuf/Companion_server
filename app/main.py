import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db import connect_db, disconnect_db
from app.redis_client import get_redis, close_redis
from app.middleware import configure_logging, configure_langsmith, RequestTimingMiddleware
from app.services.prompting.store import ensure_prompt_templates
from app.services.character import ensure_default_template
from app.services.career import ensure_default_careers
from app.services.schedule_domain.holiday_cache import reload as reload_holiday_cache
from jobs.scheduler import setup_scheduler, shutdown_scheduler

# Configure logging and tracing before anything else
configure_logging()
configure_langsmith()

logger = logging.getLogger(__name__)


async def _timed(name: str, coro):
    """Run a coroutine and log its execution time."""
    t0 = time.monotonic()
    await coro
    elapsed = (time.monotonic() - t0) * 1000
    logger.info(f"  ✓ {name} ({elapsed:.0f}ms)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    t_start = time.monotonic()
    logger.info("Starting up...")

    # Phase 1: Connect to all services in parallel
    await _timed("Database", connect_db())
    await _timed("Redis", get_redis())

    # Phase 2: Schema + seeding
    await asyncio.gather(
        _timed("Prompt templates", ensure_prompt_templates()),
        _timed("Character template", ensure_default_template()),
        _timed("Career templates", ensure_default_careers()),
    )

    # Phase 2b: Holiday cache preload. Runs sequentially (not in the gather
    # above) to avoid exhausting the Prisma pool when the other seed tasks
    # hold connections for several seconds. Failure here must not crash
    # startup — cache stays empty and `is_holiday()` falls back to lunardate.
    try:
        await _timed("Holiday cache", reload_holiday_cache())
    except Exception as e:
        logger.warning(
            f"Holiday cache preload failed ({e!r}); lunardate fallback active."
        )

    # Phase 3: Scheduler
    setup_scheduler()
    logger.info(f"  ✓ Scheduler")

    total = (time.monotonic() - t_start) * 1000
    logger.info(f"Startup complete ({total:.0f}ms)")
    yield
    # Shutdown
    shutdown_scheduler()
    await disconnect_db()
    await close_redis()


app = FastAPI(title="AI Companion", version="0.1.0", lifespan=lifespan)

app.add_middleware(RequestTimingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
from app.api.public.health import router as health_router
from app.api.public.users import router as users_router
from app.api.public.agents import router as agents_router
from app.api.public.conversations import router as conversations_router
from app.api.public.chat import router as chat_router
from app.api.public.memories import router as memories_router
from app.api.public.emotions import router as emotions_router
from app.api.public.intimacy import router as intimacy_router
from app.api.public.boundary import router as boundary_router
from app.api.public.stickers import router as stickers_router
from app.api.realtime.ws import router as ws_router
from app.api.admin.prompts import router as admin_prompts_router
from app.api.admin.holidays import router as admin_holidays_router
from app.api.public.auth import router as auth_router
from app.api.admin.users import router as admin_users_router
from app.api.admin.character import router as admin_character_router
from app.api.admin.career import router as admin_career_router
from app.api.admin.agents import router as admin_agents_router
from app.api.public.traces import router as traces_router

app.include_router(health_router)
app.include_router(users_router)
app.include_router(agents_router)
app.include_router(conversations_router)
app.include_router(chat_router)
app.include_router(memories_router)
app.include_router(emotions_router)
app.include_router(intimacy_router)
app.include_router(boundary_router)
app.include_router(stickers_router)
app.include_router(ws_router)
app.include_router(admin_prompts_router)
app.include_router(admin_holidays_router)
app.include_router(auth_router)
app.include_router(admin_users_router)
app.include_router(admin_character_router)
app.include_router(admin_career_router)
app.include_router(admin_agents_router)
app.include_router(traces_router)
