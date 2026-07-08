import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db import connect_db, disconnect_db
from app.redis_client import get_redis, close_redis, mark_redis_healthy
from app.middleware import configure_logging, configure_langsmith, RequestTimingMiddleware
from app.services.prompting.store import ensure_prompt_templates
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
    scheduler_started = False
    ws_manager = None

    try:
        settings.validate_security_config()

        # Phase 1: Connect to all services in parallel
        # DB 是硬依赖, 启动失败直接 crash; Redis 软依赖, 失败降级到 readonly mode
        # (GET 端点仍可用, 写端点 require_redis 返 503, scheduler 每 30s 重检自愈).
        await _timed("Database", connect_db())
        try:
            # get_redis 是懒初始化 (只建 client 不建连接) — 必须 ping 才真正
            # 建连+验证可达: 否则 Redis 已挂也会被误标 healthy (直到 30s 健康
            # 检查才纠正), 且首个用户请求要付连接建立成本 (预热).
            redis_client = await get_redis()
            await _timed("Redis", redis_client.ping())
            mark_redis_healthy(True)
        except Exception as e:
            logger.error(f"Redis connect failed ({e!r}); starting in readonly mode")
            mark_redis_healthy(False)

        # Phase 2: Seeding
        # Database schema changes are managed exclusively by Prisma migrations.
        await asyncio.gather(
            _timed("Prompt templates", ensure_prompt_templates()),
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

        # Phase 2c: Runtime config preload (system + agent overrides). 失败时全部
        # fallback 到 env 默认, 不阻断启动.
        try:
            from app.services.runtime_config import load_caches
            await _timed("Runtime config", load_caches())
        except Exception as e:
            logger.warning(
                f"Runtime config load failed ({e!r}); env defaults active."
            )

        try:
            from app.api.public.agents import warm_default_agent_avatars
            await _timed("Agent avatars", warm_default_agent_avatars())
        except Exception as e:
            logger.warning(f"Agent avatar warmup failed ({e!r})")

        # Phase 3: Scheduler + WS subscriber (跨进程 Pub/Sub)
        setup_scheduler()
        scheduler_started = True
        logger.info("  ✓ Scheduler")
        from app.services.runtime.ws_manager import manager as runtime_ws_manager
        ws_manager = runtime_ws_manager
        await ws_manager.start_subscriber()
        logger.info("  ✓ WS subscriber")

        total = (time.monotonic() - t_start) * 1000
        logger.info(f"Startup complete ({total:.0f}ms)")
        yield
    finally:
        if ws_manager is not None:
            try:
                await ws_manager.stop_subscriber()
            except Exception as e:
                logger.warning(f"WS subscriber shutdown failed: {e!r}")
        if scheduler_started:
            try:
                shutdown_scheduler()
            except Exception as e:
                logger.warning(f"Scheduler shutdown failed: {e!r}")
        try:
            await disconnect_db()
        except Exception as e:
            logger.warning(f"DB disconnect failed: {e!r}")
        try:
            await close_redis()
        except Exception as e:
            logger.warning(f"Redis disconnect failed: {e!r}")


app = FastAPI(title="AI Companion", version="0.1.0", lifespan=lifespan)

app.add_middleware(RequestTimingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins(),
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
from app.api.public.chat_media import router as chat_media_router
from app.api.public.chat_links import router as chat_links_router
from app.api.public.daily_share import router as daily_share_router
from app.api.public.memories import router as memories_router
from app.api.public.reminders import router as reminders_router
from app.api.public.emotions import router as emotions_router
from app.api.public.intimacy import router as intimacy_router
from app.api.public.boundary import router as boundary_router
from app.api.public.stickers import router as stickers_router
from app.api.public.store import router as store_router
from app.api.public.time_capsules import router as time_capsules_router
from app.api.public.last_wills import router as last_wills_router
from app.api.public.games import router as games_router
from app.api.public.music import router as music_router
from app.api.public.achievements import router as achievements_router
from app.api.public.wallet import router as wallet_router
from app.api.public.notifications import router as notifications_router
from app.api.public.offline import router as offline_router
from app.api.realtime.ws import router as ws_router
from app.api.admin.prompts import router as admin_prompts_router
from app.api.admin.holidays import router as admin_holidays_router
from app.api.public.auth import router as auth_router
from app.api.public.meal import router as meal_router
from app.api.admin.users import router as admin_users_router
from app.api.admin.career import router as admin_career_router
from app.api.admin.agents import router as admin_agents_router
from app.api.admin.agent_templates import router as admin_agent_templates_router
from app.api.public.traces import router as traces_router
from app.api.admin.bug_reports import router as admin_bug_reports_router
from app.api.admin.stats import router as admin_stats_router
from app.api.admin.runtime_config import router as admin_runtime_config_router
from app.api.admin.model_registry import router as admin_model_registry_router
from app.api.admin.memory_repairs import router as admin_memory_repairs_router
from app.api.admin.runtime_jobs import router as admin_runtime_jobs_router
from app.api.admin.meal import router as admin_meal_router

app.include_router(health_router)
app.include_router(users_router)
app.include_router(agents_router)
app.include_router(conversations_router)
app.include_router(chat_media_router)
app.include_router(chat_links_router)
app.include_router(chat_router)
app.include_router(daily_share_router)
app.include_router(memories_router)
app.include_router(reminders_router)
app.include_router(emotions_router)
app.include_router(intimacy_router)
app.include_router(boundary_router)
app.include_router(stickers_router)
app.include_router(time_capsules_router)
app.include_router(last_wills_router)
app.include_router(games_router)
app.include_router(music_router)
app.include_router(achievements_router)
app.include_router(wallet_router)
app.include_router(store_router)
app.include_router(notifications_router)
app.include_router(offline_router)
app.include_router(ws_router)
app.include_router(admin_prompts_router)
app.include_router(admin_holidays_router)
app.include_router(auth_router)
app.include_router(meal_router)
app.include_router(admin_users_router)
app.include_router(admin_career_router)
app.include_router(admin_agents_router)
app.include_router(admin_agent_templates_router)
app.include_router(traces_router)
app.include_router(admin_bug_reports_router)
app.include_router(admin_stats_router)
app.include_router(admin_runtime_config_router)
app.include_router(admin_model_registry_router)
app.include_router(admin_memory_repairs_router)
app.include_router(admin_runtime_jobs_router)
app.include_router(admin_meal_router)
