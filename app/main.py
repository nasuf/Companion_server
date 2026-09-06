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
from app.services.runtime.distributed_lock import (
    DistributedLockNotAcquired,
    DistributedLockUnavailable,
    distributed_lock,
)
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
def _warn_if_embedding_model_uncalibrated() -> None:
    """EMBEDDING_MODEL 与代码里标定阈值所针对的模型不一致时大声告警.

    这两处会分开漂移: 模型来自环境变量 (部署时由 CI 变量写进 .env), 而十个相似度
    阈值写死在代码里. 对不上不会报错, 只会让检索悄悄失准 —— 阈值配错一端就是
    噪声灌进 prompt, 配错另一端就是整轮失忆, 两种都不会出现在日志里.
    """
    from app.services.memory.config import CALIBRATED_EMBEDDING_MODEL

    actual = settings.embedding_model
    if actual != CALIBRATED_EMBEDDING_MODEL:
        logger.error(
            "EMBEDDING MODEL MISMATCH: running %r but similarity thresholds are "
            "calibrated for %r. Retrieval quality is silently degraded until the "
            "two agree — either set EMBEDDING_MODEL to the calibrated model or "
            "re-derive the thresholds (scripts/calibrate_embedding_thresholds.py) "
            "and re-embed the corpus.",
            actual, CALIBRATED_EMBEDDING_MODEL,
        )


async def lifespan(app: FastAPI):
    t_start = time.monotonic()
    logger.info("Starting up...")
    scheduler_started = False
    ws_manager = None

    try:
        settings.validate_security_config()
        _warn_if_embedding_model_uncalibrated()

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
        #
        # 必须串行化: 两个 seeder 都是"先查缺失再创建", 多 worker 同时启动会一起
        # 读到"缺", 一起创建。career_templates.title 没有唯一约束, 结果是每个
        # worker 各建一份重复职业; prompt_templates.key 有唯一约束, 结果是输的
        # worker 直接抛异常起不来。
        #
        # 拿不到锁就跳过 (fail-closed): 说明别的 worker 正在做同一件事, 重复做既
        # 无必要也不安全。Redis 不可用时 (fail_open=非生产) 本地仍会执行 —— 开发
        # 环境只有一个进程, 不存在竞争。
        try:
            async with distributed_lock(
                "startup:seed", ttl_s=180, fail_open=not settings.is_production(),
            ):
                await asyncio.gather(
                    _timed("Prompt templates", ensure_prompt_templates()),
                    _timed("Career templates", ensure_default_careers()),
                )
        except DistributedLockNotAcquired:
            logger.info("  ↷ Seeding skipped: another worker holds the seed lock")
        except DistributedLockUnavailable as e:
            logger.warning(f"  ↷ Seeding skipped: seed lock unavailable ({e})")

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

        from app.services.agent_avatars import validate_avatar_assets

        validate_avatar_assets()
        logger.info("  ✓ Agent avatar assets")

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
    expose_headers=[
        "X-TTS-Duration-Milliseconds",
        "X-TTS-Billable-Characters",
        "X-TTS-Cost-CNY",
    ],
)

# Register routers
from app.api.public.health import router as health_router
from app.api.public.users import router as users_router
from app.api.public.agents import router as agents_router
from app.api.public.conversations import router as conversations_router
from app.api.public.chat import router as chat_router
from app.api.public.chat_media import router as chat_media_router
from app.api.public.chat_links import router as chat_links_router
from app.api.public.speech import router as speech_router
from app.api.public.daily_share import router as daily_share_router
from app.api.public.memories import router as memories_router
from app.api.public.reminders import router as reminders_router
from app.api.public.emotions import router as emotions_router
from app.api.public.intimacy import router as intimacy_router
from app.api.public.boundary import router as boundary_router
from app.api.public.stickers import router as stickers_router
from app.api.public.store import router as store_router
from app.api.public.vip import router as vip_router
from app.api.public.time_capsules import router as time_capsules_router
from app.api.public.last_wills import router as last_wills_router
from app.api.public.native_games import router as native_games_router
from app.api.public.game_points import router as game_points_router
from app.api.public.music import router as music_router
from app.api.public.achievements import router as achievements_router
from app.api.public.wallet import router as wallet_router
from app.api.public.offerings import gift_router, router as offerings_router
from app.api.public.notifications import router as notifications_router
from app.api.public.offline import router as offline_router
from app.api.public.iap import router as iap_router
from app.api.public.iap_membership import router as iap_membership_router
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
from app.api.admin.game_configs import router as admin_game_configs_router
from app.api.admin.game_points import router as admin_game_points_router
from app.api.admin.wallet import router as admin_wallet_router
from app.api.admin.payments import router as admin_payments_router
from app.api.admin.chat_quota import router as admin_chat_quota_router
from app.api.admin.offline_settings import router as admin_offline_settings_router
from app.api.admin.achievement_settings import router as admin_achievement_settings_router
from app.api.admin.tts import (
    public_router as tts_enrollment_public_router,
    router as admin_tts_router,
)

app.include_router(health_router)
app.include_router(users_router)
app.include_router(agents_router)
app.include_router(conversations_router)
app.include_router(chat_media_router)
app.include_router(chat_links_router)
app.include_router(speech_router)
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
app.include_router(native_games_router)
app.include_router(game_points_router)
app.include_router(music_router)
app.include_router(achievements_router)
app.include_router(wallet_router)
app.include_router(offerings_router)
app.include_router(gift_router)
app.include_router(store_router)
app.include_router(vip_router)
app.include_router(notifications_router)
app.include_router(offline_router)
app.include_router(iap_router)
app.include_router(iap_membership_router)
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
app.include_router(admin_game_configs_router)
app.include_router(admin_game_points_router)
app.include_router(admin_wallet_router)
app.include_router(admin_payments_router)
app.include_router(admin_chat_quota_router)
app.include_router(admin_offline_settings_router)
app.include_router(admin_achievement_settings_router)
app.include_router(admin_tts_router)
app.include_router(tts_enrollment_public_router)
