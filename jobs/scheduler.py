"""Job scheduler for periodic tasks.

Uses APScheduler for:
- Daily: L2 动态分数调整 (spec §1.5.2), reflection, 记忆衰减
- Weekly: weekly reflection, portrait update
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.config import settings
from app.observability.events import EVT_SCHEDULER_JOB
from app.redis_client import get_redis
from app.services.memory.lifecycle.hygiene import run_memory_hygiene
from app.services.memory.lifecycle.l2_dynamics import run_l2_adjustment
from app.services.reflection import run_weekly_reflection
from app.services.portrait import update_portrait_weekly
from app.services.schedule_domain.schedule import (
    generate_and_save_life_overview, generate_daily_schedule, get_cached_schedule,
    get_current_status, get_life_overview, review_daily_schedule,
)
from app.services.mbti import get_mbti
from app.services.interaction.boundary import recover_patience_hourly
from app.services.relationship.intimacy import compute_growth_intimacy, compute_topic_intimacy
from app.services.proactive.orchestrator import scan_proactive_states
from app.services.interaction.delayed_queue import (
    enqueue_delayed_message, scan_due_delayed_messages, merge_delayed_payloads,
    try_lock_conversation, unlock_conversation,
    mark_reply_inflight, clear_reply_inflight,
)
from app.services.interaction.user_turn_aggregation import scan_due_user_turns
from app.services.proactive.triggers import scan_triggers
from app.services.proactive.special_dates import scan_special_dates_today
from app.services.notifications.capsules import scan_ready_capsule_notifications
from app.services.notifications.dispatcher import dispatch_due_notifications
from app.services.music_status import scan_music_schedule_transitions
from app.services.last_will import scan_due_last_wills
from app.services.offline.scheduler import scan_offline_triggers
from app.services.offline.providers.ali1688_token import refresh_access_token
from app.services.runtime.distributed_lock import (
    DistributedLockNotAcquired,
    DistributedLockUnavailable,
    distributed_lock,
)
from app.services.runtime.job_queue import process_runtime_jobs

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()
_ACHIEVEMENT_ROLLUP_CHECKPOINT_KEY = "achievement:daily_rollup:last_success"


async def _run_distributed_job(
    job_name: str,
    ttl_s: int,
    fn: Callable[[], object],
) -> None:
    """Run one scheduler job once across all server instances.

    Development fails open so local Redis outages do not make cron debugging
    painful. Production fails closed: if Redis cannot provide the lock, the job
    skips instead of risking duplicate reminder/proactive sends.
    """
    try:
        async with distributed_lock(
            f"scheduler:{job_name}",
            ttl_s=ttl_s,
            fail_open=not settings.is_production(),
        ):
            result = fn()
            if asyncio.iscoroutine(result):
                await result
            # 跑完即记成功. 任务体自己吞掉的异常会另外经 _job_failed 记失败,
            # 所以 fail_at 比 ok_at 新就说明这一轮实际没干成事.
            await _record_job_outcome(job_name, ok=True)
    except DistributedLockNotAcquired:
        logger.debug(
            f"[CRON] {job_name} skipped: another instance holds the lock",
            extra={"event": EVT_SCHEDULER_JOB, "task_name": job_name, "phase": "skipped_lock"},
        )
    except DistributedLockUnavailable as e:
        logger.warning(
            f"[CRON] {job_name} skipped: distributed lock unavailable ({e})",
            extra={
                "event": EVT_SCHEDULER_JOB,
                "task_name": job_name,
                "phase": "skipped_lock_unavailable",
                "error_type": type(e).__name__,
            },
        )


_JOB_HEALTH_KEY = "scheduler:health"
_JOB_HEALTH_TTL_S = 90 * 24 * 3600


async def _record_job_outcome(job_name: str, ok: bool, detail: str = "") -> None:
    """记下每个定时任务最近一次成功/失败的时刻.

    存在的理由: L2 动态分级 cron 因为一个 SQL 类型错每晚崩了几个月没人发现. 失败
    走 logger.warning, 成功则默认不出声 —— 于是"从来没成功过"和"这次没事可做"在
    日志里长得一模一样. 只看日志判断不了一个任务是不是活的.

    这里把两种结局都写进 Redis 哈希, 让"上次成功是什么时候"成为可查的事实. 记录
    失败本身不能再抛 —— 掩盖掉原始故障比丢一条健康数据更糟.
    """
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    field = "ok_at" if ok else "fail_at"
    try:
        redis = await get_redis()
        await redis.hset(_JOB_HEALTH_KEY, f"{job_name}:{field}", stamp)
        if not ok and detail:
            await redis.hset(_JOB_HEALTH_KEY, f"{job_name}:fail_reason", detail[:200])
        await redis.expire(_JOB_HEALTH_KEY, _JOB_HEALTH_TTL_S)
    except Exception:
        pass


def _job_failed(job_name: str, exc: BaseException) -> None:
    """定时任务失败的统一上报口.

    级别定在 error 而不是 warning: 一个定时任务失败意味着某个功能整块没运行, 而
    warning 在这个项目里量大到没人逐条看 —— L2 那次就是这么漏掉的.
    """
    logger.error(
        f"[CRON] {job_name} failed: {exc}",
        extra={
            "event": EVT_SCHEDULER_JOB, "task_name": job_name,
            "phase": "failed", "error_type": type(exc).__name__,
        },
    )
    asyncio.create_task(_record_job_outcome(job_name, ok=False, detail=str(exc)))


async def _run_for_all_agents(
    fn: Callable, concurrency: int = 3, task_name: str = "task",
) -> None:
    """Run an async function for all agents with concurrency control.

    每个 agent 起一个独立 usage session, cron 的 LLM 调用 (作息生成 / 画像更新 /
    月度 overview 等) 按 agent 维度落到 llm_usage 表 (scope=schedule_cron).
    无 LLM 调用的 cron (l2_adjustment 等) flush 返回 None, 不写空行.
    """
    from app.db import db
    from app.observability import bind_context
    from app.services.agent_template.registry import get_template_owner_id
    from app.services.llm.usage_tracker import usage_session
    # Only run per-agent crons for real, active agents. Exclude:
    #  - archived/provisioning agents (no live schedule/memory should grow), and
    #  - the template agent (owned by the template system user) — it is a frozen
    #    clone source and must never accumulate its own daily schedule / self-
    #    memory summaries (those would otherwise be copied into every new clone).
    owner_id = await get_template_owner_id()
    where: dict = {"status": "active"}
    if owner_id:
        where["userId"] = {"not": owner_id}
    agents = await db.aiagent.find_many(where=where)
    sem = asyncio.Semaphore(concurrency)
    n_total = len(agents)
    n_failed = 0
    started_at = asyncio.get_event_loop().time()

    logger.info(
        f"[CRON] {task_name} started for {n_total} agents",
        extra={"event": EVT_SCHEDULER_JOB, "task_name": task_name,
               "phase": "started", "n_agents": n_total},
    )

    async def _process(agent):
        nonlocal n_failed
        async with sem:
            async with usage_session(
                scope="schedule_cron", conversation_id=None,
                agent_id=agent.id, user_id=getattr(agent, "userId", None),
            ):
                # 绑 log 上下文 — 整个 cron 任务链 (生成作息/画像/记忆衰减) 的
                # log 都带 agent 字段, 便于 Axiom 按 agent 切分
                with bind_context(
                    agent_id=agent.id,
                    agent_name=getattr(agent, "name", None),
                    user_id=getattr(agent, "userId", None),
                ):
                    try:
                        await fn(agent)
                    except Exception as e:
                        n_failed += 1
                        logger.warning(
                            f"{task_name} failed for agent {agent.id}: {e}",
                            extra={"event": EVT_SCHEDULER_JOB, "task_name": task_name,
                                   "phase": "agent_failed",
                                   "error_type": type(e).__name__},
                        )

    await asyncio.gather(*[_process(a) for a in agents])

    elapsed = asyncio.get_event_loop().time() - started_at
    logger.info(
        f"[CRON] {task_name} done in {elapsed:.1f}s "
        f"({n_total - n_failed}/{n_total} ok)",
        extra={"event": EVT_SCHEDULER_JOB, "task_name": task_name,
               "phase": "completed", "n_agents": n_total, "n_failed": n_failed,
               "elapsed_sec": round(elapsed, 2)},
    )


def setup_scheduler():
    """Configure and start the job scheduler."""
    # Daily growth intimacy at 2 AM
    scheduler.add_job(
        _run_daily_intimacy,
        "cron",
        hour=2,
        minute=0,
        id="daily_intimacy",
        replace_existing=True,
    )

    # Achievement exact/end-of-day rollups shortly after local midnight.
    scheduler.add_job(
        _run_achievement_daily_rollup,
        "cron",
        hour=0,
        minute=5,
        timezone=settings.schedule_timezone,
        id="achievement_daily_rollup",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=6 * 3600,
    )
    scheduler.add_job(
        _run_achievement_daily_rollup,
        "date",
        run_date=datetime.now(ZoneInfo(settings.schedule_timezone))
        + timedelta(seconds=30),
        id="achievement_daily_rollup_startup_catchup",
        replace_existing=True,
    )

    # Weekly topic intimacy on Sunday at 2 AM
    scheduler.add_job(
        _run_weekly_topic_intimacy,
        "cron",
        day_of_week="sun",
        hour=2,
        minute=30,
        id="weekly_topic_intimacy",
        replace_existing=True,
    )

    # Daily L2 memory scoring adjustment at 2:30 AM (spec §1.5.2)
    scheduler.add_job(
        _run_l2_adjustment,
        "cron",
        hour=2,
        minute=30,
        id="l2_adjustment",
        replace_existing=True,
        max_instances=1,
    )

    # Weekly reflection on Sunday at 4 AM
    scheduler.add_job(
        _run_weekly_reflection,
        "cron",
        day_of_week="sun",
        hour=4,
        minute=0,
        id="weekly_reflection",
        replace_existing=True,
        max_instances=1,
    )

    # Weekly memory hygiene after reflection: conservative fact evolution and
    # duplicate cleanup inside each top-level memory category.
    scheduler.add_job(
        _run_memory_hygiene,
        "cron",
        day_of_week="sun",
        hour=4,
        minute=20,
        id="memory_hygiene",
        replace_existing=True,
        max_instances=1,
    )

    # Phase 2 weekly L3 consolidation (Sunday 05:10, after hygiene). Gated by
    # MEMORY_CONSOLIDATION_ENABLED (default off) — it archives original rows,
    # so it must be validated on staging data before production enablement.
    scheduler.add_job(
        _run_memory_consolidation,
        "cron",
        day_of_week="sun",
        hour=5,
        minute=10,
        id="memory_consolidation",
        replace_existing=True,
        max_instances=1,
    )

    # Weekly portrait update on Sunday at 3:45 AM (staggered from daily reflection)
    scheduler.add_job(
        _run_weekly_portraits,
        "cron",
        day_of_week="sun",
        hour=3,
        minute=45,
        id="weekly_portrait",
        replace_existing=True,
        max_instances=1,
    )

    # Daily schedule generation at 3:30 AM. spec Part 5 §4.3: 内部链式触发
    # special_dates_scan, 保证 scan 读当天新作息. max_instances=1 防 LLM 慢
    # 导致 3:30 跨日叠加.
    scheduler.add_job(
        _run_daily_schedules,
        "cron",
        hour=3,
        minute=30,
        id="daily_schedule",
        replace_existing=True,
        max_instances=1,
    )

    # Monthly life overview refresh on 1st at 5:30 AM
    scheduler.add_job(
        _run_monthly_overview_refresh,
        "cron",
        day=1,
        hour=5,
        minute=30,
        id="monthly_overview",
        replace_existing=True,
        max_instances=1,
    )

    # Daily schedule review at 4 AM
    scheduler.add_job(
        _run_schedule_review,
        "cron",
        hour=4,
        minute=0,
        id="schedule_review",
        replace_existing=True,
        max_instances=1,
    )

    scheduler.add_job(
        _run_proactive_orchestrator_scan,
        "interval",
        minutes=1,
        id="proactive_orchestrator_scan",
        replace_existing=True,
    )

    scheduler.add_job(
        _run_music_schedule_transition_scan,
        "interval",
        minutes=1,
        id="music_schedule_transition_scan",
        replace_existing=True,
        max_instances=1,
    )

    scheduler.add_job(
        _run_runtime_job_queue,
        "interval",
        seconds=5,
        id="runtime_job_queue",
        replace_existing=True,
        max_instances=1,
    )

    scheduler.add_job(
        _run_game_memory_sync_retry,
        "interval",
        minutes=1,
        id="game_memory_sync_retry",
        replace_existing=True,
        max_instances=1,
    )

    scheduler.add_job(
        _run_notification_dispatch,
        "interval",
        seconds=15,
        id="notification_dispatch",
        replace_existing=True,
        max_instances=1,
    )

    scheduler.add_job(
        _run_capsule_ready_notifications,
        "cron",
        hour=9,
        minute=0,
        id="capsule_ready_notifications",
        replace_existing=True,
        max_instances=1,
    )

    scheduler.add_job(
        _run_offline_trigger_scan,
        "cron",
        hour=10,
        minute=20,
        id="offline_trigger_scan",
        replace_existing=True,
        max_instances=1,
    )

    # Patience recovery every hour
    scheduler.add_job(
        _run_patience_recovery,
        "interval",
        hours=1,
        id="patience_recovery",
        replace_existing=True,
    )

    # Redis health recheck: flip app-level readonly mode as Redis recovers/fails
    scheduler.add_job(
        _run_redis_health_recheck,
        "interval",
        seconds=30,
        id="redis_health_recheck",
        replace_existing=True,
    )

    # spec §1.4: 后台定时任务每秒扫描延迟队列
    scheduler.add_job(
        _run_aggregation_scan,
        "interval",
        seconds=1,
        id="aggregation_scan",
        replace_existing=True,
        max_instances=1,  # prevent "max instances reached" warning
    )

    # §9.5: Time trigger scan — 15 秒一次. 之前 1 分钟一次, 配合 reminder
    # "不早响"策略导致 "两分钟后" 的提醒最坏延迟可达 1 分钟. 15s 把最坏延迟
    # 压到 15s, 同时 DB 查询负担可控 (find_many 在 ±15s 窗口, isActive=true
    # 的 timetrigger 数量级很小).
    scheduler.add_job(
        _run_trigger_scan,
        "interval",
        seconds=15,
        id="trigger_scan",
        replace_existing=True,
        # D2: 多实例错峰 — 无 jitter 时所有实例同刻抢分布式锁 (thundering herd).
        # ±3s 不影响 reminder 最坏延迟量级 (15s → 18s), 换取锁竞争均摊.
        jitter=3,
    )
    logger.info("Scheduler: trigger_scan registered (interval=15s, jitter=3s)")

    scheduler.add_job(
        _run_last_will_scan,
        "interval",
        hours=1,
        id="last_will_scan",
        replace_existing=True,
        max_instances=1,
        jitter=300,
    )

    # Part 5 §2.1: NTP 校准每 6 小时跑一次
    scheduler.add_job(
        _run_ntp_calibration,
        "cron",
        hour="*/6",
        minute=15,
        id="ntp_calibration",
        replace_existing=True,
    )

    # 节假日 DB 不走定时 cron, 也不走后端批量 refresh: 年度变化 (国务院
    # 11-12 月发布次年安排), 运营需要时在 admin UI "查询外部源" 拉候选挑
    # 选保存即可 — preview + bulk_save 工作流覆盖所有使用场景.

    # 1688 access_token 每 6 小时刷新一次 (token 有效期通常 1 天, 提前续期防过期).
    # 仅在启用 ali1688 provider 时实际执行刷新, 否则空跑跳过.
    scheduler.add_job(
        _run_ali1688_token_refresh,
        "interval",
        hours=6,
        id="ali1688_token_refresh",
        replace_existing=True,
        max_instances=1,
    )

    # 本地 trace 采集保留期清理: 删除超过 trace_retention_days 的 trace_runs 行.
    # 被查看过的 trace 已物化进 message_traces 镜像, 不受影响. 凌晨 5:10 错开
    # 其他日任务 (schedule 3:30 / review 4:00 / hygiene 4:20).
    scheduler.add_job(
        _run_trace_retention,
        "cron",
        hour=5,
        minute=10,
        id="trace_retention",
        replace_existing=True,
        max_instances=1,
    )

    scheduler.start()
    logger.info("Job scheduler started")


async def _run_weekly_portraits():
    await _run_distributed_job(
        "weekly_portrait",
        3600,
        lambda: _run_for_all_agents(
            lambda a: update_portrait_weekly(a.userId, a.id),
            concurrency=3, task_name="Portrait update",
        ),
    )


async def _run_achievement_daily_rollup():
    async def _body():
        from app.services.achievements.mode import achievement_evaluation_enabled
        from app.services.achievements.service import run_daily_rollup

        if not await achievement_evaluation_enabled():
            # "off" freezes the checkpoint so a later re-enable replays the
            # missed days via the existing catch-up loop below.
            logger.info(
                "Achievement daily rollup skipped: achievement_mode=off "
                "(checkpoint frozen for catch-up)"
            )
            return

        target_day = (
            datetime.now(ZoneInfo(settings.schedule_timezone)).date()
            - timedelta(days=1)
        )
        redis = None
        try:
            redis = await get_redis()
            raw_checkpoint = await redis.get(_ACHIEVEMENT_ROLLUP_CHECKPOINT_KEY)
        except Exception as e:
            logger.warning(f"Achievement rollup checkpoint unavailable: {e}")
            raw_checkpoint = None

        checkpoint_text = (
            raw_checkpoint.decode()
            if isinstance(raw_checkpoint, bytes)
            else str(raw_checkpoint or "")
        )
        try:
            last_success = date.fromisoformat(checkpoint_text)
        except ValueError:
            last_success = None

        if last_success and last_success >= target_day:
            return
        first_day = target_day if last_success is None else last_success + timedelta(days=1)
        if (target_day - first_day).days > 365:
            first_day = target_day - timedelta(days=365)
            logger.warning("Achievement rollup catch-up capped at 366 days")

        current_day = first_day
        while current_day <= target_day:
            local_day = datetime.combine(
                current_day,
                time.min,
                tzinfo=ZoneInfo(settings.schedule_timezone),
            )
            await run_daily_rollup(local_day)
            if redis is not None:
                await redis.set(
                    _ACHIEVEMENT_ROLLUP_CHECKPOINT_KEY,
                    current_day.isoformat(),
                )
            current_day += timedelta(days=1)

    await _run_distributed_job("achievement_daily_rollup", 1800, _body)


async def _run_weekly_reflection():
    await _run_distributed_job("weekly_reflection", 7200, run_weekly_reflection)


async def _run_daily_schedules():
    async def _body():
        async def _gen(agent):
            overview = await get_life_overview(agent.id)
            mbti = get_mbti(agent)
            await generate_daily_schedule(
                agent.id, agent.name, mbti,
                life_overview=overview, user_id=agent.userId,
            )

        await _run_for_all_agents(_gen, concurrency=3, task_name="Daily schedule")

        # spec Part 5 §4.3: "每日凌晨**生成 AI 作息表时**" 触发特殊日期扫描.
        # 链式 await 保证 scan 一定读到当天新作息 (不是昨天 cache).
        # 历史: 独立 cron 在 3:35 跑, 若 daily_schedule 慢于 5 min 仍读旧 cache.
        try:
            await scan_special_dates_today()
        except Exception as e:
            _job_failed("Special dates scan (chained)", e)

    await _run_distributed_job("daily_schedule", 7200, _body)


async def _run_monthly_overview_refresh():
    async def _refresh(agent):
        await generate_and_save_life_overview(agent)

    await _run_distributed_job(
        "monthly_overview",
        7200,
        lambda: _run_for_all_agents(_refresh, concurrency=2, task_name="Monthly overview"),
    )


async def _run_schedule_review():
    await _run_distributed_job(
        "schedule_review",
        3600,
        lambda: _run_for_all_agents(
            lambda a: review_daily_schedule(a.id, a.userId, a.name),
            concurrency=3, task_name="Schedule review",
        ),
    )


async def _run_proactive_orchestrator_scan():
    """扫描主动状态机。第一阶段仅做区间推进和互斥检查。"""
    async def _body():
        try:
            await scan_proactive_states()
        except Exception as e:
            _job_failed("Proactive orchestrator scan", e)

    await _run_distributed_job("proactive_orchestrator_scan", 55, _body)


async def _run_music_schedule_transition_scan():
    async def _body():
        try:
            await scan_music_schedule_transitions()
        except Exception as e:
            _job_failed("Music schedule transition scan", e)

    await _run_distributed_job("music_schedule_transition_scan", 55, _body)


async def _run_runtime_job_queue():
    try:
        processed = await process_runtime_jobs(max_jobs=20)
        if processed:
            logger.info(
                f"[CRON] runtime job queue processed {processed} jobs",
                extra={
                    "event": EVT_SCHEDULER_JOB,
                    "task_name": "runtime_job_queue",
                    "phase": "completed",
                    "processed": processed,
                },
            )
    except Exception as e:
        logger.warning(
            f"Runtime job queue scan failed: {e}",
            extra={
                "event": EVT_SCHEDULER_JOB,
                "task_name": "runtime_job_queue",
                "phase": "failed",
                "error_type": type(e).__name__,
            },
        )


async def _run_notification_dispatch():
    async def _body():
        try:
            processed = await dispatch_due_notifications()
            if processed:
                logger.info(f"[CRON] notification dispatch processed {processed} events")
        except Exception as e:
            _job_failed("Notification dispatch", e)

    await _run_distributed_job("notification_dispatch", 120, _body)


async def _run_capsule_ready_notifications():
    async def _body():
        try:
            processed = await scan_ready_capsule_notifications()
            if processed:
                logger.info(f"[CRON] capsule ready notifications queued for {processed} scopes")
        except Exception as e:
            _job_failed("Capsule ready notification scan", e)

    await _run_distributed_job("capsule_ready_notifications", 1800, _body)


async def _run_offline_trigger_scan():
    async def _body():
        try:
            stats = await scan_offline_triggers()
            if stats.get("activities") or stats.get("gifts") or stats.get("failed"):
                logger.info(f"[CRON] offline trigger scan: {stats}")
        except Exception as e:
            _job_failed("Offline trigger scan", e)

    await _run_distributed_job("offline_trigger_scan", 3600, _body)


async def _run_l2_adjustment():
    """Spec §1.5.2: recalculate L2 scores, promote/demote.

    Both outcomes are logged on purpose. This job crashed on a SQL type error
    every night for months and nobody noticed, because failure went out at
    warning level and success only spoke up when something was promoted or
    demoted — so a job that had never once worked looked exactly like a job
    with nothing to do. Now silence in the logs means the job did not run at
    all, which is a signal rather than the normal case.
    """
    async def _body():
        try:
            stats = await run_l2_adjustment()
            logger.info(
                f"[CRON] l2_adjustment ok: {stats}",
                extra={"event": EVT_SCHEDULER_JOB, "task_name": "l2_adjustment",
                       "phase": "ok", "adjusted": stats.get("adjusted", 0),
                       "promoted": stats.get("promoted", 0),
                       "demoted": stats.get("demoted", 0)},
            )
        except Exception as e:
            logger.error(
                f"[CRON] l2_adjustment failed: {e}",
                extra={"event": EVT_SCHEDULER_JOB, "task_name": "l2_adjustment",
                       "phase": "failed", "error_type": type(e).__name__},
            )

    await _run_distributed_job("l2_adjustment", 3600, _body)


async def _run_memory_hygiene():
    """Run weekly memory duplicate cleanup, fact evolution, and log retention."""
    async def _body():
        try:
            stats = await run_memory_hygiene()
            if stats.get("archived") or stats.get("merged") or stats.get("updated"):
                logger.info(f"Memory hygiene: {stats}")
        except Exception as e:
            _job_failed("Memory hygiene", e)
        # Retention rides the same weekly slot: purge stale `access` changelog
        # rows (13 months+). Non-access operations are kept forever (audit).
        try:
            from app.services.memory.lifecycle.changelog_retention import (
                purge_stale_access_changelog,
            )
            await purge_stale_access_changelog()
        except Exception as e:
            _job_failed("Access changelog purge", e)

    await _run_distributed_job("memory_hygiene", 7200, _body)


async def _run_memory_consolidation():
    """Phase 2: weekly L3 cluster compression (flag-gated, template-excluded)."""
    from app.config import settings

    if not settings.memory_consolidation_enabled:
        logger.debug("memory consolidation disabled (MEMORY_CONSOLIDATION_ENABLED=false)")
        return

    async def _body():
        from app.services.memory.lifecycle.consolidation import (
            compress_l3_clusters_for_workspace,
        )
        from app.services.workspace.workspaces import resolve_workspace_id

        async def _one(agent):
            workspace_id = await resolve_workspace_id(
                user_id=agent.userId, agent_id=agent.id,
            )
            if workspace_id:
                await compress_l3_clusters_for_workspace(
                    user_id=agent.userId, workspace_id=workspace_id,
                )

        # _run_for_all_agents already excludes the template agent + non-active.
        await _run_for_all_agents(_one, concurrency=2, task_name="Memory consolidation")

    await _run_distributed_job("memory_consolidation", 7200, _body)


async def _run_daily_intimacy():
    await _run_distributed_job(
        "daily_intimacy",
        3600,
        lambda: _run_for_all_agents(
            lambda a: compute_growth_intimacy(a.id, a.userId, a.createdAt),
            concurrency=3, task_name="Growth intimacy",
        ),
    )


async def _run_weekly_topic_intimacy():
    await _run_distributed_job(
        "weekly_topic_intimacy",
        3600,
        lambda: _run_for_all_agents(
            lambda a: compute_topic_intimacy(a.id, a.userId, a.createdAt),
            concurrency=3, task_name="Topic intimacy",
        ),
    )


async def _run_patience_recovery():
    await _run_distributed_job(
        "patience_recovery",
        900,
        lambda: _run_for_all_agents(
            lambda a: recover_patience_hourly(a.id, a.userId),
            concurrency=5, task_name="Patience recovery",
        ),
    )


async def _run_trigger_scan():
    """§9.5: 扫描到期的时间触发器。"""
    async def _body():
        try:
            await scan_triggers()
        except Exception as e:
            _job_failed("Trigger scan", e)

    await _run_distributed_job("trigger_scan", 120, _body)


async def _run_last_will_scan():
    """Scan inactive-login last wills and create pending deliveries."""
    async def _body():
        try:
            stats = await scan_due_last_wills()
            if stats.get("triggered") or stats.get("deliveries"):
                logger.info(f"Last will scan: {stats}")
        except Exception as e:
            _job_failed("Last will scan", e)

    await _run_distributed_job("last_will_scan", 3600, _body)


async def _run_redis_health_recheck():
    """30s 周期 ping Redis 并更新 _redis_healthy flag. 允许 Redis 故障后自愈
    (修好后下次 tick flip 回 healthy, 写 endpoints 自动重开).
    """
    try:
        from app.redis_client import recheck_redis_health
        await recheck_redis_health()
    except Exception as e:
        _job_failed("Redis health recheck", e)


async def _run_trace_retention():
    """本地 trace 采集保留期清理 (trace_backend=local 的 trace_runs 表)."""
    async def _body():
        from app.services.chat.local_tracer import purge_expired_trace_runs
        try:
            deleted = await purge_expired_trace_runs()
            if deleted:
                logger.info(f"Trace retention purge: {deleted} rows deleted")
        except Exception as e:
            _job_failed("Trace retention purge", e)

    await _run_distributed_job("trace_retention", 3600, _body)


async def _run_ntp_calibration():
    """Part 5 §2.1: NTP 校准, 漂移 > 阈值时告警."""
    async def _body():
        import asyncio
        from app.services.schedule_domain.time_service import calibrate_against_ntp
        try:
            # ntplib 是同步阻塞调用, 放线程池
            drift = await asyncio.to_thread(calibrate_against_ntp)
            if drift is None:
                logger.warning("NTP calibration failed (network or lib unavailable)")
                return
            if abs(drift) > 1.0:
                logger.warning(f"NTP drift {drift:+.3f}s exceeds 1s threshold; investigate clock source")
            else:
                logger.info(f"NTP drift {drift:+.3f}s (within threshold)")
        except Exception as e:
            _job_failed("NTP calibration job", e)

    await _run_distributed_job("ntp_calibration", 900, _body)


async def _run_ali1688_token_refresh():
    """每 6h 刷新 1688 access_token；仅在启用 ali1688 provider 时执行。"""
    async def _body():
        if "ali1688" not in {
            settings.gift_commerce_provider,
            settings.gift_logistics_provider,
        }:
            return  # 未启用 1688，空跑跳过
        try:
            result = await refresh_access_token()
            logger.info("ali1688 token refreshed: expires_in=%ss", result.get("expires_in"))
        except Exception as e:
            # 用 error 级别：刷新连续失败会让 Redis 内 token 过期后回退到可能已失效的
            # 初始 token，导致下单全失败——需要运维介入（重新授权拿 refresh_token）。
            logger.error(f"ali1688 token refresh FAILED, gift ordering will break if this persists: {e}")

    await _run_distributed_job("ali1688_token_refresh", 300, _body)


async def _run_game_memory_sync_retry():
    async def _body():
        from app.services.games.native import (
            abort_stale_sessions,
            retry_missing_chat_side_effects,
            retry_pending_memory_sync,
        )

        closed = await abort_stale_sessions()
        retried = await retry_pending_memory_sync(limit=10)
        attempted_chat_repairs = await retry_missing_chat_side_effects(limit=20)
        if closed:
            logger.info("[CRON] closed %s stale native game session(s)", closed)
        if retried:
            logger.info("[CRON] retried %s native game memory sync(s)", retried)
        if attempted_chat_repairs:
            logger.info(
                "[CRON] attempted %s native game chat projection repair(s)",
                attempted_chat_repairs,
            )

    await _run_distributed_job("game_memory_sync_retry", 180, _body)


async def _run_aggregation_scan():
    await _run_distributed_job("aggregation_scan", 120, _run_aggregation_scan_body)


async def _run_aggregation_scan_body():
    """Scan aggregation windows and due delayed replies, then deliver asynchronously."""
    from app.services.chat.orchestrator import stream_chat_response
    from app.services.runtime.ws_manager import manager
    from app.api.realtime.ws import stream_to_ws
    from app.db import db

    try:
        due_user_turns = await scan_due_user_turns()
        await _enqueue_scanned_aggregation_results(due_user_turns, manager)

        due_conversations = await scan_due_delayed_messages()
        for conv_id, payloads in due_conversations:
            # Prevent concurrent processing of the same conversation.
            # Capture the owner token so unlock only releases our own lock (CAS).
            lock_token = await try_lock_conversation(conv_id, ttl=120)
            if not lock_token:
                logger.debug(f"Conversation {conv_id[:8]} is locked, skipping this scan")
                continue

            try:
                try:
                    merged = merge_delayed_payloads(payloads)
                    if not merged:
                        continue

                    # 去重 gate: ws.py 已用 enqueue_or_append_delayed 关闭主要 race;
                    # 这里兜底 "msg1 已被 flush 出队但仍在 LLM 中, msg2 才到达" 窗口:
                    # 上一轮 LLM 数据拉取若已隐式包含本 user_msg(写到 reply 的
                    # metadata.covered_until_user_ts), 则跳过避免重复回复。
                    # 仅看 metadata 显式字段 → 不会因短路/边界 reply 误伤未覆盖消息。
                    user_msg_id = merged.get("user_message_id")
                    if user_msg_id and await _already_covered(conv_id, user_msg_id):
                        logger.info(
                            f"[DEDUP-GATE] skip conv={conv_id[:8]} "
                            f"user_msg={user_msg_id[:8]} already covered by prior reply"
                        )
                        continue

                    conv = await db.conversation.find_unique(
                        where={"id": conv_id},
                        include={"agent": True},
                    )
                    if not conv or not conv.agent:
                        continue

                    gen = stream_chat_response(
                        conversation_id=conv_id,
                        user_message=merged["user_message"],
                        agent=conv.agent,
                        user_id=merged["user_id"],
                        reply_context=merged.get("reply_context"),
                        save_user_message=False,
                        user_message_id=merged.get("user_message_id"),
                        delivered_from_queue=True,
                    )

                    # stream_to_ws 内部每条 chunk 走 manager.send_event,
                    # fast path 本地命中或 slow path publish 跨 worker, 无需手工查 WS.
                    # 离线用户 (无 WS / 跨进程 publish 也无人订阅): 仍 await 消费完
                    # generator 触发 LLM + 持久化, 避免漏存回复.
                    # 标记回复生成中: 生成期间到达的新消息在 plan_user_message_aggregation
                    # 里会被路由到 delayed queue 合并, 而非另起一轮近似重复的回复.
                    await mark_reply_inflight(conv_id)
                    await stream_to_ws(gen, conv_id)
                    logger.debug(f"Delayed reply pushed for conv={conv_id[:8]}")
                except Exception as conv_err:
                    # 单个会话的处理失败不能阻塞批次内其他会话; 推 done 事件给前端
                    # 解卡"消息处理中"状态, 防用户卡死 UI.
                    logger.exception(
                        f"Aggregation scan: conv {conv_id[:8]} processing failed: {conv_err}"
                    )
                    try:
                        await manager.send_event(
                            conv_id, "done", {"message_id": "error", "error": str(conv_err)},
                        )
                    except Exception as notify_err:
                        logger.warning(
                            f"Failed to notify conv {conv_id[:8]} of error: {notify_err}"
                        )
            finally:
                await clear_reply_inflight(conv_id)
                await unlock_conversation(conv_id, lock_token)
    except Exception as e:
        _job_failed("Aggregation scan", e)


async def _already_covered(conversation_id: str, user_msg_id: str) -> bool:
    """检查是否已有 assistant 回复在 prompt 中显式覆盖了这条 user 消息.

    依赖 orchestrator 主路径在 save_replies 时写入的 metadata.covered_until_user_ts
    (LLM 数据拉取时刻能看到的最新 user 消息时间). user_msg.createdAt 早于或等于
    任一 assistant 的 covered_until_user_ts → 视为已被覆盖, 跳过避免双发。

    短路/边界回复不写此字段 → 不会误判, 这类回复对应的 user 消息仍按原路处理。
    """
    from app.db import db

    user_msg = await db.message.find_unique(where={"id": user_msg_id})
    if not user_msg or user_msg.createdAt is None:
        return False

    # 拉取此消息之后的所有 assistant 消息 (上限 10 条防长会话扫描过大).
    later_ai = await db.message.find_many(
        where={
            "conversationId": conversation_id,
            "role": "assistant",
            "createdAt": {"gt": user_msg.createdAt},
        },
        order={"createdAt": "asc"},
        take=10,
    )
    for ai_msg in later_ai:
        md = getattr(ai_msg, "metadata", None) or {}
        covered = md.get("covered_until_user_ts") if isinstance(md, dict) else None
        if not covered:
            continue
        try:
            cutoff = datetime.fromisoformat(covered) if isinstance(covered, str) else None
        except ValueError:
            cutoff = None
        if cutoff is None:
            continue
        # 比较时统一带 tz, prisma 默认返回 aware datetime; isoformat 也带 tz.
        if cutoff >= user_msg.createdAt:
            return True
    return False


async def _enqueue_scanned_aggregation_results(results, manager) -> None:
    """Move scanned aggregation windows into the shared delayed reply queue."""
    for agent_id, user_id, combined_text, conv_id, reply_context, latest_message_id in results:
        delay_seconds = float((reply_context or {}).get("delay_seconds", 0.0) or 0.0)
        await enqueue_delayed_message(
            conv_id,
            {
                "conversation_id": conv_id,
                "agent_id": agent_id,
                "user_id": user_id,
                "message": combined_text,
                "message_id": latest_message_id,
                "reply_context": reply_context,
            },
            delay_seconds,
        )
        # send_event 跨进程 routing: scheduler 与 WS holder 不同 worker 时 publish.
        if delay_seconds > 5:
            await manager.send_event(conv_id, "delay", {"duration": delay_seconds})
        await manager.send_event(conv_id, "pending", {"status": "queued", "delay": delay_seconds})


def shutdown_scheduler():
    """Shutdown the scheduler."""
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Job scheduler stopped")
