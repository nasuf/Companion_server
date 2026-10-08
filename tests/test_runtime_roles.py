"""Role boundaries, shared handlers, health and process-failure behavior."""
import asyncio
from contextlib import asynccontextmanager
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.config import settings
from app.services.runtime.roles import require_role, validate_process_budget
from app.services.runtime.role_health import RoleHealth, probe_dependencies, cancel_and_drain
from app.services.runtime.sql_job_contracts import WorkerStopRequired


@pytest.mark.parametrize('name,api,scheduler,consumer', [
    ('integrated', True, True, True), ('api', True, False, False),
    ('scheduler', False, True, False), ('background', False, False, True),
])
def test_role_ownership(name, api, scheduler, consumer):
    role = require_role(name, api=api)
    assert (role.starts_scheduler, role.consumes_redis_jobs) == (scheduler, consumer)
    with pytest.raises(ValueError):
        require_role(name, api=not api)


@pytest.mark.parametrize('name', ['foreground', 'delivery', 'unknown', 'API', ''])
def test_unimplemented_or_mistyped_role_never_activates_consumers(name):
    with pytest.raises(ValueError):
        require_role(name, api=False)


@pytest.mark.parametrize('role,count,valid', [
    ('api', None, True), ('integrated', None, True), ('api', 1, False),
    ('background', None, False), ('background', 2, False),
    ('scheduler', 4, True), ('background', 4, True), ('api', True, False),
])
def test_split_budget_must_be_explicit(role, count, valid):
    config = SimpleNamespace(app_runtime_role=role, web_concurrency=2, llm_process_count=count)
    if valid:
        validate_process_budget(config)
    else:
        with pytest.raises(ValueError):
            validate_process_budget(config)


def test_total_llm_budget_includes_non_api_processes(monkeypatch):
    from app.services.llm.resilience import _per_worker_share
    monkeypatch.setattr(settings, 'llm_process_count', 4)
    assert _per_worker_share(64) == 16
    assert _per_worker_share(16) == 4
    monkeypatch.setattr(settings, 'llm_process_count', None)
    assert _per_worker_share(64) == 32


@pytest.mark.asyncio
async def test_worker_unknown_usage_scope_still_uses_background_quota(monkeypatch):
    from app.services.llm import resilience
    entered = []
    @asynccontextmanager
    async def slot():
        yield
    def semaphore(mapping, provider, count):
        entered.append((mapping is resilience._bg_slots, count))
        return slot()
    monkeypatch.setattr(resilience, '_get_semaphore', semaphore)
    monkeypatch.setattr(settings, 'app_runtime_role', 'background')
    monkeypatch.setattr(settings, 'llm_process_count', 4)
    async with resilience._llm_slot('synthetic'):
        pass
    assert entered == [(True, 4), (False, 16)]


def test_registration_uses_service_without_router_imports(monkeypatch):
    from app.services.runtime import job_queue
    from app.services.runtime.handler_registry import register_runtime_handlers
    from app.services.agent_initialization import _run_agent_initialization_job
    from app.services.memory.generation_lock import MEMORY_GENERATION_LOCK_TTL_S
    monkeypatch.setattr(job_queue, '_HANDLERS', {})
    monkeypatch.setattr(job_queue, '_RECOVERY_DELAYS', {})
    monkeypatch.setattr(job_queue, '_LEGACY_NO_DELAY_HANDLERS', set())
    assert register_runtime_handlers() == ('agent_initialization',)
    assert register_runtime_handlers() == ('agent_initialization',)
    assert job_queue._HANDLERS == {'agent_initialization': _run_agent_initialization_job}
    assert job_queue._RECOVERY_DELAYS['agent_initialization'] == MEMORY_GENERATION_LOCK_TTL_S
    assert job_queue._LEGACY_NO_DELAY_HANDLERS == {'agent_initialization'}


@pytest.mark.asyncio
@pytest.mark.parametrize('agent', [None, SimpleNamespace(status='archived')])
async def test_moved_initializer_preserves_missing_archived_short_circuit(monkeypatch, agent):
    from app.services import agent_initialization as service
    fake_db = SimpleNamespace(aiagent=SimpleNamespace(find_unique=AsyncMock(return_value=agent)))
    patience = AsyncMock()
    monkeypatch.setattr(service, 'db', fake_db)
    monkeypatch.setattr(service, 'init_patience', patience)
    await service._run_agent_initialization_job({'agent_id':'missing','user_id':'user'})
    patience.assert_not_awaited()


@pytest.mark.asyncio
async def test_moved_initializer_preserves_scope_and_overrides(monkeypatch):
    from app.services import agent_initialization as service
    agent = SimpleNamespace(id='a', status='provisioning')
    monkeypatch.setattr(service, 'db', SimpleNamespace(aiagent=SimpleNamespace(find_unique=AsyncMock(return_value=agent))))
    monkeypatch.setattr(service, 'init_patience', AsyncMock())
    inner = AsyncMock()
    monkeypatch.setattr(service, '_run_agent_initialization_inner', inner)
    await service._run_agent_initialization_job({'agent_id':'a','user_id':'u','workspace_id':'w','personality':{'warmth':80},'profile_override':{'identity':{}},'career_template_override':{'title':'test'}})
    inner.assert_awaited_once_with(agent, 'u', 'w', {'warmth':80}, profile_override={'identity':{}}, career_template_override={'title':'test'})


@pytest.mark.asyncio
async def test_dependency_probe_reports_failure_and_recovers():
    health = RoleHealth('background', initialized=True)
    db = SimpleNamespace(query_raw=AsyncMock(side_effect=RuntimeError('unavailable')))
    redis = SimpleNamespace(ping=AsyncMock(return_value=True))
    await probe_dependencies(health, db, redis)
    assert not health.ready() and not health.postgres and health.redis
    db.query_raw = AsyncMock(return_value=[{'one':1}])
    await probe_dependencies(health, db, redis)
    assert health.ready()
    health.last_probe -= 36
    assert not health.ready()
    health.last_probe = time.monotonic()
    health.fatal = True
    assert not health.ready()


@pytest.mark.asyncio
async def test_shutdown_cancels_cooperative_tasks():
    task = asyncio.create_task(asyncio.sleep(100))
    await cancel_and_drain([task], timeout=.1)
    assert task.cancelled()


@pytest.mark.asyncio
async def test_shutdown_rejects_uncooperative_task():
    started, release = asyncio.Event(), asyncio.Event()
    async def ignores_cancel():
        started.set()
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            await release.wait()
    task = asyncio.create_task(ignores_cancel())
    await started.wait()
    try:
        with pytest.raises(WorkerStopRequired):
            await cancel_and_drain([task], timeout=.02)
    finally:
        release.set()
        await task


@pytest.mark.asyncio
async def test_stop_required_is_never_retried_in_same_worker(monkeypatch):
    from jobs import runtime
    from app.services.runtime import job_queue
    fake = AsyncMock(side_effect=WorkerStopRequired('unsafe handler'))
    monkeypatch.setattr(job_queue, 'process_runtime_jobs', fake)
    health = RoleHealth('background', initialized=True, postgres=True, redis=True, last_probe=time.monotonic())
    with pytest.raises(WorkerStopRequired):
        await runtime.consume_background_jobs(health)
    fake.assert_awaited_once()


@pytest.mark.asyncio
async def test_scheduler_failure_is_visible_to_supervision(monkeypatch):
    from jobs import runtime, scheduler
    monkeypatch.setattr(scheduler.scheduler, 'state', 0)
    with pytest.raises(RuntimeError, match='scheduler stopped'):
        await runtime.monitor_dependencies(RoleHealth('scheduler'), MagicMock())


@pytest.mark.asyncio
async def test_internal_health_never_claims_readiness_before_startup():
    from jobs import runtime
    runtime.app.state.role_health = RoleHealth('background')
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=runtime.app), base_url='http://isolated') as client:
        assert (await client.get('/health')).status_code == 503
        runtime.app.state.role_health = RoleHealth('background', initialized=True, postgres=True, redis=True, last_probe=time.monotonic())
        response = await client.get('/health')
        assert response.status_code == 200 and response.json()['role'] == 'background'
        runtime.app.state.role_health.closing = True
        assert (await client.get('/health')).status_code == 503


@pytest.mark.asyncio
async def test_api_only_lifespan_never_starts_scheduler(monkeypatch):
    import app.main as main
    from app.services import runtime_config
    from app.services.schedule_domain import holiday_cache
    from app.services.runtime.ws_manager import manager
    from app.services import agent_avatars
    from app.services.speech_output import client as tts
    monkeypatch.setattr(settings, 'app_runtime_role', 'api')
    monkeypatch.setattr(settings, 'llm_process_count', None)
    monkeypatch.setattr(type(settings), 'validate_security_config', lambda self:None)
    for name in ('connect_db','disconnect_db','close_redis','ensure_prompt_templates','ensure_default_careers','ensure_default_names'):
        monkeypatch.setattr(main, name, AsyncMock())
    monkeypatch.setattr(main, 'get_redis', AsyncMock(return_value=SimpleNamespace(ping=AsyncMock())))
    monkeypatch.setattr(main, '_warn_if_embedding_model_uncalibrated', lambda:None)
    @asynccontextmanager
    async def seed(*args, **kwargs):
        yield
    monkeypatch.setattr(main, 'distributed_lock', seed)
    monkeypatch.setattr(holiday_cache, 'reload', AsyncMock())
    monkeypatch.setattr(main, 'reload_holiday_cache', AsyncMock())
    monkeypatch.setattr(runtime_config, 'load_caches', AsyncMock())
    monkeypatch.setattr(runtime_config, 'refresh_worker_config', lambda:asyncio.sleep(100))
    monkeypatch.setattr(agent_avatars, 'validate_avatar_assets', lambda:None)
    monkeypatch.setattr(tts, 'close_tts_client', AsyncMock())
    monkeypatch.setattr(manager, 'start_subscriber', AsyncMock())
    monkeypatch.setattr(manager, 'stop_subscriber', AsyncMock())
    start, stop = MagicMock(), MagicMock()
    monkeypatch.setattr(main, 'setup_scheduler', start)
    monkeypatch.setattr(main, 'shutdown_scheduler', stop)
    generator = main.lifespan(main.app)
    await anext(generator)
    assert not start.called
    await generator.aclose()
    assert not stop.called
    main.disconnect_db.assert_awaited_once()


def test_scheduler_definition_read_does_not_start_timers(monkeypatch):
    from jobs import scheduler
    start = MagicMock(side_effect=AssertionError('must not start'))
    monkeypatch.setattr(scheduler.AsyncIOScheduler, 'start', start)
    jobs = scheduler.scheduler_job_definitions()
    assert {'aggregation_scan','runtime_job_queue','daily_schedule','trigger_scan'} <= {job.id for job in jobs}
    assert not any(job.id.endswith('startup_catchup') for job in jobs)
    start.assert_not_called()


def test_independent_scheduler_excludes_consumer_and_local_redis_timer(monkeypatch):
    from jobs import scheduler
    fake = MagicMock()
    monkeypatch.setattr(scheduler, 'scheduler', fake)
    scheduler.setup_scheduler(role='scheduler')
    assert [call.args[0] for call in fake.remove_job.call_args_list] == ['runtime_job_queue','redis_health_recheck']
    fake.start.assert_called_once()
    with pytest.raises(ValueError):
        scheduler.setup_scheduler(role='api')


@pytest.mark.asyncio
async def test_api_process_still_detects_stale_cron_without_live_scheduler(monkeypatch):
    from app.services.ops import cron_health
    from app import redis_client
    from jobs import scheduler
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    redis = SimpleNamespace(hgetall=AsyncMock(return_value={'aggregation_scan:ok_at':(now-timedelta(seconds=10)).isoformat()}))
    monkeypatch.setattr(redis_client, 'get_redis', AsyncMock(return_value=redis))
    monkeypatch.setattr(scheduler.scheduler, 'state', 0)
    report = await cron_health.collect_cron_health(now=now)
    assert report.definitions_available
    assert next(job for job in report.jobs if job.job_id=='aggregation_scan').verdict == 'stale'


@pytest.mark.asyncio
@pytest.mark.parametrize('role', ['background', 'scheduler'])
async def test_independent_lifespan_owns_only_its_role(monkeypatch, role):
    from jobs import runtime, scheduler
    from app.services import runtime_config
    from app.services.schedule_domain import holiday_cache
    monkeypatch.setattr(settings, 'app_runtime_role', role)
    monkeypatch.setattr(settings, 'llm_process_count', 4)
    monkeypatch.setattr(type(settings), 'validate_security_config', lambda self:None)
    monkeypatch.setattr(runtime, 'configure_logging', lambda:None)
    monkeypatch.setattr(runtime, 'configure_langsmith', lambda:None)
    monkeypatch.setattr(runtime, 'connect_db', AsyncMock())
    monkeypatch.setattr(runtime, 'disconnect_db', AsyncMock())
    monkeypatch.setattr(runtime, 'close_redis', AsyncMock())
    monkeypatch.setattr(runtime, 'get_redis', AsyncMock(return_value=SimpleNamespace(ping=AsyncMock(return_value=True))))
    monkeypatch.setattr(runtime, 'db', SimpleNamespace(query_raw=AsyncMock(return_value=[{'one':1}])))
    monkeypatch.setattr(runtime_config, 'load_caches', AsyncMock())
    monkeypatch.setattr(runtime_config, 'refresh_worker_config', lambda:asyncio.sleep(100))
    monkeypatch.setattr(holiday_cache, 'reload', AsyncMock())
    monkeypatch.setattr(runtime, 'monitor_dependencies', lambda *args:asyncio.sleep(100))
    monkeypatch.setattr(runtime, 'consume_background_jobs', lambda *args:asyncio.sleep(100))
    register = MagicMock()
    start, stop = MagicMock(), MagicMock()
    monkeypatch.setattr(runtime, 'register_runtime_handlers', register)
    monkeypatch.setattr(scheduler, 'setup_scheduler', start)
    monkeypatch.setattr(scheduler, 'shutdown_scheduler', stop)
    async with runtime.lifespan(runtime.app):
        assert runtime.app.state.role_health.ready()
        assert register.called == (role == 'background')
        assert start.called == (role == 'scheduler')
        if role == 'scheduler':
            start.assert_called_once_with(role='scheduler')
    assert not runtime.app.state.role_health.ready()
    assert stop.called == (role == 'scheduler')
    runtime.disconnect_db.assert_awaited_once()
    runtime.close_redis.assert_awaited_once()


@pytest.mark.asyncio
async def test_dependency_failure_cleans_up_before_readiness(monkeypatch):
    from jobs import runtime
    monkeypatch.setattr(settings, 'app_runtime_role', 'background')
    monkeypatch.setattr(settings, 'llm_process_count', 4)
    monkeypatch.setattr(type(settings), 'validate_security_config', lambda self:None)
    monkeypatch.setattr(runtime, 'connect_db', AsyncMock(side_effect=RuntimeError('connect failed')))
    close_db, close_redis = AsyncMock(), AsyncMock()
    monkeypatch.setattr(runtime, 'disconnect_db', close_db)
    monkeypatch.setattr(runtime, 'close_redis', close_redis)
    with pytest.raises(RuntimeError, match='connect failed'):
        async with runtime.lifespan(runtime.app):
            pytest.fail('unavailable runtime started')
    assert not runtime.app.state.role_health.ready()
    close_db.assert_awaited_once()
    close_redis.assert_awaited_once()


@pytest.mark.asyncio
async def test_unexpected_loop_exit_marks_fatal_and_terminates(monkeypatch):
    from jobs import runtime
    from app.services import runtime_config
    from app.services.schedule_domain import holiday_cache
    monkeypatch.setattr(settings, 'app_runtime_role', 'background')
    monkeypatch.setattr(settings, 'llm_process_count', 4)
    monkeypatch.setattr(type(settings), 'validate_security_config', lambda self:None)
    monkeypatch.setattr(runtime, 'connect_db', AsyncMock())
    monkeypatch.setattr(runtime, 'disconnect_db', AsyncMock())
    monkeypatch.setattr(runtime, 'close_redis', AsyncMock())
    monkeypatch.setattr(runtime, 'get_redis', AsyncMock(return_value=SimpleNamespace(ping=AsyncMock(return_value=True))))
    monkeypatch.setattr(runtime, 'db', SimpleNamespace(query_raw=AsyncMock(return_value=[{'one':1}])))
    monkeypatch.setattr(runtime_config, 'load_caches', AsyncMock())
    monkeypatch.setattr(runtime_config, 'refresh_worker_config', lambda:asyncio.sleep(100))
    monkeypatch.setattr(holiday_cache, 'reload', AsyncMock())
    monkeypatch.setattr(runtime, 'consume_background_jobs', AsyncMock(side_effect=WorkerStopRequired('unsafe')))
    exit_called = asyncio.Event()
    monkeypatch.setattr(runtime, 'terminate_worker', exit_called.set)
    async with runtime.lifespan(runtime.app):
        await asyncio.wait_for(exit_called.wait(), timeout=1)
        assert runtime.app.state.role_health.fatal
        assert not runtime.app.state.role_health.ready()


@pytest.mark.asyncio
async def test_nested_job_ownership_remains_tracked_until_outer_finishes(monkeypatch):
    from jobs import scheduler
    monkeypatch.setattr(scheduler, '_record_job_outcome', AsyncMock())
    current = asyncio.current_task()
    async def inner():
        assert current in scheduler.active_scheduler_tasks()
    async def outer():
        await scheduler._run_local_job('inner', inner)
        assert current in scheduler.active_scheduler_tasks()
    await scheduler._run_local_job('outer', outer)
    assert current not in scheduler.active_scheduler_tasks()


@pytest.mark.asyncio
async def test_cancelled_job_removes_ownership_and_leaves_no_success(monkeypatch):
    from jobs import scheduler
    record = AsyncMock()
    monkeypatch.setattr(scheduler, '_record_job_outcome', record)
    started = asyncio.Event()
    async def pending():
        started.set()
        await asyncio.sleep(100)
    task = asyncio.create_task(scheduler._run_local_job('pending', pending))
    await started.wait()
    assert task in scheduler.active_scheduler_tasks()
    await cancel_and_drain([task])
    assert task not in scheduler.active_scheduler_tasks()
    record.assert_not_awaited()
