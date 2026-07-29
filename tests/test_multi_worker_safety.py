"""多 worker 安全性守卫.

2026-07-29 起服务以多个 uvicorn worker 进程运行。它们共享 Redis / DB / 文件系统,
**但不共享内存** —— 所有进程内状态都变成"每 worker 一份"。

这类问题的共同特征是: 出错时没有任何症状, 只是数据悄悄多了一份、某个上限悄悄翻了
倍、或者某个 worker 起不来而别的还在跑。所以这里钉住的是结构性不变量, 而不是行为。

上线前的完整扫描结论见 CLAUDE.md §11。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCHEDULER = _ROOT / "jobs" / "scheduler.py"
_MAIN = _ROOT / "app" / "main.py"
_DOCKERFILE = _ROOT / "Dockerfile"
_COMPOSE = _ROOT / "docker-compose.deploy.yml"


class TestSchedulerJobs:
    def test_every_job_is_either_locked_or_deliberately_per_worker(self):
        """每个 cron 要么全局只跑一次, 要么明确是"每实例各跑".

        漏一个就意味着那个任务在 N 个 worker 上同时跑 —— 对发消息类任务就是用户
        收到 N 条重复。
        """
        tree = ast.parse(_SCHEDULER.read_text(encoding="utf-8"))

        registered: dict[str, str] = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "add_job" and n.args:
                jid = next(
                    (ast.literal_eval(k.value) for k in n.keywords
                     if k.arg == "id" and isinstance(k.value, ast.Constant)),
                    None,
                )
                if jid:
                    registered[jid] = ast.unparse(n.args[0])

        guarded: set[str] = set()
        for fn in tree.body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls = {
                getattr(c.func, "id", None)
                for c in ast.walk(fn) if isinstance(c, ast.Call)
            }
            if {"_run_distributed_job", "_run_local_job"} & calls:
                guarded.add(fn.name)

        unguarded = sorted(j for j, h in registered.items() if h not in guarded)
        assert not unguarded, (
            "这些 cron 没有经过 _run_distributed_job (全局一次) 或 _run_local_job "
            f"(每实例一次), 多 worker 下会重复执行：{unguarded}"
        )

    def test_per_worker_jobs_are_an_explicit_short_list(self):
        """"每实例各跑"必须是少数几个刻意为之的, 不能随手扩张.

        redis_health_recheck 要每进程 ping 自己的连接; runtime_job_queue 是多实例
        并行消费同一个队列。其余任何任务走这条路都该先解释清楚为什么。
        """
        src = _SCHEDULER.read_text(encoding="utf-8")
        tree = ast.parse(src)
        local = {
            fn.name for fn in tree.body
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(
                isinstance(c, ast.Call) and getattr(c.func, "id", None) == "_run_local_job"
                for c in ast.walk(fn)
            )
        }
        assert local == {"_run_redis_health_recheck", "_run_runtime_job_queue"}, (
            f"「每实例各跑」的任务集合变了: {sorted(local)}。新增前先确认它在 N 个 "
            "worker 上同时跑是安全且必要的。"
        )


class TestStartupSeeding:
    def test_seeding_runs_under_a_distributed_lock(self):
        """两个 seeder 都是"先查缺失再创建", 并发启动会一起创建.

        career_templates.title 没有唯一约束 —— 结果是每个 worker 各建一份重复职业;
        prompt_templates.key 有唯一约束 —— 结果是输的那个 worker 抛异常起不来。
        """
        src = _MAIN.read_text(encoding="utf-8")
        tree = ast.parse(src)

        seed_calls = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and getattr(n.func, "id", None) in {
                "ensure_default_careers", "ensure_prompt_templates",
            }
        ]
        assert seed_calls, "找不到 seeding 调用"

        locks = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and getattr(n.func, "id", None) == "distributed_lock"
            and any(
                isinstance(a, ast.Constant) and a.value == "startup:seed" for a in n.args
            )
        ]
        assert locks, (
            "启动 seeding 没有加分布式锁 —— 多 worker 同时启动会重复创建职业模板, "
            "或撞唯一键让 worker 起不来"
        )

    def test_seed_lock_failure_does_not_crash_startup(self):
        """拿不到锁说明别人在做, 应当跳过而不是让这个 worker 起不来."""
        src = _MAIN.read_text(encoding="utf-8")
        assert "DistributedLockNotAcquired" in src
        assert "Seeding skipped" in src


class TestProcessLocalLimits:
    def test_llm_cap_is_divided_across_workers(self):
        """信号量是进程内对象; 不摊分的话 2 个 worker 就把全局上限翻倍.

        provider 的 rate limit 按全局算, 所以配置项的语义必须始终是"整个服务的
        在途上限", 不随起了几个进程而漂。
        """
        from app.config import settings
        from app.services.llm.resilience import _per_worker_share

        original = settings.web_concurrency
        try:
            settings.web_concurrency = 1
            assert _per_worker_share(64) == 64
            settings.web_concurrency = 2
            assert _per_worker_share(64) == 32
            settings.web_concurrency = 4
            assert _per_worker_share(64) == 16
        finally:
            settings.web_concurrency = original

    def test_share_never_starves_a_worker(self):
        """向上取整并保底 1: 某个 worker 拿到 0 个槽位等于它上面的聊天全部卡死,
        比稍微宽松的限流严重得多。"""
        from app.config import settings
        from app.services.llm.resilience import _per_worker_share

        original = settings.web_concurrency
        try:
            settings.web_concurrency = 32
            assert _per_worker_share(16) >= 1
            settings.web_concurrency = 3
            assert _per_worker_share(64) == 22        # ceil(64/3)
        finally:
            settings.web_concurrency = original

    def test_disabled_cap_stays_disabled(self):
        from app.services.llm.resilience import _per_worker_share

        assert _per_worker_share(0) == 0
        assert _per_worker_share(None) == 0


class TestDeployWiring:
    def test_dockerfile_uses_the_worker_env(self):
        src = _DOCKERFILE.read_text(encoding="utf-8")
        assert "--workers" in src, "Dockerfile 还是单 worker"
        assert "WEB_CONCURRENCY" in src, "worker 数应当可通过环境变量调整"

    def test_compose_passes_the_same_variable(self):
        assert "WEB_CONCURRENCY" in _COMPOSE.read_text(encoding="utf-8")

    def test_worker_count_fits_the_db_connection_budget(self):
        """每 worker 占 DB_CONNECTION_LIMIT 个连接, postgres 上限 50.

        4 worker × 12 + 其他占用就超了 —— 超限的表现是随机的连接失败, 而不是一个
        清晰的报错。
        """
        from app.config import settings

        per_worker = 12          # docker-compose DB_CONNECTION_LIMIT 默认值
        postgres_max = 50
        reserved = 5             # 迁移 / 备份 / admin 连接
        assert settings.web_concurrency * per_worker + reserved < postgres_max, (
            f"web_concurrency={settings.web_concurrency} 会把 DB 连接打到上限"
        )


class TestLimitSemantics:
    """判别原则: 保护**共享外部资源**的上限必须按 worker 摊分, 保护**本进程资源**的
    不该摊分。搞反了两个方向都出问题 —— 前者会把 provider 的配额打爆, 后者会让每个
    进程只剩几分之一的可用额度。
    """

    def test_llm_cap_is_divided_because_it_guards_a_shared_quota(self):
        """provider 的 rate limit 是全局的, N 个进程各持 64 就是 N×64 打过去."""
        import inspect

        from app.services.llm import resilience

        assert "_per_worker_share" in inspect.getsource(resilience._llm_slot)

    def test_background_task_cap_is_not_divided_because_it_guards_a_local_loop(self):
        """后台任务上限护的是本进程的事件循环, 每进程一份正是对的.

        摊分它反而有害: 2 worker 各只剩 128, 而每个进程的事件循环本来都能承载 256。
        """
        import inspect

        from app.services.runtime import tasks

        src = inspect.getsource(tasks)
        assert "_per_worker_share" not in src, (
            "后台任务上限被摊分了 —— 它护的是本进程事件循环, 不是共享资源"
        )

    def test_user_facing_rate_limits_live_in_redis(self):
        """按用户计的限流必须落在 Redis, 放进程内就等于被 worker 数放宽."""
        import inspect

        from app.api.public import speech

        src = inspect.getsource(speech._enforce_rate_limit)
        assert "redis" in src and "incr" in src, (
            "语音限流不在 Redis 上, 多 worker 下每个进程各算一份, 实际额度翻倍"
        )


class TestCheckThenCreate:
    def test_first_greeting_conversation_creation_is_serialised(self):
        """查空到创建之间要串行化.

        这个函数由 WS 连接触发。用户双设备登录或重连风暴会让两次调用并行, 双双查到
        "没有会话"再各建一个 —— 同一个 agent 下两个默认会话, 消息还会被分到两边。
        """
        import inspect

        from app.services.proactive import sender

        src = inspect.getsource(sender._ensure_first_greeting_conversations)
        assert "distributed_lock" in src, "查空到创建之间没有串行化"
        assert src.count("find_many") >= 2, (
            "拿到锁之后必须重查一次 —— 等锁期间别人可能已经建好了"
        )


class TestWebSocketCrossProcess:
    def test_conversation_send_falls_back_to_pubsub(self):
        """用户的 WS 连接只存在于某一个 worker 的内存里.

        别的 worker 上的后台任务 (主动消息 / 延迟回复) 要推给他, 必须能跨进程。
        """
        import inspect

        from app.services.runtime.ws_manager import ConnectionManager

        src = inspect.getsource(ConnectionManager.send_event)
        assert "_publish" in src, "本地未命中时没有跨进程发布, 消息会静默丢失"

    def test_publisher_filters_its_own_messages(self):
        """自己发的自己收到要跳过, 否则本地已送达的会再送一次."""
        import inspect

        from app.services.runtime.ws_manager import ConnectionManager

        assert "_instance_id" in inspect.getsource(ConnectionManager.__init__)
        assert "sender" in inspect.getsource(ConnectionManager._publish)


class TestMemoryPipelineLock:
    def test_pipeline_is_fail_closed_across_instances(self):
        """拿不到分布式锁时必须跳过本批, 不能退回本地锁继续跑.

        退回本地锁 = 两个 worker 各自"独占"地跑同一个 conversation, L1 SINGLETON
        会重复 (生产已复现过一次)。
        """
        src = (_ROOT / "app" / "services" / "chat" / "post_process.py").read_text(
            encoding="utf-8",
        )
        assert "memory_pipeline:" in src, "记忆管线没有分布式锁"
        assert "fail-closed" in src or "fail_closed" in src, (
            "缺少 fail-closed 说明; 锁不可用时若退回无锁执行会重现 L1 重复"
        )


@pytest.mark.parametrize("path", [
    "app/services/chat/post_process.py",
    "app/services/llm/resilience.py",
])
def test_process_local_state_is_documented(path: str):
    """进程内可变状态要写明多 worker 下的含义, 否则下一个人会当成全局状态用."""
    src = (_ROOT / path).read_text(encoding="utf-8")
    assert any(k in src for k in ("多 worker", "多实例", "进程内", "每实例")), (
        f"{path} 有模块级可变状态但没说明它在多进程下是每进程一份"
    )
