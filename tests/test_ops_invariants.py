"""数据不变量的阈值判定.

这组检查的价值在于抓"定时任务报了成功其实没干成事" —— cron 健康表看不出这类。
每条不变量都对应一次真实事故, 测试里按事故当时的数据形态构造用例。
"""

from __future__ import annotations

import pytest

from app.services.ops import invariants as inv


class _FakeDb:
    """按顺序返回预设的 query_raw 结果."""

    def __init__(self, *rows):
        self._rows = list(rows)
        self.calls = 0

    async def query_raw(self, *_args, **_kwargs):
        self.calls += 1
        return self._rows.pop(0)


def _patch_db(monkeypatch, fake):
    import app.db as app_db

    monkeypatch.setattr(app_db, "db", fake, raising=False)


class TestScheduleToday:
    @pytest.mark.asyncio
    async def test_zero_rows_today_is_violated(self, monkeypatch):
        """日期错位那天的真实形态: 24 个活跃 agent, 今天 0 行."""
        _patch_db(monkeypatch, _FakeDb([{"agents": 24, "scheduled": 0}]))
        result = await inv._check_schedule_today()
        assert result.status == "violated"
        assert result.observed == {"active_agents": 24, "with_schedule_today": 0}

    @pytest.mark.asyncio
    async def test_full_coverage_is_ok(self, monkeypatch):
        _patch_db(monkeypatch, _FakeDb([{"agents": 24, "scheduled": 24}]))
        assert (await inv._check_schedule_today()).status == "ok"

    @pytest.mark.asyncio
    async def test_a_couple_missing_is_tolerated(self, monkeypatch):
        """模板 agent 不参与每日生成, 差一两个属正常."""
        _patch_db(monkeypatch, _FakeDb([{"agents": 24, "scheduled": 23}]))
        assert (await inv._check_schedule_today()).status == "ok"

    @pytest.mark.asyncio
    async def test_many_missing_warns(self, monkeypatch):
        _patch_db(monkeypatch, _FakeDb([{"agents": 24, "scheduled": 10}]))
        assert (await inv._check_schedule_today()).status == "warn"

    @pytest.mark.asyncio
    async def test_no_agents_is_not_a_violation(self, monkeypatch):
        _patch_db(monkeypatch, _FakeDb([{"agents": 0, "scheduled": 0}]))
        assert (await inv._check_schedule_today()).status == "ok"


class TestL2Freshness:
    @pytest.mark.asyncio
    async def test_stale_scores_are_violated(self, monkeypatch):
        """L2 cron 死掉几个月时就是这个形态."""
        _patch_db(monkeypatch, _FakeDb([{"total": 5897, "fresh": 0}]))
        result = await inv._check_l2_scores_fresh()
        assert result.status == "violated"

    @pytest.mark.asyncio
    async def test_mostly_fresh_is_ok(self, monkeypatch):
        _patch_db(monkeypatch, _FakeDb([{"total": 5897, "fresh": 5890}]))
        assert (await inv._check_l2_scores_fresh()).status == "ok"

    @pytest.mark.asyncio
    async def test_partial_coverage_warns(self, monkeypatch):
        _patch_db(monkeypatch, _FakeDb([{"total": 1000, "fresh": 700}]))
        assert (await inv._check_l2_scores_fresh()).status == "warn"

    @pytest.mark.asyncio
    async def test_no_l2_memories_is_ok(self, monkeypatch):
        _patch_db(monkeypatch, _FakeDb([{"total": 0, "fresh": 0}]))
        assert (await inv._check_l2_scores_fresh()).status == "ok"


class TestAccessLogging:
    @pytest.mark.asyncio
    async def test_conversation_without_access_logs_is_violated(self, monkeypatch):
        _patch_db(monkeypatch, _FakeDb([{"accesses": 0, "user_messages": 60}]))
        assert (await inv._check_memory_access_logged()).status == "violated"

    @pytest.mark.asyncio
    async def test_quiet_day_is_not_a_violation(self, monkeypatch):
        """今天没人聊天时打点本来就是零, 不能误报."""
        _patch_db(monkeypatch, _FakeDb([{"accesses": 0, "user_messages": 0}]))
        assert (await inv._check_memory_access_logged()).status == "ok"

    @pytest.mark.asyncio
    async def test_normal_traffic_is_ok(self, monkeypatch):
        _patch_db(monkeypatch, _FakeDb([{"accesses": 398, "user_messages": 60}]))
        assert (await inv._check_memory_access_logged()).status == "ok"


class TestEmbeddingCoverage:
    @pytest.mark.asyncio
    async def test_missing_vectors_beyond_tolerance_is_violated(self, monkeypatch):
        _patch_db(monkeypatch, _FakeDb([{"memories": 8000, "embeddings": 7000}]))
        assert (await inv._check_embeddings_complete()).status == "violated"

    @pytest.mark.asyncio
    async def test_extra_vectors_are_fine(self, monkeypatch):
        """向量表也留着归档记忆的行, 多出来是正常的."""
        _patch_db(monkeypatch, _FakeDb([{"memories": 7998, "embeddings": 8386}]))
        assert (await inv._check_embeddings_complete()).status == "ok"

    @pytest.mark.asyncio
    async def test_a_handful_missing_only_warns(self, monkeypatch):
        _patch_db(monkeypatch, _FakeDb([{"memories": 8000, "embeddings": 7990}]))
        assert (await inv._check_embeddings_complete()).status == "warn"


class TestConsolidation:
    @pytest.mark.asyncio
    async def test_disabled_consolidation_is_not_checked(self, monkeypatch):
        from app.config import settings

        monkeypatch.setattr(settings, "memory_consolidation_enabled", False)
        assert (await inv._check_consolidation_active()).status == "ok"

    @pytest.mark.asyncio
    async def test_enabled_but_never_ran_is_violated(self, monkeypatch):
        from app.config import settings

        monkeypatch.setattr(settings, "memory_consolidation_enabled", True)
        _patch_db(monkeypatch, _FakeDb([{"runs": 0, "compressed": 0}]))
        assert (await inv._check_consolidation_active()).status == "violated"

    @pytest.mark.asyncio
    async def test_running_but_compressing_nothing_warns(self, monkeypatch):
        """聚类阈值没跟着换 embedding 模型时就是这个形态: 在跑, 但一条都压不动."""
        from app.config import settings

        monkeypatch.setattr(settings, "memory_consolidation_enabled", True)
        _patch_db(monkeypatch, _FakeDb([{"runs": 7, "compressed": 0}]))
        assert (await inv._check_consolidation_active()).status == "warn"

    @pytest.mark.asyncio
    async def test_productive_run_is_ok(self, monkeypatch):
        from app.config import settings

        monkeypatch.setattr(settings, "memory_consolidation_enabled", True)
        _patch_db(monkeypatch, _FakeDb([{"runs": 1, "compressed": 6}]))
        assert (await inv._check_consolidation_active()).status == "ok"


class TestCapacityHeadroom:
    """这条不回答"有没有坏", 回答"还能撑多久" —— 到线时提醒该上 nginx least_conn。"""

    @pytest.mark.asyncio
    async def test_idle_system_reports_headroom(self, monkeypatch):
        monkeypatch.setattr(inv, "_capacity_probe", None, raising=False)
        result = await _capacity_with(monkeypatch, cpu=0.3, ws=2)
        assert result.status == "ok"
        assert result.observed == {"cpu_percent": 0.3, "ws_connections": 2}

    @pytest.mark.asyncio
    async def test_high_cpu_warns(self, monkeypatch):
        result = await _capacity_with(monkeypatch, cpu=55.0, ws=2)
        assert result.status == "warn" and "CPU" in result.detail

    @pytest.mark.asyncio
    async def test_many_connections_warn(self, monkeypatch):
        result = await _capacity_with(monkeypatch, cpu=1.0, ws=80)
        assert result.status == "warn" and "WS" in result.detail

    @pytest.mark.asyncio
    async def test_connection_count_comes_from_redis_not_process_memory(self):
        """ConnectionManager 的内存表每个 worker 只看得到自己那一半.

        用它统计会在多 worker 下systematically 低估, 于是水位线永远够不到 —— 提醒
        变成永远不响的哑铃。
        """
        import inspect

        src = inspect.getsource(inv._check_capacity_headroom)
        assert "presence:online:ws" in src, "在线连接数没走 Redis 全局 ZSET"

    @pytest.mark.asyncio
    async def test_cpu_sampling_does_not_block_the_event_loop(self):
        """不能用 psutil.cpu_percent(interval=...) —— 它是同步阻塞的.

        在 async 函数里调它会把事件循环卡满采样时长, 单 worker 下所有在线用户一起
        卡。复用 collect_host_metrics: 它用 await asyncio.sleep 取两次 /proc/stat
        快照, 采样期间事件循环照常转。
        """
        import inspect

        import ast

        src = inspect.getsource(inv._check_capacity_headroom)
        assert "collect_host_metrics" in src

        # 只看代码不看注释 —— 注释里正解释着"为什么不用 psutil", 按纯文本匹配会
        # 把这段解释本身判成违规。
        tree = ast.parse(src.strip())
        names = {
            n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
        } | {
            a.name.split(".")[0]
            for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
        } | {
            (n.module or "").split(".")[0]
            for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
        }
        assert "psutil" not in names, (
            "psutil.cpu_percent 是同步阻塞调用, 会把事件循环卡满采样时长; "
            "而且 psutil 不是本项目的声明依赖"
        )

    @pytest.mark.asyncio
    async def test_missing_cpu_reading_does_not_crash(self, monkeypatch):
        """/proc/stat 读不到时 cpu_percent 是 None, 不能拿去做比较或格式化."""
        result = await _capacity_with(monkeypatch, cpu=None, ws=3)
        assert result.status == "ok" and "未知" in result.detail


async def _capacity_with(monkeypatch, *, cpu: float | None, ws: int):
    """跑一次容量检查, CPU 与连接数由参数注入."""
    from unittest.mock import AsyncMock

    import app.services.system_metrics as sm

    monkeypatch.setattr(
        sm, "collect_host_metrics",
        AsyncMock(return_value={"system": {"cpu_percent": cpu}}),
    )

    fake_redis = AsyncMock()
    fake_redis.zcard = AsyncMock(return_value=ws)
    import app.redis_client as rc

    monkeypatch.setattr(rc, "get_redis", AsyncMock(return_value=fake_redis))
    return await inv._check_capacity_headroom()


class TestRunner:
    @pytest.mark.asyncio
    async def test_one_broken_check_does_not_sink_the_report(self, monkeypatch):
        """一条查不动就整份报告消失, 等于又回到"看不见"."""

        async def _boom():
            raise RuntimeError("column does not exist")

        monkeypatch.setattr(
            inv, "_CHECKS",
            (
                ("boom", "会炸的检查", _boom),
                ("fine", "正常检查", _ok_check),
            ),
        )
        results = await inv.run_invariant_checks()
        assert [r.status for r in results] == ["error", "ok"]
        assert results[0].key == "boom"
        assert "column does not exist" in results[0].detail

    @pytest.mark.asyncio
    async def test_every_registered_check_reports_its_own_key(self, monkeypatch):
        """注册表里的 key 必须和检查返回的 key 一致, 否则报错时张冠李戴."""
        for key, _title, fn in inv._CHECKS:
            assert fn.__name__ == f"_check_{key}", (
                f"{fn.__name__} 与注册 key {key!r} 对不上"
            )


async def _ok_check():
    return inv.InvariantResult("fine", "正常检查", "ok")


class TestNoOversizedMemories:
    """超限记忆的巡检.

    这类失效没有任何外部症状 —— 不报错、不告警, 只是 agent 想不起某件事。产生途径
    有好几条 (profile 生成 / txt 导入 / hygiene 合并), 与其在每个入口各自防, 不如
    在结果侧盯住。
    """

    _LONG = "很长的记忆内容需要越过单条上限。" * 20

    def _rows(self, n: int, content: str):
        # 两张表各查一次, 所以要准备两批返回值。
        return [[{"id": f"m{i}", "content": content} for i in range(n)] for _ in range(2)]

    @pytest.mark.asyncio
    async def test_clean_database_is_ok(self, monkeypatch):
        _patch_db(monkeypatch, _FakeDb(*self._rows(3, "短记忆")))
        r = await inv._check_no_oversized_memories()
        assert r.status == "ok"

    @pytest.mark.asyncio
    async def test_even_a_few_oversized_rows_are_a_violation(self, monkeypatch):
        """任何一条超限都要红。

        这条断言 2026-08 反转过: 原先是"少量超限 (≤50) 只记 warn, 不该把整块巡检
        标成红色"。反转的理由是那个门槛在生产上被实测证伪 —— warn 不打 error 日志、
        不计入后台 badge, 只有人主动打开页面才看得见, 于是模板克隆持续漏出超限记忆
        整整一个月无人察觉, 而每次漏的量都远不到 51 条。

        "少量超限只影响那几条自己" 本身没说错, 错在它是存量视角: 每条超限记忆都
        意味着某个写入路径漏了, 而漏的路径不会自己停下来。
        """
        _patch_db(monkeypatch, _FakeDb(*self._rows(3, self._LONG)))
        r = await inv._check_no_oversized_memories()
        assert r.status == "violated"
        assert "检索时会被整条跳过" in r.detail

    @pytest.mark.asyncio
    async def test_large_scale_regression_is_a_violation(self, monkeypatch):
        _patch_db(monkeypatch, _FakeDb(*self._rows(40, self._LONG)))
        r = await inv._check_no_oversized_memories()
        assert r.status == "violated"

    @pytest.mark.asyncio
    async def test_reports_sample_ids_for_triage(self, monkeypatch):
        """只给个数字没法排查, 要能直接拿 id 去查."""
        _patch_db(monkeypatch, _FakeDb(*self._rows(1, self._LONG)))
        r = await inv._check_no_oversized_memories()
        assert "m0" in r.observed["sample_ids"]

    def test_is_registered_in_the_check_list(self):


        assert any(key == "no_oversized_memories" for key, _, _ in inv._CHECKS)
