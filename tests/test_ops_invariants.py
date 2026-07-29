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
