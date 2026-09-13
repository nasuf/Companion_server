"""CI smoke for the temporal_recall eval mechanics.

The real run needs Ollama for embeddings (offline tooling, run manually). Here
we pin the invariants that would silently make the eval measure the wrong
thing —— the exact class of bug that nearly shipped in this file when the
frozen-clock helper broke isinstance() and silently disabled P3 for the whole
sweep, without any test failing.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from app.services.memory.retrieval import ranking as R
from evals.temporal_recall.cases import CASES, SEED_BANK
from evals.temporal_recall.run_eval import (
    NOW_UTC,
    _FrozenNow,
    _freeze_time_for_ranking,
    _recency_boost,
)


class TestFrozenClock:
    """The frozen-clock helper is the eval's most fragile piece. If it breaks
    isinstance(x, datetime) inside ranking, _occur_recency_factor returns None
    for every candidate and P3 silently doesn't run — the sweep prints "no
    effect at any weight" and we'd conclude P3 is dead. We hit exactly this."""

    def test_datetime_now_returns_frozen_moment(self):
        with patch.object(R, "datetime", _FrozenNow()):
            assert R.datetime.now(timezone.utc) == NOW_UTC

    def test_real_datetime_still_passes_isinstance_under_patch(self):
        real = datetime.now(timezone.utc)
        with patch.object(R, "datetime", _FrozenNow()):
            # If this fails, _occur_recency_factor's `isinstance(raw, datetime)`
            # returns False and silently zeros out P3 for the entire eval.
            assert isinstance(real, R.datetime)

    def test_occur_recency_factor_fires_under_patch(self):
        occ = NOW_UTC - timedelta(days=5)
        with _freeze_time_for_ranking():
            factor = R._occur_recency_factor({"occur_time": occ})
        # 5 days at 30-day half-life ~ 0.89 (not None, not 0)
        assert factor is not None
        assert 0.85 < factor < 0.95


class TestRecencyBoostSwitch:
    def test_context_manager_restores_original(self):
        original = R._RECENCY_BOOST_WEIGHT
        with _recency_boost(2.5):
            assert R._RECENCY_BOOST_WEIGHT == 2.5
        assert R._RECENCY_BOOST_WEIGHT == original

    def test_context_manager_restores_on_exception(self):
        original = R._RECENCY_BOOST_WEIGHT
        try:
            with _recency_boost(2.5):
                raise ValueError("boom")
        except ValueError:
            pass
        assert R._RECENCY_BOOST_WEIGHT == original


class TestCaseBankShape:
    """The A/B report depends on the bank having enough queries on each side —
    P3-triggering and not-triggering — that the numbers aren't noise."""

    def test_bank_has_both_recency_and_non_recency_time_cases(self):
        time_cases = [c for c in CASES if c.needs_time]
        recency = [c for c in time_cases if R._is_recency_seeking_query(c.query)]
        non_recency = [c for c in time_cases if not R._is_recency_seeking_query(c.query)]
        # Need enough of each side that a flip means something.
        assert len(recency) >= 4, f"only {len(recency)} recency-triggering time cases"
        assert len(non_recency) >= 3, f"only {len(non_recency)} non-recency time cases"

    def test_bank_has_control_cases(self):
        control = [c for c in CASES if not c.needs_time]
        assert len(control) >= 2, "controls confirm retrieval itself isn't broken"

    def test_expected_ids_exist_in_seed_bank(self):
        seed_ids = {s.id for s in SEED_BANK}
        for c in CASES:
            for eid in c.expect_hit:
                assert eid in seed_ids, f"case {c.id} expects unknown seed {eid}"

    def test_no_duplicate_case_ids(self):
        ids = [c.id for c in CASES]
        assert len(ids) == len(set(ids)), "duplicate case ids"

    def test_seed_bank_covers_all_kinds(self):
        # 每个 kind 至少一个用例, 否则分 kind 报告的行永远为空
        kinds = {c.kind for c in CASES}
        assert kinds >= {"point", "range", "update", "relative"}

    def test_adversarial_group_has_enough_coverage(self):
        """v3 起要求求近对抗组 ≥ 8 道 —— 少于这个数, "sweet spot 存在"结论就是
        under-coverage 的假象 (v2 血泪教训: 3 道对抗时看似 +1 净胜, 扩到 10 道
        后立刻翻成净负)."""
        adversarial = [c for c in CASES if c.id.startswith("recency_")]
        assert len(adversarial) >= 8, (
            f"only {len(adversarial)} adversarial recency cases — 权重扫描"
            f"的'甜蜜点未见回退'结论可能只是缺覆盖")


class TestProdPoolShape:
    """prod_pool 模块的形状不变量 —— 具体查询依赖 DB, 只在这里锁 dataclass 与
    _to_dt 的边缘 case (那次 sampler 静默返 0 就是 _to_dt 漏了 str → datetime)."""

    def test_prod_seed_fields_match_candidate_shape(self):
        from evals.temporal_recall.prod_pool import ProdSeed
        # ProdSeed 字段必须能被 run_eval._prod_candidate 消费. 每次改 _prod_candidate
        # 都要同步这里, 否则 pool 加载 200 条 sampler 沉默返 0 那种 bug 会复发.
        required = {
            "id", "text", "main", "sub", "occur_time", "statement_time",
            "source", "importance", "level",
        }
        assert set(ProdSeed.__dataclass_fields__.keys()) == required

    def test_to_dt_handles_iso_string(self):
        # 关键: db.query_raw 把 timestamp 返成 ISO 字符串, 不是 datetime.
        # 之前 sampler 直接 isinstance(x, datetime), 静默丢光整个 pool.
        from datetime import datetime as _dt
        from evals.temporal_recall.prod_pool import _to_dt
        parsed = _to_dt("2026-07-22T20:02:14.336+00:00")
        assert isinstance(parsed, _dt)

    def test_to_dt_passthrough_datetime(self):
        from datetime import datetime as _dt, timezone
        from evals.temporal_recall.prod_pool import _to_dt
        original = _dt.now(timezone.utc)
        assert _to_dt(original) is original

    def test_to_dt_none_on_invalid(self):
        from evals.temporal_recall.prod_pool import _to_dt
        assert _to_dt(None) is None
        assert _to_dt("not a date") is None
        assert _to_dt(12345) is None
