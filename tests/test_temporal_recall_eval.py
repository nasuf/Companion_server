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
