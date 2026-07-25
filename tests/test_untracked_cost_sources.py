"""Costs that never reach the token-based pricing path.

Three spend sources used to be invisible in the admin dashboard because they
bypass `estimate_cost_cny`:

1. Ark 联网内容插件 — billed per search call against a monthly free quota.
2. 豆包视觉理解 — token-billed, but issued over raw httpx outside the
   langchain wrapper that normally records usage.
3. Fun-ASR — billed by audio duration, not tokens.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from app.api.admin import stats as stats_mod
from app.services.llm import usage_tracker


# --------------------------------------------------------------------------
# 1. Web search plugin — per-call billing
# --------------------------------------------------------------------------


def test_record_web_search_accumulates_into_session():
    token = usage_tracker.start_session()
    try:
        usage_tracker.record_web_search(2)
        usage_tracker.record_web_search(3)
    finally:
        summary = usage_tracker.flush_session(token)
    assert summary is not None
    assert summary["web_search_calls"] == 5


def test_record_web_search_ignores_non_positive_and_no_session():
    usage_tracker.record_web_search(1)  # no active session — must not raise

    token = usage_tracker.start_session()
    try:
        usage_tracker.record_web_search(0)
        usage_tracker.record_web_search(-4)
    finally:
        summary = usage_tracker.flush_session(token)
    # Nothing recorded at all → session is dropped rather than writing a zero row.
    assert summary is None


def test_search_only_session_still_persists():
    """A search that returns no usage block must not lose its billable call."""
    token = usage_tracker.start_session()
    try:
        usage_tracker.record_web_search(1)
    finally:
        summary = usage_tracker.flush_session(token)
    assert summary is not None
    assert summary["call_count"] == 0
    assert summary["web_search_calls"] == 1


@pytest.mark.asyncio
async def test_web_search_billing_free_quota_not_exceeded(monkeypatch):
    monkeypatch.setattr(stats_mod.settings, "web_search_free_calls_monthly", 20000)
    monkeypatch.setattr(stats_mod.settings, "web_search_price_cny_per_k", 4.0)
    monkeypatch.setattr(
        stats_mod, "db", SimpleNamespace(query_raw=AsyncMock(return_value=[{"calls": 1500}]))
    )

    result = await stats_mod._web_search_billing(120)

    assert result["window_calls"] == 120
    assert result["month_calls"] == 1500
    assert result["free_remaining"] == 18500
    assert result["billable_calls"] == 0
    assert result["cost_cny"] == 0.0


@pytest.mark.asyncio
async def test_web_search_billing_charges_overage_only(monkeypatch):
    monkeypatch.setattr(stats_mod.settings, "web_search_free_calls_monthly", 20000)
    monkeypatch.setattr(stats_mod.settings, "web_search_price_cny_per_k", 4.0)
    monkeypatch.setattr(
        stats_mod, "db", SimpleNamespace(query_raw=AsyncMock(return_value=[{"calls": 23500}]))
    )

    result = await stats_mod._web_search_billing(900)

    assert result["billable_calls"] == 3500
    assert result["free_remaining"] == 0
    assert result["cost_cny"] == pytest.approx(14.0)


@pytest.mark.asyncio
async def test_web_search_billing_month_boundary_uses_billing_timezone(monkeypatch):
    """账单月是北京时间的自然月, created_at 存 UTC.

    直接取 UTC 月初会把 1 号 00:00-08:00 (北京) 的调用漏算进上个月的额度 —
    月初边界必须先在业务时区取, 再换回 UTC 去比 created_at.
    """
    captured: list = []

    async def fake_query_raw(_sql, *params):
        captured.append(params)
        return [{"calls": 0}]

    monkeypatch.setattr(stats_mod, "db", SimpleNamespace(query_raw=fake_query_raw))

    await stats_mod._web_search_billing(0)

    bound = datetime.fromisoformat(captured[0][0])
    local_month_start = datetime.now(timezone.utc).astimezone(
        ZoneInfo(stats_mod.settings.schedule_timezone)
    ).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    assert bound == local_month_start.astimezone(timezone.utc).replace(tzinfo=None)
    # Asia/Shanghai is UTC+8, so the UTC instant lands on the previous month.
    assert bound.hour == 16
    assert bound.tzinfo is None  # naive UTC, matching how created_at is stored


@pytest.mark.asyncio
async def test_web_search_billing_survives_query_failure(monkeypatch):
    """Billing extras must never take the whole dashboard down."""
    monkeypatch.setattr(
        stats_mod, "db", SimpleNamespace(query_raw=AsyncMock(side_effect=RuntimeError("boom")))
    )
    assert await stats_mod._web_search_billing(5) == {}


# --------------------------------------------------------------------------
# 2. Vision — token billing over raw httpx
# --------------------------------------------------------------------------


def test_vision_usage_recorded_with_cached_tokens():
    from app.services.chat_media import vision

    token = usage_tracker.start_session()
    try:
        vision._record_vision_usage(
            "doubao-1-5-vision-pro-32k-250115",
            {
                "usage": {
                    "prompt_tokens": 1200,
                    "completion_tokens": 80,
                    "prompt_tokens_details": {"cached_tokens": 400},
                }
            },
        )
    finally:
        summary = usage_tracker.flush_session(token)

    assert summary is not None
    bucket = summary["tokens_by_model"]["ark/doubao-1-5-vision-pro-32k-250115"]
    assert bucket == {"input": 1200, "output": 80, "cached_input": 400}
    assert summary["call_count"] == 1


def test_vision_usage_tolerates_missing_usage_block():
    from app.services.chat_media import vision

    token = usage_tracker.start_session()
    try:
        vision._record_vision_usage("m", {"choices": []})
        vision._record_vision_usage("m", None)
    finally:
        assert usage_tracker.flush_session(token) is None


# --------------------------------------------------------------------------
# 3. ASR — duration billing
# --------------------------------------------------------------------------


def test_asr_billing_multiplies_seconds_by_unit_price(monkeypatch):
    monkeypatch.setattr(stats_mod.settings, "asr_price_cny_per_second", 0.00022)
    result = stats_mod._asr_billing({"count": 12, "total_seconds": 3600})
    assert result["count"] == 12
    assert result["total_seconds"] == 3600
    assert result["cost_cny"] == pytest.approx(0.792)


def test_asr_billing_reports_unpriced_instead_of_zero(monkeypatch):
    """An unset rate must read as 未配置, never as free."""
    monkeypatch.setattr(stats_mod.settings, "asr_price_cny_per_second", 0.0)
    result = stats_mod._asr_billing({"count": 3, "total_seconds": 90})
    assert result["cost_cny"] is None
    assert result["total_seconds"] == 90
