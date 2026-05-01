"""Spec Part 5 §2.1 + §2.2 + §3.2 时间中枢单测.

覆盖: getCurrentTimestamp / TimeInfo.timestamp_ms / NTP drift 真应用 /
resolve_implicit_time helper.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from app.services.schedule_domain import time_service
from app.services.schedule_domain.time_service import (
    _TZ,
    get_current_time,
    get_current_timestamp_ms,
    resolve_implicit_time,
)


# ── P1-1: spec §2.2 API ────────────────────────────────────────────


def test_get_current_timestamp_ms_returns_int_13_digits():
    """spec §2.2 getCurrentTimestamp(): Unix 毫秒级整数 (13 位 in 21st century)."""
    ts = get_current_timestamp_ms()
    assert isinstance(ts, int)
    # 2001-09-09 ~ 2286 范围内 13 位 (足够覆盖)
    assert 10**12 < ts < 10**13


def test_get_current_time_includes_timestamp_ms():
    """TimeInfo 含 timestamp_ms 字段, 跟 now 字段一致."""
    info = get_current_time()
    expected = int(info.now.timestamp() * 1000)
    # 允许 1ms 浮动 (两次 datetime.now 间隔)
    assert abs(info.timestamp_ms - expected) <= 1


# ── P1-2: NTP drift 真修正 ─────────────────────────────────────────


def test_ntp_drift_applied_to_get_current_time():
    """模拟 _NTP_DRIFT_SECONDS=2.0, get_current_time().now 应比墙钟快 2 秒."""
    real_now = datetime.now(_TZ)
    with patch.object(time_service, "_NTP_DRIFT_SECONDS", 2.0):
        corrected = get_current_time().now
    diff = (corrected - real_now).total_seconds()
    # 应该 ~= 2.0 (允许 0.1s 调度抖动)
    assert 1.9 < diff < 2.1, f"expected ~2s drift, got {diff:.3f}s"


def test_ntp_drift_applied_to_get_current_timestamp_ms():
    """模拟 drift=5s, getCurrentTimestamp 也跟着 +5000ms."""
    baseline = get_current_timestamp_ms()
    with patch.object(time_service, "_NTP_DRIFT_SECONDS", 5.0):
        with_drift = get_current_timestamp_ms()
    diff_ms = with_drift - baseline
    # 应在 4900-5100ms 之间 (调度抖动)
    assert 4900 < diff_ms < 5100, f"expected ~5000ms drift, got {diff_ms}ms"


# ── P1-4: spec §3.2 隐性时间解析 helper ────────────────────────────


@pytest.mark.asyncio
async def test_resolve_implicit_time_uses_provided_ai_status():
    """传入 ai_status (避免重复 fetch) → 不走 get_cached_schedule."""
    ai_status = {"activity": "散步", "status": "idle", "type": "leisure"}
    with patch(
        "app.services.schedule_domain.schedule.get_cached_schedule",
        new_callable=AsyncMock,
    ) as mock_load:
        now, activity = await resolve_implicit_time("agent1", ai_status)
    mock_load.assert_not_awaited()
    assert isinstance(now, datetime)
    assert "散步" in activity


@pytest.mark.asyncio
async def test_resolve_implicit_time_loads_when_no_ai_status():
    """ai_status=None → 走 get_cached_schedule + get_current_status."""
    with (
        patch(
            "app.services.schedule_domain.schedule.get_cached_schedule",
            new_callable=AsyncMock, return_value=[{"start": "00:00", "end": "23:59", "activity": "睡眠", "status": "sleep"}],
        ),
        patch(
            "app.services.schedule_domain.schedule.get_current_status",
            return_value={"activity": "睡眠", "status": "sleep", "type": "sleep"},
        ),
    ):
        now, activity = await resolve_implicit_time("agent1")
    assert isinstance(now, datetime)
    assert "睡眠" in activity


@pytest.mark.asyncio
async def test_resolve_implicit_time_handles_missing_schedule():
    """无 schedule → 返回 "(未知)"."""
    with patch(
        "app.services.schedule_domain.schedule.get_cached_schedule",
        new_callable=AsyncMock, return_value=None,
    ):
        now, activity = await resolve_implicit_time("agent1")
    assert activity == "(未知)"
