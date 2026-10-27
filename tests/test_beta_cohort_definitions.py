"""UXM01.01 口径定义的失败与边界: 未知人群、查询失败、人群指纹不可用时的缓存."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.effect import definitions, signals, store
from app.services.effect.signals import EffectMetrics


def test_unknown_cohort_fails_instead_of_silently_counting_everyone():
    for build in (
        lambda: definitions.population_sql("c.user_id", "staff"),
        lambda: definitions.user_row_population_sql("u", "staff"),
        lambda: definitions.conversation_population_sql("m.conversation_id", "staff"),
        lambda: definitions.chat_activity_sql("staff"),
    ):
        with pytest.raises(ValueError):
            build()


def test_ratio_states_never_turn_unknown_into_zero():
    assert definitions.ratio(0, 0)["state"] == "no_sample"
    assert definitions.ratio(3, 7)["state"] == "insufficient_sample"
    assert definitions.ratio(3, 7)["value"] is None
    assert definitions.ratio(5, 10, settled=False)["state"] == "unsettled"
    real_zero = definitions.ratio(0, 40)
    assert real_zero["state"] == "ok" and real_zero["value"] == 0.0


def test_describe_defaults_to_the_whole_dictionary_and_local_today_uses_utc8():
    described = definitions.describe()
    assert set(described["metrics"]) == set(definitions.METRICS)
    assert described["definition_version"] == definitions.DEFINITION_VERSION
    # UTC 2026-10-27 17:00 = 北京时间 10-28 01:00
    assert definitions.local_today(datetime(2026, 10, 27, 17, tzinfo=timezone.utc)) == date(2026, 10, 28)
    assert isinstance(definitions.local_today(), date)


async def test_population_fingerprint_handles_an_empty_result():
    fake = MagicMock(query_raw=AsyncMock(return_value=[]))
    with patch("app.db.db", fake):
        assert await definitions.population_fingerprint() == "none"


async def test_failed_queries_are_reported_and_summarised_as_failed_not_zero():
    fake = MagicMock(query_raw=AsyncMock(side_effect=RuntimeError("db down")))
    with patch("app.db.db", fake):
        day = (await signals.collect(date(2026, 10, 20),
                                     now=datetime(2026, 10, 25, tzinfo=timezone.utc))).as_dict()
    assert day["failed"] == ["core", "slices", "sessions", "proactive", "retention"]
    ratios = store.summarise([day])["ratios"]
    assert ratios["continuation"]["state"] == "failed"
    assert ratios["proactive_response"]["state"] == "failed"
    assert ratios["continuation"]["value"] is None


class _Redis:
    def __init__(self):
        self.data, self.ttls = {}, {}

    async def get(self, key):
        return self.data.get(key)

    async def set(self, key, value, ex=None):
        self.data[key], self.ttls[key] = value, ex


async def test_cache_is_bypassed_when_the_population_is_unknown():
    redis = _Redis()
    with (
        patch.object(store, "population_fingerprint", AsyncMock(side_effect=RuntimeError("db down"))),
        patch("app.redis_client.get_redis", AsyncMock(return_value=redis)) as get_redis,
        patch.object(store, "collect", AsyncMock(return_value=EffectMetrics(date="2026-10-01"))),
    ):
        payload = await store.get_day(date(2026, 10, 1))
    assert payload["date"] == "2026-10-01"
    get_redis.assert_not_called()  # 不能把未知人群的结果写进任何人群的缓存
    assert redis.data == {}


async def test_settled_day_is_cached_long_but_a_failed_day_only_briefly():
    redis = _Redis()
    old = date.today() - timedelta(days=10)
    ok = EffectMetrics(date=old.isoformat(), turns=3)
    broken = EffectMetrics(date=(old - timedelta(days=1)).isoformat(), failed=["core"])
    with (
        patch.object(store, "population_fingerprint", AsyncMock(return_value="fp")),
        patch("app.redis_client.get_redis", AsyncMock(return_value=redis)),
        patch.object(store, "collect", AsyncMock(side_effect=[ok, broken])) as collect,
    ):
        await store.get_day(old)
        cached = await store.get_day(old)  # 第二次命中缓存, 不再计算
        await store.get_day(old - timedelta(days=1))
    assert collect.await_count == 2 and cached["turns"] == 3
    ttls = {json.loads(redis.data[k])["date"]: redis.ttls[k] for k in redis.data}
    assert ttls[old.isoformat()] == store._TTL_SETTLED_S
    assert ttls[(old - timedelta(days=1)).isoformat()] == store._TTL_PENDING_S
    assert all(f":{definitions.DEFINITION_VERSION}:external:fp:" in k for k in redis.data)


async def test_range_reads_the_population_fingerprint_once_for_all_days():
    redis = _Redis()
    fingerprint = AsyncMock(return_value="fp")
    with (
        patch.object(store, "population_fingerprint", fingerprint),
        patch("app.redis_client.get_redis", AsyncMock(return_value=redis)),
        patch.object(store, "collect", AsyncMock(side_effect=lambda day, **_: EffectMetrics(date=day.isoformat()))),
    ):
        result = await store.get_range(3)
    dates = [d["date"] for d in result.days]
    assert dates == sorted(dates) and len(dates) == 3
    assert fingerprint.await_count == 1  # 一段区间只查一次内部账号集合
    assert all(":external:fp:" in key for key in redis.data)


async def test_forced_refresh_recomputes_even_when_cached():
    redis = _Redis()
    old = date.today() - timedelta(days=10)
    with (
        patch.object(store, "population_fingerprint", AsyncMock(return_value="fp")),
        patch("app.redis_client.get_redis", AsyncMock(return_value=redis)),
        patch.object(store, "collect", AsyncMock(side_effect=[
            EffectMetrics(date=old.isoformat(), turns=1), EffectMetrics(date=old.isoformat(), turns=2),
        ])) as collect,
    ):
        await store.get_day(old)
        refreshed = await store.get_day(old, force=True)
    assert collect.await_count == 2 and refreshed["turns"] == 2


async def test_reply_effect_endpoint_accepts_only_known_cohorts(api_client):
    from app.api.jwt_auth import require_admin_jwt
    from app.main import app

    app.dependency_overrides[require_admin_jwt] = lambda: {"role": "admin"}
    try:
        with patch("app.services.effect.store.get_range", AsyncMock(
            return_value=store.EffectRange(days=[EffectMetrics(date="2026-10-01", cohort="all").as_dict()]),
        )) as get_range:
            ok = api_client.get("/admin-api/stats/reply-effect", params={"days": 3, "cohort": "all"})
            bad = api_client.get("/admin-api/stats/reply-effect", params={"cohort": "staff"})
        assert ok.status_code == 200 and bad.status_code == 422
        body = ok.json()
        assert body["cohort"] == "all"
        assert body["definitions"]["definition_version"] == definitions.DEFINITION_VERSION
        assert set(body["definitions"]["metrics"]) == {"continuation", "proactive_response", "next_day_return"}
        assert body["summary"]["ratios"]["continuation"]["state"] == "no_sample"
        assert get_range.await_args.kwargs == {"cohort": "all"} and get_range.await_args.args == (3,)
    finally:
        app.dependency_overrides.pop(require_admin_jwt, None)


async def test_daily_refresh_warms_both_cohorts():
    with (
        patch.object(store, "population_fingerprint", AsyncMock(return_value="fp")),
        patch.object(store, "get_day", AsyncMock(return_value={})) as get_day,
    ):
        await store.refresh_recent(days=2)
    cohorts = [call.kwargs["cohort"] for call in get_day.await_args_list]
    assert cohorts == ["external", "external", "all", "all"]
    assert {call.kwargs["fingerprint"] for call in get_day.await_args_list} == {"fp", "-"}
