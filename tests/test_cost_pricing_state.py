"""Recorded fees stay priced; only fees never recorded are unpriced; caches and traces stay linked."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services.effect import costs

PRICE = {"input_cost_per_million": 1.0, "output_cost_per_million": 2.0,
         "cached_input_cost_per_million": None}
PRICES = {"deepseek/deepseek-v4-flash": PRICE}


def bucket(i=1000, o=100, **more):
    return {"input": i, "output": o, "cached_input": 0, **more}


def test_recorded_fee_of_removed_model_stays_priced():
    # 2026-08-24: ark/doubao-1-5-vision-pro-32k-250115 was billed, then removed.
    assert costs.unpriced_models({"ark/vision-removed": bucket()}, 0.0047, PRICES) == []


def test_zero_fee_is_unpriced_only_when_a_price_would_have_charged():
    assert costs.unpriced_models({"ark/vision-removed": bucket()}, 0, PRICES) == ["ark/vision-removed"]
    # Registered later (deepseek/ added 2026-05-26 with NULL prices): the 0 was never priced.
    assert costs.unpriced_models({"deepseek/deepseek-v4-flash": bucket()}, 0, PRICES) == [
        "deepseek/deepseek-v4-flash"]
    # Below 0.000001 the stored 6-decimal amount is legitimately 0.
    assert costs.unpriced_models({"deepseek/deepseek-v4-flash": bucket(i=0, o=0)}, 0, PRICES) == []
    # 1 token at 1 CNY/M is 0.000001: storable, so a 0 means it was never priced.
    assert costs.unpriced_models({"deepseek/deepseek-v4-flash": bucket(i=1, o=0)}, 0, PRICES) == [
        "deepseek/deepseek-v4-flash"]
    cheap = {"m/cheap": {**PRICE, "input_cost_per_million": 0.2}}
    assert costs.unpriced_models({"m/cheap": bucket(i=1, o=0)}, 0, cheap) == []  # 2e-7 rounds to 0


def test_write_time_flags_win_and_ignore_zero_token_buckets():
    models = {
        "ark/known": bucket(priced=True),
        "ark/unknown": bucket(priced=False),
        "ark/attempt": {"input": 0, "output": 0, "cached_input": 0, "unmetered_calls": 1, "priced": False},
    }
    # A multi-model row with a recorded fee still exposes the one model it could not price.
    assert costs.unpriced_models(models, 0.5, {}) == ["ark/unknown"]
    assert costs.unpriced_models({"ark/known": bucket(priced=True)}, 0, {}) == []


def test_accumulate_keeps_per_model_unpriced():
    total = costs.blank()
    costs.accumulate(total, {"models": {"a": {**bucket(), "cost_cny": 0.1, "unpriced": False}}})
    costs.accumulate(total, {"models": {"a": {**bucket(), "cost_cny": 0, "unpriced": True}}})
    assert total["models"]["a"]["unpriced"] is True
    assert total["models"]["a"]["input"] == 2000 and total["models"]["a"]["cost_cny"] == 0.1


async def test_usage_row_records_whether_each_model_had_a_price(monkeypatch):
    from app.services.llm import usage_repo

    create = AsyncMock()
    monkeypatch.setattr(usage_repo, "db", SimpleNamespace(llmusage=SimpleNamespace(create=create)))
    prices = {"deepseek/deepseek-v4-flash": {"input": 1.0, "output": 2.0, "cached_input": 1.0}}
    summary = {
        "tokens_by_model": {"deepseek/deepseek-v4-flash": bucket(), "ark/unregistered": bucket()},
        "input_tokens": 2000, "output_tokens": 200, "cached_input_tokens": 0, "call_count": 2,
        "web_search_calls": 0, "latency_ms_total": 0, "latency_count": 0, "failure_count": 0,
        "fallback_count": 0, "circuit_open_count": 0,
    }
    with patch("app.services.runtime_config.get_pricing", side_effect=prices.get):
        await usage_repo.write_usage_row(summary=summary, conversation_id="c", agent_id="a",
                                         user_id="u", trace_id="t", scope="chat")
    data = create.await_args.kwargs["data"]
    stored = data["tokensByModel"].data
    assert stored["deepseek/deepseek-v4-flash"]["priced"] is True
    assert stored["ark/unregistered"]["priced"] is False
    assert data["costCny"] == pytest.approx(0.0012)
    assert "priced" not in summary["tokens_by_model"]["ark/unregistered"]  # caller's summary untouched


def test_invalidate_caches_clears_every_model_factory(monkeypatch):
    from app.services import runtime_config
    from app.services.llm import models

    cleared = []
    names = ("get_chat_model", "get_utility_model", "get_grounding_model",
             "get_fallback_chat_model", "get_fallback_utility_model")
    for name in names:
        def clear(name=name):
            cleared.append(name)
            if name == "get_chat_model":
                raise RuntimeError("synthetic")  # one failure must not skip the rest
        monkeypatch.setattr(getattr(models, name), "cache_clear", clear)
    runtime_config.invalidate_caches()
    assert cleared == list(names)


def test_grounding_and_fallback_utility_factories_expose_cache_clear():
    from app.services.llm import models

    assert models.get_grounding_model.cache_clear == models._build_grounding_model.cache_clear
    assert models.get_fallback_utility_model.cache_clear == models._build_fallback_utility_model.cache_clear
