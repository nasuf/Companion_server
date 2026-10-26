import json
from pathlib import Path
import pytest
from evals.context_embedding.measurement import (
    ARK_MODEL,
    ark_usage,
    priced_call,
    recover_ark_trace_cache,
    report_routes,
    search_bill_increment,
    stable_prefix,
)

SAMPLES = json.loads(
    (Path(__file__).parent / "fixtures/ark_raw_usage_20261026.json").read_text()
)


@pytest.mark.parametrize("sample", SAMPLES)
def test_raw_provider_zero_is_measured_and_priced(sample):
    m = ark_usage(sample["usage"], sample["route"])
    assert m["status"] == "measured" and m["cached_input_tokens"] == 0
    p = priced_call(ARK_MODEL, sample["route"], m)
    assert p["status"] == "priced" and p["cost_cny"] > 0
    assert p["search_cost_cny"] == (
        0.004 if sample["route"] == "responses.web_search" else 0
    )


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"prompt_tokens": 77, "completion_tokens": 1},
        {"prompt_tokens": 77, "completion_tokens": 1, "prompt_tokens_details": {}},
        {
            "prompt_tokens": 77,
            "completion_tokens": 1,
            "prompt_tokens_details": {"cached_tokens": None},
        },
    ],
)
def test_missing_is_never_zero(usage):
    m = ark_usage(usage, "chat.completions")
    assert m["status"] != "measured"
    assert priced_call(ARK_MODEL, "chat.completions", m)["cost_cny"] is None


@pytest.mark.parametrize("bad", [-1, True, "0", 1.5, 100])
def test_invalid_cache_values(bad):
    assert (
        ark_usage(
            {
                "prompt_tokens": 77,
                "completion_tokens": 1,
                "prompt_tokens_details": {"cached_tokens": bad},
            },
            "chat.completions",
        )["status"]
        != "measured"
    )


@pytest.mark.parametrize("n,status", [(32000, "priced"), (32001, "price_unknown")])
def test_price_tier_gate(n, status):
    m = ark_usage(
        {
            "prompt_tokens": n,
            "completion_tokens": 1,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
        "chat.completions",
    )
    assert priced_call(ARK_MODEL, "chat.completions", m)["status"] == status
    assert (
        priced_call("ark/unregistered", "chat.completions", m)["status"]
        == "price_unknown"
    )


def test_search_count_and_no_silent_default():
    m = ark_usage(
        {
            "input_tokens": 30000,
            "output_tokens": 70,
            "input_tokens_details": {"cached_tokens": 1000},
            "tool_usage": {"web_search": 3},
        },
        "responses.web_search",
    )
    p = priced_call(ARK_MODEL, "responses.web_search", m)
    assert p["search_cost_cny"] == 0.012
    assert p["cost_cny"] == pytest.approx(
        (29000 * 0.8 + 1000 * 0.16 + 70 * 2) / 1e6 + 0.012
    )
    assert (
        ark_usage(
            {
                "input_tokens": 30000,
                "output_tokens": 70,
                "input_tokens_details": {"cached_tokens": 0},
            },
            "responses.web_search",
        )["status"]
        == "search_usage_missing"
    )
    assert search_bill_increment(3, 19999) == 0.008
    assert search_bill_increment(3, 150) == 0


def test_missing_search_count_does_not_hide_input_tier():
    m = ark_usage(
        {
            "input_tokens": 32001,
            "output_tokens": 2,
            "input_tokens_details": {"cached_tokens": 0},
        },
        "responses.web_search",
    )
    assert m["status"] == "search_usage_missing" and m["input_tokens"] == 32001
    p = priced_call(ARK_MODEL, "responses.web_search", m)
    assert p["status"] == "price_unknown" and p["cost_cny"] is None


def test_raw_trace_zero_recovery_scoped_to_ark():
    row = {
        "model_name": ARK_MODEL,
        "outputs_json": {
            "generations": [
                [
                    {
                        "message": {
                            "kwargs": {
                                "usage_metadata": {
                                    "input_token_details": {"cache_read": 0}
                                }
                            }
                        }
                    }
                ]
            ]
        },
    }
    assert recover_ark_trace_cache(row) == 0
    row["model_name"] = "other/model"
    assert recover_ark_trace_cache(row) is None
    assert recover_ark_trace_cache({"model_name": ARK_MODEL}) is None


def test_route_maximum_and_unknown_are_not_averaged_away():
    calls = []
    for route, n in [
        ("chat.completions", 77),
        ("responses.web_search", 30600),
        ("responses.web_search", 32001),
    ]:
        u = (
            {
                "prompt_tokens": n,
                "completion_tokens": 1,
                "prompt_tokens_details": {"cached_tokens": 0},
            }
            if route == "chat.completions"
            else {
                "input_tokens": n,
                "output_tokens": 1,
                "input_tokens_details": {"cached_tokens": 0},
                "tool_usage": {"web_search": 1},
            }
        )
        m = ark_usage(u, route)
        calls.append(
            dict(
                model=ARK_MODEL,
                route=route,
                measurement=m,
                price=priced_call(ARK_MODEL, route, m),
                ttft_seconds=1,
            )
        )
    reports = report_routes(calls)
    assert len(reports) == 2 and reports[0]["status"] == "qualified"
    assert (
        reports[1]["max_input_tokens"] == 32001
        and reports[1]["status"] == "not_qualified"
    )
    assert reports[1]["normalized_cost_cny"] is None


def test_stable_prefix_is_byte_exact():
    a = [{"role": "system", "content": "稳定段\n"}]
    assert stable_prefix(a) == stable_prefix(
        a + [{"role": "user", "content": "新内容"}]
    )
    assert stable_prefix(a) != stable_prefix([{"role": "system", "content": "稳定段"}])
    assert stable_prefix([{"role": "user", "content": "消息"}]) is None


def test_corpus_category_minimum_and_unique_scenarios():
    from collections import Counter
    from evals.context_embedding.build_cases import cases, CATEGORIES

    corpus = cases()
    counts = Counter(c["category"] for c in corpus)
    assert set(CATEGORIES) <= set(counts)
    assert all(n >= 8 for n in counts.values())
    assert len(corpus) == len({c["id"] for c in corpus}) == 104
    assert all(
        len(c["turns"]) >= 2 and all(t["criterion"] for t in c["turns"]) for c in corpus
    )
    assert any(c["route"] == "web_search" for c in corpus)
