"""Offline checks of the repetition-gate metering proxy and controller, without keys or network."""

import json
from pathlib import Path

import httpx
import pytest

from app.services.quality.budget import BudgetExhausted
from evals.reply_repetition import gate, meter

ROUTE = ["ark", "doubao-seed-character-260628"]
PRICE = {"input": 0.8, "output": 2, "cached_input": 0.16}
PRICES = {"ark/doubao-seed-character-260628": PRICE, "dashscope/qwen3.5-plus": PRICE,
          "dashscope/qwen3.5-flash": PRICE, "dashscope/text-embedding-v4": PRICE}


class FakeControl:
    def __init__(self, refuse=()):
        self.refuse = set(refuse)
        self.reserved, self.settled, self.uncertain_ids = [], {}, []
        self.fail_settle = False

    async def reserve_block(self, model, amount):
        if amount in self.refuse:
            raise BudgetExhausted("monthly")
        self.reserved.append((model, amount))
        return f"b{len(self.reserved)}"

    async def settle_block(self, charge_id, usage):
        if self.fail_settle:
            raise RuntimeError("down")
        self.settled[charge_id] = usage

    async def uncertain(self, charge_id):
        self.uncertain_ids.append(charge_id)


def upstream(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def chat(prompt=100, completion=10, cached=0, status=200):
    def handler(request):
        assert request.headers["Authorization"] == "Bearer real-ark"
        body = {"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                                         "prompt_tokens_details": {"cached_tokens": cached}}}
        return httpx.Response(status, json=body)
    return handler


def make(handler, control=None):
    control = control or FakeControl()
    m = meter.Meter(control, PRICES, {"ark": "https://ark.invalid/v3/", "dashscope": "https://ds.invalid/v1"},
                    {"ark": "real-ark", "dashscope": "real-ds"}, ROUTE, client=upstream(handler))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=m.app), base_url="http://proxy")
    return m, control, client


async def post(client, body, path="/ark/chat/completions"):
    return await client.post(path, content=json.dumps(body))


BODY = {"model": ROUTE[1], "messages": [{"role": "user", "content": "合成"}], "max_tokens": 100}


async def test_billed_calls_share_a_block_and_settle_on_close():
    m, control, client = make(chat())
    for _ in range(3):
        assert (await post(client, BODY)).status_code == 200
    await m.close()
    assert control.reserved == [("ark/doubao-seed-character-260628", meter.BLOCK_STEPS[0])]
    assert control.settled["b1"] == dict(input=300, output=30, cached_input=0, calls=3, unknown_cny=0.0)
    summary = m.summary()
    assert summary["models"]["ark/doubao-seed-character-260628"]["calls"] == 3
    assert summary["cost_cny"] == pytest.approx(3 * (100 * 0.8 + 10 * 2) / 1e6, abs=1e-4)


@pytest.mark.parametrize("body,path,code", [
    ({**BODY, "model": "other"}, "/ark/chat/completions", 403),
    ({**BODY, "stream": True}, "/ark/chat/completions", 400),
    ({**BODY, "max_tokens": 5000}, "/ark/chat/completions", 400),
    ({**BODY, "max_tokens": 1.5}, "/ark/chat/completions", 400),
    ([], "/ark/chat/completions", 403),
    (BODY, "/ark/images", 403),
])
async def test_requests_outside_the_policy_are_refused_before_any_reservation(body, path, code):
    m, control, client = make(chat())
    assert (await post(client, body, path)).status_code == code
    assert (await client.post("/ark/chat/completions", content=b"{")).status_code == 400
    assert (await client.post("/ark/chat/completions", content=b"x" * (meter.MAX_BODY + 1))).status_code == 413
    await m.close()
    assert control.reserved == []


async def test_ambiguous_outcomes_are_charged_and_rejections_are_free():
    calls = iter([httpx.Response(429, json={}), httpx.Response(500, json={}),
                  httpx.Response(200, json={"usage": {"prompt_tokens": 10**9, "completion_tokens": 1}})])
    m, control, client = make(lambda request: next(calls))
    assert (await post(client, BODY)).status_code == 429
    assert (await post(client, BODY)).status_code == 500
    assert (await post(client, BODY)).status_code == 200  # implausible usage: worst case
    await m.close()
    stats = m.summary()["models"]["ark/doubao-seed-character-260628"]
    assert stats["rejected"] == 1 and stats["unknown_calls"] == 2
    usage = control.settled["b1"]
    assert usage["calls"] == 2 and usage["input"] == 0 and usage["unknown_cny"] > 0


async def test_transport_failure_is_502_and_charged():
    def broken(request):
        raise httpx.ConnectError("down")
    m, control, client = make(broken)
    assert (await post(client, BODY)).status_code == 502
    await m.close()
    assert control.settled["b1"]["unknown_cny"] > 0


async def test_full_block_rolls_over_and_falls_back_to_smaller_blocks():
    control = FakeControl(refuse={meter.BLOCK_STEPS[0]})
    m, control, client = make(chat(), control)
    assert (await post(client, BODY)).status_code == 200
    block = m.blocks["ark/doubao-seed-character-260628"]
    assert block.amount == float(meter.BLOCK_STEPS[1])
    block.committed = block.amount  # next call no longer fits
    assert (await post(client, BODY)).status_code == 200
    await m.close()
    assert [a for _, a in control.reserved] == [meter.BLOCK_STEPS[1]] * 2
    assert set(control.settled) == {"b1", "b2"}


async def test_exhausted_budget_answers_402_without_upstream_calls():
    control = FakeControl(refuse=set(meter.BLOCK_STEPS))
    seen = []
    m, control, client = make(lambda r: seen.append(r) or httpx.Response(200, json={}), control)
    assert (await post(client, BODY)).status_code == 402
    assert m.exhausted and (await post(client, BODY)).status_code == 402
    await m.close()
    assert seen == []


async def test_control_outage_is_503_and_failed_settlement_marks_uncertain():
    class Down(FakeControl):
        async def reserve_block(self, model, amount):
            raise RuntimeError("down")
    m, _, client = make(chat(), Down())
    assert (await post(client, BODY)).status_code == 503
    await m.close()
    assert m.errors == ["reserve_RuntimeError"]

    m, control, client = make(chat())
    control.fail_settle = True
    assert (await post(client, BODY)).status_code == 200
    await m.close()
    assert control.uncertain_ids == ["b1"] and m.errors == ["settle_RuntimeError"]


def test_embedding_usage_and_bounds():
    response = httpx.Response(200, json={"usage": {"total_tokens": 7}})
    assert meter.usage("embeddings", response, 100, 0) == dict(input=7, output=0, cached_input=0)
    assert meter.usage("embeddings", httpx.Response(200, text="x"), 100, 0) is None
    bad = httpx.Response(200, json={"usage": {"prompt_tokens": "7", "completion_tokens": 1}})
    assert meter.usage("chat/completions", bad, 100, 10) is None


async def test_serve_binds_loopback_and_stops():
    m, _, _ = make(chat())
    port, stop = await meter.serve(m)
    async with httpx.AsyncClient(trust_env=False) as client:
        response = await client.post(f"http://127.0.0.1:{port}/ark/chat/completions", content=b"{")
    await stop()
    await m.close()
    assert response.status_code == 400


def test_arms_alternate_and_prompts_follow_code_unless_web_managed():
    assert gate.arms("gate", 2) == [("base", "base-1"), ("candidate", "cand-1"),
                                    ("base", "base-2"), ("candidate", "cand-2")]
    assert gate.arms("calibrate", 2) == [("base", "base-1"), ("base", "base-2")]
    manifest = {"protected_prompts": ["a"], "prompts": [{"key": "a", "content": "web"},
                                                        {"key": "b", "content": "prod"}]}
    assert [p["content"] for p in gate.prompt_rows(manifest)] == ["web", None]


def test_code_prompts_use_this_tree_defaults():
    from app.services.prompting.registry import PROMPT_DEFINITION_MAP
    from evals.reply_repetition.run_replay import code_prompts

    key = next(iter(PROMPT_DEFINITION_MAP))
    rows = code_prompts([{"key": key, "content": None}, {"key": key, "content": "web"},
                         {"key": "removed-from-code", "content": None}])
    assert [r["content"] for r in rows] == [PROMPT_DEFINITION_MAP[key].default_text, "web"]


def test_keys_from_files_are_deleted_and_children_get_none(tmp_path, monkeypatch):
    (tmp_path / "ark").write_text("real-ark\n")
    monkeypatch.setenv("EVAL_KEYS_DIR", str(tmp_path))
    monkeypatch.setenv("EVAL_DASHSCOPE_API_KEY", "real-ds")
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "oidc")
    assert gate.read_keys() == {"ark": "real-ark", "dashscope": "real-ds"}
    assert not (tmp_path / "ark").exists()
    env = gate.child_env(1234)
    assert "real" not in json.dumps(env) and "ACTIONS_ID_TOKEN_REQUEST_TOKEN" not in env
    assert env["ARK_BASE_URL"] == "http://127.0.0.1:1234/ark"


def test_candidate_and_calibration_must_match(tmp_path):
    (tmp_path / "evals/reply_repetition").mkdir(parents=True)
    (tmp_path / "evals/reply_repetition/run_replay.py").write_text("old")
    with pytest.raises(RuntimeError, match="predates"):
        gate.check_candidate(tmp_path)
    gate.check_candidate(Path(gate.__file__).resolve().parents[2])
    good = {"case_set": "s", "reply_model": ROUTE[1], "samples_per_run": 5, "runs_per_arm": 8}
    assert gate.check_calibration(good, "s", ROUTE) == 8
    for change, reason in (({"case_set": "x"}, "case_set"), ({"reply_model": "x"}, "reply_model"),
                           ({"samples_per_run": 3}, "sample_count")):
        with pytest.raises(RuntimeError, match=reason):
            gate.check_calibration({**good, **change}, "s", ROUTE)


def test_public_log_has_numbers_only():
    report = {"mode": "gate", "spend": {"cost_cny": 1.5}, "replays": [{"arm": "base-1", "done": 9, "failed": 0,
                                                                       "exit_codes": [0]}],
              "gate": {"passed": True, "checks": {"x": True}},
              "paired_diff": {"positive_repeated": {"diff": -0.1, "ci95": [-0.2, 0], "cases": ["secret"]}}}
    out = gate.public("completed", report)
    assert "secret" not in json.dumps(out) and out["gate_passed"] and out["cost_cny"] == 1.5
