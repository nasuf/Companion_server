"""Adversarial identity, metering, gates and corpus coverage checks."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from app.services.quality import oidc, gates, transport, judge


def test_oidc_rejects_other_repo_ref_workflow_and_unsigned(monkeypatch):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    monkeypatch.setattr(
        oidc._jwks,
        "get_signing_key_from_jwt",
        lambda _: SimpleNamespace(key=key.public_key()),
    )
    now = datetime.now(UTC)
    claims = dict(
        iss=oidc.ISSUER,
        aud=oidc.AUDIENCE,
        sub="repo:nasuf/Companion_server:ref:refs/heads/main",
        repository="nasuf/Companion_server",
        repository_id="1180603795",
        ref="refs/heads/main",
        sha="a" * 40,
        run_id="123",
        run_attempt="1",
        workflow_ref="nasuf/Companion_server/.github/workflows/quality-nightly.yml@refs/heads/main",
        event_name="schedule",
        iat=now,
        nbf=now,
        exp=now + timedelta(minutes=5),
    )
    assert oidc.verify(jwt.encode(claims, key, algorithm="RS256"))["run_id"] == "123"
    for change in [
        dict(repository="attacker/fork"),
        dict(repository_id="123"),
        dict(ref="refs/pull/4/merge"),
        dict(event_name="pull_request_target"),
        dict(
            workflow_ref="nasuf/Companion_server/.github/workflows/ci.yml@refs/heads/main"
        ),
        dict(aud="other"),
        dict(exp=now - timedelta(minutes=1)),
        dict(sub="repo:nasuf/Companion_server:environment:prod"),
    ]:
        with pytest.raises((ValueError, jwt.PyJWTError)):
            oidc.verify(jwt.encode({**claims, **change}, key, algorithm="RS256"))
    with pytest.raises(jwt.PyJWTError):
        oidc.verify(jwt.encode(claims, key=None, algorithm="none"))


def test_gate_baseline_missing_incomplete_critical_and_noise():
    rows = [
        dict(case_id=str(i), repetition=0, critical=i < 3, **{"pass": True})
        for i in range(10)
    ]
    assert gates.compare(rows, [])["status"] == "baseline_recorded"
    assert gates.compare(rows, rows[:-1])["status"] == "incomplete"
    changed = [{**r, "pass": False if r["case_id"] == "0" else True} for r in rows]
    assert gates.compare(changed, rows)["status"] == "failed"
    changed = [
        {**r, "pass": False if r["case_id"] in {"8", "9"} else True} for r in rows
    ]
    assert gates.compare(changed, rows)["status"] == "passed"
    changed[-3]["pass"] = False
    assert gates.compare(changed, rows)["status"] == "failed"
    changed[-3]["pass"] = None
    assert gates.compare(changed, rows)["status"] == "incomplete"


def test_temporal_dialogue_replay_matches_authored_harness():
    from evals.quality.corpus import load, TEMPORAL_DIALOGUE_JUDGING

    cases = [c for c in load() if c["suite"] == "temporal_dialogue"]
    assert len(cases) == 240
    for c in cases:
        assert c["judging_instruction"] == TEMPORAL_DIALOGUE_JUDGING
        statements = [m for m in c["history_rows"] if m["role"] == "user"]
        if c["category"] == "gap":
            assert c["memories"] == []
        elif all(m["id"].startswith("statement-") for m in c["memories"]):
            assert [m["content"] for m in c["memories"]] == [s["content"] for s in statements]
            assert all(
                m["occur_time"] is None and m["statement_time"] == s["createdAt"]
                for m, s in zip(c["memories"], statements)
            )
    authored = [c for c in cases if c["memories"] and not c["memories"][0]["id"].startswith("statement-")]
    assert len(authored) == 2


def test_corpus_provenance_and_quick_all_suites():
    from evals.quality.corpus import load

    full = load()
    quick = load(quick=True)
    assert len(quick) == 30 and len({c["id"] for c in full}) == len(full)
    assert {c["suite"] for c in quick} == {c["suite"] for c in full}
    for c in full:
        assert isinstance(c["history"], list)
        assert c["turns"] and all(t["message"] and t["criterion"] for t in c["turns"])
        assert c["route"] in {
            "main",
            "ordinary",
            "tier",
            "current_state",
            "boundary",
            "memory_extraction",
            "persona_grounding",
            "web_search",
        }


async def test_confirmed_cases_cannot_expand_quick_gate_unbounded(monkeypatch):
    from unittest.mock import AsyncMock
    from app.services.quality import regression
    from evals.quality.corpus import load

    rows = [
        dict(
            id=str(i),
            case_data=dict(
                history=[dict(role="user", content="合成问题")],
                criterion="不捏造",
                priority="P0" if i == 0 else "P1",
            ),
        )
        for i in range(40)
    ]
    monkeypatch.setattr(
        regression, "db", SimpleNamespace(query_raw=AsyncMock(return_value=rows))
    )
    quick = await regression.corpus({"kind": "prompt"})
    assert len(quick) == 30
    human = [c for c in quick if c["suite"] == "human_confirmed"]
    assert len(human) == 10
    assert human[0]["id"] == "human_confirmed:0"
    assert human[1]["id"] == "human_confirmed:39"
    assert {c["suite"] for c in load(quick=True)} <= {c["suite"] for c in quick}
    full = await regression.corpus({"kind": "nightly"})
    assert len(full) == len(load()) + 40


def test_price_unknown_and_judge_unknown_not_success(monkeypatch):
    monkeypatch.setattr(transport, "get_pricing", lambda _: None)
    with pytest.raises(transport.NotConfigured):
        transport.price_bound("ark", "x", [{"role": "user", "content": "hello"}], 900)
    with pytest.raises(ValueError, match="32k"):
        transport.price_bound(
            "ark", "x", [{"role": "user", "content": "中" * 12000}], 900
        )
    with pytest.raises(ValueError):
        judge.parse('{"categories": {"hallucination":"pass"}}')


async def test_transport_reserves_before_http_and_overrides_output_limit(monkeypatch):
    import httpx
    from unittest.mock import AsyncMock
    from app.services import runtime_config
    from app.services.quality import budget

    monkeypatch.setattr(
        runtime_config,
        "_PRICING_CACHE",
        {"dashscope/qwen3.5-flash": {"input": 0.2, "output": 2, "cached_input": 0.04}},
    )
    reserve = AsyncMock(return_value="charge")
    settle = AsyncMock()
    unknown = AsyncMock()
    monkeypatch.setattr(budget, "reserve", reserve)
    monkeypatch.setattr(budget, "settle", settle)
    monkeypatch.setattr(budget, "uncertain", unknown)

    def response(request):
        import json

        assert reserve.await_count == 1
        assert json.loads(request.content)["max_tokens"] == 900
        return httpx.Response(
            200,
            json={
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(response))
    monkeypatch.setattr(transport.httpx, "AsyncClient", lambda **_: client)
    text, usage = await transport.complete(
        {"id": "job", "lease_token": "lease"},
        "dashscope",
        "qwen3.5-flash",
        [{"role": "user", "content": "你好"}],
        key_override="synthetic-only",
    )
    assert (
        text == "ok"
        and usage["input"] == 10
        and settle.await_count == 1
        and unknown.await_count == 0
    )


async def test_transport_missing_usage_remains_charged(monkeypatch):
    import httpx
    from unittest.mock import AsyncMock
    from app.services import runtime_config
    from app.services.quality import budget

    monkeypatch.setattr(
        runtime_config,
        "_PRICING_CACHE",
        {"dashscope/qwen3.5-flash": {"input": 0.2, "output": 2}},
    )
    reserve = AsyncMock(return_value="charge")
    settle = AsyncMock()
    unknown = AsyncMock()
    monkeypatch.setattr(budget, "reserve", reserve)
    monkeypatch.setattr(budget, "settle", settle)
    monkeypatch.setattr(budget, "uncertain", unknown)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"choices": []})
        )
    )
    monkeypatch.setattr(transport.httpx, "AsyncClient", lambda **_: client)
    with pytest.raises(ValueError, match="usage_missing"):
        await transport.complete(
            {"id": "job", "lease_token": "lease"},
            "dashscope",
            "qwen3.5-flash",
            [{"role": "user", "content": "你好"}],
            key_override="synthetic-only",
        )
    unknown.assert_awaited_once_with("charge")
    settle.assert_not_awaited()


@pytest.mark.parametrize(
    "usage,choice,settled",
    [
        (
            {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 5},
            },
            {},
            False,
        ),
        ({"prompt_tokens": 4, "completion_tokens": 901}, {}, False),
        (
            {"prompt_tokens": 4, "completion_tokens": 2},
            {"finish_reason": "length", "message": {"content": "partial"}},
            True,
        ),
    ],
)
async def test_invalid_or_incomplete_provider_receipts_cannot_pass(
    monkeypatch, usage, choice, settled
):
    import httpx
    from unittest.mock import AsyncMock
    from app.services.quality import budget

    monkeypatch.setattr(budget, "reserve", AsyncMock(return_value="receipt"))
    monkeypatch.setattr(budget, "settle", AsyncMock())
    monkeypatch.setattr(budget, "uncertain", AsyncMock())
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"usage": usage, "choices": [choice]})
        )
    )
    monkeypatch.setattr(transport.httpx, "AsyncClient", lambda **_: client)
    with pytest.raises(ValueError):
        await transport.complete(
            dict(
                id="job",
                lease_token="lease",
                snapshot=dict(
                    prices={"dashscope/qwen3.5-flash": {"input": 0.2, "output": 2}}
                ),
            ),
            "dashscope",
            "qwen3.5-flash",
            [{"role": "user", "content": "hi"}],
            key_override="synthetic-only",
        )
    assert budget.settle.await_count == int(settled)
    budget.uncertain.assert_awaited_once_with("receipt")


@pytest.mark.parametrize(
    "provider,messages,prices,operation,error",
    [
        ("untrusted", [{"role": "user", "content": "hi"}], {}, "chat", ValueError),
        ("ark", [], {}, "chat", ValueError),
        (
            "ark",
            [{"role": "user", "content": "hi"}],
            {"input": 0.8, "output": 2, "cached_input": 1},
            "chat",
            transport.NotConfigured,
        ),
        (
            "ark",
            [{"role": "user", "content": "hi"}],
            {"input": 2, "output": 6},
            "web_search",
            transport.NotConfigured,
        ),
        (
            "ark",
            [{"role": "user", "content": "hi"}],
            {"input": 0.8, "output": 2},
            "unsupported",
            ValueError,
        ),
    ],
)
def test_unqualified_provider_contracts_rejected_before_reservation(
    provider, messages, prices, operation, error
):
    with pytest.raises(error):
        transport.price_bound(
            provider,
            "doubao-seed-character-260628",
            messages,
            900,
            prices={provider + "/doubao-seed-character-260628": prices},
            operation=operation,
        )
