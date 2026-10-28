"""Independent ledger fixtures: exact, interrupted, missing, expired and unpriced."""

import json
from datetime import timedelta
from uuid import uuid4
from unittest.mock import AsyncMock, patch
import pytest

from app.services.effect import reconciliation as r, daily_facts, definitions as defs
from tests.test_beta_daily_facts_postgres import (
    pg,
    _user,
    _conversation,
    _at,
    _now,
    DAY,
)


async def trace(
    pg,
    conv,
    *,
    usage=(100, 10),
    tokens=(100, 10),
    unmetered=0,
    model="qwen3.5-flash",
    qualified=True,
):
    tid = uuid4().hex
    await pg.db.execute_raw(
        """INSERT INTO trace_runs(run_id,trace_id,name,run_type,status,started_at,inputs_json,extra_json)
      VALUES($1,$1,'chat_request','chain','success',$2::timestamp,$3::jsonb,$4::jsonb)""",
        tid,
        _at(DAY, 23, 59).isoformat(),
        json.dumps({"conversation_id": conv}),
        json.dumps(
            {
                "metadata": {
                    "usage_accounting_version": r.ACCOUNTING_VERSION
                    if qualified
                    else "old"
                }
            }
        ),
    )
    await pg.db.execute_raw(
        """INSERT INTO trace_runs(run_id,trace_id,parent_id,name,run_type,status,started_at,
      model_name,prompt_tokens,completion_tokens) VALUES($1,$2,$2,'reply','llm',$3,$4::timestamp,$5,$6,$7)""",
        uuid4().hex,
        tid,
        "cancelled" if tokens == (None, None) else "success",
        _at(DAY, 23, 59, 1).isoformat(),
        model,
        *tokens,
    )
    if usage is not None:
        user = (
            await pg.db.query_raw("SELECT user_id FROM conversations WHERE id=$1", conv)
        )[0]["user_id"]
        await pg.db.execute_raw(
            """INSERT INTO llm_usage(id,trace_id,conversation_id,user_id,scope,input_tokens,output_tokens,
          tokens_by_model,call_count,cost_cny,created_at) VALUES($1,$2,$3,$4,'chat',$5,$6,$7::jsonb,1,.001,$8::timestamp)""",
            uuid4().hex,
            tid,
            conv,
            user,
            *usage,
            json.dumps(
                {
                    "dashscope/" + model: {
                        "input": usage[0],
                        "output": usage[1],
                        "unmetered_calls": unmetered,
                    }
                }
            ),
            _at(DAY + timedelta(days=1), 0, 1).isoformat(),
        )
    return tid


async def test_reconciliation_cross_midnight_exact_unknown_and_missing(pg):
    user = await _user(pg)
    conv, _ = await _conversation(pg, user)
    traces = []
    try:
        exact = await trace(pg, conv)
        traces.append(exact)
        missing = await trace(pg, conv, usage=None)
        traces.append(missing)
        unknown = await trace(pg, conv, usage=(0, 0), tokens=(None, None), unmetered=1)
        traces.append(unknown)
        mismatch = await trace(pg, conv, usage=(101, 10))
        traces.append(mismatch)
        unpriced = await trace(pg, conv, model="not-registered")
        traces.append(unpriced)
        await daily_facts.compute_day(DAY, now=_now(DAY + timedelta(days=2)))
        out = await r.read(DAY, DAY, now=_now(DAY + timedelta(days=2)))
        byid = {x["trace_id"]: x for x in out["items"]}
        assert (
            byid[exact]["status"] == "matched"
            and byid[exact]["data"]["usage_input"] == 100
        )
        assert byid[missing]["status"] == "missing_usage"
        assert byid[unknown]["status"] == "unmetered" and out["unmetered_calls"] == 1
        assert byid[mismatch]["status"] == "mismatch"
        assert byid[unpriced]["data"]["unpriced_models"] == ["dashscope/not-registered"]
        assert out["counts"]["matched"] == 2 and out["comparable_traces"] == 4
        assert out["match"]["state"] == "insufficient_sample"
        assert (
            len(
                (
                    await r.read(
                        DAY,
                        DAY,
                        status="missing_usage",
                        now=_now(DAY + timedelta(days=2)),
                    )
                )["items"]
            )
            == 1
        )
        assert (await r.read(DAY, DAY, now=_now(DAY + timedelta(days=31))))[
            "state"
        ] == "expired"
        await daily_facts.compute_day(
            DAY + timedelta(days=1), now=_now(DAY + timedelta(days=2))
        )
        assert (
            await r.read(
                DAY + timedelta(days=1),
                DAY + timedelta(days=1),
                now=_now(DAY + timedelta(days=2)),
            )
        )["total"] == 0
    finally:
        await pg.db.execute_raw(
            "DELETE FROM trace_runs WHERE trace_id=ANY($1::text[])", traces
        )
        await pg.db.execute_raw(
            "DELETE FROM llm_usage WHERE trace_id=ANY($1::text[])", traces
        )


@pytest.mark.parametrize(
    "kwargs",
    [{"status": "secret"}, {"page_size": 101}, {"page": 0}, {"agent": "x" * 129}],
)
async def test_reconciliation_rejects_unbounded_filters(kwargs):
    with pytest.raises(ValueError):
        await r.read(DAY, DAY, **kwargs)


@pytest.mark.parametrize(
    "missing,mismatch,unpriced,band,total,expected",
    [
        (0, 0, 0, 0, 0, "ok"),
        (1, 0, 0, 0, 100, "warn"),
        (0, 2, 0, 0, 100, "warn"),
        (0, 6, 0, 0, 100, "violated"),
        (0, 0, 1, 0, 100, "violated"),
        (0, 0, 0, 1, 100, "warn"),
        (0, 1, 0, 0, 100, "ok"),
    ],
)
async def test_release_invariant_thresholds(
    missing, mismatch, unpriced, band, total, expected
):
    from app.services.ops.invariants import _check_token_usage_reconciled

    report = dict(
        state="ok",
        counts=dict(
            missing_usage=missing,
            mismatch=mismatch,
            unpriced=unpriced,
            band_exceeded=band,
        ),
        comparable_traces=total,
        unmetered_calls=3,
    )
    with patch.object(r, "read", AsyncMock(return_value=report)) as read:
        result = await _check_token_usage_reconciled()
    assert result.status == expected
    assert read.call_args.kwargs["qualified_only"] is True


def test_expiry_and_max_are_not_sums_or_zero():
    from app.services.effect.dashboard import aggregate_points

    points = [
        dict(state="ok", value=30000, settled=True),
        dict(state="ok", value=20000, settled=True),
    ]
    assert aggregate_points("token_max_input", points)["value"] == 30000
    assert (
        aggregate_points("token_max_input", [dict(state="expired", value=None)])[
            "state"
        ]
        == "expired"
    )
    assert defs.METRICS["token_exact_match"]["cohorts"] == ("all",)
