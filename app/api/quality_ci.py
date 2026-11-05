"""Narrow OIDC-only evaluation control, with no business administration rights."""

import json
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal
from app.config import settings
from app.db import db
from app.services.quality import (
    oidc,
    jobs,
    budget,
    snapshots,
    regression,
    repetition_gate,
    transport,
    search,
)

router = APIRouter(prefix="/admin-api/quality-ci", tags=["quality-ci"])


class Begin(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_digest: str = Field(pattern="^[a-f0-9]{64}$")
    keys_configured: bool


@router.post("/begin")
async def begin(body: Begin, identity=Depends(oidc.NIGHTLY)):
    snapshot = await snapshots.capture()
    ident = await jobs.enqueue(
        "nightly",
        oidc.source_key(identity),
        identity["sha"],
        {**snapshot, "source_digest": body.source_digest},
    )
    row = (await db.query_raw("SELECT * FROM quality_jobs WHERE id=$1", ident))[0]
    if row["status"] != "pending":
        # Retrying begin must never replay an ambiguous billed job.
        return dict(status=row["status"], report=row["report"])
    reason = None
    if not settings.quality_enabled:
        reason = "quality_not_enabled"
    elif not body.keys_configured:
        reason = "evaluation_secrets_not_configured"
    else:
        previous = await db.query_raw(
            """SELECT id FROM quality_jobs WHERE kind='nightly' AND status='completed'
          AND snapshot->>'source_digest'=$1 LIMIT 1""",
            body.source_digest,
        )
        if previous:
            reason = "chat_source_unchanged"
    if reason:
        status = "skipped" if reason == "chat_source_unchanged" else "not_configured"
        await db.execute_raw(
            "UPDATE quality_jobs SET status=$2,report=$3::jsonb WHERE id=$1 AND status='pending'",
            ident,
            status,
            json.dumps({"reason": reason}),
        )
        return dict(status=status, reason=reason)
    job = await jobs.claim(job_id=ident)
    if job is None:
        raise HTTPException(409, "quality_executor_busy")
    try:
        cases = await regression.corpus(job)
        models = {
            "/".join(snapshot["route"]),
            "dashscope/qwen3.5-flash",
            "dashscope/qwen3.5-plus",
        }
        prices = {key: snapshot["prices"].get(key) for key in models}
        return dict(
            status="running",
            job=job,
            cases=cases,
            prices=prices,
            policy=dict(critical_repeats=settings.quality_critical_repeats),
        )
    except BaseException:
        await jobs.finish(job, "failed", {"reason": "manifest_unavailable"})
        raise


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lease_token: str = Field(min_length=32, max_length=32)
    operation: Literal[
        "renew",
        "prepare",
        "result",
        "baseline",
        "finish",
        "reserve",
        "settle",
        "uncertain",
    ]
    data: dict = Field(default_factory=dict)


@router.post("/jobs/{job_id}")
async def action(job_id: str, body: Action, identity=Depends(oidc.NIGHTLY)):
    if len(json.dumps(body.data).encode()) > 2_000_000:
        raise HTTPException(413, "quality_request_too_large")
    rows = await db.query_raw(
        "SELECT * FROM quality_jobs WHERE id=$1 AND kind='nightly' AND source_key=$2 AND source_version=$3",
        job_id,
        oidc.source_key(identity),
        identity["sha"],
    )
    if not rows:
        raise HTTPException(404, "quality_job_not_found")
    job = {**rows[0], "lease_token": body.lease_token}
    sink = regression.LocalSink(job)
    data = body.data
    try:
        # Every mutation is bound to this exact repository/run/lease.
        await jobs.renew(job_id, body.lease_token)
        if body.operation == "renew":
            return {"ok": True}
        if body.operation == "prepare":
            await sink.prepare(data)
            return {"ok": True}
        if body.operation == "result":
            allowed = {"case_id", "repetition", "critical", "suite", "pass", "turns"}
            if (
                set(data) != allowed
                or type(data["pass"]) not in (bool, type(None))
                or not 0 <= data["repetition"] < 5
            ):
                raise ValueError("invalid_quality_result")
            await sink.result(data)
            return {"ok": True}
        if body.operation == "baseline":
            ident, results = await sink.baseline(data["corpus_hash"])
            return dict(id=ident, results=results)
        if body.operation == "finish":
            await sink.finish(data["status"], data["report"])
            return {"ok": True}
        if body.operation == "reserve":
            provider, model = data["model"].split("/", 1)
            allowed = {
                tuple(job["snapshot"]["route"]),
                ("dashscope", "qwen3.5-flash"),
                ("dashscope", "qwen3.5-plus"),
            }
            if (provider, model) not in allowed:
                raise ValueError("unapproved_evaluation_model")
            _, ceiling = transport.price_bound(
                provider,
                model,
                data["messages"],
                data["max_tokens"],
                prices=job["snapshot"]["prices"],
                operation=data.get("operation", "chat"),
            )
            charge = await budget.reserve(
                job_id,
                data["model"],
                ceiling,
                lease_token=body.lease_token,
                request=data,
            )
            return dict(id=charge)
        charge = await db.query_raw(
            "SELECT model,reserved_cny,request_policy FROM quality_charges WHERE id=$1 AND job_id=$2",
            data["charge_id"],
            job_id,
        )
        if not charge:
            raise ValueError("charge_not_owned_by_run")
        if body.operation == "uncertain":
            await budget.uncertain(data["charge_id"])
            return {"ok": True}
        usage = data["usage"]
        if (
            any(
                type(usage.get(k)) is not int or usage[k] < 0
                for k in ("input", "output", "cached_input")
            )
            or usage["cached_input"] > usage["input"]
        ):
            raise ValueError("invalid_quality_usage")
        policy = charge[0]["request_policy"]
        search_request = policy.get("operation") == "web_search"
        if search_request:
            search.validate(usage, policy["max_tokens"])
        elif usage.get("operation", "chat") != "chat" or usage.get(
            "web_search_calls", 0
        ):
            raise ValueError("settlement_operation_mismatch")
        cost = transport.price_cost(
            transport.pinned_price(charge[0]["model"], job["snapshot"]["prices"]),
            usage["input"],
            usage["output"],
            usage["cached_input"],
        )
        if search_request:
            cost += float(search.tool_cost(usage))
        await budget.settle(data["charge_id"], cost, usage)
        return {"ok": True}
    except jobs.LostLease as exc:
        raise HTTPException(409, "quality_lease_lost") from exc
    except budget.BudgetExhausted as exc:
        raise HTTPException(429, "quality_budget_exhausted") from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(400, "invalid_quality_action") from exc


class GateBegin(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["gate", "calibrate"]
    case_set: str = Field(pattern="^[a-z0-9-]{1,40}$")
    # The candidate tree compared against this main; absent when calibrating.
    candidate_sha: str | None = Field(default=None, pattern="^[a-f0-9]{40}$")
    keys_configured: bool


@router.post("/repetition-gate/begin")
async def gate_begin(body: GateBegin, identity=Depends(oidc.REPETITION_GATE)):
    if (body.mode == "gate") != (body.candidate_sha is not None):
        raise HTTPException(400, "candidate_required_only_for_gate")
    snapshot = await repetition_gate.snapshot()
    ident = await jobs.enqueue(
        repetition_gate.GATE_KIND,
        oidc.source_key(identity),
        identity["sha"],
        {**snapshot, **body.model_dump(exclude={"keys_configured"})},
    )
    row = (await db.query_raw("SELECT * FROM quality_jobs WHERE id=$1", ident))[0]
    if row["status"] != "pending":
        return dict(status=row["status"], report=row["report"])
    cases, contexts = await repetition_gate.cases(body.case_set)
    reason = None
    if not settings.quality_enabled:
        reason = "quality_not_enabled"
    elif not body.keys_configured:
        reason = "evaluation_secrets_not_configured"
    elif not cases:
        reason = "case_set_not_loaded"
    elif body.mode == "gate":
        previous = await db.query_raw(
            """SELECT id FROM quality_jobs WHERE kind=$1 AND status='completed' AND id<>$2
          AND source_version=$3 AND snapshot->>'candidate_sha'=$4 AND snapshot->>'case_set'=$5 LIMIT 1""",
            repetition_gate.GATE_KIND,
            ident,
            identity["sha"],
            body.candidate_sha,
            body.case_set,
        )
        if previous:
            reason = "already_evaluated"
    if reason:
        status = "skipped" if reason == "already_evaluated" else "not_configured"
        await db.execute_raw(
            "UPDATE quality_jobs SET status=$2,report=$3::jsonb WHERE id=$1 AND status='pending'",
            ident,
            status,
            json.dumps({"reason": reason}),
        )
        return dict(status=status, reason=reason)
    job = await jobs.claim(job_id=ident, executor=repetition_gate.GATE_KIND)
    if job is None:
        # Another gate holds the executor; this run never starts, so it is not left pending.
        await db.execute_raw(
            "UPDATE quality_jobs SET status='skipped',report=$2::jsonb WHERE id=$1 AND status='pending'",
            ident,
            json.dumps({"reason": "executor_busy"}),
        )
        raise HTTPException(409, "quality_executor_busy")
    return dict(
        status="running",
        job=dict(id=job["id"], lease_token=job["lease_token"]),
        cases=cases,
        contexts=contexts,
        prompts=snapshot["prompts"],
        protected_prompts=snapshot["protected_prompts"],
        route=snapshot["route"],
        prices={m: snapshot["prices"].get(m) for m in repetition_gate.models(job)},
        max_block_cny=str(repetition_gate.MAX_BLOCK_CNY),
    )


class GateAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lease_token: str = Field(min_length=32, max_length=32)
    operation: Literal["renew", "reserve_block", "settle_block", "uncertain", "finish"]
    data: dict = Field(default_factory=dict)


@router.post("/repetition-gate/jobs/{job_id}")
async def gate_action(job_id: str, body: GateAction, identity=Depends(oidc.REPETITION_GATE)):
    if len(json.dumps(body.data).encode()) > 200_000:
        raise HTTPException(413, "quality_request_too_large")
    rows = await db.query_raw(
        "SELECT * FROM quality_jobs WHERE id=$1 AND kind=$2 AND source_key=$3 AND source_version=$4",
        job_id,
        repetition_gate.GATE_KIND,
        oidc.source_key(identity),
        identity["sha"],
    )
    if not rows:
        raise HTTPException(404, "quality_job_not_found")
    job = {**rows[0], "lease_token": body.lease_token}
    data = body.data
    try:
        await jobs.renew(job_id, body.lease_token)
        if body.operation == "renew":
            return {"ok": True}
        if body.operation == "finish":
            if data["status"] not in {"completed", "failed", "budget_exhausted"}:
                raise ValueError("invalid_gate_status")
            await jobs.finish(job, data["status"], dict(data["report"]))
            return {"ok": True}
        if body.operation == "reserve_block":
            if data["model"] not in repetition_gate.models(job):
                raise ValueError("unapproved_evaluation_model")
            charge = await budget.reserve(
                job_id,
                data["model"],
                repetition_gate.block_amount(data["amount"]),
                lease_token=body.lease_token,
                request={"operation": "block"},
            )
            return dict(id=charge)
        charge = await db.query_raw(
            "SELECT model FROM quality_charges WHERE id=$1 AND job_id=$2",
            data["charge_id"],
            job_id,
        )
        if not charge:
            raise ValueError("charge_not_owned_by_run")
        if body.operation == "uncertain":
            await budget.uncertain(data["charge_id"])
            return {"ok": True}
        usage = dict(data["usage"])
        cost = repetition_gate.block_cost(job, charge[0]["model"], usage)
        await budget.settle(data["charge_id"], cost, usage)
        return {"ok": True}
    except jobs.LostLease as exc:
        raise HTTPException(409, "quality_lease_lost") from exc
    except budget.BudgetExhausted as exc:
        raise HTTPException(429, "quality_budget_exhausted") from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(400, "invalid_quality_action") from exc
