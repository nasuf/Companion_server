"""Sequential, budgeted scenario replay and exact baseline comparison."""

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timedelta
from app.config import settings
from app.db import db
from app.services.quality import jobs, budget, transport, gates, replay
from evals.quality.corpus import load, normalise, VERSION


async def corpus(job):
    cases = load(quick=job["kind"] == "prompt")
    rows = await db.query_raw("""SELECT c.id,c.case_data FROM quality_cases c JOIN quality_samples s ON s.id=c.sample_id
      JOIN users u ON u.id=s.user_id AND u.status<>'deleting' WHERE c.enabled
      ORDER BY c.created_at,c.id LIMIT 501""")
    if len(rows) > 500:
        raise ValueError("confirmed_regression_cases_exceed_bound")
    # Prompt saves stay a quick gate even as confirmed cases accumulate. Keep
    # every built-in suite plus the newest critical human cases; full/nightly
    # runs still include all confirmed cases. Selection is deterministic.
    if job["kind"] == "prompt":
        rows = sorted(
            reversed(rows), key=lambda row: row["case_data"].get("priority") != "P0"
        )[:10]
        cases = cases[: 30 - len(rows)]
    for row in rows:
        c = row["case_data"]
        history = c["history"]
        if not history or history[-1]["role"] != "user":
            raise ValueError("review_case_missing_latest_user_turn")
        cases.append(
            normalise(
                "human_confirmed",
                dict(
                    id=row["id"],
                    history=[[m["role"], m["content"]] for m in history[:-1]],
                    message=history[-1]["content"],
                    criterion=c["criterion"],
                    critical=c.get("priority") == "P0",
                    route="recorded_prompt",
                    recorded_messages=c.get("recorded_messages"),
                    now=c["history"][-1].get("at", "2026-10-29T04:00:00+00:00"),
                ),
            )
        )
    return cases


class LocalSink:
    def __init__(self, job):
        self.job = job

    async def prepare(self, report):
        async with db.tx(timeout=timedelta(seconds=8)) as tx:
            await jobs.require_lease(tx, self.job["id"], self.job["lease_token"])
            await tx.execute_raw(
                "UPDATE quality_jobs SET report=$2::jsonb WHERE id=$1",
                self.job["id"],
                json.dumps(report),
            )

    async def renew(self):
        await jobs.renew(self.job["id"], self.job["lease_token"])

    async def result(self, row):
        async with db.tx(timeout=timedelta(seconds=8)) as tx:
            await jobs.require_lease(tx, self.job["id"], self.job["lease_token"])
            await tx.execute_raw(
                """INSERT INTO quality_results(job_id,case_id,repetition,result) VALUES($1,$2,$3,$4::jsonb)
              ON CONFLICT(job_id,case_id,repetition) DO NOTHING""",
                self.job["id"],
                row["case_id"],
                row["repetition"],
                json.dumps(row),
            )

    async def baseline(self, fingerprint):
        baseline = await db.query_raw(
            """SELECT id FROM quality_jobs WHERE id<>$1 AND status='completed'
          AND report->>'corpus_hash'=$2 AND report->>'repeats'=$3
          AND report->>'gate_status' IN ('baseline_recorded','passed') ORDER BY created_at ASC LIMIT 1""",
            self.job["id"],
            fingerprint,
            str(settings.quality_critical_repeats),
        )
        rows = (
            await db.query_raw(
                "SELECT result FROM quality_results WHERE job_id=$1 ORDER BY case_id,repetition",
                baseline[0]["id"],
            )
            if baseline
            else []
        )
        return baseline[0]["id"] if baseline else None, [r["result"] for r in rows]

    async def finish(self, status, report):
        await jobs.finish(self.job, status, report)


async def run(job, *, cases=None, key_overrides=None, sink=None):
    sink = sink or LocalSink(job)
    cases = cases if cases is not None else await corpus(job)
    fingerprint = hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()
    snapshot = job["snapshot"]
    if not snapshot.get("prompts") or not snapshot.get("route"):
        return await sink.finish(
            "not_configured", {"reason": "missing_immutable_snapshot"}
        )
    report = dict(
        corpus_version=VERSION,
        corpus_hash=fingerprint,
        scenarios=len(cases),
        repeats=settings.quality_critical_repeats,
        warning_only=True,
        forced_routing=True,
        quick_selection="提示词快速集共30个场景，最多10个最新人工确认问题（关键问题优先）；完整回放保留全部已启用问题。"
        if job["kind"] == "prompt"
        else None,
        human_cases_scope="原始完整输入重放，仅验模型；不声称重建历史业务状态或验证新模板渲染。缺少精确来源记未知。",
    )
    await sink.prepare(report)
    results = []
    route_usage = {}
    terminal = "completed"
    try:
        for case in cases:
            for repetition in range(
                settings.quality_critical_repeats if case["critical"] else 1
            ):
                history = deepcopy(case["history"])
                anchor = datetime.fromisoformat(case["now"].replace("Z", "+00:00"))
                history_rows = deepcopy(case.get("history_rows"))
                if history_rows is None:
                    history_rows = [
                        dict(
                            role=role,
                            content=text,
                            createdAt=(
                                anchor
                                - timedelta(
                                    seconds=case.get("gap", 120)
                                    + (len(history) - i - 1) * 30
                                )
                            ).isoformat(),
                        )
                        for i, (role, text) in enumerate(history)
                    ]
                passes = []
                for turn, step in enumerate(case["turns"]):
                    await sink.renew()
                    at = (
                        step.get("now")
                        or (anchor + timedelta(minutes=turn)).isoformat()
                    )
                    current = {
                        **case,
                        **step,
                        "history": history,
                        "history_rows": history_rows,
                        "now": at,
                    }
                    # Future user turns/criteria are not evidence for this turn.
                    # Leaking them to the judge falsely rejects correct replies.
                    current.pop("turns", None)
                    current.pop("boundary", None)
                    if case["route"] == "boundary":
                        current["boundary_turn"] = case["boundary"]["turns"][turn]
                    result = await replay.replay_turn(
                        job,
                        current,
                        snapshot["prompts"],
                        case.get("model_route", snapshot["route"]),
                        key_overrides=key_overrides,
                    )
                    for call in result.get("calls", []):
                        key = call.get("route", "chat.completions")
                        metric = route_usage.setdefault(
                            key,
                            dict(
                                calls=0,
                                input=0,
                                output=0,
                                cached_input=0,
                                max_input=0,
                                search_calls=0,
                                cost_cny=0,
                            ),
                        )
                        metric["calls"] += 1
                        for field in ("input", "output", "cached_input", "cost_cny"):
                            metric[field] += call.get(field, 0)
                        metric["max_input"] = max(
                            metric["max_input"], call.get("input", 0)
                        )
                        metric["search_calls"] += call.get("web_search_calls", 0)
                    passes.append(result["judge"]["pass"])
                    history += [
                        ["user", step["message"]],
                        ["assistant", result["reply"]],
                    ]
                    # Preserve earlier statement dates across gaps; never move
                    # the whole history forward to the current turn's clock.
                    history_rows = history_rows + [
                        dict(role="user", content=step["message"], createdAt=at),
                        dict(
                            role="assistant",
                            content=result["reply"],
                            createdAt=(
                                datetime.fromisoformat(at.replace("Z", "+00:00"))
                                + timedelta(seconds=1)
                            ).isoformat(),
                        ),
                    ]
                row = dict(
                    case_id=case["id"],
                    repetition=repetition,
                    critical=case["critical"],
                    suite=case["suite"],
                    **{"pass": None if any(p is None for p in passes) else all(passes)},
                    turns=len(passes),
                )
                if case.get("transcript_scorer") == "long_companion_sim":
                    from evals.long_companion_sim import score_transcript

                    scored = score_transcript(
                        [
                            dict(
                                day=max(
                                    1,
                                    1
                                    + (
                                        datetime.fromisoformat(
                                            r["createdAt"].replace("Z", "+00:00")
                                        )
                                        - anchor
                                    ).days,
                                ),
                                role=r["role"],
                                content=r["content"],
                            )
                            for r in history_rows
                        ]
                    )
                    if row["pass"] is not None:
                        row["pass"] = row["pass"] and scored["passed"]
                # No raw private conversation/reply/judge reason is retained in
                # these generic rows. Owned originals stay in quality_samples.
                await sink.result(row)
                results.append(row)
    except budget.BudgetExhausted:
        terminal = "budget_exhausted"
    except transport.NotConfigured:
        terminal = "not_configured"
    except jobs.LostLease:
        raise
    except Exception as exc:
        terminal = "failed"
        report["error_type"] = type(exc).__name__
        report["error_case_id"] = case["id"]
        if isinstance(exc, replay.RendererError):
            report["renderer_diagnostic"] = exc.diagnostic
    baseline_id, previous = await sink.baseline(fingerprint)
    gate = (
        gates.compare(results, previous)
        if terminal == "completed"
        else {"status": "incomplete"}
    )
    expected = sum(
        settings.quality_critical_repeats if c["critical"] else 1 for c in cases
    )
    if len(results) != expected:
        gate = {"status": "incomplete", "reason": "missing_scenarios"}
    report.update(
        gate=gate,
        gate_status=gate["status"],
        baseline_job_id=baseline_id,
        evaluated=len(results),
        expected=expected,
        route_usage=route_usage,
        search_scope="联网评测限定每次一轮、一个关键词、10条结果、公开网页；生产搜索设置不变。超过32k输入不通过。",
    )
    if gate["status"] in {"failed", "incomplete"} and terminal == "completed":
        terminal = "failed"
    await sink.finish(terminal, report)
