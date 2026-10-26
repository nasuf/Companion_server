"""Sequential synthetic conversation replay through the real reply renderer.

Production prompts are an explicit read-only snapshot. DB, Redis, scheduling,
voice and persistence use the existing isolated incident harness. Routing is
forced to separately measure main/tier/current-state/search paths; this is not
an intent-classifier or production-delivery E2E. No monkeypatches run concurrently.
"""

from __future__ import annotations
import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
import time

from evals.context_embedding.measurement import (
    ARK_MODEL,
    ark_usage,
    priced_call,
    report_routes,
    stable_prefix,
)
from evals.context_embedding.judge import (
    DEFAULT_JUDGE_MODEL,
    JUDGING_INSTRUCTION,
    judge_signature,
)


class Transport:
    def __init__(self, client, model, search, calls):
        self.client, self.model, self.search, self.calls = client, model, search, calls

    async def __call__(self, messages, **runtime_options):
        from app.config import settings
        from app.services.llm.providers import _model_runtime_options
        from app.services.llm.ark_web_search import _to_responses_input, _extract_output
        from app.services.chat.prompt_builder import fold_trailing_system

        route = "responses.web_search" if self.search else "chat.completions"
        base = settings.ark_base_url.rstrip("/")
        headers = {"Authorization": "Bearer " + settings.ark_api_key}
        started = time.monotonic()
        ttft = None
        usage = None
        parts = []
        usage_payloads = []
        record = {
            "model": f"ark/{self.model}",
            "route": route,
            "stable_prefix_sha256": stable_prefix(messages),
        }
        self.calls.append(record)  # failed requests remain visible
        try:
            if self.search:
                if runtime_options.get("needs_web_search") is not True:
                    raise ValueError(
                        "Search route must exercise the real search prompt branch"
                    )
                location = runtime_options.get("web_search_location")
                # Match production nonstream Responses requests, including tool settings.
                body = {
                    "model": self.model,
                    "input": _to_responses_input(fold_trailing_system(messages)),
                    "tools": [
                        {
                            "type": "web_search",
                            **({"user_location": location} if location else {}),
                        }
                    ],
                    "tool_choice": "required",
                    "stream": False,
                }
                response = await self.client.post(
                    base + "/responses", headers=headers, json=body
                )
                response.raise_for_status()
                payload = response.json()
                usage = payload.get("usage")
                if payload.get("status") != "completed":
                    raise RuntimeError("Incomplete Responses result")
                text = _extract_output(payload).text
                record["time_to_complete_seconds"] = time.monotonic() - started
                record["ttft_kind"] = "not_available_nonstream_production_route"
                record["caching"] = payload.get("caching")
                record["response_status"] = payload.get("status")
                record["web_search_location_supplied"] = bool(location)
            else:
                body = {
                    "model": self.model,
                    "messages": messages,
                    **_model_runtime_options("ark", self.model),
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                finished = False
                finish_reason = None
                async with self.client.stream(
                    "POST", base + "/chat/completions", headers=headers, json=body
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        if line == "data: [DONE]":
                            finished = True
                            continue
                        event = json.loads(line[6:])
                        if event.get("usage") is not None:
                            usage = event["usage"]
                            usage_payloads.append(usage)
                        for choice in event.get("choices", []):
                            content = choice.get("delta", {}).get("content")
                            if content:
                                if ttft is None:
                                    ttft = time.monotonic() - started
                                parts.append(content)
                            if choice.get("finish_reason"):
                                finish_reason = choice["finish_reason"]
                if not finished or finish_reason not in {"stop"}:
                    raise RuntimeError("Incomplete or truncated model stream")
                text = "".join(parts)
                record["ttft_kind"] = "first_nonempty_content_delta"
            measurement = ark_usage(usage, route)
            record.update(
                raw_usage=usage,
                raw_usage_payloads=usage_payloads,
                measurement=measurement,
                price=priced_call(f"ark/{self.model}", route, measurement),
                ttft_seconds=ttft,
                duration_seconds=time.monotonic() - started,
            )
            if not text:
                raise RuntimeError("Empty model response")
            return {"text": text, "usage": usage}
        except Exception as exc:
            record.update(
                error_type=type(exc).__name__,
                measurement={"status": "request_failed"},
                price={"status": "request_failed", "cost_cny": None},
                ttft_seconds=ttft,
                duration_seconds=time.monotonic() - started,
            )
            raise


def save(path, report):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


async def run(args):
    if (args.source_root / ".env").exists():
        raise ValueError("Source checkout must not contain .env")
    os.environ.update(
        DATABASE_URL="postgresql://test:test@127.0.0.1:1/context_eval",
        REDIS_URL="redis://127.0.0.1:1/15",
        PYTHON_DOTENV_DISABLED="1",
        APP_ENV="test",
        TRACE_BACKEND="off",
    )
    import httpx
    from app.services.chat import orchestrator as chat
    from app.services.llm.providers import _model_runtime_options
    from evals.temporal_awareness.run_incident_eval import run as run_incident

    original_context = chat._fetch_intent_context
    corpus = json.loads(args.cases.read_text())
    if args.case_ids:
        corpus = [c for c in corpus if c["id"] in args.case_ids.split(",")]
    if not corpus:
        raise ValueError("No selected cases")
    if args.max_cost_cny <= 0:
        raise ValueError("Cost limit must be positive")
    if args.output.exists():
        raise ValueError("Output exists: retain it and use a fresh run path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_files = [
        *args.source_root.joinpath("app").rglob("*.py"),
        *args.source_root.joinpath("evals/context_embedding").rglob("*.py"),
        *args.source_root.joinpath("evals/temporal_awareness").rglob("*.py"),
    ]
    report = {
        "schema_version": 1,
        "model": args.model,
        "model_options": _model_runtime_options("ark", args.model),
        "judge_protocol": judge_signature(args.judge_model),
        "repeats": args.repeats,
        "case_count": len(corpus),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.source_root, text=True
        ).strip(),
        "source_files_sha256": {
            str(p.relative_to(args.source_root)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sorted(source_files)
        },
        "product_cost_budget_cny": args.max_cost_cny,
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "prompt_snapshot_sha256": hashlib.sha256(
            args.prompt_snapshot.read_bytes()
        ).hexdigest(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "runs": [],
        "limitations": [
            "forced routes; synthetic state and side effects",
            "live main/tier/state/search generation; no production delivery",
            "search route is nonstream in production: TTFT unavailable, report completion latency separately",
            "original app snapshot and same model options; no prompt/window changes",
            "judge and temporal guard use existing incident harness; judge fees not included in product reply costs",
        ],
        "status": "running",
    }
    save(args.output, report)
    try:
        async with httpx.AsyncClient(timeout=90) as client:
            for case in corpus:
                for sample in range(args.repeats):
                    history = list(case["history"])
                    previous_prefix = None
                    for turn, step in enumerate(case["turns"]):
                        known_cost = sum(
                            c["price"].get("cost_cny") or 0
                            for r in report["runs"]
                            for c in r["calls"]
                        )
                        if known_cost >= args.max_cost_cny:
                            raise RuntimeError("Evaluation product-cost budget reached")
                        calls = []
                        transport = Transport(
                            client, args.model, case["route"] == "web_search", calls
                        )

                        def fixture_hook(patches, chat_module, io, current_case):
                            if case["route"] == "web_search":
                                chat_module.fetch_parallel_context.return_value.needs_web_search = (
                                    True
                                )
                            patches.setattr(
                                chat_module, "_fetch_intent_context", original_context
                            )
                            generate = chat_module._generate_reply

                            async def force_route(**kwargs):
                                if case["route"] in ("main", "web_search"):
                                    kwargs["force_main_prompt"] = True
                                return await generate(**kwargs)

                            patches.setattr(chat_module, "_generate_reply", force_route)

                        now = datetime(
                            2026, 10, 26, 4, tzinfo=timezone.utc
                        ) + timedelta(seconds=turn * 60)
                        incident = dict(
                            id=case["id"],
                            now=now.isoformat(),
                            gap=30,
                            history=history,
                            message=step["message"],
                            criterion=step["criterion"],
                            memory=False,
                            route=(
                                "current_state"
                                if case["route"] == "current_state"
                                else "ordinary"
                            ),
                        )
                        detail = (
                            args.output.parent
                            / f'{args.output.stem}-{case["id"]}-{sample}-{turn}.json'
                        )
                        if detail.exists():
                            raise ValueError("Detail exists: refusing overwrite")
                        runner_args = SimpleNamespace(
                            source_root=args.source_root,
                            prompt_snapshot=args.prompt_snapshot,
                            model=args.model,
                            judge_model=args.judge_model,
                            judging_instruction=JUDGING_INSTRUCTION,
                            guard_model="qwen3.5-flash",
                            samples=1,
                            output=detail,
                            provider_call=transport,
                            fixture_hook=fixture_hook,
                        )
                        row = {
                            "case": case["id"],
                            "category": case["category"],
                            "target_route": case["route"],
                            "sample": sample,
                            "turn": turn,
                            "critical": case["critical"],
                            "calls": calls,
                            "detail": str(detail),
                        }
                        report["runs"].append(row)
                        try:
                            await run_incident(runner_args, [incident])
                            evidence = json.loads(detail.read_text())
                            r = evidence["runs"][0]
                            row.update(
                                reply=r["reply"],
                                judge=r["judge"],
                                location_queries=r["location_queries"],
                                response_diagnostics=r["response_diagnostics"],
                                status="completed",
                            )
                            actual_path = (
                                r["response_diagnostics"].get("reply_path")
                                or "current_state_shortcircuit"
                            )
                            for call in calls:
                                call["reply_path"] = actual_path
                            prefix = (
                                calls[0].get("stable_prefix_sha256") if calls else None
                            )
                            row["stable_prefix_equal_previous_turn"] = (
                                prefix == previous_prefix
                                if prefix and previous_prefix
                                else None
                            )
                            previous_prefix = prefix
                            history += [
                                ["user", step["message"]],
                                ["assistant", r["reply"]],
                            ]
                        except Exception as exc:
                            row.update(status="failed", error_type=type(exc).__name__)
                            raise
                        finally:
                            report["routes"] = report_routes(
                                [c for r in report["runs"] for c in r["calls"]]
                            )
                            save(args.output, report)
                        print(
                            json.dumps(
                                {
                                    "case": case["id"],
                                    "sample": sample,
                                    "turn": turn,
                                    "judge": row["judge"]["pass"],
                                }
                            ),
                            flush=True,
                        )
            report["status"] = "baseline_recorded"
            report["completed_turns"] = len(report["runs"])
            report["judged_pass"] = sum(
                r.get("judge", {}).get("pass") is True for r in report["runs"]
            )
            report["measurement_qualified"] = all(
                r["status"] == "qualified" for r in report["routes"]
            )
            if not report["measurement_qualified"]:
                report["status"] = "measurement_not_qualified"
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        raise
    finally:
        report["ended_at"] = datetime.now(timezone.utc).isoformat()
        save(args.output, report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--prompt-snapshot", type=Path, required=True)
    p.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.json"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case-ids")
    p.add_argument("--model", default=ARK_MODEL.split("/", 1)[1])
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--repeats", type=int, choices=range(1, 6), default=3)
    p.add_argument("--max-cost-cny", type=float, default=10)
    p.add_argument("--live", action="store_true", required=True)
    result = asyncio.run(run(p.parse_args()))
    raise SystemExit(0 if result["measurement_qualified"] else 2)
