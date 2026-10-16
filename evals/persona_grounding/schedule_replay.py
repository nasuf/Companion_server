"""Replay real schedules through the location verifier (persona.grounding_check).

The synthetic set next door (run_eval.py / cases.json) pins the rules one case
at a time. This one measures what the verifier does to schedules the model
actually wrote: does every fabricated place stay rejected, and do generic
places (附近公园 / 老城区市集 / 街角咖啡馆) stop being rejected?

Cases are the gated slots (`needs_location_verification(text, implicit_self=True)`)
of the production schedules of 2026-10-03..10-09. Grounding went live on
10-09 17:19 and no schedule of that week carries its neutral placeholder, so
every slot is the model's original wording. Each is paired with its agent's
real profile and labelled by hand:

  reject  an Agent location the profile does not support: a named street /
          district / city / landmark absent from the profile, or a scene that
          implies being in another city (another city's places, 海边 for an
          inland city)
  allow   generic places (公园, 市集, 老城区, 街角, 江边 without a name), places
          literally in the profile, or no place at all
  either  generic on its own, but verified next to a fabricated trip of the
          same day; a rejection is defensible, so it is not scored

The case files (schedule_replay.jsonl, schedule_replay_profiles.json) are
production data and are not in the repository: export them locally with the
SQL below and label them by hand. Export hashes agent ids and keeps no user
id; remove personal names that schedules quote from chats yourself. A
re-export keeps the labels of unchanged slots, so redacted slots come back
unlabelled and must be redacted and labelled again.

Subcommands:
  export    build/refresh the case file from a read-only psql export; labels of
            unchanged slots are kept, changed slots come back unlabelled
  validate  check the case file offline
  run       live calibration, batched like the schedule generation path

`run` mirrors generation-time verification as batched in schedule.py
(_GROUNDING_BATCH / _GENERATION_GROUNDING_TIMEOUT): gated slots two at a time,
the batches of one schedule concurrently, 12s each. That batching ships
together with this calibration; the read path's 5s hot-path budget is not
replayed. Requests match production (temperature 0, no thinking, JSON by
prompt only) and verdicts are accepted on the terms of verify_generated_locations.

`run` needs PERSONA_GROUNDING_API_KEY and PERSONA_GROUNDING_BASE_URL set
explicitly. Reports go to evals/results/ (gitignored); they contain case ids,
verdicts, the verifier's reasons and timing, never credentials.

Export (read-only, on the database host):

  SELECT json_build_object('agent_id', s.agent_id, 'date', s.date,
         'schedule', s.schedule_data, 'name', a.name, 'city', a.city,
         'occupation', a.occupation, 'age', a.age,
         'nickname', to_jsonb(a) ->> 'nickname')
  FROM ai_daily_schedules s JOIN ai_agents a ON a.id = s.agent_id
  WHERE s.date BETWEEN '2026-10-03' AND '2026-10-09'
  ORDER BY s.agent_id, s.date;

  psql -At ... > export.jsonl
  python evals/persona_grounding/schedule_replay.py export export.jsonl
"""
from __future__ import annotations

import argparse
import ast
import asyncio
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.services.llm.models import _extract_json  # noqa: E402
from app.services.persona_grounding import canonical_facts, needs_location_verification  # noqa: E402

HERE = Path(__file__).resolve().parent
CASES = HERE / "schedule_replay.jsonl"
PROFILES = HERE / "schedule_replay_profiles.json"
RESULTS = ROOT / "evals" / "results"
LABELS = {"reject", "allow", "either"}
# See the module docstring. An unverified batch is replaced like a rejected one.
BATCH = 2
GENERATION_TIMEOUT_S = 12


def slot_text(slot: dict) -> str:
    return str(slot.get("event") or slot.get("activity") or "")


def agent_key(agent_id: str) -> str:
    return "a" + hashlib.sha256(agent_id.encode()).hexdigest()[:8]


def load_cases() -> list[dict]:
    return [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]


def write_cases(cases: list[dict]) -> None:
    CASES.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases))


def export(source: Path) -> None:
    previous = {c["id"]: c for c in load_cases()} if CASES.exists() else {}
    profiles, schedules, cases = {}, {}, []
    for line in source.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        slots = row["schedule"]
        if isinstance(slots, str):
            slots = json.loads(slots)
        if not isinstance(slots, list):
            continue
        key = agent_key(row["agent_id"])
        profiles[key] = {k: row[k] for k in ("name", "nickname", "city", "occupation", "age") if row.get(k)}
        schedule = f"{key}@{str(row['date'])[:10]}"
        schedules[schedule] = {"agent": key, "slots": len(slots)}
        for i, slot in enumerate(slots):
            text = slot_text(slot) if isinstance(slot, dict) else ""
            if not needs_location_verification(text, implicit_self=True):
                continue
            case_id = f"{schedule}#{i}"
            old = previous.get(case_id)
            kept = old if old and old["text"] == text else {}
            cases.append({"id": case_id, "schedule": schedule, "slot": i, "text": text,
                          "label": kept.get("label"), "note": kept.get("note", "")})
    write_cases(cases)
    PROFILES.write_text(json.dumps({"profiles": profiles, "schedules": schedules},
                                   ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    unlabelled = sum(c["label"] is None for c in cases)
    print(json.dumps({"schedules": len(schedules), "cases": len(cases), "unlabelled": unlabelled}))


def require_data(*paths: Path) -> None:
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"{', '.join(missing)} not found: production data is not in the repository, "
                         "export it locally first (see the module docstring)")


def validate() -> tuple[list[dict], dict]:
    require_data(CASES, PROFILES)
    cases, meta = load_cases(), json.loads(PROFILES.read_text())
    profiles, schedules = meta["profiles"], meta["schedules"]
    problems = []
    if len({c["id"] for c in cases}) != len(cases):
        problems.append("duplicate case ids")
    for c in cases:
        schedule = schedules.get(c["schedule"])
        if schedule is None or not profiles.get(schedule["agent"], {}).get("city"):
            problems.append(f"{c['id']}: schedule or profile city missing")
        if c["label"] not in LABELS or not c["note"].strip():
            problems.append(f"{c['id']}: needs a label in {sorted(LABELS)} and a note")
        if not needs_location_verification(c["text"], implicit_self=True):
            # The gate changed; this slot no longer reaches the verifier.
            problems.append(f"{c['id']}: no longer gated, re-export")
    if problems:
        raise SystemExit("\n".join(problems[:50]))
    return cases, meta


def load_policy(ref: str | None) -> str:
    if ref is None:
        from app.services.prompting.defaults import PERSONA_GROUNDING_CHECK_PROMPT
        return PERSONA_GROUNDING_CHECK_PROMPT
    source = subprocess.run(["git", "show", f"{ref}:app/services/prompting/defaults.py"],
                            cwd=ROOT, check=True, capture_output=True, text=True).stdout
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "PERSONA_GROUNDING_CHECK_PROMPT" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit(f"PERSONA_GROUNDING_CHECK_PROMPT not found at {ref}")


async def verify(client, model: str, policy: str, facts: dict, batch: list[dict], *,
                 kind: str = "schedule", question: str = "", timeout: float = GENERATION_TIMEOUT_S) -> dict:
    """One verifier call, accepted only on the terms verify_generated_locations uses.

    A single attempt: production may retry inside the same budget, so a
    transient provider error counts as unverified here more often than live.
    """
    items = [{"index": j, "text": c["text"]} for j, c in enumerate(batch)]
    prompt = policy.format(
        facts=json.dumps(facts, ensure_ascii=False, sort_keys=True), kind=kind,
        items=json.dumps(items, ensure_ascii=False), question=json.dumps(question, ensure_ascii=False),
    )
    # Same request as production: temperature 0, no thinking, JSON by prompt only.
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "enable_thinking": False}
    start = time.monotonic()
    try:
        async with asyncio.timeout(timeout):
            response = await client.post("/chat/completions", json=body)
            response.raise_for_status()
        verdicts = _extract_json(response.json()["choices"][0]["message"]["content"])["verdicts"]
        if not isinstance(verdicts, list) or len(verdicts) != len(batch):
            raise ValueError("Incomplete verdict set")
        by_index = {}
        for v in verdicts:
            j, allowed = v["index"], v["allowed"]
            if type(j) is not int or not 0 <= j < len(batch) or j in by_index or type(allowed) is not bool:
                raise ValueError("Invalid grounding verdict")
            by_index[j] = ("allowed" if allowed else "rejected", str(v.get("reason", ""))[:200])
        outcomes, error = [by_index[j] for j in range(len(batch))], None
    except Exception as exc:  # as verify_generated_locations: anything else is unverified
        outcomes, error = [("unverified", "")] * len(batch), type(exc).__name__
    seconds = round(time.monotonic() - start, 3)
    return {"cases": [{"id": c["id"], "outcome": o, "reason": r, "error": error}
                      for c, (o, r) in zip(batch, outcomes)], "seconds": seconds}


def provider_client(timeout: float) -> httpx.AsyncClient:
    base = os.environ["PERSONA_GROUNDING_BASE_URL"].rstrip("/")
    if not base.startswith("https://"):
        raise ValueError("Live calibration requires an explicit HTTPS provider")
    headers = {"Authorization": "Bearer " + os.environ["PERSONA_GROUNDING_API_KEY"]}
    return httpx.AsyncClient(base_url=base, headers=headers, timeout=timeout + 3)


async def replay(cases: list[dict], meta: dict, policy: str, model: str, concurrency: int) -> list[dict]:
    by_schedule: dict[str, list[dict]] = {}
    for c in sorted(cases, key=lambda c: (c["schedule"], c["slot"])):
        by_schedule.setdefault(c["schedule"], []).append(c)
    gate = asyncio.Semaphore(concurrency)

    async def one_schedule(client, schedule: str, slots: list[dict]) -> list[dict]:
        agent = meta["profiles"][meta["schedules"][schedule]["agent"]]
        facts = canonical_facts(SimpleNamespace(**agent))
        async with gate:
            return await asyncio.gather(*(
                verify(client, model, policy, facts, slots[k:k + BATCH]) for k in range(0, len(slots), BATCH)
            ))

    async with provider_client(GENERATION_TIMEOUT_S) as client:
        nested = await asyncio.gather(*(one_schedule(client, s, v) for s, v in by_schedule.items()))
    return [batch for schedule in nested for batch in schedule]


def summarize(cases: list[dict], meta: dict, batches: list[dict]) -> dict:
    label = {c["id"]: c["label"] for c in cases}
    outcome = {r["id"]: r for b in batches for r in b["cases"]}
    table = Counter((label[i], r["outcome"]) for i, r in outcome.items())
    seconds = sorted(b["seconds"] for b in batches)
    replaced = sum(r["outcome"] != "allowed" for r in outcome.values())
    total_slots = sum(s["slots"] for s in meta["schedules"].values())
    return {
        "cases": len(cases),
        "by_label": {f"{lab}/{out}": n for (lab, out), n in sorted(table.items())},
        # 漏放: a fabricated place that would reach live state
        "missed": sorted(i for i, r in outcome.items() if label[i] == "reject" and r["outcome"] == "allowed"),
        # 误杀: a generic or grounded slot replaced by a neutral activity
        "false_rejects": sorted(i for i, r in outcome.items() if label[i] == "allow" and r["outcome"] == "rejected"),
        "unverified": sorted(i for i, r in outcome.items() if r["outcome"] == "unverified"),
        "replaced_slots": f"{replaced}/{total_slots}",
        "batch_seconds": {"p50": statistics.median(seconds),
                          "p90": seconds[max(0, math.ceil(len(seconds) * 0.9) - 1)]},
    }


def run(args) -> None:
    cases, meta = validate()
    policy = load_policy(args.policy_ref)
    digest = hashlib.sha256(policy.encode()).hexdigest()[:12]
    text = {c["id"]: c["text"] for c in cases}
    runs = []
    for n in range(args.repeat):
        batches = asyncio.run(replay(cases, meta, policy, args.model, args.concurrency))
        summary = summarize(cases, meta, batches)
        runs.append({"summary": summary, "batches": batches})
        shown = {k: v for k, v in summary.items() if k not in {"missed", "false_rejects", "unverified"}}
        print(json.dumps({"run": n + 1, "policy": digest, **shown,
                          "missed": len(summary["missed"]), "false_rejects": len(summary["false_rejects"]),
                          "unverified": len(summary["unverified"])}, ensure_ascii=False))
        for kind in ("missed", "false_rejects"):
            for i in summary[kind]:
                print(f"  {kind}: {i} {text[i]}")
    RESULTS.mkdir(exist_ok=True)
    output = args.output or RESULTS / f"persona_grounding_schedule_{digest}.json"
    output.write_text(json.dumps({"model": args.model, "policy_ref": args.policy_ref or "working tree",
                                  "policy_sha256": digest, "runs": runs}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(output)}))
    # Unverified slots are replaced live too, and a dead provider must not pass
    # as "nothing missed, nothing rejected".
    if any(s["missed"] or len(s["false_rejects"]) > args.max_false_rejects
           or len(s["unverified"]) > args.max_unverified for s in (r["summary"] for r in runs)):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("export").add_argument("source", type=Path, help="psql -At JSONL export")
    sub.add_parser("validate")
    live = sub.add_parser("run")
    live.add_argument("--model", default="qwen3.5-plus")
    live.add_argument("--policy-ref", help="git ref to take the policy from; default: working tree")
    live.add_argument("--repeat", type=int, default=1)
    live.add_argument("--concurrency", type=int, default=4, help="schedules verified at once")
    live.add_argument("--max-false-rejects", type=int, default=0)
    live.add_argument("--max-unverified", type=int, default=15, help="timeouts/invalid verdicts per run")
    live.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "run" and args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if args.command == "export":
        export(args.source)
    elif args.command == "validate":
        cases, meta = validate()
        print(json.dumps({"schedules": len(meta["schedules"]), "cases": len(cases),
                          **Counter(c["label"] for c in cases)}))
    else:
        run(args)


if __name__ == "__main__":
    main()
