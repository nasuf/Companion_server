"""Replay real chat replies and AI memories through persona.grounding_check.

The same policy guards three paths. schedule_replay.py covers schedules; this
covers the two that verify one text at a time:

  reply   guard_reply: an assistant turn (its bubbles joined by a space) with
          the user message it answers as `question`; a location question
          forces verification (`always`). A rejected reply is replaced
          wholesale by a "I got the place wrong" correction.
  memory  _generated_self_memory_allowed: a generated AI-side memory (not a
          profile/knowledge seed, not a stable identity claim). A rejected
          memory is not stored.

Cases are the gated texts of production 2026-09-16..10-16. Before grounding
went live (10-09 17:19) the text is the model's original; after it, only
what the old policy let through survives, which is still what a new policy
must not start rejecting. Labels are by hand:

  reject  an Agent location the profile does not support: a named place
          absent from the profile, a claim of being in / having been to
          another city, denying the profile city
  allow   the user's places and experiences, generic places, profile places,
          discussion and plans, dish names, no place at all
  either  cannot be decided from the verifier's inputs (e.g. "is your home
          near here?" without knowing where the user lives), or the profile
          names the neighbourhood but not that exact street (平江路 for
          平江街道)

The case files (reply_memory_replay.jsonl, reply_memory_replay_profiles.json)
are production data and are not in the repository: export them locally with
the SQL below and label them by hand. Export hashes agent and memory ids,
keeps no user id and strips shared coordinates; users' names, home addresses
and street-level locations in questions and replies must be redacted by hand.
A re-export keeps the labels of unchanged texts, so redacted texts come back
unlabelled and must be redacted and labelled again.

Subcommands mirror schedule_replay.py. `run` uses the hot-path budget of 5s
per call (VERIFY_TIMEOUT_S); a reply or memory that cannot be verified is
treated like a rejected one in production.

Export (read-only, on the database host), then
`python evals/persona_grounding/reply_memory_replay.py export messages.jsonl memories.jsonl`:

  SELECT json_build_object('conversation_id', m.conversation_id, 'agent_id', c.agent_id,
         'role', m.role, 'content', m.content, 'created_at', m.created_at,
         'proactive', coalesce(m.metadata->>'proactive', 'false'), 'name', a.name,
         'nickname', to_jsonb(a) ->> 'nickname', 'city', a.city, 'occupation', a.occupation, 'age', a.age)
  FROM messages m JOIN conversations c ON c.id = m.conversation_id JOIN ai_agents a ON a.id = c.agent_id
  WHERE m.created_at >= '2026-09-16' AND m.role IN ('user', 'assistant')
  ORDER BY m.conversation_id, m.created_at, m.id;

  SELECT json_build_object('memory_id', m.id, 'agent_id', w.agent_id, 'content', m.content,
         'provenance', m.provenance, 'created_at', m.created_at, 'name', a.name,
         'nickname', to_jsonb(a) ->> 'nickname', 'city', a.city, 'occupation', a.occupation, 'age', a.age)
  FROM memories_ai m JOIN chat_workspaces w ON w.id = m.workspace_id JOIN ai_agents a ON a.id = w.agent_id
  WHERE m.created_at >= '2026-09-16' AND coalesce(m.provenance, '') NOT IN ('profile_seed', 'knowledge_seed')
  ORDER BY w.agent_id, m.created_at, m.id;
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schedule_replay import RESULTS, agent_key, load_policy, provider_client, require_data, verify  # noqa: E402
from app.services.persona_grounding import (  # noqa: E402
    canonical_facts, generated_stable_self_claim, needs_location_verification,
)

HERE = Path(__file__).resolve().parent
CASES = HERE / "reply_memory_replay.jsonl"
PROFILES = HERE / "reply_memory_replay_profiles.json"
LABELS = {"reject", "allow", "either"}
HOT_PATH_TIMEOUT_S = 5
GO_LIVE = "2026-10-09T17:19"
# guard_reply's test for a location question (a rejected rewrite of it would
# make the eval silently diverge, so validate checks it against the source).
LOCATION_QUESTION = (r"你.{0,16}(?:住|(?<!现)在(?!\s*(?:干嘛|干什么|做什么|忙|吃|喝|想|看|听|写|工作|学习|睡|休息|玩|等|陪))"
                     r".{0,8}(?:哪|吗|？|\?)|哪(?:里|儿|个城市)|搬家|回家)")
_COORDINATES = re.compile(r"[（(]约?\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*[）)]")


def load_cases() -> list[dict]:
    return [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]


def _profile(row: dict) -> dict:
    return {k: row[k] for k in ("name", "nickname", "city", "occupation", "age") if row.get(k)}


def _turns(rows: list[dict]):
    """Assistant bubbles between two user messages form one guarded turn."""
    turn, asked = None, []
    for row in rows:
        if turn and (row["conversation_id"] != turn["conversation_id"]
                     or row["role"] != "assistant" or row["proactive"] == "true"):
            yield turn
            turn = None
        if asked and asked[-1]["conversation_id"] != row["conversation_id"]:
            asked = []
        if row["role"] == "user":
            asked.append(row)
        elif row["proactive"] != "true":
            if turn is None:
                turn = {**row, "parts": [], "question": "".join(r["content"] for r in asked)}
                asked = []
            turn["parts"].append(row["content"])
    if turn:
        yield turn


def export(messages: Path, memories: Path) -> None:
    previous = {c["id"]: c for c in load_cases()} if CASES.exists() else {}
    profiles, cases = {}, []

    def add(case_id: str, row: dict, kind: str, text: str, **extra) -> None:
        key = agent_key(row["agent_id"])
        profiles[key] = _profile(row)
        old = previous.get(case_id)
        kept = old if old and old["text"] == text else {}
        period = "pre" if str(row["created_at"]) < GO_LIVE else "post"
        cases.append({"id": case_id, "kind": kind, "agent": key, "period": period, "text": text, **extra,
                      "label": kept.get("label"), "note": kept.get("note", "")})

    rows = [json.loads(line) for line in messages.read_text().splitlines() if line.strip()]
    for turn in _turns(rows):
        text = " ".join(turn["parts"])
        question = _COORDINATES.sub("", turn["question"])
        always = bool(re.search(LOCATION_QUESTION, question))
        if always or needs_location_verification(text):
            digest = hashlib.sha256(f"{turn['conversation_id']}{turn['created_at']}".encode()).hexdigest()[:10]
            add(f"r-{digest}", turn, "reply", text, question=question, always=always)
    for line in memories.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        text = row["content"]
        if generated_stable_self_claim(text) or not needs_location_verification(text, implicit_self=True):
            continue
        add(f"m-{hashlib.sha256(row['memory_id'].encode()).hexdigest()[:10]}", row, "memory", text,
            provenance=row.get("provenance"))
    CASES.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases))
    PROFILES.write_text(json.dumps(profiles, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cases": len(cases), **Counter(c["kind"] for c in cases),
                      "unlabelled": sum(c["label"] is None for c in cases)}))


def validate() -> tuple[list[dict], dict]:
    import inspect
    from app.services import persona_grounding

    require_data(CASES, PROFILES)
    cases, profiles = load_cases(), json.loads(PROFILES.read_text())
    problems = []
    if LOCATION_QUESTION not in inspect.getsource(persona_grounding.guard_reply).replace('"\n        r"', ""):
        problems.append("guard_reply's location-question pattern changed; update LOCATION_QUESTION")
    if len({c["id"] for c in cases}) != len(cases):
        problems.append("duplicate case ids")
    for c in cases:
        if not profiles.get(c["agent"], {}).get("city"):
            problems.append(f"{c['id']}: profile city missing")
        if c["label"] not in LABELS or not c["note"].strip():
            problems.append(f"{c['id']}: needs a label in {sorted(LABELS)} and a note")
        gated = (c["always"] or needs_location_verification(c["text"])) if c["kind"] == "reply" else (
            not generated_stable_self_claim(c["text"]) and needs_location_verification(c["text"], implicit_self=True))
        if not gated:
            problems.append(f"{c['id']}: no longer gated, re-export")
    if problems:
        raise SystemExit("\n".join(problems[:50]))
    return cases, profiles


async def replay(cases: list[dict], profiles: dict, policy: str, model: str, concurrency: int) -> list[dict]:
    gate = asyncio.Semaphore(concurrency)

    async def one(client, case: dict) -> dict:
        facts = canonical_facts(SimpleNamespace(**profiles[case["agent"]]))
        async with gate:
            return await verify(client, model, policy, facts, [case], kind=case["kind"],
                                question=case.get("question", ""), timeout=HOT_PATH_TIMEOUT_S)

    async with provider_client(HOT_PATH_TIMEOUT_S) as client:
        return await asyncio.gather(*(one(client, c) for c in cases))


def summarize(cases: list[dict], calls: list[dict]) -> dict:
    case = {c["id"]: c for c in cases}
    outcome = {r["id"]: r["outcome"] for b in calls for r in b["cases"]}
    table = Counter((case[i]["kind"], case[i]["label"], o) for i, o in outcome.items())
    return {
        "cases": len(cases),
        "by_kind_label": {f"{k}/{lab}/{o}": n for (k, lab, o), n in sorted(table.items())},
        "missed": sorted(i for i, o in outcome.items() if case[i]["label"] == "reject" and o == "allowed"),
        "false_rejects": sorted(i for i, o in outcome.items() if case[i]["label"] == "allow" and o == "rejected"),
        "unverified": sorted(i for i, o in outcome.items() if o == "unverified"),
    }


def run(args) -> None:
    cases, profiles = validate()
    if args.kind != "all":
        cases = [c for c in cases if c["kind"] == args.kind]
    policy = load_policy(args.policy_ref)
    digest = hashlib.sha256(policy.encode()).hexdigest()[:12]
    text = {c["id"]: f"[{c['kind']}] {c['text'][:80]}" for c in cases}
    runs = []
    for n in range(args.repeat):
        calls = asyncio.run(replay(cases, profiles, policy, args.model, args.concurrency))
        summary = summarize(cases, calls)
        runs.append({"summary": summary, "calls": calls})
        print(json.dumps({"run": n + 1, "policy": digest, "cases": summary["cases"],
                          "by_kind_label": summary["by_kind_label"], "missed": len(summary["missed"]),
                          "false_rejects": len(summary["false_rejects"]),
                          "unverified": len(summary["unverified"])}, ensure_ascii=False))
        for kind in ("missed", "false_rejects"):
            for i in summary[kind]:
                print(f"  {kind}: {i} {text[i]}")
    RESULTS.mkdir(exist_ok=True)
    output = args.output or RESULTS / f"persona_grounding_reply_memory_{digest}.json"
    output.write_text(json.dumps({"model": args.model, "policy_ref": args.policy_ref or "working tree",
                                  "policy_sha256": digest, "runs": runs}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(output)}))
    if any(s["missed"] or len(s["false_rejects"]) > args.max_false_rejects
           or len(s["unverified"]) > args.max_unverified for s in (r["summary"] for r in runs)):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("messages", type=Path)
    exp.add_argument("memories", type=Path)
    sub.add_parser("validate")
    live = sub.add_parser("run")
    live.add_argument("--kind", choices=["reply", "memory", "all"], default="all")
    live.add_argument("--model", default="qwen3.5-plus")
    live.add_argument("--policy-ref", help="git ref to take the policy from; default: working tree")
    live.add_argument("--repeat", type=int, default=1)
    live.add_argument("--concurrency", type=int, default=8)
    live.add_argument("--max-false-rejects", type=int, default=0)
    live.add_argument("--max-unverified", type=int, default=30, help="5s timeouts/invalid verdicts per run")
    live.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "run" and args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if args.command == "export":
        export(args.messages, args.memories)
    elif args.command == "validate":
        cases, profiles = validate()
        print(json.dumps({"agents": len(profiles), "cases": len(cases),
                          **Counter(f"{c['kind']}/{c['label']}" for c in cases)}))
    else:
        run(args)


if __name__ == "__main__":
    main()
