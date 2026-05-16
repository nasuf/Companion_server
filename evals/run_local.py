"""Run Companion agent eval cases.

Use --validate-only for CI-safe schema/grader validation. Server mode requires a
running backend, a conversation id, and either a bearer token or login creds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.graders import grade_reply, validate_case  # noqa: E402


DEFAULT_CASES = Path(__file__).with_name("cases.jsonl")
DEFAULT_RESULTS = Path(__file__).with_name("results") / "latest.json"


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            case["_line"] = line_no
            cases.append(case)
    return cases


def validate_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    seen: set[str] = set()
    for case in cases:
        errors = validate_case(case)
        case_id = str(case.get("id") or f"line:{case.get('_line')}")
        if case_id in seen:
            errors.append("duplicate id")
        seen.add(case_id)
        if errors:
            failures.append({"id": case_id, "line": case.get("_line"), "errors": errors})
    return failures


def _parse_sse(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    event: dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if event:
                events.append(event)
                event = {}
            continue
        if line.startswith("event:"):
            event["event"] = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            payload = line.removeprefix("data:").strip()
            try:
                event["data"] = json.loads(payload)
            except json.JSONDecodeError:
                event["data"] = payload
    if event:
        events.append(event)
    return events


def _auth_headers(client: httpx.Client, args: argparse.Namespace) -> dict[str, str]:
    token = args.token or os.getenv("COMPANION_EVAL_TOKEN")
    if token:
        return {"Authorization": f"Bearer {token}"}

    username = args.username or os.getenv("COMPANION_EVAL_USERNAME")
    password = args.password or os.getenv("COMPANION_EVAL_PASSWORD")
    if not username or not password:
        raise RuntimeError("server mode requires --token or --username/--password")
    resp = client.post("/auth/login", json={"username": username, "password": password})
    resp.raise_for_status()
    return {"Authorization": f"Bearer {resp.json()['token']}"}


def _latest_messages(
    client: httpx.Client,
    conversation_id: str,
    headers: dict[str, str],
    limit: int = 30,
) -> list[dict[str, Any]]:
    resp = client.get(
        f"/conversations/{conversation_id}/messages",
        params={"limit": limit, "include_metadata": "true"},
        headers=headers,
    )
    resp.raise_for_status()
    return resp.json()


def _wait_for_assistant_reply(
    client: httpx.Client,
    conversation_id: str,
    headers: dict[str, str],
    known_ids: set[str],
    timeout_s: float,
) -> list[dict[str, Any]]:
    deadline = time.time() + timeout_s
    found: dict[str, dict[str, Any]] = {}
    while time.time() < deadline:
        for message in _latest_messages(client, conversation_id, headers):
            if message.get("id") in known_ids:
                continue
            if message.get("role") == "assistant":
                found[str(message.get("id"))] = message
        if found:
            ordered = list(found.values())
            ordered.sort(key=lambda item: item.get("created_at") or "")
            return ordered
        time.sleep(1.0)
    return []


def run_case(
    client: httpx.Client,
    headers: dict[str, str],
    conversation_id: str,
    case: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    before = _latest_messages(client, conversation_id, headers)
    known_ids = {str(m.get("id")) for m in before if m.get("id")}
    sse_events: list[dict[str, Any]] = []

    for turn in case["turns"]:
        resp = client.post(
            f"/chat/{conversation_id}",
            json={"message": turn["content"]},
            headers=headers,
        )
        sse_events.extend(_parse_sse(resp.text))
        resp.raise_for_status()

    replies = _wait_for_assistant_reply(client, conversation_id, headers, known_ids, timeout_s)
    reply_text = "\n".join(str(item.get("content") or "") for item in replies)
    grade = grade_reply(reply_text, case["assertions"])
    return {
        "id": case["id"],
        "category": case["category"],
        "priority": case["priority"],
        "passed": grade["passed"],
        "reply": reply_text,
        "assistant_message_ids": [item.get("id") for item in replies],
        "sse_events": sse_events,
        "grade": grade,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--base-url", default=os.getenv("COMPANION_EVAL_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--conversation-id", default=os.getenv("COMPANION_EVAL_CONVERSATION_ID"))
    parser.add_argument("--token", default=None)
    parser.add_argument("--username", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--priority", choices=["P0", "P1", "P2"], action="append")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = load_cases(args.cases)
    if args.priority:
        allowed = set(args.priority)
        cases = [case for case in cases if case.get("priority") in allowed]

    validation_failures = validate_cases(cases)
    if validation_failures:
        print(json.dumps({"validation_failures": validation_failures}, ensure_ascii=False, indent=2))
        return 2

    if args.validate_only:
        print(json.dumps({"ok": True, "validated_cases": len(cases)}, ensure_ascii=False, indent=2))
        return 0

    if not args.conversation_id:
        raise RuntimeError("server mode requires --conversation-id")

    with httpx.Client(base_url=args.base_url, timeout=httpx.Timeout(args.timeout), trust_env=False) as client:
        headers = _auth_headers(client, args)
        health = client.get("/health")
        health.raise_for_status()
        results = [
            run_case(client, headers, args.conversation_id, case, timeout_s=args.timeout)
            for case in cases
        ]

    summary = {
        "passed": all(item["passed"] for item in results),
        "total": len(results),
        "failed": sum(1 for item in results if not item["passed"]),
        "results": results,
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

