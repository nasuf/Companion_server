"""Synthetic live-model calibration; no DB, Redis, tracing, or real chats.

Set PERSONA_GROUNDING_API_KEY and PERSONA_GROUNDING_BASE_URL explicitly.
Results contain case IDs, verdicts and timing; never credentials or raw replies.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.services.prompting.defaults import PERSONA_GROUNDING_CHECK_PROMPT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.5-plus")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    cases = json.loads(Path(__file__).with_name("cases.json").read_text())
    assert len({c["id"] for c in cases}) == len(cases)
    assert all(c["city"] and (all(type(v["allowed"]) is bool and v["text"] for v in c["items"])
               if "items" in c else type(c["allowed"]) is bool and c["text"]) for c in cases)
    if args.validate_only:
        print(json.dumps({"cases": len(cases), "valid": True}))
        return
    base = os.environ["PERSONA_GROUNDING_BASE_URL"].rstrip("/")
    if not base.startswith("https://"):
        raise ValueError("Live calibration requires an explicit HTTPS provider")
    key = os.environ["PERSONA_GROUNDING_API_KEY"]
    results = []
    for case in cases:
        items = case.get("items", [{"text": case.get("text"), "allowed": case.get("allowed")}])
        expected = [x["allowed"] for x in items]
        prompt = PERSONA_GROUNDING_CHECK_PROMPT.format(
            facts=json.dumps({"name": "小伴", "city": case["city"], "occupation": "客服员", "age": 22}, ensure_ascii=False),
            kind=case.get("kind", "reply"), question=json.dumps(case.get("question", ""), ensure_ascii=False),
            items=json.dumps([{"index": i, "text": x["text"]} for i, x in enumerate(items)], ensure_ascii=False),
        )
        body = {"model": args.model, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0, "enable_thinking": False,
                "response_format": {"type": "json_object"}}
        request = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        start = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                reply = json.load(response)["choices"][0]["message"]["content"]
            verdict = json.loads(reply)["verdicts"]
            assert len(verdict) == len(items)
            assert all(type(x["index"]) is int and type(x["allowed"]) is bool for x in verdict)
            assert sorted(x["index"] for x in verdict) == list(range(len(items)))
            allowed = [x["allowed"] for x in sorted(verdict, key=lambda x: x["index"])]
            passed = allowed == expected
            error = None
        except Exception as exc:
            allowed, passed, error = None, False, type(exc).__name__
        results.append({"id": case["id"], "passed": passed, "allowed": allowed,
                        "expected": expected, "seconds": round(time.monotonic() - start, 3), "error": error})
        args.output.write_text(json.dumps({"model": args.model, "results": results}, indent=2) + "\n")
    print(json.dumps({"model": args.model, "passed": sum(x["passed"] for x in results), "total": len(cases)}))
    if not all(x["passed"] and x["seconds"] < 5 for x in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
