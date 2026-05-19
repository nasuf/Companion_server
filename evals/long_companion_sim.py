"""Deterministic long-companion transcript simulation checks.

This module scores a 30-day style transcript without LLM calls. It is meant to
catch obvious regressions before deeper server-mode evals exist: persona leaks,
mechanical comfort loops, goal continuity loss, and overactive proactive sends.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

PERSONA_LEAKS = ("作为AI", "作为人工智能", "语言模型", "机器人")
MECHANICAL_COMFORT = ("积极一点", "想开点", "不要焦虑就好", "加油就好了")
GOAL_TERMS = ("睡前复盘", "代码学习")


def _norm(text: str) -> str:
    return "".join(str(text or "").split()).lower()


def _contains_any(text: str, values: tuple[str, ...]) -> bool:
    haystack = _norm(text)
    return any(_norm(value) in haystack for value in values)


def build_reference_transcript(days: int = 30) -> list[dict[str, Any]]:
    """A passing synthetic transcript used for CI-safe harness validation."""
    rows: list[dict[str, Any]] = [
        {
            "day": 1,
            "role": "user",
            "content": "我想开始睡前复盘，把代码学习坚持下来。",
        },
        {
            "day": 1,
            "role": "assistant",
            "content": "我记住这个方向：睡前复盘服务于代码学习，今晚先写十分钟就好。",
        },
    ]
    for day in range(2, days + 1):
        if day in {5, 12, 19, 26}:
            rows.append({
                "day": day,
                "role": "assistant",
                "content": "轻轻提醒一下：今天的代码学习睡前复盘，写一个卡点就够。",
                "proactive": True,
            })
        if day in {7, 14, 21, 28}:
            rows.append({
                "day": day,
                "role": "user",
                "content": "这周推进得有点慢。",
            })
            rows.append({
                "day": day,
                "role": "assistant",
                "content": "慢也算在路上。我们先看代码学习里最小的一步，睡前复盘只记一个问题。",
            })
    return rows


def load_transcript(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("transcript must be a JSON array")
    return data


def validate_transcript(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if not rows:
        return ["transcript must not be empty"]
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"rows[{index}] must be an object")
            continue
        if row.get("role") not in {"user", "assistant"}:
            errors.append(f"rows[{index}].role must be user or assistant")
        try:
            day = int(row.get("day"))
        except (TypeError, ValueError):
            errors.append(f"rows[{index}].day must be an integer")
        else:
            if day < 1:
                errors.append(f"rows[{index}].day must be positive")
        if not isinstance(row.get("content"), str) or not row["content"].strip():
            errors.append(f"rows[{index}].content must be a non-empty string")
    return errors


def score_transcript(rows: list[dict[str, Any]]) -> dict[str, Any]:
    assistant_rows = [row for row in rows if row.get("role") == "assistant"]
    persona_leak_count = sum(
        1 for row in assistant_rows if _contains_any(str(row.get("content")), PERSONA_LEAKS)
    )
    mechanical_comfort_count = sum(
        1 for row in assistant_rows if _contains_any(str(row.get("content")), MECHANICAL_COMFORT)
    )

    goal_introduced = any(
        row.get("role") == "user" and _contains_any(str(row.get("content")), GOAL_TERMS)
        for row in rows
    )
    goal_mentions_after_intro = 0
    seen_goal = False
    for row in rows:
        if row.get("role") == "user" and _contains_any(str(row.get("content")), GOAL_TERMS):
            seen_goal = True
            continue
        if seen_goal and row.get("role") == "assistant" and _contains_any(str(row.get("content")), GOAL_TERMS):
            goal_mentions_after_intro += 1

    proactive_by_day: Counter[int] = Counter()
    for row in assistant_rows:
        if row.get("proactive"):
            proactive_by_day[int(row.get("day") or 0)] += 1
    max_proactive_per_day = max(proactive_by_day.values(), default=0)
    proactive_days = sorted(day for day, count in proactive_by_day.items() if count > 0)

    passed = (
        persona_leak_count == 0
        and mechanical_comfort_count == 0
        and (not goal_introduced or goal_mentions_after_intro >= 3)
        and max_proactive_per_day <= 3
    )
    return {
        "passed": passed,
        "metrics": {
            "assistant_turns": len(assistant_rows),
            "persona_leak_count": persona_leak_count,
            "mechanical_comfort_count": mechanical_comfort_count,
            "goal_introduced": goal_introduced,
            "goal_mentions_after_intro": goal_mentions_after_intro,
            "max_proactive_per_day": max_proactive_per_day,
            "proactive_days": proactive_days,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--results", type=Path, default=Path("evals/results/long_companion_latest.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_reference_transcript() if args.validate_only or not args.transcript else load_transcript(args.transcript)
    errors = validate_transcript(rows)
    result = {"validation_errors": errors, **score_transcript(rows)}
    if args.results:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors and result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
