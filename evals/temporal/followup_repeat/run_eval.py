"""TIME06: score baseline lexical repeat check vs. candidate FollowupDecision on frozen cases."""
import json, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

NOW = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)


def history(case):
    return [{"id": f"m{i}", "role": role, "content": text,
             "created_at": (NOW - timedelta(minutes=minutes)).isoformat()}
            for i, (minutes, role, text) in enumerate(case["history"])]


class MemoryDB:
    """Serves the same two bounded reads as production (dense page, family ledger)."""

    def __init__(self, rows):
        self.rows = rows

    async def query_raw(self, _sql, _user, _workspace, now, days, limit, pattern=None):
        import re
        cutoff = datetime.fromisoformat(now) - timedelta(days=days)
        rows = [r for r in self.rows if cutoff < datetime.fromisoformat(r["created_at"])
                <= datetime.fromisoformat(now) and (pattern is None or re.search(pattern, r["content"]))]
        return sorted(rows, key=lambda r: r["created_at"], reverse=True)[:limit]


def baseline(case):
    from app.services.time_domain.followup import FollowupPolicy, repeats_recent_question, _SENTENCE, _is_question
    # Baseline = TIME04 behaviour: recent questions only within the 3h session.
    qs = tuple(s.strip() for minutes, role, text in case["history"] if role == "assistant" and minutes <= 180
               for s in _SENTENCE.findall(text) if _is_question(s))
    return "suppress" if repeats_recent_question(case["question"], FollowupPolicy(recent_questions=qs)) else "allow"


def candidate(case):
    import asyncio
    from app.services.time_domain.followup_decision import decide_reply
    text, decisions, _ = asyncio.run(decide_reply(
        case["question"], database=MemoryDB(history(case)), user_id="u", workspace_id="w",
        now=NOW, current_user_message=case["current_user_message"]))
    if not decisions:
        return "allow", "not_an_outcome_followup"
    return ("allow" if text else "suppress"), decisions[0].reason


def main(path):
    cases = json.loads(Path(path).read_text())["cases"]
    rows, b_ok, c_ok = [], 0, 0
    for case in cases:
        b = baseline(case)
        c, reason = candidate(case)
        b_ok += b == case["expected"]; c_ok += c == case["expected"]
        rows.append({"id": case["id"], "family": case["template_family"], "expected": case["expected"],
                     "baseline": b, "candidate": c, "reason": reason})
    out = {"cases": len(cases), "baseline_correct": b_ok, "candidate_correct": c_ok,
           "candidate_failures": [r for r in rows if r["candidate"] != r["expected"]],
           "false_suppress": sum(r["candidate"] == "suppress" and r["expected"] == "allow" for r in rows),
           "missed_repeat": sum(r["candidate"] == "allow" and r["expected"] == "suppress" for r in rows),
           "rows": rows}
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
