from pathlib import Path

from evals.graders import grade_reply
from evals.run_local import load_cases, validate_cases


ROOT = Path(__file__).resolve().parents[1]


def test_eval_cases_are_valid():
    cases = load_cases(ROOT / "evals" / "cases.jsonl")
    assert cases
    assert validate_cases(cases) == []


def test_memory_hallucination_case_fails_on_specific_invention():
    cases = load_cases(ROOT / "evals" / "cases.jsonl")
    case = next(c for c in cases if c["id"] == "memory_no_unsupported_preference")
    result = grade_reply("我记得你最喜欢的歌手是周兴哲。", case["assertions"])
    assert result["passed"] is False
    assert result["n_failed"] >= 1


def test_memory_hallucination_case_accepts_hedged_no_evidence_reply():
    cases = load_cases(ROOT / "evals" / "cases.jsonl")
    case = next(c for c in cases if c["id"] == "memory_no_unsupported_preference")
    result = grade_reply("我这里没有看到你明确说过最喜欢的歌手，所以我不确定。", case["assertions"])
    assert result["passed"] is True

