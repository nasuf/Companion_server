from pathlib import Path

from evals.graders import grade_reply
from evals.run_local import _grade_text_for_case, load_cases, validate_cases


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


def test_p2_goal_tracking_case_fails_when_goal_is_lost():
    cases = load_cases(ROOT / "evals" / "cases.jsonl")
    case = next(c for c in cases if c["id"] == "p2_relationship_goal_tracking_multiturn")
    result = grade_reply("今晚你可以先放松一下。", case["assertions"])
    assert result["passed"] is False


def test_p2_goal_tracking_case_accepts_specific_continuity():
    cases = load_cases(ROOT / "evals" / "cases.jsonl")
    case = next(c for c in cases if c["id"] == "p2_relationship_goal_tracking_multiturn")
    result = grade_reply(
        "今晚先做十分钟睡前复盘，把代码学习里卡住的一点写下来就好。",
        case["assertions"],
    )
    assert result["passed"] is True


def test_eval_grade_target_can_use_last_reply_only():
    case = {"grade_target": "last_reply"}
    replies = [
        {"content": "第一轮里可以有别的内容"},
        {"content": "最后一轮才是要评分的回复"},
    ]
    assert _grade_text_for_case(replies, case) == "最后一轮才是要评分的回复"
