from pathlib import Path

from evals.graders import grade_reply
from evals.long_companion_sim import build_reference_transcript, score_transcript, validate_transcript
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


def test_long_companion_reference_simulation_passes():
    rows = build_reference_transcript()
    assert validate_transcript(rows) == []
    result = score_transcript(rows)
    assert result["passed"] is True
    assert result["metrics"]["goal_mentions_after_intro"] >= 3


def test_long_companion_simulation_fails_persona_leak_and_overactive_proactive():
    rows = [
        {"day": 1, "role": "user", "content": "我想睡前复盘，坚持代码学习。"},
        {"day": 1, "role": "assistant", "content": "作为AI，我会提醒你。"},
        {"day": 2, "role": "assistant", "content": "积极一点就好了。", "proactive": True},
        {"day": 2, "role": "assistant", "content": "继续提醒。", "proactive": True},
        {"day": 2, "role": "assistant", "content": "继续提醒。", "proactive": True},
        {"day": 2, "role": "assistant", "content": "继续提醒。", "proactive": True},
    ]
    result = score_transcript(rows)
    assert result["passed"] is False
    assert result["metrics"]["persona_leak_count"] == 1
    assert result["metrics"]["mechanical_comfort_count"] == 1
    assert result["metrics"]["max_proactive_per_day"] == 4
