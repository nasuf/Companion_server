"""LongMemEval adapter 的正确性.

这套东西的输出是我们判断"时间感知做得怎么样"的唯一依据 —— 指标算错了不会报错,
只会给出一个看起来合理但错误的数字, 然后我们照着它改代码。所以切分和计数这两块
必须锁死。
"""

from __future__ import annotations

import json
from pathlib import Path

from evals.external.longmemeval import (
    MAX_TURN_CHARS,
    all_evidence_at_k,
    load_temporal_cases,
    parse_lme_date,
    recall_at_k,
)
from evals.external.longmemeval import _rounds_of_session as rounds_of


def _turn(role: str, content: str, has_answer: bool = False) -> dict:
    t = {"role": role, "content": content}
    if has_answer:
        t["has_answer"] = True
    return t


class TestRoundSplitting:
    def test_one_user_plus_replies_is_one_round(self):
        session = [
            _turn("user", "问题一"),
            _turn("assistant", "回答一"),
            _turn("user", "问题二"),
            _turn("assistant", "回答二"),
        ]
        rounds = rounds_of(session, 0, None, "q1")
        assert len(rounds) == 2
        assert "问题一" in rounds[0].text and "回答一" in rounds[0].text
        assert "问题二" in rounds[1].text

    def test_evidence_on_the_assistant_side_still_marks_the_round(self):
        """标签可能落在任一侧.

        只看 user 侧会漏掉"答案在 AI 回复里"的题 (LongMemEval 有
        single-session-assistant 这一类), 召回率会被系统性低估。
        """
        session = [_turn("user", "问"), _turn("assistant", "答", has_answer=True)]
        assert rounds_of(session, 0, None, "q")[0].is_evidence is True

    def test_evidence_on_the_user_side_marks_the_round(self):
        session = [_turn("user", "问", has_answer=True), _turn("assistant", "答")]
        assert rounds_of(session, 0, None, "q")[0].is_evidence is True

    def test_round_without_label_is_not_evidence(self):
        session = [_turn("user", "闲聊"), _turn("assistant", "嗯")]
        assert rounds_of(session, 0, None, "q")[0].is_evidence is False

    def test_ids_are_unique_within_a_question(self):
        session = [_turn("user", f"第{i}问") for i in range(5)]
        ids = [r.id for r in rounds_of(session, 3, None, "qX")]
        assert len(set(ids)) == len(ids)
        assert all(i.startswith("qX:s3:r") for i in ids)

    def test_long_turns_are_truncated(self):
        """不截会把本地 embedder 打挂; 而且不截也不代表我们的真实行为 —— 我们存的
        是抽取后的短事实, 不是整段对话。"""
        session = [_turn("user", "x" * 5000)]
        text = rounds_of(session, 0, None, "q")[0].text
        assert len(text) < MAX_TURN_CHARS + 40

    def test_session_starting_with_assistant_does_not_crash(self):
        session = [_turn("assistant", "开场白"), _turn("user", "回应")]
        rounds = rounds_of(session, 0, None, "q")
        assert len(rounds) == 2

    def test_empty_session(self):
        assert rounds_of([], 0, None, "q") == []


class TestMetrics:
    def test_recall_counts_coverage_not_any_hit(self):
        """order 题要两条证据都召回主模型才比得了, 只中一条不能算通过."""
        assert recall_at_k(["a", "b", "c"], {"a", "z"}, 3) == 0.5

    def test_recall_respects_k(self):
        assert recall_at_k(["x", "y", "a"], {"a"}, 2) == 0.0
        assert recall_at_k(["x", "y", "a"], {"a"}, 3) == 1.0

    def test_all_evidence_requires_every_piece(self):
        assert all_evidence_at_k(["a", "b"], {"a", "b"}, 2) is True
        assert all_evidence_at_k(["a", "b"], {"a", "b", "c"}, 2) is False

    def test_empty_evidence_is_not_a_pass(self):
        """没有证据标注的题不能算满分 —— 那会把数据缺陷记成系统能力."""
        assert recall_at_k(["a"], set(), 5) == 0.0
        assert all_evidence_at_k(["a"], set(), 5) is False


class TestGranularityComparability:
    """两种粒度的分数必须同口径, 否则拿来对比就是自欺.

    chunk 模式下一轮会切出好几块全标成证据, 直接算的话"证据条数"跟 round 模式不是
    一回事 —— 分数低不代表更差, 只代表分母变了。收敛到轮次级再计分才能比。
    """

    def test_chunk_ids_collapse_to_their_turn(self):
        from evals.external.longmemeval import parent_turn

        assert parent_turn("q:s1:t2:c3") == "q:s1:t2"
        assert parent_turn("q:s1:r0") == "q:s1:r0"  # round id 不带 :c, 原样返回

    def test_multiple_chunks_of_one_turn_count_once(self):
        ranked = ["q:s0:t0:c0", "q:s0:t0:c1", "q:s0:t0:c2", "q:s9:t1:c0"]
        # 前三个同属一轮; top2(轮次级) = {s0:t0, s9:t1}, 证据全中
        assert all_evidence_at_k(ranked, {"q:s9:t1:c0"}, 2) is True

    def test_round_mode_behaviour_is_unchanged(self):
        """round id 里没有 :c, 收敛是恒等变换 —— 老基线数字不受影响."""
        assert recall_at_k(["a:r0", "b:r1"], {"a:r0"}, 1) == 1.0
        assert recall_at_k(["b:r1", "a:r0"], {"a:r0"}, 1) == 0.0


class TestLoading:
    def test_date_parsing(self):
        d = parse_lme_date("2023/04/10 (Mon) 17:50")
        assert d is not None and (d.year, d.month, d.day, d.hour) == (2023, 4, 10, 17)

    def test_bad_date_is_none_not_a_crash(self):
        assert parse_lme_date("not a date") is None
        assert parse_lme_date("") is None

    def test_only_temporal_questions_are_loaded(self, tmp_path: Path):
        data = [
            {
                "question_id": "t1", "question_type": "temporal-reasoning",
                "question": "Q", "answer": "A", "question_date": "2023/04/10 (Mon) 17:50",
                "haystack_dates": ["2023/04/09 (Sun) 10:00"],
                "haystack_sessions": [[_turn("user", "hi", has_answer=True)]],
            },
            {
                "question_id": "m1", "question_type": "multi-session",
                "question": "Q", "answer": "A", "question_date": "2023/04/10 (Mon) 17:50",
                "haystack_dates": ["2023/04/09 (Sun) 10:00"],
                "haystack_sessions": [[_turn("user", "hi", has_answer=True)]],
            },
        ]
        p = tmp_path / "d.json"
        p.write_text(json.dumps(data))
        cases = load_temporal_cases(p)
        assert [c.question_id for c in cases] == ["t1"]

    def test_questions_without_evidence_are_skipped(self, tmp_path: Path):
        """没有标注就测不了召回。跳过而不是当成失败 —— 否则分数被数据缺陷拉低."""
        data = [{
            "question_id": "t1", "question_type": "temporal-reasoning",
            "question": "Q", "answer": "A", "question_date": "2023/04/10 (Mon) 17:50",
            "haystack_dates": ["2023/04/09 (Sun) 10:00"],
            "haystack_sessions": [[_turn("user", "no label")]],
        }]
        p = tmp_path / "d.json"
        p.write_text(json.dumps(data))
        assert load_temporal_cases(p) == []

    def test_missing_dates_do_not_crash(self, tmp_path: Path):
        """haystack_dates 比 sessions 短时不能越界 —— 真实数据里出现过."""
        data = [{
            "question_id": "t1", "question_type": "temporal-reasoning",
            "question": "Q", "answer": "A", "question_date": "2023/04/10 (Mon) 17:50",
            "haystack_dates": [],
            "haystack_sessions": [[_turn("user", "hi", has_answer=True)]],
        }]
        p = tmp_path / "d.json"
        p.write_text(json.dumps(data))
        cases = load_temporal_cases(p)
        assert len(cases) == 1
        assert cases[0].rounds[0].at is None
