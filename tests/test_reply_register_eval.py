"""Guards for the reply-register eval's non-LLM parts.

The eval itself needs a server and real model calls, so it cannot run in CI.
Its deterministic pieces can, and they are the pieces that decide whether a
threshold is met — a silent regression here would move the numbers without
anything changing in the product.
"""

from __future__ import annotations

import pytest

from evals.reply_register import judge as J
from evals.reply_register.cases import ALL_CASES, GROUPS
from evals.reply_register.standard import ACKNOWLEDGE_STRATEGIES, ACTION_STRATEGIES


class TestCaseBank:
    def test_ids_unique_and_groups_known(self):
        ids = [c.id for c in ALL_CASES]
        assert len(ids) == len(set(ids))
        assert {c.group for c in ALL_CASES} == set(GROUPS)

    def test_each_group_has_enough_cases(self):
        for group in GROUPS:
            assert sum(1 for c in ALL_CASES if c.group == group) >= 20

    def test_messages_match_real_im_length(self):
        """真实中文 IM 是 5.64 字/行 —— 案例库偏离太多就测不到真实分布."""
        avg = sum(len(c.message) for c in ALL_CASES) / len(ALL_CASES)
        assert 4.0 <= avg <= 9.0, f"avg message length {avg:.1f} 偏离 IM 基准过远"

    def test_production_failure_is_kept_verbatim(self):
        """golden case 改写就失去了意义 (含原句的错字与两问合一)."""
        case = next(c for c in ALL_CASES if c.id == "fact_yongle_palace")
        assert case.message == "你知道运城永乐宫建于哪一年 有那怎样的历史故事吗？"


class TestFormatMetrics:
    def test_counts_bubbles_and_strips_emotion_marker(self):
        m = J.analyse_format("元代建的||1247年就有了||你要去玩吗 [EMO:中性/50]")
        assert m.bubbles == 3
        assert m.max_bubble_chars == 8
        assert m.format_ok

    def test_blank_lines_count_as_bubble_separators(self):
        assert J.analyse_format("第一句\n\n第二句").bubbles == 2

    def test_emoji_counted_individually(self):
        assert J.analyse_format("好呀🙂").emoji_count == 1
        assert J.analyse_format("好呀🙂😊").emoji_count == 2
        assert not J.analyse_format("好呀🙂😊").format_ok

    def test_long_bubble_fails_format(self):
        assert not J.analyse_format("一" * 21).format_ok


class TestVerdictParsing:
    def test_parses_json_verdict(self):
        assert J.parse_verdict("fact", '{"verdict": "encyclopedic", "reason": "只有事实"}') == (
            "encyclopedic"
        )

    def test_falls_back_to_bare_label_longest_first(self):
        # "question" 是 "providing_suggestions" 之外的独立标签, 但短标签不能
        # 抢先命中长标签的子串.
        assert J.parse_verdict("emotion", "providing_suggestions") == "providing_suggestions"

    def test_unparseable_returns_none(self):
        """解析不出来必须是 None — 静默归类会把评审故障算成真实判定."""
        assert J.parse_verdict("fact", "我觉得还行") is None


class TestAcknowledgementRule:
    """情绪首句是否"先接住" —— 词法判定, 刻意不经过评审器.

    评审器在这条边界上跨轮翻供, 且错判全部偏向通过侧 (详见 judge.py 注释),
    而阈值恰好落在它的摇摆区间里.
    """

    @pytest.mark.parametrize("reply", [
        "啊？怎么了？", "唉 咋啦", "辛苦了 咋了 忙啥了", "太难受了吧",
        "那也太好啦！", "哟 啥好事说来听听", "天啊 怎么会这样",
    ])
    def test_acknowledged(self, reply):
        assert J.opens_with_acknowledgment(reply)

    @pytest.mark.parametrize("reply", [
        "怎么了呀？", "为什么", "是有心事吗",
        "怎么更烦了呀？ 是又碰到糟心事了吗？",
        "怎么回事呀||是有什么烦心事吗",
    ])
    def test_bare_probe_is_not_acknowledgement(self, reply):
        assert not J.opens_with_acknowledgment(reply)

    def test_only_the_first_bubble_counts(self):
        """第二句才安慰不算 —— 产品规则要的是"第一句先接住"."""
        assert not J.opens_with_acknowledgment("怎么了呀||太难受了吧")

    def test_advice_first_beats_a_sympathetic_opener(self):
        """建议类由评审器判, 优先级高于词法接应: 先安慰再教方法仍是越过情绪办事."""
        assert J.classify_emotion_opening(
            "providing_suggestions", "辛苦了 你可以试试深呼吸",
        ) == "advice_first"

    def test_taxonomy_labels_still_map(self):
        assert J.classify_emotion_opening("reflection_of_feelings", "") == "acknowledge_first"
        assert J.classify_emotion_opening("question", "怎么了呀") == "question_first"


def test_calibration_labels_are_valid():
    """校准样本贴了不存在的标签就永远判 MISS, 把评测整个卡死."""
    for group, _message, _reply, expected in J.CALIBRATION:
        assert expected in J._VALID_VERDICTS[group], f"{group}: {expected}"


def test_strategy_sets_do_not_overlap():
    assert not (ACKNOWLEDGE_STRATEGIES & ACTION_STRATEGIES)
