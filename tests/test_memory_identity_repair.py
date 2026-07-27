"""身份事实的确定性兜底.

生产 2026-07-27, 82 条用户记忆里 12 条身份类, 5 条落进「身份/其他」并被压到
0.80 (L2). 同一句式「用户叫X」在同一个模型上 2 对 2 —— 07-19 13:57 判对、
13:58 判错. 类目字段同时决定能不能进 L1 和走不走 SINGLETON 去重, 不能只靠模型
的单次发挥.
"""

from __future__ import annotations

import pytest

from app.services.memory.recording.identity_repair import (
    IDENTITY_IMPORTANCE_FLOOR,
    REPAIRABLE_SUBS,
    detect_identity_sub,
    repair_identity_classification,
)
from app.services.memory.taxonomy import L1_SINGLETON_SUBS


def _repair(summary: str, main: str = "身份", sub: str = "其他", imp: float = 0.80):
    return repair_identity_classification(
        summary=summary, main_category=main, sub_category=sub, importance=imp,
    )


class TestProductionFailures:
    """五条真实误分类, 逐条锁死."""

    @pytest.mark.parametrize("summary,want_sub", [
        ("用户叫李杰", "姓名"),
        ("用户叫Kiki", "姓名"),
        ("用户要求他人称呼自己为阿山", "姓名"),
        ("用户是广东的", "出生地"),
        ("用户称和AI说过自己是男生", "性别"),
    ])
    def test_repaired_to_the_right_sub_and_promoted(self, summary, want_sub):
        main, sub, imp, reason = _repair(summary)
        assert (main, sub) == ("身份", want_sub)
        assert imp == IDENTITY_IMPORTANCE_FLOOR
        assert reason


class TestDoesNotOverreach:
    """误判成身份比漏判严重得多 —— 它会污染 L1 和 singleton 槽位."""

    @pytest.mark.parametrize("summary,main,sub", [
        ("用户喜欢看哈利波特", "偏好", "审美爱好"),
        ("用户不喜欢打游戏", "偏好", "生活习惯"),
        ("用户说他同事叫李杰", "生活", "人际"),
        ("用户提到公司叫星澜互娱", "生活", "工作"),
        ("用户觉得住在大城市压力大", "思维", "社会观点"),
        ("用户在讨论男生女生的差异", "思维", "社会观点"),
    ])
    def test_non_identity_left_alone(self, summary, main, sub):
        got_main, got_sub, imp, reason = _repair(summary, main=main, sub=sub, imp=0.80)
        assert (got_main, got_sub, imp) == (main, sub, 0.80)
        assert reason is None

    def test_third_party_name_is_not_the_user_name(self):
        """「用户说他同事叫李杰」不能被当成用户自己的姓名."""
        assert detect_identity_sub("用户说他同事叫李杰") is None
        assert detect_identity_sub("用户的猫叫小花") is None

    @pytest.mark.parametrize("summary", [
        "用户叫我明天提醒他",
        "用户叫我起床",
        "用户叫他别走",
        "用户叫外卖",
        "用户叫车",
        "用户叫她过来",
    ])
    def test_jiao_as_a_verb_is_not_a_name(self, summary):
        """中文「叫」既是"名叫"也是"叫某人做某事".

        第一版正则把「用户叫我明天提醒他」判成姓名「我明天提醒他」并抬到 0.90 —
        既污染 L1 又占掉 singleton 槽位, 比漏判严重得多.
        """
        assert detect_identity_sub(summary) is None

    @pytest.mark.parametrize("summary", [
        "用户是内向的人",
        "用户是已婚的人",
        "用户是同性恋的",
        "用户是年轻人",
        "用户是好人",
        "用户是设计师",
        "用户是自由职业的",
    ])
    def test_trait_statements_are_not_birthplaces(self, summary):
        """「用户是X的/X人」的 X 位置什么都能塞 —— 性格、状态、年龄段、职业.

        第一版靠句式匹配, 把「用户是内向的人」判成出生地并占掉 singleton 槽位,
        后续真正的籍贯会被当成重复拒写. 形容词是开放集合, 停用词表列不完,
        所以改成正向要求地名标志.
        """
        assert detect_identity_sub(summary) is None

    @pytest.mark.parametrize("summary", [
        "用户是广东的",       # 省级简称
        "用户是北京人",       # 直辖市 + 人
        "用户是苏州的",       # 地级市
        "用户是浙江省的",     # 带行政后缀
        "用户来自四川",
        "用户的老家在成都",
    ])
    def test_real_places_still_recognised(self, summary):
        assert detect_identity_sub(summary) == "出生地"

    @pytest.mark.parametrize("summary", [
        "用户的籍贯是工程师",          # 老家/籍贯 也要校验载荷是不是地名
        "用户现在住在心里",            # 比喻不是现居地
        "用户是女生缘很好",            # 性别词出现在句中而非句尾
        "用户觉得对方是女生",          # 说的是第三人
        "用户说他同事是男生",          # 同上
        "用户的生日是明天，记得提醒",   # 相对时间 + 提醒请求, 钉成 L1 明天就成假记忆
        "用户的生日快到了",            # 没有具体日期
        "用户29岁的时候去了北京",      # 往事, 不是当前年龄
        "用户希望别人称呼自己为老板的助理",  # 「的」不出现在人名里
    ])
    def test_prefix_match_without_payload_check_is_rejected(self, summary):
        """整类缺陷: 只认前缀不看载荷.

        三轮 review 里这一类反复出现 (出生地→性格句、姓名→动词句、现居地→比喻句、
        生日→提醒句). 现在每条规则要么校验命名组, 要么用 $ 锚死整句。
        """
        assert detect_identity_sub(summary) is None

    @pytest.mark.parametrize("summary", [
        "用户是本市人",
        "用户住在大城市",
        "用户籍贯在外省",
    ])
    def test_generic_place_words_are_not_places(self, summary):
        """带行政后缀 ≠ 具体地点. 「本市」「大城市」「外省」写进 singleton 槽位,
        会把真正的籍贯挡在门外."""
        assert detect_identity_sub(summary) is None

    def test_unlisted_place_misses_rather_than_guesses(self):
        """没收录的地名回到修复前的状态 (漏判), 而不是瞎猜 —— 加一行数据即可."""
        from app.services.memory.recording.identity_repair import _looks_like_a_place

        assert _looks_like_a_place("广东")
        assert _looks_like_a_place("余杭区")
        assert not _looks_like_a_place("内向")
        assert not _looks_like_a_place("大城市")

    def test_longer_prefix_wins_in_the_alternation(self):
        """「叫做」必须整体匹配. 交替分支若把「叫」排前面, 剩下「做阿山」会被
        停用词判否, 好好一个名字就漏了."""
        assert detect_identity_sub("用户叫做阿山") == "姓名"

    def test_llm_specific_sub_is_never_overridden(self):
        """模型给了具体子类就说明它做了判断, 正则不该推翻."""
        main, sub, imp, reason = _repair("用户叫小明", sub="外貌特征", imp=0.70)
        assert (main, sub, imp) == ("身份", "外貌特征", 0.70)
        assert reason is None


class TestImportanceFloor:
    def test_floor_applies_even_when_sub_was_already_right(self):
        """生产可见: 子类分对了但 importance 仍给 0.80, 照样进不了 L1."""
        _, sub, imp, reason = _repair("用户今年29岁", sub="年龄", imp=0.80)
        assert sub == "年龄"
        assert imp == IDENTITY_IMPORTANCE_FLOOR
        assert "importance" in (reason or "")

    def test_higher_importance_is_not_lowered(self):
        _, _, imp, reason = _repair("用户叫小明", sub="姓名", imp=0.98)
        assert imp == 0.98
        assert reason is None

    def test_floor_clears_the_l1_threshold(self):
        """下限必须高于 pipeline 的 L1 门槛, 否则这层修了个寂寞."""
        assert IDENTITY_IMPORTANCE_FLOOR >= 0.85


class TestScope:
    def test_every_repairable_sub_is_an_l1_singleton(self):
        """兜底的价值在于恢复 singleton 保护; 修到非 singleton 子类既无必要
        也更容易误伤."""
        for sub in REPAIRABLE_SUBS:
            assert ("身份", sub) in L1_SINGLETON_SUBS, sub

    def test_empty_and_none_inputs_are_safe(self):
        assert detect_identity_sub("") is None
        assert detect_identity_sub("   ") is None
        main, sub, imp, reason = repair_identity_classification(
            summary="", main_category=None, sub_category=None, importance=0.5,
        )
        assert (main, sub, imp, reason) == ("", "", 0.5, None)

    def test_vague_main_category_is_also_corrected(self):
        """模型把身份事实丢进「生活/其他」时, main 也要拉回来."""
        main, sub, _, reason = _repair("用户叫小明", main="生活", sub="其他")
        assert (main, sub) == ("身份", "姓名")
        assert reason
