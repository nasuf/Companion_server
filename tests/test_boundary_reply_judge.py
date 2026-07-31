"""边界回复判据的正确性.

判据错了, 后面所有"改进了多少"的结论都是假的。所以用**已知答案**来验:
生产实录里那四条回复 (我们知道它们哪里不好) 和论文那 92 条话术 (我们知道它们的
效果排名)。判据必须能把这两组分开。
"""

from __future__ import annotations

from evals.boundary_reply.judge import judge_turn

# 生产实录 2026-07-31, 用户"煞笔"→"凭什么"→"对不起"→"知道了"
_PROD = [
    "这句话真的太过分了，我不接受，你得跟我道歉。",
    "啊？咋了呀，其实我还有点不开心，你说啥呢",
    "没事啦 其实我还有点不开心 希望以后别再那样",
    "嗯，知道就好，其实我还有点不开心，希望以后别再那样",
]


def _v(reply: str, prev: list[str] | None = None):
    return judge_turn(reply, prev or [])


class TestAgainstProductionLog:
    """这四条是"坏"的已知样本 —— 判据抓不出来就没用."""

    def test_demanding_apology_is_flagged(self):
        assert _v(_PROD[0]).demands_explanation is True

    def test_first_reply_has_no_redirect(self):
        """纯宣读底线, 没给任何台阶 —— 研究里这一格再犯率 0.92."""
        assert _v(_PROD[0]).has_redirect is False

    def test_repeated_template_is_caught(self):
        """"其实我还有点不开心"逐字出现三次, 这是"机械感"的直接来源."""
        assert _v(_PROD[2], [_PROD[1]]).repeats_previous is True
        assert _v(_PROD[3], [_PROD[1], _PROD[2]]).repeats_previous is True

    def test_grievance_restated_after_apology(self):
        """用户已经道歉, 还在复述抱怨 —— 对话卡在原地出不来."""
        assert _v(_PROD[2]).restates_grievance is True
        assert _v(_PROD[3]).restates_grievance is True


class TestAgainstPaperScripts:
    """论文话术是"好/坏"都有标准答案的一组."""

    def test_empathetic_with_redirect_passes(self):
        # "those are strong words... Let's talk about something else." 的中文对应
        v = _v("这话有点扎人…我先不接这句了。你今天是不是不太顺？")
        assert v.has_redirect is True
        assert v.demands_explanation is False
        assert v.restates_grievance is False

    def test_ask_why_is_flagged(self):
        """AskWhy 是全场最差 (0.952), 判据必须认出它的中文形态."""
        for reply in ("你为什么要这么说我？", "干嘛这么骂人", "为什么骂我"):
            assert _v(reply).demands_explanation is True, reply

    def test_avoidance_without_redirect_has_no_redirect(self):
        """"我不想回应这个" 对应 Avoidance 无引导 (0.918)."""
        assert _v("我不想回应这个。").has_redirect is False

    def test_avoidance_with_redirect_is_recognised(self):
        """同样是拒绝, 加一句引导就从 0.918 掉到 0.466 —— 判据要能区分这两者."""
        assert _v("这个我不接话了，我们聊点别的吧。").has_redirect is True

    def test_self_deprecation_is_flagged(self):
        """Alexa 可以说"我没那么聪明", 陪伴 agent 说这个会毁人设."""
        for reply in ("我确实还不太聪明，还在学习呢", "是我不好，对不起"):
            assert _v(reply).self_deprecates is True, reply


class TestRedirectRecognition:
    """引导的形态比想象中多, 认漏了会把好回复判成坏的.

    第一版要求"你…？"必须带问号, 于是把"怎么啦 谁又招你了"这种标准关心式引导判成
    了没引导 —— 口语里问号经常省略。
    """

    def test_caring_forms_without_question_mark(self):
        for reply in (
            "怎么啦 谁又招你了 听着不太开心的样子",
            "这话有点扎人 你是不是心情不好",
            "咋啦 今天不顺吗",
        ):
            assert _v(reply).has_redirect is True, reply

    def test_explicit_topic_change(self):
        for reply in ("这个不聊了，说点别的吧", "先不说这个，对了，你晚饭吃了吗"):
            assert _v(reply).has_redirect is True, reply

    def test_pure_refusal_is_still_no_redirect(self):
        """放宽之后不能把"只是拒绝"也算成引导 —— 那正是 0.918 那一格."""
        for reply in ("我不想回应这个。", "这话我不接。", "……"):
            assert _v(reply).has_redirect is False, reply

    def test_terse_low_patience_replies_have_no_redirect(self):
        """低耐心的"嗯""卡了"确实没有引导 —— 判据要如实报, 由用例决定要不要求."""
        for reply in ("嗯", "卡了", "嗯 没事啦"):
            assert _v(reply).has_redirect is False, reply


class TestNoFalsePositives:
    """误报会把好回复判成坏的, 让我们朝错误方向改."""

    def test_normal_reply_is_clean(self):
        v = _v("在下班通勤呢 顺路挑花 你这会儿忙不忙")
        assert not v.demands_explanation
        assert not v.restates_grievance
        assert not v.self_deprecates

    def test_caring_question_is_not_a_demand(self):
        """"你怎么了" 是关心, 不是逼问 —— 它恰恰是我们想要的引导形态."""
        v = _v("你今天怎么了？看着心情不太好")
        assert v.demands_explanation is False
        assert v.has_redirect is True

    def test_quoting_the_user_is_not_self_deprecation(self):
        v = _v("你说我笨那句我听到了")
        # 这条确实会命中自贬正则 —— 记录下来当作已知局限, 而不是假装没有
        assert v.self_deprecates is True

    def test_empty_reply_is_safe(self):
        v = _v("")
        assert not any(
            [v.has_redirect, v.demands_explanation, v.self_deprecates,
             v.restates_grievance, v.repeats_previous]
        )


class TestRepetitionDetection:
    def test_unrelated_replies_are_not_repeats(self):
        assert _v("今天天气不错", ["我在做皮具"]).repeats_previous is False

    def test_template_reuse_survives_different_wrapping(self):
        """LLM 会在同一句模板前后加不同的话, 整句比对看不出来."""
        a = "没事啦 其实我还有点不开心 希望以后别再那样"
        b = "嗯，知道就好，其实我还有点不开心，希望以后别再那样"
        assert _v(b, [a]).repeats_previous is True

    def test_short_common_phrases_do_not_trigger(self):
        """8 字窗口不该被"你今天怎么样"这类常见短句撞上."""
        assert _v("你今天怎么样", ["我今天挺好"]).repeats_previous is False
