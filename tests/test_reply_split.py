"""Reply split + whitespace cleanup tests.

Regression: LLM 偶尔用 \\n\\n 分段而不是约定的 ||, 老逻辑只 split("||")
不识别空行 → 整段进同一回复 → 前端 .msg-bubble p { white-space: pre-wrap }
把 \\n\\n 渲染成空行, 视觉上像一条带空行的消息.
"""

from __future__ import annotations

from app.services.chat.orchestrator import (
    _clean_reply_part,
    split_and_validate_replies,
    truncate_at_sentence,
)


class TestCleanReplyPart:
    def test_collapses_intra_reply_newline_to_space(self):
        assert _clean_reply_part("你好\n吗") == "你好 吗"

    def test_strips_leading_trailing_whitespace(self):
        assert _clean_reply_part("  hi  ") == "hi"

    def test_collapses_multiple_newlines(self):
        assert _clean_reply_part("a\n\n\nb") == "a b"

    def test_preserves_content(self):
        assert _clean_reply_part("正常一句话。") == "正常一句话。"

    def test_strips_leading_full_timestamp_prefix(self):
        # LLM 模仿注入的历史时间前缀 [MM-DD HH:MM], 不应泄漏到回复
        assert _clean_reply_part("[07-04 16:56] 啊？你昨天不就夸了句月亮。") == (
            "啊？你昨天不就夸了句月亮。"
        )

    def test_strips_leading_bare_hhmm_prefix(self):
        assert _clean_reply_part("[16:56] 在忙呢") == "在忙呢"

    def test_strips_repeated_timestamp_prefix(self):
        assert _clean_reply_part("[07-04 16:56] [07-04 16:56] 内容") == "内容"

    def test_keeps_non_timestamp_bracket(self):
        # 只剥时间戳格式, 不误删普通方括号内容
        assert _clean_reply_part("[重要] 记得喝水") == "[重要] 记得喝水"


class TestSplitAndValidateReplies:
    def test_pipe_separator(self):
        result = split_and_validate_replies("第一句||第二句")
        assert result == ["第一句", "第二句"]

    def test_strips_timestamp_prefix_on_each_split_part(self):
        # 复现生产 bug: LLM 两条回复都带 [MM-DD HH:MM] 前缀
        result = split_and_validate_replies(
            "[07-04 16:56] 啊？你昨天不就夸了句月亮好圆嘛。"
            "||[07-04 16:56] 难道你昨天还干了啥大事没汇报？"
        )
        assert result == [
            "啊？你昨天不就夸了句月亮好圆嘛。",
            "难道你昨天还干了啥大事没汇报？",
        ]

    def test_double_newline_treated_as_pipe(self):
        """LLM 用空行分段 — 必须切, 不能让 \\n\\n 留在单条内被 pre-wrap 渲染."""
        result = split_and_validate_replies(
            "抱抱。这种闷闷的感觉确实挺难受的。\n\n我也刚因为算账焦虑了一会儿。"
        )
        assert len(result) == 2
        assert "\n" not in result[0]
        assert "\n" not in result[1]
        assert result[0].startswith("抱抱")
        assert "算账" in result[1]

    def test_triple_newline_also_splits(self):
        result = split_and_validate_replies("第一段\n\n\n第二段")
        assert result == ["第一段", "第二段"]

    def test_intra_part_single_newline_collapsed(self):
        """单 \\n 不切, 但要折成空格防止单条内断行."""
        result = split_and_validate_replies("你好\n吗")
        assert result == ["你好 吗"]

    def test_mixed_pipe_and_newline(self):
        result = split_and_validate_replies("一\n\n二||三")
        assert result == ["一", "二", "三"]

    def test_strips_dangling_comma_from_split_boundary(self):
        result = split_and_validate_replies("还有《小森林》，||那种慢慢生活的感觉很好")
        assert result == ["还有《小森林》", "那种慢慢生活的感觉很好"]

    def test_preserves_terminal_punctuation_at_split_boundary(self):
        result = split_and_validate_replies("我超喜欢！||你呢？||可以聊聊。")
        assert result == ["我超喜欢！", "你呢？", "可以聊聊。"]

    def test_strips_soft_punctuation_from_overlong_local_split(self):
        raw = "好呀，我也喜欢那种很慢的电影，像《小森林》，那种按季节生活的感觉很舒服。"
        result = split_and_validate_replies(raw, max_count=3, max_per_reply=24, max_total=90)
        assert len(result) > 1
        assert not result[0].endswith(("，", ",", "、", "；", ";", "：", ":"))

    def test_single_reply_no_split(self):
        result = split_and_validate_replies("整个就是一句话。")
        assert result == ["整个就是一句话。"]

    def test_empty_input_returns_placeholder(self):
        assert split_and_validate_replies("") == ["..."]
        assert split_and_validate_replies("   \n\n  ") == ["..."]

    def test_max_count_enforced(self):
        result = split_and_validate_replies("a||b||c||d||e", max_count=3)
        assert len(result) == 3

    def test_max_per_reply_truncates(self):
        long_text = "我" * 100  # 100 chars
        result = split_and_validate_replies(long_text, max_per_reply=20)
        assert len(result) == 1
        assert len(result[0]) <= 20

    def test_overlong_single_reply_splits_at_punctuation(self):
        """主 LLM 忘记输出 || 时，不能把气泡截断在半句话中间。"""
        raw = (
            "好呀，我的工作说起来有点野生——我专门给机械小动物做义肢和升级，"
            "比如给瘸腿兔子装音乐关节，或者给断翅膀的鸟做碳纤维翅膀。"
        )
        result = split_and_validate_replies(raw, max_count=3, max_per_reply=36, max_total=120)

        assert len(result) > 1
        assert all(len(part) <= 36 for part in result)
        assert not result[0].endswith("或者给断翅膀的")
        assert "".join(result).startswith("好呀，我的工作说起来")


class TestTruncateAtSentence:
    def test_under_max_no_change(self):
        assert truncate_at_sentence("短句。", 100) == "短句。"

    def test_truncate_at_period(self):
        text = "第一句话。第二句话。第三句话。"
        result = truncate_at_sentence(text, 7)
        assert result.endswith("。")
        assert len(result) <= 7
