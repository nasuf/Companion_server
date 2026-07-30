"""多事实记忆的拆分.

起因: 检索时 8% 的 AI 记忆因超过单条 token 上限被整条跳过 —— 躺在库里但任何检索都
注入不了它们。追查发现两种形态, 只有一种该拆:

    多事实拼接 (含「；」)  → 按分号拆。它本来就该是几条独立记忆。
    单段完整叙事           → 按**句子边界**拆, 每段补回 "标题：" 前缀。

2026-07-30 修订: 单段叙事原先是"绝不拆"。实测 63 段存量长记忆后反转 —— 真正不能做
的是切在句子中间或切进引号里, 不是拆本身。带标题前缀按句号拆出来的两条各自读得通,
且都能被单独检索到; 不拆的代价是整条 100% 检索不到, 严格更差。

所以最要紧的测试不再是"一条都别拆", 而是**拆出来的每一条都得读得通**: 不切句中、
不切进引号、不丢字、续段有主语。
"""

from __future__ import annotations

import pytest

from app.services.memory.recording.splitting import (
    MAX_SEGMENTS,
    MIN_SEGMENT_CHARS,
    should_split,
    split_multi_fact,
)

# 生产真实样本 (截断)。用真数据而不是编的, 免得测出来的规则在实际内容上不成立。
_REAL_MULTI_FACT = (
    "我的工作是在线与电话咨询解答：通过公司内部通讯工具和400客服热线，实时解答用户在"
    "使用App过程中遇到的各类功能性问题；用户情绪安抚与引导：这是我工作中最具挑战也最"
    "有价值的部分，当用户因孤独焦虑向AI伴侣倾诉无果时我需要介入；BUG复现与报告：接到"
    "复杂技术问题时我需要在测试设备上一步步操作重现用户描述的问题"
)
_REAL_NARRATIVE = (
    "与“大橘”的首次相遇：在那个下着冬雨的黄昏，她在公司楼下的车轮旁发现了瑟瑟发抖、"
    "瘦得皮包骨的大橘。它的眼睛因为发炎而黏在一起，叫声像是从很远的地方传来，她蹲下来"
    "看了很久，最后脱下外套把它裹住带回了家"
)


class TestDoesNotSplitWhatItShouldnt:
    """反例优先: 误拆比不拆严重得多."""

    def test_narrative_is_not_semicolon_split(self):
        """叙事里没有「；」, 不该走多事实那条路 (句级拆分是另一条, 见下面的类)."""
        assert should_split(_REAL_NARRATIVE) is False

    def test_narrative_under_the_limit_is_untouched(self):
        """没超限就不动 —— 拆分只为救"存了也检索不到"的条目, 不是无差别切碎."""
        from app.services.memory.retrieval.context_selector import (
            MAX_MEMORY_TOKENS_PER_ITEM,
            estimate_tokens,
        )

        assert estimate_tokens(_REAL_NARRATIVE) <= MAX_MEMORY_TOKENS_PER_ITEM
        assert split_multi_fact(_REAL_NARRATIVE) == [_REAL_NARRATIVE]

    def test_commas_and_periods_are_not_separators(self):
        """中文叙事里逗号句号到处都是, 按它们切会把任何长文本切碎."""
        text = "我喜欢在周末去爬山，山顶的风很大。有时候会带上相机，拍一些云海的照片。"
        assert split_multi_fact(text) == [text]

    def test_short_text_untouched(self):
        assert split_multi_fact("我叫小伴") == ["我叫小伴"]

    def test_single_semicolon_with_one_real_segment_stays(self):
        """只有一段实质内容时拆了也是一条, 不如不动."""
        text = "我喜欢喝手冲咖啡；"
        assert split_multi_fact(text) == [text]

    def test_too_many_segments_left_alone(self):
        """切出十几条会让这个 agent 的某个话题在检索里过度膨胀, 挤掉其他类目."""
        text = "；".join(f"这是第{i}件事情它有足够长的描述文字来通过最小长度检查" for i in range(MAX_SEGMENTS + 3))
        assert split_multi_fact(text) == [text]

    def test_empty_and_none_safe(self):
        assert split_multi_fact("") == [""]
        assert should_split("") is False


class TestSplitsMultiFact:
    def test_real_production_sample_splits(self):
        parts = split_multi_fact(_REAL_MULTI_FACT)
        assert len(parts) == 3, f"应拆成 3 条, 实际 {len(parts)}: {parts}"

    def test_each_part_keeps_the_subject(self):
        """第二段起若丢了引导语, 单独被检索出来时读者不知道这在说什么 ——
        而记忆恰恰是被单独检索出来用的。"""
        parts = split_multi_fact(_REAL_MULTI_FACT)
        assert all(p.startswith("我的工作是") for p in parts), parts

    def test_parts_are_shorter_than_the_original(self):
        parts = split_multi_fact(_REAL_MULTI_FACT)
        assert all(len(p) < len(_REAL_MULTI_FACT) for p in parts)

    def test_no_content_is_lost(self):
        """拆分不能丢内容 —— 每一段的实质文字都要出现在某条结果里."""
        parts = split_multi_fact(_REAL_MULTI_FACT)
        joined = "".join(parts)
        for chunk in ("400客服热线", "情绪安抚", "BUG复现", "测试设备"):
            assert chunk in joined, f"拆分丢了「{chunk}」"

    def test_short_tail_merges_instead_of_becoming_its_own_row(self):
        """过短的段并进上一条而不是独立成条 —— 半句话既检索不到也没信息."""
        text = "我的爱好是长跑，每周固定跑三次风雨无阻；偶尔也游泳；我还喜欢在跑完之后去吃一碗热汤面"
        parts = split_multi_fact(text)
        assert all(len(p) >= MIN_SEGMENT_CHARS for p in parts)
        assert "游泳" in "".join(parts), "短段被丢了"

    def test_trailing_punctuation_normalised(self):
        parts = split_multi_fact(_REAL_MULTI_FACT)
        assert all(p.endswith("。") for p in parts)


class TestSecondPassSplit:
    """按「；」拆完仍超限的, 再按句号切一次.

    生产样本里有段落一次拆完还剩 172 token, 逼近 180 上限 —— 内容再长一点就"拆了
    也白拆"。
    """

    def test_long_segment_gets_split_further(self):
        from app.services.memory.retrieval.context_selector import (
            MAX_MEMORY_TOKENS_PER_ITEM,
            estimate_tokens,
        )

        long_seg = "我的工作是情绪安抚：" + "这是一段很长的说明文字用来撑满长度限制。" * 12
        text = f"{long_seg}；我的工作是另一件事情它也有足够的长度描述"
        for part in split_multi_fact(text):
            assert estimate_tokens(part) <= MAX_MEMORY_TOKENS_PER_ITEM, (
                f"二次拆分后仍超限: {estimate_tokens(part)} tok"
            )

    def test_single_unsplittable_sentence_kept_intact(self):
        """单句就超限时原样保留 —— 硬切会切在句子中间, 比超限更糟."""
        from app.services.memory.retrieval.context_selector import estimate_tokens

        one = "我的工作是" + "很长的没有句号的描述内容" * 40
        text = f"{one}；我的工作是另一件足够长的事情描述在这里"
        parts = split_multi_fact(text)
        big = [p for p in parts if estimate_tokens(p) > 180]
        assert all("。" not in p[:-1] for p in big), "把句子从中间切开了"


class TestNarrativeSentenceSplit:
    """超限的单段叙事按句子拆, 每段补回标题.

    这类记忆不拆就是死的 (检索时整条跳过), 所以拆; 但拆坏了比死掉更糟 —— 用户会读到
    半句话。下面每条都对着"读得通"这个标准。
    """

    def _oversized_narrative(self) -> str:
        return (
            "入职第一天的“手抖”：第一天正式接听电话，虽然培训时已滚瓜烂熟，但听到电话"
            "铃声响起的一瞬间，她还是紧张得手抖。接起电话后，是一位语气温和的阿姨咨询"
            "怎么修改AI伴侣的发型。我因为紧张，说话有点结巴，但阿姨反而安慰她："
            "“小姑娘，别急，慢慢来，我不赶时间。”在对方的鼓励下，她顺利完成了第一通"
            "服务。挂断电话后，她长长地舒了一口气。"
        )

    def test_never_splits_inside_a_quote(self):
        """生产样本踩过的坑.

        按句号裸切会在「我不赶时间。」处断开 —— 前一条引号不闭合, 后一条以一个孤零零
        的 ” 开头, 两条都读不通。
        """
        parts = split_multi_fact(self._oversized_narrative())
        for p in parts:
            assert not p.lstrip().startswith("”"), f"片段以孤立右引号开头: {p[:30]}"
            assert p.count("“") == p.count("”"), f"引号不配对: {p[:60]}"

    def test_每段都补回标题前缀(self):
        parts = split_multi_fact(self._oversized_narrative())
        assert len(parts) >= 2
        for p in parts:
            assert p.startswith("入职第一天的“手抖”："), f"续段丢了标题: {p[:30]}"

    def test_all_parts_fit_the_limit(self):
        from app.services.memory.retrieval.context_selector import (
            MAX_MEMORY_TOKENS_PER_ITEM,
            estimate_tokens,
        )

        for p in split_multi_fact(self._oversized_narrative()):
            assert estimate_tokens(p) <= MAX_MEMORY_TOKENS_PER_ITEM

    def test_no_characters_are_lost(self):
        """拆分只重排不删字 —— 补的前缀之外, 原文每个字都要还在."""
        import re

        src = self._oversized_narrative()
        joined = "".join(split_multi_fact(src))
        strip = lambda s: re.sub(r"[\s。；;，,、：:“”「」]", "", s)  # noqa: E731
        assert set(strip(src)) <= set(strip(joined))

    def test_unbalanced_quotes_degrade_to_no_split(self):
        """源数据引号不配对时不产出坏片段, 退化成不拆 (与修改前行为一致)."""
        text = "标题：" + "他说：“这段话的引号一直没有闭合。" * 14
        parts = split_multi_fact(text)
        assert parts == [text]

    def test_narrative_without_title_still_splits(self):
        """没有 "标题：" 的叙事各句自带主语, 不补前缀也读得通."""
        from app.services.memory.retrieval.context_selector import (
            MAX_MEMORY_TOKENS_PER_ITEM,
            estimate_tokens,
        )

        text = (
            "大橘是林昕刚工作时在公司园区附近发现的流浪猫。当时它还是一只瘦弱的小奶猫，"
            "躲在车轮下避雨，叫声微弱。林昕于心不忍，便将它带回了公寓。起初只是打算暂时"
            "喂养找个领养人，但养着养着就有了感情。大橘非常聪明乖巧从不拆家，会在她下班"
            "回家时准时蹲在门口迎接，用头蹭她的脚踝。对于独自居住的林昕而言，大橘不仅是"
            "宠物，更是这个家不可缺少的家庭成员和精神寄托。"
        )
        assert estimate_tokens(text) > MAX_MEMORY_TOKENS_PER_ITEM
        parts = split_multi_fact(text)
        assert len(parts) >= 2
        for p in parts:
            assert estimate_tokens(p) <= MAX_MEMORY_TOKENS_PER_ITEM
            assert p.endswith("。"), f"切在了句子中间: …{p[-25:]}"


class TestOversizeRelief:
    def test_splitting_brings_parts_under_the_injection_limit(self):
        """拆分的目的就是让每条能进注入集 —— 拆完还超限就白拆了."""
        from app.services.memory.retrieval.context_selector import (
            MAX_MEMORY_TOKENS_PER_ITEM,
            estimate_tokens,
        )

        assert estimate_tokens(_REAL_MULTI_FACT) > MAX_MEMORY_TOKENS_PER_ITEM, (
            "样本本身没超限, 这个测试就失去意义了"
        )
        for part in split_multi_fact(_REAL_MULTI_FACT):
            assert estimate_tokens(part) <= MAX_MEMORY_TOKENS_PER_ITEM, (
                f"拆完仍超限 ({estimate_tokens(part)} tok): {part[:40]}"
            )


@pytest.mark.parametrize("text", [_REAL_MULTI_FACT, _REAL_NARRATIVE])
def test_split_is_idempotent(text: str):
    """对已拆过的结果再拆一次不该继续变化 —— 否则修复脚本重跑会越拆越碎."""
    once = split_multi_fact(text)
    twice = [p for part in once for p in split_multi_fact(part)]
    assert once == twice
