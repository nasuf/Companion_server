"""写入侧的长度收口.

`store_memory` 是全部 11 个写入方 (聊天抽取 / 每日总结 / L3 整合 / 矛盾纠正 /
提醒 / 知识导入 …) 的唯一入口, 之前它对长度不做任何处理。超过检索单条上限的记忆
会被 context_selector 整条跳过 —— 存进去了, 但任何对话都不会用到它, 且没有任何
外部症状。

在这一处拆一次就覆盖全部路径。2026-07 生产实测: 每日总结最长已到 171 token, 距
180 的上限只剩 9 —— 靠 prompt 里写"简洁"挡不住。
"""

from __future__ import annotations

from app.services.memory.retrieval.context_selector import (
    MAX_MEMORY_TOKENS_PER_ITEM,
    estimate_tokens,
)
from app.services.memory.storage.persistence import _split_for_storage
from app.services.memory.taxonomy import L1_SINGLETON_SUBS, resolve_taxonomy


def _taxonomy(main: str, sub: str, source: str = "ai", level: int = 2):
    return resolve_taxonomy(
        main_category=main, sub_category=sub,
        legacy_type=None, source=source, level=level,
    )


_LONG_NARRATIVE = (
    "入职第一天的“手抖”：第一天正式接听电话，虽然培训时已滚瓜烂熟，但听到电话铃声"
    "响起的一瞬间，她还是紧张得手抖。接起电话后，是一位语气温和的阿姨咨询怎么修改"
    "AI伴侣的发型。我因为紧张，说话有点结巴，但阿姨反而安慰她：“小姑娘，别急，慢慢"
    "来，我不赶时间。”在对方的鼓励下，她顺利完成了第一通服务。挂断电话后，她长长地"
    "舒了一口气。"
)


class TestSplitDecision:
    def test_short_content_is_untouched(self):
        t = _taxonomy("生活", "工作")
        assert _split_for_storage("今天开了个会。", t) == ["今天开了个会。"]

    def test_empty_content_is_safe(self):
        t = _taxonomy("生活", "工作")
        assert _split_for_storage("", t) == [""]

    def test_oversized_content_is_split(self):
        t = _taxonomy("生活", "工作")
        assert estimate_tokens(_LONG_NARRATIVE) > MAX_MEMORY_TOKENS_PER_ITEM
        pieces = _split_for_storage(_LONG_NARRATIVE, t)
        assert len(pieces) >= 2

    def test_every_piece_fits_the_limit(self):
        t = _taxonomy("生活", "工作")
        for p in _split_for_storage(_LONG_NARRATIVE, t):
            assert estimate_tokens(p) <= MAX_MEMORY_TOKENS_PER_ITEM

    def test_singleton_categories_are_never_split(self):
        """单例子类每个 agent 只允许一行.

        拆成两条会被 singleton 闸门拦下第二条 —— 结果是内容丢一半, 比超限更糟。
        这些子类 (姓名/年龄/生日) 天然很短, 实际走不到这, 但不能靠"应该不会发生"。
        """
        assert L1_SINGLETON_SUBS, "单例清单为空, 这个测试就没意义了"
        main, sub = next(iter(L1_SINGLETON_SUBS))
        t = _taxonomy(main, sub, level=1)
        assert _split_for_storage(_LONG_NARRATIVE, t) == [_LONG_NARRATIVE]

    def test_unsplittable_content_passes_through(self):
        """拆不动就原样存 —— 硬切会切在句子中间, 那比超限更糟."""
        t = _taxonomy("生活", "工作")
        one_sentence = "那个下着冬雨的黄昏她在公司楼下等了我很久很久始终没有离开" * 8
        assert _split_for_storage(one_sentence, t) == [one_sentence]


class TestWiring:
    def test_store_memory_accepts_the_reentry_flag(self):
        """递归重入靠 _split_done 终止, 少了它会无限递归."""
        import inspect

        from app.services.memory.storage.persistence import store_memory

        assert "_split_done" in inspect.signature(store_memory).parameters

    def test_split_happens_after_taxonomy_resolution(self):
        """必须先归一类目再决定拆不拆 —— 单例判定依赖归一后的 (main, sub).

        调用方传进来的 sub_category 可能是别名 ("纪念日" → "重要日期"), 拿原始值
        判断会漏掉单例。
        """
        import inspect

        from app.services.memory.storage.persistence import store_memory

        src = inspect.getsource(store_memory)
        assert src.index("resolve_taxonomy(") < src.index("_split_for_storage(")
