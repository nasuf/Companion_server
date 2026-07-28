"""记忆管线第一级只做硬拒, 不做猜测.

这一级曾经是个加权打分器 (长度/第一人称/情感词/时间词/自我暴露句式, 满 2 分才
放行). 拿 619 条真实用户消息实测它拒掉 56%, 而 spec §2.1.1 字面只要求拒纯语气词.
抽查其中 120 条交给两个模型评审 (带上前一句 AI 的话), **双方都判该记的占 21%** ——
折算下来约 73 条被误丢, 而系统当时全部用户记忆才 77 条.

根因是这一级只看单条消息的字面, 而对话里的意义常常在上文:

    AI: 你今天还好吗
    用户: 不好            ← 打分器给不出 2 分, 但这是明确的情绪记录

补关键词治不了 —— 缺的是上下文不是词汇。而且这一级的假阴性无法挽回: 被拒的消息
不进预筛, 也不进抽取。

所以规则只保留"任何上下文下都不值得记"的硬拒, 判断交给小模型 (DuLeMon 外部标注
上召回 92.6%)。这个文件守住这条分工不被重新滑回打分器。
"""

from __future__ import annotations

import pytest

from app.services.memory.recording.filter import should_extract_memory

# 真实生产消息, 全部曾被打分器拒掉, 且两个评审都认为该记
CONTEXT_DEPENDENT = [
    "被领导骂了",
    "我好烦",
    "经常这样",
    "消不了",
    "完全不好奇😂",
    "还不知道",
]

ALWAYS_JUNK = [
    "嗯", "好的", "哈哈", "你好", "早上好", "你呢", "是吗", "在吗",
    "🥺", "😴😴😴", "", "   ",
]


@pytest.mark.parametrize("message", CONTEXT_DEPENDENT)
def test_short_but_meaningful_replies_reach_the_model(message):
    """短不等于没内容。这些的意义在上一句里, 规则看不到, 所以不能替模型下结论."""
    assert should_extract_memory(message), (
        f"「{message}」被第一级拒了 —— 它不会进预筛也不会进抽取，这条信息永久丢失。"
        "如果是为了省小模型调用而加的规则，代价不成比例：这一级的假阴性无法挽回。"
    )


@pytest.mark.parametrize("message", ALWAYS_JUNK)
def test_contentless_messages_are_still_rejected_cheaply(message):
    """放宽不等于不拦。这些在任何上下文下都不值得记, 送小模型是白花钱."""
    assert not should_extract_memory(message)


def test_filter_does_not_score_signals_anymore():
    """守住分工: 一旦这里又变成加权打分, 上面那批消息会重新被吞掉。

    检查的是"有没有实义内容", 不是"像不像值得记的话" —— 后者要上下文, 是模型的活。
    """
    import inspect

    from app.services.memory.recording import filter as filter_module

    source = inspect.getsource(filter_module.should_extract_memory)
    for banned in ("total_weight", ">= 2", "FIRST_PERSON", "EMOTION_WORDS"):
        assert banned not in source, (
            f"should_extract_memory 里出现了 {banned!r} —— 这一级又开始靠字面"
            "信号猜该不该记了，实测这样会误丢五分之一以上该记的内容。"
        )


def test_a_long_disclosure_still_passes():
    """回归保护: 放宽逻辑不能把明显该记的搞丢."""
    assert should_extract_memory("我是一个程序员，在北京工作")
    assert should_extract_memory("我昨天去看了电影")
