"""记忆消息过滤器。

多信号准入，决定是否对消息进行记忆提取。
核心目标是过滤纯寒暄/回声词，同时避免错过简短但高价值的自我披露。
"""

from __future__ import annotations

import re

from app.services.rules.memory_keywords import (
    CORE_PROFILE_PATTERNS,
    FIRST_PERSON_TERMS,
    FIRST_PERSON_TERMS_EN,
    MEMORY_EMOTION_WORDS,
    MEMORY_EMOTION_WORDS_EN,
    MEMORY_FACT_WORDS,
    MEMORY_FACT_WORDS_EN,
    MEMORY_FILLER_WORDS,
    MEMORY_TIME_WORDS,
    MEMORY_TIME_WORDS_EN,
    SELF_DISCLOSURE_PATTERNS,
    SELF_DISCLOSURE_PATTERNS_EN,
)


def _word_set(message: str) -> set[str]:
    """Extract lowercase latin words for lightweight English heuristics."""
    return {w.lower() for w in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", message)}


# 有实义内容的最小判据: 至少一个汉字/字母/数字. 纯表情或纯标点没有可抽取的东西,
# 挡在这里能省一次小模型调用.
_HAS_CONTENT = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")


def should_extract_memory(message: str) -> bool:
    """Spec §2.1.1: 只硬拒纯语气词和无实义内容, 其余一律交给小模型预筛.

    这里曾经是一个加权打分器 (长度/第一人称/情感词/时间词/事实词/自我暴露句式,
    满 2 分才放行). 拿 619 条真实用户消息实测, 它拒掉 56%, 而 spec 字面只要求拒
    纯语气词 (对应 1%). 多拒的 341 条从未被记录为 spec 偏离.

    代价是实打实的: 抽查 120 条被它拒掉的消息, 带上前一句 AI 的话交给两个模型
    评审, **双方都认为该记的占 21%** (单个较宽松的评审给到 52%). 按下界折算,
    348 条被拒消息里约 73 条含有值得记的内容 —— 而系统当时全部用户记忆才 77 条。
    过滤器丢掉的和它留下的一样多。

    根因不是词表不全, 是这一级**只看单条消息的字面**, 而对话里的意义常常在上文:

        AI: 你今天还好吗
        用户: 不好            ← 两个字, 打分器给不出 2 分, 但这是明确的情绪记录
        AI: 怎么一直加班
        用户: 经常这样        ← 同理

    加关键词补不上这个洞 —— 缺的是上下文, 不是词汇。而这一级的假阴性没有任何
    后续环节能补: 被拒的消息不进预筛, 也不进抽取。

    所以退回 spec 的职责划分: 规则只做"一眼就知道不用记"的硬拒, 判断交给小模型。
    小模型在这件事上有外部标注支撑 —— DuLeMon 人工标为 persona 事实的话语, 预筛
    召回 92.6%, 正是这一级最需要的属性。预筛跑在 _bg_memory_pipeline 后台, 不占
    回复延迟; 多出来的调用按 qwen3.5-flash 计价可以忽略。
    """
    if not message or not message.strip():
        return False

    msg = message.strip()
    if msg.lower() in MEMORY_FILLER_WORDS:
        return False
    if not _HAS_CONTENT.search(msg):
        return False
    return True
