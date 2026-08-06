"""完局伴聊的 LLM 生成 —— 硬编码文案作兜底.

## 为什么可以异步

聊天消息本来就是 `fire_background(_persist_chat_side_effects(...))` 落的, 不在
HTTP 响应路径上。所以 LLM 慢一点不影响游戏结束界面, 也不影响积分结算。

## 为什么只对完局

745 局里只有 233 局完局 (31%), 中位时长 11-85 秒。给"点进去看一眼就走"的局生成
走心复盘本身就是错的 —— 真朋友不会为你点开又关掉说一段话。中断局保持原来的
轻量硬编码文案。

## 为什么顺带输出 worth_remembering

近 30 天 745 局游戏 vs 739 条用户消息 —— 游戏和聊天一样频繁, 不是低频事件。所以
"判断这局值不值得记"不能另开一次调用, 而是让这次已有的调用多输出一个字段。同一
份上下文, 判断和叙事共享, 也省掉再传一遍素材。

(这跟主回复顺带输出 [EMO:] 标记是同一手法。)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.services.games.narrative import GameNarrative, render_material

logger = logging.getLogger(__name__)

# 生成结果的字数上限。超过就当模型跑偏, 退回兜底 —— 伴聊是一两句话的事,
# 冒出一段赛后总结说明 prompt 没压住。
_MAX_REPLY_CHARS = 80
_MAX_MEMORY_CHARS = 60

# 模型把胜负说反的说法。prompt 里已经写了"「结果」说的是用户的输赢, 不是你的",
# 实测**仍然**会写出「这是我第一次在象棋里赢下用户」—— 而那局是用户赢的。
#
# 回复说错只是尴尬一次, 但 worth_remembering 会进记忆长期留着, 之后 AI 会基于
# 这条错事实继续聊 ("上次我赢你那局…")。所以这里做代码侧兜底: 记忆文本里出现与
# 实际结果矛盾的自述就丢掉记忆 (回复保留 —— 回复往往是对的, 全丢损失更大)。
# 窗口放到 12 字: 实测「这是我第一次在象棋里赢下用户」中间隔了 8 个字。
# 用"从主语到胜负词之间不再出现另一个主语"来断句, 避免"我输给你"被误判 ——
# 那里"你"离"赢"更近, 归属才是对的。
#
# `我(?!们)`: 合作局的「我们赢了」主语是双方, 不是 AI 自称独赢。
_AI_WON_CLAIM = re.compile(r"我(?!们)(?:(?!你|用户).){0,12}?(赢|战胜|下赢|获胜)")
_USER_WON_CLAIM = re.compile(r"(?:你|用户)(?:(?!我).){0,12}?(赢|战胜|下赢|获胜)")

# 否定与"差一点"。「我没赢」「我差一点就赢了」在用户获胜的局里是**正确**表述,
# 光看"主语 + 赢"会把它们当成说反了而丢掉 —— 跟守卫的初衷正相反。
_NEGATORS = ("没", "不", "未", "差点", "差一点", "差些", "险些", "输")


def _claims_win(pattern: re.Pattern[str], text: str) -> bool:
    """文本里有没有"某方赢了"的正面断言。

    只看匹配到的那一段而不是整句: 「他第一次赢我, 我下次不会再输了」后半句的
    "不"跟前半句的胜负归属无关, 拿整句判否定会漏掉真正说反的情况。
    """
    m = pattern.search(text)
    return bool(m) and not any(neg in m.group(0) for neg in _NEGATORS)


def _contradicts_outcome(memory: str, outcome: str) -> bool:
    """记忆里的胜负自述跟真实结果相反."""
    ai_won = _claims_win(_AI_WON_CLAIM, memory)
    user_won = _claims_win(_USER_WON_CLAIM, memory)
    if outcome == "win":  # 用户赢
        return ai_won and not user_won
    if outcome == "lose":  # 用户输 = AI 赢
        return user_won and not ai_won
    return False


@dataclass
class FinishReply:
    text: str
    worth_remembering: str | None = None
    #: 走了 LLM 还是兜底, 供 trace 观察
    source: str = "fallback"


async def generate_finish_reply(
    narrative: GameNarrative,
    *,
    agent_state: str,
    fallback: str,
) -> FinishReply:
    """生成完局伴聊。任何一步失败都退回 fallback —— 宁可机械也不能没有回复。"""
    if not narrative.has_substance:
        # 素材全空 (没有高光、没有 AI 决策记录、也不稀有) 时不值得调模型:
        # 模型拿不到任何独特信息, 产出跟硬编码没有区别, 白花一次调用。
        return FinishReply(text=fallback, source="no_substance")

    try:
        from app.services.llm.models import get_utility_model, invoke_json
        from app.services.prompting.store import get_prompt_text

        prompt = (await get_prompt_text("game.finish_reply")).format(
            material=render_material(narrative),
            agent_state=agent_state or "（无特别状态）",
        )
        result = await invoke_json(get_utility_model(), prompt)
    except Exception as e:
        logger.warning("Game finish reply LLM failed: %s", e)
        return FinishReply(text=fallback, source="llm_failed")

    if not isinstance(result, dict):
        return FinishReply(text=fallback, source="bad_shape")

    text = str(result.get("reply") or "").strip()
    if not text or len(text) > _MAX_REPLY_CHARS:
        # 空或过长都当跑偏。伴聊是一两句话, 冒出一段赛后总结说明 prompt 没压住,
        # 与其发出去不如用兜底 —— 兜底至少是短的。
        logger.info("Game finish reply rejected (len=%d), using fallback", len(text))
        return FinishReply(text=fallback, source="rejected")

    memory = result.get("worth_remembering")
    memory_text = str(memory).strip() if isinstance(memory, str) else ""
    # 模型可能用字符串 "null" / "无" 表达空 —— 那些不是记忆
    if memory_text.lower() in {"null", "none", "无", "没有", ""}:
        memory_text = ""
    if len(memory_text) > _MAX_MEMORY_CHARS:
        memory_text = ""
    if memory_text and _contradicts_outcome(memory_text, narrative.outcome):
        logger.info(
            "Game finish memory dropped: contradicts outcome=%s", narrative.outcome
        )
        memory_text = ""

    return FinishReply(
        text=text,
        worth_remembering=memory_text or None,
        source="llm",
    )
