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
from dataclasses import dataclass

from app.services.games.narrative import GameNarrative, render_material

logger = logging.getLogger(__name__)

# 生成结果的字数上限。超过就当模型跑偏, 退回兜底 —— 伴聊是一两句话的事,
# 冒出一段赛后总结说明 prompt 没压住。
_MAX_REPLY_CHARS = 80
_MAX_MEMORY_CHARS = 60


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

    return FinishReply(
        text=text,
        worth_remembering=memory_text or None,
        source="llm",
    )
