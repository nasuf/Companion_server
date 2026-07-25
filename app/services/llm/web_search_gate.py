"""Decide whether a user message needs live web information.

Why an explicit gate instead of letting the model choose:
measured against the real production system prompt (~4.7k chars of persona +
format constraints), `doubao-seed-character` with `tool_choice: "auto"`
triggered a search in 0 of 16 trials — it answers "我帮你查下" and never calls
the tool. The same prompt with `tool_choice: "required"` searches reliably and
answers correctly. So the decision has to happen on our side, and the search
call must force the tool.

Two stages, cheapest first:
1. `looks_like_realtime_question` — keyword prefilter, no LLM. Ordinary chat
   stops here, so the classifier cost applies only to candidate messages.
2. `needs_web_search` — small-model confirm on candidates, killing the
   prefilter's false positives ("最近怎么样" / "你新剪的头发好看" ...).

Both stages fail closed (no search) so this can never break a normal reply.
"""

from __future__ import annotations

import logging

from app.services.llm.models import get_utility_model, invoke_text
from app.services.prompting.utils import render_prompt

logger = logging.getLogger(__name__)

# Topic markers for information that changes over time and cannot come from
# memory. Recall-oriented on purpose: stage 2 removes the false positives.
_REALTIME_HINTS: tuple[str, ...] = (
    # 天气 / 环境
    "天气", "气温", "下雨", "下雪", "台风", "空气质量", "限号",
    # 新闻 / 热点
    "新闻", "热搜", "热点", "最新", "刚刚发生", "出什么事",
    # 影视 / 娱乐 / 作品
    "上映", "新片", "新电影", "新剧", "票房", "评分", "豆瓣", "演唱会", "新歌", "新专辑",
    # 行情 / 价格
    "股价", "股票", "金价", "汇率", "油价", "币价", "多少钱", "价格", "涨了", "跌了",
    # 赛事
    "比分", "比赛结果", "夺冠", "赛程",
    # 榜单 / 推荐类时效问题
    "排行", "榜单", "推荐几个", "有什么好看的", "有什么好玩的",
)

# Message shorter than this is chit-chat ("嗯"/"在吗") — never worth a search.
_MIN_LENGTH = 4


def looks_like_realtime_question(message: str) -> bool:
    """Cheap prefilter: could this message need information we cannot know?"""
    text = (message or "").strip()
    if len(text) < _MIN_LENGTH:
        return False
    return any(hint in text for hint in _REALTIME_HINTS)


async def needs_web_search(message: str, context: str = "") -> bool:
    """Small-model confirm for messages the prefilter flagged.

    Returns False on any failure — a missed search degrades to today's
    behaviour, while a wrong search costs a plugin call and ~2s of latency.
    """
    try:
        raw = await render_prompt(
            "chat.web_search_decision",
            {"message": message, "context": context or "(无)"},
            lambda p: invoke_text(get_utility_model(), p),
        )
    except Exception as e:  # noqa: BLE001 — gate must never break the reply
        logger.warning(f"[WEB-SEARCH-GATE] classify failed: {e}")
        return False
    answer = (raw or "").strip()
    # "不需要联网" contains "需要联网", so the negative must be checked first.
    if "不需要" in answer or "无需" in answer:
        return False
    return "需要联网" in answer or answer.startswith("需要")
