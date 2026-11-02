"""Bounded evidence check for date-specific replies with mismatched memories.

Only the narrow missing-event case incurs a utility-model call. No DB writes,
new memory claims, date backfills, or changes to crisis/confirmation replies.
"""
from __future__ import annotations

import asyncio
import logging

from app.services.llm.models import get_utility_model, invoke_json
from app.services.prompting.store import get_prompt_text, PromptDisabledError
from app.services.prompting.utils import render_template

logger = logging.getLogger(__name__)
_FALLBACK = "这个具体的我一下子想不起来了"


async def _fallback_reply() -> str:
    """兜底那一句按人设口吻写在提示词里; 读不到时用代码里的同义句。"""
    try:
        text = (await get_prompt_text("chat.temporal_fallback") or "").strip()
    except Exception:
        text = ""
    return text or _FALLBACK


async def guard_temporal_reply(reply: str, *, question: str, evidence: str,
                               diagnostics: dict, agent_evidence: str = "",
                               agent_day_in_prompt: bool = False) -> str:
    """agent_evidence: AI 自己的固定作息与所问那天的安排 (TIME12), 交给核验一起看。

    agent_day_in_prompt: 所问的就是今天, 且主回复已带上「你的作息」段 —— 被排除的
    只有 AI 自己别的日子的经历时, 回复按的是同一份作息, 不再核验。2026-10-30 生产:
    "今晚加班了, 你呢?" 因为人设里一条随机日期的旧经历被判缺证据, 回复被换成
    "这件事的具体细节，我这会儿记不太清了。" 问的是过去某天时照常核验 (主回复里
    没有那天的作息)。
    """
    state = diagnostics.get("temporal_context", {})
    if not state.get("out_of_window_event_count") or state.get("in_window_event_count"):
        return reply
    user_side = state["out_of_window_event_count"] - state.get("out_of_window_ai_event_count", 0)
    if agent_day_in_prompt and user_side <= 0:
        state["event_check_status"] = "agent_schedule_evidence"
        return reply
    if agent_evidence:
        evidence = f"{evidence}\n【角色作息】\n{agent_evidence}"
    try:
        template = await get_prompt_text("chat.temporal_check")
        if not template:
            state["event_check_status"] = "disabled"
            return reply
        prompt = render_template(template, {
            "date": state["requested_event_date"], "question": question[:1000],
            "evidence": evidence[:8000], "reply": reply[:1500],
        })
        result = await asyncio.wait_for(invoke_json(get_utility_model(), prompt), timeout=5)
        if not isinstance(result, dict) or type(result.get("grounded")) is not bool:
            raise ValueError("Invalid temporal verdict")
        state["event_check_status"] = "grounded" if result["grounded"] else "rejected"
        # Never publish a second model's unverified replacement story.
        return reply if result["grounded"] else await _fallback_reply()
    except PromptDisabledError:
        state["event_check_status"] = "disabled"
        return reply
    except Exception as exc:
        logger.warning("[TEMPORAL] event check unavailable: %s", type(exc).__name__)
        state["event_check_status"] = "unavailable"
        # This request already has an explicit evidence gap. A failed check must
        # not turn an unsupported date-specific claim into a delivered fact.
        return await _fallback_reply()
