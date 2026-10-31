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
_FALLBACK = "这件事的具体细节，我这会儿记不太清了。"


async def guard_temporal_reply(reply: str, *, question: str, evidence: str,
                               diagnostics: dict) -> str:
    state = diagnostics.get("temporal_context", {})
    if not state.get("out_of_window_event_count") or state.get("in_window_event_count"):
        return reply
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
        return reply if result["grounded"] else _FALLBACK
    except PromptDisabledError:
        state["event_check_status"] = "disabled"
        return reply
    except Exception as exc:
        logger.warning("[TEMPORAL] event check unavailable: %s", type(exc).__name__)
        state["event_check_status"] = "unavailable"
        # This request already has an explicit evidence gap. A failed check must
        # not turn an unsupported date-specific claim into a delivered fact.
        return _FALLBACK
