"""Main-reply web search via Volcengine Ark Responses API.

Why a separate module instead of the langchain chat-completions path:
the built-in `web_search` tool is only exposed on Ark's Responses API
(`POST {ARK_BASE_URL}/responses`), not on chat completions. The model decides
per turn whether to actually search (casual chat does not trigger a search,
so no plugin fee); when it does search, results are injected server-side and
the answer comes back with url citations.

Integration contract (see reply_generate._run_main_llm):
- `generate_with_web_search(chat_messages, model)` returns the reply text,
  or None on ANY failure — the caller falls back to the normal streaming
  path, so this feature can never take the main reply down (fail-open).
- Usage is recorded to usage_tracker under the same "ark/<model>" key the
  streaming path uses, keeping admin cost stats accurate.
- The call is also recorded as a manual trace step: it bypasses langchain's
  callback handler, so without that the main reply would be missing from the
  trace tree entirely (only the small-model steps would show).

Verified against production 2026-07-25: multi-turn system/assistant/user
input, on-demand triggering, [EMO:] marker instruction honored, ~3s when a
search fires vs ~1.2s when not.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from app.config import settings
from app.observability.events import EVT_REPLY_WEB_SEARCH
from app.services.chat.local_tracer import record_manual_llm_run
from app.services.llm import usage_tracker

logger = logging.getLogger(__name__)

# Search turns take noticeably longer than plain ones (observed ~3s vs ~1.2s);
# generous ceiling so a slow search degrades to fallback instead of hanging.
_TIMEOUT_S = 45.0

_ALLOWED_ROLES = {"system", "user", "assistant"}


def _to_responses_input(chat_messages: list[dict]) -> list[dict[str, Any]]:
    """chat-completions style dicts → Responses API input array.

    Roles outside system/user/assistant (e.g. tool) never appear in our main
    reply prompt; drop defensively rather than 400 the whole call.
    """
    out: list[dict[str, Any]] = []
    for m in chat_messages:
        role = str(m.get("role") or "")
        content = m.get("content")
        if role not in _ALLOWED_ROLES or not isinstance(content, str):
            continue
        out.append({"role": role, "content": content})
    return out


def _extract_output_text(payload: dict) -> tuple[str, int]:
    """Return (reply_text, web_search_call_count) from a Responses API body."""
    outputs = payload.get("output")
    if not isinstance(outputs, list):
        return "", 0
    search_calls = 0
    texts: list[str] = []
    for item in outputs:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "web_search_call":
            search_calls += 1
            continue
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "output_text":
                text = part.get("text")
                if isinstance(text, str) and text:
                    texts.append(text)
    return "".join(texts).strip(), search_calls


def _usage_tokens(payload: dict) -> tuple[int, int, int] | None:
    """(input, output, cached_input) from a Responses API body; None if absent."""
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    input_details = usage.get("input_tokens_details")
    cached = 0
    if isinstance(input_details, dict):
        cached = int(input_details.get("cached_tokens", 0) or 0)
    return (
        int(usage.get("input_tokens", 0) or 0),
        int(usage.get("output_tokens", 0) or 0),
        cached,
    )


async def generate_with_web_search(
    chat_messages: list[dict], *, model: str,
) -> str | None:
    """One-shot main reply through Ark Responses API, forcing a web search.

    `tool_choice="required"` is not optional: with the production system
    prompt the model answers "我帮你查下" and skips the tool under the default
    "auto" (0/16 in measurement). Callers gate on web_search_gate, so by the
    time we get here a search is known to be wanted.

    Returns reply text, or None when the call cannot produce one (missing
    key, HTTP error, timeout, empty output). Callers must treat None as
    "fall back to the normal LLM path".
    """
    if not settings.ark_api_key or not model:
        return None
    body = {
        "model": model,
        "input": _to_responses_input(chat_messages),
        "tools": [{"type": "web_search"}],
        "tool_choice": "required",
        "stream": False,
    }
    if not body["input"]:
        return None
    endpoint = settings.ark_base_url.rstrip("/") + "/responses"
    started_at = datetime.now(timezone.utc)
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            response = await client.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {settings.ark_api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
        if response.status_code != 200:
            # ToolNotOpen (plugin deactivated), quota errors, 5xx — all fall back.
            logger.warning(
                "[WEB-SEARCH] responses API status=%s body=%s",
                response.status_code,
                (response.text or "")[:300],
            )
            return None
        payload = response.json()
    except Exception as e:  # noqa: BLE001 — any transport/parse error falls back
        logger.warning(f"[WEB-SEARCH] responses API call failed: {e}")
        return None

    text, search_calls = _extract_output_text(payload)
    if not text:
        logger.warning("[WEB-SEARCH] empty output, falling back to normal path")
        return None

    tokens = _usage_tokens(payload)
    if tokens is not None:
        usage_tracker.record(
            f"ark/{model}", tokens[0], tokens[1], cached_input_tokens=tokens[2],
        )
    # This call never touches langchain, so nothing would appear in the trace
    # tree without an explicit record — the main reply would look missing.
    record_manual_llm_run(
        name="ArkResponsesAPI",
        model_name=model,
        provider="ark",
        messages=chat_messages,
        output_text=text,
        started_at=started_at,
        ended_at=datetime.now(timezone.utc),
        input_tokens=tokens[0] if tokens else None,
        output_tokens=tokens[1] if tokens else None,
        cached_input_tokens=tokens[2] if tokens else None,
        metadata={"web_search_calls": search_calls},
    )
    logger.info(
        f"[WEB-SEARCH] reply len={len(text)} search_calls={search_calls}",
        extra={
            "event": EVT_REPLY_WEB_SEARCH,
            "web_search_calls": search_calls,
            "raw_response_len": len(text),
        },
    )
    return text
