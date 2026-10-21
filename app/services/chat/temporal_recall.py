"""Bounded read-only recall answers with local source IDs and a separate verdict.

Only ordinary, short recall questions use this path. It never changes recall,
memories, schedules or pending actions. Source association is not fact proof.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from time import monotonic
from zoneinfo import ZoneInfo

from langchain_core.messages import HumanMessage, SystemMessage

from app.services.llm.models import get_grounding_model, invoke_json
from app.services.llm.resilience import CallProfile
from app.services.prompting.store import get_prompt_text, PromptDisabledError
from app.services.prompting.utils import render_template
from app.services.prompting.reply_prefix import build_reply_prefix
from app.services.temporal import parse_time

logger = logging.getLogger(__name__)
MAX_MEMORIES = 20
MAX_EVIDENCE_CHARS = 6000
_FALLBACK = "这件事的时间和细节，我还没完全对上，先不乱说。"
_QUESTION = re.compile(r"[?？]|什么|哪家|哪里|在哪|哪天|哪年|几号|店名|记得|记不记得")
_REFERENCE = re.compile(r"那次|上次|之前|以前|说过|提过|记得|昨天|前天|上个?月|今年|去年|[一二三四五六七八九十\d]{1,4}[年月日号]")
# These need event completeness, validity or clock precision beyond this batch.
_UNSUPPORTED = re.compile(r"多少次|几次|第几|第一次|最后一次|多久|多少天|几年|几个小时|几点|昨晚|早上|上午|下午|晚上|凌晨|仍然|现在还")
_PROFILE = CallProfile(timeout_s=8, max_retries=0, retry_backoff_s=(), allow_ollama_fallback=False)
_CHECK_PROFILE = CallProfile(timeout_s=5, max_retries=0, retry_backoff_s=(), allow_ollama_fallback=False)
# A partial recall set cannot prove non-occurrence. This is a conservative
# extra check for common denial forms, not a complete Chinese entailment model.
_DENIAL = re.compile(r"(?:没(?:有)?|未(?:曾)?|不曾)(?:(?:真的|真|一起|实际|正式|确实|曾经|曾|还|再)|\s){0,3}"
                     r"(?:(?:对|跟|和)(?:我|你|他|她))?"
                     r"(下单|参加|成行|报名|领证|结婚|离职|去|到|来|住|买|购|办|订|提|说|做|玩|吃|喝|用|见|听)")
_UNCERTAIN_CLAUSE = re.compile(r"不确定|不知道|不清楚|未确认|没确认|是否|有没有|假如|要是|如果|吗")


def has_unsupported_denial(reply: str, records: list[dict], recent_user_context: str) -> bool:
    """Require explicit negative material before letting the model check it.

    Questions, assistant history, style examples and character cities are excluded
    as absence evidence. A matching predicate is necessary, never sufficient;
    the subsequent model check must still verify subject, event and date.
    """
    def predicates(text):
        return {"purchase" if m in {"下单", "订", "买", "购"} else "attendance" if m in {"去", "来", "到", "参加", "成行"} else m
                for clause in re.findall(r"[^，,。.!?！？；;|\n]+[，,。.!?！？；;|\n]*", text)
                if not _UNCERTAIN_CLAUSE.search(clause) and "?" not in clause and "？" not in clause
                for m in _DENIAL.findall(clause)}
    supported = predicates("\n".join(r["content"] for r in records) + "\n" + recent_user_context)
    return bool(predicates(reply) - supported)


_MENTION_DAY = re.compile(r"(?:那天|当天)[^，,。.!?！？；;|\n]{0,12}(?:告诉|跟我说|和我说|说过|提过|聊过|说的)")


def has_unsupported_same_day_mention(reply: str, records: list[dict]) -> bool:
    """An event day cannot silently become the day it was mentioned.

    Conservatively reject this deictic claim when the associated records do not
    establish a shared day. This does not infer which day 'that day' denotes;
    even aligned dates must still pass the independent semantic check.
    """
    if not _MENTION_DAY.search(reply):
        return False
    return not records or any(not r["event_date"] or not r["statement_date"]
                              or r["event_date"] != r["statement_date"] for r in records)


def validate_answer(value: object, *, records: list[dict], max_chars: int) -> tuple[str, list[dict]]:
    """Validate local candidate IDs, never ask a model to invent source offsets.

    An association is a model claim, not proof of entailment. The independent
    answer check below sees all candidates and must validate the actual reply.
    """
    if not isinstance(value, dict) or set(value) != {"memory_indices", "reply"}:
        raise ValueError("invalid_contract")
    matches, reply = value["memory_indices"], value["reply"]
    if (not isinstance(reply, str) or not reply.strip() or len(reply) > max_chars
            or not isinstance(matches, list) or len(matches) > len(records)):
        raise ValueError("invalid_bounds")
    seen, associations = set(), []
    for index in matches:
        if type(index) is not int or not 0 <= index < len(records) or index in seen:
            raise ValueError("invalid_index")
        seen.add(index)
        associations.append({"index": index, "source": records[index]["source"]})
    return reply.strip(), associations


def recall_payload(question: str, memories: list, query: dict) -> tuple[dict | None, str]:
    if not _QUESTION.search(question) or not _REFERENCE.search(question) or _UNSUPPORTED.search(question):
        return None, "not_recall_question"
    if query.get("operation", {}).get("type") in {"count", "order", "duration", "as_of", "current", "planned"}:
        return None, "unsupported_operation"
    anchor = parse_time(query.get("anchor_at"))
    if anchor is None or query.get("anchor_source") not in {"persisted_message", "server_received"}:
        return None, "unknown_anchor"
    if not memories:
        return None, "no_candidates"
    if len(question) > 1000 or len(memories) > MAX_MEMORIES or sum(len(m.text) for m in memories) > MAX_EVIDENCE_CHARS:
        return None, "input_limit"
    zone = ZoneInfo(query["timezone"])
    day = anchor.astimezone(zone).date()
    def date(value):
        parsed = parse_time(value)
        return parsed.astimezone(zone).date() if parsed else None
    ranges = []
    if query.get("status") == "resolved" and query.get("date_basis", {}).get("event_range_eligible") is True:
        for period in query.get("ranges", [])[:8]:
            start, end = date(period.get("start")), date(period.get("end"))
            if start is not None and end is not None and start <= end:
                ranges.append({"start": str(start), "end": str(end)})
    records = []
    for i, memory in enumerate(memories):
        if memory.source not in {"user", "ai"} or not memory.text or len(memory.text) > 1000:
            return None, "invalid_source"
        event, statement = date(memory.occur_time), date(memory.statement_time)
        records.append({"index": i, "content": memory.text, "source": memory.source,
                        "event_date": str(event) if event else None,
                        "in_question_date_range": (any(r["start"] <= str(event) <= r["end"] for r in ranges)
                                                   if event and ranges else None),
                        "statement_date": str(statement) if statement else None,
                        "days_ago": (day-event).days if event else None,
                        "months_ago": (day.year-event.year)*12+day.month-event.month if event else None,
                        "years_ago": day.year-event.year if event else None})
    return {"question": question, "calendar": {"today": str(day),
             "timezone": query["timezone"]}, "question_date_ranges": ranges, "memories": records}, "eligible"


async def answer_temporal_recall(question: str, memories: list, diagnostics: dict, *, max_chars: int,
                                 recent_context: str = "", personality: str = "",
                                 recent_user_context: str = "") -> str | None:
    state = diagnostics.get("temporal_context")
    if not isinstance(state, dict):
        return None
    info = {"version": 1, "status": "not_requested", "reason": "not_recall_question",
            "candidate_count": len(memories), "matches": []}
    state["recall"] = info
    started = monotonic()
    try:
        payload, reason = recall_payload(question, memories, state.get("query") or {})
        info["reason"] = reason
        if payload is None:
            return None
        if len(recent_context) > 4000 or len(personality) > 4000 or len(recent_user_context) > 4000:
            info["reason"] = "input_limit"
            return None
        payload.update(recent_context=recent_context, character=personality)
        # A disabled check must disable the whole route, never deliver unchecked text.
        templates = [await get_prompt_text(key) for key in ("chat.temporal_recall", "chat.temporal_recall_check")]
        if not all(templates):
            info.update(status="disabled", reason="prompt_disabled")
            return None
        info.update(status="running", reason="eligible")
        # Keep Web-managed reply style and character facts on this reply path.
        # Factual checks receive data only; style preferences cannot change
        # the factual contract or manufacture details.
        style = await build_reply_prefix()
        if len(style) > 8000:
            info.update(status="not_requested", reason="input_limit")
            return None
        # Style is a preference, not a second source of event facts. Keeping
        # one instruction message also avoids provider-specific multi-system
        # precedence. It must not override the factual output contract.
        generation_payload = {**payload, "style_preferences": style}
        prompt = [SystemMessage(content=render_template(templates[0], {})),
                  HumanMessage(content=json.dumps(generation_payload, ensure_ascii=False))]
        value = await asyncio.wait_for(invoke_json(get_grounding_model(), prompt, temperature=0,
                        max_tokens=1800, profile=_PROFILE), timeout=9)
        reply, matches = validate_answer(value, records=payload["memories"], max_chars=max_chars)
        info.update(matches=matches)
        matched = [payload["memories"][m["index"]] for m in matches]
        if has_unsupported_same_day_mention(reply, matched):
            info.update(status="rejected", reason="unsupported_mention_time")
            return _FALLBACK
        if has_unsupported_denial(reply, matched, recent_user_context):
            info.update(status="rejected", reason="unsupported_absence")
            return _FALLBACK
        check = [SystemMessage(content=render_template(templates[1], {})),
                 HumanMessage(content=json.dumps({"input": payload, "answer": value}, ensure_ascii=False))]
        verdict = await asyncio.wait_for(invoke_json(get_grounding_model(), check, temperature=0,
                        max_tokens=300, profile=_CHECK_PROFILE), timeout=6)
        if not isinstance(verdict, dict) or set(verdict) != {"grounded"} or type(verdict["grounded"]) is not bool:
            raise ValueError("invalid_verdict")
        if not verdict["grounded"]:
            info.update(status="rejected", reason="answer_not_grounded")
            return _FALLBACK
        info.update(status="accepted", reason="checked")
        return reply
    except PromptDisabledError:
        info.update(status="disabled", reason="prompt_disabled")
        return None
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        info.update(status="unavailable", reason="validation_failed" if isinstance(exc, ValueError) else "service_failed")
        logger.warning("[TEMPORAL-RECALL] unavailable: %s", type(exc).__name__)
        return _FALLBACK
    finally:
        info["duration_ms"] = round((monotonic()-started)*1000)
