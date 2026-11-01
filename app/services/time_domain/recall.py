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

from app.services.llm.models import (
    get_chat_model,
    get_grounding_model,
    get_utility_model,
    invoke_json,
)
from app.services.llm.resilience import CallProfile, provider_name
from app.services.prompting.store import get_prompt_text, PromptDisabledError
from app.services.prompting.utils import render_template
from app.services.prompting.reply_prefix import build_reply_prefix
from app.services.time_domain.timestamps import parse_time
from app.services.time_domain.source_evidence import (
    MAX_SOURCE_RECORDS,
    attach_source_date_spans,
    validate_as_of_support,
    calendar_evidence_options,
    validate_calendar_labels,
)

logger = logging.getLogger(__name__)
MAX_MEMORIES = MAX_SOURCE_RECORDS
MAX_EVIDENCE_CHARS = 6000
_FALLBACK = "这件事的时间和细节，我还没完全对上，先不乱说。"
_QUESTION = re.compile(r"[?？]|什么|哪家|哪里|在哪|哪天|哪年|几号|店名|记得|记不记得")
_REFERENCE = re.compile(
    r"那次|上次|之前|以前|说过|提过|记得|昨天|前天|上个?月|今年|去年|[一二三四五六七八九十\d]{1,4}[年月日号]"
)
# These need event completeness, validity or clock precision beyond this batch.
_UNSUPPORTED = re.compile(
    r"多少次|几次|第几|第一次|最后一次|多久|多少天|几年|几个小时|几点|昨晚|早上|上午|下午|晚上|凌晨|仍然|现在还"
)
_PROFILE = CallProfile(
    timeout_s=8, max_retries=0, retry_backoff_s=(), allow_ollama_fallback=False
)
_CHECK_PROFILE = CallProfile(
    timeout_s=5, max_retries=0, retry_backoff_s=(), allow_ollama_fallback=False
)
# A partial recall set cannot prove non-occurrence. This is a conservative
# extra check for common denial forms, not a complete Chinese entailment model.
_DENIAL = re.compile(
    r"(?:没(?:有)?|未(?:曾)?|不曾|不)(?:(?:真的|真|一起|实际|正式|确实|曾经|曾|还|再)|\s){0,3}"
    r"(?:(?:对|跟|和)(?:我|你|他|她))?"
    r"(下单|参加|成行|报名|领证|结婚|离职|在|去|到|来|住|买|购|办|订|提|说|做|玩|吃|喝|用|见|听)"
)
_UNCERTAIN_CLAUSE = re.compile(
    r"不确定|不知道|不清楚|未确认|没确认|是否|有没有|假如|要是|如果|吗"
)


def has_unsupported_denial(
    reply: str, records: list[dict], recent_user_context: str
) -> bool:
    """Require explicit negative material before letting the model check it.

    Questions, assistant history, style examples and character cities are excluded
    as absence evidence. A matching predicate is necessary, never sufficient;
    the subsequent model check must still verify subject, event and date.
    """

    def predicates(text):
        return {
            (
                "purchase"
                if m in {"下单", "订", "买", "购"}
                else "attendance" if m in {"去", "来", "到", "参加", "成行"} else m
            )
            for clause in re.findall(
                r"[^，,。.!?！？；;|\n]+[，,。.!?！？；;|\n]*", text
            )
            if not _UNCERTAIN_CLAUSE.search(clause)
            and "?" not in clause
            and "？" not in clause
            for m in _DENIAL.findall(clause)
        }

    supported = predicates(
        "\n".join(r["content"] for r in records) + "\n" + recent_user_context
    )
    return bool(predicates(reply) - supported)


_MENTION_DAY = re.compile(
    r"(?:那天|当天)[^，,。.!?！？；;|\n]{0,12}(?:告诉|跟我说|和我说|说过|提过|聊过|说的)"
)


def has_unsupported_same_day_mention(reply: str, records: list[dict]) -> bool:
    """An event day cannot silently become the day it was mentioned.

    Conservatively reject this deictic claim when the associated records do not
    establish a shared day. This does not infer which day 'that day' denotes;
    even aligned dates must still pass the independent semantic check.
    """
    if not _MENTION_DAY.search(reply):
        return False
    return not records or any(
        not r["event_date"]
        or not r["statement_date"]
        or r["event_date"] != r["statement_date"]
        for r in records
    )


def validate_answer(
    value: object, *, records: list[dict], max_chars: int
) -> tuple[str, list[dict]]:
    """Validate local candidate IDs, never ask a model to invent source offsets.

    An association is a model claim, not proof of entailment. The independent
    answer check below sees all candidates and must validate the actual reply.
    """
    if not isinstance(value, dict) or set(value) != {"memory_indices", "reply"}:
        raise ValueError("invalid_contract")
    matches, reply = value["memory_indices"], value["reply"]
    if (
        not isinstance(reply, str)
        or not reply.strip()
        or len(reply) > max_chars
        or not isinstance(matches, list)
        or len(matches) > len(records)
    ):
        raise ValueError("invalid_bounds")
    seen, associations = set(), []
    for index in matches:
        if type(index) is not int or not 0 <= index < len(records) or index in seen:
            raise ValueError("invalid_index")
        seen.add(index)
        associations.append({"index": index, "source": records[index]["source"]})
    return reply.strip(), associations


def recall_payload(
    question: str, memories: list, query: dict, *, validity_records: dict | None = None
) -> tuple[dict | None, str]:
    as_of = query.get("operation", {}).get("type") == "as_of"
    if not as_of and (
        not _QUESTION.search(question)
        or not _REFERENCE.search(question)
        or _UNSUPPORTED.search(question)
    ):
        return None, "not_recall_question"
    if query.get("operation", {}).get("type") in {
        "count",
        "order",
        "duration",
        "current",
        "planned",
    }:
        return None, "unsupported_operation"
    anchor = parse_time(query.get("anchor_at"))
    if anchor is None or query.get("anchor_source") not in {
        "persisted_message",
        "server_received",
    }:
        return None, "unknown_anchor"
    if not memories and not (as_of and validity_records):
        return None, "no_candidates"
    if (
        len(question) > 1000
        or len(memories) > MAX_MEMORIES
        or sum(len(m.text) for m in memories) > MAX_EVIDENCE_CHARS
    ):
        return None, "input_limit"
    zone = ZoneInfo(query["timezone"])
    day = anchor.astimezone(zone).date()

    def date(value):
        parsed = parse_time(value)
        return parsed.astimezone(zone).date() if parsed else None

    ranges = []
    if (
        query.get("status") == "resolved"
        and query.get("date_basis", {}).get("event_range_eligible") is True
    ):
        for period in query.get("ranges", [])[:8]:
            start, end = date(period.get("start")), date(period.get("end"))
            if start is not None and end is not None and start <= end:
                ranges.append({"start": str(start), "end": str(end)})
    records = []
    if as_of:
        if query.get("status") != "resolved" or len(query.get("ranges", [])) != 1:
            return None, "unknown_as_of_time"
        if query.get("date_basis", {}).get("kind") != "event_candidate":
            return None, "unbound_as_of_time"
        from app.services.memory.recording.validity import public_validity

        # Only resolve explicit personal predicates. "你记得我住哪" asks about
        # the user; a bare "你记得" is not evidence about the Agent's life.
        subjects = set(
            re.findall(
                r"(我|你)(?:当时|那时|那会儿|那天|的|也|还|仍|已经|曾经|曾|是否|到底|真的|实际|正|一直|都|就|又|确定|肯定|以前){0,4}"
                r"(?:住|在|去|做|工作|职业|家|身体|病|计划|安排|用|学|拿|参加)",
                question,
            )
        )
        owner = (
            {"我": "user", "你": "ai"}.get(next(iter(subjects)))
            if len(subjects) == 1
            else None
        )
        if owner == "ai" and re.search(r"住|在|去|到|所在地|住址|家", question):
            # These assertions are assistant-authored utterances, not an
            # independently confirmed itinerary. Repeating an earlier invented
            # trip would bypass the character grounding of the ordinary path.
            # Current profile location also cannot establish a past location.
            return None, "agent_location_not_verified"
        for memory in memories:
            if owner is not None and memory.source != owner:
                continue
            for evidence in (validity_records or {}).get(
                (memory.source, memory.id), []
            ):
                value = public_validity(evidence.get("validity"))
                if (
                    value["state"] != "asserted"
                    or evidence.get("source") != memory.source
                ):
                    continue
                # An out-of-period assertion cannot establish either presence
                # OR absence during the requested period (the person may have
                # returned later). Do not let the model turn it into a denial.
                if evidence.get("membership") == "outside":
                    continue
                content = evidence.get("content")
                if not isinstance(content, str) or not 1 <= len(content) <= 1000:
                    return None, "input_limit"
                event, statement = date(evidence.get("event_time")), date(
                    evidence.get("statement_time")
                )
                recorded = parse_time(evidence.get("recorded_at"))
                retracted = parse_time(evidence.get("retracted_at"))
                records.append(
                    {
                        "index": len(records),
                        "content": content,
                        "source": memory.source,
                        "source_excerpt": evidence.get("source_excerpt"),
                        "event_date": str(event) if event else None,
                        "statement_date": str(statement) if statement else None,
                        "statement_at": (
                            parse_time(evidence.get("statement_time")).isoformat()
                            if parse_time(evidence.get("statement_time"))
                            else None
                        ),
                        "validity": value,
                        "validity_membership": evidence.get("membership", "unknown"),
                        "system_recorded_at": (
                            recorded.isoformat() if recorded else None
                        ),
                        "system_retracted_at": (
                            retracted.isoformat() if retracted else None
                        ),
                    }
                )
        # These are exact, scope-checked user messages, not inferred memories.
        # Preserve their original wording for both generation and checking;
        # statement time must never stand in for a fact's effective date.
        if owner != "ai":
            for key, evidence in (validity_records or {}).items():
                if key[0] not in {"dialogue", "linked"}:
                    continue
                for item in evidence:
                    content = item.get("content")
                    if (
                        item.get("evidence_type")
                        not in {"recent_statement", "linked_statement"}
                        or item.get("source") != "user"
                        or not isinstance(content, str)
                        or not 1 <= len(content) <= 800
                    ):
                        return None, "invalid_source"
                    statement = date(item.get("statement_time"))
                    records.append(
                        {
                            "index": len(records),
                            "content": content,
                            "source": "user",
                            "source_excerpt": content,
                            "event_date": None,
                            "statement_date": str(statement) if statement else None,
                            "statement_at": (
                                parse_time(item.get("statement_time")).isoformat()
                                if parse_time(item.get("statement_time"))
                                else None
                            ),
                            "evidence_type": item["evidence_type"],
                            "validity_membership": "not_materialized",
                        }
                    )
        if not records:
            return None, "no_as_of_evidence"
        attach_source_date_spans(records, timezone_name=query["timezone"])
        if (
            len(records) > MAX_MEMORIES
            or len(json.dumps(records, ensure_ascii=False)) > MAX_EVIDENCE_CHARS
        ):
            return None, "input_limit"
        return {
            "question": question,
            "operation": "as_of",
            "calendar": {"today": str(day), "timezone": query["timezone"]},
            "question_date_ranges": [],
            "requested_validity_range": query["ranges"][0],
            "memories": records,
        }, "eligible"
    for i, memory in enumerate(memories):
        if (
            memory.source not in {"user", "ai"}
            or not memory.text
            or len(memory.text) > 1000
        ):
            return None, "invalid_source"
        event, statement = date(memory.occur_time), date(memory.statement_time)
        records.append(
            {
                "index": i,
                "content": memory.text,
                "source": memory.source,
                "event_date": str(event) if event else None,
                "in_question_date_range": (
                    any(r["start"] <= str(event) <= r["end"] for r in ranges)
                    if event and ranges
                    else None
                ),
                "statement_date": str(statement) if statement else None,
                "days_ago": (day - event).days if event else None,
                "months_ago": (
                    (day.year - event.year) * 12 + day.month - event.month
                    if event
                    else None
                ),
                "years_ago": day.year - event.year if event else None,
            }
        )
    return {
        "question": question,
        "calendar": {"today": str(day), "timezone": query["timezone"]},
        "question_date_ranges": ranges,
        "memories": records,
    }, "eligible"


async def answer_temporal_recall(
    question: str,
    memories: list,
    diagnostics: dict,
    *,
    max_chars: int,
    recent_context: str = "",
    personality: str = "",
    recent_user_context: str = "",
    validity_records: dict | None = None,
) -> str | None:
    state = diagnostics.get("temporal_context")
    if not isinstance(state, dict):
        return None
    info = {
        "version": 1,
        "status": "not_requested",
        "reason": "not_recall_question",
        "candidate_count": len(memories),
        "matches": [],
    }
    state["recall"] = info
    started = monotonic()
    try:
        payload, reason = recall_payload(
            question,
            memories,
            state.get("query") or {},
            validity_records=validity_records,
        )
        info["reason"] = reason
        if payload is None:
            return None
        if (
            len(recent_context) > 4000
            or len(personality) > 4000
            or len(recent_user_context) > 4000
        ):
            info["reason"] = "input_limit"
            return None
        if payload.get("operation") == "as_of":
            from app.services.time_domain.timestamps import exclusive_end

            period = payload["requested_validity_range"]
            requested_end = exclusive_end(
                parse_time(period["end"]), period["end_boundary"]
            )
            requested_start = parse_time(period["start"])
            for record in payload["memories"]:
                for span in record.get("literal_date_spans", []):
                    begin, end = parse_time(span["start"]), parse_time(
                        span["end_exclusive"]
                    )
                    span["relation_to_query"] = (
                        "contains"
                        if begin <= requested_start and end >= requested_end
                        else (
                            "ended_before"
                            if span["precision"] == "range" and end <= requested_start
                            else "outside_or_partial"
                        )
                    )
            requested_day = parse_time(payload["requested_validity_range"]["start"])
            payload["calendar"]["requested_weekday"] = requested_day.astimezone(
                ZoneInfo(payload["calendar"]["timezone"])
            ).isoweekday()
        payload.update(recent_context=recent_context, character=personality)
        # A disabled check must disable the whole route, never deliver unchecked text.
        templates = [
            await get_prompt_text(key)
            for key in ("chat.temporal_recall", "chat.temporal_recall_check")
        ]
        if not all(templates):
            info.update(status="disabled", reason="prompt_disabled")
            return None
        info.update(
            status="running",
            reason="eligible",
            candidate_count=len(payload["memories"]),
            selected_memory_count=len(memories),
        )
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
        prompt = [
            SystemMessage(content=render_template(templates[0], {})),
            HumanMessage(content=json.dumps(generation_payload, ensure_ascii=False)),
        ]
        generation_model = (
            get_chat_model()
            if payload.get("operation") == "as_of"
            else get_grounding_model()
        )
        generation_options = (
            {"response_format": {"type": "json_object"}}
            if payload.get("operation") == "as_of"
            and provider_name(generation_model) in {"ark", "dashscope", "deepseek"}
            else {}
        )
        value = await asyncio.wait_for(
            invoke_json(
                generation_model,
                prompt,
                temperature=0,
                max_tokens=1800,
                profile=_PROFILE,
                **generation_options,
            ),
            timeout=9,
        )
        reply, matches = validate_answer(
            value, records=payload["memories"], max_chars=max_chars
        )
        info.update(matches=matches)
        matched = [payload["memories"][m["index"]] for m in matches]
        if has_unsupported_same_day_mention(reply, matched):
            info.update(status="rejected", reason="unsupported_mention_time")
            return _FALLBACK
        if has_unsupported_denial(reply, matched, recent_user_context):
            info.update(status="rejected", reason="unsupported_absence")
            return _FALLBACK
        historical = payload.get("operation") == "as_of"
        check_payload = (
            {**payload, "calendar_evidence": calendar_evidence_options(payload)}
            if historical
            else payload
        )
        check = [
            SystemMessage(content=render_template(templates[1], {})),
            HumanMessage(
                content=json.dumps(
                    {"input": check_payload, "answer": value}, ensure_ascii=False
                )
            ),
        ]
        verdict = await asyncio.wait_for(
            invoke_json(
                get_utility_model() if historical else get_grounding_model(),
                check,
                temperature=0,
                max_tokens=700 if historical else 300,
                profile=_CHECK_PROFILE,
            ),
            timeout=5 if historical else 6,
        )
        # Keep the old managed prompt schema readable during code/prompt rollout.
        allowed_keys = (
            [{"grounded", "time_support"}, {"grounded", "calendar_evidence_labels"}]
            if historical
            else [{"grounded"}]
        )
        if (
            not isinstance(verdict, dict)
            or set(verdict) not in allowed_keys
            or type(verdict["grounded"]) is not bool
        ):
            raise ValueError("invalid_verdict")
        if not verdict["grounded"]:
            info.update(status="rejected", reason="answer_not_grounded")
            return _FALLBACK
        calendar_validator = (
            validate_calendar_labels
            if "calendar_evidence_labels" in verdict
            else validate_as_of_support
        )
        if historical and not calendar_validator(verdict, payload, matches):
            info.update(status="rejected", reason="unsupported_historical_time")
            return _FALLBACK
        info.update(status="accepted", reason="checked")
        return reply
    except PromptDisabledError:
        info.update(status="disabled", reason="prompt_disabled")
        return None
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        info.update(
            status="unavailable",
            reason=(
                "validation_failed" if isinstance(exc, ValueError) else "service_failed"
            ),
        )
        logger.warning("[TEMPORAL-RECALL] unavailable: %s", type(exc).__name__)
        return _FALLBACK
    finally:
        info["duration_ms"] = round((monotonic() - started) * 1000)
