"""After the main reply is generated: rewrite once, then drop copied bubbles.

Order is prevent (prompt sections) → rewrite once → drop near-verbatim single
bubbles; everything happens before the reply is sent. Diagnostics go to
response_diagnostics["reply_repetition"]; the older "restated_retry" and
"restated_bubbles" keys are still written for the Trace view and old readers.
"""
from __future__ import annotations

from app.services.chat.reply_repetition import breaker
from app.services.chat.reply_repetition.policy import Policy, current
from app.services.chat.reply_repetition.rules import (
    drop_restated_bubbles,
    previous_ai_turn,
    previous_reply_situation,
    restated_overlap,
    restated_retry_note,
)

DIAGNOSTICS_VERSION = 1


def _diagnostics(ctx, policy: Policy) -> dict:
    record = ctx.response_diagnostics.setdefault("reply_repetition", {})
    record.setdefault("version", DIAGNOSTICS_VERSION)
    record["policy_revision"] = policy.revision[:12]
    if not policy.valid:
        record["stored_policy_valid"] = False
    return record


def _skipped(ctx) -> bool:
    return bool(ctx.reply_is_fallback or ctx.contradiction_inquiry or ctx.crisis_care_turn)


async def rewrite_if_restated(ctx, services, generate_kwargs, policy: Policy | None = None) -> None:
    """Every bubble restates the previous turn (ack / overlap turns only): rewrite once."""
    policy = current() if policy is None else policy
    record = _diagnostics(ctx, policy)
    if _skipped(ctx):
        record["rewrite"] = "not_applicable"
        return
    await breaker.record("turns")
    # Only the two situations prone to restating; ordinary turns rely on the prompt section.
    situation = previous_reply_situation(ctx.messages_dicts, ctx.current_turn_ids, ctx.user_message)
    if situation is None:
        record["rewrite"] = "ordinary_turn"
        return
    previous, _ = previous_ai_turn(ctx.messages_dicts, ctx.current_turn_ids)
    if not drop_restated_bubbles(ctx.replies, previous, ctx.user_message,
                                 threshold=policy.rewrite_overlap)[2]:
        record["rewrite"] = "not_restated"
        return
    if not policy.rewrite_enabled:
        record["rewrite"] = "disabled"
        return
    gate = await breaker.state(policy)
    record["breaker"] = gate
    if gate["state"] == "open":
        record["rewrite"] = "breaker_open"
        return
    note = await restated_retry_note(previous)
    if note is None:
        record["rewrite"] = "prompt_disabled"
        ctx.response_diagnostics["restated_retry"] = "disabled"
        return
    build = ctx._build_main_chat_messages

    async def with_note():
        return [*await build(), {"role": "system", "content": note}]

    await breaker.record("rewrites")
    retry = await services._generate_reply(
        **{**generate_kwargs, "chat_messages_factory": with_note, "force_main_prompt": True}
    )
    replies, _, is_fallback, emotion = retry
    if is_fallback or drop_restated_bubbles(replies, previous, ctx.user_message,
                                            threshold=policy.rewrite_overlap)[2]:
        record["rewrite"] = ctx.response_diagnostics["restated_retry"] = "still_restated"
        return
    ctx.replies, ctx.reply_emotion_pre = replies, emotion
    record["rewrite"] = ctx.response_diagnostics["restated_retry"] = "rewritten"


def drop_restated(ctx, policy: Policy | None = None) -> None:
    """Backstop: don't send a bubble that is basically a line from the previous turn.

    Ack / overlap turns also catch paraphrases; other turns only near-verbatim
    copies. Topic-continuation lead lines go back to the previous topic on
    purpose and are not judged; a reply that is all restatement is left alone
    (the rewrite step already had its one try).
    """
    policy = current() if policy is None else policy
    record = _diagnostics(ctx, policy)
    if ctx.crisis_care_turn or ctx.reply_is_fallback or ctx.contradiction_inquiry:
        record["backstop"] = {"skipped": "not_applicable"}
        return
    situation = previous_reply_situation(ctx.messages_dicts, ctx.current_turn_ids, ctx.user_message)
    previous, _ = previous_ai_turn(ctx.messages_dicts, ctx.current_turn_ids)
    if not previous:
        record["backstop"] = {"skipped": "no_previous_turn"}
        return
    if not policy.backstop_enabled:
        record["backstop"] = {"skipped": "disabled"}
        return
    lead = list(ctx.continuation_lines or [])
    lead = lead if ctx.replies[: len(lead)] == lead else []
    rule = ({"threshold": policy.backstop_overlap} if situation
            else {"threshold": policy.backstop_verbatim_overlap, "whole_turn": False})
    body, dropped, all_restated = drop_restated_bubbles(
        ctx.replies[len(lead):], previous, ctx.user_message, **rule
    )
    if dropped:
        ctx.replies = [*lead, *body]
        ctx.reply_emotion_pre = None  # emotion / sticker must follow the text actually sent
    outcome = {
        "situation": situation[0] if situation else "other",
        "previous_bubbles": len(previous),
        "dropped": dropped,
        "all_restated": all_restated,
        "max_overlap": round(max((restated_overlap(r, previous) for r in body), default=0.0), 2),
    }
    ctx.response_diagnostics["restated_bubbles"] = outcome
    record["backstop"] = {**outcome, "threshold": rule["threshold"]}
