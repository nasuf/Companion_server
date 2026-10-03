"""Contracts of the first graph adapter; these are not durable snapshots.

Domain values are invocation-owned. Only decisions and diagnostics are JSON
state. R01/R02 must add replay-safe business snapshots before persistence.
"""

from dataclasses import dataclass


class ChatNodeContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class NodeContract:
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


CONTRACTS = {
    "load_turn": NodeContract(
        ("conversation_id", "user_message"), ("messages_dicts", "current_turn_ids")
    ),
    "guard": NodeContract(
        ("messages_dicts", "current_turn_ids"), ("crisis_decision", "cached_patience")
    ),
    "pending": NodeContract(("crisis_decision",), ()),
    "prepare_reads": NodeContract(("crisis_decision",), ("response_diagnostics",)),
    "identify": NodeContract(("response_diagnostics",), ("detected_intent",)),
    "prepare_routes": NodeContract(
        ("detected_intent",), ("sc_ctx", "pending_sub_fragments")
    ),
    "early_route": NodeContract(("sc_ctx",), ()),
    "prepare_context": NodeContract(
        ("detected_intent",), ("fetched", "memory_relevance")
    ),
    "special_route": NodeContract(("fetched", "sc_ctx"), ()),
    "prepare_reply": NodeContract(
        ("fetched",), ("reply_count", "_build_main_chat_messages")
    ),
    "generate_reply": NodeContract(
        ("_build_main_chat_messages",), ("replies", "reply_is_fallback")
    ),
    "normalize_reply": NodeContract(("replies",), ("emitted_replies",)),
    "persist_reply": NodeContract(
        ("emitted_replies",), ("first_assistant_message_id",)
    ),
}


def validate_fields(frame, phase, *, output=False):
    contract = CONTRACTS[phase]
    for name in contract.outputs if output else contract.inputs:
        if getattr(frame, name, None) is None:
            raise ChatNodeContractError(f"Chat phase {phase} omitted {name}")
