"""Immutable scoped import/edit inputs, without asserting their factual truth.

Callers supply authenticated, server-validated inputs before asynchronous work.
The resulting snapshot is private; administration exposes only its kind/format.
"""
from dataclasses import dataclass
import json

from app.services.memory.evidence import EvidenceSource, content_version

INPUT_FORMAT = "memory-input-v1"
INPUT_MAX_BYTES = 8 * 1024 * 1024
INPUT_KINDS = frozenset({"knowledge_document", "admin_edit", "user_edit", "admin_repair", "contradiction_confirmation", "daily_summary"})


@dataclass(frozen=True)
class InputOrigin:
    payload: str
    version: str
    kind: str


def prepare_input_origin(*, kind: str, payload: dict) -> InputOrigin:
    """Freeze a canonical input snapshot; mutable caller data is never retained."""
    if not isinstance(kind, str) or kind not in INPUT_KINDS or not isinstance(payload, dict):
        raise ValueError("invalid_input_origin")
    try:
        serialized = json.dumps(
            {"format": INPUT_FORMAT, "kind": kind, "input": payload},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
        )
        size = len(serialized.encode("utf-8"))
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("invalid_input_payload") from exc
    if size > INPUT_MAX_BYTES:
        raise ValueError("input_origin_too_large")
    return InputOrigin(serialized, content_version(serialized), kind)


async def persist_input_origin(database, *, user_id: str, workspace_id: str,
                               agent_id: str, origin: InputOrigin) -> EvidenceSource:
    """Persist through the caller's transaction; same scoped input is replayable.

    Database guards independently enforce envelope/hash, scope and immutability.
    The caller must commit the memory/vector/audit/link with this same transaction.
    """
    ref = content_version(json.dumps([user_id, workspace_id, agent_id, origin.kind, origin.version]))
    await database.execute_raw(
        """INSERT INTO memory_input_origins
           (id,user_id,workspace_id,agent_id,source_version,source_status,format_version,payload_text)
           VALUES ($1,$2,$3,$4,$5,$6,$7,$8) ON CONFLICT (id) DO NOTHING""",
        ref, user_id, workspace_id, agent_id, origin.version, origin.kind, INPUT_FORMAT, origin.payload,
    )
    return EvidenceSource("input", ref, relation="recorded_from", expected_version=origin.version)
