"""One immutable initialization input/profile snapshot per scoped batch.

These are fictional character settings, not verified real-world experiences.
The snapshot contains invocation arguments, not an exact model/prompt receipt.
"""
from dataclasses import dataclass
import json

from app.services.memory.evidence import EvidenceSource, content_version

PROFILE_FORMAT = "persona-profile-v1"
PROFILE_EXTRACTOR = "persona-init-v1"
PROFILE_MAX_BYTES = 262144
PROFILE_KINDS = {"generated_profile", "imported_profile", "provided_profile"}


@dataclass(frozen=True)
class ProfileOrigin:
    payload: str
    version: str
    kind: str
    inputs_recorded: bool


def prepare_profile_origin(profile: dict, career: dict | None, *,
                           inputs: dict | None = None,
                           kind: str = "provided_profile") -> ProfileOrigin:
    """Freeze the actual conversion input before awaits or mutable processing."""
    if kind not in PROFILE_KINDS or not isinstance(profile, dict):
        raise ValueError("invalid_profile_origin")
    if career is not None and not isinstance(career, dict):
        raise ValueError("invalid_profile_career")
    if inputs is not None and not isinstance(inputs, dict):
        raise ValueError("invalid_profile_inputs")
    payload = json.dumps({"format": PROFILE_FORMAT, "kind": kind, "profile": profile,
                          "career": career, "invocation_inputs": inputs},
                         ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(payload.encode("utf-8")) > PROFILE_MAX_BYTES:
        raise ValueError("profile_origin_too_large")
    return ProfileOrigin(payload, content_version(payload), kind, inputs is not None)


async def persist_profile_origin(database, *, user_id: str, workspace_id: str,
                                 agent_id: str, origin: ProfileOrigin) -> EvidenceSource:
    # Scope is part of identity; identical profiles in two Agents never share
    # a private source object. Database guards also protect direct SQL writes.
    ref = content_version(json.dumps([user_id, workspace_id, agent_id, origin.version]))
    await database.execute_raw(
        """INSERT INTO memory_profile_origins
           (id,user_id,workspace_id,agent_id,source_version,source_status,input_status,format_version,payload_text)
           VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9) ON CONFLICT (id) DO NOTHING""",
        ref, user_id, workspace_id, agent_id, origin.version, origin.kind,
        "recorded" if origin.inputs_recorded else "uncollected", PROFILE_FORMAT, origin.payload)
    return EvidenceSource("profile", ref, relation="derived_from", expected_version=origin.version)
