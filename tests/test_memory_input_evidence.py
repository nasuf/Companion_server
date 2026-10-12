"""Pure canonical import/edit snapshots and scoped receipt identity."""
import json
from unittest.mock import AsyncMock

import pytest

from app.services.memory.evidence import content_version
from app.services.memory import input_evidence as inputs


@pytest.mark.parametrize("kind", sorted(inputs.INPUT_KINDS))
def test_input_snapshot_is_canonical_frozen_and_not_a_truth_verdict(kind):
    payload = {"text": "私有合成输入", "nested": {"b": 2, "a": [1]}}
    origin = inputs.prepare_input_origin(kind=kind, payload=payload)
    reordered = {"nested": {"a": [1], "b": 2}, "text": "私有合成输入"}
    assert origin == inputs.prepare_input_origin(kind=kind, payload=reordered)
    payload["nested"]["a"].append(9)
    snapshot = json.loads(origin.payload)
    assert snapshot == {"format": inputs.INPUT_FORMAT, "kind": kind, "input": reordered}
    assert origin.version == content_version(origin.payload)
    assert "verified" not in snapshot


@pytest.mark.parametrize("kind,payload,error", [
    ("import", {}, "invalid_input_origin"),
    (None, {}, "invalid_input_origin"),
    ([], {}, "invalid_input_origin"),
    ("user_edit", [], "invalid_input_origin"),
    ("user_edit", {"value": float("nan")}, "invalid_input_payload"),
    ("admin_edit", {"value": object()}, "invalid_input_payload"),
    ("user_edit", {"value": "\ud800"}, "invalid_input_payload"),
])
def test_bad_input_cannot_form_a_receipt(kind, payload, error):
    with pytest.raises(ValueError, match=error):
        inputs.prepare_input_origin(kind=kind, payload=payload)


def test_input_cap_counts_utf8_bytes_including_envelope(monkeypatch):
    origin = inputs.prepare_input_origin(kind="knowledge_document", payload={"text": "合成" * 60})
    size = len(origin.payload.encode("utf-8"))
    monkeypatch.setattr(inputs, "INPUT_MAX_BYTES", size)
    assert inputs.prepare_input_origin(kind="knowledge_document", payload={"text": "合成" * 60}) == origin
    monkeypatch.setattr(inputs, "INPUT_MAX_BYTES", size - 1)
    with pytest.raises(ValueError, match="input_origin_too_large"):
        inputs.prepare_input_origin(kind="knowledge_document", payload={"text": "合成" * 60})


async def test_persist_identity_contains_scope_and_replay_keeps_hash():
    database = AsyncMock()
    origin = inputs.prepare_input_origin(kind="user_edit", payload={"after_content": "合成"})
    scopes = [("u", "w", "a"), ("u2", "w", "a"), ("u", "w2", "a"), ("u", "w", "a2")]
    refs = []
    for user_id, workspace_id, agent_id in scopes + [scopes[0]]:
        source = await inputs.persist_input_origin(database, user_id=user_id,
            workspace_id=workspace_id, agent_id=agent_id, origin=origin)
        refs.append(source.ref)
        assert source.kind == "input" and source.relation == "recorded_from"
        assert source.expected_version == origin.version and source.side is None
        sql, *args = database.execute_raw.call_args.args
        assert "ON CONFLICT (id) DO NOTHING" in sql
        assert args[1:] == [user_id, workspace_id, agent_id, origin.version,
                            "user_edit", inputs.INPUT_FORMAT, origin.payload]
    assert len(set(refs[:4])) == 4 and refs[0] == refs[4]
