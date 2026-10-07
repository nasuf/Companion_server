"""Retry identity and immutable capture contracts, without database writes."""
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import json

import pytest

from app.services.runtime.chat_ingress_contracts import (
    MAX_JSON_BYTES, ChatAggregationPolicy, ChatExecutionSnapshot,
    ChatRequestInput, PreparedChatMessage, canonical_object,
)


def test_original_body_identity_is_independent_of_json_key_order():
    a = ChatRequestInput.from_client(client_id="same", text="你好🙂",
                                   component_card={"payload": {"b": 2, "a": 1}, "type": "location"})
    b = ChatRequestInput.from_client(client_id="same", text="你好🙂",
                                   component_card={"type": "location", "payload": {"a": 1, "b": 2}})
    assert a == b and a.fingerprint == b.fingerprint
    assert a.request_key == "client:same"


def test_client_and_wechat_identity_namespaces_do_not_collide():
    a = ChatRequestInput.from_client(client_id="42", text="same")
    b = ChatRequestInput.from_wechat(message_id="42", text="same")
    assert a.fingerprint == b.fingerprint
    assert a.request_key != b.request_key
    assert b.client_id is None


def test_attachment_order_and_exact_text_are_part_of_retry_identity():
    build = lambda text, ids: ChatRequestInput.from_client(client_id="same", text=text, attachment_ids=ids)
    assert build("hi", ["a", "b"]).fingerprint != build("hi", ["b", "a"]).fingerprint
    assert build("hi", []).fingerprint != build("hi ", []).fingerprint


def test_captured_inputs_and_snapshots_have_no_mutable_aliases():
    card = {"type": "location", "payload": {"name": "original"}}
    request = ChatRequestInput.from_client(client_id="same", text="x", component_card=card)
    config = {"model": {"revision": 1}}
    snapshot = ChatExecutionSnapshot.capture(executor="langgraph", graph_version="g1", state_version=1,
                                             config=config, prompts={"p": "original"}, budget={"calls": 8})
    card["payload"]["name"] = "changed"
    config["model"]["revision"] = 2
    assert json.loads(request.input_json)["component_card"]["payload"]["name"] == "original"
    assert json.loads(snapshot.config_json)["model"]["revision"] == 1
    with pytest.raises(FrozenInstanceError):
        snapshot.graph_version = "changed"


@pytest.mark.parametrize("client_id", ["", " ", "a\n", "x" * 250, "\ud800", None, 1])
def test_invalid_client_identity_is_rejected_before_io(client_id):
    with pytest.raises(ValueError):
        ChatRequestInput.from_client(client_id=client_id, text="test")


@pytest.mark.parametrize("value", [{1: "value"}, {"nested": {1: "value"}}, {"x": float("nan")},
                                  {"x": float("inf")}, {"x": {1, 2}}, {"x": (1, 2)},
                                  {"x": "\ud800"}, {"x": "x" * MAX_JSON_BYTES}, []])
def test_invalid_json_is_rejected(value):
    with pytest.raises(ValueError):
        canonical_object(value)


@pytest.mark.parametrize("value", [{"x": "\x00"}, {"\x00": "x"}])
def test_postgres_unrepresentable_json_strings_are_rejected(value):
    with pytest.raises(ValueError):
        canonical_object(value)


def test_direct_provider_contract_requires_a_nonempty_message_identity():
    with pytest.raises(ValueError):
        ChatRequestInput("wechat:", "wechat", canonical_object({"text": "x",
            "attachment_ids": [], "component_card": None}))


@pytest.mark.parametrize("kwargs", [{"text": " "}, {"text": "x" * 32769},
                                   {"text": "x", "attachment_ids": ["a", "a"]},
                                   {"text": "x", "attachment_ids": ""},
                                   {"text": "x", "attachment_ids": ["a"] * 21},
                                   {"text": "x", "component_card": []}])
def test_invalid_message_input_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ChatRequestInput.from_client(client_id="a", **kwargs)


def test_card_or_attachment_input_can_have_an_empty_bubble():
    ChatRequestInput.from_client(client_id="a", text="", attachment_ids=["attachment"])
    ChatRequestInput.from_client(client_id="b", text="", component_card={"type": "location"})


@pytest.mark.parametrize("kwargs", [{"mode": "unknown"}, {"mode": "turn_window"},
                                   {"mode": "turn_window", "quiet_seconds": 1, "max_wait_seconds": .5},
                                   {"mode": "immediate", "quiet_seconds": 1},
                                   {"mode": "immediate", "delay_seconds": float("nan")},
                                   {"mode": "immediate", "delay_seconds": True},
                                   {"mode": "immediate", "delay_seconds": None},
                                   {"mode": "turn_window", "quiet_seconds": None},
                                   {"mode": "immediate", "allow_join": 1}])
def test_invalid_aggregation_policy_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ChatAggregationPolicy(**kwargs)


def test_prepared_metadata_and_context_are_captured_separately():
    context = {"received_at": "2026-10-07T00:00:00+00:00"}
    metadata = {"attachments": [{"id": "a"}]}
    prepared = PreparedChatMessage.capture(persisted_text="original", prompt_text="rendered",
        metadata=metadata, reply_context=context, received_at=datetime.now(timezone.utc))
    context.clear(); metadata["attachments"].clear()
    assert json.loads(prepared.metadata_json)["attachments"] == [{"id": "a"}]
    assert "received_at" in json.loads(prepared.reply_context_json)


def test_receipt_and_deadline_require_timezones():
    with pytest.raises(ValueError):
        PreparedChatMessage.capture(persisted_text="x", prompt_text="x", metadata={},
                                    reply_context={}, received_at=datetime.now())
    with pytest.raises(ValueError):
        ChatExecutionSnapshot.capture(executor="langgraph", graph_version="g1", state_version=1,
            config={}, prompts={}, budget={}, deadline_at=datetime.now())


@pytest.mark.parametrize("text", ["\x00", "\ud800"])
def test_prepared_text_must_be_representable_in_postgres(text):
    with pytest.raises(ValueError):
        PreparedChatMessage.capture(persisted_text=text, prompt_text="x", metadata={},
            reply_context={}, received_at=datetime.now(timezone.utc))


@pytest.mark.parametrize("field,value", [("state_version", True), ("state_version", 0),
                                       ("max_attempts", 0), ("max_attempts", True),
                                       ("executor", "untrusted"), ("graph_version", " ")])
def test_invalid_execution_snapshot_is_rejected(field, value):
    arguments = dict(executor="langgraph", graph_version="g1", state_version=1,
                     config={}, prompts={}, budget={})
    with pytest.raises(ValueError):
        ChatExecutionSnapshot.capture(**{**arguments, field: value})
