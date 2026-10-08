"""Transport, snapshot, SQL-policy and post-commit event contracts."""
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.config import settings
from app.services.runtime import chat_entrypoint as entry
from app.services.runtime import chat_ingress_activation as activation
from app.services.runtime.chat_ingress import AcceptedChatMessage, ChatRequestConflict
from app.services.runtime.chat_ingress_effects import ChatIngressQuotaBlocked, ChatIngressResourceInvalid
from app.services.runtime.execution_scope import ExecutionScopeUnavailable
from tests.test_chat_endpoint_security import secured_chat


@pytest.fixture
def receipt():
    return AcceptedChatMessage("message", "run", "job", 1, True, "client", "queued",
                               "pending", "collecting", datetime.now(timezone.utc)+timedelta(seconds=5))


def test_wire_input_preserves_original_body_and_shared_identity():
    payload = {"client_id":"id", "message":"  hi  ", "attachments":[{"id":"a"}, "b"],
               "component_card":{"type":"gift", "payload":{"offering_id":"gift"}}}
    captured = entry.parse_client_chat_input(payload)
    assert json.loads(captured.request.input_json)["text"] == "  hi  "
    assert json.loads(captured.request.input_json)["attachment_ids"] == ["a", "b"]
    payload["component_card"]["payload"]["offering_id"] = "changed"
    assert json.loads(captured.request.input_json)["component_card"]["payload"]["offering_id"] == "gift"
    assert captured.request.request_key == "client:id"


@pytest.mark.parametrize("change", [
    {"client_id":None}, {"client_id":""}, {"client_id":1}, {"client_id":" x"},
    {"message":False}, {"message":None}, {"message":""}, {"component_card":[]},
    {"attachments":"a"}, {"attachments":["a","a"]}, {"attachments":[None]},
    {"attachments":[{}]}, {"attachments":["a","b","c","d"]},
    *[{"paid_confirmed":value} for value in ("false","true",0,1,None,[],{})],
])
def test_malformed_input_is_rejected_before_sql_or_providers(change):
    with pytest.raises(ValueError):
        entry.parse_client_chat_input({"client_id":"id", "message":"synthetic", **change})


@pytest.mark.parametrize("payload", [None, [], "message", 5])
def test_non_object_envelope_rejected(payload):
    with pytest.raises(ValueError):entry.parse_client_chat_input(payload)


def test_payment_confirmation_is_outside_request_identity():
    body={"client_id":"id", "message":"synthetic"}
    assert entry.parse_client_chat_input(body).request == entry.parse_client_chat_input({**body,"paid_confirmed":True}).request


def test_sql_activation_cannot_be_enabled_by_environment_alone(monkeypatch):
    monkeypatch.setattr(settings,"chat_ingress_backend","redis")
    assert not activation.sql_chat_ingress_enabled()
    monkeypatch.setattr(settings,"chat_ingress_backend","sql")
    with pytest.raises(RuntimeError, match="consumer/delivery/handover"):
        activation.sql_chat_ingress_enabled()


def test_invalid_backend_fails_closed():
    with pytest.raises(RuntimeError):activation.validate_chat_ingress_backend("shadow")


def test_ack_only_after_receipt_and_never_contains_stored_provider_errors(receipt):
    result = entry.accepted_chat_events(replace(receipt,error_json='{"secret":"private"}'))
    assert [event["type"] for event in result] == ["ack","pending"]
    assert result[0]["data"]["message_id"]=="message"
    assert result[0]["data"]["defer_ui"] and result[1]["data"]["status"]=="aggregating"
    assert result[0]["data"]["ui_delay_seconds"]==0
    assert "private" not in json.dumps(result)


def test_ready_delayed_ack_uses_timer_without_deferring_until_processing(receipt):
    ack=entry.accepted_chat_events(replace(receipt,phase="ready"))[0]["data"]
    assert not ack["defer_ui"] and 0 < ack["ui_delay_seconds"] <= 5


@pytest.mark.parametrize("state",["succeeded","failed","cancelled"])
def test_terminal_retry_ack_has_current_state_without_restart_or_pending(receipt,state):
    events=entry.accepted_chat_events(replace(receipt,created=False,run_status=state))
    assert len(events)==1 and events[0]["data"]["duplicate"]
    assert events[0]["data"]["run_status"]==state


@pytest.mark.parametrize("failure,expected",[
    (ChatRequestConflict(),"request_conflict"),(ChatIngressResourceInvalid(),"invalid_resource"),
    (entry.ChatIngressInputInvalid("private body"),"invalid_request"),
    (ValueError("private wallet"),"storage_unavailable"),(RuntimeError("private SQL"),"storage_unavailable"),
])
async def test_socket_failure_has_no_ack_or_legacy_fallback(monkeypatch,failure,expected):
    monkeypatch.setattr(entry,"receive_client_chat",AsyncMock(side_effect=failure))
    ws=SimpleNamespace(send_json=AsyncMock(),close=AsyncMock())
    await entry.handle_sql_chat_frame(ws,"actor","conv",{"client_id":"id","message":"synthetic"})
    event=ws.send_json.await_args.args[0]
    assert event["type"]=="error" and event["data"]["code"]==expected
    assert "private" not in json.dumps(event)


async def test_socket_scope_failure_closes_without_ack(monkeypatch):
    monkeypatch.setattr(entry,"receive_client_chat",AsyncMock(side_effect=ExecutionScopeUnavailable()))
    ws=SimpleNamespace(send_json=AsyncMock(),close=AsyncMock())
    await entry.handle_sql_chat_frame(ws,"actor","conv",{"client_id":"id","message":"synthetic"})
    ws.close.assert_awaited_once_with(code=4403,reason="conversation_access_denied")
    ws.send_json.assert_not_called()


async def test_quota_blocked_socket_preserves_draft_identity(monkeypatch):
    monkeypatch.setattr(entry,"receive_client_chat",AsyncMock(side_effect=ChatIngressQuotaBlocked({
        "reason":"paid_confirm","per_msg_cost":0.5,"spendable_tickets":2})))
    ws=SimpleNamespace(send_json=AsyncMock())
    await entry.handle_sql_chat_frame(ws,"actor","conv",{"client_id":"id","message":"synthetic"})
    assert ws.send_json.await_args.args[0]=={"type":"quota_blocked","data":{
        "reason":"paid_confirm","per_msg_cost":0.5,"spendable_tickets":2,"client_id":"id"}}


async def test_socket_ack_transmission_loss_does_not_resubmit(monkeypatch,receipt):
    receiver=AsyncMock(return_value=receipt);monkeypatch.setattr(entry,"receive_client_chat",receiver)
    ws=SimpleNamespace(send_json=AsyncMock(side_effect=ConnectionError("gone")))
    with pytest.raises(ConnectionError):
        await entry.handle_sql_chat_frame(ws,"actor","conv",{"client_id":"id","message":"synthetic"})
    receiver.assert_awaited_once()


@pytest.mark.parametrize("text,offering,enabled,mode",[
    ("想",False,True,"fragment_window"),("好的",False,True,"turn_window"),
    ("今天挺开心",False,True,"turn_window"),("提醒我明天早上起床",False,True,"immediate"),
    ("gift",True,True,"immediate"),("想",False,False,"immediate"),
])
async def test_sql_planner_uses_domain_rules_without_mutating_redis(monkeypatch,text,offering,enabled,mode):
    from app.services.interaction import user_turn_aggregation as turn,chat_management
    monkeypatch.setattr(chat_management,"user_message_aggregation_enabled",lambda:enabled)
    monkeypatch.setattr(turn,"load_pending_action",AsyncMock(return_value=None))
    monkeypatch.setattr(turn,"load_pending_contradiction",AsyncMock(return_value=None))
    forbidden=[]
    for name in ("flush_pending","has_turn_pending","has_pending_delayed_messages","is_reply_inflight","push_pending","push_turn_pending"):
        call=AsyncMock(side_effect=AssertionError("SQL planner touched Redis queues"))
        monkeypatch.setattr(turn,name,call);forbidden.append(call)
    policy=await turn.plan_sql_user_message_aggregation(conversation_id="conv",text=text,
        reply_context={"delay_seconds":8},offering=offering)
    assert policy.mode==mode and policy.delay_seconds==(0 if offering else 8)
    for call in forbidden:call.assert_not_called()


def test_bound_snapshot_has_actual_prompt_content_and_no_env_credentials(monkeypatch):
    from app.services import runtime_config as config
    from app.services.prompting import store
    from app.services.runtime_config import ConfigurationSnapshot
    monkeypatch.setattr(settings,"chat_executor","langgraph")
    monkeypatch.setattr(settings,"chat_graph_all_conversations",True)
    monkeypatch.setattr(settings,"dashscope_api_key","must-not-be-exported")
    token=config._current_snapshot.set(ConfigurationSnapshot(None,{},
        {"synthetic/model":{"input":1.5,"output":2.5}},{}))
    prompt_token=store.bind_prompt_snapshot({"chat.system_base":("synthetic Web content",False)})
    try:
        snap=entry._bound_execution_snapshot("conv")
        assert snap.executor=="langgraph" and snap.graph_version=="chat-g01-v1"
        prompts=json.loads(snap.prompts_json)["templates"]
        assert prompts["chat.system_base"]["content"]=="synthetic Web content"
        assert not prompts["chat.system_base"]["enabled"]
        assert len(prompts)>=188 and "must-not-be-exported" not in snap.config_json
        assert json.loads(snap.config_json)["pricing"]["synthetic/model"]["input"]==1.5
        config._current_snapshot.get().pricing["synthetic/model"]["input"]=99
        assert json.loads(snap.config_json)["pricing"]["synthetic/model"]["input"]==1.5
    finally:
        store.reset_prompt_snapshot(prompt_token);config._current_snapshot.reset(token)


async def test_sql_preparation_requires_a_complete_configuration_view(monkeypatch):
    from app.services import runtime_config as config
    monkeypatch.setattr(config,"_CACHE_LOADED",False)
    monkeypatch.setattr(config,"load_caches",AsyncMock())
    before=config._current_snapshot.get()
    with pytest.raises(RuntimeError,match="complete database"):
        await config.bind_agent_context("agent",require_loaded=True)
    assert config._current_snapshot.get() is before


@pytest.mark.parametrize("failure,status,code",[
    (ChatRequestConflict(),409,"request_conflict"),
    (ChatIngressResourceInvalid(),422,"invalid_resource"),
    (entry.ChatIngressInputInvalid(),422,"invalid_request"),
    (ExecutionScopeUnavailable(),403,"conversation_access_denied"),
    (RuntimeError("private storage"),503,"storage_unavailable"),
    (ValueError("private ledger"),503,"storage_unavailable"),
])
def test_http_sql_failure_never_acks_or_enqueues_redis(secured_chat,monkeypatch,auth_header,failure,status,code):
    from app.api.public import chat
    monkeypatch.setattr(chat,"sql_chat_ingress_enabled",lambda:True)
    receiver=AsyncMock(side_effect=failure)
    monkeypatch.setattr(entry,"receive_client_chat",receiver)
    response=secured_chat.client.post('/chat/conv-1',json={"message":"synthetic","client_id":"id"},headers=auth_header("owner"))
    assert response.status_code==status and response.json()["detail"]["code"]==code
    assert "private" not in response.text and "event: ack" not in response.text
    secured_chat.db.message.create.assert_not_called()
    for call in secured_chat.business.values():call.assert_not_called()
    assert receiver.await_args.kwargs["actor_user_id"]=="owner"


@pytest.mark.parametrize("value",["false","true",0,1,None])
def test_http_rejects_non_boolean_confirmation_before_business(secured_chat,auth_header,value):
    response=secured_chat.client.post('/chat/conv-1',json={"message":"synthetic","paid_confirmed":value},headers=auth_header("owner"))
    assert response.status_code==422
    secured_chat.db.message.create.assert_not_called()
