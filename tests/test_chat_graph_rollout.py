"""G03 must hold on incomplete, drifting or ambiguous production evidence."""

import copy
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.ops.chat_graph_rollout import Observation, assess, collect

NOW = datetime(2026, 10, 4, 3, tzinfo=timezone.utc)


@pytest.fixture
def evidence():
    observation = Observation(
        ("c-1",), "langgraph", "chat-g01-v1",
        NOW - timedelta(hours=26), NOW - timedelta(hours=1),
    )
    runtime = {
        "executor": "langgraph", "allowlist": "c-1", "graph_version": "chat-g01-v1",
        "durable_execution_ready": False, "trace_backend": "local",
    }
    roots, nodes, usage, messages = [], [], [], []
    for i in range(20):
        trace = f"t-{i}"
        started = observation.since + timedelta(minutes=i)
        roots.append({
            "trace_id": trace, "conversation_id": "c-1", "status": "success",
            "started_at": started, "ended_at": started + timedelta(seconds=1),
            "duration_ms": 1000., "executor": "langgraph", "graph_version": "chat-g01-v1",
            "checkpoint_enabled": "false",
            "usage_expected": "true",
        })
        nodes.append({"trace_id": trace, "graph_count": 1, "finish_count": 1,
                      "version_mismatches": 0, "llm_count": 4})
        usage.append({"trace_id": trace, "rows": 1, "cost_cny": .01, "call_count": 4,
                      "failure_count": 0, "fallback_count": 0})
        messages.append({"trace_id": trace, "reply_count": 2})
    return SimpleNamespace(observation=observation, runtime=runtime, roots=roots,
                           nodes=nodes, usage=usage, messages=messages)


def report(e):
    return assess(e.observation, e.runtime, e.roots, e.nodes, e.usage, e.messages, now=NOW)


def codes(e):
    return {v["code"] for v in report(e)["issues"]}


def test_complete_telemetry_is_not_full_release_qualification(evidence):
    result = report(evidence)
    assert result["telemetry_ready"]
    assert result["release_ready"] is False
    assert result["metrics"]["chat_model_calls"] == 80
    assert result["metrics"]["p95_root_duration_ms"] == 1000
    assert result["metrics"]["mean_chat_cost_cny"] == pytest.approx(.01)
    assert len(result["limitations"]) == 5


def test_full_rollout_observation_keeps_a_bounded_sample_without_cohort_false_alarm(evidence):
    o = evidence.observation
    evidence.observation = Observation(o.conversation_ids, o.executor, o.graph_version, o.since, o.until, True)
    evidence.runtime.update(all_conversations=True, allowlist="")
    result = report(evidence)
    assert result["telemetry_ready"]
    assert result["coverage"] == "all_conversations"
    assert result["release_ready"] is False
    evidence.runtime["all_conversations"] = False
    assert "coverage_drift" in codes(evidence)


def test_unexpected_full_rollout_or_mixed_modes_are_detected(evidence):
    evidence.runtime["all_conversations"] = True
    assert "coverage_drift" in codes(evidence)
    o = evidence.observation
    evidence.observation = Observation(o.conversation_ids, o.executor, o.graph_version, o.since, o.until, True)
    assert "cohort_drift" in codes(evidence)


def test_full_rollout_observation_cannot_claim_legacy(evidence):
    o = evidence.observation
    request = Observation(o.conversation_ids, "legacy", o.graph_version, o.since, o.until, True)
    with pytest.raises(ValueError):
        request.validate(NOW)


@pytest.mark.parametrize("field,value,code", [
    ("executor", "legacy", "executor_drift"),
    ("allowlist", "c-1,c-other", "cohort_drift"),
    ("allowlist", "", "cohort_drift"),
    ("graph_version", "chat-future", "version_drift"),
    ("durable_execution_ready", True, "durability_changed"),
    ("trace_backend", "off", "trace_backend"),
])
def test_runtime_drift_holds(evidence, field, value, code):
    evidence.runtime[field] = value
    assert code in codes(evidence)
    assert not report(evidence)["telemetry_ready"]


@pytest.mark.parametrize("field,value,code", [
    ("executor", "legacy", "observed_executor"),
    ("executor", None, "unmarked_root"),
    ("graph_version", "chat-old", "observed_version"),
    ("checkpoint_enabled", "true", "checkpoint_evidence"),
    ("checkpoint_enabled", None, "checkpoint_unrecorded"),
    ("status", "error", "root_failed"),
    ("status", "unknown", "root_failed"),
    ("ended_at", None, "root_failed"),
    ("duration_ms", -1, "duration_missing"),
    ("duration_ms", float("nan"), "duration_missing"),
])
def test_invalid_root_never_passes(evidence, field, value, code):
    evidence.roots[0][field] = value
    assert code in codes(evidence)


@pytest.mark.parametrize("field,value,code", [
    ("graph_count", 0, "graph_completion"),
    ("graph_count", 2, "graph_completion"),
    ("finish_count", 0, "graph_completion"),
    ("finish_count", 2, "graph_completion"),
    ("version_mismatches", 1, "node_version"),
])
def test_graph_evidence_is_complete_and_single_parent(evidence, field, value, code):
    evidence.nodes[0][field] = value
    assert code in codes(evidence)


def test_legacy_rollback_needs_positive_root_identity(evidence):
    evidence.observation = Observation(
        ("c-1",), "legacy", "chat-g01-v1",
        evidence.observation.since, evidence.observation.until,
    )
    evidence.runtime["executor"] = "legacy"
    for root in evidence.roots:
        root.update(executor="legacy", graph_version=None)
    evidence.nodes.clear()
    assert report(evidence)["telemetry_ready"]
    evidence.roots[0]["executor"] = None
    assert "unmarked_root" in codes(evidence)


def test_empty_window_and_missing_conversation_do_not_count_as_success(evidence):
    evidence.roots.clear()
    assert {"samples_short", "cohort_unobserved"} <= codes(evidence)


def test_short_observation_cannot_be_qualified_by_high_traffic(evidence):
    evidence.observation = Observation(
        ("c-1",), "langgraph", "chat-g01-v1",
        NOW - timedelta(hours=2), NOW - timedelta(hours=1),
    )
    assert "window_short" in codes(evidence)


def test_truncation_duplicate_scope_and_summary_hold(evidence):
    evidence.roots += [copy.deepcopy(evidence.roots[0])] * 1000
    evidence.roots[0]["conversation_id"] = "unauthorized"
    evidence.usage.append(copy.deepcopy(evidence.usage[0]))
    assert {"truncated", "scope_or_duplicate_root", "duplicate_summary"} <= codes(evidence)


@pytest.mark.parametrize("field,value,code", [
    ("rows", 2, "usage_missing_or_duplicate"),
    ("cost_cny", -1, "cost_invalid"),
    ("cost_cny", float("inf"), "cost_invalid"),
    ("call_count", -1, "usage_invalid"),
    ("call_count", None, "usage_invalid"),
    ("call_count", "4", "usage_invalid"),
    ("call_count", True, "usage_invalid"),
])
def test_invalid_usage_holds(evidence, field, value, code):
    evidence.usage[0][field] = value
    assert code in codes(evidence)


def test_missing_usage_and_reply_hold(evidence):
    evidence.usage.pop()
    evidence.messages[0]["reply_count"] = 0
    assert {"usage_missing_or_duplicate", "reply_missing"} <= codes(evidence)


def test_zero_model_turn_requires_session_and_trace_evidence(evidence):
    evidence.usage.pop()
    evidence.nodes[-1]["llm_count"] = 0
    assert "usage_missing_or_duplicate" in codes(evidence)
    evidence.roots[-1]["usage_expected"] = "false"
    result = report(evidence)
    assert result["telemetry_ready"]
    assert result["metrics"]["zero_model_turns"] == 1
    assert result["metrics"]["mean_chat_cost_cny"] == pytest.approx(.0095)
    evidence.nodes[-1]["llm_count"] = 1
    assert "usage_missing_or_duplicate" in codes(evidence)


def test_stale_and_recent_running_are_not_success(evidence):
    evidence.roots[0].update(status="running", ended_at=None, duration_ms=None)
    assert "stale_root" in codes(evidence)
    evidence.roots[0]["started_at"] = NOW - timedelta(minutes=1)
    assert "pending_root" in codes(evidence)


def test_inflight_turn_waits_for_evidence_without_false_failure(evidence):
    evidence.roots[-1].update(
        status="running", started_at=NOW - timedelta(minutes=1),
        ended_at=None, duration_ms=None,
    )
    evidence.nodes[-1]["finish_count"] = 0
    evidence.usage.pop()
    evidence.messages.pop()
    result = report(evidence)
    assert result["telemetry_ready"] is False
    assert result["issues"] == [{"code": "pending_root", "severity": "wait",
                                "detail": "A root has not completed"}]


def test_summary_never_contains_private_text_or_errors(evidence):
    evidence.roots[0].update(inputs_json={"message": "private-text"}, error="secret-url")
    encoded = json.dumps(report(evidence))
    assert "private-text" not in encoded and "secret-url" not in encoded
    assert "conversation_id" not in encoded


def add_auxiliary(evidence, *, scope=None, status="success", billed=True):
    root = {**evidence.roots[0], "trace_id": "auxiliary", "executor": None,
            "graph_version": None, "checkpoint_enabled": None,
            "usage_scope": scope, "usage_expected": billed, "status": status}
    evidence.roots.append(root)
    evidence.nodes.append({"trace_id": "auxiliary", "llm_count": int(billed),
                           "graph_count": 0, "finish_count": 0})
    if billed:
        evidence.usage.append({"trace_id": "auxiliary", "rows": 0,
                               "auxiliary_rows": 1, "auxiliary_scopes": ["proactive"]})
    return root


@pytest.mark.parametrize("scope", [None, "proactive"])
@pytest.mark.parametrize("status", ["success", "error", "running"])
def test_auxiliary_without_chat_reply_is_reported_separately(evidence, scope, status):
    add_auxiliary(evidence, scope=scope, status=status)
    result = report(evidence)
    assert result["telemetry_ready"]
    assert result["metrics"]["traced_turns"] == 20
    assert result["metrics"]["chat_model_calls"] == 80
    assert result["metrics"]["mean_chat_cost_cny"] == pytest.approx(.01)
    assert result["auxiliary_metrics"] == {
        "traced_operations": 1, "root_statuses": {status: 1},
        "usage_scopes": {"proactive": 1}, "expected_usage_missing": 0,
    }


def test_explicit_zero_model_auxiliary_needs_no_chat_reply_or_billing(evidence):
    add_auxiliary(evidence, scope="proactive", billed=False)
    assert report(evidence)["telemetry_ready"]
    assert report(evidence)["auxiliary_metrics"]["expected_usage_missing"] == 0
    evidence.roots[-1]["usage_expected"] = True
    assert report(evidence)["auxiliary_metrics"]["expected_usage_missing"] == 1


def test_auxiliary_only_window_does_not_qualify_chat_samples(evidence):
    add_auxiliary(evidence, scope="proactive")
    evidence.roots = [evidence.roots[-1]] * 20
    for i, root in enumerate(evidence.roots):
        evidence.roots[i] = {**root, "trace_id": f"aux-{i}"}
    result = report(evidence)
    assert result["metrics"]["traced_turns"] == 0
    assert {"samples_short", "cohort_unobserved"} <= codes(evidence)
    assert result["auxiliary_metrics"]["traced_operations"] == 20


def test_user_text_and_fingerprints_cannot_hide_an_unmarked_root(evidence):
    root = add_auxiliary(evidence, billed=False)
    root.update(inputs_json={"message": "[proactive:followup_unfinished]"},
                name="proactive.topic_completion_judge")
    assert {"unmarked_root", "reply_missing"} <= codes(evidence)
    assert report(evidence)["auxiliary_metrics"]["traced_operations"] == 0


@pytest.mark.parametrize("kind", ["executor", "graph", "finish", "billing"])
def test_declared_auxiliary_cannot_hide_chat_execution_or_lost_reply(evidence, kind):
    root = add_auxiliary(evidence, scope="proactive")
    if kind == "executor":
        root.update(executor="langgraph", graph_version="chat-g01-v1", checkpoint_enabled=False)
    elif kind == "graph":
        evidence.nodes[-1]["graph_count"] = 1
    elif kind == "finish":
        evidence.nodes[-1]["finish_count"] = 1
    else:
        evidence.usage[-1].update(rows=1, cost_cny=.01, call_count=1,
                                 failure_count=0, fallback_count=0)
    assert {"trace_scope_conflict", "reply_missing"} <= codes(evidence)
    assert not report(evidence)["telemetry_ready"]


def test_chat_root_with_background_usage_remains_a_chat_turn(evidence):
    evidence.usage[0].update(auxiliary_rows=1, auxiliary_scopes=["post_process"])
    result = report(evidence)
    assert result["telemetry_ready"]
    assert result["auxiliary_metrics"]["traced_operations"] == 0
    assert result["metrics"]["mean_chat_cost_cny"] == pytest.approx(.01)


def test_unmarked_chat_reply_with_background_billing_is_not_an_auxiliary_turn(evidence):
    add_auxiliary(evidence)
    evidence.messages.append({"trace_id": "auxiliary", "reply_count": 1, "chat_reply_count": 1})
    result = report(evidence)
    assert "unmarked_root" in codes(evidence)
    assert result["metrics"]["traced_turns"] == 21
    assert result["auxiliary_metrics"]["traced_operations"] == 0


@pytest.mark.parametrize("source", ["root", "billing"])
def test_unknown_scope_is_not_silently_excluded(evidence, source):
    root = add_auxiliary(evidence)
    if source == "root":
        root["usage_scope"] = "future-unknown"
        code = "unknown_usage_scope"
    else:
        evidence.usage[-1]["auxiliary_scopes"] = ["future-unknown"]
        code = "unknown_billing_scope"
    assert code in codes(evidence)
    assert "reply_missing" in codes(evidence)


def test_foreign_or_duplicate_auxiliary_cannot_bypass_scope_guard(evidence):
    root = add_auxiliary(evidence, scope="proactive")
    root["conversation_id"] = "unauthorized"
    evidence.roots.append(dict(root))
    assert "scope_or_duplicate_root" in codes(evidence)


@pytest.mark.parametrize("ids", [(), ("*",), ("c-1,c-2",), ("c-1", "c-1"),
                                tuple(f"c-{i}" for i in range(101))])
async def test_invalid_scope_rejected_before_database_access(evidence, ids):
    db = MagicMock()
    o = evidence.observation
    request = Observation(ids, o.executor, o.graph_version, o.since, o.until)
    with pytest.raises(ValueError):
        await collect(db, request, evidence.runtime, now=NOW)
    db.tx.assert_not_called()


@pytest.mark.parametrize("since,until", [
    (NOW.replace(tzinfo=None) - timedelta(days=1), NOW),
    (NOW, NOW + timedelta(seconds=1)),
    (NOW - timedelta(days=8), NOW),
    (NOW, NOW),
])
def test_window_validation(evidence, since, until):
    request = Observation(("c-1",), "langgraph", "chat-g01-v1", since, until)
    with pytest.raises(ValueError):
        request.validate(NOW)


async def test_collector_uses_read_only_snapshot_and_bound_scope(evidence):
    tx = SimpleNamespace(execute_raw=AsyncMock(),
                         query_raw=AsyncMock(side_effect=[
                             evidence.roots, evidence.nodes, evidence.usage, evidence.messages,
                         ]))
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=tx)
    context.__aexit__ = AsyncMock(return_value=False)
    db = SimpleNamespace(tx=MagicMock(return_value=context))
    result = await collect(db, evidence.observation, evidence.runtime, now=NOW)
    assert result["telemetry_ready"]
    assert tx.execute_raw.await_args_list[0].args == (
        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY",
    )
    assert "statement_timeout" in tx.execute_raw.await_args_list[1].args[0]
    assert tx.query_raw.await_args_list[0].args[1] == ["c-1"]
    for call in tx.query_raw.await_args_list:
        assert call.args[0].strip().startswith("SELECT")
        assert "c-1" not in call.args[0]  # IDs are parameters, never SQL interpolation.
