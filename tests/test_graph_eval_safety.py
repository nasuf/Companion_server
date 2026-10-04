"""Live evaluation must be unable to contact business services or fake a PASS."""

import socket
from types import SimpleNamespace

import httpx
import pytest

from evals.graph_equivalence.safety import DeniedIO, model_network_fence, provider_origin


@pytest.mark.parametrize("url", ["http://model.test/v1", "https://u:p@model.test/v1", "https://model.test:8000/v1", "file:///x"])
def test_provider_origins_fail_closed(url):
    with pytest.raises(ValueError):
        provider_origin(url)


def test_business_io_is_not_a_permissive_mock():
    with pytest.raises(RuntimeError, match="forbidden business IO"):
        DeniedIO().message


def resolved(*args, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("203.0.113.1", 443))]


def test_socket_and_dns_fence(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", resolved)
    calls = []
    monkeypatch.setattr(socket.socket, "connect", lambda self, addr: calls.append(addr))
    with model_network_fence(["https://model.test/v1"]):
        with socket.socket() as connection:
            connection.connect(("203.0.113.1", 443))
            assert socket.getaddrinfo("203.0.113.1", 443)
            for address in (("203.0.113.1", 5432), ("127.0.0.1", 6379), ("203.0.113.2", 443)):
                with pytest.raises(RuntimeError, match="non-model socket"):
                    connection.connect(address)
        with pytest.raises(RuntimeError, match="non-model DNS"):
            socket.getaddrinfo("prod.test", 443)
    assert calls == [("203.0.113.1", 443)]


@pytest.mark.asyncio
async def test_http_redirect_to_business_host_blocked_even_on_shared_ip(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", resolved)
    seen = []
    async def handler(request):
        seen.append(request.url.host)
        return httpx.Response(302, headers={"location": "https://prod.test/admin"})
    with model_network_fence(["https://model.test/v1"]):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
            with pytest.raises(RuntimeError, match="non-model HTTP"):
                await client.get("https://model.test/v1")
            with pytest.raises(RuntimeError, match="non-model HTTP"):
                await client.get("https://prod.test/admin")
    assert seen == ["model.test"]


def test_sync_redirect_is_also_fenced(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", resolved)
    with model_network_fence(["https://model.test/v1"]):
        with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(302, headers={"location": "https://prod.test"})), follow_redirects=True) as client:
            with pytest.raises(RuntimeError, match="non-model HTTP"):
                client.get("https://model.test/v1")


def test_partial_eval_and_unknown_prices_cannot_qualify():
    from evals.graph_equivalence.run_eval import report_summary
    case = SimpleNamespace(id="only", group="chitchat")
    result = report_summary([], [case], 5)
    assert not result["passed"] and not result["complete"]
    # Reusing a row, an unknown case or fewer samples cannot fill the bank.
    # Minimal rows deliberately use an error so no quality grade is invented.
    rows = [{"case": "wrong", "sample": i, "executor": executor, "error": "RuntimeError"}
            for i in range(5) for executor in ("legacy", "langgraph")]
    result = report_summary(rows, [case], 5)
    assert not result["complete"] and not result["passed"]


def valid_rows(samples=5):
    return [{"case": "only", "group": "chitchat", "sample": i, "executor": executor,
             "message": "今天好累", "reply": "歇歇吧", "verdict": "natural", "persona_leak": False,
             "bubbles": 1, "total_chars": 3, "max_bubble_chars": 3, "emoji_count": 0,
             "format_ok": True, "latency_ms": 100, "normalized_cost_cny": 0.01,
             "usage": {"call_count": 2}, "prices_known": True,
             "requests": [{"phase": "response", "model": "synthetic", "input_hash": "same"}],
             "case_precondition": {"case": "only", "stage": "model_input", "passed": True, "requirements": [], "issues": []}}
            for i in range(samples) for executor in ("legacy", "langgraph")]


@pytest.mark.parametrize("failure", ["unknown_price", "persona_leak", "judge_failure", "latency", "cost", "calls", "input", "duplicate", "missing", "too_few_samples"])
def test_complete_eval_gates_cannot_be_bypassed(failure):
    from evals.graph_equivalence.run_eval import report_summary
    rows = valid_rows(1 if failure == "too_few_samples" else 5)
    row = next(r for r in rows if r["executor"] == "langgraph")
    if failure == "unknown_price":
        row["prices_known"] = False
    elif failure == "persona_leak":
        row["persona_leak"] = True
    elif failure == "judge_failure":
        # Existing quality summarisation excludes unparsed verdicts. A partly
        # judged bank must not gain qualification from selective denominators.
        row["verdict"] = None
    elif failure == "latency":
        row["latency_ms"] = 1000
    elif failure == "cost":
        row["normalized_cost_cny"] = 0.1
    elif failure == "calls":
        row["usage"] = {"call_count": 20}
    elif failure == "input":
        row["requests"] = [{"phase": "response", "model": "synthetic", "input_hash": "changed"}]
    elif failure == "duplicate":
        rows[-1] = dict(rows[0])
    elif failure == "missing":
        rows.pop()
    result = report_summary(rows, [SimpleNamespace(id="only", group="chitchat")], 1 if failure == "too_few_samples" else 5)
    assert not result["passed"]


def test_passing_pair_ignores_only_stochastic_emotion_input():
    from evals.graph_equivalence.run_eval import report_summary
    rows = valid_rows()
    for index, row in enumerate(rows):
        row["requests"].append({"phase": "reply_emotion", "model": "synthetic", "input_hash": str(index)})
    assert report_summary(rows, [SimpleNamespace(id="only", group="chitchat")], 5)["passed"]


def test_call_count_gate_includes_cancelled_or_unreported_model_starts():
    from evals.graph_equivalence.run_eval import report_summary
    rows = valid_rows()
    for row in rows:
        row["model_attempts"] = 5 if row["executor"] == "langgraph" else 3
    summary = report_summary(rows, [SimpleNamespace(id="only", group="chitchat")], 5)
    assert summary["performance"]["langgraph"]["mean_calls"] == 5
    assert not summary["performance_passed"] and not summary["passed"]


def test_retry_input_dedup_preserves_actual_attempt_evidence():
    from evals.graph_equivalence.run_eval import logical_inputs
    inputs = [{"phase": "response", "model": "synthetic", "input_hash": "same"}] * 2
    assert len(logical_inputs(inputs)) == 1
    assert len(inputs) == 2


def test_provider_dns_is_pinned_without_new_resolution(monkeypatch):
    count = 0
    def changing_dns(*args, **kwargs):
        nonlocal count
        count += 1
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (f"203.0.113.{count}", 443))]
    monkeypatch.setattr(socket, "getaddrinfo", changing_dns)
    with model_network_fence(["https://model.test/v1"]):
        assert socket.getaddrinfo("model.test", 443)[0][4] == ("203.0.113.1", 443)
        assert socket.getaddrinfo("model.test", 443)[0][4] == ("203.0.113.1", 443)
    assert count == 1


def test_recovered_fence_error_cannot_qualify():
    from evals.graph_equivalence.run_eval import report_summary
    rows = valid_rows()
    rows[0]["model_errors"] = [{"type": "ExceptionGroup", "children": [{"fence": "Evaluation blocked a non-model socket"}]}]
    assert not report_summary(rows, [SimpleNamespace(id="only", group="chitchat")], 5)["passed"]


def test_non_model_violation_is_recorded_even_when_business_code_catches_it(monkeypatch):
    from evals.graph_equivalence.run_eval import report_summary
    monkeypatch.setattr(socket, "getaddrinfo", resolved)
    with model_network_fence(["https://model.test/v1"]) as fence:
        try:
            socket.getaddrinfo("prod.test", 6379)
        except RuntimeError:
            pass
        assert fence.violations == ["Evaluation blocked a non-model DNS request"]
    rows = valid_rows()
    rows[0]["network_fence_hits"] = len(fence.violations)
    assert not report_summary(rows, [SimpleNamespace(id="only", group="chitchat")], 5)["passed"]


def test_parallel_read_interleaving_does_not_hide_within_role_input_changes():
    from evals.graph_equivalence.run_eval import logical_inputs
    intent = {"phase": "intent", "model": "synthetic", "input_hash": "i"}
    relevance = {"phase": "memory_relevance", "model": "synthetic", "input_hash": "r"}
    reply = {"phase": "response", "model": "synthetic", "input_hash": "reply"}
    assert logical_inputs([intent, relevance, reply]) == logical_inputs([relevance, intent, reply])
    changed_reply = {**reply, "input_hash": "other"}
    assert logical_inputs([intent, reply, changed_reply]) != logical_inputs([intent, changed_reply, reply])


def test_input_difference_diagnostics_do_not_relax_qualification():
    from evals.graph_equivalence.run_eval import report_summary
    rows = valid_rows()
    rows[1]["requests"][0]["input_hash"] = "changed"
    result = report_summary(rows, [SimpleNamespace(id="only", group="chitchat")], 5)
    assert result["model_input_differences"] == [{"case": "only", "sample": 0, "phases": ["response"]}]
    assert not result["non_emotion_model_input_sequence_equal"]
    assert not result["passed"]


def test_classifier_pair_copies_source_and_consumed_decisions():
    from app.services.chat.intent_dispatcher import IntentResult, IntentType
    from evals.graph_equivalence.run_eval import ClassifierPair
    pair = ClassifierPair()
    pair.begin(recording=True)
    original = IntentResult(intent=IntentType.NONE, metadata={"fragments": ["original"]})
    consumed = pair.choose("intent", {"message": "test"}, original)
    consumed.metadata["fragments"].append("source mutated")
    original.metadata["fragments"].clear()
    pair.finish()
    pair.begin(recording=False)
    independent = IntentResult(intent=IntentType.DELETION)
    replay = pair.choose("intent", {"message": "test"}, independent)
    assert replay.intent == IntentType.NONE
    assert replay.metadata == {"fragments": ["original"]}
    replay.metadata["fragments"].clear()
    pair.finish()
    assert pair.recorded["intent"][0][1].metadata == {"fragments": ["original"]}


@pytest.mark.parametrize("failure", ["extra", "changed_input", "missing"])
def test_classifier_pair_rejects_differences_even_when_caught(failure):
    from evals.graph_equivalence.run_eval import ClassifierPair, report_summary
    pair = ClassifierPair()
    pair.begin(recording=True)
    pair.choose("intent", {"message": "expected"}, "source")
    pair.begin(recording=False)
    try:
        if failure == "extra":
            pair.choose("other", {"message": "expected"}, "independent")
        elif failure == "changed_input":
            pair.choose("intent", {"message": "changed"}, "independent")
    except RuntimeError:
        pass
    with pytest.raises(RuntimeError, match="Classifier pair contract failed"):
        pair.finish()
    rows = valid_rows()
    rows[1]["classifier_pair_valid"] = False
    assert not report_summary(rows, [SimpleNamespace(id="only", group="chitchat")], 5)["passed"]


def test_classifier_pair_allows_parallel_roles_but_preserves_call_order():
    from evals.graph_equivalence.run_eval import ClassifierPair
    pair = ClassifierPair()
    pair.begin(recording=True)
    pair.choose("intent", {"message": "first"}, 1)
    pair.choose("memory_relevance", {"message": "first"}, 2)
    pair.choose("intent", {"message": "second"}, 3)
    pair.begin(recording=False)
    assert pair.choose("memory_relevance", {"message": "first"}, 99) == 2
    assert pair.choose("intent", {"message": "first"}, 99) == 1
    assert pair.choose("intent", {"message": "second"}, 99) == 3
    pair.finish()


@pytest.mark.parametrize("source_cancelled,second_cancelled", [(True, False), (False, True), (True, True)])
def test_unused_relevance_completion_race_preserves_call_contract(source_cancelled, second_cancelled):
    from evals.graph_equivalence.run_eval import ClassifierPair
    pair = ClassifierPair()
    inputs = {"message": "ending turn"}
    pair.begin(recording=True)
    if source_cancelled:
        pair.cancelled("memory_relevance", inputs)
    else:
        pair.choose("memory_relevance", inputs, "source decision")
    pair.finish(reply_uses_context=False)
    pair.begin(recording=False)
    if second_cancelled:
        pair.cancelled("memory_relevance", inputs)
    else:
        actual = pair.choose("memory_relevance", inputs, "independent decision")
        assert actual == ("independent decision" if source_cancelled else "source decision")
    pair.finish(reply_uses_context=False)
    assert not pair.violations
    assert bool(pair.independent_unused_results) == (source_cancelled and not second_cancelled)


@pytest.mark.parametrize("failure", ["missing", "extra", "changed_input", "consumed", "intent"])
def test_cancelled_relevance_never_waives_input_call_or_consumption_guards(failure):
    from evals.graph_equivalence.run_eval import ClassifierPair
    pair = ClassifierPair()
    pair.begin(recording=True)
    pair.cancelled("memory_relevance", {"message": "expected"})
    pair.finish(reply_uses_context=False)
    pair.begin(recording=False)
    with pytest.raises(RuntimeError, match="Classifier pair contract failed"):
        if failure == "missing":
            pair.finish(reply_uses_context=False)
        elif failure == "changed_input":
            pair.cancelled("memory_relevance", {"message": "changed"})
        elif failure == "intent":
            pair.cancelled("intent", {"message": "expected"})
        else:
            pair.choose("memory_relevance", {"message": "expected"}, "independent decision")
            if failure == "extra":
                pair.choose("memory_relevance", {"message": "expected"}, "extra decision")
            else:
                pair.finish(reply_uses_context=True)
    assert pair.violations


def test_cancelled_counterpart_cannot_hide_context_used_by_source_reply():
    from evals.graph_equivalence.run_eval import ClassifierPair
    pair = ClassifierPair()
    pair.begin(recording=True)
    pair.choose("memory_relevance", {"message": "test"}, "weak")
    pair.finish(reply_uses_context=True)
    pair.begin(recording=False)
    pair.cancelled("memory_relevance", {"message": "test"})
    with pytest.raises(RuntimeError):
        pair.finish(reply_uses_context=False)


@pytest.mark.asyncio
async def test_live_relevance_cancellation_is_recorded_and_propagated(monkeypatch):
    import asyncio
    from unittest.mock import AsyncMock
    from app.services.chat import data_fetch_phase
    from evals.graph_equivalence.run_eval import ClassifierPair, Patches, REQUEST_PHASE, configure_live_context
    started = asyncio.Event()
    async def pending(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()
    monkeypatch.setattr(data_fetch_phase, "classify_memory_relevance", pending)
    pair = ClassifierPair()
    pair.begin(recording=True)
    decisions = []
    with Patches() as patches:
        configure_live_context(patches, decisions, pair)
        task = asyncio.create_task(data_fetch_phase._classify_relevance("ending turn", context=""))
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    pair.finish(reply_uses_context=False)
    assert pair.cancellations and decisions == [{"phase": "memory_relevance_cancelled"}]
    assert REQUEST_PHASE.get() == "response"


def test_paired_classifier_inputs_normalize_call_syntax():
    from evals.graph_equivalence.run_eval import ClassifierPair, paired_decision
    def classify(message, context=""):
        pass
    pair = ClassifierPair()
    pair.begin(recording=True)
    assert paired_decision(pair, "intent", classify, ("hello",), {}, "source") == "source"
    pair.begin(recording=False)
    assert paired_decision(pair, "intent", classify, (), {"message": "hello", "context": ""}, "other") == "source"
    pair.finish()


@pytest.mark.asyncio
async def test_paired_relevance_invokes_real_classifier_on_both_sides(monkeypatch):
    from unittest.mock import AsyncMock
    from app.services.chat import data_fetch_phase
    from app.services.memory.retrieval.relevance import RelevanceResult
    from evals.graph_equivalence.run_eval import ClassifierPair, Patches, configure_live_context
    classifier = AsyncMock(side_effect=[RelevanceResult(level="medium", enhanced_query=""),
                                        RelevanceResult(level="strong", enhanced_query="independent")])
    monkeypatch.setattr(data_fetch_phase, "classify_memory_relevance", classifier)
    pair = ClassifierPair()
    for recording in (True, False):
        pair.begin(recording=recording)
        decisions = []
        with Patches() as patches:
            configure_live_context(patches, decisions, pair)
            result = await data_fetch_phase.fetch_parallel_context(
                user_id="synthetic-user", agent_id="synthetic-agent", workspace_id="synthetic-workspace",
                user_message="我今天很难过", messages_dicts=[], parsed_times=[],
            )
            assert result.memory_relevance == "medium"
            assert decisions[-1]["phase"] == "memory_relevance_consumed"
            assert decisions[0]["level"] == ("medium" if recording else "strong")
            pair.finish()
    assert classifier.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("text,expected_calls,expected_level", [("嗯", 0, "weak"), ("我今天很难过", 1, "medium")])
async def test_live_context_retains_production_fast_gate_and_relevance_classifier(monkeypatch, text, expected_calls, expected_level):
    from unittest.mock import AsyncMock
    from app.services.chat import data_fetch_phase
    from app.services.memory.retrieval.relevance import RelevanceResult
    from evals.graph_equivalence.run_eval import Patches, configure_live_context
    classifier = AsyncMock(return_value=RelevanceResult(level="medium", enhanced_query=""))
    monkeypatch.setattr(data_fetch_phase, "classify_memory_relevance", classifier)
    with Patches() as patches:
        decisions = []
        configure_live_context(patches, decisions)
        result = await data_fetch_phase.fetch_parallel_context(
            user_id="synthetic-user", agent_id="synthetic-agent", workspace_id="synthetic-workspace",
            user_message=text, messages_dicts=[], parsed_times=[],
        )
        assert result.memory_relevance == expected_level
        assert result.classified_memories is None
        assert classifier.await_count == expected_calls
        assert decisions == ([{"phase": "memory_relevance", "level": "medium", "enhanced_query": ""}]
                             if expected_calls else [])
