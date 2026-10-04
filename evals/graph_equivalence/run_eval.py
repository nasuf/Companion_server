"""Paired real-model G02 evaluation with synthetic domain IO.

    python -m evals.graph_equivalence.run_eval --read-only-production-snapshot \
        --output /tmp/g02.json

Run in a disposable process, never import into an application worker. The only
production DB access is an explicit read-only transaction loading configuration,
prompt templates and model prices; no conversation/user/agent rows are read.
After disconnect, business DB and Redis access are denied, notifications and
background actions are closed, and network access is fenced to model providers.
Credentials remain in that process; reports include hashes, never prompt text or
credentials. Real-model and judge calls are opt-in and consume provider quota.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import re
import traceback
from contextlib import ExitStack
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from importlib import import_module
from importlib.metadata import version
from inspect import signature
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain_core.callbacks import BaseCallbackHandler

from app.config import settings
from app.services import runtime_config as rc
from app.services.chat import data_fetch_phase, intent_dispatcher, intent_replies, orchestrator as chat, reply_generate
from app.services.llm import models, usage_repo
from app.services.llm.pricing import estimate_cost_cny, is_known_model
from app.services.prompting import store
from evals.graph_equivalence.safety import DeniedIO, model_network_fence
from evals.graph_equivalence.preconditions import audit_bank, audit_case, fixture_facts, row_precondition_valid
from evals.graph_equivalence.scenarios import SCENARIO_VERSION, history_for, witness_for
from evals.reply_register import judge as J
from evals.reply_register.cases import ALL_CASES
from evals.reply_register.run_eval import run_calibration, summarise
from evals.reply_register.standard import SAMPLES_PER_CASE
from tests.g02_harness_support import configure_pair, synthetic_agent

MAX_REGRESSION = 0.10
EVALUATION_FILES = (
    "evals/graph_equivalence/run_eval.py", "evals/graph_equivalence/safety.py", "evals/graph_equivalence/preconditions.py",
    "evals/graph_equivalence/scenarios.py", "evals/reply_register/run_eval.py",
    "evals/reply_register/cases.py", "evals/reply_register/judge.py", "evals/reply_register/standard.py",
    "tests/g02_harness_support.py", "tests/graph_harness_support.py",
)
REQUEST_PHASE = ContextVar("g02_request_phase", default="response")
_CANCELLED_CLASSIFIER = object()


class Patches(ExitStack):
    def setattr(self, target, name, value):
        self.enter_context(patch.object(target, name, value))


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


class Requests(BaseCallbackHandler):
    def __init__(self):
        self.rows = []
        self.errors = []
        self.decisions = []
        self.response_inputs = []
        self.trusted_references = []
        self.judge_prompt = None

    def on_chat_model_start(self, serialized, messages, **kwargs):
        payload = [[{"type": m.type, "content": m.content} for m in group] for group in messages]
        if REQUEST_PHASE.get() == "response":
            self.response_inputs.extend([[str(m.content) for m in group] for group in messages])
        self.rows.append({"phase": REQUEST_PHASE.get(), "input_hash": digest(payload), "model":
                          (kwargs.get("invocation_params") or {}).get("model_name") or
                          (kwargs.get("invocation_params") or {}).get("model")})

    def on_llm_error(self, error, **_):
        def safe_kind(exc):
            if isinstance(exc, BaseExceptionGroup):
                return {"type": type(exc).__name__, "children": [safe_kind(child) for child in exc.exceptions]}
            if isinstance(exc, RuntimeError) and str(exc).startswith("Evaluation blocked"):
                return {"type": type(exc).__name__, "fence": str(exc)}
            return {"type": type(exc).__name__}
        self.errors.append(safe_kind(error))


class SampleRejected(RuntimeError):
    def __init__(self, usage, requests, errors):
        self.usage = usage
        self.requests = requests
        self.model_errors = errors


class ClassifierPair:
    """Common parsed decisions, with real classifier calls on both sides.

    Each pair records the first executor's decisions. The second executor still
    invokes the real classifier and records its independent result, but consumes
    a copy of the first result. This controls classifier randomness without
    skipping model calls, hiding extra calls or mutating a production model.
    """

    def __init__(self):
        self.recorded = {}
        self.positions = {}
        self.violations = []
        self.recording = True
        self.cancellations = []
        self.source_context_used = None
        self.independent_unused_results = []

    def begin(self, *, recording):
        self.recording = recording
        self.positions = {}
        self.cancellations = []
        self.independent_unused_results = []
        if recording:
            self.recorded = {}
            self.violations = []
            self.source_context_used = None

    def choose(self, phase, inputs, actual):
        fingerprint = digest(inputs)
        if self.recording:
            value = actual if actual is _CANCELLED_CLASSIFIER else deepcopy(actual)
            self.recorded.setdefault(phase, []).append((fingerprint, value))
            return None if actual is _CANCELLED_CLASSIFIER else deepcopy(actual)
        index = self.positions.get(phase, 0)
        entries = self.recorded.get(phase, [])
        if index >= len(entries) or entries[index][0] != fingerprint:
            self.violations.append("Unexpected classifier input or call")
            raise RuntimeError("Classifier pair contract failed")
        self.positions[phase] = index + 1
        if actual is _CANCELLED_CLASSIFIER:
            return None
        if entries[index][1] is _CANCELLED_CLASSIFIER:
            # Only an unused speculative relevance read may lack a source
            # result. finish() rejects any response path that consumes context;
            # the full report still requires equal actual model request inputs.
            self.independent_unused_results.append({"phase": phase, "input_hash": fingerprint})
            return deepcopy(actual)
        return deepcopy(entries[index][1])

    def cancelled(self, phase, inputs):
        if phase != "memory_relevance":
            self.violations.append("Required classifier cancelled")
            raise RuntimeError("Classifier pair contract failed")
        self.cancellations.append({"phase": phase, "input_hash": digest(inputs)})
        self.choose(phase, inputs, _CANCELLED_CLASSIFIER)

    def finish(self, *, reply_uses_context=True):
        if self.recording:
            self.source_context_used = reply_uses_context
        cancelled_source = any(value is _CANCELLED_CLASSIFIER for entries in self.recorded.values()
                               for _, value in entries)
        context_used = reply_uses_context or (not self.recording and self.source_context_used is not False)
        if context_used and (self.cancellations or cancelled_source):
            self.violations.append("Cancelled classifier needed by response")
        if not self.recording and any(self.positions.get(phase, 0) != len(entries)
                                      for phase, entries in self.recorded.items()):
            self.violations.append("Missing classifier call")
        if self.violations:
            raise RuntimeError("Classifier pair contract failed")


def paired_decision(pair, phase, classify, args, kwargs, actual):
    if pair is None:
        return actual
    bound = signature(classify).bind(*args, **kwargs)
    bound.apply_defaults()
    return pair.choose(phase, bound.arguments, actual)


async def read_snapshot():
    from app.db import db
    await db.connect()
    try:
        async with db.tx() as tx:
            await tx.execute_raw("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            system = await tx.systemconfig.find_unique(where={"id": 1})
            overrides = await tx.agentconfigoverride.find_many()
            prompts = await tx.prompttemplate.find_many()
            prices = await tx.modelregistry.find_many()
    finally:
        await db.disconnect()
    if system is None:
        raise RuntimeError("System config missing; evaluation cannot seed production")
    # The hot path normally reads Redis first. A stale cache would make a DB-only
    # snapshot score different prompts than users currently receive. Read only;
    # never refill, delete or repair production cache during an evaluation.
    from app.redis_client import close_redis, get_redis
    try:
        redis = await get_redis()
        cached_texts = await redis.mget([store.PROMPT_KEY_PREFIX + row.key for row in prompts])
        cached_enabled = await redis.mget([store.PROMPT_ENABLED_KEY_PREFIX + row.key for row in prompts])
        for row, text, enabled in zip(prompts, cached_texts, cached_enabled, strict=True):
            if text and text != row.content:
                raise RuntimeError("Production prompt cache differs from DB snapshot")
            if enabled is not None and (str(enabled) != "0") != row.isEnabled:
                raise RuntimeError("Production prompt enabled cache differs from DB snapshot")
    finally:
        await close_redis()
    rc._GLOBAL_CACHE = rc._row_to_dict(system)
    rc._AGENT_CACHE = {row.agentId: rc._row_to_dict(row) for row in overrides}
    rc._CACHE_LOADED = True
    rc._PRICING_CACHE = {
        f"{row.provider}/{row.identifier}": {
            "input": row.inputCostPerMillion or 0.0,
            "output": row.outputCostPerMillion or 0.0,
            "cached_input": row.cachedInputCostPerMillion if row.cachedInputCostPerMillion is not None else row.inputCostPerMillion or 0.0,
        } for row in prices if getattr(row, "modelKind", "llm") == "llm"
    }
    # Qualification here covers the global model route, never arbitrary users.
    # Distinct per-agent routes need their own isolated run before full rollout.
    if overrides:
        raise RuntimeError("Per-agent overrides require separate synthetic route qualification")
    return {row.key: row for row in prompts}


def percentile(values, fraction):
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def prospective_prompt_snapshot(snapshot):
    """Emulate this candidate's code_sync entirely in the evaluation process.

    Never update production DB/cache. Unreviewed keys or customized rows fail
    closed; Web policy overrides retain their exact production content.
    """
    from app.services.prompting.registry import PROMPT_DEFINITION_MAP
    allowed = {f"memory.{kind}_reply" for kind in ("weak", "medium", "strong", "l3")}
    if not allowed <= snapshot.keys():
        raise RuntimeError("Candidate tier prompt row missing")
    effective, changes = {}, []
    for key, row in snapshot.items():
        definition = PROMPT_DEFINITION_MAP.get(key)
        if definition and row.defaultContent != definition.default_text:
            if key not in allowed or row.content != row.defaultContent:
                raise RuntimeError("Candidate code_sync would change an unreviewed or customized prompt")
            replacement = deepcopy(row)
            replacement.content = replacement.defaultContent = definition.default_text
            effective[key] = replacement
            changes.append({"key": key, "original_content_hash": digest(row.content),
                            "candidate_content_hash": digest(replacement.content),
                            "expected_updated_at": row.updatedAt.isoformat(), "customized": False})
        else:
            effective[key] = row
    return effective, changes


def report_summary(rows, expected_cases, samples):
    result = {executor: summarise([r for r in rows if r["executor"] == executor])
              for executor in ("legacy", "langgraph")}
    successful = [r for r in rows if "error" not in r]
    paired = {}
    for row in successful:
        paired.setdefault((row["case"], row["sample"]), {})[row["executor"]] = row
    expected_pairs = {(case.id, sample) for case in expected_cases for sample in range(samples)}
    complete = len(rows) == len(expected_cases) * samples * 2 and all(
        set(pair) == {"legacy", "langgraph"} for pair in paired.values()
    ) and set(paired) == expected_pairs
    inputs_equal = bool(paired) and complete and all(
        logical_inputs(pair["legacy"]["requests"]) == logical_inputs(pair["langgraph"]["requests"])
        for pair in paired.values()
    )
    input_differences = []
    for (case_id, sample), pair in paired.items():
        if set(pair) != {"legacy", "langgraph"}:
            continue
        left, right = (logical_inputs(pair[e]["requests"]) for e in ("legacy", "langgraph"))
        if left != right:
            phases = sorted({r["phase"] for r in left + right})
            input_differences.append({"case": case_id, "sample": sample, "phases": [
                phase for phase in phases if [r for r in left if r["phase"] == phase] !=
                [r for r in right if r["phase"] == phase]
            ]})
    # Cache billing is order-sensitive. Keep billed estimates and use the same
    # uncached-input price for both sides as the architecture cost gate.
    performance = {}
    for executor in ("legacy", "langgraph"):
        g = [r for r in successful if r["executor"] == executor]
        performance[executor] = {
            "p95_ms": percentile([r["latency_ms"] for r in g], 0.95),
            "mean_cost_cny": sum(r["normalized_cost_cny"] for r in g) / len(g) if g else None,
            "mean_calls": sum(r.get("model_attempts", r["usage"]["call_count"]) for r in g) / len(g) if g else None,
            "mean_reported_usage_calls": sum(r["usage"]["call_count"] for r in g) / len(g) if g else None,
            "cancelled_relevance_reads": sum(len(r.get("classifier_cancellations", [])) for r in g),
            "prices_known": bool(g) and all(r["prices_known"] for r in g),
        }
    base, graph = performance["legacy"], performance["langgraph"]
    perf_ok = complete and all(base[k] is not None and base[k] > 0 and graph[k] is not None
                              and graph[k] <= base[k] * (1 + MAX_REGRESSION)
                              for k in ("p95_ms", "mean_cost_cny", "mean_calls"))
    perf_ok = perf_ok and base["prices_known"] and graph["prices_known"]
    persona_ok = bool(successful) and all(not r["persona_leak"] for r in successful)
    judge_ok = complete and all(r.get("verdict") is not None for r in successful)
    fence_ok = all(not r.get("network_fence_hits", 0) and
                   "\"fence\"" not in json.dumps(r.get("model_errors", [])) for r in rows)
    classifier_pair_ok = all(r.get("classifier_pair_valid", True) for r in rows)
    case_by_id = {case.id: case for case in expected_cases}
    preconditions_ok = complete and all(row_precondition_valid(r, case_by_id[r["case"]]) for r in successful)
    return {"complete": complete, "minimum_samples_met": samples >= SAMPLES_PER_CASE,
            "non_emotion_model_input_sequence_equal": inputs_equal,
            "model_input_differences": input_differences,
            "persona_leak_free": persona_ok, "quality": result,
            "judge_complete": judge_ok,
            "network_fence_not_hit": fence_ok,
            "classifier_pair_contract_passed": classifier_pair_ok,
            "case_preconditions_passed": preconditions_ok,
            "performance": performance, "performance_passed": perf_ok,
            "passed": complete and samples >= SAMPLES_PER_CASE and inputs_equal and perf_ok and persona_ok and judge_ok and fence_ok and classifier_pair_ok and preconditions_ok
                      and all(s["passed"] for s in result.values())}


def logical_inputs(requests):
    by_phase = {}
    for request in requests:
        if request["phase"] == "reply_emotion":
            continue
        result = by_phase.setdefault(request["phase"], [])
        # Existing resilience can retry a transient failed HTTP call. Compare
        # logical model inputs; retain all attempts and failures in the report.
        if not result or request != result[-1]:
            result.append(request)
    # Intent and relevance are independent parallel reads. Their start-order
    # interleaving is not a semantic difference; preserve order within each role.
    return [request for phase in sorted(by_phase) for request in by_phase[phase]]


def configure_live_context(patches, decisions=None, classifier_pair=None):
    """Keep production relevance gating; substitute only domain reads/actions."""
    for name, value in (
        ("_load_schedule", []), ("_load_portrait", None),
        ("_load_topic_intimacy", 50.0), ("_load_time_memories", []),
        ("analyze_user_emotion", None), ("_decide_web_search", False),
        ("probe_knowledge_memories", []),
        ("_do_retrieval", {"memories": None, "memory_strings": None, "graph_context": None}),
    ):
        patches.setattr(data_fetch_phase, name, AsyncMock(return_value=value))
    patches.setattr(data_fetch_phase, "replace_latest_retrieval_selection", lambda **_: None)
    classify = data_fetch_phase._classify_relevance
    async def relevance(*args, **kwargs):
        token = REQUEST_PHASE.set("memory_relevance")
        try:
            result = await classify(*args, **kwargs)
            if decisions is not None:
                decisions.append({"phase": "memory_relevance", "level": result.level,
                                  "enhanced_query": result.enhanced_query})
            consumed = paired_decision(classifier_pair, "memory_relevance", classify, args, kwargs, result)
            if classifier_pair is not None and decisions is not None:
                decisions.append({"phase": "memory_relevance_consumed", "level": consumed.level,
                                  "enhanced_query": consumed.enhanced_query})
            return consumed
        except asyncio.CancelledError:
            if classifier_pair is not None:
                bound = signature(classify).bind(*args, **kwargs)
                bound.apply_defaults()
                classifier_pair.cancelled("memory_relevance", bound.arguments)
            if decisions is not None:
                decisions.append({"phase": "memory_relevance_cancelled"})
            raise
        finally:
            REQUEST_PHASE.reset(token)
    patches.setattr(data_fetch_phase, "_classify_relevance", relevance)
    patches.setattr(chat, "fetch_parallel_context", data_fetch_phase.fetch_parallel_context)


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    temp.replace(path)


async def grade_response(model, group, prompt):
    """Retry only invalid judge output, with the exact same frozen input.

    Valid negative verdicts are final. Keep every attempt as evidence; after
    three invalid outputs the row remains ungraded and cannot qualify. Provider
    or fence exceptions still fail the row without retry or relaxed parsing.
    """
    attempts = []
    token = REQUEST_PHASE.set("judge")
    try:
        for _ in range(3):
            raw = await model.ainvoke(prompt)
            output = str(raw.content)
            verdict = J.parse_verdict(group, output)
            attempts.append({"input_hash": digest(prompt), "output_hash": digest(output),
                             "output_chars": len(output), "verdict": verdict})
            if verdict is not None:
                return verdict, attempts
        return None, attempts
    finally:
        REQUEST_PHASE.reset(token)


async def evaluate_turn(case, executor, snapshot, requests, live_main, classifier_pair=None):
    with Patches() as patches:
        io = configure_pair(patches)
        configure_live_context(patches, requests.decisions, classifier_pair)
        patches.setattr(settings, "chat_executor", executor)
        patches.setattr(settings, "trace_backend", "off")
        patches.setattr(store, "db", SimpleNamespace(prompttemplate=SimpleNamespace(
            find_unique=AsyncMock(side_effect=lambda *, where: snapshot.get(where["key"])),
        )))
        async def main_reply(*args, **kwargs):
            requests.decisions.append({"phase": "reply_path", "path": "main"})
            requests.trusted_references.extend(str(m["content"]) for m in args[0] if m["role"] == "system")
            return await live_main(*args, **kwargs)
        patches.setattr(reply_generate, "_run_main_llm", main_reply)
        async def intent(*args, **kwargs):
            token = REQUEST_PHASE.set("intent")
            try:
                result = await intent_dispatcher.detect_intent_unified(*args, **kwargs)
                requests.decisions.append({"phase": "intent", "intent": result.intent.value,
                                          "confidence": result.confidence,
                                          "metadata": json.loads(json.dumps(result.metadata, default=str))})
                consumed = paired_decision(classifier_pair, "intent", intent_dispatcher.detect_intent_unified,
                                           args, kwargs, result)
                if classifier_pair is not None:
                    requests.decisions.append({"phase": "intent_consumed", "intent": consumed.intent.value,
                                              "confidence": consumed.confidence,
                                              "metadata": json.loads(json.dumps(consumed.metadata, default=str))})
                return consumed
            finally:
                REQUEST_PHASE.reset(token)
        patches.setattr(chat, "detect_intent_unified", intent)
        for kind in ("weak", "medium", "strong", "l3"):
            async def tier_reply(_kind=kind, _reply=getattr(intent_replies, f"memory_{kind}_reply"), **kwargs):
                requests.decisions.append({"phase": "reply_path", "path": _kind})
                requests.trusted_references.extend(str(kwargs[key]) for key in ("personality_brief", "ai_memory", "l3_memory")
                                                   if kwargs.get(key))
                return await _reply(**kwargs)
            patches.setattr(chat, f"_memory_{kind}_reply", tier_reply)
        async def emotion(text):
            token = REQUEST_PHASE.set("reply_emotion")
            try:
                return await intent_replies.ai_reply_emotion(text)
            finally:
                REQUEST_PHASE.reset(token)
        patches.setattr(chat, "_ai_reply_emotion", emotion)
        effective_history = history_for(case)
        context = "\n".join(f"{'AI' if role == 'assistant' else '用户'}: {text}" for role, text in effective_history)
        chat._fetch_intent_context.return_value = context
        # Fixed time and fresh IO per side, no agent/user production content.
        history = [SimpleNamespace(id=f"history-{i}", role=role, content=text, metadata={},
                   createdAt=io.now - timedelta(minutes=3 * (len(effective_history) - i)))
                   for i, (role, text) in enumerate(effective_history)]
        history.append(SimpleNamespace(id="u-new", role="user", content=case.message, metadata={}, createdAt=io.now))
        io.db.message.find_many.side_effect = lambda **_: list(reversed(history))
        usage = AsyncMock()
        patches.setattr(usage_repo, "write_usage_row", usage)
        requests.rows.clear()
        requests.errors.clear()
        requests.decisions.clear()
        requests.response_inputs.clear()
        requests.trusted_references.clear()
        requests.judge_prompt = None
        witness = witness_for(case)
        if witness:
            requests.trusted_references.append(witness)
        started = perf_counter()
        events = [event async for event in chat.stream_chat_response("c-1", case.message, io.agent, "u-1")]
        latency = (perf_counter() - started) * 1000
        if classifier_pair is not None:
            classifier_pair.finish(reply_uses_context=any(d["phase"] == "reply_path" for d in requests.decisions))
        reply_events = [json.loads(e["data"]) for e in events if e["event"] == "reply"]
        if sum(e["event"] == "done" for e in events) != 1 or not reply_events:
            raise RuntimeError("Missing reply or non-unique completion")
        if chat.finish_assistant_turn.await_count != 1 or io.db.message.create.await_count != 1:
            raise RuntimeError("Non-unique turn side effects")
        if usage.await_count != 1:
            raise RuntimeError("Missing or duplicate usage summary")
        summary = usage.await_args.kwargs["summary"]
        if summary["fallback_count"] or any(r.get("reply_failed") for r in reply_events):
            raise SampleRejected(summary, list(requests.rows), list(requests.errors))
        reply = "||".join(r["text"] for r in reply_events)
        fmt = J.analyse_format(reply)
        facts = fixture_facts(io.agent)
        from evals.graph_equivalence.preconditions import REQUIREMENTS
        requirement = REQUIREMENTS.get(case.id)
        judge_facts = facts[requirement.key].statement if requirement else None
        requests.judge_prompt = J.build_judge_prompt(case.group, "\n".join(f"{r}: {c}" for r, c in effective_history),
                                                     case.message, reply, known_facts=judge_facts)
        precondition = audit_case(case, facts, J.GROUNDED_FALSE_PREMISE_ASSUMPTIONS,
                                  trusted_references=requests.trusted_references, response_inputs=requests.response_inputs,
                                  judge_references=[judge_facts] if judge_facts else [], judge_inputs=[[requests.judge_prompt]])
        costs = sum(estimate_cost_cny(model, t["input"], t["output"], t["cached_input"])
                    for model, t in summary["tokens_by_model"].items())
        normalized = sum(estimate_cost_cny(model, t["input"], t["output"])
                         for model, t in summary["tokens_by_model"].items())
        return {"message": case.message, "reply": reply, "latency_ms": latency,
                "requests": list(requests.rows), "usage": summary, "cost_cny": costs,
                "routing_decisions": list(requests.decisions),
                "case_precondition": precondition,
                "judge_input_hash": digest(requests.judge_prompt),
                "classifier_pair_valid": classifier_pair is None or not classifier_pair.violations,
                "classifier_cancellations": list(classifier_pair.cancellations) if classifier_pair else [],
                "independently_completed_unused_classifiers": list(classifier_pair.independent_unused_results) if classifier_pair else [],
                "model_errors": list(requests.errors), "model_attempts": len(requests.rows),
                "normalized_cost_cny": normalized,
                "persona_leak": bool(re.search(r"作为\s*(?:AI|人工智能)|语言模型|我是\s*AI", reply, re.I)),
                "prices_known": all(is_known_model(m) for m in summary["tokens_by_model"]),
                "bubbles": fmt.bubbles, "max_bubble_chars": fmt.max_bubble_chars,
                "total_chars": fmt.total_chars, "emoji_count": fmt.emoji_count,
                "format_ok": fmt.format_ok}


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--read-only-production-snapshot", action="store_true")
    parser.add_argument("--preflight-only", action="store_true", help="Offline case-evidence audit; no DB, Redis or model calls")
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_CASE)
    parser.add_argument("--paired-classifiers", action="store_true",
                        help="Both classifiers run live; consume the first executor's decisions on both sides")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--case-limit", type=int, help="Smoke only; cannot qualify the entire bank")
    selection.add_argument("--case-id", action="append", help="Select named diagnostic cases; repeatable")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.preflight_only and not args.read_only_production_snapshot:
        parser.error("Live evaluation requires --read-only-production-snapshot")
    if args.samples < 1 or (args.case_limit is not None and args.case_limit < 1):
        parser.error("Sample/case counts must be positive")
    if args.case_id:
        unknown = set(args.case_id) - {case.id for case in ALL_CASES}
        if unknown:
            parser.error("Unknown case IDs")
        cases = [case for case in ALL_CASES if case.id in args.case_id]
    else:
        cases = ALL_CASES[:args.case_limit] if args.case_limit else ALL_CASES
    preconditions = audit_bank(cases, fixture_facts(synthetic_agent()), J.GROUNDED_FALSE_PREMISE_ASSUMPTIONS)
    report = {"schema_version": 6, "started_at": datetime.now(timezone.utc).isoformat(), "scenario_version": SCENARIO_VERSION,
              "scenario_witnesses": {case.id: digest(witness_for(case)) for case in cases if witness_for(case)},
              "case_preconditions": preconditions, "full_case_bank": len(cases) == len(ALL_CASES),
              "evaluation_hashes": {name: hashlib.sha256((Path(__file__).resolve().parents[2] / name).read_bytes()).hexdigest()
                                    for name in EVALUATION_FILES},
              "expected_case_ids": [case.id for case in cases], "rows": [], "calibration_passed": False,
              "qualification_passed": False}
    save(args.output, report)
    if args.preflight_only or not preconditions["passed"]:
        report["stage"] = "preflight_only" if args.preflight_only else "blocked_by_case_preconditions"
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save(args.output, report)
        print(json.dumps({"stage": report["stage"], "case_count": preconditions["case_count"],
                          "blocked_case_count": preconditions["blocked_case_count"],
                          "qualification_passed": False}, ensure_ascii=False), flush=True)
        if not preconditions["passed"]:
            raise SystemExit(2)
        return
    logging.basicConfig(level=logging.ERROR)
    settings.trace_backend = "off"
    os.environ["LANGSMITH_TRACING"] = "false"
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(name, None)
    snapshot, prompt_changes = prospective_prompt_snapshot(await read_snapshot())
    # Replace the DB root before importing additional application helpers.
    import app.db
    app.db.db = DeniedIO()
    settings.database_url = settings.direct_database_url = "postgresql://synthetic:synthetic@127.0.0.1:1/synthetic"
    settings.redis_url = "redis://127.0.0.1:1/0"
    import app.redis_client
    app.redis_client.get_redis = AsyncMock(side_effect=RuntimeError("Evaluation forbids business Redis"))
    requests = Requests()
    chat_model, utility_model = models.get_chat_model(), models.get_utility_model()
    for model in (chat_model, utility_model):
        model.callbacks = [requests]
    urls = [str(getattr(model, "openai_api_base", "")) for model in (chat_model, utility_model)]
    report.update({
              "comparison_mode": "common parsed classifier decisions; both sides invoke real classifiers" if args.paired_classifiers
                                 else "independently sampled classifiers",
              "scope": "Real intent/relevance/tier/main/reply-emotion models; synthetic domain, no external business actions",
              "routes": {"chat_provider": models._provider_for("chat"), "chat_model": models._chat_model_name(),
                         "utility_provider": models._provider_for("utility"), "utility_model": models._utility_model_name()},
              "dependencies": {name: version(name) for name in ("langgraph", "langchain-core", "langchain-openai")},
              "prospective_code_sync": prompt_changes,
              "source_hashes": {name: hashlib.sha256(Path(import_module(name[:-3].replace("/", ".")).__file__).read_bytes()).hexdigest() for name in (
                  "app/services/chat/orchestrator.py", "app/services/chat/graph_phases.py",
                  "app/services/chat/graph_runtime.py", "app/services/chat/main_graph.py",
                  "app/services/chat/reply_generate.py", "app/services/chat/intent_dispatcher.py",
                  "app/services/chat/data_fetch_phase.py", "app/services/memory/retrieval/relevance.py")},
              "synthetic_conditions": {"personality": "neutral", "memory_relevance": "production classifier and fast gates",
                                       "memory_candidates": "empty", "user_emotion_signal": None,
                                       "web_search": False, "current_activity": "在家整理工作台",
                                       "time": "2026-10-04T04:00:00+00:00", "reply_count": 2,
                                       "business_background_disabled": True, "artificial_sleep_disabled": True},
              "prompt_versions": [{"key": row.key, "hash": digest(row.content), "enabled": row.isEnabled,
                                  "updated_at": row.updatedAt.isoformat()} for row in snapshot.values()],
              "calibration_passed": False, "rows": []})
    save(args.output, report)
    live_main = reply_generate._run_main_llm
    with model_network_fence(urls) as fence:
        report["calibration_passed"] = await run_calibration(utility_model, concurrency=1, grounded_false_premises=True)
        report["network_fence_violations"] = list(fence.violations)
        save(args.output, report)
        if not report["calibration_passed"] or fence.violations:
            raise SystemExit("Judge calibration failed; no qualification claim or gate relaxation")
        for case_index, case in enumerate(cases):
            for sample in range(args.samples):
                # Alternate pair order to balance warm-cache/time-of-day bias.
                order = ("legacy", "langgraph") if (case_index + sample) % 2 == 0 else ("langgraph", "legacy")
                classifier_pair = ClassifierPair() if args.paired_classifiers else None
                for index, executor in enumerate(order):
                    if classifier_pair is not None:
                        classifier_pair.begin(recording=index == 0)
                    row = {"case": case.id, "group": case.group, "sample": sample, "executor": executor}
                    hits_before = len(fence.violations)
                    try:
                        row.update(await evaluate_turn(case, executor, snapshot, requests, live_main, classifier_pair))
                        if row["case_precondition"]["passed"]:
                            row["verdict"], row["judge_attempts"] = await grade_response(
                                utility_model, case.group, requests.judge_prompt,
                            )
                        else:
                            row["verdict"] = None
                    except Exception as error:
                        row["error"] = type(error).__name__
                        # Preserve actual request/decision/cancellation evidence
                        # even when a contract fails before turn analysis returns.
                        row.setdefault("requests", list(requests.rows))
                        row.setdefault("routing_decisions", list(requests.decisions))
                        row.setdefault("model_errors", list(requests.errors))
                        row.setdefault("classifier_cancellations", list(classifier_pair.cancellations) if classifier_pair else [])
                        if isinstance(error, SampleRejected):
                            row.update(usage=error.usage, requests=error.requests, model_errors=error.model_errors)
                        row["error_location"] = [{"file": Path(frame.filename).name, "line": frame.lineno,
                                                  "function": frame.name} for frame in traceback.extract_tb(error.__traceback__)]
                    row["network_fence_hits"] = len(fence.violations) - hits_before
                    if classifier_pair is not None:
                        row["classifier_pair_valid"] = not classifier_pair.violations
                        row["classifier_decision_source"] = order[0]
                    report["network_fence_violations"] = list(fence.violations)
                    report["rows"].append(row)
                    report["summary"] = report_summary(report["rows"], cases, args.samples)
                    report["full_case_bank"] = len(cases) == len(ALL_CASES)
                    save(args.output, report)
                    print(f"pair {case.id} #{sample + 1} {executor}: {'error' if 'error' in row else 'recorded'}", flush=True)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["qualification_passed"] = report["summary"]["passed"] and report["full_case_bank"]
    save(args.output, report)
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)
    if not (report["summary"]["passed"] and report["full_case_bank"]):
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as error:
        # Provider/DB exceptions can contain URLs or headers; never print them.
        print(f"Evaluation aborted: {type(error).__name__}", flush=True)
        raise SystemExit(2) from None
