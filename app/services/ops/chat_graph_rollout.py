"""Read-only, scoped telemetry for G03; never activates or replays a turn."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

MAX_COHORT = 100
MAX_ROOTS = 1000
MIN_TURNS = 20
MIN_OBSERVATION = timedelta(hours=24)
MAX_WINDOW = timedelta(days=7)


@dataclass(frozen=True)
class Observation:
    conversation_ids: tuple[str, ...]
    executor: str
    graph_version: str
    since: datetime
    until: datetime
    all_conversations: bool = False

    def validate(self, now: datetime) -> None:
        if (
            not self.conversation_ids
            or len(self.conversation_ids) > MAX_COHORT
            or len(set(self.conversation_ids)) != len(self.conversation_ids)
            or any(not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", v) for v in self.conversation_ids)
        ):
            raise ValueError("Specify 1–100 distinct, exact authorized conversation IDs")
        if self.executor not in {"legacy", "langgraph"} or not self.graph_version:
            raise ValueError("An executor and expected graph version are required")
        if type(self.all_conversations) is not bool or (self.all_conversations and self.executor != "langgraph"):
            raise ValueError("Full rollout observation requires the graph executor")
        for value in (self.since, self.until, now):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("Observation timestamps must include a timezone")
        if not self.since < self.until <= now or self.until - self.since > MAX_WINDOW:
            raise ValueError("Use a past observation window of at most seven days")


def assess(
    observation: Observation, runtime: dict, roots: list[dict],
    nodes: list[dict], usage: list[dict], messages: list[dict], *,
    now: datetime,
) -> dict[str, Any]:
    """A telemetry gate, not proof of successful client delivery or release."""
    observation.validate(now)
    issues: list[dict] = []

    def issue(code: str, severity: str, detail: str) -> None:
        issues.append({"code": code, "severity": severity, "detail": detail})

    cohort = set(observation.conversation_ids)
    allowed = {v.strip() for v in runtime["allowlist"].split(",") if v.strip()}
    if runtime.get("all_conversations", False) != observation.all_conversations:
        issue("coverage_drift", "stop", "Runtime full-rollout mode differs from the expected mode")
    if (observation.all_conversations and allowed) or (
        not observation.all_conversations
        and (allowed - cohort or (observation.executor == "langgraph" and allowed != cohort))
    ):
        issue("cohort_drift", "stop", "Runtime allowlist differs from the authorized cohort")
    if runtime["executor"] != observation.executor:
        issue("executor_drift", "stop", "Runtime executor differs from the expected executor")
    if runtime["graph_version"] != observation.graph_version:
        issue("version_drift", "stop", "Installed graph version differs from the expected version")
    if runtime["durable_execution_ready"] is not False:
        issue("durability_changed", "stop", "Checkpoint qualification is outside G03")
    if runtime["trace_backend"] != "local":
        issue("trace_backend", "stop", "This collector requires local trace evidence")
    if len(roots) > MAX_ROOTS:
        issue("truncated", "wait", "Root limit exceeded; use a smaller cohort/window")
        roots = roots[:MAX_ROOTS]
    if observation.until - observation.since < MIN_OBSERVATION:
        issue("window_short", "wait", "At least 24 observed hours are required")
    if len(roots) < MIN_TURNS:
        issue("samples_short", "wait", f"At least {MIN_TURNS} traced turns are required")
    if cohort - {r["conversation_id"] for r in roots}:
        issue("cohort_unobserved", "wait", "Some authorized conversations have no traced turns")

    def indexed(rows: list[dict]) -> dict[str, dict]:
        index = {}
        for row in rows:
            trace_id = row["trace_id"]
            if trace_id in index:
                issue("duplicate_summary", "stop", "Duplicate per-trace telemetry summary")
            index[trace_id] = row
        return index

    node_by_id, usage_by_id, message_by_id = map(indexed, (nodes, usage, messages))
    counts: Counter = Counter()
    durations, costs = [], []
    total_calls = total_failures = total_fallbacks = zero_model_turns = 0
    seen = set()
    for root in roots:
        trace_id = root["trace_id"]
        if trace_id in seen or root["conversation_id"] not in cohort:
            issue("scope_or_duplicate_root", "stop", "Root scope or uniqueness is invalid")
        seen.add(trace_id)
        counts[str(root["status"])] += 1
        if root["executor"] not in {"legacy", "langgraph"}:
            issue("unmarked_root", "wait", "Historical/unmarked roots cannot prove executor selection")
        elif root["executor"] != observation.executor:
            issue("observed_executor", "stop", "A root ran a different executor")
        if root["checkpoint_enabled"] is None:
            issue("checkpoint_unrecorded", "wait", "Historical checkpoint metadata is missing")
        elif root["checkpoint_enabled"] not in ("false", False):
            issue("checkpoint_evidence", "stop", "Missing or enabled checkpoint metadata")
        if root["executor"] == "langgraph" and root["graph_version"] != observation.graph_version:
            issue("observed_version", "stop", "A root ran a different graph version")
        if root["status"] == "running":
            started = root["started_at"]
            if isinstance(started, str):
                started = datetime.fromisoformat(started.replace("Z", "+00:00"))
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)  # Prisma UTC timestamp columns.
            issue("stale_root" if now - started >= timedelta(minutes=10) else "pending_root",
                  "stop" if now - started >= timedelta(minutes=10) else "wait",
                  "A root has not completed")
            # In-flight replies, finish nodes and billing are not due yet.
            # Missing completion evidence must not turn a recent pending turn
            # into a reported production failure.
            continue
        elif root["status"] != "success" or root["ended_at"] is None:
            issue("root_failed", "stop", "A root failed or lacks a completion timestamp")
        node = node_by_id.get(trace_id, {})
        if root["executor"] == "langgraph":
            if node.get("graph_count") != 1 or node.get("finish_count") != 1:
                issue("graph_completion", "stop", "Graph or parent finish-node evidence is incomplete")
            if node.get("version_mismatches", 0):
                issue("node_version", "stop", "Graph-node metadata differs from the root version")
        elif node.get("graph_count", 0):
            issue("legacy_has_graph", "stop", "A legacy root contains a graph invocation")
        if root["status"] == "success" and message_by_id.get(trace_id, {}).get("reply_count", 0) < 1:
            issue("reply_missing", "stop", "Successful root has no persisted assistant reply")
        record = usage_by_id.get(trace_id)
        if (
            record is None and root.get("usage_expected") in ("false", False)
            and node.get("llm_count", 0) == 0
        ):
            # Explicit foreground-session evidence plus the complete trace tree;
            # do not infer zero cost merely from a missing billing row.
            zero_model_turns += 1
            costs.append(0.)
        elif record is None or record.get("rows") != 1:
            issue("usage_missing_or_duplicate", "wait" if record is None else "stop",
                  "Expected one parent chat-usage row per traced turn")
        else:
            cost = record["cost_cny"]
            if not isinstance(cost, int | float) or not math.isfinite(cost) or cost < 0:
                issue("cost_invalid", "stop", "Cost telemetry is invalid")
            else:
                costs.append(cost)
            valid_counters = {}
            for field in ("call_count", "failure_count", "fallback_count"):
                if (
                    not isinstance(record.get(field), int) or isinstance(record[field], bool)
                    or record[field] < 0
                ):
                    issue("usage_invalid", "stop", "Usage counters must be nonnegative integers")
                    valid_counters[field] = 0
                else:
                    valid_counters[field] = record[field]
            total_calls += valid_counters["call_count"]
            total_failures += valid_counters["failure_count"]
            total_fallbacks += valid_counters["fallback_count"]
        duration = root["duration_ms"]
        if isinstance(duration, int | float) and math.isfinite(duration) and duration >= 0:
            durations.append(duration)
        elif root["status"] == "success":
            issue("duration_missing", "wait", "Completed root lacks valid duration telemetry")

    # Summaries only: never return message text, model inputs/outputs or error text.
    unique_issues = list({(v["code"], v["severity"]): v for v in issues}.values())
    return {
        "schema_version": 1,
        "generated_at": now.isoformat(),
        "cohort_hash": hashlib.sha256(json.dumps(sorted(cohort)).encode()).hexdigest(),
        "cohort_size": len(cohort),
        "expected_executor": observation.executor,
        "expected_graph_version": observation.graph_version,
        "since": observation.since.isoformat(), "until": observation.until.isoformat(),
        "runtime": {k: runtime[k] for k in ("executor", "graph_version", "trace_backend")},
        "coverage": "all_conversations" if observation.all_conversations else "cohort",
        "telemetry_ready": not unique_issues,
        "release_ready": False,
        "issues": unique_issues,
        "metrics": {
            "traced_turns": len(roots), "root_statuses": dict(counts),
            "p95_root_duration_ms": sorted(durations)[math.ceil(.95 * len(durations)) - 1]
            if durations else None,
            "mean_chat_cost_cny": sum(costs) / len(costs) if costs else None,
            "chat_model_calls": total_calls, "model_failures": total_failures,
            "model_fallbacks": total_fallbacks,
            "zero_model_turns": zero_model_turns,
        },
        "limitations": [
            "Only persisted, explicitly marked trace roots are assessed.",
            "Client ack/done, delivery, side effects and quality require controlled smoke/E2E.",
            "Chat token cost is an estimate; background, speech and other charges are excluded.",
            "This report performs no matched-model latency/cost comparison.",
            "A telemetry pass alone does not authorize coverage expansion or checkpoint replay.",
        ],
    }


ROOTS_SQL = """
SELECT trace_id, inputs_json->>'conversation_id' AS conversation_id, status,
       started_at, ended_at,
       (EXTRACT(EPOCH FROM (ended_at-started_at))*1000)::float8 AS duration_ms,
       extra_json->'metadata'->>'executor' AS executor,
       extra_json->'metadata'->>'graph_version' AS graph_version,
       extra_json->'metadata'->>'checkpoint_enabled' AS checkpoint_enabled,
       extra_json->'metadata'->>'usage_expected' AS usage_expected
FROM trace_runs
WHERE name='chat_request' AND parent_id IS NULL AND run_id=trace_id
  AND inputs_json->>'conversation_id'=ANY($1::text[])
  AND started_at >= ($2::timestamptz AT TIME ZONE 'UTC')
  AND started_at < ($3::timestamptz AT TIME ZONE 'UTC')
  AND created_at >= ($2::timestamptz AT TIME ZONE 'UTC') - interval '10 minutes'
ORDER BY started_at, trace_id LIMIT 1001
"""
NODES_SQL = """
SELECT trace_id,
       COUNT(*) FILTER (WHERE run_type='llm')::int AS llm_count,
       COUNT(*) FILTER (WHERE name='main_chat')::int AS graph_count,
       COUNT(*) FILTER (WHERE name='finish_turn' AND status='success')::int AS finish_count,
       COUNT(*) FILTER (WHERE extra_json->'metadata'->>'executor'='langgraph'
         AND (extra_json->'metadata'->>'graph_version' IS DISTINCT FROM $2::text
           OR extra_json->'metadata'->>'checkpoint_enabled' IS DISTINCT FROM 'false'))::int
         AS version_mismatches
FROM trace_runs WHERE trace_id=ANY($1::text[]) AND parent_id IS NOT NULL
GROUP BY trace_id
"""
USAGE_SQL = """
SELECT trace_id, COUNT(*)::int AS rows, SUM(cost_cny)::float8 AS cost_cny,
       SUM(call_count)::int AS call_count, SUM(failure_count)::int AS failure_count,
       SUM(fallback_count)::int AS fallback_count
FROM llm_usage WHERE trace_id=ANY($1::text[]) AND conversation_id=ANY($2::text[])
  AND scope='chat'
GROUP BY trace_id
"""
MESSAGES_SQL = """
SELECT metadata->>'trace_id' AS trace_id, COUNT(*)::int AS reply_count
FROM messages WHERE conversation_id=ANY($1::text[]) AND role='assistant'
  AND metadata->>'trace_id'=ANY($2::text[])
  AND created_at >= ($3::timestamptz AT TIME ZONE 'UTC')
GROUP BY metadata->>'trace_id'
"""


async def collect(
    database, observation: Observation, runtime: dict, *, now: datetime,
) -> dict:
    """Read one consistent SQL snapshot. No Redis/model/job/notification calls."""
    observation.validate(now)  # Validate scope before even opening a transaction.
    async with database.tx(timeout=timedelta(seconds=30)) as tx:
        await tx.execute_raw("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        await tx.execute_raw("SET LOCAL statement_timeout = '5s'")
        roots = await tx.query_raw(
            ROOTS_SQL, list(observation.conversation_ids),
            observation.since.isoformat(), observation.until.isoformat(),
        )
        trace_ids = [r["trace_id"] for r in roots[:MAX_ROOTS]]
        if trace_ids:
            nodes = await tx.query_raw(NODES_SQL, trace_ids, observation.graph_version)
            usage = await tx.query_raw(USAGE_SQL, trace_ids, list(observation.conversation_ids))
            messages = await tx.query_raw(
                MESSAGES_SQL, list(observation.conversation_ids), trace_ids,
                observation.since.isoformat(),
            )
        else:
            nodes = usage = messages = []
    return assess(observation, runtime, roots, nodes, usage, messages, now=now)
