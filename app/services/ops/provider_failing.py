"""Bounded, content-free LLM availability observations.

Calls only append to memory. Each API worker flushes through the existing 30s
health tick; a SQL row lock merges workers and deduplicates retried batches by process sequence.
An incident survives an empty window/restart. Only a same-route complete success
STARTED after the latest failure can recover it. No balances or text are stored.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from app.config import settings
from app.db import db

_pending: list[dict] = []
_batch: dict | None = None
_lost = 0
_lock: asyncio.Lock | None = None
_worker_id = uuid4().hex
_sequence = 0


def policy() -> dict:
    return {
        "window_seconds": settings.provider_failing_window_seconds,
        "failure_threshold": settings.provider_failing_count,
        "failure_ratio_threshold": settings.provider_failing_ratio,
        "route_limit": settings.provider_failing_route_limit,
        "buffer_limit": settings.provider_failing_buffer_limit,
        "sample_interval_seconds": 30,
    }


def _at() -> float:
    return datetime.now(UTC).timestamp()


def _route(provider: str, model: str, role: str) -> str | None:
    if model.startswith(provider + "/"):
        model = model[len(provider) + 1 :]
    if (
        not re.fullmatch(r"[a-zA-Z0-9._:-]{1,40}", provider)
        or not re.fullmatch(r"[a-zA-Z0-9._:-]{1,160}", model)
        or role not in {"chat", "utility", "grounding", "other"}
    ):
        return None
    return f"{provider}/{model}/{role}"


def _append(event: dict) -> None:
    global _lost
    if len(_pending) >= settings.provider_failing_buffer_limit:
        _lost += 1
    else:
        _pending.append(event)


class Call:
    """One logical invocation; attempts share an ID, backup outcomes reference it."""

    def __init__(self, provider: str, model: str, role: str = "chat"):
        self.route = _route(provider, model, role)
        self.id = uuid4().hex

    def record(
        self,
        kind: str,
        started: float,
        *,
        error: Exception | None = None,
        circuit_open: bool = False,
    ) -> None:
        if not self.route:
            global _lost
            _lost += 1  # unknown route cannot silently count as healthy coverage
            return
        from app.services.ops.provider_budget import classify

        if error is not None and classify(self.route.split("/")[0], error):
            kind = "billing"
        _append(
            {
                "route": self.route,
                "request": self.id,
                "kind": kind,
                "started": started,
                "at": _at(),
                "circuit": circuit_open,
            }
        )

    def fallback(self, outcome: str) -> None:
        if self.route:
            _append(
                {
                    "route": self.route,
                    "request": self.id,
                    "kind": "fallback",
                    "outcome": outcome,
                    "at": _at(),
                }
            )


def _window(route: dict, at: float, config: dict) -> tuple[int, int]:
    # One-second buckets bound persisted state independently of request volume.
    cutoff = int(at - config["window_seconds"])
    route["buckets"] = {
        k: v for k, v in route.get("buckets", {}).items() if int(k) > cutoff
    }
    calls = sum(b[0] for b in route["buckets"].values())
    failures = sum(b[1] for b in route["buckets"].values())
    return calls, failures


def reduce_snapshot(
    snapshot: dict,
    events: list[dict],
    now: float,
    config: dict,
    *,
    evaluate: bool = True,
) -> dict:
    """Pure reducer shared by live collection and historical rule replay."""
    routes = snapshot.setdefault("routes", {})
    for route in routes.values():
        _window(route, now, config)
    for e in sorted(events, key=lambda x: (x["at"], x.get("started", x["at"]))):
        key = e["route"]
        if key not in routes:
            if len(routes) >= config["route_limit"]:
                snapshot["incomplete"] = True
                continue
            routes[key] = {"buckets": {}}
        r = routes[key]
        at, kind = e["at"], e["kind"]
        if kind == "fallback":
            if e["request"] == r.get("last_failure_request"):
                r["fallback"] = e["outcome"]
            continue
        if (
            kind in {"success", "failure", "billing"}
            and at > now - config["window_seconds"]
        ):
            bucket = r["buckets"].setdefault(str(int(at)), [0, 0])
            bucket[0] += 1
            bucket[1] += int(kind == "failure")
        if kind == "success":
            r["success_started"] = max(r.get("success_started", 0), e["started"])
        if kind in {"failure", "circuit"}:
            if at >= r.get("last_failure_at", 0):
                r.update(
                    last_failure_at=at,
                    last_failure_request=e["request"],
                    fallback="pending",
                    reason="circuit_open"
                    if e.get("circuit") or kind == "circuit"
                    else "failure_rate",
                )
        if kind == "circuit" or (kind == "failure" and e.get("circuit")):
            r["circuit_at"] = max(r.get("circuit_at", 0), at)
        if r.get("success_started", 0) > r.get("last_failure_at", 0):
            r.pop("incident_at", None)
        r["last_observed_at"] = max(r.get("last_observed_at", 0), at)
    if evaluate:
        evaluate_incidents(snapshot, now, config)
    return snapshot


def evaluate_incidents(snapshot: dict, now: float, config: dict) -> None:
    # Latch at the service sample, after merging worker batches. Latching while
    # each worker flushes can turn its temporarily partial denominator into an
    # irreversible false incident (late, older successes cannot recover it).
    for r in snapshot.get("routes", {}).values():
        calls, failures = _window(r, now, config)
        triggered = r.get("circuit_at", 0) > r.get("success_started", 0) or (
            failures >= config["failure_threshold"]
            and failures / max(calls, 1) >= config["failure_ratio_threshold"]
        )
        if triggered and r.get("last_failure_at", 0) >= r.get("success_started", 0):
            r.setdefault("incident_at", r["last_failure_at"])


async def flush() -> None:
    """No request waits on this. Retry a frozen batch after uncertain SQL outcome."""
    global _batch, _lost, _lock, _sequence
    if _lock is None:
        _lock = asyncio.Lock()
    async with _lock:
        if _batch is None:
            # Swapping is synchronous; calls arriving during SQL await use the new list.
            events = _pending[: settings.provider_failing_buffer_limit]
            del _pending[: len(events)]
            _sequence += 1
            _batch = {
                "worker": _worker_id,
                "sequence": _sequence,
                "events": events,
                "lost": _lost,
                "at": _at(),
            }
            _lost = 0
        batch = _batch
        await _persist_batch(batch)
        _batch = None


async def _persist_batch(batch: dict) -> None:
    async with asyncio.timeout(3):
        async with db.tx(timeout=timedelta(seconds=3)) as tx:
            await tx.execute_raw("SET LOCAL lock_timeout='1s'")
            await tx.execute_raw("SET LOCAL statement_timeout='1500ms'")
            await tx.execute_raw(
                "INSERT INTO ops_provider_health(id) VALUES('primary') ON CONFLICT DO NOTHING"
            )
            row = (
                await tx.query_raw(
                    "SELECT snapshot FROM ops_provider_health WHERE id='primary' FOR UPDATE"
                )
            )[0]
            snapshot = row["snapshot"]
            if isinstance(snapshot, str):
                snapshot = json.loads(snapshot)
            now = _at()
            # A watermark per worker (not one UUID per flush) bounds daily write
            # amplification. Gaps/expired receipts never masquerade as complete data.
            receipts = {
                k: v
                for k, v in snapshot.get("receipts", {}).items()
                if v["at"] > now - settings.provider_failing_receipt_seconds
            }
            previous_sequence = receipts.get(batch["worker"], {}).get("sequence", 0)
            if batch["sequence"] > previous_sequence:
                if batch["sequence"] != previous_sequence + 1:
                    snapshot["incomplete"] = True
                if batch["at"] <= now - settings.provider_failing_receipt_seconds:
                    snapshot["incomplete"] = True
                else:
                    snapshot = reduce_snapshot(
                        snapshot, batch["events"], now, policy(), evaluate=False
                    )
                snapshot["lost_observations"] = (
                    snapshot.get("lost_observations", 0) + batch["lost"]
                )
                if batch["lost"]:
                    snapshot["incomplete"] = True
                receipts[batch["worker"]] = {
                    "sequence": batch["sequence"],
                    "at": now,
                    "observed_at": batch["at"],
                }
            if len(receipts) > settings.provider_failing_receipt_limit:
                snapshot["incomplete"] = True
                receipts = dict(
                    sorted(receipts.items(), key=lambda item: item[1]["at"])[
                        -settings.provider_failing_receipt_limit :
                    ]
                )
            snapshot["receipts"] = receipts
            await tx.execute_raw(
                "UPDATE ops_provider_health SET snapshot=$1::jsonb, sampled_at=$2::timestamptz WHERE id='primary'",
                json.dumps(snapshot),
                datetime.fromtimestamp(now, UTC).isoformat(),
            )


async def measure(now: datetime, carry: dict) -> dict:
    config = policy()
    base = {
        "threshold": 1,
        **config,
        "recovery": "same_route_later_complete_success",
        "counter_unit": "completed_provider_attempts_including_retries",
        "window_precision_seconds": 1,
    }
    async with db.tx(timeout=timedelta(seconds=3)) as tx:
        await tx.execute_raw("SET LOCAL lock_timeout='1s'")
        await tx.execute_raw("SET LOCAL statement_timeout='1500ms'")
        rows = await tx.query_raw(
            "SELECT snapshot,sampled_at FROM ops_provider_health WHERE id='primary' FOR UPDATE"
        )
        if not rows:
            return {
                **base,
                "status": "unknown",
                "alerting": False,
                "reason": "not_collected",
            }
        snapshot = rows[0]["snapshot"]
        if isinstance(snapshot, str):
            snapshot = json.loads(snapshot)
        # A healthy writer must not hide another worker's delayed batch. Use
        # capture time, not commit time (a retry can commit minutes later).
        # Retire silent peers only after their whole counting window ages out.
        peers = [
            r
            for r in snapshot.get("receipts", {}).values()
            if r["at"] > now.timestamp() - config["window_seconds"]
        ]
        peer_incomplete = any(
            r.get("observed_at", r["at"])
            < now.timestamp() - settings.provider_failing_stale_seconds
            for r in peers
        )
        if not peer_incomplete:
            evaluate_incidents(snapshot, now.timestamp(), config)
        # Do not refresh sampled_at here: a reader cannot fabricate collector health.
        await tx.execute_raw(
            "UPDATE ops_provider_health SET snapshot=$1::jsonb WHERE id='primary'",
            json.dumps(snapshot),
        )
    stamp = rows[0]["sampled_at"]
    if isinstance(stamp, str):
        stamp = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    age = (now - stamp.replace(tzinfo=UTC)).total_seconds()
    routes = []
    for key, r in snapshot.get("routes", {}).items():
        calls, failures = _window(r, now.timestamp(), config)
        if not calls and not r.get("incident_at"):
            continue
        provider, model, role = key.split("/")

        def iso(value):
            return datetime.fromtimestamp(value, UTC).isoformat() if value else None

        routes.append(
            {
                "provider": provider,
                "model": model,
                "role": role,
                "calls": calls,
                "failures": failures,
                "failure_ratio": failures / calls if calls else None,
                "alerting": bool(r.get("incident_at")),
                "first_failure_at": iso(r.get("incident_at")),
                "last_failure_at": iso(r.get("last_failure_at")),
                "fallback": r.get("fallback", "not_needed"),
                "reason": r.get("reason"),
            }
        )
    incidents = [r for r in routes if r["alerting"]]
    incomplete = bool(
        peer_incomplete
        or snapshot.get("incomplete")
        or _lost
        or age > settings.provider_failing_stale_seconds
        or (
            _batch
            and _batch["at"] < now.timestamp() - settings.provider_failing_stale_seconds
        )
    )
    return {
        **base,
        "status": "unknown" if incomplete else "measured",
        "count": len(incidents),
        "sample_state": "sampled" if routes else "no_sample",
        "alerting": bool(incidents),
        "routes": routes,
        "incidents": incidents,
        "sampled_at": stamp.isoformat(),
        "lost_observations": snapshot.get("lost_observations", 0) + _lost,
        "incomplete": incomplete,
        "peer_collection_incomplete": peer_incomplete,
        "observed_workers": len(peers),
    }
