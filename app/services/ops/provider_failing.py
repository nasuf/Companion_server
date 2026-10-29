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


ROLE_LABELS = {
    "chat": "主聊天模型",
    "utility": "小模型",
    "grounding": "人设校验模型",
    "other": "其他模型",
}
# A dormant incident (see evaluate_incidents) is listed for a day, then forgotten.
_FORGET_DORMANT_AFTER = 86400


def policy() -> dict:
    return {
        "window_seconds": settings.provider_failing_window_seconds,
        "failure_threshold": settings.provider_failing_count,
        "failure_ratio_threshold": settings.provider_failing_ratio,
        "route_limit": settings.provider_failing_route_limit,
        "buffer_limit": settings.provider_failing_buffer_limit,
        # Never shorter than the window, or a dormant route would re-trigger on its own failures.
        "dormant_seconds": max(
            settings.provider_failing_dormant_seconds,
            settings.provider_failing_window_seconds,
        ),
        "sample_interval_seconds": 30,
    }


def _at() -> float:
    return datetime.now(UTC).timestamp()


def _route(provider: str, model: str, role: str) -> str | None:
    if not all(isinstance(value, str) for value in (provider, model, role)):
        return None
    if model.startswith(provider + "/"):
        model = model[len(provider) + 1 :]
    if (
        not re.fullmatch(r"[a-zA-Z0-9._:-]{1,40}", provider)
        or not re.fullmatch(r"[a-zA-Z0-9._:-]{1,160}", model)
        or role not in ROLE_LABELS
    ):
        return None
    return f"{provider}/{model}/{role}"


def _lose() -> None:
    global _lost
    _lost += 1


def _append(event: dict) -> None:
    if len(_pending) >= settings.provider_failing_buffer_limit:
        _lose()
    else:
        _pending.append(event)


def _mark_gap(snapshot: dict, at: float) -> None:
    # Missing observations can only hide failures inside the rolling window, so a
    # gap makes the check incomplete for one window rather than forever.
    snapshot["gap_at"] = max(snapshot.get("gap_at", 0), at)


class Call:
    """One logical invocation; attempts share an ID, backup outcomes reference it.

    Recording runs inside the model call's own try blocks: an exception here would
    turn a completed reply into a failure. So it never raises; problems count as lost.
    """

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
        try:
            if not self.route:
                _lose()  # unknown route cannot silently count as healthy coverage
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
        except Exception:
            _lose()

    def fallback(self, outcome: str) -> None:
        try:
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
        except Exception:
            _lose()


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
                _mark_gap(snapshot, now)
                continue
            routes[key] = {"buckets": {}}
        r = routes[key]
        at, kind = e["at"], e["kind"]
        if kind == "fallback":
            if e["request"] == r.get("last_failure_request"):
                r["fallback"] = e["outcome"]
            continue
        r.pop("dormant_at", None)  # traffic is back: it either recovers or alerts again
        if (
            kind in {"success", "failure", "billing"}
            and at > now - config["window_seconds"]
        ):
            bucket = r["buckets"].setdefault(str(int(at)), [0, 0])
            bucket[0] += 1
            bucket[1] += int(kind == "failure")
        if kind == "success":
            r["success_started"] = max(r.get("success_started", 0), e["started"])
            if e["request"] == r.get("last_failure_request"):
                # A retry of that request worked; no backup ran.
                r["fallback"] = "retry_succeeded"
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
    routes = snapshot.get("routes", {})
    for key, r in list(routes.items()):
        calls, failures = _window(r, now, config)
        if (
            r.get("incident_at")
            and now - r.get("last_observed_at", 0) >= config["dormant_seconds"]
        ):
            # A retired route or an idle backup gets no call that could prove
            # recovery; an open incident would hold every alert at warning forever.
            # Stop alerting, and keep it listed as unconfirmed rather than recovered.
            r.pop("incident_at")
            r["dormant_at"] = now
        circuit = r.get("circuit_at", 0) > max(
            r.get("success_started", 0), now - config["window_seconds"]
        )
        triggered = circuit or (
            failures >= config["failure_threshold"]
            and failures / max(calls, 1) >= config["failure_ratio_threshold"]
        )
        if triggered and r.get("last_failure_at", 0) >= r.get("success_started", 0):
            r.setdefault("incident_at", r["last_failure_at"])
        idle_since = max(r.get("last_observed_at", 0), r.get("dormant_at", 0))
        if (
            not r.get("incident_at")
            and not r["buckets"]
            and now - idle_since > _FORGET_DORMANT_AFTER
        ):
            del routes[key]  # keeps route_limit for routes still in use


async def retire() -> None:
    """Graceful shutdown: write what this worker saw and mark its receipt closed.

    A stopped worker's receipt otherwise looks like a lagging peer for a whole
    window, so every deploy pushed a false "服务巡检异常" and then "恢复".
    A worker that dies without this still counts as lagging, which is correct.
    """
    await flush()  # includes a frozen retry batch, if any
    await flush(closing=True)


async def flush(*, closing: bool = False) -> None:
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
        await _persist_batch(batch, closing=closing)
        _batch = None


async def _persist_batch(batch: dict, *, closing: bool = False) -> None:
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
                    _mark_gap(snapshot, now)
                if batch["at"] <= now - settings.provider_failing_receipt_seconds:
                    _mark_gap(snapshot, now)
                else:
                    snapshot = reduce_snapshot(
                        snapshot, batch["events"], now, policy(), evaluate=False
                    )
                snapshot["lost_observations"] = (
                    snapshot.get("lost_observations", 0) + batch["lost"]
                )
                if batch["lost"]:
                    _mark_gap(snapshot, now)
                receipts[batch["worker"]] = {
                    "sequence": batch["sequence"],
                    "at": now,
                    "observed_at": batch["at"],
                }
            if closing and batch["worker"] in receipts:
                receipts[batch["worker"]]["closed"] = True
            snapshot.pop("incomplete", None)  # legacy permanent flag; gaps now age out
            if len(receipts) > settings.provider_failing_receipt_limit:
                _mark_gap(snapshot, now)
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
            and not r.get("closed")  # a gracefully stopped worker is not lagging
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

    def iso(value):
        return datetime.fromtimestamp(value, UTC).isoformat() if value else None

    routes = []
    for key, r in snapshot.get("routes", {}).items():
        calls, failures = _window(r, now.timestamp(), config)
        if not calls and not r.get("incident_at") and not r.get("dormant_at"):
            continue
        provider, model, role = key.split("/")
        routes.append(
            {
                "provider": provider,
                "model": model,
                "role": role,
                "role_label": ROLE_LABELS.get(role, role),
                "calls": calls,
                "failures": failures,
                "failure_ratio": failures / calls if calls else None,
                "alerting": bool(r.get("incident_at")),
                "dormant": bool(r.get("dormant_at")),
                "dormant_at": iso(r.get("dormant_at")),
                "first_failure_at": iso(r.get("incident_at")),
                "last_failure_at": iso(r.get("last_failure_at")),
                "fallback": r.get("fallback", "not_needed"),
                "reason": r.get("reason"),
            }
        )
    incidents = [r for r in routes if r["alerting"]]
    # A gap can hide failures only inside the rolling window, so it ages out.
    gap_until = snapshot.get("gap_at", 0) + config["window_seconds"]
    incomplete = bool(
        peer_incomplete
        or gap_until > now.timestamp()
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
        "gap_until": iso(gap_until) if gap_until > now.timestamp() else None,
        "peer_collection_incomplete": peer_incomplete,
        "observed_workers": len(peers),
    }
