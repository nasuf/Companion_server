"""Self-hosted trace collection replacing LangSmith run storage.

Collection side:
    `LocalTraceHandler` subclasses langchain's `AsyncBaseTracer`, which already
    assembles `Run` objects in the exact shape LangSmith ingests (inputs via
    `dumpd`, outputs via `LLMResult.model_dump` with `dumpd` messages). We only
    persist each run into the `trace_runs` table: one row inserted on run
    start, upserted with outputs on run end. Late background runs (memory
    extraction fired after the reply was emitted) keep writing rows because
    the handler travels with `contextvars.copy_context()`.

    `LocalTracer` mirrors the `LangSmithTracer` interface
    (enter / attach_to_parent / close / trace_id / safe_trace_id / is_active)
    and manages the per-request lifecycle: a synthetic root run
    ("chat_request"), installing the handler into a ContextVar registered with
    langchain's configure hooks (so every LLM call auto-attaches without
    touching call sites), and closing the root run.

Read side:
    `load_local_trace` reads the rows back and produces the same normalized
    `{"trace": ..., "steps": [...]}` shape as `public_trace.load_public_trace`,
    including synthesized `dotted_order` / `child_ids`, so `trace_enrich` and
    the web Trace panel work unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import uuid
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from typing import Any

from langchain_core.tracers.base import AsyncBaseTracer
from prisma import Json

from app.config import settings

logger = logging.getLogger(__name__)

# Runs still marked running after this window are shown as "cancelled": the
# LLM call was likely killed by asyncio.wait_for, which skips langchain's
# end/error callbacks (the same root cause that left LangSmith traces
# eternally "pending").
STALE_RUNNING_AFTER = timedelta(minutes=10)

ROOT_RUN_NAME = "chat_request"

_local_trace_handler: ContextVar[Any | None] = ContextVar(
    "local_trace_handler",
    default=None,
)

_configure_hook_registered = False


def _ensure_configure_hook() -> None:
    """Register the handler ContextVar with langchain's callback configure hooks.

    Once registered, every langchain callback manager picks up the handler
    stored in `_local_trace_handler` (inheritable → child async tasks and
    `copy_context` background tasks keep tracing). Registration is process
    global and idempotent.
    """
    global _configure_hook_registered
    if _configure_hook_registered:
        return
    from langchain_core.tracers.context import register_configure_hook

    register_configure_hook(_local_trace_handler, inheritable=True)
    _disable_langsmith_env_tracing()
    _configure_hook_registered = True


def _disable_langsmith_env_tracing() -> None:
    """Safety net: kill langchain's env-driven LangSmith auto-uploads.

    langchain reads LANGSMITH_TRACING / LANGCHAIN_TRACING_V2 directly from the
    environment on every call. If ops forgets to flip them off while
    trace_backend=local, every LLM call would still upload runs to LangSmith
    (burning the monthly quota this backend exists to escape) — observed live
    as `LangSmithRateLimitError: Monthly unique traces usage limit exceeded`.
    """
    import os

    stale = [
        var for var in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING")
        if os.environ.get(var, "").strip().lower() in ("true", "1", "yes")
    ]
    if not stale:
        return
    for var in stale:
        os.environ[var] = "false"
    logger.warning(
        f"[local-trace] trace_backend=local but {'/'.join(stale)} was true; "
        "forcibly disabled LangSmith env tracing to protect quota. "
        "Set LANGSMITH_TRACING=false in .env to silence this warning."
    )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _json_safe(value: Any) -> Any:
    """Best-effort conversion to JSON-serializable data (datetime → str)."""
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return None


def _fire(coro) -> None:
    """Schedule a fire-and-forget DB write; drop silently without a loop.

    Trace persistence must never break or slow the chat hot path. The wrapped
    coroutines swallow their own exceptions, so no done-callback is needed.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        coro.close()
        return
    loop.create_task(coro)


# ─────────────────────────────────────────────────────────────────
# Collection: langchain callback handler + per-request tracer
# ─────────────────────────────────────────────────────────────────


def _extract_model_name_from_run(run: Any) -> str | None:
    """Same lookup order as public_trace._extract_model_name, at write time."""
    extra = getattr(run, "extra", None) or {}
    metadata = extra.get("metadata") or {}
    invocation = extra.get("invocation_params") or {}
    for candidate in (
        metadata.get("ls_model_name"),
        metadata.get("model_name"),
        metadata.get("model"),
        invocation.get("model_name"),
        invocation.get("model"),
    ):
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_usage_from_outputs(outputs: dict[str, Any] | None) -> dict[str, Any]:
    """Extract token usage from an LLM run's serialized outputs.

    Prefers langchain's normalized `usage_metadata` on the generation message
    (input_tokens / output_tokens / input_token_details.cache_read), falls
    back to provider `llm_output.token_usage` (prompt_tokens / ...).
    `prompt_token_details` uses the LangSmith key `cache_read` that the web
    panel already understands.
    """
    result: dict[str, Any] = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "prompt_token_details": None,
    }
    if not isinstance(outputs, dict):
        return result

    usage_meta: dict[str, Any] | None = None
    generations = outputs.get("generations")
    if isinstance(generations, list) and generations:
        first_group = generations[0]
        first = first_group[0] if isinstance(first_group, list) and first_group else first_group
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                kwargs = message.get("kwargs") or {}
                candidate = kwargs.get("usage_metadata")
                if isinstance(candidate, dict):
                    usage_meta = candidate

    if usage_meta:
        result["prompt_tokens"] = _as_int(usage_meta.get("input_tokens"))
        result["completion_tokens"] = _as_int(usage_meta.get("output_tokens"))
        result["total_tokens"] = _as_int(usage_meta.get("total_tokens"))
        details = usage_meta.get("input_token_details")
        if isinstance(details, dict):
            cache_read = _as_int(details.get("cache_read") or details.get("cached_tokens"))
            if cache_read is not None:
                result["prompt_token_details"] = {"cache_read": cache_read}
        if result["total_tokens"] is None and (
            result["prompt_tokens"] is not None or result["completion_tokens"] is not None
        ):
            result["total_tokens"] = (result["prompt_tokens"] or 0) + (result["completion_tokens"] or 0)
        return result

    llm_output = outputs.get("llm_output")
    if isinstance(llm_output, dict):
        token_usage = llm_output.get("token_usage") or llm_output.get("usage")
        if isinstance(token_usage, dict):
            result["prompt_tokens"] = _as_int(token_usage.get("prompt_tokens"))
            result["completion_tokens"] = _as_int(token_usage.get("completion_tokens"))
            result["total_tokens"] = _as_int(token_usage.get("total_tokens"))
            details = token_usage.get("prompt_tokens_details")
            if isinstance(details, dict):
                cache_read = _as_int(details.get("cached_tokens"))
                if cache_read is not None:
                    result["prompt_token_details"] = {"cache_read": cache_read}
    return result


def _trim_events(events: list[Any] | None) -> list[dict[str, Any]]:
    """Keep start/end/error plus the first new_token event.

    Streaming runs append one new_token event per token; storing them all
    would bloat rows for zero display value (the panel only needs the first
    token timestamp for latency).
    """
    if not events:
        return []
    trimmed: list[dict[str, Any]] = []
    token_seen = False
    for event in events:
        if not isinstance(event, dict):
            continue
        name = event.get("name")
        if name == "new_token":
            if token_seen:
                continue
            token_seen = True
            # Drop the chunk payload; only the timestamp matters.
            trimmed.append({"name": "new_token", "time": event.get("time")})
            continue
        trimmed.append({k: v for k, v in event.items() if k != "kwargs"})
    return trimmed


def _first_token_time(events: list[Any] | None) -> datetime | None:
    for event in events or []:
        if isinstance(event, dict) and event.get("name") == "new_token":
            time_value = event.get("time")
            if isinstance(time_value, datetime):
                return time_value
            if isinstance(time_value, str):
                try:
                    return datetime.fromisoformat(time_value.replace("Z", "+00:00"))
                except ValueError:
                    return None
    return None


def _normalize_run_type(run_type: str | None) -> str:
    # langchain 1.x tags chat-model runs as "chat_model"; LangSmith (and the
    # web panel) call them "llm".
    return "llm" if run_type in ("chat_model", "llm") else (run_type or "chain")


class LocalTraceHandler(AsyncBaseTracer):
    """Async langchain tracer persisting each run into trace_runs.

    Instantiated per request by `LocalTracer.enter()`; langchain discovers it
    through the ContextVar registered in `_ensure_configure_hook`.
    """

    # Never raise into the LLM call path (langchain also guards handlers).
    raise_error = False

    def __init__(self, trace_id: str, root_run_id: str) -> None:
        # "original+chat" enables on_chat_model_start (same as LangChainTracer).
        super().__init__(_schema_format="original+chat")
        self.local_trace_id = trace_id
        self.local_root_run_id = root_run_id

    async def _persist_run(self, run: Any) -> None:
        # Rows are written incrementally in _on_run_create/_on_run_update; the
        # root-run persist hook of BaseTracer is unused.
        return None

    async def _on_run_create(self, run: Any) -> None:
        _fire(_write_run_start(self, run))

    async def _on_run_update(self, run: Any) -> None:
        _fire(_write_run_end(self, run))


def _run_row_base(handler: LocalTraceHandler, run: Any) -> dict[str, Any]:
    parent_id = getattr(run, "parent_run_id", None)
    return {
        "id": str(run.id),
        "traceId": handler.local_trace_id,
        # Top-level langchain runs have no parent; attach them to our
        # synthetic per-request root so the panel renders one tree.
        "parentId": str(parent_id) if parent_id else handler.local_root_run_id,
        "name": getattr(run, "name", None) or "run",
        "runType": _normalize_run_type(getattr(run, "run_type", None)),
        "startedAt": getattr(run, "start_time", None) or _utcnow(),
    }


async def _write_run_start(handler: LocalTraceHandler, run: Any) -> None:
    try:
        from app.db import db

        base = _run_row_base(handler, run)
        inputs = _json_safe(getattr(run, "inputs", None))
        extra = _json_safe(getattr(run, "extra", None))
        create_data: dict[str, Any] = {
            **base,
            "status": "running",
            "modelName": _extract_model_name_from_run(run),
        }
        if inputs is not None:
            create_data["inputsJson"] = Json(inputs)
        if extra is not None:
            create_data["extraJson"] = Json(extra)
        await db.tracerun.create(data=create_data)
    except Exception as e:
        # Duplicate key is expected when the end upsert won the race.
        logger.debug(f"[local-trace] run start write skipped {run.id}: {type(e).__name__}: {e}")


async def _write_run_end(handler: LocalTraceHandler, run: Any) -> None:
    try:
        from app.db import db

        base = _run_row_base(handler, run)
        error_text = getattr(run, "error", None)
        outputs = _json_safe(getattr(run, "outputs", None))
        events = getattr(run, "events", None) or []
        usage = _extract_usage_from_outputs(outputs if isinstance(outputs, dict) else None)
        ended_at = getattr(run, "end_time", None) or _utcnow()
        first_token_at = _first_token_time(events)
        trimmed_events = _json_safe(_trim_events(events)) or []
        inputs = _json_safe(getattr(run, "inputs", None))
        extra = _json_safe(getattr(run, "extra", None))

        update_data: dict[str, Any] = {
            "status": "error" if error_text else "success",
            "error": str(error_text) if error_text else None,
            "endedAt": ended_at,
            "eventsJson": Json(trimmed_events),
            "modelName": _extract_model_name_from_run(run),
            "promptTokens": usage["prompt_tokens"],
            "completionTokens": usage["completion_tokens"],
            "totalTokens": usage["total_tokens"],
        }
        if first_token_at is not None:
            update_data["firstTokenAt"] = first_token_at
        if usage["prompt_token_details"] is not None:
            update_data["promptTokenDetails"] = Json(usage["prompt_token_details"])
        if outputs is not None:
            update_data["outputsJson"] = Json(outputs)

        create_data = {**base, **update_data}
        if inputs is not None:
            create_data["inputsJson"] = Json(inputs)
        if extra is not None:
            create_data["extraJson"] = Json(extra)

        # Upsert converges even when the start insert lost the race or failed.
        await db.tracerun.upsert(
            where={"id": base["id"]},
            data={"create": create_data, "update": update_data},
        )
    except Exception as e:
        logger.debug(f"[local-trace] run end write failed {run.id}: {type(e).__name__}: {e}")


async def _write_root_start(
    trace_id: str, user_message: str, conversation_id: str, started_at: datetime,
) -> None:
    try:
        from app.db import db

        await db.tracerun.create(data={
            "id": trace_id,
            "traceId": trace_id,
            "name": ROOT_RUN_NAME,
            "runType": "chain",
            "status": "running",
            "startedAt": started_at,
            "inputsJson": Json({
                "message": user_message,
                "conversation_id": conversation_id,
            }),
        })
    except Exception as e:
        logger.debug(f"[local-trace] root start write failed {trace_id}: {type(e).__name__}: {e}")


async def _write_root_end(trace_id: str, ended_at: datetime) -> None:
    try:
        from app.db import db

        await db.tracerun.update(
            where={"id": trace_id},
            data={"status": "success", "endedAt": ended_at},
        )
    except Exception as e:
        logger.debug(f"[local-trace] root end write failed {trace_id}: {type(e).__name__}: {e}")


class LocalTracer:
    """Per-request local trace lifecycle; drop-in interface for LangSmithTracer."""

    def __init__(self, user_message: str, conversation_id: str) -> None:
        self._user_message = user_message
        self._conversation_id = conversation_id
        self._closed = False
        self._attached = False
        self._cv_token = None
        self.trace_id: str | None = None

    @property
    def is_active(self) -> bool:
        return settings.trace_backend == "local"

    @property
    def safe_trace_id(self) -> str | None:
        """is_active 时返回 trace_id, 否则 None — 调用方写消息 metadata 用."""
        return self.trace_id if self.is_active else None

    def enter(self) -> "LocalTracer":
        """Open the trace: root run row + handler install. Root call only."""
        if not self.is_active:
            return self
        _ensure_configure_hook()
        self.trace_id = str(uuid.uuid4())
        handler = LocalTraceHandler(self.trace_id, self.trace_id)
        self._cv_token = _local_trace_handler.set(handler)
        _fire(_write_root_start(
            self.trace_id, self._user_message, self._conversation_id, _utcnow(),
        ))
        return self

    def attach_to_parent(self, parent_trace_id: str | None) -> "LocalTracer":
        """sub_intent mode: reuse the parent's trace_id and handler.

        The parent's handler is still installed in the current context, so LLM
        calls made while processing the sub fragment keep attaching to the
        parent tree; we only propagate the trace_id for message metadata.
        """
        self.trace_id = parent_trace_id
        self._attached = True
        return self

    def close(self) -> None:
        """Close the root run and uninstall the handler. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._attached or not self.trace_id:
            return
        if self._cv_token is not None:
            try:
                _local_trace_handler.reset(self._cv_token)
            except ValueError:
                # Token created in another context (defensive); background
                # tasks hold their own context copies either way.
                _local_trace_handler.set(None)
            self._cv_token = None
        _fire(_write_root_end(self.trace_id, _utcnow()))


# ─────────────────────────────────────────────────────────────────
# Read side: trace_runs rows → normalized detail (load_public_trace shape)
# ─────────────────────────────────────────────────────────────────


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else None


def _duration_ms(start: datetime | None, end: datetime | None) -> int | None:
    if not isinstance(start, datetime) or not isinstance(end, datetime):
        return None
    return max(0, math.floor((end - start).total_seconds() * 1000))


def _dotted_segment(started_at: datetime | None, run_id: str) -> str:
    stamp = (started_at or _utcnow()).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}{run_id}"


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _row_to_step(row: Any) -> dict[str, Any]:
    """Map a trace_runs row to the normalized step shape of _normalize_step."""
    started_at: datetime | None = row.startedAt
    ended_at: datetime | None = row.endedAt
    first_token_at: datetime | None = row.firstTokenAt

    status = row.status
    if status == "running":
        # The web panel understands LangSmith's "pending"; stale runs are
        # flagged "cancelled" by _mark_stale_running afterwards.
        status = "pending"

    step: dict[str, Any] = {
        "id": row.id,
        "name": row.name,
        "run_type": row.runType,
        "status": status,
        "parent_id": row.parentId,
        "parent_ids": [row.parentId] if row.parentId else [],
        "child_ids": [],  # filled by _assign_tree_fields
        "trace_id": row.traceId,
        "dotted_order": None,  # filled by _assign_tree_fields
        "started_at": _iso(started_at),
        "ended_at": _iso(ended_at),
        "duration_ms": _duration_ms(started_at, ended_at),
        "first_token_ms": _duration_ms(started_at, first_token_at),
        "first_token_time": _iso(first_token_at),
        "model_name": row.modelName,
        "total_tokens": row.totalTokens,
        "prompt_tokens": row.promptTokens,
        "completion_tokens": row.completionTokens,
        "prompt_token_details": row.promptTokenDetails,
        "completion_token_details": None,
        "inputs": row.inputsJson,
        "outputs": row.outputsJson,
        "error": row.error,
        "events": row.eventsJson or [],
        "extra": row.extraJson,
        "app_path": None,
    }
    # "raw" powers the panel's 原始 JSON view; reference the normalized fields
    # (minus the bulky payloads) instead of duplicating the whole row.
    step["raw"] = {
        **{k: v for k, v in step.items() if k not in ("inputs", "outputs")},
        "source": "local",
    }
    return step


def _mark_stale_running(steps: list[dict[str, Any]], *, now: datetime | None = None) -> None:
    """Flag long-running "pending" steps as cancelled (skipped end callbacks)."""
    current = now or _utcnow()
    for step in steps:
        if step.get("status") != "pending":
            continue
        started = _parse_iso(step.get("started_at"))
        if started is not None and current - started > STALE_RUNNING_AFTER:
            step["status"] = "cancelled"
            raw = step.get("raw")
            if isinstance(raw, dict):
                raw["status"] = "cancelled"


def _assign_tree_fields(steps: list[dict[str, Any]]) -> None:
    """Fill child_ids and synthesize LangSmith-style dotted_order."""
    by_id = {step["id"]: step for step in steps}
    children: dict[str | None, list[dict[str, Any]]] = {}
    for step in steps:
        parent_id = step.get("parent_id")
        if parent_id not in by_id:
            # Dangling parent (e.g. partial retention purge): treat as a tree
            # root for traversal while keeping the original parent_id field.
            parent_id = None
        children.setdefault(parent_id, []).append(step)
        if parent_id:
            by_id[parent_id]["child_ids"].append(step["id"])

    def _start_key(step: dict[str, Any]) -> str:
        return step.get("started_at") or ""

    def _walk(step: dict[str, Any], prefix: str) -> None:
        segment = _dotted_segment(_parse_iso(step.get("started_at")), step["id"])
        step["dotted_order"] = f"{prefix}.{segment}" if prefix else segment
        raw = step.get("raw")
        if isinstance(raw, dict):
            raw["dotted_order"] = step["dotted_order"]
            raw["child_ids"] = step["child_ids"]
        for child in sorted(children.get(step["id"], []), key=_start_key):
            _walk(child, step["dotted_order"])

    for root in sorted(children.get(None, []), key=_start_key):
        _walk(root, "")


async def count_local_trace_runs(trace_id: str) -> int:
    """Row count for mirror freshness checks; 0 when purged/absent."""
    if not trace_id:
        return 0
    try:
        from app.db import db

        return await db.tracerun.count(where={"traceId": trace_id})
    except Exception as e:
        logger.warning(f"[local-trace] count failed for {trace_id}: {e}")
        return 0


async def load_local_trace(trace_id: str) -> dict[str, Any] | None:
    """Build a load_public_trace-shaped detail from local rows.

    Returns None when no rows exist (legacy LangSmith trace or purged by
    retention). The trace summary carries `settled=False` while some runs are
    still in flight so the caller can defer the mirror write.
    """
    if not trace_id:
        return None
    try:
        from app.db import db

        rows = await db.tracerun.find_many(
            where={"traceId": trace_id},
            order={"startedAt": "asc"},
        )
    except Exception as e:
        logger.warning(f"[local-trace] read failed for {trace_id}: {e}")
        return None
    if not rows:
        return None

    steps = [_row_to_step(row) for row in rows]
    _mark_stale_running(steps)
    _assign_tree_fields(steps)
    steps.sort(key=lambda step: (step.get("dotted_order") or "", step.get("id") or ""))

    from app.services.chat.trace_enrich import enrich_steps

    enrich_steps(steps)

    root_step = next(
        (step for step in steps if step["id"] == trace_id),
        next((step for step in steps if not step.get("parent_id")), steps[0]),
    )
    llm_steps = [step for step in steps if step.get("run_type") == "llm"]
    total_tokens = sum(
        int(step.get("total_tokens") or 0)
        for step in steps
        if isinstance(step.get("total_tokens"), int | float)
    )
    root_inputs = root_step.get("inputs") if isinstance(root_step.get("inputs"), dict) else {}
    settled = not any(step.get("status") == "pending" for step in steps)

    # Root duration: while the root row is still open (or background runs
    # extend past it), fall back to the latest child end time so the header
    # shows a sane wall-clock number.
    duration_ms = root_step.get("duration_ms")
    if duration_ms is None:
        root_started = _parse_iso(root_step.get("started_at"))
        ends = [_parse_iso(step.get("ended_at")) for step in steps]
        ends = [e for e in ends if e is not None]
        if ends and root_started is not None:
            duration_ms = _duration_ms(root_started, max(ends))

    return {
        "trace": {
            "share_token": None,
            "run_id": root_step["id"],
            "external_url": None,
            "source": "local",
            "settled": settled,
            "root_id": root_step["id"],
            "trace_id": trace_id,
            "name": root_step.get("name"),
            "run_type": root_step.get("run_type"),
            "status": root_step.get("status"),
            "started_at": root_step.get("started_at"),
            "ended_at": root_step.get("ended_at"),
            "duration_ms": duration_ms,
            "conversation_id": root_inputs.get("conversation_id"),
            "message": root_inputs.get("message"),
            "step_count": len(steps),
            "llm_step_count": len(llm_steps),
            "total_tokens": total_tokens,
        },
        "steps": steps,
    }


async def purge_expired_trace_runs(*, retention_days: int | None = None) -> int:
    """Delete trace_runs rows older than the retention window.

    Viewed traces live on in the message_traces mirror; unviewed ones become
    unresolvable after purge (the resolve endpoint reports trace_expired).
    """
    days = retention_days if retention_days is not None else settings.trace_retention_days
    if days <= 0:
        return 0
    cutoff = _utcnow() - timedelta(days=days)
    from app.db import db

    deleted = await db.tracerun.delete_many(where={"createdAt": {"lt": cutoff}})
    if deleted:
        logger.info(f"[local-trace] purged {deleted} trace_runs rows older than {days}d")
    return deleted
