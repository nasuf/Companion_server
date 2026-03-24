from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException, status

LANGSMITH_PUBLIC_HOST = "smith.langchain.com"
LANGSMITH_PUBLIC_API = "https://api.smith.langchain.com"
TRACE_URL_RE = re.compile(
    r"^/public/(?P<share_token>[0-9a-f-]+)/r(?:/(?P<run_id>[0-9a-f-]+))?$",
    re.IGNORECASE,
)
RUN_SELECT_FIELDS = [
    "id",
    "name",
    "run_type",
    "status",
    "start_time",
    "end_time",
    "parent_run_id",
    "parent_run_ids",
    "child_run_ids",
    "direct_child_run_ids",
    "trace_id",
    "dotted_order",
    "inputs",
    "outputs",
    "error",
    "events",
    "extra",
    "app_path",
    "total_tokens",
    "prompt_tokens",
    "completion_tokens",
    "first_token_time",
    "prompt_token_details",
    "completion_token_details",
]


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _duration_ms(start: str | None, end: str | None) -> int | None:
    start_dt = _parse_iso(start)
    end_dt = _parse_iso(end)
    if not start_dt or not end_dt:
        return None
    return max(0, math.floor((end_dt - start_dt).total_seconds() * 1000))


def _latency_ms(start: str | None, first_token: str | None) -> int | None:
    start_dt = _parse_iso(start)
    first_dt = _parse_iso(first_token)
    if not start_dt or not first_dt:
        return None
    return max(0, math.floor((first_dt - start_dt).total_seconds() * 1000))


def _parse_trace_url(trace_url: str) -> tuple[str, str | None]:
    parsed = urlparse(trace_url.strip())
    if parsed.scheme != "https" or parsed.netloc != LANGSMITH_PUBLIC_HOST:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported trace URL",
        )
    match = TRACE_URL_RE.match(parsed.path)
    if not match:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid trace URL",
        )
    return match.group("share_token"), match.group("run_id")


def _extract_model_name(run: dict[str, Any]) -> str | None:
    extra = run.get("extra") or {}
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


def _normalize_step(run: dict[str, Any]) -> dict[str, Any]:
    start_time = run.get("start_time")
    end_time = run.get("end_time")
    first_token_time = run.get("first_token_time")
    child_ids = run.get("child_run_ids") or []
    parent_ids = run.get("parent_run_ids") or []
    return {
        "id": run.get("id"),
        "name": run.get("name"),
        "run_type": run.get("run_type"),
        "status": run.get("status"),
        "parent_id": run.get("parent_run_id"),
        "parent_ids": parent_ids if isinstance(parent_ids, list) else [],
        "child_ids": child_ids if isinstance(child_ids, list) else [],
        "trace_id": run.get("trace_id"),
        "dotted_order": run.get("dotted_order"),
        "started_at": start_time,
        "ended_at": end_time,
        "duration_ms": _duration_ms(start_time, end_time),
        "first_token_ms": _latency_ms(start_time, first_token_time),
        "first_token_time": first_token_time,
        "model_name": _extract_model_name(run),
        "total_tokens": run.get("total_tokens"),
        "prompt_tokens": run.get("prompt_tokens"),
        "completion_tokens": run.get("completion_tokens"),
        "prompt_token_details": run.get("prompt_token_details"),
        "completion_token_details": run.get("completion_token_details"),
        "inputs": run.get("inputs"),
        "outputs": run.get("outputs"),
        "error": run.get("error"),
        "events": run.get("events") or [],
        "extra": run.get("extra"),
        "app_path": run.get("app_path"),
        "raw": run,
    }


async def _fetch_run(
    client: httpx.AsyncClient,
    share_token: str,
    run_id: str | None,
) -> dict[str, Any]:
    path = (
        f"{LANGSMITH_PUBLIC_API}/public/{share_token}/run/{run_id}"
        if run_id
        else f"{LANGSMITH_PUBLIC_API}/public/{share_token}/run"
    )
    response = await client.get(
        path,
        params={
            "exclude_s3_stored_attributes": "true",
            "exclude_serialized": "true",
        },
    )
    if response.status_code == status.HTTP_404_NOT_FOUND:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trace not found")
    response.raise_for_status()
    return response.json()


async def _fetch_runs_by_ids(
    client: httpx.AsyncClient,
    share_token: str,
    run_ids: list[str],
) -> list[dict[str, Any]]:
    if not run_ids:
        return []
    response = await client.post(
        f"{LANGSMITH_PUBLIC_API}/public/{share_token}/runs/query",
        json={"id": run_ids, "select": RUN_SELECT_FIELDS},
        headers={"Content-Type": "application/json"},
    )
    response.raise_for_status()
    payload = response.json()
    runs = payload.get("runs") or []
    return [run for run in runs if isinstance(run, dict)]


async def load_public_trace(trace_url: str) -> dict[str, Any]:
    share_token, run_id = _parse_trace_url(trace_url)
    timeout = httpx.Timeout(20.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            root_run = await _fetch_run(client, share_token, run_id)
            runs_by_id: dict[str, dict[str, Any]] = {
                str(root_run["id"]): root_run,
            }
            pending_ids = [
                child_id
                for child_id in (root_run.get("child_run_ids") or [])
                if isinstance(child_id, str)
            ]
            seen_ids = set(runs_by_id)

            while pending_ids:
                batch = [run_id for run_id in pending_ids if run_id not in seen_ids][:50]
                pending_ids = [run_id for run_id in pending_ids if run_id not in batch]
                if not batch:
                    break
                children = await _fetch_runs_by_ids(client, share_token, batch)
                for child in children:
                    child_id = child.get("id")
                    if not isinstance(child_id, str) or child_id in seen_ids:
                        continue
                    runs_by_id[child_id] = child
                    seen_ids.add(child_id)
                    for grandchild_id in child.get("child_run_ids") or []:
                        if isinstance(grandchild_id, str) and grandchild_id not in seen_ids:
                            pending_ids.append(grandchild_id)
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Trace fetch failed: {exc.response.status_code}",
            ) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Trace fetch failed",
            ) from exc

    normalized_steps = [_normalize_step(run) for run in runs_by_id.values()]
    normalized_steps.sort(key=lambda step: (step.get("dotted_order") or "", step.get("id") or ""))

    llm_steps = [step for step in normalized_steps if step.get("run_type") == "llm"]
    total_tokens = sum(
        int(step.get("total_tokens") or 0)
        for step in normalized_steps
        if isinstance(step.get("total_tokens"), int | float)
    )

    root_step = next((step for step in normalized_steps if step["id"] == root_run["id"]), normalized_steps[0])
    return {
        "trace": {
            "share_token": share_token,
            "run_id": run_id or root_step["id"],
            "external_url": trace_url,
            "root_id": root_step["id"],
            "trace_id": root_step.get("trace_id"),
            "name": root_step.get("name"),
            "run_type": root_step.get("run_type"),
            "status": root_step.get("status"),
            "started_at": root_step.get("started_at"),
            "ended_at": root_step.get("ended_at"),
            "duration_ms": root_step.get("duration_ms"),
            "conversation_id": ((root_step.get("inputs") or {}).get("conversation_id")),
            "message": ((root_step.get("inputs") or {}).get("message")),
            "step_count": len(normalized_steps),
            "llm_step_count": len(llm_steps),
            "total_tokens": total_tokens,
        },
        "steps": normalized_steps,
    }
