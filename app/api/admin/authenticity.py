"""Admin "真人判定": per-user AI-relay probability with its evidence.

Reads only derived signal tables and message metadata. No raw IP, no draft
text; message bodies are reduced to lengths and punctuation flags in SQL.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.admin.users import _utc_iso
from app.api.jwt_auth import require_admin_jwt
from app.db import db
from app.services.authenticity import scoring

router = APIRouter(prefix="/admin-api/users", tags=["admin-users"])


def _user_summary(row: dict) -> dict:
    return {
        "user_id": row["user_id"],
        "username": row.get("username"),
        "display_name": row.get("display_name"),
        "internal": bool(row.get("internal")),
        "messages": int(row.get("messages") or 0),
        "last_message_at": _utc_iso(row.get("last_message_at")),
    }


@router.get("/authenticity")
async def authenticity_ranking(
    days: int = Query(30, ge=1, le=180),
    limit: int = Query(50, ge=1, le=200),
    include_internal: bool = False,
    _: dict = Depends(require_admin_jwt),
):
    rows = await scoring.feature_rows(days)
    items = []
    for row in rows:
        if row.get("internal") and not include_internal:
            continue
        assessed = scoring.assess(row)
        items.append(_user_summary(row) | {
            "probability": assessed["probability"], "verdict": assessed["verdict"]})
    items.sort(key=lambda item: (item["verdict"] == "insufficient", -item["probability"], -item["messages"]))
    return {"days": days, "model_version": scoring.MODEL_VERSION, "total": len(items), "items": items[:limit]}


@router.get("/{user_id}/authenticity")
async def user_authenticity(
    user_id: str,
    days: int = Query(30, ge=1, le=180),
    _: dict = Depends(require_admin_jwt),
):
    if not await db.query_raw("SELECT 1 FROM users WHERE id=$1", user_id):
        raise HTTPException(status_code=404, detail="User not found")
    rows = await scoring.feature_rows(days)
    row = next((r for r in rows if r["user_id"] == user_id), None)
    assessed = scoring.assess(row) if row else {
        "probability": None, "verdict": "insufficient", "contributions": []}
    networks = await db.query_raw(
        """SELECT o.ip_prefix, o.country, o.province, o.city, o.isp, o.network_kind, o.client_kind,
                  o.user_agent, o.source, o.hits, o.first_seen_at, o.last_seen_at,
                  COALESCE(array_agg(DISTINCT p.user_id) FILTER (WHERE p.user_id IS NOT NULL),
                           '{}') AS peers
           FROM user_network_observations o
           LEFT JOIN user_network_observations p ON p.ip_hash=o.ip_hash AND p.user_id<>o.user_id
           WHERE o.user_id=$1
           GROUP BY o.user_id, o.ip_hash
           ORDER BY o.last_seen_at DESC LIMIT 50""",
        user_id,
    )
    inputs = await db.query_raw(
        """SELECT created_at, chars, server_compose_ms, client_compose_ms, max_insert_chars,
                  edit_count, source
           FROM user_input_signals WHERE user_id=$1 ORDER BY created_at DESC LIMIT 60""",
        user_id,
    )
    return {
        "days": days,
        "model_version": scoring.MODEL_VERSION,
        "user": _user_summary(row) if row else {"user_id": user_id, "messages": 0},
        **assessed,
        "reference": scoring.reference_ranges(rows, exclude=user_id),
        "networks": [
            {k: n[k] for k in ("ip_prefix", "country", "province", "city", "isp", "network_kind",
                               "client_kind", "user_agent", "source", "hits")}
            | {"first_seen_at": _utc_iso(n["first_seen_at"]), "last_seen_at": _utc_iso(n["last_seen_at"]),
               "shared_account_count": len(n["peers"] or []), "shared_user_ids": list(n["peers"] or [])[:10]}
            for n in networks
        ],
        "inputs": [
            {"created_at": _utc_iso(i["created_at"]), "chars": i["chars"],
             "compose_ms": i["client_compose_ms"] if i["client_compose_ms"] is not None else i["server_compose_ms"],
             "compose_source": "app" if i["client_compose_ms"] is not None else (
                 "server" if i["server_compose_ms"] is not None else None),
             "max_insert_chars": i["max_insert_chars"], "edit_count": i["edit_count"], "source": i["source"]}
            for i in inputs
        ],
    }
