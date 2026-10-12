"""Admin APIs for reviewing and claiming a human takeover."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.jwt_auth import require_admin_jwt
from app.services.admin_takeover.service import (
    STATUS_OVERRIDE_VALUES,
    TakeoverClaimError,
    claim_takeover,
    dismiss_takeover,
    list_takeovers,
    mark_takeover_read,
    review_takeover,
    send_admin_message,
    set_status_override,
)
from app.services.runtime.ws_auth import (
    TicketInvalid,
    TicketRateLimited,
    TicketUnavailable,
    issue_ticket,
)

router = APIRouter(prefix="/admin-api/takeovers", tags=["admin-takeovers"])


class TakeoverMessageRequest(BaseModel):
    text: str = ""
    component_card: dict[str, Any] | None = None
    attachments: list[dict[str, Any]] = Field(default_factory=list)

    def normalized_text(self) -> str:
        return self.text.strip()


class TakeoverStatusRequest(BaseModel):
    status: str


def _admin_id(claims: dict) -> str:
    admin_id = str(claims.get("sub") or "")
    if not admin_id:
        raise HTTPException(status_code=401, detail="Admin access required")
    return admin_id


def _claim_http(error: TakeoverClaimError) -> HTTPException:
    if error.code == "not_found":
        return HTTPException(status_code=404, detail="Takeover not found")
    if error.code == "already_taken_over":
        return HTTPException(
            status_code=409,
            detail={
                "code": "already_taken_over",
                "reason": "already_taken_over",
                "owner_name": error.owner_name,
                "message": f"该用户已由{error.owner_name}接管",
            },
        )
    if error.code == "agent_not_ready":
        return HTTPException(status_code=409, detail={"code": "agent_not_ready"})
    if error.code == "not_owner":
        return HTTPException(status_code=403, detail="Not the owning admin")
    return HTTPException(status_code=409, detail={"code": error.code})


@router.get("")
async def get_takeovers(
    scope: str = "pending",
    claims: dict = Depends(require_admin_jwt),
):
    if scope not in {"pending", "mine"}:
        raise HTTPException(status_code=400, detail="scope must be pending or mine")
    items = await list_takeovers(_admin_id(claims), scope=scope)
    return {"items": items}


@router.post("/inbox-ticket")
async def inbox_ticket(claims: dict = Depends(require_admin_jwt)):
    admin_id = _admin_id(claims)
    try:
        return await issue_ticket(f"admin-inbox:{admin_id}", claims)
    except TicketInvalid:
        raise HTTPException(status_code=401, detail="Invalid or expired token") from None
    except TicketRateLimited:
        raise HTTPException(status_code=429, detail="Too many connection attempts") from None
    except TicketUnavailable:
        raise HTTPException(status_code=503, detail="Connection authentication unavailable") from None


@router.get("/{takeover_id}")
async def get_takeover(takeover_id: str, _: dict = Depends(require_admin_jwt)):
    detail = await review_takeover(takeover_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Takeover not found")
    return detail


@router.post("/{takeover_id}/dismiss")
async def post_dismiss(takeover_id: str, claims: dict = Depends(require_admin_jwt)):
    ok = await dismiss_takeover(takeover_id, _admin_id(claims))
    if not ok:
        raise HTTPException(status_code=404, detail="Pending takeover not found")
    return {"ok": True}


@router.post("/{takeover_id}/claim")
async def post_claim(takeover_id: str, claims: dict = Depends(require_admin_jwt)):
    try:
        return await claim_takeover(takeover_id, _admin_id(claims))
    except TakeoverClaimError as error:
        raise _claim_http(error) from None


@router.post("/{takeover_id}/read")
async def post_read(takeover_id: str, claims: dict = Depends(require_admin_jwt)):
    ok = await mark_takeover_read(takeover_id, _admin_id(claims))
    if not ok:
        raise HTTPException(status_code=403, detail="Not the owning admin")
    return {"ok": True}


@router.post("/{takeover_id}/messages")
async def post_message(
    takeover_id: str,
    body: TakeoverMessageRequest,
    claims: dict = Depends(require_admin_jwt),
):
    text = body.normalized_text()
    if not text and not body.component_card and not body.attachments:
        raise HTTPException(status_code=400, detail="Empty message")
    if len(text) > 4000:
        raise HTTPException(status_code=400, detail="Message is too long")
    try:
        return await send_admin_message(
            takeover_id=takeover_id,
            admin_id=_admin_id(claims),
            text=text,
            component_card=body.component_card,
            attachments=body.attachments,
        )
    except TakeoverClaimError as error:
        raise _claim_http(error) from None


@router.patch("/{takeover_id}/status")
async def patch_status(
    takeover_id: str,
    body: TakeoverStatusRequest,
    claims: dict = Depends(require_admin_jwt),
):
    status = body.status.strip()
    if status not in STATUS_OVERRIDE_VALUES:
        raise HTTPException(status_code=400, detail="Unknown status")
    try:
        return await set_status_override(takeover_id, _admin_id(claims), status)
    except TakeoverClaimError as error:
        raise _claim_http(error) from None
    except ValueError as error:
        raise HTTPException(status_code=400, detail="Unknown status") from error
