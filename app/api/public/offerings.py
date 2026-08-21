from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.jwt_auth import require_user
from app.models.offerings import (
    GiftSendResponse,
    RedPacketGetResponse,
    RedPacketSendResponse,
    SendGiftRequest,
    SendRedPacketRequest,
)
from app.models.wallet import WalletBalanceResponse
from app.services import offerings

router = APIRouter(prefix="/red-packets", tags=["red-packets"])
gift_router = APIRouter(prefix="/gifts", tags=["gifts"])


def _http_error(exc: ValueError) -> HTTPException:
    code = str(exc)
    if code == "insufficient_ticket_balance":
        return HTTPException(status_code=409, detail=code)
    if code == "insufficient_inventory":
        return HTTPException(status_code=409, detail=code)
    if code in {"conversation_not_found", "offering_not_found"}:
        return HTTPException(status_code=404, detail=code)
    if code in {"offering_forbidden", "offering_already_bound"}:
        return HTTPException(status_code=409, detail=code)
    if code in {"invalid_amount", "not_giftable"}:
        return HTTPException(status_code=400, detail=code)
    return HTTPException(status_code=400, detail=code)


@router.post("", response_model=RedPacketSendResponse)
async def send_red_packet(
    data: SendRedPacketRequest,
    payload: dict = Depends(require_user),
):
    try:
        result = await offerings.send_red_packet(
            user_id=str(payload["sub"]),
            conversation_id=data.conversation_id,
            ticket_amount=data.ticket_amount,
            blessing=data.blessing,
        )
    except ValueError as exc:
        raise _http_error(exc) from exc
    wallet = result["wallet"]
    return {
        "offering": offerings.public_offering(result["offering"]),
        "component_card": result["component_card"],
        "wallet": WalletBalanceResponse(
            ticket_balance=int(wallet["ticket_balance"]),
            point_balance=int(wallet["point_balance"]),
            achievement_points_synced=int(wallet["achievement_points_synced"]),
        ),
    }


@router.get("/{offering_id}", response_model=RedPacketGetResponse)
async def get_red_packet(
    offering_id: str,
    payload: dict = Depends(require_user),
):
    try:
        result = await offerings.get_red_packet(
            offering_id=offering_id,
            user_id=str(payload["sub"]),
        )
    except ValueError as exc:
        raise _http_error(exc) from exc
    return {
        "offering": offerings.public_offering(result["offering"]),
        "component_card": result["component_card"],
    }


def _wallet_response(wallet: dict) -> WalletBalanceResponse:
    return WalletBalanceResponse(
        ticket_balance=int(wallet["ticket_balance"]),
        point_balance=int(wallet["point_balance"]),
        achievement_points_synced=int(wallet["achievement_points_synced"]),
    )


@gift_router.post("", response_model=GiftSendResponse)
async def send_gift(
    data: SendGiftRequest,
    payload: dict = Depends(require_user),
):
    try:
        result = await offerings.send_gift(
            user_id=str(payload["sub"]),
            conversation_id=data.conversation_id,
            product_kind=data.product_kind,
        )
    except ValueError as exc:
        raise _http_error(exc) from exc
    return {
        "offering": offerings.public_offering(result["offering"]),
        "component_card": result["component_card"],
        "wallet": _wallet_response(result["wallet"]),
        "inventory_item": result["inventory_item"],
    }


@gift_router.get("/{offering_id}", response_model=RedPacketGetResponse)
async def get_gift(
    offering_id: str,
    payload: dict = Depends(require_user),
):
    try:
        result = await offerings.get_gift(
            offering_id=offering_id,
            user_id=str(payload["sub"]),
        )
    except ValueError as exc:
        raise _http_error(exc) from exc
    return {
        "offering": offerings.public_offering(result["offering"]),
        "component_card": result["component_card"],
    }
