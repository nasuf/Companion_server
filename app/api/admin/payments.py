from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.jwt_auth import require_admin_jwt
from app.models.iap import (
    AdminIapNotificationItem,
    AdminIapSubscriptionItem,
    AdminIapTransactionItem,
)
from app.services.payments import admin as payments_admin

router = APIRouter(
    prefix="/admin-api/payments",
    tags=["admin", "payments"],
    dependencies=[Depends(require_admin_jwt)],
)


@router.get("/transactions", response_model=list[AdminIapTransactionItem])
async def list_transactions(
    transaction_id: str | None = Query(default=None),
    user_id: str | None = Query(default=None),
    status: str | None = Query(default=None),
    environment: str | None = Query(default=None),
    kind: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    return await payments_admin.list_transactions(
        transaction_id=transaction_id,
        user_id=user_id,
        status=status,
        environment=environment,
        kind=kind,
        limit=limit,
        offset=offset,
    )


@router.get("/subscriptions", response_model=list[AdminIapSubscriptionItem])
async def list_subscriptions(
    user_id: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    return await payments_admin.list_subscriptions(
        user_id=user_id, status=status, limit=limit, offset=offset
    )


@router.get("/notifications", response_model=list[AdminIapNotificationItem])
async def list_notifications(
    notification_type: str | None = Query(default=None),
    unprocessed_only: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    return await payments_admin.list_notifications(
        notification_type=notification_type,
        unprocessed_only=unprocessed_only,
        limit=limit,
        offset=offset,
    )
