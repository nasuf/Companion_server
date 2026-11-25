"""Purchase switches (Apple / WeChat Pay); activation codes and internal wallet spending are separate."""
from fastapi import APIRouter, Depends, Response

from app.api.jwt_auth import require_admin_jwt
from app.models.payment_settings import AdminPaymentSettings, AdminPaymentSettingsUpdate
from app.services.payments.availability import get_payment_settings, set_payment_settings

router = APIRouter(
    prefix="/admin-api/payment-settings",
    tags=["admin", "payments"],
    dependencies=[Depends(require_admin_jwt)],
)


@router.get("", response_model=AdminPaymentSettings)
async def read_payment_settings(response: Response):
    response.headers["Cache-Control"] = "no-store"
    return await get_payment_settings()


@router.put("", response_model=AdminPaymentSettings)
async def update_payment_settings(
    data: AdminPaymentSettingsUpdate, payload: dict = Depends(require_admin_jwt)
):
    return await set_payment_settings(actor_id=str(payload["sub"]), **data.changes())
