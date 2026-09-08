"""Admin API: last-will SMS smoke test."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.api.jwt_auth import require_admin_jwt
from app.services.last_will import send_test_last_will_sms
from app.services.sms.tencent import SmsSendError

router = APIRouter(
    prefix="/admin-api/last-wills",
    tags=["admin", "last-wills"],
    dependencies=[Depends(require_admin_jwt)],
)


class LastWillSmsTestRequest(BaseModel):
    phone: str = Field(min_length=5, max_length=20)


class LastWillSmsTestResponse(BaseModel):
    ok: bool = True
    phone_tail: str
    mock: bool


@router.post("/sms-test", response_model=LastWillSmsTestResponse)
async def test_last_will_sms(payload: LastWillSmsTestRequest) -> LastWillSmsTestResponse:
    """Send one last-will notification SMS to the given mainland-CN phone."""
    try:
        result = await send_test_last_will_sms(payload.phone)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="手机号格式不正确",
        ) from None
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="遗言短信未配置或未启用",
        ) from None
    except SmsSendError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    return LastWillSmsTestResponse(
        phone_tail=str(result.get("phone_tail") or ""),
        mock=bool(result.get("mock")),
    )
