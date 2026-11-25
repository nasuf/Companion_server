from typing import Any

from pydantic import BaseModel, ConfigDict, StrictBool, model_validator


class PaymentSettings(BaseModel):
    """Apple in-app purchase gate, as read by the App (``/iap/apple/*``)."""

    model_config = ConfigDict(extra="forbid")
    real_money_payments_enabled: StrictBool


class AdminPaymentSettings(BaseModel):
    """Both purchase gates, as shown in the admin console."""

    real_money_payments_enabled: StrictBool  # Apple 内购
    wechat_pay_purchases_enabled: StrictBool  # 服务号 H5 微信支付


class AdminPaymentSettingsUpdate(BaseModel):
    """Partial update: send only the switch being flipped.

    Older admin clients send ``{"real_money_payments_enabled": ...}`` alone and
    must keep working without touching the WeChat Pay switch.
    """

    model_config = ConfigDict(extra="forbid")
    real_money_payments_enabled: StrictBool | None = None
    wechat_pay_purchases_enabled: StrictBool | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_empty_or_null(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data:
                raise ValueError("at least one switch is required")
            if any(value is None for value in data.values()):
                raise ValueError("switch values must be true or false")
        return data

    def changes(self) -> dict[str, bool]:
        return self.model_dump(exclude_unset=True)
