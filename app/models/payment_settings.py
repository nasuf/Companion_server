from typing import Any

from pydantic import BaseModel, ConfigDict, StrictBool, model_validator


class PaymentSettings(BaseModel):
    """Apple in-app purchase gate, as read by the App (``/iap/apple/*``)."""

    model_config = ConfigDict(extra="forbid")
    real_money_payments_enabled: StrictBool


class ApplePurchasePermissionRequest(BaseModel):
    """购买前预检的可选请求体。旧客户端不带请求体，只做开关检查。"""

    model_config = ConfigDict(extra="forbid")
    product_id: str | None = None


class AndroidStoreSettings(BaseModel):
    """安卓商城开关，由 Android 客户端读取（``/store/android-settings``）。"""

    model_config = ConfigDict(extra="forbid")
    wechat_pay_purchases_enabled: StrictBool
    android_auto_renew_enabled: StrictBool


class AdminPaymentSettings(BaseModel):
    """All purchase gates, as shown in the admin console."""

    real_money_payments_enabled: StrictBool  # Apple 内购
    wechat_pay_purchases_enabled: StrictBool  # 服务号 H5 微信支付
    android_auto_renew_enabled: StrictBool  # 安卓是否展示连续包月


class AdminPaymentSettingsUpdate(BaseModel):
    """Partial update: send only the switch being flipped.

    Older admin clients send ``{"real_money_payments_enabled": ...}`` alone and
    must keep working without touching the WeChat Pay switch.
    """

    model_config = ConfigDict(extra="forbid")
    real_money_payments_enabled: StrictBool | None = None
    wechat_pay_purchases_enabled: StrictBool | None = None
    android_auto_renew_enabled: StrictBool | None = None

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
