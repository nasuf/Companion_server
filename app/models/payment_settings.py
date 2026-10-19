from pydantic import BaseModel, ConfigDict, StrictBool


class PaymentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    real_money_payments_enabled: StrictBool
