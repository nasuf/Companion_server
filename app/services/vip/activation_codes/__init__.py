"""VIP activation codes: admin generation, user redeem, entitlement stacking."""

from app.services.vip.activation_codes.errors import VipActivationError
from app.services.vip.activation_codes.redeem import redeem_code
from app.services.vip.activation_codes.admin import (
    create_codes,
    list_codes,
    list_redemptions,
    revoke_redemption,
    set_code_enabled,
)

__all__ = [
    "VipActivationError",
    "redeem_code",
    "create_codes",
    "list_codes",
    "list_redemptions",
    "revoke_redemption",
    "set_code_enabled",
]
