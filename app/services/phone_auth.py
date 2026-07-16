"""Phone-number identity: login and two-way binding with WeChat accounts.

Table design — no schema change. A phone number is an ``auth_identities`` row
(``provider='phone'``, ``providerAccountId=<11-digit number>``), structurally
identical to the WeChat identity rows:

* one table owns every login method; ``unique(provider, providerAccountId)``
  guarantees a phone belongs to exactly one account;
* "bind phone to WeChat account" / "bind WeChat to phone account" are both just
  inserting (or re-pointing) an identity row on the same ``User``;
* conflict detection ("this phone/WeChat already belongs to another account")
  is a single indexed lookup.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime

from prisma import Json
from prisma.errors import UniqueViolationError

from app.db import db

logger = logging.getLogger(__name__)

PHONE_PROVIDER = "phone"


class IdentityConflict(Exception):
    """The target phone/WeChat identity already belongs to another account."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _phone_username(phone: str) -> str:
    digest = hashlib.sha256(f"phone:{phone}".encode("utf-8")).hexdigest()[:16]
    return f"ph_{digest}"


async def find_or_create_phone_user(
    phone: str,
    *,
    signup_fields: dict[str, str] | None = None,
):
    """Login path: resolve the account owning ``phone``, creating one if new.

    ``signup_fields`` (signupSource/...) are persisted only on the create
    branch — existing accounts keep their original registration origin.
    """
    identity = await db.authidentity.find_first(
        where={"provider": PHONE_PROVIDER, "providerAccountId": phone}
    )
    if identity:
        await db.authidentity.update(
            where={"id": identity.id},
            data={"lastLoginAt": datetime.now(UTC)},
        )
        return await db.user.find_unique(where={"id": identity.userId})

    user_data: dict[str, object] = {
        "username": _phone_username(phone),
        "hashedPassword": None,
        "role": "user",
    }
    if signup_fields:
        user_data.update(signup_fields)
    try:
        async with db.tx() as tx:
            user = await tx.user.create(data=user_data)
            await tx.authidentity.create(
                data={
                    "user": {"connect": {"id": user.id}},
                    "provider": PHONE_PROVIDER,
                    "providerAccountId": phone,
                    "rawProfile": Json({"phone": phone}),
                    "lastLoginAt": datetime.now(UTC),
                }
            )
            return user
    except UniqueViolationError:
        # Concurrent first-login for the same phone — converge on the winner.
        identity = await db.authidentity.find_first(
            where={"provider": PHONE_PROVIDER, "providerAccountId": phone}
        )
        if not identity:
            raise
        return await db.user.find_unique(where={"id": identity.userId})


async def bind_phone_to_user(user_id: str, phone: str) -> None:
    """Attach ``phone`` to an existing account (e.g. a WeChat-login user).

    Semantics:
    * phone already on this account         -> no-op (idempotent)
    * phone owned by another account        -> IdentityConflict("phone_taken")
    * account already has a different phone -> re-point that row (换绑)
    * otherwise                             -> create the identity row
    """
    existing = await db.authidentity.find_first(
        where={"provider": PHONE_PROVIDER, "providerAccountId": phone}
    )
    if existing:
        if existing.userId != user_id:
            raise IdentityConflict("phone_taken")
        return

    mine = await db.authidentity.find_first(
        where={"provider": PHONE_PROVIDER, "userId": user_id}
    )
    try:
        if mine:
            await db.authidentity.update(
                where={"id": mine.id},
                data={
                    "providerAccountId": phone,
                    "rawProfile": Json({"phone": phone}),
                    "lastLoginAt": datetime.now(UTC),
                },
            )
        else:
            await db.authidentity.create(
                data={
                    "user": {"connect": {"id": user_id}},
                    "provider": PHONE_PROVIDER,
                    "providerAccountId": phone,
                    "rawProfile": Json({"phone": phone}),
                    "lastLoginAt": datetime.now(UTC),
                }
            )
    except UniqueViolationError as exc:
        # Lost a race against another account binding the same phone.
        raise IdentityConflict("phone_taken") from exc
    logger.info(
        "phone bound to user",
        extra={"event": "phone_bound", "user_id": user_id, "phone_tail": phone[-4:]},
    )


async def get_identity_summary(user_id: str) -> tuple[str | None, bool]:
    """Return (phone, wechat_bound) for surfacing binding state to clients."""
    identities = await db.authidentity.find_many(where={"userId": user_id})
    phone: str | None = None
    wechat_bound = False
    for identity in identities:
        if identity.provider == PHONE_PROVIDER:
            phone = identity.providerAccountId
        elif identity.provider == "wechat":
            wechat_bound = True
    return phone, wechat_bound
