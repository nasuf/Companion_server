from __future__ import annotations

import base64
import hashlib
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from app.config import settings

_PREFIX = "enc:v1:"


def _raw_key() -> str:
    return settings.last_will_encryption_key.strip() or settings.jwt_secret.strip()


def _fernet() -> Fernet | None:
    raw = _raw_key()
    if not raw:
        return None
    key = base64.urlsafe_b64encode(hashlib.sha256(raw.encode("utf-8")).digest())
    return Fernet(key)


def protect_text(value: str | None) -> str:
    text = value or ""
    if not text or text.startswith(_PREFIX):
        return text
    fernet = _fernet()
    if fernet is None:
        return text
    token = fernet.encrypt(text.encode("utf-8")).decode("ascii")
    return f"{_PREFIX}{token}"


def reveal_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if not text.startswith(_PREFIX):
        return text
    fernet = _fernet()
    if fernet is None:
        raise RuntimeError("Last-will encryption key is required to read encrypted data")
    token = text[len(_PREFIX) :].encode("ascii")
    try:
        return fernet.decrypt(token).decode("utf-8")
    except InvalidToken as exc:
        raise RuntimeError("Last-will encrypted payload could not be decrypted") from exc


def protect_contact(contact: dict[str, Any]) -> dict[str, Any]:
    protected: dict[str, Any] = {}
    for key in ("name", "email", "phone"):
        value = contact.get(key)
        if value is not None:
            protected[key] = protect_text(str(value))
    return protected


def reveal_contact(contact: Any) -> dict[str, Any]:
    contact = getattr(contact, "data", contact)
    if not isinstance(contact, dict):
        return {}
    revealed: dict[str, Any] = {}
    for key in ("name", "email", "phone"):
        value = contact.get(key)
        if value is not None:
            revealed[key] = reveal_text(value)
    return revealed
