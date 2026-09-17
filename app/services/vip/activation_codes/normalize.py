from __future__ import annotations

import re

_CODE_RE = re.compile(r"[^A-Z0-9]")


def strip_code_separators(raw: str) -> str:
    """Uppercase alphanumeric only (for search / partial match)."""
    return _CODE_RE.sub("", (raw or "").upper())


def normalize_code(raw: str) -> str:
    """Uppercase alphanumeric only (strip separators/spaces)."""
    cleaned = strip_code_separators(raw)
    if not cleaned:
        raise ValueError("empty_code")
    return cleaned


def format_code_display(normalized: str) -> str:
    """Human-readable activation code (XXXX-XXXX for new 8-char codes)."""
    code = strip_code_separators(normalized)
    if not code:
        return normalized or ""

    if len(code) == 8:
        return f"{code[:4]}-{code[4:]}"

    # Legacy VIP-prefixed codes stored as VIP + 8 alnum (11 chars total).
    if code.startswith("VIP") and len(code) == 11:
        body = code[3:]
        return f"VIP-{body[:4]}-{body[4:]}"

    return code
