from __future__ import annotations

import re

_CODE_RE = re.compile(r"[^A-Z0-9]")


def normalize_code(raw: str) -> str:
    """Uppercase alphanumeric only (strip separators/spaces)."""
    cleaned = _CODE_RE.sub("", (raw or "").upper())
    if not cleaned:
        raise ValueError("empty_code")
    return cleaned
