"""Validate forwarded JPush configuration without printing credential values."""
from __future__ import annotations

import os
import re


def validate(environ: dict[str, str]) -> None:
    enabled = environ.get("JPUSH_ENABLED", "false").lower()
    if enabled not in {"true", "false"}:
        raise ValueError("JPUSH_ENABLED must be true or false")
    formats = {
        "JIGUANG_PUSH_APP_KEY": r"[0-9a-fA-F]{24}",
        "JIGUANG_PUSH_MASTER_SECRET": r"[A-Za-z0-9_-]{16,128}",
        "JIGUANG_PUSH_PACKAGE_NAME": r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)+",
    }
    for name, pattern in formats.items():
        value = environ.get(name, "")
        if (enabled == "true" or value) and not re.fullmatch(pattern, value):
            raise ValueError(f"{name} is missing or invalid; values redacted")


if __name__ == "__main__":
    validate(dict(os.environ))
    print("JPush deployment configuration passed (values redacted)")
