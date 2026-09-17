from __future__ import annotations

import secrets
import string

_ALPHABET = string.ascii_uppercase + string.digits
# Crockford-ish: omit I/O to reduce confusion in printed codes.
_ALPHABET = "".join(c for c in _ALPHABET if c not in "IO")


def generate_code_string(*, segments: int = 2, segment_len: int = 4) -> str:
    parts = [
        "".join(secrets.choice(_ALPHABET) for _ in range(segment_len))
        for _ in range(segments)
    ]
    return f"VIP-{'-'.join(parts)}"
