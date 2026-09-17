from __future__ import annotations

import secrets
import string

_ALPHABET = string.ascii_uppercase + string.digits
# Crockford-ish: omit I/O to reduce confusion in printed codes.
_ALPHABET = "".join(c for c in _ALPHABET if c not in "IO")


def _mixed_segment(length: int) -> str:
    """Segment with at least one letter and one digit."""
    while True:
        seg = "".join(secrets.choice(_ALPHABET) for _ in range(length))
        if any(c.isalpha() for c in seg) and any(c.isdigit() for c in seg):
            return seg


def generate_code_string(*, segment_len: int = 4) -> str:
    left = _mixed_segment(segment_len)
    right = _mixed_segment(segment_len)
    return f"{left}-{right}"
