from __future__ import annotations

import re

from app.services.vip.activation_codes.generator import generate_code_string
from app.services.vip.activation_codes.normalize import (
    format_code_display,
    normalize_code,
    strip_code_separators,
)


_CODE_PATTERN = re.compile(r"^[A-HJ-NP-Z0-9]{4}-[A-HJ-NP-Z0-9]{4}$")


def test_generate_code_string_format():
    for _ in range(50):
        code = generate_code_string()
        assert _CODE_PATTERN.match(code), code
        normalized = normalize_code(code)
        assert len(normalized) == 8
        assert any(c.isalpha() for c in normalized)
        assert any(c.isdigit() for c in normalized)
        assert "I" not in normalized and "O" not in normalized


def test_format_code_display_new_and_legacy():
    assert format_code_display("ABCD1234") == "ABCD-1234"
    assert format_code_display("ABCD-1234") == "ABCD-1234"
    assert format_code_display("VIPABCD1234") == "VIP-ABCD-1234"


def test_strip_code_separators_for_search():
    assert strip_code_separators("abcd-1234") == "ABCD1234"
    assert strip_code_separators("VIP-ABCD-1234") == "VIPABCD1234"
