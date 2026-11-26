"""Per-model input token counts calibrated against production billing.

No offline tokenizer exists for doubao, so counts are a linear fit of billed
prompt tokens on CJK characters, other characters and message count. Fitted
2026-11-25 on the latest 14 days of trace_runs (ChatOpenAI calls; 5-fold
cross-validated relative error, calls of 1000+ billed tokens):

  doubao-seed-character-260628  1141 calls  p5 -3.9%  p95 +4.6%
  qwen3.5-flash                 2894 calls  p5 -11%   p95 +3.1%
  qwen3.5-plus                   103 calls  p5 -0.4%  p95 +0.4% (one prompt shape)

The context selector's estimate_tokens (CJK 1.5, other 0.25) runs 1.3-1.8x the
billed count; budgets still use it so behaviour does not move until someone
decides to re-tune them. Unknown models use a bound no calibrated model
exceeded (billed never passed 0.825 tokens per character). Re-fit when a model
changes: /admin-api/chat-context-policy/token-calibration compares both counts with billing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rate:
    cjk: float
    other: float
    per_message: float
    calibrated: bool = True


RATES = {
    "doubao-seed-character-260628": Rate(0.795, 0.392, 16.6),
    "qwen3.5-flash": Rate(0.491, 0.803, 54.8),
    "qwen3.5-plus": Rate(0.708, 0.365, 16.6),
}
UNCALIBRATED = Rate(1.0, 0.85, 20.0, calibrated=False)


def rate(model: str | None) -> Rate:
    name = (model or "").rsplit("/", 1)[-1]
    return RATES.get(name, UNCALIBRATED)


def _split(text: str) -> tuple[int, int]:
    cjk = sum(1 for c in text if "一" <= c <= "鿿")
    return cjk, len(text) - cjk


def count_text(text: str, model: str | None) -> int:
    """Tokens for a bare text span, without per-message overhead."""
    r = rate(model)
    cjk, other = _split(text or "")
    return round(cjk * r.cjk + other * r.other)


def count_messages(messages: list[dict], model: str | None) -> int:
    """Billed input tokens expected for a chat request with these messages."""
    r = rate(model)
    cjk = other = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            a, b = _split(content)
            cjk, other = cjk + a, other + b
    return round(cjk * r.cjk + other * r.other + len(messages) * r.per_message)
