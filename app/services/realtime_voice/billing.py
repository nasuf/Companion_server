"""Token metering for the DashScope realtime model (calls and voice messages).

DashScope prices text and audio tokens separately on both sides, and its
usage block splits them (input_tokens_details / output_tokens_details). Each
model therefore has a sibling price-only registry row, ``<model>:audio``,
holding the audio prices, so the shared cost estimator sums both buckets.
"""

from __future__ import annotations

from typing import Any

AUDIO_PRICE_SUFFIX = ":audio"


def audio_price_key(model: str) -> str:
    return f"{model}{AUDIO_PRICE_SUFFIX}"


def is_price_only(identifier: str) -> bool:
    """A registry row that only carries audio prices, never a selectable model."""
    return identifier.endswith(AUDIO_PRICE_SUFFIX)


def _count(value: Any) -> int:
    return value if isinstance(value, int) and value > 0 else 0


def split_usage(usage: dict) -> dict[str, int]:
    """Text and audio token counts of one response.done usage block.

    Tokens the details do not name (e.g. a missing details block) count as
    audio, the dearer kind, so cost is never underestimated.
    """
    def part(total_key: str, details_key: str) -> tuple[int, int]:
        total = _count(usage.get(total_key))
        details = usage.get(details_key)
        details = details if isinstance(details, dict) else {}
        text = _count(details.get("text_tokens"))
        audio = _count(details.get("audio_tokens"))
        return text, audio + max(0, total - text - audio)

    input_text, input_audio = part("input_tokens", "input_tokens_details")
    output_text, output_audio = part("output_tokens", "output_tokens_details")
    return {
        "input_text": input_text,
        "input_audio": input_audio,
        "output_text": output_text,
        "output_audio": output_audio,
    }


def tokens_by_model(model: str, split: dict[str, int]) -> dict[str, dict[str, int]]:
    return {
        model: {"input": split["input_text"], "output": split["output_text"], "cached_input": 0},
        audio_price_key(model): {
            "input": split["input_audio"], "output": split["output_audio"], "cached_input": 0,
        },
    }


def cost_cny(model: str, split: dict[str, int]) -> float:
    from app.services.llm.pricing import estimate_cost_cny

    return round(sum(
        estimate_cost_cny(name, bucket["input"], bucket["output"])
        for name, bucket in tokens_by_model(model, split).items()
    ), 6)
