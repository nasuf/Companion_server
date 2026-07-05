"""Aggregated fragment turns must appear as one coherent user message in the
reply prompt, not N separate fragment rows (else the LLM answers only the last).
"""

from __future__ import annotations

from app.services.chat.message_utils import collapse_turn_fragments


def _msgs():
    return [
        {"id": "a0", "role": "assistant", "content": "在的", "createdAt": "2026-07-05T09:00:00+00:00"},
        {"id": "u1", "role": "user", "content": "我", "createdAt": "2026-07-05T10:00:00+00:00"},
        {"id": "u2", "role": "user", "content": "喜欢", "createdAt": "2026-07-05T10:00:01+00:00"},
        {"id": "u3", "role": "user", "content": "你", "createdAt": "2026-07-05T10:00:02+00:00"},
    ]


def test_collapses_fragments_into_one_turn():
    out = collapse_turn_fragments(
        _msgs(),
        turn_message_ids={"u1", "u2", "u3"},
        combined_text="我喜欢你",
        combined_id="u3",
    )
    # assistant kept + single combined user message
    assert [m["role"] for m in out] == ["assistant", "user"]
    combined = out[-1]
    assert combined["content"] == "我喜欢你"
    assert combined["id"] == "u3"
    assert combined["coalesced_turn"] is True
    # latest fragment timestamp preserved
    assert combined["createdAt"] == "2026-07-05T10:00:02+00:00"


def test_single_fragment_is_noop():
    msgs = _msgs()
    out = collapse_turn_fragments(
        msgs, turn_message_ids={"u3"}, combined_text="你", combined_id="u3",
    )
    assert out == msgs


def test_empty_combined_text_is_noop():
    msgs = _msgs()
    out = collapse_turn_fragments(
        msgs, turn_message_ids={"u1", "u2", "u3"}, combined_text="  ", combined_id="u3",
    )
    assert out == msgs


def test_no_matching_ids_is_noop():
    msgs = _msgs()
    out = collapse_turn_fragments(
        msgs, turn_message_ids={"x1", "x2"}, combined_text="whatever", combined_id="x2",
    )
    assert out == msgs


def test_preserves_order_and_other_messages():
    msgs = _msgs()
    out = collapse_turn_fragments(
        msgs, turn_message_ids={"u1", "u2", "u3"},
        combined_text="我喜欢你", combined_id="u3",
    )
    # the assistant message stays first, combined user turn appended last
    assert out[0]["id"] == "a0"
    assert out[-1]["content"] == "我喜欢你"
    assert len(out) == 2
