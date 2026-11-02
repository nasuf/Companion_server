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


def test_current_message_sent_before_previous_reply_moves_to_the_end():
    """第二句在上一轮还没回完时就到了: 不能让 prompt 以 AI 自己的回复收尾 (2026-11-02 生产)。"""
    msgs = [
        {"id": "u1", "role": "user", "content": "转塘是哪里啊", "createdAt": "2026-11-02T09:51:16+00:00"},
        {"id": "u2", "role": "user", "content": "不过听起来是看日落的好地方", "createdAt": "2026-11-02T09:51:20+00:00"},
        {"id": "a1", "role": "assistant", "content": "是杭州西湖区的一个街道呀", "createdAt": "2026-11-02T09:51:29+00:00"},
        {"id": "a2", "role": "assistant", "content": "有好多好看的日落点", "createdAt": "2026-11-02T09:51:29+00:00"},
    ]
    out = collapse_turn_fragments(msgs, turn_message_ids={"u2"}, combined_text="不过听起来是看日落的好地方",
                                  combined_id="u2")
    assert [m["id"] for m in out] == ["u1", "a1", "a2", "u2"]


def test_not_reordered_when_a_later_user_message_follows():
    """下一条用户消息已经入库: 不是 "回复晚于本条" 的竞态, 不挪, 免得新旧两句倒过来合并。"""
    msgs = [
        {"id": "a0", "role": "assistant", "content": "在吗", "createdAt": "2026-11-02T09:00:00+00:00"},
        {"id": "u1", "role": "user", "content": "刚下班", "createdAt": "2026-11-02T09:01:00+00:00"},
        {"id": "u2", "role": "user", "content": "今天好累啊你呢", "createdAt": "2026-11-02T09:01:30+00:00"},
    ]
    assert collapse_turn_fragments(msgs, turn_message_ids={"u1"}, combined_text="刚下班",
                                   combined_id="u1") == msgs


def test_not_reordered_for_a_sub_intent_fragment():
    """子意图: 末尾补的是合成的子片段, 父消息不能挪到它后面。"""
    msgs = [
        {"id": "u1", "role": "user", "content": "我好难过，不想聊了", "createdAt": "2026-11-02T09:00:00+00:00"},
        {"id": "a1", "role": "assistant", "content": "怎么啦", "createdAt": "2026-11-02T09:00:05+00:00"},
        {"id": None, "role": "user", "content": "不想聊了", "createdAt": "2026-11-02T09:00:00+00:00",
         "synthetic_current": True},
    ]
    assert collapse_turn_fragments(msgs, turn_message_ids={"u1"}, combined_text="不想聊了",
                                   combined_id=None) == msgs
