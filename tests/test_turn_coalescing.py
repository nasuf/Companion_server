"""Turn-level semantic coalescing tests."""

from __future__ import annotations

from app.services.interaction.turn_coalescing import coalesce_turn_messages


def test_coalesces_rewritten_ai_age_query():
    result = coalesce_turn_messages(["你多大了？", "几岁？"])

    assert result.texts == ["你多大了？"]
    assert result.combined_text == "你多大了？"
    assert result.metadata["coalesced_count"] == 1
    assert result.metadata["dropped"][0]["text"] == "几岁？"
    assert "fact_query|ai|identity|age" in result.metadata["dropped"][0]["signature"]


def test_coalesces_rewritten_stable_profile_query_beyond_age():
    result = coalesce_turn_messages(["你高中在哪读的？", "哪个学校？"])

    assert result.texts == ["你高中在哪读的？"]
    assert result.metadata["coalesced_count"] == 1
    assert "fact_query|ai|identity|education" in result.metadata["dropped"][0]["signature"]


def test_coalesces_rewritten_preference_query_with_same_slot():
    result = coalesce_turn_messages(["你喜欢听什么歌？", "喜欢什么音乐？"])

    assert result.texts == ["你喜欢听什么歌？"]
    assert result.metadata["coalesced_count"] == 1
    assert "fact_query|ai|preference|music" in result.metadata["dropped"][0]["signature"]


def test_keeps_distinct_identity_slots():
    result = coalesce_turn_messages(["你多大了？", "生日哪天？"])

    assert result.texts == ["你多大了？", "生日哪天？"]
    assert result.metadata is None


def test_keeps_distinct_identity_subjects():
    result = coalesce_turn_messages(["你叫什么？", "我叫什么？"])

    assert result.texts == ["你叫什么？", "我叫什么？"]
    assert result.metadata is None


def test_keeps_distinct_profile_domains():
    result = coalesce_turn_messages(["你喜欢什么颜色？", "你喜欢听什么歌？"])

    assert result.texts == ["你喜欢什么颜色？", "你喜欢听什么歌？"]
    assert result.metadata is None


def test_keeps_context_question_with_profile_topic_words():
    result = coalesce_turn_messages(["你刚才说《小森林》是电影吗？", "那是电影吗？"])

    assert result.texts == ["你刚才说《小森林》是电影吗？", "那是电影吗？"]
    assert result.metadata is None


def test_coalesces_schedule_followup_without_new_time_scope():
    result = coalesce_turn_messages(["你明天忙吗？", "那有空吗？"])

    assert result.texts == ["你明天忙吗？"]
    assert result.metadata["coalesced_count"] == 1
    assert result.metadata["dropped"][0]["text"] == "那有空吗？"
    assert "schedule_query|ai|schedule|availability" in result.metadata["dropped"][0]["signature"]


def test_keeps_schedule_followup_with_new_time_scope():
    result = coalesce_turn_messages(["你明天忙吗？", "后天呢？"])

    assert result.texts == ["你明天忙吗？", "后天呢？"]
    assert result.metadata is None


def test_coalesces_current_state_variants():
    result = coalesce_turn_messages(["在忙吗", "干嘛呢"])

    assert result.texts == ["在忙吗"]
    assert result.metadata["coalesced_count"] == 1
    assert result.metadata["dropped"][0]["text"] == "干嘛呢"


def test_coalesces_current_busy_followup_without_subject():
    result = coalesce_turn_messages(["你忙吗现在", "忙啥呢"])

    assert result.texts == ["你忙吗现在"]
    assert result.metadata["coalesced_count"] == 1
    assert result.metadata["dropped"][0]["text"] == "忙啥呢"


def test_keeps_write_actions_unmodified():
    result = coalesce_turn_messages(["明天提醒我喝水", "提醒我喝水"])

    assert result.texts == ["明天提醒我喝水", "提醒我喝水"]
    assert result.metadata is None
