"""life_story.convert_profile_to_memories v2 测试: life_events / emotion_events
数组展开为多条记忆 + occur_time 策略 + v2 新字段映射."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.services.life_story import (
    convert_profile_to_memories,
    _LIFE_EVENT_SUB_MAP,
    _EMOTION_EVENT_SUB_MAP,
    _LIFE_EVENT_TIME_RANGE,
    _random_past_time,
)


def _profile(**overrides) -> dict:
    base = {
        "identity": {"gender": "女", "age": 25, "name": "李小雨"},
    }
    base.update(overrides)
    return base


def test_life_events_each_scene_becomes_separate_memory():
    """life_events 每个 tag 元素 → 1 条独立 memories_ai."""
    profile = _profile(life_events={
        "education": ["初中遇到改变人生的语文老师", "高考前两周突发肠胃炎", "大三那年挂了一门必修"],
        "work": ["第一次独立承接项目"],
    })
    memories = convert_profile_to_memories(profile, None)
    edu_mems = [m for m in memories if m["sub_category"] == "教育" and m["main_category"] == "生活"]
    work_mems = [m for m in memories if m["sub_category"] == "工作" and m["main_category"] == "生活"]
    assert len(edu_mems) == 3
    assert len(work_mems) == 1


def test_life_event_memories_have_past_occur_time():
    profile = _profile(life_events={"education": ["A 事件"]})
    memories = convert_profile_to_memories(profile, None)
    edu = next(m for m in memories if m["sub_category"] == "教育")
    assert "occur_time" in edu
    assert edu["occur_time"] < datetime.now(UTC)


def test_life_event_education_occur_time_in_past_3_to_12_years():
    profile = _profile(life_events={"education": ["事件"]})
    memories = convert_profile_to_memories(profile, None)
    edu = next(m for m in memories if m["sub_category"] == "教育")
    now = datetime.now(UTC)
    years_ago = (now - edu["occur_time"]).days / 365.25
    # 3-12 年前
    assert 2.9 < years_ago < 12.1


def test_emotion_events_15_subs_all_mapped():
    """15 个情绪类都能正确生成记忆."""
    profile = _profile(emotion_events={key: [f"情境{key}"] for key in _EMOTION_EVENT_SUB_MAP})
    memories = convert_profile_to_memories(profile, None)
    emo_subs = {m["sub_category"] for m in memories if m["main_category"] == "情绪"}
    expected = set(_EMOTION_EVENT_SUB_MAP.values())
    assert emo_subs == expected


def test_emotion_event_memories_have_occur_time():
    profile = _profile(emotion_events={"happy": ["A", "B"]})
    memories = convert_profile_to_memories(profile, None)
    happy = [m for m in memories if m["sub_category"] == "高兴"]
    assert len(happy) == 2
    for m in happy:
        assert "occur_time" in m
        assert m["occur_time"] < datetime.now(UTC)


def test_v2_new_identity_fields_create_memories():
    profile = _profile(identity={
        "gender": "女", "age": 25, "name": "李小雨",
        "zodiac": "蛇", "constellation": "双鱼座",
        "social_relations": "朋友 3-5 个真心",
        "pet_profile": "一只猫名叫团子",
    })
    memories = convert_profile_to_memories(profile, None)
    sub_to_mem = {m["sub_category"]: m for m in memories if m["main_category"] == "身份"}
    assert "姓名" in sub_to_mem
    assert "生肖" in sub_to_mem
    assert "星座" in sub_to_mem
    assert "社会关系" in sub_to_mem
    assert "宠物" in sub_to_mem


def test_pet_profile_no_pet_skips_memory():
    """pet_profile = 「无」时不生成宠物记忆."""
    profile = _profile(identity={
        "gender": "女", "age": 25, "name": "X", "pet_profile": "无",
    })
    memories = convert_profile_to_memories(profile, None)
    pet = [m for m in memories if m["sub_category"] == "宠物"]
    assert len(pet) == 0


def test_v2_new_values_fields_create_memories():
    """worldview / interpersonal_view / social_view / faith 都能生成记忆."""
    profile = _profile(values={
        "motto": "活出自己",
        "worldview": "善意循环",
        "interpersonal_view": "亲情至上",
        "social_view": "关注弱势",
        "faith": "艺术是寄托",
    })
    memories = convert_profile_to_memories(profile, None)
    sub_to_mem = {m["sub_category"]: m for m in memories if m["main_category"] == "思维"}
    assert "人生观" in sub_to_mem  # motto
    assert "世界观" in sub_to_mem
    assert "人际关系观" in sub_to_mem
    assert "社会观点" in sub_to_mem
    assert "信仰/寄托" in sub_to_mem


def test_v2_interpersonal_lifestyle_taboo_fields():
    profile = _profile(
        interpersonal={"liked_traits": ["真诚"], "disliked_traits": ["虚伪"]},
        lifestyle={"routine": "晚睡早起", "hygiene": "洁癖", "leisure": "看书"},
        taboo={"items": ["说谎", "迟到"]},
    )
    memories = convert_profile_to_memories(profile, None)
    sub_to_mems = {}
    for m in memories:
        if m["main_category"] == "偏好":
            sub_to_mems.setdefault(m["sub_category"], []).append(m)
    assert "人际喜好" in sub_to_mems
    assert "人际厌恶" in sub_to_mems
    assert "生活习惯" in sub_to_mems
    assert "禁忌/雷区" in sub_to_mems
    assert len(sub_to_mems["禁忌/雷区"]) >= 2  # 2 个 taboo items


def test_old_schema_fears_compat_still_maps_to_taboo():
    """旧 schema 的 fears 分类仍然能转记忆 (历史 profile_data 兼容)."""
    profile = _profile(fears={"animals": ["蛇"], "objects": ["针"]})
    memories = convert_profile_to_memories(profile, None)
    taboo_mems = [m for m in memories if m["sub_category"] == "禁忌/雷区"]
    assert len(taboo_mems) >= 2


def test_random_past_time_within_range():
    now = datetime.now(UTC)
    for _ in range(20):
        ts = _random_past_time(2.0, 5.0)
        years_ago = (now - ts).days / 365.25
        assert 1.99 < years_ago < 5.01


def test_life_event_sub_map_covers_taxonomy_life_subs():
    """life_events schema 11 个字段映射到 taxonomy 生活的 11 个核心子类."""
    expected = {
        "交互", "教育", "工作", "旅行", "居住", "健康", "宠物",
        "人际", "技能", "生活", "其他特殊事件",
    }
    assert set(_LIFE_EVENT_SUB_MAP.values()) == expected


def test_emotion_event_sub_map_covers_15_emotions():
    expected = {
        "高兴", "悲伤", "愤怒", "恐惧", "厌恶", "焦虑", "失望", "自豪",
        "感动", "尴尬", "遗憾", "孤独", "惊讶", "感激", "释怀",
    }
    assert set(_EMOTION_EVENT_SUB_MAP.values()) == expected


def test_life_event_time_ranges_all_in_past():
    """所有时间区间 min_years_ago >= 0."""
    for field, (lo, hi) in _LIFE_EVENT_TIME_RANGE.items():
        assert lo >= 0, f"{field} has negative min"
        assert hi > lo, f"{field} hi <= lo"
