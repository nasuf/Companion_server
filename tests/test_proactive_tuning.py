"""后台「聊天管理 → 主动交流」可调参数: 登记表、校验、运行时读取."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.services.proactive import gates, policy, tuning

UTC = timezone.utc


@pytest.fixture
def set_overrides(monkeypatch):
    """模拟后台保存: 每次换一份新的覆盖 dict (生产里配置刷新也是整体替换, 从不就地改)."""
    def _set(**values):
        snapshot = dict(values)
        monkeypatch.setattr(
            "app.services.runtime_config.proactive_policy_overrides", lambda: snapshot,
        )
    _set()
    return _set


# ── 默认值 = 上线前的硬编码, 没有覆盖时行为不变 ─────────────────────────

def test_defaults_match_previous_hard_coded_behaviour():
    d = tuning.DEFAULTS
    assert [d[f"a_window_{i}_rate"] for i in range(1, 5)] == [0.05, 0.12, 0.25, 0.35]
    assert d["a_window_rate_cap"] == 0.50
    assert (d["trigger_silence_wakeup"], d["trigger_memory_proactive"], d["trigger_scheduled_scene"]) \
        == (0.30, 0.30, 0.40)
    assert (d["fallback_silence_wakeup"], d["fallback_memory_proactive"]) == (0.5, 0.5)
    assert (d["music_probability"], d["trending_interest_first"], d["recent_dialogue_care_followup"]) \
        == (0.04, 0.5, 0.5)
    assert (d["followup_judge_delay_minutes"], d["followup_max_per_session"],
            d["followup_respect_active_hours"], d["return_line_min_gap_minutes"]) == (5, 1, True, 10)
    assert (d["active_hour_start"], d["active_hour_end"], d["cooldown_minutes"], d["daily_max"]) \
        == (8, 22, 30, 3)


def test_default_source_tables_match_previous_tables():
    positive = lambda dist: {k: v for k, v in dist.items() if v > 0}  # noqa: E731
    assert positive(tuning.default_distribution("silence", "cold")) == {
        "greeting": 0.70, "ai_l1": 0.20, "ai_schedule": 0.10,
    }
    assert positive(tuning.default_distribution("silence", "intimate")) == {
        "greeting": 0.50, "ai_l1": 0.05, "ai_l2": 0.05, "ai_schedule": 0.10,
        "user_l1": 0.10, "user_l2": 0.10, "recent_dialogue": 0.10,
    }
    assert positive(tuning.default_distribution("memory", "cold")) == {"ai_l1": 1.0}
    assert positive(tuning.default_distribution("memory", "warming")) == {
        "ai_l1": 0.50, "ai_l2": 0.05, "user_l1": 0.10, "user_l2": 0.10,
        "relationship": 0.10, "recent_dialogue": 0.15,
    }


def test_registry_is_complete_and_well_formed():
    keys = [p.key for p in tuning.PARAMS]
    assert len(keys) == len(set(keys))
    group_keys = {g.key for g in tuning.GROUPS}
    for p in tuning.PARAMS:
        assert p.group in group_keys and p.label and p.description
        assert p.minimum <= float(p.default) <= p.maximum or p.kind == "bool"
    for g in tuning.SUM_GROUPS:
        members = [p for p in tuning.PARAMS if p.sum_group == g.key]
        assert members and sum(float(p.default) for p in members) == pytest.approx(1.0)
    assert tuning.resolve(None) == tuning.DEFAULTS


# ── 后台保存的校验 ───────────────────────────────────────────────────

def test_patch_stores_only_values_that_differ_from_default():
    stored = tuning.apply_patch({}, {"a_window_1_rate": 0.5, "daily_max": 3})
    assert stored == {"a_window_1_rate": 0.5}
    assert tuning.apply_patch(stored, {"a_window_1_rate": None}) == {}


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"nope": 1}, "未知参数"),
        ({"a_window_1_rate": 1.5}, "0%–100%"),
        ({"daily_max": 2.5}, "整数"),
        ({"daily_max": True}, "数字"),
        ({"followup_respect_active_hours": 1}, "开关"),
        ({"trigger_scheduled_scene": 0.5}, "合计为 110.0%"),
        ({"active_hour_start": 22}, "开始必须早于结束"),
    ],
)
def test_patch_rejects_invalid_values(patch, message):
    with pytest.raises(tuning.TuningError) as err:
        tuning.apply_patch({}, patch)
    assert message in str(err.value)


def test_a_whole_source_table_can_be_rebalanced_in_one_patch():
    patch = {f"silence_cold_{s}": 0.0 for s, _, _ in tuning._SILENCE_SOURCES}
    patch.update({"silence_cold_greeting": 0.5, "silence_cold_recent_dialogue": 0.5})
    stored = tuning.apply_patch({}, patch)
    assert tuning.resolve(stored)["silence_cold_recent_dialogue"] == 0.5


# ── 坏数据绝不让主动交流停摆 ─────────────────────────────────────────

def test_resolve_falls_back_on_bad_stored_data():
    values = tuning.resolve({
        "a_window_1_rate": "lots",            # 类型不对 → 这一项回落默认
        "trigger_scheduled_scene": 0.9,        # 表合计不为 100% → 整张表回落
        "active_hour_start": 23,               # 早于结束不成立 → 两项回落
        "removed_param": 1,                    # 登记表删掉的旧参数 → 忽略
    })
    assert values["a_window_1_rate"] == 0.05
    assert values["trigger_scheduled_scene"] == 0.40
    assert (values["active_hour_start"], values["active_hour_end"]) == (8, 22)


def test_current_follows_override_changes(set_overrides):
    assert tuning.value("daily_max") == 3
    set_overrides(daily_max=5)
    assert tuning.value("daily_max") == 5
    set_overrides()
    assert tuning.value("daily_max") == 3


# ── 各读取点确实读的是后台值 ─────────────────────────────────────────

def _state(window: int):
    return SimpleNamespace(current_window_index=window)


def test_window_rate_and_cap_are_tunable(set_overrides):
    set_overrides(a_window_1_rate=0.9, a_window_rate_cap=0.6)
    _, rate = policy.should_hit_window(_state(1))
    assert rate == 0.6
    _, rate = policy.should_hit_window(_state(0))  # B 模式判定窗永远不参与概率
    assert rate == 0.0


def test_trigger_split_is_tunable(set_overrides):
    set_overrides(
        trigger_silence_wakeup=0.0, trigger_memory_proactive=0.0, trigger_scheduled_scene=1.0,
        fallback_silence_wakeup=0.0, fallback_memory_proactive=1.0,
    )
    assert {policy.select_trigger_type() for _ in range(50)} == {"scheduled_scene"}
    assert {policy.fallback_trigger_type() for _ in range(50)} == {"memory_proactive"}


def test_source_tables_are_tunable_per_stage(set_overrides):
    table = {f"silence_cold_{s}": 0.0 for s, _, _ in tuning._SILENCE_SOURCES}
    set_overrides(**{**table, "silence_cold_recent_dialogue": 1.0})
    assert {policy.select_topic_source("p1_cold", "silence_wakeup") for _ in range(30)} == {"recent_dialogue"}
    # 其他阶段不受影响
    picks = Counter(policy.select_topic_source("intimate", "silence_wakeup") for _ in range(400))
    assert picks["greeting"] > 100


def test_active_hours_are_tunable(set_overrides):
    late = datetime(2026, 10, 17, 15, 30, tzinfo=UTC)  # 北京时间 23:30
    assert gates.is_in_active_hours(late) is False
    set_overrides(active_hour_end=24)
    assert gates.is_in_active_hours(late) is True


def test_weighted_choice_never_returns_a_zero_weight_option(monkeypatch):
    monkeypatch.setattr(policy.random, "random", lambda: 0.9999999999999999)
    assert policy._weighted_choice({"greeting": 0.0, "ai_l1": 0.3, "user_l1": 0.7}) == "user_l1"
