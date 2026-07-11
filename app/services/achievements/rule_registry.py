"""Canonical execution registry for every product achievement."""

from __future__ import annotations

from dataclasses import dataclass

from app.services.achievements.definitions import ACHIEVEMENT_BY_ID


@dataclass(frozen=True, slots=True)
class AchievementRuleSpec:
    id: int
    source: str
    timing: str
    evaluator: str
    enabled: bool = True


DISABLED_ACHIEVEMENT_IDS = frozenset({
    3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 22, 24, 32, 34, 40,
})


def _rule(
    achievement_id: int,
    source: str,
    timing: str,
    evaluator: str,
) -> AchievementRuleSpec:
    return AchievementRuleSpec(
        id=achievement_id,
        source=source,
        timing=timing,
        evaluator=evaluator,
        enabled=achievement_id not in DISABLED_ACHIEVEMENT_IDS,
    )


ACHIEVEMENT_RULES: dict[int, AchievementRuleSpec] = {
    1: _rule(1, "user_message", "realtime", "first_user_message"),
    2: _rule(2, "daily_rollup", "day_end", "single_message_day"),
    3: _rule(3, "user_message", "disabled", "disabled"),
    4: _rule(4, "user_message", "disabled", "disabled"),
    5: _rule(5, "daily_rollup", "rolling_48h", "unique_messages_48h"),
    6: _rule(6, "daily_rollup", "three_day_streak", "evening_message_streak"),
    7: _rule(7, "daily_rollup", "day_end", "midday_only"),
    8: _rule(8, "user_message", "realtime", "four_short_messages"),
    9: _rule(9, "user_message", "realtime", "exact_haha"),
    10: _rule(10, "memory", "disabled", "disabled"),
    11: _rule(11, "memory", "disabled", "disabled"),
    12: _rule(12, "memory", "disabled", "disabled"),
    13: _rule(13, "memory", "disabled", "disabled"),
    14: _rule(14, "user_message", "disabled", "disabled"),
    15: _rule(15, "intent", "disabled", "disabled"),
    16: _rule(16, "memory", "disabled", "disabled"),
    17: _rule(17, "user_message", "disabled", "disabled"),
    18: _rule(18, "daily_rollup", "day_end", "twelve_hour_span"),
    19: _rule(19, "user_message", "realtime", "same_first_character"),
    20: _rule(20, "intent", "realtime", "first_schedule_adjustment"),
    21: _rule(21, "user_message", "realtime", "future_plan_query"),
    22: _rule(22, "emotion", "disabled", "disabled"),
    23: _rule(23, "memory", "realtime", "first_sad_memory"),
    24: _rule(24, "assistant_message", "disabled", "disabled"),
    25: _rule(25, "proactive_response", "realtime", "first_proactive_response"),
    26: _rule(26, "daily_rollup", "two_day_streak", "same_daily_opener"),
    27: _rule(27, "user_message", "realtime", "wave_suffix"),
    28: _rule(28, "daily_rollup", "day_end", "message_count_multiple_of_three"),
    29: _rule(29, "memory", "realtime", "first_name_memory"),
    30: _rule(30, "assistant_message", "realtime", "first_slow_reply"),
    31: _rule(31, "user_message", "realtime", "five_questions"),
    32: _rule(32, "user_message", "disabled", "disabled"),
    33: _rule(33, "daily_rollup", "day_end", "twenty_short_messages"),
    34: _rule(34, "aggregation", "disabled", "disabled"),
    35: _rule(35, "user_message", "cumulative_days", "seven_chat_days"),
    36: _rule(36, "daily_rollup", "day_end", "symbol_free_day"),
    37: _rule(37, "user_message", "realtime", "ten_unique_first_characters"),
    38: _rule(38, "user_message", "realtime", "increasing_lengths"),
    39: _rule(39, "intimacy", "realtime", "intimacy_above_400"),
    40: _rule(40, "daily_rollup", "disabled", "disabled"),
    41: _rule(41, "user_message", "realtime", "decreasing_lengths"),
    42: _rule(42, "user_message", "realtime", "long_short_transitions"),
    43: _rule(43, "daily_rollup", "day_end", "matching_first_last_length"),
    44: _rule(44, "daily_rollup", "day_end", "assistant_three_times_longer"),
    45: _rule(45, "daily_rollup", "day_end", "all_even_lengths"),
    46: _rule(46, "daily_rollup", "day_end", "all_odd_lengths"),
    47: _rule(47, "user_message", "realtime", "shared_character_six_messages"),
    48: _rule(48, "schedule_status", "cumulative", "ten_sleep_wakeups"),
    49: _rule(49, "memory", "realtime", "first_goal_memory"),
    50: _rule(50, "daily_rollup", "three_day_streak", "twelve_hour_span_streak"),
    51: _rule(51, "daily_rollup", "two_day_streak", "symbol_free_streak"),
    52: _rule(52, "user_message", "cumulative_days", "late_night_days"),
    53: _rule(53, "user_message", "cumulative_days", "early_morning_days"),
    54: _rule(54, "user_message", "cumulative_days", "fifteen_chat_days"),
    55: _rule(55, "assistant_message", "realtime", "first_memory_proactive"),
    56: _rule(56, "daily_rollup", "seven_day_streak", "sleep_respect_streak"),
    57: _rule(57, "user_message", "realtime", "three_repeated_messages"),
    58: _rule(58, "user_message", "cumulative", "three_late_goodnights"),
    59: _rule(59, "assistant_message", "cumulative", "one_hundred_stickers"),
    60: _rule(60, "user_message", "realtime", "ten_thousand_user_characters"),
    61: _rule(61, "proactive_response", "realtime", "holiday_response"),
    62: _rule(62, "daily_rollup", "day_end", "all_quick_replies"),
    63: _rule(63, "user_message", "realtime", "three_time_windows"),
    64: _rule(64, "user_message", "realtime", "one_hundred_messages"),
    65: _rule(65, "memory", "cumulative", "twenty_preferences"),
    66: _rule(66, "memory", "cumulative", "ten_fears"),
    67: _rule(67, "intimacy", "realtime", "intimacy_above_600"),
    68: _rule(68, "proactive_response", "realtime", "birthday_response"),
    69: _rule(69, "user_message", "realtime", "time_digit_sum_length"),
    70: _rule(70, "user_message", "realtime", "ten_question_messages"),
    71: _rule(71, "user_message", "realtime", "ten_identical_messages"),
    72: _rule(72, "user_message", "cumulative", "fifty_um_messages"),
    73: _rule(73, "user_message", "realtime", "two_hundred_messages"),
    74: _rule(74, "schedule_status", "seven_day_streak", "all_statuses_seven_days"),
    75: _rule(75, "intimacy", "realtime", "intimacy_above_800"),
    76: _rule(76, "user_message", "realtime", "user_message_at_1314"),
    77: _rule(77, "user_message", "cumulative_days", "thirty_chat_days"),
    78: _rule(78, "proactive_response", "realtime", "holiday_birthday_response"),
    79: _rule(79, "assistant_message", "cumulative", "five_hundred_short_replies"),
    80: _rule(80, "daily_rollup", "day_end", "exactly_one_hundred_daily_chars"),
    81: _rule(81, "assistant_turn", "realtime", "exactly_one_hundred_turn_chars"),
    82: _rule(82, "user_message", "realtime", "user_message_at_0520"),
    83: _rule(83, "user_message", "realtime", "ai_birthday_greeting"),
    84: _rule(84, "assistant_message", "realtime", "proactive_at_1314"),
    85: _rule(85, "user_message", "realtime", "three_echo_lengths"),
    86: _rule(86, "aggregation", "cumulative", "fifty_completed_fragment_windows"),
    87: _rule(87, "intent", "cumulative", "fifty_schedule_adjustments"),
    88: _rule(88, "intimacy", "realtime", "maximum_intimacy"),
    89: _rule(89, "proactive_response", "cumulative", "one_hundred_proactive_responses"),
    90: _rule(90, "daily_rollup", "day_end", "mirrored_first_last_times"),
    91: _rule(91, "user_message", "cumulative_days", "ninety_chat_days"),
    92: _rule(92, "proactive_response", "cumulative", "all_proactive_replies_quick"),
    93: _rule(93, "schedule_status", "thirty_day_streak", "all_statuses_thirty_days"),
    94: _rule(94, "user_message", "cumulative", "ten_midnight_edge_messages"),
    95: _rule(95, "user_message", "realtime", "ai_birthday_clock_time"),
    96: _rule(96, "assistant_message", "realtime", "user_birthday_clock_time"),
    97: _rule(97, "user_message", "cumulative_days", "one_hundred_eighty_chat_days"),
}


def validate_rule_registry() -> None:
    definition_ids = set(ACHIEVEMENT_BY_ID)
    rule_ids = set(ACHIEVEMENT_RULES)
    if rule_ids != definition_ids:
        missing = sorted(definition_ids - rule_ids)
        unknown = sorted(rule_ids - definition_ids)
        raise RuntimeError(
            f"Achievement rule registry mismatch: missing={missing}, unknown={unknown}"
        )
    disabled = {
        achievement_id
        for achievement_id, rule in ACHIEVEMENT_RULES.items()
        if not rule.enabled
    }
    if disabled != set(DISABLED_ACHIEVEMENT_IDS):
        raise RuntimeError(
            f"Achievement disabled registry mismatch: {sorted(disabled)}"
        )


validate_rule_registry()
