from __future__ import annotations

import pytest

from app.services import game_points


def _tiers() -> list[dict[str, int]]:
    # A trimmed ladder mirroring the seed shape.
    return [
        {"sort_order": 1, "stage_name": "白手套", "tier_name": "1 阶", "cumulative_points": 0},
        {"sort_order": 2, "stage_name": "白手套", "tier_name": "2 阶", "cumulative_points": 50},
        {"sort_order": 3, "stage_name": "白手套", "tier_name": "3 阶", "cumulative_points": 150},
        {"sort_order": 6, "stage_name": "蓝手套", "tier_name": "1 阶", "cumulative_points": 750},
    ]


def test_resolve_level_picks_highest_reached_tier():
    level, nxt = game_points._resolve_level(160, _tiers())
    assert level["cumulative_points"] == 150
    assert nxt["cumulative_points"] == 750


def test_resolve_level_below_first_threshold_maps_to_first_tier():
    level, nxt = game_points._resolve_level(0, _tiers())
    assert level["sort_order"] == 1
    assert nxt["cumulative_points"] == 50


def test_resolve_level_at_max_has_no_next_tier():
    level, nxt = game_points._resolve_level(10_000, _tiers())
    assert level["cumulative_points"] == 750
    assert nxt is None


def test_outcome_delta_maps_quit_to_aborted():
    rules = {"win": 4, "lose": -3, "draw": 0, "quit": -3}
    assert game_points._outcome_delta(rules, "win") == 4
    assert game_points._outcome_delta(rules, "lose") == -3
    assert game_points._outcome_delta(rules, "aborted") == -3
    assert game_points._outcome_delta(rules, "draw") == 0


def test_milestone_delta_awards_highest_reached_tile():
    rules = {
        "type": "milestone",
        "milestones": [
            {"tile": 128, "points": 2},
            {"tile": 256, "points": 5},
            {"tile": 512, "points": 6},
            {"tile": 1024, "points": 15},
            {"tile": 2048, "points": 25},
        ],
        "quit_below_threshold": {"threshold": 128, "below": -2, "at_or_above": 0},
    }
    assert game_points._milestone_delta(rules, "win", 2048) == 25
    assert game_points._milestone_delta(rules, "lose", 1024) == 15
    assert game_points._milestone_delta(rules, "lose", 200) == 2
    assert game_points._milestone_delta(rules, "lose", 64) == 0
    # Quit uses the threshold rule, not the milestone ladder.
    assert game_points._milestone_delta(rules, "aborted", 64) == -2
    assert game_points._milestone_delta(rules, "aborted", 256) == 0


def test_validate_rules_outcome_rejects_non_int():
    with pytest.raises(ValueError):
        game_points._validate_rules({"type": "outcome", "win": "x"})


def test_validate_rules_outcome_roundtrip_keeps_pending_flag():
    cleaned = game_points._validate_rules(
        {"type": "outcome", "win": 3, "lose": -2, "draw": 0, "quit": -2, "pending_pm": True}
    )
    assert cleaned == {
        "type": "outcome",
        "win": 3,
        "lose": -2,
        "draw": 0,
        "quit": -2,
        "pending_pm": True,
    }


def test_validate_rules_milestone_requires_ascending_tiles():
    with pytest.raises(ValueError):
        game_points._validate_rules(
            {
                "type": "milestone",
                "milestones": [
                    {"tile": 256, "points": 5},
                    {"tile": 128, "points": 2},
                ],
            }
        )


def test_validate_rules_rejects_unknown_type():
    with pytest.raises(ValueError):
        game_points._validate_rules({"type": "mystery"})
