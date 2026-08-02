from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services import game_points


def _tier(order: int, stage: str, caption: str, colour: str, points: int) -> dict:
    return {
        "sort_order": order,
        "stage_name": stage,
        "stage_caption": caption,
        "tier_name": colour,
        "cumulative_points": points,
    }


def _tiers() -> list[dict]:
    # A trimmed ladder mirroring the seed shape.
    return [
        _tier(1, "皮革手套", "初学起步", "白", 0),
        _tier(2, "皮革手套", "初学起步", "绿", 50),
        _tier(3, "皮革手套", "初学起步", "黄", 150),
        _tier(6, "尼龙手套", "进阶提升", "白", 750),
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


def test_levels_endpoint_returns_the_ladder(api_client, auth_header):
    ladder = [
        {
            "sort_order": 1,
            "stage_name": "皮革手套",
            "stage_caption": "初学起步",
            "tier_name": "白",
            "upgrade_points": 0,
            "cumulative_points": 0,
        }
    ]
    with patch(
        "app.api.public.game_points.game_points.list_level_tiers",
        new_callable=AsyncMock,
        return_value=ladder,
    ):
        response = api_client.get("/game-wallet/levels", headers=auth_header("u1"))
    assert response.status_code == 200
    assert response.json() == ladder


def test_levels_endpoint_requires_a_signed_in_user(api_client):
    assert api_client.get("/game-wallet/levels").status_code in (401, 403)
