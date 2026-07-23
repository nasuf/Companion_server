from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GameLevelInfo(BaseModel):
    sort_order: int
    stage_name: str
    tier_name: str
    upgrade_points: int
    cumulative_points: int


class GameWalletResponse(BaseModel):
    balance: int
    lifetime_earned: int
    can_play: bool
    daily_grant: int
    convert_floor: int
    convert_rate: int
    convertible: int
    level: GameLevelInfo | None = None
    next_tier: GameLevelInfo | None = None
    # Net points settled for a specific game; only populated when the request
    # scopes to a game_key (per-game display on each game screen).
    game_points_for_game: int | None = None


class GamePointConvertRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    amount: int = Field(gt=0)


class GamePointConvertResponse(BaseModel):
    game_balance: int
    shop_point_balance: int
    converted: int
    shop_point_delta: int


# ── Admin ──


class GameLevelTierPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage_name: str = Field(min_length=1, max_length=40)
    tier_name: str = Field(min_length=1, max_length=60)
    upgrade_points: int = Field(ge=0)
    cumulative_points: int = Field(ge=0)


class GameLevelTiersPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tiers: list[GameLevelTierPayload] = Field(min_length=1)


class GamePointRulePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rules: dict[str, Any]


class GamePointRuleResponse(BaseModel):
    game_key: str
    title: str
    rules: dict[str, Any]
