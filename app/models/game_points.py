from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GameLevelInfo(BaseModel):
    sort_order: int
    # `皮革手套` — the glove itself.
    stage_name: str
    # `初学起步` — the descriptive line clients show under the name.
    stage_caption: str = ""
    # `白` — the colour that ranks this step inside the stage.
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
    # This game's scoring rules, so the client can show the real value of a
    # round on the result screen. Only populated alongside a game_key.
    rules: dict[str, Any] | None = None


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
    stage_caption: str = Field(default="", max_length=40)
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


class GamePointGrantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1)
    # Positive adds balance (official grant); the level is never changed.
    amount: int = Field(gt=0)
    note: str | None = Field(default=None, max_length=200)


class GamePointGrantResponse(BaseModel):
    user_id: str
    balance: int
    delta: int


class AdminUserSearchItem(BaseModel):
    user_id: str
    username: str
    nickname: str | None = None


class AdminGamePointLedgerItem(BaseModel):
    id: str
    user_id: str
    username: str | None = None
    nickname: str | None = None
    delta: int
    balance_after: int
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    lifetime_after: int
    level_name: str | None = None
    level_up: bool = False
