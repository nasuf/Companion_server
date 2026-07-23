from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


GameDifficulty = Literal["newbie", "normal", "hard"]
GamePlayMode = Literal["versus", "cooperate"]


class GamePlayerInfo(BaseModel):
    uid: str
    nick_name: str
    avatar_url: str = ""
    gender: str = ""
    is_ai: int = 0
    ai_level: int = 0


class GameSessionRow(BaseModel):
    """Raw representation of a `game_sessions` row shared by the session-support
    helpers before it is projected into a `NativeSessionResponse`."""

    id: str
    provider: str = "native"
    game_key: str | None = None
    status: str
    user_id: str
    agent_id: str
    workspace_id: str | None = None
    conversation_id: str | None = None
    room_id: str
    play_mode: GamePlayMode
    difficulty: GameDifficulty
    ai_level: int
    user_player: GamePlayerInfo
    ai_player: GamePlayerInfo
    companion_reply: str | None = None
    result: dict[str, Any] | None = None
    duration_seconds: int | None = None
    started_at: str | None = None
    ended_at: str | None = None
    created_at: str | None = None


NativeGameKey = Literal[
    "go",
    "reversi",
    "gomoku",
    "xiangqi",
    "chess",
    "chinese_checkers",
    "match3",
    "minesweeper",
    "number_merge",
    "tetris_duel",
]
NativeGameSource = Literal["client", "replay"]


class NativeCreateSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str
    workspace_id: str | None = None
    conversation_id: str | None = None
    game_key: NativeGameKey = "gomoku"


class NativeGameEventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: str = Field(min_length=1, max_length=80)
    state: str | None = Field(default=None, max_length=120)
    payload: dict[str, Any] = Field(default_factory=dict)
    source: NativeGameSource = "client"
    client_event_id: str | None = Field(default=None, min_length=1, max_length=120)


class NativeSessionResponse(BaseModel):
    id: str
    provider: Literal["native"] = "native"
    game_key: NativeGameKey
    status: str
    user_id: str
    agent_id: str
    workspace_id: str | None = None
    conversation_id: str | None = None
    room_id: str
    play_mode: GamePlayMode
    difficulty: Literal["normal"] = "normal"
    ai_level: int
    config_version: int = 1
    effective_strength: int = 50
    engine_config: dict[str, Any] = Field(default_factory=dict)
    user_player: GamePlayerInfo
    ai_player: GamePlayerInfo
    companion_reply: str | None = None
    result: dict[str, Any] | None = None
    duration_seconds: int | None = None
    started_at: str | None = None
    ended_at: str | None = None
    created_at: str | None = None


class NativeGameEventResponse(BaseModel):
    session: NativeSessionResponse
    companion_reply: str | None = None
    persisted_event_id: str | None = None
    duplicate: bool = False


class NativeGameEventRecord(BaseModel):
    id: str
    event_type: str
    state: str | None = None
    source: str
    payload: dict[str, Any] = Field(default_factory=dict)
    companion_reply: str | None = None
    created_at: str
