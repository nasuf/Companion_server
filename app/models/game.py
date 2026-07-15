from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


GameDifficulty = Literal["newbie", "normal", "hard"]
GamePlayMode = Literal["versus", "cooperate"]


class SudConfigResponse(BaseModel):
    provider: str = "sud"
    sdk_enabled: bool
    sdk_package: str = "sud_gip_plugin"
    app_id: str
    app_key: str
    bundle_id: str
    is_test_env: bool
    default_mg_id: str
    missing_config: list[str]
    callbacks: dict[str, str]


class SudCreateSessionRequest(BaseModel):
    agent_id: str
    workspace_id: str | None = None
    conversation_id: str | None = None
    mg_id: str | None = None
    room_id: str | None = None
    play_mode: GamePlayMode = "versus"
    difficulty: GameDifficulty = "newbie"


class SudPlayerInfo(BaseModel):
    uid: str
    nick_name: str
    avatar_url: str = ""
    gender: str = ""
    is_ai: int = 0
    ai_level: int = 0


class SudSessionResponse(BaseModel):
    id: str
    provider: str = "sud"
    game_key: str | None = None
    status: str
    sdk_enabled: bool
    user_id: str
    agent_id: str
    workspace_id: str | None = None
    conversation_id: str | None = None
    app_id: str
    app_key: str
    bundle_id: str
    is_test_env: bool
    mg_id: str
    room_id: str
    code: str
    code_expires_at: str
    play_mode: GamePlayMode
    difficulty: GameDifficulty
    ai_level: int
    user_player: SudPlayerInfo
    ai_player: SudPlayerInfo
    companion_reply: str | None = None
    result: dict[str, Any] | None = None
    duration_seconds: int | None = None
    started_at: str | None = None
    ended_at: str | None = None
    created_at: str | None = None


class SudGameEventRequest(BaseModel):
    event_type: str = Field(min_length=1, max_length=80)
    state: str | None = Field(default=None, max_length=120)
    payload: dict[str, Any] = Field(default_factory=dict)
    source: Literal["client", "sud_callback", "mock"] = "client"


class SudGameEventResponse(BaseModel):
    session: SudSessionResponse
    companion_reply: str | None = None
    persisted_event_id: str | None = None


class SudCallbackGetSsTokenRequest(BaseModel):
    code: str


class SudCallbackUpdateSsTokenRequest(BaseModel):
    ss_token: str


class SudCallbackGetUserInfoRequest(BaseModel):
    ss_token: str


class SudCallbackReportGameInfoRequest(BaseModel):
    report_type: str
    report_msg: dict[str, Any]
    uid: str | None = None
    ss_token: str | None = None


class SudCallbackNotifyRequest(BaseModel):
    app_id: str | None = None
    notify_id: str | None = None
    notify_time: str | None = None
    notify_event: str
    data: dict[str, Any] = Field(default_factory=dict)
    ss_token: str | None = None


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
    user_player: SudPlayerInfo
    ai_player: SudPlayerInfo
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
