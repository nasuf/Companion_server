from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MusicConfigResponse(BaseModel):
    provider: str = "jamendo"
    api_enabled: bool
    default_libraries: list[str]
    missing_config: list[str]


class MusicLibrary(BaseModel):
    id: str
    title: str
    subtitle: str = ""


class MusicLibrariesResponse(BaseModel):
    libraries: list[MusicLibrary]
    provider: str = "jamendo"
    default_library: str


class MusicTrack(BaseModel):
    id: str
    title: str
    artist: str = "Jamendo"
    album: str = "Jamendo Library"
    library: str = "focus"
    url: str = ""
    duration_sec: int = 0
    cover_key: str = "music-cover-01.jpg"
    accent_a: str = "#1f6fff"
    accent_b: str = "#18c6c0"
    source: str = "jamendo"
    is_favorite: bool = False
    played_by_agent: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class MusicTracksResponse(BaseModel):
    tracks: list[MusicTrack]
    provider: str = "jamendo"
    api_enabled: bool
    library: str | None = None
    cache_ttl_seconds: int = 0


class MusicTrackPlayUrlResponse(BaseModel):
    provider: str = "jamendo"
    track_id: str
    url: str
    expires_at: str | None = None


class MusicTrackPayload(BaseModel):
    id: str = Field(min_length=1, max_length=160)
    title: str = Field(min_length=1, max_length=240)
    artist: str = Field(default="Jamendo", max_length=160)
    album: str = Field(default="Jamendo Library", max_length=240)
    library: str = Field(default="focus", max_length=120)
    url: str = Field(default="", max_length=2000)
    duration_sec: int = Field(default=0, ge=0, le=24 * 60 * 60)
    cover_key: str = Field(default="music-cover-01.jpg", max_length=160)
    accent_a: str = Field(default="#1f6fff", max_length=32)
    accent_b: str = Field(default="#18c6c0", max_length=32)
    source: str = Field(default="jamendo", max_length=80)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MusicFavoriteRequest(BaseModel):
    agent_id: str
    workspace_id: str | None = None
    track: MusicTrackPayload


class MusicFavoriteResponse(BaseModel):
    track: MusicTrack


class MusicPlaybackRequest(BaseModel):
    agent_id: str
    workspace_id: str | None = None
    conversation_id: str | None = None
    track: MusicTrackPayload
    position_seconds: int = Field(default=0, ge=0, le=24 * 60 * 60)
    is_playing: bool = True


class MusicPlaybackResponse(BaseModel):
    track: MusicTrack | None
    position_seconds: int = 0
    is_playing: bool = False
    updated_at: str | None = None


class MusicCoListeningResponse(BaseModel):
    status: str = "ended"
    track: MusicTrack | None = None
    position_seconds: int = 0
    is_playing: bool = False
    initiated_by: str | None = None
    ended_reason: str | None = None
    updated_at: str | None = None


class MusicCoListeningEndRequest(BaseModel):
    agent_id: str
    conversation_id: str
    reason: str = Field(default="user_exit", max_length=80)
