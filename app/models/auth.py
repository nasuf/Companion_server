from typing import Literal

from pydantic import BaseModel, Field, field_validator

_PLATFORM_MAX_LEN = 32
_VERSION_MAX_LEN = 64


def _clean_optional_str(value: object, max_len: int) -> str | None:
    """Best-effort sanitizer for analytics-only client metadata.

    Never raises: junk values degrade to None instead of failing the whole
    auth request over an optional telemetry field.
    """
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    return cleaned[:max_len]


class ClientInfoMixin(BaseModel):
    """Optional device metadata clients attach for signup-source analytics."""

    # Device platform, e.g. ios / android / harmony / devtools / windows / mac / web
    platform: str | None = None
    # OS version, e.g. "iOS 17.5.1" / "Android 14"
    os_version: str | None = None
    # Client build version, e.g. Flutter "0.1.10+1" or Mini Program release version
    app_version: str | None = None

    @field_validator("platform", mode="before")
    @classmethod
    def _clean_platform(cls, value: object) -> str | None:
        cleaned = _clean_optional_str(value, _PLATFORM_MAX_LEN)
        return cleaned.lower() if cleaned else None

    @field_validator("os_version", "app_version", mode="before")
    @classmethod
    def _clean_version(cls, value: object) -> str | None:
        return _clean_optional_str(value, _VERSION_MAX_LEN)


class RegisterRequest(ClientInfoMixin):
    username: str = Field(min_length=2, max_length=30, pattern=r"^[a-zA-Z0-9_\u4e00-\u9fff]+$")
    password: str = Field(min_length=6, max_length=128)
    # Which client surface the password registration came from. None = legacy
    # client that predates the field; the signup source then stays "password".
    channel: Literal["app", "miniprogram", "h5", "web"] | None = None


class LoginRequest(BaseModel):
    username: str
    password: str


class WeChatMobileLoginRequest(ClientInfoMixin):
    code: str = Field(min_length=1, max_length=512)
    # Narrower than the mixin field: the Flutter app is the only caller and
    # always reports one of these values.
    platform: Literal["ios", "android", "harmony"] = "ios"

    @field_validator("code")
    @classmethod
    def strip_code(cls, value: str) -> str:
        code = value.strip()
        if not code:
            raise ValueError("code must not be blank")
        return code


class WeChatMiniLoginRequest(ClientInfoMixin):
    code: str = Field(min_length=1, max_length=512)

    @field_validator("code")
    @classmethod
    def strip_code(cls, value: str) -> str:
        code = value.strip()
        if not code:
            raise ValueError("code must not be blank")
        return code


class WeChatH5LoginRequest(ClientInfoMixin):
    """公众号网页授权 (H5) 登录: 前端携带 OAuth 回调的 code."""

    code: str = Field(min_length=1, max_length=512)

    @field_validator("code")
    @classmethod
    def strip_code(cls, value: str) -> str:
        code = value.strip()
        if not code:
            raise ValueError("code must not be blank")
        return code


class WeChatProfileUpdate(BaseModel):
    nickname: str | None = Field(default=None, max_length=64)
    # Optional base64 avatar image (from the Mini Program chooseAvatar button).
    avatar_base64: str | None = None
    avatar_mime: str | None = None


class AuthResponse(BaseModel):
    token: str
    user_id: str
    username: str
    user_display_name: str | None = None
    user_avatar_url: str | None = None
    role: str
    has_agent: bool
    agent_id: str | None = None
    agent_name: str | None = None
    agent_avatar_key: str | None = None
    agent_avatar_url: str | None = None
    agent_city: str | None = None
    workspace_id: str | None = None
    conversation_id: str | None = None
