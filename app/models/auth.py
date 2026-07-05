from typing import Literal

from pydantic import BaseModel, Field, field_validator


class RegisterRequest(BaseModel):
    username: str = Field(min_length=2, max_length=30, pattern=r"^[a-zA-Z0-9_\u4e00-\u9fff]+$")
    password: str = Field(min_length=6, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


class WeChatMobileLoginRequest(BaseModel):
    code: str = Field(min_length=1, max_length=512)
    platform: Literal["ios", "android", "harmony"] = "ios"

    @field_validator("code")
    @classmethod
    def strip_code(cls, value: str) -> str:
        code = value.strip()
        if not code:
            raise ValueError("code must not be blank")
        return code


class WeChatMiniLoginRequest(BaseModel):
    code: str = Field(min_length=1, max_length=512)

    @field_validator("code")
    @classmethod
    def strip_code(cls, value: str) -> str:
        code = value.strip()
        if not code:
            raise ValueError("code must not be blank")
        return code


class WeChatH5LoginRequest(BaseModel):
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
