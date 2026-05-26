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


class AuthResponse(BaseModel):
    token: str
    user_id: str
    username: str
    role: str
    has_agent: bool
    agent_id: str | None = None
    agent_name: str | None = None
    workspace_id: str | None = None
    conversation_id: str | None = None
