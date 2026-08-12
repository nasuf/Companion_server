from pydantic import BaseModel, Field

# 昵称长度上限。两个写入端点 (App 的 /users/me/profile 与小程序的
# /auth/wechat/profile) 共用同一个值 —— 分开定义会让其中一条路径悄悄允许更长的
# 名字, 而两边写的是同一列。
MAX_DISPLAY_NAME_LENGTH = 32


class UserUpdate(BaseModel):
    email: str | None = None


class UserResponse(BaseModel):
    id: str
    username: str
    email: str | None = None
    created_at: str | None = None


class UserProfileUpdate(BaseModel):
    """昵称修改。头像走 multipart 的 POST /users/me/avatar, 不在这里。"""

    display_name: str = Field(min_length=1, max_length=MAX_DISPLAY_NAME_LENGTH)


class UserProfileResponse(BaseModel):
    """改完之后客户端 copyWith 进本地 session 用的最小回执。

    刻意不返回新 JWT: 昵称头像跟鉴权无关, 重签 token 会让所有端的 session 存储
    跟着抖一次。
    """

    display_name: str | None = None
    avatar_url: str | None = None


class ProfileStatsResponse(BaseModel):
    workspace_id: str
    intimacy_stage: str
    intimacy_stage_label: str
    topic_intimacy: float
    intimacy_subtitle: str
    companion_days: int
    companion_started_on: str | None = None
    chat_hours: int
    chat_minutes: int
    chat_duration_label: str
    chat_duration_subtitle: str
    message_count: int
    recent_7d_message_count: int
    recent_7d_message_label: str
    companion_summary: str
    backpack_count: int = 0
    member_is_active: bool = False
    member_expires_on: str | None = None


class ChatRecordsClearResponse(BaseModel):
    workspace_id: str
    cleared_conversations: int


class UserLocationRequest(BaseModel):
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    city: str | None = Field(default=None, max_length=80)
    region: str | None = Field(default=None, max_length=80)
    country: str | None = Field(default=None, max_length=80)
    source: str = Field(default="device", max_length=40)
    permission_status: str = Field(default="unknown", max_length=40)


class UserLocationResponse(BaseModel):
    has_location: bool
    latitude: float | None = None
    longitude: float | None = None
    city: str | None = None
    region: str | None = None
    country: str | None = None
    permission_status: str | None = None
    updated_at: str | None = None
