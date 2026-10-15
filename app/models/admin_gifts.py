"""Web 后台「礼物管理」的请求/响应模型。

这些接口只给管理员用：会返回完整收货地址（运营要据此寄快递）和未脱敏的礼物详情。
用户侧视图见 app.models.offline.RealWorldGiftItem（签收前脱敏为神秘礼物）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

AdminGiftStatusFilter = Literal["all", "in_transit", "out_for_delivery", "delivered", "closed"]
ManualTrackingStatus = Literal["ordered", "packed", "shipping", "out_for_delivery"]

# 10MB 图片的 base64 约 13.4M 字符。
_MAX_IMAGE_BASE64 = 14_000_000


class AdminGiftUser(BaseModel):
    user_id: str
    username: str
    nickname: str | None = None
    display_name: str | None = None
    created_at: str | None = None


class AdminGiftWorkspace(BaseModel):
    workspace_id: str
    agent_id: str | None = None
    agent_name: str | None = None
    conversation_id: str | None = None


class AdminGiftAddress(BaseModel):
    recipient_name: str | None = None
    phone: str | None = None
    province: str | None = None
    city: str | None = None
    district: str | None = None
    detail: str | None = None
    display: str | None = None


class AdminGiftBudget(BaseModel):
    recharge_total_cents: int
    historical_spend_cents: int
    available_cents: int


class AdminGiftTrackingEvent(BaseModel):
    id: str
    status: str
    title: str
    description: str | None = None
    location: str | None = None
    occurred_at: str
    created_at: str | None = None


class AdminGiftMessage(BaseModel):
    id: str
    role: str
    trigger_type: str | None = None
    text_preview: str
    has_trace: bool = False
    created_at: str


class AdminGiftItem(BaseModel):
    id: str
    user_id: str
    username: str | None = None
    nickname: str | None = None
    agent_id: str | None = None
    agent_name: str | None = None
    workspace_id: str | None = None
    conversation_id: str | None = None
    status: str
    trigger_type: str
    provider: str
    gift_name: str
    gift_reason: str | None = None
    gift_note: str | None = None
    product_image_url: str | None = None
    paid_amount_cents: int = 0
    carrier: str | None = None
    tracking_number: str | None = None
    address_snapshot: dict[str, Any] = Field(default_factory=dict)
    recommendation: dict[str, Any] | None = None
    failure_reason: str | None = None
    thanks_message: str | None = None
    thanks_sent_at: str | None = None
    ordered_at: str | None = None
    shipped_at: str | None = None
    delivered_at: str | None = None
    created_at: str
    updated_at: str
    last_event_title: str | None = None
    last_event_at: str | None = None


class AdminGiftDetail(AdminGiftItem):
    events: list[AdminGiftTrackingEvent] = Field(default_factory=list)
    messages: list[AdminGiftMessage] = Field(default_factory=list)


class AdminGiftListResponse(BaseModel):
    items: list[AdminGiftItem]
    total: int


class AdminGiftUserOverview(BaseModel):
    user: AdminGiftUser
    workspaces: list[AdminGiftWorkspace]
    address: AdminGiftAddress | None = None
    budget: AdminGiftBudget
    tags: list[str] = Field(default_factory=list)
    memory_count: int = 0
    gifts: list[AdminGiftItem] = Field(default_factory=list)
    # 自动送礼（定时触发 + provider 下单）开关与当前 provider，供运营判断是否会撞车。
    auto_gift_enabled: bool = False
    commerce_provider: str = "mock"


class AdminGiftRecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: str | None = Field(default=None, max_length=64)
    budget_min_yuan: int | None = Field(default=None, ge=1, le=100_000)
    budget_max_yuan: int | None = Field(default=None, ge=1, le=100_000)
    # 本轮已经看过、不想再看到的礼物名（「换一批」时由前端带上）。
    exclude_names: list[str] = Field(default_factory=list, max_length=60)
    count: int = Field(default=5, ge=1, le=8)

    @field_validator("exclude_names")
    @classmethod
    def _trim_names(cls, value: list[str]) -> list[str]:
        return [name.strip()[:40] for name in value if isinstance(name, str) and name.strip()]


class AdminGiftEvidence(BaseModel):
    ref: str
    feature: str
    source_type: Literal["memory", "tag"]
    source_text: str
    source_category: str | None = None
    source_date: str | None = None


class AdminGiftCandidate(BaseModel):
    gift_name: str
    search_keywords: list[str] = Field(default_factory=list)
    price_min_yuan: int | None = None
    price_max_yuan: int | None = None
    reason: str
    gift_note: str
    evidence: list[AdminGiftEvidence] = Field(default_factory=list)
    # False：模型给出的依据编号全部无效（可能是编造），需要运营自己判断。
    grounded: bool = False


class AdminGiftRecommendationResponse(BaseModel):
    batch_id: str
    generated_at: str
    workspace_id: str | None = None
    budget_label: str
    tags: list[str] = Field(default_factory=list)
    memory_count: int = 0
    candidates: list[AdminGiftCandidate]
    # 被丢弃的候选数（重复 / 与已送礼物同名 / 缺字段），非零时提示可以再换一批。
    dropped_count: int = 0
    trace_id: str | None = None


class _ImageFields(BaseModel):
    image_url: str | None = Field(default=None, max_length=1000)
    image_base64: str | None = Field(default=None, max_length=_MAX_IMAGE_BASE64)
    image_mime: str | None = Field(default=None, max_length=40)


class AdminGiftCreateRequest(_ImageFields):
    model_config = ConfigDict(extra="forbid")

    # 前端每次打开寄送表单生成一个 UUID，作为礼物 id：重复提交只会得到同一份礼物。
    request_id: str = Field(min_length=32, max_length=36)
    user_id: str = Field(min_length=1, max_length=64)
    workspace_id: str | None = Field(default=None, max_length=64)
    gift_name: str = Field(min_length=1, max_length=40)
    gift_note: str | None = Field(default=None, max_length=180)
    gift_reason: str | None = Field(default=None, max_length=200)
    paid_amount_yuan: float = Field(default=0, ge=0, le=100_000)
    carrier: str | None = Field(default=None, max_length=40)
    tracking_number: str | None = Field(default=None, max_length=60)
    initial_status: Literal["ordered", "shipping"] = "ordered"
    # 选中的推荐候选（理由/依据），只做审计留档。
    recommendation: dict[str, Any] | None = None

    @field_validator("gift_name", "gift_note", "gift_reason", "carrier", "tracking_number", mode="before")
    @classmethod
    def _trim(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


class AdminGiftUpdateRequest(_ImageFields):
    model_config = ConfigDict(extra="forbid")

    gift_name: str | None = Field(default=None, min_length=1, max_length=40)
    gift_note: str | None = Field(default=None, max_length=180)
    gift_reason: str | None = Field(default=None, max_length=200)
    paid_amount_yuan: float | None = Field(default=None, ge=0, le=100_000)
    carrier: str | None = Field(default=None, max_length=40)
    tracking_number: str | None = Field(default=None, max_length=60)

    @field_validator("gift_name", "gift_note", "gift_reason", "carrier", "tracking_number", mode="before")
    @classmethod
    def _trim(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


class AdminGiftTrackingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ManualTrackingStatus
    title: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=300)
    location: str | None = Field(default=None, max_length=60)
    occurred_at: datetime | None = None

    @field_validator("title", "description", "location", mode="before")
    @classmethod
    def _trim(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


class AdminGiftCancelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(default="", max_length=200)


class AdminGiftActionResponse(BaseModel):
    gift: AdminGiftDetail
    # 本次操作触发的 AI 聊天通知：sent 已发出 / failed 发送失败（礼物状态已生效）/
    # skipped 本次不需要通知（例如重复提交、非首次派送）。
    notification: Literal["sent", "failed", "skipped"] = "skipped"
    assistant_message: str | None = None
