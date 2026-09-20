from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.models.chat_media import ChatAttachmentResponse


ActivityStatus = Literal["pending", "accepted", "ignored", "cancelled", "expired", "completed"]
GiftStatus = Literal[
    "pending_address",
    "selecting",
    "ordered",
    "shipping",
    "delivered",
    "failed",
    "skipped",
]


class OfflineActivityItem(BaseModel):
    id: str
    status: ActivityStatus
    title: str
    summary: str
    description: str
    category: str | None = None
    city: str | None = None
    location_name: str | None = None
    address: str | None = None
    starts_at: str | None = None
    ends_at: str | None = None
    official_url: str | None = None
    image_urls: list[str] = Field(default_factory=list)
    task_hint: str | None = None
    easter_egg_task: dict[str, Any] | None = None
    search_sources: list[dict[str, Any]] = Field(default_factory=list)
    # 打卡闭环状态（拍摄条件明文永不下发前端）
    reached: bool = False
    arrival_confirmed_at: str | None = None
    prophecy_text: str | None = None
    auto_archive_at: str | None = None
    fragment_count: int = 0
    accepted_at: str | None = None
    ignored_at: str | None = None
    completed_at: str | None = None
    expires_at: str | None = None
    completion_feedback: "OfflineActivityCompletionFeedback | None" = None
    created_at: str
    updated_at: str


class OfflineActivityCompletionFeedback(BaseModel):
    text: str = ""
    photo_attachments: list[ChatAttachmentResponse] = Field(default_factory=list)
    audio_attachment: ChatAttachmentResponse | None = None
    created_at: str | None = None


class OfflineActivityFragmentItem(BaseModel):
    id: str
    tier: str
    text: str
    lead_in: str | None = None


class OfflineActivityReviewResponse(BaseModel):
    id: str
    title: str
    address: str | None = None
    cover_url: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    story: str = ""
    gallery: list[str] = Field(default_factory=list)
    fragments: list[OfflineActivityFragmentItem] = Field(default_factory=list)
    event_tags: list[str] = Field(default_factory=list)
    has_memory_note: bool = False
    travel_note: str | None = None


class OfflineMemoryNoteResponse(BaseModel):
    title: str
    date_text: str
    cover_url: str | None = None
    travel_note: str
    fragment_tags: list[str] = Field(default_factory=list)


class OfflineHomeResponse(BaseModel):
    pending_activity_count: int = 0
    accepted_activity_count: int = 0
    completed_activity_count: int = 0
    gift_count: int = 0
    shipping_gift_count: int = 0
    has_location: bool = False
    tags: list[str] = Field(default_factory=list)
    latest_activity: OfflineActivityItem | None = None
    gift_summary: str = "你有一份惊喜在路上"


class OfflineActivitiesResponse(BaseModel):
    latest: OfflineActivityItem | None = None
    pending: list[OfflineActivityItem] = Field(default_factory=list)
    ignored: list[OfflineActivityItem] = Field(default_factory=list)
    completed: list[OfflineActivityItem] = Field(default_factory=list)


class OfflineActivityClearResponse(BaseModel):
    deleted_activities: int = 0
    deleted_feedback: int = 0


class OfflineActivityCompleteRequest(BaseModel):
    text: str = Field(default="", max_length=1000)
    photo_attachment_ids: list[str] = Field(default_factory=list, max_length=3)
    audio_attachment_id: str | None = Field(default=None, max_length=80)


class OfflineActivityArriveRequest(BaseModel):
    """确认到达时上报的当前 GPS（可选）。

    有坐标且该地点已地理编码时才做 ≤200m 直线校验；否则荣誉制放行（允许用户直接
    点击到达）。地理编码 Key 未配置时地点无坐标，等同放行。
    """

    lat: float | None = Field(default=None, ge=-90, le=90)
    lng: float | None = Field(default=None, ge=-180, le=180)


class OfflineActivityImageUpload(BaseModel):
    kind: Literal["image", "audio"] = "image"
    name: str | None = Field(default=None, max_length=120)
    mime: str = Field(default="image/jpeg", max_length=80)
    size: int = Field(ge=1, le=5 * 1024 * 1024)
    width: int | None = Field(default=None, ge=1, le=10000)
    height: int | None = Field(default=None, ge=1, le=10000)
    duration_seconds: int | None = Field(default=None, ge=1, le=180)
    base64: str = Field(min_length=1)


class GiftAddressResponse(BaseModel):
    id: str | None = None
    recipient_name: str | None = None
    phone: str | None = None
    province: str | None = None
    city: str | None = None
    district: str | None = None
    detail: str | None = None
    display: str | None = None


class GiftAddressRequest(BaseModel):
    recipient_name: str = Field(min_length=1, max_length=40)
    phone: str = Field(min_length=6, max_length=30)
    province: str = Field(default="", max_length=40)
    city: str = Field(min_length=1, max_length=40)
    district: str = Field(default="", max_length=40)
    detail: str = Field(min_length=3, max_length=160)

    @field_validator("recipient_name", "phone", "province", "city", "district", "detail", mode="before")
    @classmethod
    def _trim_string(cls, value: str) -> str:
        return value.strip() if isinstance(value, str) else value

    @field_validator("phone")
    @classmethod
    def _validate_phone(cls, value: str) -> str:
        compact = re.sub(r"[\s-]+", "", value)
        if not re.fullmatch(r"\+?\d{6,20}", compact):
            raise ValueError("invalid phone number")
        return compact


class GiftTrackingEvent(BaseModel):
    id: str
    status: str
    title: str
    description: str | None = None
    location: str | None = None
    occurred_at: str


class RealWorldGiftItem(BaseModel):
    id: str
    status: GiftStatus
    trigger_type: str
    gift_name: str
    gift_reason: str | None = None
    gift_note: str | None = None
    product_image_url: str | None = None
    provider: str = "mock"
    provider_product_id: str | None = None
    provider_order_id: str | None = None
    product_url: str | None = None
    paid_amount_cents: int = 0
    tracking_number: str | None = None
    logistics_provider: str | None = None
    last_tracking_synced_at: str | None = None
    thanks_sent_at: str | None = None
    ordered_at: str | None = None
    shipped_at: str | None = None
    delivered_at: str | None = None
    created_at: str
    updated_at: str


class GiftYearGroup(BaseModel):
    year: int
    gifts: list[RealWorldGiftItem] = Field(default_factory=list)


class GiftsHomeResponse(BaseModel):
    address: GiftAddressResponse | None = None
    shipping_gift: RealWorldGiftItem | None = None
    groups: list[GiftYearGroup] = Field(default_factory=list)


class GiftThanksRequest(BaseModel):
    message: str = Field(min_length=1, max_length=300)
    client_id: str | None = Field(default=None, max_length=120)


class GiftThanksResponse(BaseModel):
    ok: bool
    gift: RealWorldGiftItem
    assistant_message: str | None = None


class GiftTrackingResponse(BaseModel):
    gift_id: str
    events: list[GiftTrackingEvent] = Field(default_factory=list)
