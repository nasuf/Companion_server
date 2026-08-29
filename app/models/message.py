from typing import Literal

from pydantic import BaseModel


class MessageCreate(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    metadata: dict | None = None


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    metadata: dict | None = None
    created_at: str | None = None


class ChatRequest(BaseModel):
    message: str


class MessageSearchHit(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    metadata: dict | None = None
    created_at: str | None = None
    match_type: Literal["text", "card", "image"]
    # 该消息在 list_messages 那种标准 desc 排序里的位置, 供客户端用现成的
    # loadMessages(limit, offset=rank-N) 拉一段上下文窗口 (查找结果跳转用).
    rank: int
    matched_attachment_id: str | None = None


class MessageSearchResponse(BaseModel):
    text: list[MessageSearchHit] = []
    cards: list[MessageSearchHit] = []
    images: list[MessageSearchHit] = []
    has_more_text: bool = False
    has_more_cards: bool = False
    has_more_images: bool = False
