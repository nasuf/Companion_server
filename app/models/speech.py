from typing import Literal

from pydantic import AliasChoices, BaseModel, Field

from app.models.chat_media import ChatAttachmentResponse


class ChatAudioTranscriptionRequest(BaseModel):
    conversation_id: str = Field(min_length=1)
    name: str | None = None
    mime: str | None = None
    size: int | None = Field(default=None, ge=1)
    duration_seconds: int = Field(ge=1)
    display_mode: Literal["voice", "text"] = "voice"
    base64: str = Field(
        min_length=1,
        max_length=3 * 1024 * 1024,
        validation_alias=AliasChoices("base64", "base64Data"),
    )


class ChatAudioTranscriptionResponse(BaseModel):
    text: str
    duration_seconds: int
    model: str
    request_id: str | None = None
    attachment: ChatAttachmentResponse | None = None
