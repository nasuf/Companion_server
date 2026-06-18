from pydantic import AliasChoices, BaseModel, Field


class ChatImageUpload(BaseModel):
    conversation_id: str
    name: str | None = None
    mime: str | None = None
    size: int | None = None
    width: int | None = None
    height: int | None = None
    base64: str = Field(
        min_length=1,
        validation_alias=AliasChoices("base64", "base64Data"),
    )


class ChatAttachmentResponse(BaseModel):
    id: str
    kind: str = "image"
    name: str | None = None
    mime: str
    size: int
    width: int | None = None
    height: int | None = None
    url: str
    vision_status: str = "pending"
    vision_summary: str | None = None
    created_at: str | None = None
