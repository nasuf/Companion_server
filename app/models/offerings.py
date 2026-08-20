from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.models.wallet import WalletBalanceResponse


class SendRedPacketRequest(BaseModel):
    conversation_id: str
    ticket_amount: int = Field(ge=1, le=1_000_000)
    blessing: str | None = Field(default=None, max_length=40)


class RedPacketOffering(BaseModel):
    id: str
    kind: str
    ticket_amount: int
    agent_value_yuan: int
    status: str
    blessing: str | None = None
    conversation_id: str | None = None
    message_id: str | None = None
    agent_id: str
    created_at: str
    received_at: str | None = None


class RedPacketSendResponse(BaseModel):
    offering: RedPacketOffering
    component_card: dict[str, Any]
    wallet: WalletBalanceResponse


class RedPacketGetResponse(BaseModel):
    offering: RedPacketOffering
    component_card: dict[str, Any]
