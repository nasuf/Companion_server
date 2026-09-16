from datetime import date

from pydantic import BaseModel, Field


class InteractionDayStatus(BaseModel):
    date: date
    source: str | None = None
    makeup_eligible: bool = False


class InteractionOverviewResponse(BaseModel):
    current_streak: int
    today: date
    today_marked: bool
    makeup_cards: int
    lookback_days: int
    workspace_created_on: date | None = None
    month: str
    days: list[InteractionDayStatus] = Field(default_factory=list)


class InteractionMakeupRequest(BaseModel):
    date: date


class InteractionMakeupResponse(BaseModel):
    current_streak: int
    today: date
    today_marked: bool
    makeup_cards: int
    day: InteractionDayStatus
