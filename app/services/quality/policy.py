"""All budgets and sampling limits have one configurable source."""

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from app.config import settings

CATEGORIES = ("hallucination", "irrelevance", "temporal", "persona_tone", "repetition")
CRITICAL = ("hallucination", "speaker_confusion", "fabricated_action")
DEFINITION_VERSION = "quality-review-v1"


def local_day(now: datetime):
    return now.astimezone(ZoneInfo(settings.schedule_timezone)).date()


def money(value) -> Decimal:
    return Decimal(str(value)).quantize(Decimal(".00000001"))


def run_limit(kind: str) -> Decimal:
    return money(
        settings.quality_daily_cny
        if kind in ("daily", "calibration")
        else settings.quality_prompt_run_cny
        if kind == "prompt"
        else settings.quality_full_run_cny
    )


def public_policy() -> dict:
    return dict(
        definition_version=DEFINITION_VERSION,
        timezone=settings.schedule_timezone,
        daily_turns=settings.quality_daily_turns,
        review_limit=settings.quality_review_limit,
        judge=f"dashscope/{settings.quality_judge_model}",
        daily_cny=settings.quality_daily_cny,
        prompt_run_cny=settings.quality_prompt_run_cny,
        prompt_daily_cny=settings.quality_prompt_daily_cny,
        full_run_cny=settings.quality_full_run_cny,
        monthly_cny=settings.quality_monthly_cny,
        prompt_debounce_seconds=settings.quality_prompt_debounce_seconds,
        critical_repeats=settings.quality_critical_repeats,
        warning_only=True,
    )
