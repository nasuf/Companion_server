"""Fast fact extractor for hot-path working memory.

Uses a small utility model to synchronously extract a handful of
high-confidence facts from the current user message, then stores them
in Redis as conversation-scoped working memory.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from app.redis_client import get_redis
from app.services.llm.models import get_utility_model, invoke_json
from app.services.prompt_store import get_prompt_text

logger = logging.getLogger(__name__)

WORKING_FACTS_TTL = 86400 * 7
MIN_FACT_CONFIDENCE = 0.78
MAX_WORKING_FACTS = 8

_ALLOWED_CATEGORIES = {
    "name",
    "age",
    "location",
    "occupation",
    "education",
    "preference",
    "dislike",
    "relationship",
    "plan",
    "schedule",
}


def _working_facts_key(conversation_id: str) -> str:
    return f"working_facts:{conversation_id}"


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).lower()


def _fact_identity(fact: dict[str, Any]) -> tuple[str, str]:
    category = str(fact.get("category", "")).strip().lower()
    key = str(fact.get("key", "")).strip().lower()
    if not key:
        key = category
    return category, key


def _safe_ttl_days(value: Any) -> int:
    try:
        days = int(value)
    except (TypeError, ValueError):
        days = 7
    return max(1, min(days, 30))


def _fact_expired(fact: dict[str, Any], now: datetime) -> bool:
    expires_at = fact.get("expires_at")
    if not expires_at:
        return False
    try:
        return datetime.fromisoformat(str(expires_at)) <= now
    except ValueError:
        return False


def _is_fast_fact_candidate(message: str) -> bool:
    msg = message.strip()
    if not msg:
        return False

    lower = msg.lower()
    if any(token in msg for token in ("我叫", "我是", "我在", "我住", "我喜欢", "我不喜欢", "我讨厌", "我", "我的")):
        return True
    if any(token in lower for token in ("i am", "i'm", "im ", "my ", "i live", "i work", "i study", "i like", "i love", "i hate", "i plan")):
        return True
    if re.search(r"\d{1,2}\s*岁", msg) or re.search(r"\b\d{1,2}\b", lower):
        return True
    return False


def _heuristic_extract(message: str) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    msg = message.strip()
    lower = msg.lower()

    match = re.search(r"我叫([\w\u4e00-\u9fff·]{1,20})", msg)
    if match:
        facts.append({
            "category": "name",
            "key": "name",
            "value": match.group(1),
            "confidence": 0.95,
            "ttl_days": 30,
        })

    match = re.search(r"我(\d{1,2})岁", msg)
    if match:
        facts.append({
            "category": "age",
            "key": "age",
            "value": f"{match.group(1)}岁",
            "confidence": 0.95,
            "ttl_days": 30,
        })

    match = re.search(r"\b(?:i am|i'm|im)\s+(\d{1,2})\b", lower)
    if match:
        facts.append({
            "category": "age",
            "key": "age",
            "value": match.group(1),
            "confidence": 0.92,
            "ttl_days": 30,
        })

    match = re.search(r"我在(.+?)(工作|上班|读书|上学)", msg)
    if match:
        facts.append({
            "category": "location",
            "key": "current_place",
            "value": match.group(1).strip(),
            "confidence": 0.82,
            "ttl_days": 14,
        })

    match = re.search(r"我喜欢(.+?)(?:。|，|$)", msg)
    if match:
        facts.append({
            "category": "preference",
            "key": "explicit_like",
            "value": match.group(1).strip(),
            "confidence": 0.86,
            "ttl_days": 21,
        })

    match = re.search(r"我(不喜欢|讨厌)(.+?)(?:。|，|$)", msg)
    if match:
        facts.append({
            "category": "dislike",
            "key": "explicit_dislike",
            "value": match.group(2).strip(),
            "confidence": 0.88,
            "ttl_days": 21,
        })

    match = re.search(r"\bi\s+(?:like|love)\s+(.+?)(?:[.!?,]|$)", lower)
    if match:
        facts.append({
            "category": "preference",
            "key": "explicit_like",
            "value": match.group(1).strip(),
            "confidence": 0.82,
            "ttl_days": 21,
        })

    return facts[:3]


def _sanitize_fact(raw: dict[str, Any], now: datetime) -> dict[str, Any] | None:
    category = str(raw.get("category", "")).strip().lower()
    key = str(raw.get("key", "")).strip().lower()
    value = str(raw.get("value", "")).strip()
    confidence = float(raw.get("confidence", 0.0) or 0.0)

    if category not in _ALLOWED_CATEGORIES or not value:
        return None
    if confidence < MIN_FACT_CONFIDENCE:
        return None

    safe_key = re.sub(r"[^a-z0-9_]+", "_", key or category).strip("_") or category
    ttl_days = _safe_ttl_days(raw.get("ttl_days", 7))
    expires_at = now + timedelta(days=ttl_days)

    return {
        "category": category,
        "key": safe_key,
        "value": value,
        "confidence": round(confidence, 3),
        "updated_at": now.isoformat(),
        "expires_at": expires_at.isoformat(),
    }


def merge_working_facts(
    existing: list[dict[str, Any]],
    new_facts: list[dict[str, Any]],
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    now = now or datetime.now(timezone.utc)
    merged: dict[tuple[str, str], dict[str, Any]] = {}

    for fact in existing:
        if _fact_expired(fact, now):
            continue
        ident = _fact_identity(fact)
        merged[ident] = fact

    for fact in new_facts:
        ident = _fact_identity(fact)
        current = merged.get(ident)
        if current is None:
            merged[ident] = fact
            continue

        if _normalize_text(current.get("value", "")) == _normalize_text(fact.get("value", "")):
            if float(fact.get("confidence", 0.0)) >= float(current.get("confidence", 0.0)):
                merged[ident] = fact
            continue

        if float(fact.get("confidence", 0.0)) >= float(current.get("confidence", 0.0)):
            merged[ident] = fact

    ordered = sorted(
        merged.values(),
        key=lambda item: (
            float(item.get("confidence", 0.0)),
            str(item.get("updated_at", "")),
        ),
        reverse=True,
    )
    return ordered[:MAX_WORKING_FACTS]


def facts_for_prompt(facts: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for fact in facts:
        category = str(fact.get("category", "")).strip()
        value = str(fact.get("value", "")).strip()
        if category and value:
            lines.append(f"[{category}] {value}")
    return lines


async def get_working_facts(conversation_id: str) -> list[dict[str, Any]]:
    redis = await get_redis()
    raw = await redis.get(_working_facts_key(conversation_id))
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []

    facts = payload.get("facts", []) if isinstance(payload, dict) else []
    if not isinstance(facts, list):
        return []

    now = datetime.now(timezone.utc)
    cleaned = [fact for fact in facts if isinstance(fact, dict) and not _fact_expired(fact, now)]
    return cleaned[:MAX_WORKING_FACTS]


async def save_working_facts(conversation_id: str, facts: list[dict[str, Any]]) -> None:
    redis = await get_redis()
    await redis.set(
        _working_facts_key(conversation_id),
        json.dumps({"facts": facts}, ensure_ascii=False),
        ex=WORKING_FACTS_TTL,
    )


async def extract_fast_facts(message: str) -> list[dict[str, Any]]:
    if not _is_fast_fact_candidate(message):
        return []

    now = datetime.now(timezone.utc)
    facts: list[dict[str, Any]] = []
    try:
        prompt_template = await get_prompt_text("memory.fast_fact")
        result = await invoke_json(
            get_utility_model(),
            prompt_template.format(message=message),
        )
        raw_facts = result.get("facts", []) if isinstance(result, dict) else []
        if isinstance(raw_facts, list):
            for item in raw_facts[:3]:
                if isinstance(item, dict):
                    fact = _sanitize_fact(item, now)
                    if fact:
                        facts.append(fact)
    except Exception as e:
        logger.warning(f"Fast fact extraction failed: {e}")

    if facts:
        return facts

    return [
        fact
        for fact in (_sanitize_fact(item, now) for item in _heuristic_extract(message))
        if fact
    ]


async def update_working_facts(
    conversation_id: str,
    message: str,
) -> list[dict[str, Any]]:
    existing = await get_working_facts(conversation_id)
    new_facts = await extract_fast_facts(message)
    if not new_facts:
        return existing

    merged = merge_working_facts(existing, new_facts)
    await save_working_facts(conversation_id, merged)
    return merged
