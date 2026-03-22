"""Prompt storage service backed by Redis + Prisma."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict

from app.db import db
from app.redis_client import get_redis
from app.services.prompt_registry import PROMPT_DEFINITION_MAP, PROMPT_DEFINITIONS, PromptDefinition

logger = logging.getLogger(__name__)

PROMPT_KEY_PREFIX = "prompt_template:"


def _redis_key(key: str) -> str:
    return f"{PROMPT_KEY_PREFIX}{key}"


async def ensure_prompt_templates() -> None:
    """Ensure prompt templates exist in DB and warm Redis from DB state."""
    for definition in PROMPT_DEFINITIONS:
        existing = await db.prompttemplate.find_unique(where={"key": definition.key})
        if existing:
            content = existing.content or definition.default_text
            needs_update = (
                existing.stage != definition.stage
                or existing.category != definition.category
                or existing.title != definition.title
                or (existing.description or "") != definition.description
                or existing.defaultContent != definition.default_text
            )
            if needs_update:
                await db.prompttemplate.update(
                    where={"key": definition.key},
                    data={
                        "stage": definition.stage,
                        "category": definition.category,
                        "title": definition.title,
                        "description": definition.description,
                        "defaultContent": definition.default_text,
                    },
                )
        else:
            content = definition.default_text
            await db.prompttemplate.create(
                data={
                    "key": definition.key,
                    "stage": definition.stage,
                    "category": definition.category,
                    "title": definition.title,
                    "description": definition.description,
                    "content": definition.default_text,
                    "defaultContent": definition.default_text,
                    "isEnabled": True,
                }
            )
        redis = await get_redis()
        await redis.set(_redis_key(definition.key), content)


async def get_prompt_text(key: str) -> str:
    """Fetch latest prompt text from Redis, falling back to DB/default."""
    definition = PROMPT_DEFINITION_MAP.get(key)
    if not definition:
        raise KeyError(f"Unknown prompt key: {key}")

    redis = await get_redis()
    cached = await redis.get(_redis_key(key))
    if cached:
        return cached

    record = await db.prompttemplate.find_unique(where={"key": key})
    content = record.content if record and record.content else definition.default_text
    await redis.set(_redis_key(key), content)
    return content


async def list_prompts() -> list[dict]:
    """Return prompt definitions merged with DB and Redis state."""
    redis = await get_redis()
    rows = await db.prompttemplate.find_many(order=[{"stage": "asc"}, {"title": "asc"}])
    row_map = {row.key: row for row in rows}

    prompts: list[dict] = []
    for definition in PROMPT_DEFINITIONS:
        row = row_map.get(definition.key)
        cached = await redis.get(_redis_key(definition.key))
        content = cached or (row.content if row else definition.default_text)
        prompts.append(
            {
                **asdict(definition),
                "content": content,
                "is_enabled": bool(row.isEnabled) if row else True,
                "updated_at": row.updatedAt.isoformat() if row else None,
                "source": "redis" if cached else ("db" if row else "default"),
            }
        )
    return prompts


async def _persist_prompt_update(key: str, content: str) -> None:
    definition = PROMPT_DEFINITION_MAP[key]
    try:
        existing = await db.prompttemplate.find_unique(where={"key": key})
        if existing:
            await db.prompttemplate.update(
                where={"key": key},
                data={
                    "content": content,
                    "stage": definition.stage,
                    "category": definition.category,
                    "title": definition.title,
                    "description": definition.description,
                    "defaultContent": definition.default_text,
                },
            )
        else:
            await db.prompttemplate.create(
                data={
                    "key": key,
                    "stage": definition.stage,
                    "category": definition.category,
                    "title": definition.title,
                    "description": definition.description,
                    "content": content,
                    "defaultContent": definition.default_text,
                    "isEnabled": True,
                }
            )
    except Exception as exc:
        logger.error("Failed to persist prompt %s: %s", key, exc)


async def update_prompt_text(key: str, content: str) -> dict:
    """Write prompt to Redis immediately and persist to DB asynchronously."""
    definition = PROMPT_DEFINITION_MAP.get(key)
    if not definition:
        raise KeyError(f"Unknown prompt key: {key}")

    normalized = content.strip()
    if not normalized:
        raise ValueError("Prompt content cannot be empty")

    redis = await get_redis()
    await redis.set(_redis_key(key), normalized)
    task = asyncio.create_task(_persist_prompt_update(key, normalized))
    task.add_done_callback(lambda t: t.exception() and logger.error("Prompt save task failed: %s", t.exception()))

    return {
        **asdict(definition),
        "content": normalized,
        "source": "redis",
    }


async def reset_prompt_text(key: str) -> dict:
    """Reset prompt to default in Redis and DB."""
    definition = PROMPT_DEFINITION_MAP.get(key)
    if not definition:
        raise KeyError(f"Unknown prompt key: {key}")

    redis = await get_redis()
    await redis.set(_redis_key(key), definition.default_text)
    await _persist_prompt_update(key, definition.default_text)
    row = await db.prompttemplate.find_unique(where={"key": key})
    return {
        **asdict(definition),
        "content": definition.default_text,
        "source": "default",
        "updated_at": row.updatedAt.isoformat() if row else None,
    }
