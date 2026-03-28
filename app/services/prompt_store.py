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
            # Logic: If current content matches the old default, update it to the new default.
            # This ensures code-level prompt updates are applied unless the user has customized them in the DB.
            is_unmodified = existing.content == existing.defaultContent
            new_content = definition.default_text if is_unmodified else existing.content
            
            # Content for Redis sync (will be updated below if needs_update is True)
            content = new_content
            
            needs_update = (
                existing.stage != definition.stage
                or existing.category != definition.category
                or existing.title != definition.title
                or (existing.description or "") != definition.description
                or existing.defaultContent != definition.default_text
                or (is_unmodified and existing.content != definition.default_text)
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
                        "content": new_content,
                    },
                )
                content = new_content
                
            version_count = await db.prompttemplateversion.count(
                where={"promptId": existing.id}
            )
            if version_count == 0:
                await _create_prompt_version(
                    prompt_id=existing.id,
                    prompt_key=definition.key,
                    content=content,
                    source="db",
                    change_type="bootstrap",
                )
        else:
            content = definition.default_text
            created = await db.prompttemplate.create(
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
            await _create_prompt_version(
                prompt_id=created.id,
                prompt_key=definition.key,
                content=definition.default_text,
                source="default",
                change_type="bootstrap",
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


async def _create_prompt_version(
    *,
    prompt_id: str,
    prompt_key: str,
    content: str,
    source: str,
    change_type: str,
) -> None:
    await db.prompttemplateversion.create(
        data={
            "promptId": prompt_id,
            "promptKey": prompt_key,
            "content": content,
            "source": source,
            "changeType": change_type,
        }
    )


async def list_prompt_versions(key: str, limit: int = 20) -> list[dict]:
    definition = PROMPT_DEFINITION_MAP.get(key)
    if not definition:
        raise KeyError(f"Unknown prompt key: {key}")

    versions = await db.prompttemplateversion.find_many(
        where={"promptKey": key},
        order={"createdAt": "desc"},
        take=limit,
    )
    return [
        {
            "id": version.id,
            "prompt_key": version.promptKey,
            "content": version.content,
            "source": version.source,
            "change_type": version.changeType,
            "persistence": "synced",
            "created_at": version.createdAt.isoformat(),
        }
        for version in versions
    ]


async def _persist_prompt_update(
    key: str,
    content: str,
    *,
    source: str,
    change_type: str,
) -> dict:
    definition = PROMPT_DEFINITION_MAP[key]
    try:
        existing = await db.prompttemplate.find_unique(where={"key": key})
        if existing:
            row = await db.prompttemplate.update(
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
            row = await db.prompttemplate.create(
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
        await _create_prompt_version(
            prompt_id=row.id,
            prompt_key=key,
            content=content,
            source=source,
            change_type=change_type,
        )
        return row
    except Exception as exc:
        logger.error("Failed to persist prompt %s: %s", key, exc)
        raise


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
    task = asyncio.create_task(
        _persist_prompt_update(
            key,
            normalized,
            source="redis",
            change_type="manual_save",
        )
    )
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
    row = await _persist_prompt_update(
        key,
        definition.default_text,
        source="default",
        change_type="reset_default",
    )
    return {
        **asdict(definition),
        "content": definition.default_text,
        "source": "default",
        "updated_at": row.updatedAt.isoformat() if row else None,
    }


async def restore_prompt_version(key: str, version_id: str) -> dict:
    definition = PROMPT_DEFINITION_MAP.get(key)
    if not definition:
        raise KeyError(f"Unknown prompt key: {key}")

    version = await db.prompttemplateversion.find_unique(where={"id": version_id})
    if not version or version.promptKey != key:
        raise KeyError(f"Unknown prompt version for key: {key}")

    redis = await get_redis()
    await redis.set(_redis_key(key), version.content)
    row = await _persist_prompt_update(
        key,
        version.content,
        source="version_restore",
        change_type=f"restore:{version_id}",
    )
    return {
        **asdict(definition),
        "content": version.content,
        "source": "redis",
        "updated_at": row.updatedAt.isoformat() if row else None,
    }
