"""Prompt storage service backed by Redis + Prisma."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import string
from contextvars import ContextVar, Token
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prisma import Json

from app.db import db
from app.redis_client import get_redis
from app.services.prompting.registry import PROMPT_DEFINITION_MAP, PROMPT_DEFINITIONS, PromptDefinition
from app.services.prompting.trace_components import ManagedPromptText

logger = logging.getLogger(__name__)

PROMPT_KEY_PREFIX = "prompt_template:"
PROMPT_CANARY_KEY_PREFIX = "prompt_canary:"
_ROOT = Path(__file__).resolve().parents[3]
_EVAL_CASES = _ROOT / "evals" / "cases.jsonl"
_prompt_runtime_context: ContextVar[dict[str, str | None] | None] = ContextVar(
    "prompt_runtime_context",
    default=None,
)


def _redis_key(key: str) -> str:
    return f"{PROMPT_KEY_PREFIX}{key}"


def _canary_redis_key(key: str) -> str:
    return f"{PROMPT_CANARY_KEY_PREFIX}{key}"


def set_prompt_runtime_context(
    *,
    agent_id: str | None = None,
    user_id: str | None = None,
) -> Token[dict[str, str | None] | None]:
    return _prompt_runtime_context.set({"agent_id": agent_id, "user_id": user_id})


def reset_prompt_runtime_context(token: Token[dict[str, str | None] | None]) -> None:
    _prompt_runtime_context.reset(token)


def _json_or_none(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _prompt_eval_result(*, prompt_key: str, change_type: str) -> dict[str, Any]:
    """CI-safe eval gate snapshot attached to prompt changes."""
    try:
        from evals.long_companion_sim import build_reference_transcript, score_transcript, validate_transcript
        from evals.run_local import load_cases, validate_cases

        cases = load_cases(_EVAL_CASES)
        validation_failures = validate_cases(cases)
        long_rows = build_reference_transcript()
        long_errors = validate_transcript(long_rows)
        long_result = score_transcript(long_rows)
        ok = not validation_failures and not long_errors and bool(long_result.get("passed"))
        return {
            "mode": "validate_only",
            "ok": ok,
            "prompt_key": prompt_key,
            "change_type": change_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "agent_eval": {
                "validated_cases": len(cases),
                "validation_failures": validation_failures,
            },
            "long_companion": {
                "validation_errors": long_errors,
                **long_result,
            },
        }
    except Exception as exc:
        logger.warning("[PROMPT-EVAL] validate snapshot failed key=%s: %s", prompt_key, exc)
        return {
            "mode": "validate_only",
            "ok": False,
            "prompt_key": prompt_key,
            "change_type": change_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "error": type(exc).__name__,
            "message": str(exc),
        }


def _stable_bucket(*values: str | None) -> int:
    raw = "|".join(str(v or "") for v in values)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def _template_fields(text: str) -> set[str]:
    fields: set[str] = set()
    for _, field_name, _, _ in string.Formatter().parse(text):
        if field_name:
            fields.add(field_name.split(".", 1)[0].split("[", 1)[0])
    return fields


def _missing_required_placeholders(reference: str, candidate: str) -> list[str]:
    return sorted(_template_fields(reference) - _template_fields(candidate))


def _normalize_canary_config(
    *,
    prompt_key: str,
    is_enabled: bool,
    mode: str,
    content: str | None,
    agent_ids: list[str] | None = None,
    rollout_percent: int = 0,
    eval_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_mode = mode if mode in {"off", "agents", "percent"} else "off"
    normalized_content = (content or "").strip()
    ids = sorted({str(item).strip() for item in (agent_ids or []) if str(item).strip()})
    percent = max(0, min(100, int(rollout_percent or 0)))
    enabled = bool(is_enabled) and normalized_mode != "off"
    if enabled and not normalized_content:
        raise ValueError("Canary content cannot be empty when enabled")
    if normalized_mode == "agents" and enabled and not ids:
        raise ValueError("agent_ids required for agents canary")
    if normalized_mode == "percent" and enabled and percent <= 0:
        raise ValueError("rollout_percent must be greater than 0 for percent canary")
    if normalized_mode == "off":
        enabled = False
        percent = 0
    return {
        "prompt_key": prompt_key,
        "is_enabled": enabled,
        "mode": normalized_mode,
        "content": normalized_content or None,
        "agent_ids": ids,
        "rollout_percent": percent,
        "eval_result": eval_result,
    }


def _canary_matches(config: dict[str, Any], *, agent_id: str | None, user_id: str | None) -> bool:
    if not config.get("is_enabled") or not config.get("content"):
        return False
    mode = str(config.get("mode") or "off")
    if mode == "agents":
        return bool(agent_id and agent_id in set(config.get("agent_ids") or []))
    if mode == "percent":
        percent = int(config.get("rollout_percent") or 0)
        if percent <= 0:
            return False
        return _stable_bucket(config.get("prompt_key"), agent_id, user_id) < percent
    return False


async def _load_canary_config(key: str) -> dict[str, Any] | None:
    redis = await get_redis()
    cached = await redis.get(_canary_redis_key(key))
    cached_config = _json_or_none(cached)
    if cached_config is not None:
        return cached_config

    rows = await db.query_raw(
        """
        SELECT canary_config
        FROM prompt_templates
        WHERE key = $1
        LIMIT 1
        """,
        key,
    )
    config = _json_or_none(rows[0].get("canary_config")) if rows else None
    if config is None:
        config = _normalize_canary_config(
            prompt_key=key,
            is_enabled=False,
            mode="off",
            content=None,
        )
    await redis.set(_canary_redis_key(key), json.dumps(config, ensure_ascii=False), ex=300)
    return config


async def ensure_prompt_templates() -> None:
    """Ensure prompt templates exist in DB and warm Redis from DB state.

    Also reconciles orphan rows — any key in DB/Redis that is no longer in
    `PROMPT_DEFINITIONS` gets deleted so the data source stays in sync with code.

    Optimized: batch-load all existing prompts + versions in 2 queries,
    then only write to DB for missing/changed entries.
    """
    # Batch load: 1 query for all prompts, 1 for version counts
    all_existing = await db.prompttemplate.find_many()
    existing_map = {t.key: t for t in all_existing}

    # Get prompt IDs that have at least one version (single query)
    version_prompt_ids: set[str] = set()
    if all_existing:
        versions = await db.query_raw(
            "SELECT DISTINCT prompt_id FROM prompt_template_versions",
        )
        version_prompt_ids = {str(v["prompt_id"]) for v in versions}

    redis = await get_redis()
    pipe = redis.pipeline()

    # Reconcile orphans: DB keys not in registry → delete (rows + versions + Redis)
    registry_keys = {d.key for d in PROMPT_DEFINITIONS}
    orphan_keys = [k for k in existing_map if k not in registry_keys]
    if orphan_keys:
        logger.warning(
            "Deleting %d orphan prompt template(s) not in registry: %s",
            len(orphan_keys), ", ".join(sorted(orphan_keys)),
        )
        await db.prompttemplateversion.delete_many(where={"promptKey": {"in": orphan_keys}})
        await db.prompttemplate.delete_many(where={"key": {"in": orphan_keys}})
        for k in orphan_keys:
            pipe.delete(_redis_key(k))

    # 代码 default 视为 prompt 终极真理: defaultContent 与代码不一致时同步覆盖
    # content, UI 定制作废; default 未变期间保留 UI 定制 (update_prompt_text
    # 写 Redis 立即生效).
    code_sync_keys: list[str] = []
    for definition in PROMPT_DEFINITIONS:
        existing = existing_map.get(definition.key)
        if existing:
            default_changed = existing.defaultContent != definition.default_text
            metadata_changed = (
                existing.stage != definition.stage
                or existing.category != definition.category
                or existing.title != definition.title
                or (existing.description or "") != definition.description
            )
            metadata_data = {
                "stage": definition.stage,
                "category": definition.category,
                "title": definition.title,
                "description": definition.description,
            }

            if default_changed:
                content = definition.default_text
                await db.prompttemplate.update(
                    where={"key": definition.key},
                    data={
                        **metadata_data,
                        "content": definition.default_text,
                        "defaultContent": definition.default_text,
                    },
                )
                await _create_prompt_version(
                    prompt_id=existing.id,
                    prompt_key=definition.key,
                    content=definition.default_text,
                    source="default",
                    change_type="code_sync",
                    attach_eval=True,
                )
                code_sync_keys.append(definition.key)
                logger.info(f"[PROMPT-SYNC] key={definition.key} overridden by code default")
            else:
                content = existing.content
                if metadata_changed:
                    await db.prompttemplate.update(
                        where={"key": definition.key},
                        data=metadata_data,
                    )

            if existing.id not in version_prompt_ids:
                # 已有行无 version 记录 (早期 seed 时版本表还没引入), 补一条作起点
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
        pipe.set(_redis_key(definition.key), content)

    await pipe.execute()
    if code_sync_keys:
        logger.info(
            f"[PROMPT-SYNC] code_sync_count={len(code_sync_keys)} "
            f"keys={sorted(code_sync_keys)}"
        )


async def get_prompt_text(key: str) -> str:
    ctx = _prompt_runtime_context.get() or {}
    return await get_prompt_text_for_context(
        key,
        agent_id=ctx.get("agent_id"),
        user_id=ctx.get("user_id"),
    )


async def get_prompt_text_for_context(
    key: str,
    *,
    agent_id: str | None = None,
    user_id: str | None = None,
) -> str:
    """Fetch latest prompt text from Redis, falling back to DB/default."""
    definition = PROMPT_DEFINITION_MAP.get(key)
    if not definition:
        raise KeyError(f"Unknown prompt key: {key}")

    if agent_id or user_id:
        config = await _load_canary_config(key)
        if config and _canary_matches(config, agent_id=agent_id, user_id=user_id):
            return ManagedPromptText(str(config["content"]), key, prompt_variant="canary")

    redis = await get_redis()
    cached = await redis.get(_redis_key(key))
    if cached:
        return ManagedPromptText(cached, key)

    record = await db.prompttemplate.find_unique(where={"key": key})
    content = record.content if record and record.content else definition.default_text
    await redis.set(_redis_key(key), content)
    return ManagedPromptText(content, key)


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
                "canary_config": _json_or_none(getattr(row, "canaryConfig", None)) if row else None,
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
    attach_eval: bool = False,
) -> None:
    eval_result = _prompt_eval_result(
        prompt_key=prompt_key,
        change_type=change_type,
    ) if attach_eval else None
    await db.prompttemplateversion.create(
        data={
            "promptId": prompt_id,
            "promptKey": prompt_key,
            "content": content,
            "source": source,
            "changeType": change_type,
            "evalResult": Json(eval_result) if eval_result is not None else None,
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
            "eval_result": _json_or_none(getattr(version, "evalResult", None)),
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
            # defaultContent 是 ensure_prompt_templates 的同步哨兵 (上次 startup 时
            # 代码 default 的快照), 只归 bootstrap + startup sync 写. UI 保存 / reset
            # / restore 路径不能触它, 否则下次代码改 default 时 sync 认为"哨兵对得
            # 上" → 放弃覆盖 → UI 永远停在旧版本.
            row = await db.prompttemplate.update(
                where={"key": key},
                data={
                    "content": content,
                    "stage": definition.stage,
                    "category": definition.category,
                    "title": definition.title,
                    "description": definition.description,
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
            attach_eval=change_type != "bootstrap",
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
    missing = _missing_required_placeholders(definition.default_text, normalized)
    if missing:
        raise ValueError(
            "Prompt content is missing required placeholders: "
            + ", ".join(f"{{{name}}}" for name in missing)
        )

    redis = await get_redis()
    await redis.set(_redis_key(key), normalized)
    canary_config = await _load_canary_config(key)
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
        "canary_config": canary_config,
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
        "canary_config": _json_or_none(getattr(row, "canaryConfig", None)),
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
        "canary_config": _json_or_none(getattr(row, "canaryConfig", None)),
        "source": "redis",
        "updated_at": row.updatedAt.isoformat() if row else None,
    }


async def get_prompt_canary_config(key: str) -> dict[str, Any]:
    if key not in PROMPT_DEFINITION_MAP:
        raise KeyError(f"Unknown prompt key: {key}")
    config = await _load_canary_config(key)
    return config or _normalize_canary_config(
        prompt_key=key,
        is_enabled=False,
        mode="off",
        content=None,
    )


async def update_prompt_canary_config(
    key: str,
    *,
    is_enabled: bool,
    mode: str,
    content: str | None,
    agent_ids: list[str] | None = None,
    rollout_percent: int = 0,
) -> dict[str, Any]:
    definition = PROMPT_DEFINITION_MAP.get(key)
    if not definition:
        raise KeyError(f"Unknown prompt key: {key}")

    eval_result = _prompt_eval_result(prompt_key=key, change_type="canary_update")
    if is_enabled:
        missing = _missing_required_placeholders(definition.default_text, content or "")
        if missing:
            raise ValueError(
                "Canary content is missing required placeholders: "
                + ", ".join(f"{{{name}}}" for name in missing)
            )
    config = _normalize_canary_config(
        prompt_key=key,
        is_enabled=is_enabled,
        mode=mode,
        content=content,
        agent_ids=agent_ids,
        rollout_percent=rollout_percent,
        eval_result=eval_result,
    )

    row = await db.prompttemplate.find_unique(where={"key": key})
    if row:
        await db.prompttemplate.update(
            where={"key": key},
            data={"canaryConfig": Json(config)},
        )
        updated_at = datetime.now(timezone.utc).isoformat()
    else:
        created = await db.prompttemplate.create(
            data={
                "key": key,
                "stage": definition.stage,
                "category": definition.category,
                "title": definition.title,
                "description": definition.description,
                "content": definition.default_text,
                "defaultContent": definition.default_text,
                "canaryConfig": Json(config),
                "isEnabled": True,
            }
        )
        updated_at = created.updatedAt.isoformat() if getattr(created, "updatedAt", None) else None

    redis = await get_redis()
    await redis.set(_canary_redis_key(key), json.dumps(config, ensure_ascii=False), ex=300)
    return {**config, "updated_at": updated_at}


async def disable_prompt_canary(key: str) -> dict[str, Any]:
    return await update_prompt_canary_config(
        key,
        is_enabled=False,
        mode="off",
        content=None,
        agent_ids=[],
        rollout_percent=0,
    )
