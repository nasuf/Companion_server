"""Prompt storage service backed by Redis + Prisma."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import string
import time
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
PROMPT_ENABLED_KEY_PREFIX = "prompt_enabled:"

# enabled 状态的进程内缓存 TTL. 热路径每次 get_prompt_text 都要判断 enabled,
# 不能每次打 Redis; 10s 内以本进程缓存为准 (多 worker 下停用最多 10s 后全量生效,
# 跟 canary 配置的 5min Redis TTL 相比已经严格得多).
_ENABLED_LOCAL_TTL_SECONDS = 10.0
_enabled_local_cache: dict[str, tuple[bool, float]] = {}


class PromptDisabledError(Exception):
    """Raised when a prompt template is disabled by admin.

    - render_prompt 捕获后返回 None (调用方已有 fallback 语义).
    - prompt_builder 捕获后跳过对应 section (从最终模型输入中彻底移除).
    """

    def __init__(self, key: str):
        super().__init__(f"Prompt disabled: {key}")
        self.key = key


class PromptUpdateConflictError(Exception):
    """Raised when an optimistic-lock save detects a concurrent modification."""
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


def _enabled_redis_key(key: str) -> str:
    return f"{PROMPT_ENABLED_KEY_PREFIX}{key}"


def _cache_enabled_local(key: str, enabled: bool) -> None:
    _enabled_local_cache[key] = (enabled, time.monotonic() + _ENABLED_LOCAL_TTL_SECONDS)


async def is_prompt_enabled(key: str) -> bool:
    """Return the admin enable/disable state (local cache → Redis → DB)."""
    if key not in PROMPT_DEFINITION_MAP:
        raise KeyError(f"Unknown prompt key: {key}")
    cached = _enabled_local_cache.get(key)
    if cached is not None and cached[1] > time.monotonic():
        return cached[0]

    redis = await get_redis()
    raw = await redis.get(_enabled_redis_key(key))
    if raw is None:
        record = await db.prompttemplate.find_unique(where={"key": key})
        enabled = bool(record.isEnabled) if record else True
        await redis.set(_enabled_redis_key(key), "1" if enabled else "0")
    else:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "ignore")
        enabled = str(raw) != "0"
    _cache_enabled_local(key, enabled)
    return enabled


async def set_prompt_enabled(key: str, enabled: bool) -> dict:
    """Persist enable/disable to DB + Redis and record an audit version entry."""
    definition = PROMPT_DEFINITION_MAP.get(key)
    if not definition:
        raise KeyError(f"Unknown prompt key: {key}")

    row = await db.prompttemplate.find_unique(where={"key": key})
    if row:
        row = await db.prompttemplate.update(
            where={"key": key},
            data={"isEnabled": enabled},
        )
    else:
        row = await db.prompttemplate.create(
            data={
                "key": key,
                "stage": definition.stage,
                "category": definition.category,
                "title": definition.title,
                "description": definition.description,
                "content": definition.default_text,
                "defaultContent": definition.default_text,
                "isEnabled": enabled,
            }
        )

    redis = await get_redis()
    await redis.set(_enabled_redis_key(key), "1" if enabled else "0")
    _cache_enabled_local(key, enabled)

    # enable/disable 也进版本表留审计痕迹 (content 记录当时生效内容, 便于追溯).
    await _create_prompt_version(
        prompt_id=row.id,
        prompt_key=key,
        content=row.content,
        source="redis",
        change_type="enable" if enabled else "disable",
    )
    logger.info("[PROMPT-ENABLED] key=%s enabled=%s", key, enabled)

    cached = await redis.get(_redis_key(key))
    return {
        **asdict(definition),
        "content": cached or row.content,
        "is_enabled": enabled,
        "canary_config": _json_or_none(getattr(row, "canaryConfig", None)),
        "updated_at": row.updatedAt.isoformat() if getattr(row, "updatedAt", None) else None,
        "source": "redis" if cached else "db",
    }


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


# Cosmetic reply-formatting placeholders that the runtime *supplies* but the
# admin may legitimately drop (e.g. hardcoding "最多3条/每条15字" instead of
# {n}/{max_per}/{total}). Rendering tolerates unused params, so removing these
# is safe. Data-input placeholders ({message}/{context}/{new_conversation}/…)
# are NOT in this set and stay required — dropping them would silently break
# the prompt (LLM never sees the user input).
_OPTIONAL_TEMPLATE_PLACEHOLDERS = {"n", "max_per", "total", "max_total", "max_reply"}


def _template_fields(text: str) -> set[str]:
    fields: set[str] = set()
    for _, field_name, _, _ in string.Formatter().parse(text):
        if field_name:
            fields.add(field_name.split(".", 1)[0].split("[", 1)[0])
    return fields


def _missing_required_placeholders(reference: str, candidate: str) -> list[str]:
    """Placeholders the default has but the new content dropped, excluding the
    cosmetic reply-formatting ones an admin may intentionally remove.

    The admin already verifies output via the trace replay before saving, and
    rendering is crash-safe for missing params, so only *data-input*
    placeholders are treated as required.
    """
    required = _template_fields(reference) - _OPTIONAL_TEMPLATE_PLACEHOLDERS
    return sorted(required - _template_fields(candidate))


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
            pipe.delete(_enabled_redis_key(k))

    # 代码 default 视为 prompt 终极真理: defaultContent 与代码不一致时同步覆盖
    # content, UI 定制作废; default 未变期间保留 UI 定制 (update_prompt_text
    # 写 Redis 立即生效).
    code_sync_keys: list[str] = []
    for definition in PROMPT_DEFINITIONS:
        existing = existing_map.get(definition.key)
        pipe.set(
            _enabled_redis_key(definition.key),
            "0" if (existing and not existing.isEnabled) else "1",
        )
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
    """Fetch latest prompt text from Redis, falling back to DB/default.

    Raises PromptDisabledError when the template is disabled by admin —
    callers must treat that as "本段/本功能提示词彻底不存在".

    回复类模板 (reply_prefix.REPLY_PROMPT_KEYS) 在此统一注入固定前置
    (通用回复规则 + 反幻觉) — 所有 AI 用户可见消息共享同一套核心规则,
    包括主动消息. 前置来源模板自身不在集合内 (防递归).
    """
    definition = PROMPT_DEFINITION_MAP.get(key)
    if not definition:
        raise KeyError(f"Unknown prompt key: {key}")

    if not await is_prompt_enabled(key):
        raise PromptDisabledError(key)

    variant = "active"
    content: str | None = None
    if agent_id or user_id:
        config = await _load_canary_config(key)
        if config and _canary_matches(config, agent_id=agent_id, user_id=user_id):
            content = str(config["content"])
            variant = "canary"

    if content is None:
        redis = await get_redis()
        cached = await redis.get(_redis_key(key))
        if cached:
            content = cached
        else:
            record = await db.prompttemplate.find_unique(where={"key": key})
            content = record.content if record and record.content else definition.default_text
            await redis.set(_redis_key(key), content)

    from app.services.prompting.reply_prefix import REPLY_PROMPT_KEYS, build_reply_prefix

    if key in REPLY_PROMPT_KEYS:
        try:
            prefix = await build_reply_prefix(agent_id=agent_id, user_id=user_id)
        except Exception as e:  # noqa: BLE001 — 前置故障不能放大成全回复链路故障
            logger.warning(f"[REPLY-PREFIX] build failed for {key}, using bare template: {e}")
            prefix = ""
        if prefix:
            content = f"{prefix}\n\n{content}"

    return ManagedPromptText(content, key, prompt_variant=variant)


async def get_prompt_text_or_default(key: str) -> str:
    """Like get_prompt_text, but disabled prompts fall back to code default.

    仅用于「停用会让必需链路断裂」的结构性兜底指令 (如终结意图兜底/作息缺上下文),
    这些场景必须有文本可用; 停用语义退化为「回到代码默认文案」。
    普通模板请用 get_prompt_text 并处理 PromptDisabledError.
    """
    try:
        return await get_prompt_text(key)
    except PromptDisabledError:
        definition = PROMPT_DEFINITION_MAP[key]
        logger.info("[PROMPT-DISABLED] structural key=%s falls back to code default", key)
        return ManagedPromptText(definition.default_text, key, prompt_variant="default")


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


async def _attach_eval_to_version(version_id: str, prompt_key: str, change_type: str) -> None:
    """Background eval snapshot — 版本行先落库保证持久性, eval 慢跑后回填."""
    try:
        eval_result = await asyncio.to_thread(
            _prompt_eval_result,
            prompt_key=prompt_key,
            change_type=change_type,
        )
        await db.prompttemplateversion.update(
            where={"id": version_id},
            data={"evalResult": Json(eval_result)},
        )
    except Exception as exc:
        logger.warning("[PROMPT-EVAL] attach failed version=%s key=%s: %s", version_id, prompt_key, exc)


async def _create_prompt_version(
    *,
    prompt_id: str,
    prompt_key: str,
    content: str,
    source: str,
    change_type: str,
    attach_eval: bool = False,
) -> None:
    version = await db.prompttemplateversion.create(
        data={
            "promptId": prompt_id,
            "promptKey": prompt_key,
            "content": content,
            "source": source,
            "changeType": change_type,
        }
    )
    if attach_eval:
        # eval 快照可能要跑本地模拟 (秒级), 移出保存关键路径; 失败只丢 eval 徽章,
        # 不影响版本记录本身的持久性. fire_background 统一处理错误日志/取消/
        # request-scoped ContextVar 隔离.
        from app.services.runtime.tasks import fire_background

        fire_background(_attach_eval_to_version(version.id, prompt_key, change_type))


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
    require_updated_at: Any = None,
) -> Any:
    """Persist content + version row.

    require_updated_at: 乐观锁的原子形态 — 提供时用条件 update_many
    (where key AND updatedAt) 代替无条件 update, 命中 0 行说明校验与写入的
    间隙有并发修改, 抛 PromptUpdateConflictError. 仅 check-then-act 的
    find_unique 比对挡不住毫秒级并发双写.
    """
    definition = PROMPT_DEFINITION_MAP[key]
    try:
        existing = await db.prompttemplate.find_unique(where={"key": key})
        if existing:
            # defaultContent 是 ensure_prompt_templates 的同步哨兵 (上次 startup 时
            # 代码 default 的快照), 只归 bootstrap + startup sync 写. UI 保存 / reset
            # / restore 路径不能触它, 否则下次代码改 default 时 sync 认为"哨兵对得
            # 上" → 放弃覆盖 → UI 永远停在旧版本.
            data = {
                "content": content,
                "stage": definition.stage,
                "category": definition.category,
                "title": definition.title,
                "description": definition.description,
            }
            if require_updated_at is not None:
                count = await db.prompttemplate.update_many(
                    where={"key": key, "updatedAt": require_updated_at},
                    data=data,
                )
                if not count:
                    raise PromptUpdateConflictError(
                        f"Prompt {key} was modified concurrently during save"
                    )
                row = await db.prompttemplate.find_unique(where={"key": key})
            else:
                row = await db.prompttemplate.update(
                    where={"key": key},
                    data=data,
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


async def update_prompt_text(
    key: str,
    content: str,
    *,
    expected_updated_at: str | None = None,
) -> dict:
    """Persist prompt update: Redis 先写立即生效, DB + 版本记录同步落库.

    历史实现 DB 持久化是 fire-and-forget task, 存在两类丢失窗口:
    1. 进程在 task 完成前崩溃 → 版本历史缺条, 且下次启动 Redis 被 DB 旧值覆盖;
    2. 并发保存 task 完成顺序不定 → DB 内容与 Redis 分叉.
    现改为: 校验 → 内容去重 → (可选) 乐观锁 → Redis 写入 → DB 同步落库,
    DB 失败时回滚 Redis 并抛错, 保证 Redis/DB/版本表三者一致.

    expected_updated_at: 前端携带其所见的 updated_at 快照; 与 DB 当前值不一致说明
    有人并发改过 → 抛 PromptUpdateConflictError (API 层转 409), 防止静默互相覆盖.
    """
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

    existing = await db.prompttemplate.find_unique(where={"key": key})
    if (
        expected_updated_at
        and existing
        and existing.updatedAt
        and existing.updatedAt.isoformat() != expected_updated_at
    ):
        raise PromptUpdateConflictError(
            f"Prompt {key} was modified by someone else at {existing.updatedAt.isoformat()}"
        )

    redis = await get_redis()
    previous_cached = await redis.get(_redis_key(key))
    current_effective = previous_cached or (
        existing.content if existing and existing.content else definition.default_text
    )
    canary_config = await _load_canary_config(key)

    if (
        normalized == current_effective
        and existing is not None
        and existing.content == normalized
    ):
        # 内容没变: 不产生重复版本记录, 直接返回当前状态.
        # 必须同时要求 DB 一致 — 若 Redis 与 DB 分叉 (历史异步落库失败的存量),
        # 管理员原样保存当前可见内容应当落 DB 修复分叉, 而不是静默 no-op
        # (否则下次重启 ensure_prompt_templates 会用 DB 旧值覆盖 Redis 回退文案).
        return {
            **asdict(definition),
            "content": normalized,
            "is_enabled": bool(existing.isEnabled),
            "canary_config": canary_config,
            "updated_at": existing.updatedAt.isoformat() if existing.updatedAt else None,
            "source": "redis" if previous_cached else "db",
        }

    await redis.set(_redis_key(key), normalized)
    try:
        row = await _persist_prompt_update(
            key,
            normalized,
            source="redis",
            change_type="manual_save",
            # 带乐观锁请求时把校验下推为原子条件更新, 覆盖 find_unique 比对
            # 与 update 之间的并发窗口.
            require_updated_at=(
                existing.updatedAt if expected_updated_at and existing else None
            ),
        )
    except Exception:
        # DB 落库失败 / 并发冲突 → 回滚 Redis, 保持一致性 (宁可保存失败也不要静默分叉).
        if previous_cached is not None:
            await redis.set(_redis_key(key), previous_cached)
        else:
            await redis.delete(_redis_key(key))
        raise

    return {
        **asdict(definition),
        "content": normalized,
        "is_enabled": bool(getattr(row, "isEnabled", True)),
        "canary_config": canary_config,
        "updated_at": row.updatedAt.isoformat() if getattr(row, "updatedAt", None) else None,
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
        # is_enabled 必须回传真实 DB 值: PromptTemplateResponse 默认 True,
        # 漏传会让前端把已停用模板显示成已启用 (UI 与运行时状态分叉).
        "is_enabled": bool(getattr(row, "isEnabled", True)),
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
        "is_enabled": bool(getattr(row, "isEnabled", True)),
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
