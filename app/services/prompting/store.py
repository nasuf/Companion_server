"""Database-authoritative prompt storage with transactional audit and fenced cache."""
from __future__ import annotations

import asyncio
import json
import logging
from contextvars import ContextVar
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from prisma import Json
from app.db import db
from app.redis_client import get_redis
from app.services.prompting.registry import PROMPT_DEFINITION_MAP, PROMPT_DEFINITIONS
from app.services.prompting.trace_components import ManagedPromptText

logger = logging.getLogger(__name__)
PROMPT_KEY_PREFIX = "prompt_template:"
PROMPT_ENABLED_KEY_PREFIX = "prompt_enabled:"
_ROOT = Path(__file__).resolve().parents[3]
_EVAL_CASES = _ROOT / "evals" / "cases.jsonl"
_prompt_snapshot: ContextVar[dict | None] = ContextVar("prompt_snapshot", default=None)

class PromptDisabledError(Exception):
    def __init__(self, key: str):
        super().__init__(f"Prompt disabled: {key}")
        self.key = key

class PromptUpdateConflictError(Exception):
    """The version seen by an administrator is no longer current."""


def _redis_key(key: str) -> str:
    return f"{PROMPT_KEY_PREFIX}{key}"


def _enabled_redis_key(key: str) -> str:
    return f"{PROMPT_ENABLED_KEY_PREFIX}{key}"


_CACHE_LUA = """
local revision = tonumber(redis.call('HGET', KEYS[1], 'revision') or '-1')
if redis.call('HGET', KEYS[1], 'prompt_id') == ARGV[4] and revision > tonumber(ARGV[1]) then return 0 end
redis.call('HSET', KEYS[1], 'revision', ARGV[1], 'content', ARGV[2], 'enabled', ARGV[3], 'prompt_id', ARGV[4])
redis.call('SET', KEYS[2], ARGV[2])
redis.call('SET', KEYS[3], ARGV[3])
return 1
"""


async def _sync_cache(row) -> bool:
    """A failed cache write never rolls back a committed save. Older writes lose."""
    try:
        # Redis can evict its revision fence. Validate under a database row lock as
        # well, so a delayed publisher cannot resurrect an older snapshot after eviction.
        async with db.tx() as tx:
            await _lock(tx, row.key)
            current = await tx.prompttemplate.find_unique(where={"key": row.key})
            if current is None or current.id != row.id or current.revision != row.revision:
                return False
            redis = await get_redis()
            result = await redis.eval(_CACHE_LUA, 3, f"prompt_snapshot:{row.key}",
                                      _redis_key(row.key), _enabled_redis_key(row.key),
                                      row.revision, row.content, "1" if row.isEnabled else "0", row.id)
            return bool(result)
    except Exception:
        logger.warning("[PROMPT-CACHE] sync pending key=%s revision=%s", getattr(row, "key", "unknown"), getattr(row, "revision", "unknown"))
        return False


def _response(definition, row, *, cache_synced=True, version_id=None, publication=None) -> dict:
    return {**asdict(definition), "content": row.content if row else definition.default_text,
            "is_enabled": row.isEnabled if row else True, "source": "db" if row else "default",
            "updated_at": row.updatedAt.isoformat() if row else None,
            "revision": row.revision if row else 0, "web_managed": row.webManaged if row else False,
            "cache_synced": cache_synced, "version_id": version_id,
            **(publication or {"content_version_type": "unverified", "web_version": None,
                               "content_version_id": None})}


def _publication_metadata(row, versions, numbers):
    """Only verified content publishes receive a public Web version label."""
    published = [v for v in versions if v.id in numbers]
    if published:
        latest = max(published, key=lambda v: numbers[v.id])
        if row and latest.content == row.content:
            return {"content_version_type": "web", "web_version": numbers[latest.id],
                    "content_version_id": latest.id}
    elif (row is None and not versions) or (row and not history_requires_web(row, versions)):
        matches = [v for v in versions if v.changeType in {"bootstrap", "code_sync"}
                   and row and v.content == row.content]
        latest = max(matches, key=lambda v: (v.createdAt, v.id)) if matches else None
        return {"content_version_type": "default", "web_version": None,
                "content_version_id": latest.id if latest else None}
    return {"content_version_type": "unverified", "web_version": None,
            "content_version_id": None}


async def _publication_data(tx, key=None):
    # Complete history for displayed registry keys, including deleted-row
    # history. Unrelated retired keys must not make this read-only transaction
    # load every historical template body or expire under registry growth.
    where = {"promptKey": key} if key else {"promptKey": {"in": [d.key for d in PROMPT_DEFINITIONS]}}
    versions = await tx.prompttemplateversion.find_many(where=where)
    mappings = await tx.promptpublicationversion.find_many(where=where)
    return versions, {p.versionId: p.number for p in mappings}


def _check_expected(row, expected_updated_at=None, expected_revision=None):
    if expected_revision is not None and (row.revision if row else 0) != expected_revision:
        raise PromptUpdateConflictError("提示词已被修改，请刷新后比对草稿再保存。")
    if expected_updated_at:
        try:
            expected = datetime.fromisoformat(expected_updated_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("Invalid expected_updated_at") from exc
        if not row or expected != row.updatedAt:
            raise PromptUpdateConflictError("提示词已被修改，请刷新后比对草稿再保存。")


async def _lock(tx, key):
    # Serializes bootstrap/save/reset/restore/enable, including a missing row.
    await tx.query_raw("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))::text AS locked", "prompt:" + key)
    # Also serialize writers using the old store (which knows no advisory lock).
    await tx.query_raw("SELECT id FROM prompt_templates WHERE key=$1 FOR UPDATE", key)


def _new_data(definition):
    return {"key": definition.key, "stage": definition.stage, "category": definition.category,
            "title": definition.title, "description": definition.description,
            "content": definition.default_text, "defaultContent": definition.default_text}


async def _version(tx, row, change_type, source):
    return await tx.prompttemplateversion.create(data={"promptId": row.id, "promptKey": row.key,
        "content": row.content, "source": source, "changeType": change_type, "revision": row.revision,
        "createdAt": datetime.now(timezone.utc)})


def _schedule_eval(version):
    from app.services.runtime.tasks import fire_background
    fire_background(_attach_eval_to_version(version.id, version.promptKey, version.changeType))


def get_prompt_snapshot() -> dict | None:
    return _prompt_snapshot.get()


def bind_prompt_snapshot(snapshot: dict | None):
    return _prompt_snapshot.set(snapshot)


def reset_prompt_snapshot(token) -> None:
    _prompt_snapshot.reset(token)


async def is_prompt_enabled(key: str) -> bool:
    if key not in PROMPT_DEFINITION_MAP:
        raise KeyError(f"Unknown prompt key: {key}")
    row = await db.prompttemplate.find_unique(where={"key": key})
    snapshot = _prompt_snapshot.get()
    # A live disable is a safety switch; enabling a prompt mid-turn waits for the next turn.
    return (bool(row.isEnabled) if row else True) and (not snapshot or snapshot.get(key, (None, True))[1])


async def _mutate(key, change_type, *, content=None, enabled=None, version_id=None,
                  expected_updated_at=None, expected_revision=None, publish_version=False):
    definition = PROMPT_DEFINITION_MAP.get(key)
    if not definition:
        raise KeyError(f"Unknown prompt key: {key}")
    version = None
    async with db.tx() as tx:
        await _lock(tx, key)
        row = await tx.prompttemplate.find_unique(where={"key": key})
        _check_expected(row, expected_updated_at, expected_revision)
        if version_id is not None:
            previous = await tx.prompttemplateversion.find_unique(where={"id": version_id})
            if not previous or previous.promptKey != key:
                raise KeyError(f"Unknown prompt version for key: {key}")
            content = previous.content
        # A same-content publication must preserve legacy surrounding whitespace;
        # edited drafts retain the normal Web trimming policy.
        if publish_version and change_type == "manual_save" and row and content == row.content.strip():
            content = row.content
        no_change = row and ((change_type == "manual_save" and content == row.content and not publish_version)
                            or (enabled is not None and enabled == row.isEnabled))
        if not no_change:
            data = {}
            if content is not None:
                data.update(content=content, webManaged=True)
            if enabled is not None:
                data["isEnabled"] = enabled
            if row:
                row = await tx.prompttemplate.update(where={"key": key}, data=data)
            else:
                row = await tx.prompttemplate.create(data={**_new_data(definition), **data})
            version = await _version(tx, row, change_type, "default" if change_type == "reset_default" else "db")
        history, numbers = await _publication_data(tx, key)
        publication = _publication_metadata(row, history, numbers)
    cache_synced = await _sync_cache(row)
    if version:
        _schedule_eval(version)
    return _response(definition, row, cache_synced=cache_synced,
                     version_id=version.id if version else None, publication=publication)


async def update_prompt_text(key: str, content: str, *, expected_updated_at=None, expected_revision=None,
                             publish_version=False):
    normalized = content.strip()
    if not normalized:
        raise ValueError("Prompt content cannot be empty")
    return await _mutate(key, "manual_save", content=normalized,
                         expected_updated_at=expected_updated_at, expected_revision=expected_revision,
                         publish_version=publish_version)


async def set_prompt_enabled(key: str, enabled: bool, *, expected_updated_at=None, expected_revision=None):
    return await _mutate(key, "enable" if enabled else "disable", enabled=enabled,
                         expected_updated_at=expected_updated_at, expected_revision=expected_revision)


async def reset_prompt_text(key: str, *, expected_updated_at=None, expected_revision=None):
    definition = PROMPT_DEFINITION_MAP.get(key)
    if not definition:
        raise KeyError(key)
    return await _mutate(key, "reset_default", content=definition.default_text,
                         expected_updated_at=expected_updated_at, expected_revision=expected_revision)


async def restore_prompt_version(key: str, version_id: str, *, expected_updated_at=None, expected_revision=None):
    return await _mutate(key, f"restore:{version_id}", version_id=version_id,
                         expected_updated_at=expected_updated_at, expected_revision=expected_revision)


def history_requires_web(row, versions) -> bool:
    """Check complete history, never just whether the content differs today.

    A bootstrap recorded long after creation cannot prove earlier edits did not exist.
    The one-minute window tolerates separate transactions used by the old seeder.
    """
    created = getattr(row, "createdAt", None)
    initial = [v for v in versions if v.changeType == "bootstrap" and
               created is not None and getattr(v, "createdAt", None) is not None and
               abs(v.createdAt - created) <= timedelta(minutes=1)]
    return bool(row.webManaged or row.content != row.defaultContent or not initial or any(
        v.changeType not in {"bootstrap", "code_sync", "enable", "disable"} for v in versions))


async def ensure_prompt_templates() -> None:
    # Orphan keys retain their audit history; the registry limits runtime/UI access.
    for definition in PROMPT_DEFINITIONS:
        version = None
        async with db.tx() as tx:
            await _lock(tx, definition.key)
            row = await tx.prompttemplate.find_unique(where={"key": definition.key})
            if not row:
                row = await tx.prompttemplate.create(data=_new_data(definition))
                await _version(tx, row, "bootstrap", "default")
            else:
                # Ownership belongs to the registry key, even after a row is recreated.
                history = await tx.prompttemplateversion.find_many(where={"promptKey": row.key})
                protected = history_requires_web(row, history)
                data = {}
                for name, value in {"stage": definition.stage, "category": definition.category,
                                    "title": definition.title, "description": definition.description}.items():
                    if (getattr(row, name) or "") != (value or ""):
                        data[name] = value
                if protected and not row.webManaged:
                    data["webManaged"] = True
                changed = row.defaultContent != definition.default_text
                if changed:
                    data["defaultContent"] = definition.default_text
                    if not protected:
                        data["content"] = definition.default_text
                    else:
                        logger.info("[PROMPT-SYNC] preserved Web-managed key=%s", row.key)
                if data:
                    row = await tx.prompttemplate.update(where={"key": row.key}, data=data)
                if changed and not protected:
                    version = await _version(tx, row, "code_sync", "default")
                elif not history:
                    await _version(tx, row, "bootstrap", "db")
        await _sync_cache(row)
        if version:
            _schedule_eval(version)


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



async def get_prompt_text(key: str) -> str:
    definition = PROMPT_DEFINITION_MAP.get(key)
    if not definition:
        raise KeyError(f"Unknown prompt key: {key}")
    if not await is_prompt_enabled(key):
        raise PromptDisabledError(key)
    snapshot = _prompt_snapshot.get()
    if snapshot is not None:
        content = snapshot.get(key, (definition.default_text, True))[0]
    else:
        row = await db.prompttemplate.find_unique(where={"key": key})
        # Re-check the row used for content (a disable may race the first read).
        if row and not row.isEnabled:
            raise PromptDisabledError(key)
        content = row.content if row else definition.default_text
        if row:
            await _sync_cache(row)

    from app.services.prompting.reply_prefix import (
        REPLY_PROMPT_KEYS, PROACTIVE_REPLY_PROMPT_KEYS, PROACTIVE_COMMON_KEY,
        build_reply_prefix,
    )

    if key in PROACTIVE_REPLY_PROMPT_KEYS:
        # Keep source components separate so Web trace editing targets the right key.
        try:
            prefix = await get_prompt_text(PROACTIVE_COMMON_KEY)
        except PromptDisabledError:
            prefix = ""
        if prefix:
            return ManagedPromptText.compose([
                (PROACTIVE_COMMON_KEY, str(prefix)), (key, content),
            ], key)
    elif key in REPLY_PROMPT_KEYS:
        try:
            prefix = await build_reply_prefix()
        except Exception as e:  # noqa: BLE001 — 前置故障不能放大成全回复链路故障
            logger.warning(f"[REPLY-PREFIX] build failed for {key}, using bare template: {e}")
            prefix = ""
        if prefix:
            content = f"{prefix}\n\n{content}"

    return ManagedPromptText(content, key, prompt_variant="active")


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
    async with db.tx() as tx:
        await tx.execute_raw("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        rows = await tx.prompttemplate.find_many()
        history, numbers = await _publication_data(tx)
    by_key = {row.key: row for row in rows}
    by_history = {}
    for version in history:
        by_history.setdefault(version.promptKey, []).append(version)
    return [_response(d, by_key.get(d.key), publication=_publication_metadata(
        by_key.get(d.key), by_history.get(d.key, []), numbers)) for d in PROMPT_DEFINITIONS]


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



async def list_prompt_versions(key: str, limit: int = 20) -> list[dict]:
    definition = PROMPT_DEFINITION_MAP.get(key)
    if not definition:
        raise KeyError(f"Unknown prompt key: {key}")

    async with db.tx() as tx:
        await tx.execute_raw("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        versions, numbers = await _publication_data(tx, key)
    versions = sorted(versions, key=lambda v: (v.createdAt, v.id), reverse=True)[:limit]
    return [
        {
            "id": version.id,
            "prompt_key": version.promptKey,
            "content": version.content,
            "source": version.source,
            "change_type": version.changeType,
            "eval_result": _json_or_none(getattr(version, "evalResult", None)),
            "persistence": "synced",
            "revision": version.revision,
            "created_at": version.createdAt.isoformat(),
            "web_version": numbers.get(version.id),
        }
        for version in versions
    ]
