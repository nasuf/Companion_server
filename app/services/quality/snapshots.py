"""Bounded immutable prompt and model metadata; no secrets or live histories."""

from datetime import timedelta
import hashlib
import json
from app.db import db
from app.services import runtime_config


async def capture():
    await runtime_config.ensure_loaded()
    async with db.tx(timeout=timedelta(seconds=8)) as tx:
        await tx.execute_raw("SET LOCAL statement_timeout='5s'")
        rows = await tx.query_raw(
            "SELECT key,content,is_enabled,updated_at FROM prompt_templates ORDER BY key LIMIT 501"
        )
    if len(rows) > 500 or len(json.dumps(rows, default=str).encode()) > 1_500_000:
        raise ValueError("prompt_snapshot_exceeds_bound")
    config = runtime_config.resolve_config_sync(None)
    route = [config.remote_chat_provider, config.remote_chat_model]
    if not config.online_model or route[0] not in {"ark", "dashscope"}:
        raise ValueError("quality_route_not_supported")
    value = dict(
        prompts=[
            dict(key=r["key"], content=r["content"], is_enabled=r["is_enabled"])
            for r in rows
        ],
        route=route,
        context_policy=runtime_config.chat_context_policy_overrides(),
        prices={
            key: runtime_config.get_pricing(key)
            for key in {
                "/".join(route),
                "dashscope/qwen3.5-flash",
                "dashscope/qwen3.5-plus",
            }
        },
    )
    value["hash"] = hashlib.sha256(
        json.dumps(value, sort_keys=True).encode()
    ).hexdigest()
    return value
