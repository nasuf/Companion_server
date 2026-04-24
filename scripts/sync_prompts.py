"""强制把所有 prompt 的 DB.content + DB.defaultContent + Redis 对齐到代码 defaults.

用于清理 UI 历史保存导致的 drift. 跑一次即可, 之后 startup sync 会增量维护.

Usage:
    cd Companion_server
    uv run python scripts/sync_prompts.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.db import db
from app.redis_client import get_redis, close_redis
from app.services.prompting.registry import PROMPT_DEFINITIONS
from app.services.prompting.store import PROMPT_KEY_PREFIX


async def sync() -> None:
    await db.connect()
    redis = await get_redis()
    try:
        print(f"Force-syncing {len(PROMPT_DEFINITIONS)} prompts to code defaults...")
        pipe = redis.pipeline()
        for definition in PROMPT_DEFINITIONS:
            await db.execute_raw(
                'UPDATE "prompt_templates" SET "content" = $1, "default_content" = $1, "updated_at" = NOW() WHERE "key" = $2',
                definition.default_text,
                definition.key,
            )
            pipe.set(f"{PROMPT_KEY_PREFIX}{definition.key}", definition.default_text)
            print(f"  ✓ {definition.key}")
        await pipe.execute()
        print("\nDone. DB.content, DB.default_content, Redis 已全部对齐代码 defaults.py.")
    finally:
        await close_redis()
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(sync())
