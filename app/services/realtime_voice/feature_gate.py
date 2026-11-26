"""聊天「语音通话」全局开关 (SystemConfig.voice_call_enabled)。

默认关闭，读库失败也按关闭处理：开关的作用是在功能未就绪时把入口藏起来，
读不到时宁可让用户看不到，也不要把未就绪的通话放出去。
读取不缓存：调用点是通话建立与客户端轮询，频率低，直接读库保证后台切换实时生效。
"""
from __future__ import annotations

import logging

from app.db import db

logger = logging.getLogger(__name__)


async def is_voice_call_enabled() -> bool:
    try:
        row = await db.systemconfig.find_unique(where={"id": 1})
    except Exception:
        logger.exception("voice call gate read failed")
        return False
    return bool(getattr(row, "voiceCallEnabled", False))


async def set_voice_call_enabled(*, enabled: bool, actor_id: str) -> bool:
    await db.systemconfig.upsert(
        where={"id": 1},
        data={
            "create": {"id": 1, "voiceCallEnabled": enabled},
            "update": {"voiceCallEnabled": enabled},
        },
    )
    logger.info(
        "voice call gate updated",
        extra={"event": "voice_call_gate_updated", "enabled": enabled, "actor_id": actor_id},
    )
    return await is_voice_call_enabled()
