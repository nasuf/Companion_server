"""LangSmith trace 全生命周期管理。

把原 orchestrator 散落的 4 个函数（_langsmith_trace_ctx / _get_langsmith_client /
_bg_share_trace / _end_trace）收敛成一个 `LangSmithTracer` 类，并把下游 phase
ctx 中的 `(trace_ctx, trace_id, end_trace_fn)` 三元组合并为单一 `tracer` 字段。

用法：
    tracer = LangSmithTracer(user_message, conversation_id).enter()
    try:
        ...                # 主流程
    finally:
        tracer.close()     # 关闭 ctx + 后台 share

`tracer.is_active`：是否启用 LangSmith。关闭时 enter/close 都是 no-op。
`tracer.trace_id`：本次 trace 的 ID（启用时填充，下游写消息 metadata 用）。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from prisma import Json

from app.config import settings
from app.db import db

logger = logging.getLogger(__name__)


def get_langsmith_client():
    """LangSmith Client 单例。仅在 enabled 时调用。"""
    cache = getattr(get_langsmith_client, "_instance", None)
    if cache is not None:
        return cache
    from langsmith import Client
    instance = Client()
    get_langsmith_client._instance = instance  # type: ignore[attr-defined]
    return instance


def _fire_background(coro) -> None:
    task = asyncio.create_task(coro)
    task.add_done_callback(
        lambda t: None if t.cancelled() or not t.exception()
        else logger.warning(f"Background trace task failed: {t.exception()}")
    )


class LangSmithTracer:
    """spec-orthogonal: 聊天请求的 LangSmith trace 全生命周期管理。

    设计为同步 enter/close 而非 `__enter__/__exit__`，因为 stream_chat_response
    是 async generator，无法直接套 `with` 语句。
    """

    def __init__(self, user_message: str, conversation_id: str) -> None:
        self._user_message = user_message
        self._conversation_id = conversation_id
        self._ctx: Any = None
        self._closed = False
        self.trace_id: str | None = None

    @property
    def is_active(self) -> bool:
        return bool(settings.langsmith_tracing)

    def enter(self) -> "LangSmithTracer":
        """打开 trace ctx，填充 trace_id。"""
        if self.is_active:
            from langsmith import trace as ls_trace
            self._ctx = ls_trace(
                name="chat_request",
                run_type="chain",
                inputs={
                    "message": self._user_message,
                    "conversation_id": self._conversation_id,
                },
                project_name="ai-companion",
            )
        else:
            self._ctx = contextlib.nullcontext()
        run_tree = self._ctx.__enter__()
        self.trace_id = str(run_tree.id) if run_tree else None
        return self

    def close(self) -> None:
        """关闭 trace ctx，启用时 fire-and-forget share。幂等。"""
        if self._closed or self._ctx is None:
            return
        self._closed = True
        self._ctx.__exit__(None, None, None)
        if self.is_active and self.trace_id:
            _fire_background(self._share())

    async def _share(self) -> None:
        """后台：share trace + 更新对应消息 metadata + WS 通知。"""
        try:
            loop = asyncio.get_running_loop()
            client = get_langsmith_client()
            public_url = await loop.run_in_executor(
                None, client.share_run, self.trace_id,
            )
            logger.info(f"Trace shared: {public_url}")

            updated_message_id: str | None = None
            msgs = await db.message.find_many(
                where={"conversationId": self._conversation_id, "role": "assistant"},
                order={"createdAt": "desc"},
                take=20,
            )
            for msg in msgs:
                meta = msg.metadata or {}
                if isinstance(meta, dict) and meta.get("trace_id") == self.trace_id:
                    await db.message.update(
                        where={"id": msg.id},
                        data={
                            "metadata": Json({
                                **meta,
                                "trace_url": public_url,
                                "trace_pending": False,
                            })
                        },
                    )
                    updated_message_id = msg.id
                    break

            if updated_message_id:
                from app.services.runtime.ws_manager import manager
                ws = manager.get(self._conversation_id)
                if ws:
                    await ws.send_json({
                        "type": "trace_ready",
                        "data": {
                            "message_id": updated_message_id,
                            "trace_url": public_url,
                        },
                    })
        except Exception as e:
            logger.warning(f"Failed to share trace: {e}")
