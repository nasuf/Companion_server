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
from app.services.runtime.tasks import fire_background as _fire_background

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
        # is_active=True 但 run_tree=None 表示 LangSmith SDK 初始化失败 (网络/认证).
        # 记录下来便于诊断 "trace 偶尔消失" — 此后整条请求不会有 trace, 消息
        # metadata 也不会带 trace_id, 前端永远不显示按钮.
        if self.is_active and self.trace_id is None:
            logger.warning(
                f"[TRACE] LangSmith ls_trace returned no run_tree "
                f"(conv={self._conversation_id[:8]}); trace will be missing for this request"
            )
        return self

    def close(self) -> None:
        """关闭 trace ctx，启用时 fire-and-forget share。幂等。"""
        if self._closed or self._ctx is None:
            return
        self._closed = True
        self._ctx.__exit__(None, None, None)
        if self.is_active and self.trace_id:
            _fire_background(self._share())

    async def _share_run_with_retry(self) -> str:
        """调 client.share_run, 失败重试 3 次指数退避 (1s, 2s, 4s).

        share_run 是网络调用 (~500ms-2s 正常, 抽风时 5xx). 不重试时偶发失败
        会让 trace_pending 永远卡在 true, 用户感觉 "trace 时有时无".
        """
        loop = asyncio.get_running_loop()
        client = get_langsmith_client()
        delays = (1.0, 2.0, 4.0)
        last_exc: Exception | None = None
        for attempt, delay in enumerate(delays, start=1):
            try:
                return await loop.run_in_executor(None, client.share_run, self.trace_id)
            except Exception as e:
                last_exc = e
                logger.warning(
                    f"[TRACE] share_run attempt {attempt}/{len(delays)} failed "
                    f"trace_id={self.trace_id} err={type(e).__name__}: {e}"
                )
                if attempt < len(delays):
                    await asyncio.sleep(delay)
        # 全失败, 抛最后一次异常给上层 logger.exception 捕获 stack
        assert last_exc is not None
        raise last_exc

    async def _share(self) -> None:
        """后台：share trace + 更新对应消息 metadata + WS 通知。"""
        try:
            public_url = await self._share_run_with_retry()
            logger.info(f"[TRACE] shared trace_id={self.trace_id} url={public_url}")
        except Exception:
            # 已重试 3 次仍失败. 用 exception 打完整 stack 便于诊断 (quota?
            # 网络? API 变更?). 消息的 trace_pending 留 true, 前端永远等不到
            # trace_ready — 这是已知不修复路径 (重启服务也补不回来).
            logger.exception(
                f"[TRACE] share_run permanently failed after retries "
                f"trace_id={self.trace_id} conv={self._conversation_id[:8]}"
            )
            return

        try:
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
                await manager.send_event(
                    self._conversation_id,
                    "trace_ready",
                    {
                        "message_id": updated_message_id,
                        "trace_url": public_url,
                    },
                )
            else:
                # share_run 成功了但找不到对应 message — 边缘竞争 (save_replies
                # 还没落库 / sub_intent 模式还没回 parent). 不致命, 只是按钮不亮.
                logger.warning(
                    f"[TRACE] no message found with trace_id={self.trace_id} "
                    f"in conv={self._conversation_id[:8]} (last 20 assistant msgs)"
                )
        except Exception:
            logger.exception(
                f"[TRACE] failed to update message metadata after share "
                f"trace_id={self.trace_id} conv={self._conversation_id[:8]}"
            )
