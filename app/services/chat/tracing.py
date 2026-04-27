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

    async def _share(self) -> None:
        """后台：share trace + 更新对应消息 metadata + WS 通知。"""
        # close() 已守 `if self.trace_id:`, 进到这里 trace_id 必非空.
        trace_id = self.trace_id
        assert trace_id is not None
        try:
            public_url = await share_run_with_retry(
                trace_id, conversation_id=self._conversation_id,
            )
            logger.info(f"[TRACE] shared trace_id={trace_id} url={public_url}")
        except Exception:
            # 已重试 3 次仍失败. 用 exception 打完整 stack 便于诊断 (quota?
            # 网络? API 变更?). 把 trace_failed=true 写回 message metadata,
            # 前端据此显示"重试"按钮替代永久 loading.
            logger.exception(
                f"[TRACE] share_run permanently failed after retries "
                f"trace_id={trace_id} conv={self._conversation_id[:8]}"
            )
            await _mark_trace_failed(self._conversation_id, trace_id)
            return

        await _persist_share_success(
            self._conversation_id, trace_id, public_url,
        )


# ─────────────────────────────────────────────────────────────────────
# Module-level helpers — 同时给 LangSmithTracer._share() 和 retry endpoint 用.
# 抽出来是为了消除 "实例方法 vs 单条消息重试" 两条路径的代码重复, 并保证
# 失败时统一打 trace_failed=true 让前端显示重试按钮.
# ─────────────────────────────────────────────────────────────────────


_SHARE_RETRY_DELAYS = (1.0, 2.0, 4.0)


async def share_run_with_retry(
    trace_id: str | None,
    *,
    conversation_id: str = "?",
) -> str:
    """调 LangSmith client.share_run, 失败重试 3 次指数退避 (1s, 2s, 4s).

    share_run 是网络调用 (~500ms-2s 正常, 抽风时 5xx). 不重试时偶发失败
    会让 trace_pending 永远卡在 true, 用户感觉 "trace 时有时无".

    Raises 最后一次异常给调用方 (LangSmithTracer._share / retry endpoint),
    供 logger.exception 打完整 stack.
    """
    if not trace_id:
        raise ValueError("share_run_with_retry requires a non-empty trace_id")
    loop = asyncio.get_running_loop()
    client = get_langsmith_client()
    last_exc: Exception | None = None
    for attempt, delay in enumerate(_SHARE_RETRY_DELAYS, start=1):
        try:
            return await loop.run_in_executor(None, client.share_run, trace_id)
        except Exception as e:
            last_exc = e
            logger.warning(
                f"[TRACE] share_run attempt {attempt}/{len(_SHARE_RETRY_DELAYS)} failed "
                f"trace_id={trace_id} conv={conversation_id[:8]} "
                f"err={type(e).__name__}: {e}"
            )
            if attempt < len(_SHARE_RETRY_DELAYS):
                await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


async def _find_message_by_trace_id(conversation_id: str, trace_id: str):
    """找该会话下 metadata.trace_id 匹配的最近 assistant 消息. 找不到返回 None.

    扫最近 20 条够覆盖正常场景 (一条用户消息 → ≤3 条 split assistant + 同会话
    其他消息); trace_id 只挂在第一条 split 上.
    """
    msgs = await db.message.find_many(
        where={"conversationId": conversation_id, "role": "assistant"},
        order={"createdAt": "desc"},
        take=20,
    )
    for msg in msgs:
        meta = msg.metadata or {}
        if isinstance(meta, dict) and meta.get("trace_id") == trace_id:
            return msg
    return None


async def _patch_message_metadata(message_id: str, current_meta: dict, **patch) -> None:
    """合并 patch 进 metadata 并写库. 保留原有 key (e.g. reply_index, sticker_url)."""
    await db.message.update(
        where={"id": message_id},
        data={"metadata": Json({**current_meta, **patch})},
    )


async def _persist_share_success(
    conversation_id: str, trace_id: str, public_url: str,
) -> str | None:
    """share_run 成功后回填 message metadata + WS 推 trace_ready. 返回 message_id."""
    msg = await _find_message_by_trace_id(conversation_id, trace_id)
    if not msg:
        # share_run 成功了但找不到对应 message — 边缘竞争 (save_replies 还没
        # 落库 / sub_intent 模式还没回 parent). 不致命, 只是按钮不亮.
        logger.warning(
            f"[TRACE] no message found with trace_id={trace_id} "
            f"in conv={conversation_id[:8]} (last 20 assistant msgs)"
        )
        return None
    try:
        await _patch_message_metadata(
            msg.id, msg.metadata or {},
            trace_url=public_url, trace_pending=False, trace_failed=False,
        )
        from app.services.runtime.ws_manager import manager
        await manager.send_event(
            conversation_id,
            "trace_ready",
            {"message_id": msg.id, "trace_url": public_url},
        )
        return msg.id
    except Exception:
        logger.exception(
            f"[TRACE] failed to update message metadata after share "
            f"trace_id={trace_id} conv={conversation_id[:8]}"
        )
        return None


async def _mark_trace_failed(conversation_id: str, trace_id: str | None) -> None:
    """share_run 永久失败后, 在对应 message metadata 上打 trace_failed=true,
    前端据此显示"重试"按钮替代永久 loading.
    """
    if not trace_id:
        return
    msg = await _find_message_by_trace_id(conversation_id, trace_id)
    if not msg:
        return
    try:
        await _patch_message_metadata(
            msg.id, msg.metadata or {},
            trace_failed=True, trace_pending=False,
        )
        from app.services.runtime.ws_manager import manager
        await manager.send_event(
            conversation_id,
            "trace_failed",
            {"message_id": msg.id},
        )
    except Exception:
        logger.exception(
            f"[TRACE] failed to mark trace_failed trace_id={trace_id} "
            f"conv={conversation_id[:8]}"
        )


async def retry_share_for_message(
    message_id: str, *, user_id: str,
) -> dict[str, str]:
    """重试单条消息的 trace 分享. 由 /traces/retry/{message_id} endpoint 调用.

    流程: 校验消息归属当前用户 → 取 metadata.trace_id → share_run + retry →
    成功更新 metadata + WS 通知. 失败抛 ValueError / RuntimeError 让 endpoint
    转 4xx/5xx.
    """
    msg = await db.message.find_unique(
        where={"id": message_id},
        include={"conversation": True},
    )
    if not msg:
        raise ValueError("message_not_found")
    conv = getattr(msg, "conversation", None)
    if not conv or conv.userId != user_id:
        raise PermissionError("not_your_message")

    meta = msg.metadata or {}
    if not isinstance(meta, dict):
        raise ValueError("metadata_invalid")
    trace_id = meta.get("trace_id")
    if not trace_id:
        # 老消息没有 trace_id, 没有底层 LangSmith run 可分享.
        raise ValueError("no_trace_id")
    if meta.get("trace_url"):
        # 已经成功过了, 直接返回现有 url (前端可能拿到旧状态).
        return {"trace_url": str(meta["trace_url"])}

    public_url = await share_run_with_retry(
        trace_id, conversation_id=conv.id,
    )
    await _patch_message_metadata(
        msg.id, meta,
        trace_url=public_url, trace_pending=False, trace_failed=False,
    )
    from app.services.runtime.ws_manager import manager
    await manager.send_event(
        conv.id,
        "trace_ready",
        {"message_id": msg.id, "trace_url": public_url},
    )
    return {"trace_url": public_url}
