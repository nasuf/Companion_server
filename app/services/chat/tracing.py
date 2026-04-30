"""LangSmith trace 全生命周期管理.

用法:
    tracer = LangSmithTracer(user_message, conversation_id).enter()
    try:
        ...                # 主流程 — LLM 调用自动 attach 到此 trace
    finally:
        tracer.close()     # exit ctx; share + mirror 由用户点击时懒触发

`tracer.is_active`: 是否启用 LangSmith. 关闭时 enter/close 都是 no-op.
`tracer.trace_id`: 本次 trace 的 ID, 写到首条 reply 的 metadata.trace_id, 用户
点 Trace 按钮时通过 /traces/resolve/{message_id} 懒 share + 加载 mirror.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from prisma import Json

from app.config import settings
from app.db import db
from app.services.chat.trace_mirror import (
    get_trace_mirror_by_message,
    write_trace_mirror,
)
from app.services.public_trace import load_public_trace
from app.services.runtime.ws_manager import manager

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
        # _attached=True: sub_intent 模式, trace_id 来自 parent, close() 不 share.
        # 这种模式下 LangSmith 仍会把 sub 内的 LLM 调用记到 parent 的 ls_trace ctx
        # (parent ctx 还活着), 所以 trace tree 是完整的, 用户点击任意 reply 的
        # trace 按钮跳到 parent run_id, 看到完整的 root + nested 视图.
        self._attached = False
        self.trace_id: str | None = None

    @property
    def is_active(self) -> bool:
        return bool(settings.langsmith_tracing)

    @property
    def safe_trace_id(self) -> str | None:
        """is_active 时返回 trace_id, 否则 None — 调用方写消息 metadata 用."""
        return self.trace_id if self.is_active else None

    def enter(self) -> "LangSmithTracer":
        """打开 trace ctx，填充 trace_id。主调用 (root chat_request) 用这个."""
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

    def attach_to_parent(self, parent_trace_id: str | None) -> "LangSmithTracer":
        """sub_intent 模式: 不开新 ls_trace ctx, 复用 parent trace_id 给消息 metadata.

        sub_intent 调用是发生在 parent 的 ls_trace ctx 内 (orchestrator 还没
        调 parent.close()), 所以 sub 内所有 LLM 调用会自动 attach 到 parent
        run tree (LangSmith 通过 contextvars 跟踪父子关系). 我们这里不开新
        chat_request span, 仅把 parent_trace_id 透传到 sub 产生的消息 metadata,
        让用户点 trace 按钮时跳到 parent run_id, 看到完整 root + nested 视图.

        若 parent_trace_id 为 None (LangSmith 未启用 / parent enter 失败),
        sub 也跳过 trace, 行为退化到无 trace 状态.
        """
        self._ctx = contextlib.nullcontext()
        self._ctx.__enter__()
        self.trace_id = parent_trace_id
        self._attached = True
        return self

    def close(self) -> None:
        """关闭 trace ctx. 幂等.

        share_run + mirror 写入改为懒触发 (用户点 trace 按钮时通过
        /traces/resolve/{message_id} endpoint 调). 这里只 exit ctx 让
        LangSmith SDK 把 run 上报到 LangSmith private project. Share
        API quota 从"每次回复"降到"实际查看的 ~5%", mirror 表 95% 行省.
        """
        if self._closed or self._ctx is None:
            return
        self._closed = True
        self._ctx.__exit__(None, None, None)


_SHARE_RETRY_DELAYS = (1.0, 2.0, 4.0)


async def share_run_with_retry(
    trace_id: str | None,
    *,
    conversation_id: str = "?",
) -> str:
    """调 LangSmith client.share_run, 失败重试 3 次指数退避 (1s, 2s, 4s).

    share_run 是网络调用 (~500ms-2s 正常, 抽风时 5xx). 偶发失败会让用户感觉
    "trace 时有时无", 重试可大幅降低这种感受.

    409 Conflict ("Run already shared") 不是失败 — race 场景下 tracer.close()
    后台 _share() 已抢先完成, 用户点击 resolve 才到. 检测 409 改读现有 share
    link 直接返回 (无需重试, 不是 transient).

    其他错误三次都失败时把最后一次异常抛给调用方 (resolve endpoint) 转 503.
    """
    from langsmith.utils import LangSmithConflictError

    if not trace_id:
        raise ValueError("share_run_with_retry requires a non-empty trace_id")
    loop = asyncio.get_running_loop()
    client = get_langsmith_client()
    last_exc: Exception | None = None
    for attempt, delay in enumerate(_SHARE_RETRY_DELAYS, start=1):
        try:
            return await loop.run_in_executor(None, client.share_run, trace_id)
        except LangSmithConflictError:
            # Run 已被另一路径 share (典型: tracer.close 后台 task 抢先).
            # 读现有 link 返回, 跟新 share 等价.
            existing = await loop.run_in_executor(
                None, client.read_run_shared_link, trace_id,
            )
            if existing:
                logger.info(
                    f"[TRACE] share_run race resolved via read_run_shared_link "
                    f"trace_id={trace_id} conv={conversation_id[:8]}"
                )
                return existing
            # 极罕见: 409 但 read 返 None — 让外层 retry 兜底
            last_exc = RuntimeError(
                f"share_run conflict but read_run_shared_link returned None "
                f"trace_id={trace_id}"
            )
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


async def _patch_message_metadata(message_id: str, current_meta: dict, **patch) -> None:
    """合并 patch 进 metadata 并写库. 保留原有 key (e.g. reply_index, sticker_url)."""
    await db.message.update(
        where={"id": message_id},
        data={"metadata": Json({**current_meta, **patch})},
    )


async def resolve_trace_for_message(
    message_id: str, *, user_id: str, is_admin: bool = False,
) -> dict[str, Any]:
    """懒触发: 用户首次点 Trace 按钮时调用, 返回完整 trace detail.

    流程: 校验消息归属 (admin 跳过, 用于后台管理跨用户调试) → 取 metadata.trace_id →
    本地 mirror 命中直接返回 → 否则 share_run + load_public_trace + 写 mirror →
    返回 {trace_url, detail}. 失败抛 ValueError / RuntimeError, endpoint 转 4xx/5xx.

    一次调用搞定 share + mirror + 读取, 避免前端"mirror→retry→public-detail"
    三次串行 RTT 的等待 + 重复 load_public_trace.
    """
    msg = await db.message.find_unique(
        where={"id": message_id},
        include={"conversation": True},
    )
    if not msg:
        raise ValueError("message_not_found")
    conv = getattr(msg, "conversation", None)
    if not conv:
        raise ValueError("message_not_found")
    if not is_admin and conv.userId != user_id:
        raise PermissionError("not_your_message")

    meta = msg.metadata or {}
    if not isinstance(meta, dict):
        raise ValueError("metadata_invalid")
    trace_id = meta.get("trace_id")
    if not trace_id:
        raise ValueError("no_trace_id")

    # 已 share 过 — 优先 mirror 命中, 落空再走 LangSmith 公开 API
    existing_url = meta.get("trace_url")
    if existing_url:
        cached = await get_trace_mirror_by_message(message_id)
        if cached:
            return {"trace_url": str(existing_url), "detail": cached}
        detail = await load_public_trace(str(existing_url))
        await write_trace_mirror(detail=detail, message_id=msg.id)
        return {"trace_url": str(existing_url), "detail": detail}

    # 首次 share — share_run + load + mirror 一气呵成
    public_url = await share_run_with_retry(
        trace_id, conversation_id=conv.id,
    )
    detail = await load_public_trace(public_url)
    await write_trace_mirror(detail=detail, message_id=msg.id)
    await _patch_message_metadata(msg.id, meta, trace_url=public_url)
    await manager.send_event(
        conv.id,
        "trace_ready",
        {"message_id": msg.id, "trace_url": public_url},
    )
    return {"trace_url": public_url, "detail": detail}
