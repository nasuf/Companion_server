"""按 env 配置初始化 AxiomHandler.

env:
  AXIOM_TOKEN     必需, xaat- 开头, ingest 权限
  AXIOM_DATASET   必需, e.g. companion-dev / companion-server
  AXIOM_ORG_ID    可选, 账号下多 org 时必需
  AXIOM_LOG_LEVEL 可选, 默认 INFO (DEBUG 会把 quota 打爆, 不建议)

任一必需 env 缺失 → 跳过装载, 程序继续, console handler 仍工作 — 本地 dev
零依赖能跑. axiom-py 没装也跳过, 防 import 失败 crash app.

**异步包装**: AxiomHandler.emit 同步, 内部用 threading.Timer + 阻塞 HTTP POST,
buffer 满 1000 时 inline flush. 在 asyncio 应用里直接挂会潜在阻塞 event loop.
用 stdlib QueueHandler + QueueListener 解耦: app 线程只 enqueue (lock-free),
QueueListener 单独后台线程 drain → 真 AxiomHandler. 进程退出时 listener.stop()
确保剩余 buffer flush.
"""

from __future__ import annotations

import atexit
import logging
import logging.handlers
import os
import queue

logger = logging.getLogger(__name__)

# 模块级保留 listener 引用 — atexit 关闭 + 防 GC.
_listener: logging.handlers.QueueListener | None = None


def setup_axiom() -> bool:
    """读 env 装 AxiomHandler (经 QueueHandler 包装). 返回是否真的装上了."""
    global _listener
    token = os.getenv("AXIOM_TOKEN")
    dataset = os.getenv("AXIOM_DATASET")
    if not token or not dataset:
        logger.info("[axiom] AXIOM_TOKEN/DATASET 未配置, 跳过远程日志")
        return False
    try:
        from axiom_py import Client
        from axiom_py.logging import AxiomHandler
    except ImportError:
        logger.warning("[axiom] axiom-py 未安装, 跳过 (uv add axiom-py)")
        return False

    try:
        from app.observability.log_filter import ContextInjectionFilter
        client = Client(token=token, org_id=os.getenv("AXIOM_ORG_ID"))
        axiom_handler = AxiomHandler(client, dataset)
        level_name = os.getenv("AXIOM_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
        axiom_handler.setLevel(level)
        # Filter 挂在 handler 链尾端: QueueHandler 把 record 入队 → QueueListener
        # 在后台线程 dispatch 到 axiom_handler.handle(), Filter 在 axiom_handler
        # 触发. Filter 必须挂在这里才能拿到 ContextVar (record 入队时已离开应用线程).
        # 故: Filter 必须在 QueueHandler 上, 让应用线程进 handle 时就快照 ContextVar.
        # axiom_handler 自身仍 addFilter 防漏 (e.g. 未来加新 source).
        axiom_handler.addFilter(ContextInjectionFilter())

        # 用 unbounded queue: blocking 风险归零. 理论上 listener 卡死会涨内存,
        # 实践中 axiom-py 自带 1s flush + 1000 batch, listener 不会真卡.
        log_queue: queue.Queue = queue.Queue(maxsize=-1)
        queue_handler = logging.handlers.QueueHandler(log_queue)
        queue_handler.setLevel(level)
        # ContextVar 必须在应用线程 (即 enqueue 瞬间) 快照, 否则 listener 后台
        # 线程跑时 ContextVar 已不在 — 故 Filter 也挂 QueueHandler.
        queue_handler.addFilter(ContextInjectionFilter())
        logging.getLogger().addHandler(queue_handler)

        # 启动 listener (后台 daemon 线程)
        _listener = logging.handlers.QueueListener(
            log_queue, axiom_handler, respect_handler_level=True,
        )
        _listener.start()
        atexit.register(_shutdown_listener)
    except Exception as e:
        logger.warning(f"[axiom] handler 装载失败, 降级本地日志: {e}")
        return False

    logger.info(f"[axiom] handler attached dataset={dataset} level={level_name} (queued)")
    return True


def _shutdown_listener() -> None:
    """进程退出前 flush — 防最后一波 log 丢失."""
    global _listener
    if _listener is not None:
        try:
            _listener.stop()
        except Exception:
            pass
        _listener = None
