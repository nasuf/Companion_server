"""_enqueue_scanned_aggregation_results 的逐条容错.

scan_due_user_turns() 用 Lua 脚本原子地把每条结果从聚合窗口弹出——弹出那一刻
它就已经从源头消失了。如果这里不逐条兜住异常，某一条入队失败会让异常冒到
外层 _run_aggregation_scan_body 的 try/except，直接跳过本批次里排在它后面的
所有会话：那些会话的消息已经被弹出且再也拿不回来，却因为跟它们无关的另一条
失败被连坐丢弃，且没有任何事件通知对应用户。
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from jobs import scheduler as scheduler_mod


def _result(conv_id: str) -> tuple:
    return ("agent-1", "user-1", "你好", conv_id, {"delay_seconds": 0.0}, "msg-1")


async def test_one_failing_conversation_does_not_block_the_rest(monkeypatch):
    enqueue_mock = AsyncMock(
        side_effect=lambda conv_id, *a, **kw: (
            (_ for _ in ()).throw(RuntimeError("redis down"))
            if conv_id == "conv-bad"
            else None
        )
    )
    monkeypatch.setattr(scheduler_mod, "enqueue_delayed_message", enqueue_mock)
    manager = AsyncMock()

    results = [_result("conv-bad"), _result("conv-good")]

    # 必须不抛异常: 一条失败不能打断循环, 也不能冒到外层杀掉整批。
    await scheduler_mod._enqueue_scanned_aggregation_results(results, manager)

    assert enqueue_mock.await_count == 2
    good_events = [
        call for call in manager.send_event.await_args_list if call.args[0] == "conv-good"
    ]
    assert any(call.args[1] == "pending" for call in good_events)


async def test_failing_conversation_gets_a_compensating_done_event(monkeypatch):
    """入队失败的那条会话必须收到一个能解卡前端"发送中"状态的事件, 而不是
    彻底沉默 (发送方永远等不到任何响应)。
    """
    monkeypatch.setattr(
        scheduler_mod,
        "enqueue_delayed_message",
        AsyncMock(side_effect=RuntimeError("redis down")),
    )
    manager = AsyncMock()

    await scheduler_mod._enqueue_scanned_aggregation_results(
        [_result("conv-bad")], manager
    )

    done_calls = [
        call for call in manager.send_event.await_args_list
        if call.args[0] == "conv-bad" and call.args[1] == "done"
    ]
    assert len(done_calls) == 1
    assert done_calls[0].args[2]["message_id"] == "error"


async def test_notify_failure_itself_does_not_raise(monkeypatch):
    """就算连"通知失败"这一步本身也失败 (比如 manager 发布也炸了), 也只能
    记日志, 绝不能让异常再冒出去影响下一条会话的处理。
    """
    monkeypatch.setattr(
        scheduler_mod,
        "enqueue_delayed_message",
        AsyncMock(side_effect=RuntimeError("redis down")),
    )
    manager = AsyncMock()
    manager.send_event = AsyncMock(side_effect=RuntimeError("publish also failed"))

    # 不应该抛出 —— 双重失败也只是走 warning 分支。
    await scheduler_mod._enqueue_scanned_aggregation_results(
        [_result("conv-bad"), _result("conv-good")], manager
    )
