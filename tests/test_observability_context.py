"""ContextVar bind/reset + asyncio task propagation 验证.

确保 bind_context:
- 能正确 set/reset (with 块语义)
- 嵌套绑定 LIFO 还原
- 自动跨 asyncio.create_task 传播 (Python 3.7+ ContextVar 语义保证)
- 跨 fire_background (用 copy_context) 传播
"""

from __future__ import annotations

import asyncio

import pytest

from app.observability.context import bind_context, current_context


def test_bind_context_sets_and_resets():
    assert current_context() == {}
    with bind_context(conversation_id="conv-X", agent_name="Alice"):
        ctx = current_context()
        assert ctx["conversation_id"] == "conv-X"
        assert ctx["agent_name"] == "Alice"
    # 退出 with 后必须清空
    assert current_context() == {}


def test_bind_context_nested_overrides_then_restores():
    with bind_context(conversation_id="outer"):
        assert current_context()["conversation_id"] == "outer"
        with bind_context(conversation_id="inner", agent_id="a-1"):
            ctx = current_context()
            assert ctx["conversation_id"] == "inner"
            assert ctx["agent_id"] == "a-1"
        # 退出内层后, 外层 conversation_id 还原, agent_id 清掉
        ctx = current_context()
        assert ctx["conversation_id"] == "outer"
        assert "agent_id" not in ctx


def test_bind_context_unknown_key_silently_ignored():
    """传未注册字段不应抛, 也不应留下垃圾数据."""
    with bind_context(conversation_id="X", nonexistent_field="oops"):
        ctx = current_context()
        assert ctx == {"conversation_id": "X"}


def test_bind_context_explicit_none_clears_outer():
    """显式传 None 等于"清掉这个字段", 跟"不传"区分."""
    with bind_context(conversation_id="outer", user_id="u1"):
        with bind_context(conversation_id=None):
            ctx = current_context()
            # conversation_id 被清掉, user_id 保留 (没传)
            assert "conversation_id" not in ctx
            assert ctx["user_id"] == "u1"


@pytest.mark.asyncio
async def test_propagates_to_asyncio_create_task():
    """asyncio.create_task 创建子 task 时复制当前 context — 子 task 能读外层绑定."""
    captured: dict = {}

    async def child():
        captured.update(current_context())

    with bind_context(conversation_id="conv-async", agent_id="a-async"):
        t = asyncio.create_task(child())
        await t

    assert captured["conversation_id"] == "conv-async"
    assert captured["agent_id"] == "a-async"


@pytest.mark.asyncio
async def test_propagates_through_fire_background():
    """fire_background 用 contextvars.copy_context() 隔离; ContextVar 应保留."""
    from app.services.runtime.tasks import fire_background

    captured: dict = {}

    async def background():
        captured.update(current_context())

    with bind_context(conversation_id="conv-bg", workspace_id="w-bg"):
        t = fire_background(background())
        await t

    assert captured["conversation_id"] == "conv-bg"
    assert captured["workspace_id"] == "w-bg"


@pytest.mark.asyncio
async def test_concurrent_tasks_have_isolated_context():
    """gather 多个 task — 每个 task 设的 ContextVar 互相不影响."""
    captured: dict[str, str] = {}

    async def worker(label: str):
        with bind_context(conversation_id=f"conv-{label}"):
            await asyncio.sleep(0.01)
            captured[label] = current_context()["conversation_id"]

    await asyncio.gather(worker("A"), worker("B"), worker("C"))
    assert captured == {"A": "conv-A", "B": "conv-B", "C": "conv-C"}
    # 父 context 没被污染
    assert current_context() == {}
