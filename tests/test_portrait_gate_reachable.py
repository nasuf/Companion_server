"""画像的前置条件必须在真实数据分布下可达.

这个文件存在的理由: 旧门槛 `L2 ≥ 20 AND L1 ≥ 5` 在生产上**一次都没被满足过**,
user_portraits 表从上线起一直是空的, 而且没有任何报错 —— 前置条件不满足只会记一行
info 日志, 看起来跟"这周没人够格"没有区别。

失效的机制值得记下来: 层级由 importance 推导, 于是这两个数在真实数据里此消彼长。
说身份事实的用户攒 L1, 聊日常的用户攒 L2, 很少有人两样都攒够。两个此消彼长的量
做 AND, 门就关死了。

所以下面用**生产上实际观察到的两种分布**做参数, 而不是随手编的数字。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services import portrait as portrait_module
from app.services.portrait import (
    MIN_MEMORIES_FOR_PORTRAIT,
    MIN_USER_MESSAGES_FOR_PORTRAIT,
    check_portrait_preconditions,
)

# 生产实测的两种真实分布 (2026-07-28, 15 个有记忆的用户):
#   身份型  L1=23 L2= 5   —— 用户主动陈述姓名/居住地/生日这类事实
#   日常型  L1= 0 L2=23   —— 用户只聊日常, 没说过身份信息
# 旧门槛下两者都不达标, 各差一个条件。
PRODUCTION_SHAPES = [
    pytest.param(23, 5, id="身份型 L1=23 L2=5"),
    pytest.param(0, 23, id="日常型 L1=0 L2=23"),
    pytest.param(8, 9, id="混合型 L1=8 L2=9"),
]


def _agent(age_hours: float = 48):
    return SimpleNamespace(
        id="agent-1",
        createdAt=datetime.now(UTC) - timedelta(hours=age_hours),
    )


async def _check(l1: int, l2: int, user_messages: int = 100) -> bool:
    """按给定的记忆/消息量跑一次前置检查。"""
    with patch.object(portrait_module, "db") as mock_db, \
         patch.object(
             portrait_module, "resolve_workspace_id", AsyncMock(return_value="ws-1")
         ), \
         patch.object(
             portrait_module.memory_repo, "count", AsyncMock(return_value=l1 + l2)
         ):
        mock_db.aiagent = MagicMock(find_unique=AsyncMock(return_value=_agent()))
        mock_db.message = MagicMock(count=AsyncMock(return_value=user_messages))
        return await check_portrait_preconditions("user-1", "agent-1")


@pytest.mark.asyncio
@pytest.mark.parametrize("l1,l2", PRODUCTION_SHAPES)
async def test_real_world_distributions_can_qualify(l1, l2):
    """三种真实形态都该够格 —— 它们的共同点是总量够, 差别只在落在哪一层。"""
    assert await _check(l1, l2) is True, (
        f"L1={l1} L2={l2} (合计 {l1 + l2}) 仍然不够格; "
        "门槛又变成看层级构成了"
    )


@pytest.mark.asyncio
async def test_gate_does_not_depend_on_how_memories_split_across_levels():
    """同样的总量, 无论怎么分布在 L1/L2, 结论必须一致。

    这条直接钉住旧 bug 的成因: 一旦有人再按层级分别设下限, 此消彼长的问题就回来了。
    """
    total = MIN_MEMORIES_FOR_PORTRAIT
    verdicts = {await _check(l1, total - l1) for l1 in range(0, total + 1)}
    assert verdicts == {True}, "同样总量下结论随层级分布变化了"


@pytest.mark.asyncio
async def test_thin_users_are_still_rejected():
    """放宽不等于取消 —— 素材不足时写出来的画像是编的。"""
    assert await _check(2, 3) is False


@pytest.mark.asyncio
async def test_memories_without_conversation_are_rejected():
    """只有记忆没有对话时, 画像会写得像档案摘要而不是对一个人的印象。"""
    assert await _check(50, 50, user_messages=MIN_USER_MESSAGES_FOR_PORTRAIT - 1) is False


@pytest.mark.asyncio
async def test_brand_new_agent_is_rejected():
    with patch.object(portrait_module, "db") as mock_db:
        mock_db.aiagent = MagicMock(
            find_unique=AsyncMock(return_value=_agent(age_hours=2))
        )
        assert await check_portrait_preconditions("user-1", "agent-1") is False


@pytest.mark.asyncio
async def test_counts_recallable_memories_not_a_single_level():
    """确认查询本身问的是 L1+L2, 而不是某一层。"""
    captured: dict = {}

    async def _count(**kwargs):
        captured.update(kwargs)
        return 100

    with patch.object(portrait_module, "db") as mock_db, \
         patch.object(
             portrait_module, "resolve_workspace_id", AsyncMock(return_value="ws-1")
         ), \
         patch.object(portrait_module.memory_repo, "count", _count):
        mock_db.aiagent = MagicMock(find_unique=AsyncMock(return_value=_agent()))
        mock_db.message = MagicMock(count=AsyncMock(return_value=100))
        await check_portrait_preconditions("user-1", "agent-1")

    assert captured["where"]["level"] == {"in": [1, 2]}
    assert captured["source"] == "user"
