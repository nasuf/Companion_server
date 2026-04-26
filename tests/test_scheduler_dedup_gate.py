"""Regression tests for scheduler dedup gate.

用户连发非碎片消息时, 同一条 user_msg 不应被处理两次:
ws.py 的 enqueue_or_append_delayed 关闭主 race; 这里覆盖兜底 gate
`_already_covered` —— 仅当上一轮 LLM 数据拉取已显式包含本 user_msg
(即 reply.metadata.covered_until_user_ts 比 user_msg.createdAt 大)
才跳过, 短路/边界回复无此字段绝不误杀。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from jobs import scheduler as scheduler_mod


def _msg(*, id: str, role: str, created_at: datetime, metadata: dict | None = None):
    return SimpleNamespace(id=id, role=role, createdAt=created_at, metadata=metadata)


@pytest.fixture
def base_ts():
    return datetime(2026, 4, 25, 14, 57, 0, tzinfo=timezone.utc)


@pytest.fixture
def fake_db(monkeypatch):
    """Patch app.db.db with controllable AsyncMocks for message.find_unique/find_many."""
    fake = SimpleNamespace(
        message=SimpleNamespace(
            find_unique=AsyncMock(),
            find_many=AsyncMock(),
        )
    )
    import app.db as db_mod

    monkeypatch.setattr(db_mod, "db", fake)
    return fake


async def test_skips_when_prior_reply_explicitly_covers(fake_db, base_ts):
    """主 LLM 已隐式回复 (covered_until_user_ts >= user_msg.createdAt) → skip."""
    user_ts = base_ts
    fake_db.message.find_unique.return_value = _msg(
        id="m2", role="user", created_at=user_ts,
    )
    # 同一轮 LLM 的 reply, covered_until_user_ts 等于 m2.createdAt → 已覆盖
    fake_db.message.find_many.return_value = [
        _msg(
            id="r1", role="assistant",
            created_at=user_ts + timedelta(seconds=15),
            metadata={"covered_until_user_ts": user_ts.isoformat()},
        ),
    ]

    assert await scheduler_mod._already_covered("conv", "m2") is True


async def test_no_skip_when_reply_data_fetched_before_user_msg(fake_db, base_ts):
    """msg2 在前一轮 LLM 数据拉取后才到达 (covered_until_user_ts < m2.createdAt) → 必须正常处理."""
    user_ts = base_ts
    earlier_ts = user_ts - timedelta(seconds=10)
    fake_db.message.find_unique.return_value = _msg(
        id="m2", role="user", created_at=user_ts,
    )
    fake_db.message.find_many.return_value = [
        _msg(
            id="r1", role="assistant",
            created_at=user_ts + timedelta(seconds=5),
            # 上轮 LLM 只看到 m1 (earlier_ts), 没看到 m2
            metadata={"covered_until_user_ts": earlier_ts.isoformat()},
        ),
    ]

    assert await scheduler_mod._already_covered("conv", "m2") is False


async def test_no_skip_for_short_circuit_reply_without_metadata_field(fake_db, base_ts):
    """短路/边界回复不写 covered_until_user_ts → 不能因它们误杀后续消息."""
    user_ts = base_ts
    fake_db.message.find_unique.return_value = _msg(
        id="m2", role="user", created_at=user_ts,
    )
    fake_db.message.find_many.return_value = [
        _msg(
            id="r1", role="assistant",
            created_at=user_ts + timedelta(seconds=10),
            metadata={"boundary": "violation"},  # 短路 metadata, 无 covered_until
        ),
    ]

    assert await scheduler_mod._already_covered("conv", "m2") is False


async def test_no_skip_when_no_later_assistant(fake_db, base_ts):
    """user_msg 之后无任何 assistant 消息 → 必须处理."""
    fake_db.message.find_unique.return_value = _msg(
        id="m1", role="user", created_at=base_ts,
    )
    fake_db.message.find_many.return_value = []

    assert await scheduler_mod._already_covered("conv", "m1") is False


async def test_no_skip_when_user_msg_missing(fake_db):
    """user_msg 不存在 (race / 已删除) → fail-open 处理."""
    fake_db.message.find_unique.return_value = None

    assert await scheduler_mod._already_covered("conv", "m-ghost") is False


async def test_handles_corrupt_metadata_gracefully(fake_db, base_ts):
    """metadata 不是 dict / covered_until_user_ts 不是合法 ISO → 不 skip."""
    fake_db.message.find_unique.return_value = _msg(
        id="m1", role="user", created_at=base_ts,
    )
    fake_db.message.find_many.return_value = [
        _msg(
            id="r-bad-iso", role="assistant",
            created_at=base_ts + timedelta(seconds=5),
            metadata={"covered_until_user_ts": "not-a-date"},
        ),
        _msg(
            id="r-non-dict-md", role="assistant",
            created_at=base_ts + timedelta(seconds=10),
            metadata="legacy string metadata",  # type: ignore[arg-type]
        ),
        _msg(
            id="r-none-md", role="assistant",
            created_at=base_ts + timedelta(seconds=15),
            metadata=None,
        ),
    ]

    assert await scheduler_mod._already_covered("conv", "m1") is False


async def test_skips_when_any_of_multiple_replies_covers(fake_db, base_ts):
    """多条后续 assistant 中只要任一条覆盖即 skip (覆盖判定是 OR 关系)."""
    user_ts = base_ts
    fake_db.message.find_unique.return_value = _msg(
        id="m2", role="user", created_at=user_ts,
    )
    fake_db.message.find_many.return_value = [
        # 第一条短路 reply 不覆盖
        _msg(
            id="r-short", role="assistant",
            created_at=user_ts + timedelta(seconds=3),
            metadata={"intent": "current_state"},
        ),
        # 第二条主 LLM reply 显式覆盖
        _msg(
            id="r-main", role="assistant",
            created_at=user_ts + timedelta(seconds=12),
            metadata={"covered_until_user_ts": user_ts.isoformat()},
        ),
    ]

    assert await scheduler_mod._already_covered("conv", "m2") is True
