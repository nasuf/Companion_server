"""AI 寄出的现实礼物写进记忆：寄出不泄露礼物名、签收双侧落库、道谢、撤销，以及通用抽取不重复记。"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from app.services.chat import post_process
from app.services.offline import gift_memory

GIFT = {
    "id": "g1",
    "gift_name": "空白迷你灵感便签本",
    "gift_note": "想到小灵感就写下来呀",
    "ordered_at": "2026-10-14T16:30:00Z",  # 北京时间 10-15 凌晨
}


@pytest.fixture
def stored(monkeypatch):
    """同步跑完后台写入，返回每次 store_memory 的调用参数。"""
    calls = []

    async def fake_store(user_id, content, **kwargs):
        calls.append({"user_id": user_id, "content": content, **kwargs})
        return f"mem-{len(calls)}"

    pending = []
    monkeypatch.setattr(gift_memory, "store_memory", fake_store)
    monkeypatch.setattr(gift_memory, "fire_background", pending.append)

    async def flush():
        while pending:
            await pending.pop(0)
        return calls

    return flush


def _by_side(calls):
    return {call["source"]: call for call in calls}


@pytest.mark.asyncio
async def test_sent_is_an_ai_memory_without_the_gift_name(stored):
    gift_memory.remember_gift_sent("user-1", "ws-1", GIFT)
    [call] = await stored()

    assert call["source"] == "ai" and call["workspace_id"] == "ws-1"
    assert (call["main_category"], call["sub_category"], call["level"]) == ("生活", "馈赠", 2)
    assert call["provenance"] == "ai_authored" and call["skip_reconciliation"] is True
    assert "便签本" not in call["content"]  # 记忆会注入聊天，签收前不能出现礼物名
    assert "保密" not in call["content"]  # 写事实不写指令：指令进了长期记忆，揭晓后还会生效
    assert "寄了一份礼物" in call["content"]


@pytest.mark.asyncio
async def test_received_is_remembered_by_both_sides_with_the_reason(stored):
    gift_memory.remember_gift_received(
        "user-1", "ws-1", GIFT, agent_name="昕昕", reveal_message="其实是便签本||你总有小灵感",
    )
    sides = _by_side(await stored())

    ai, user = sides["ai"], sides["user"]
    assert "用户签收了我寄的礼物：空白迷你灵感便签本" in ai["content"]
    assert "「其实是便签本 你总有小灵感」" in ai["content"]
    assert "想到小灵感就写下来呀" in ai["content"]
    assert user["content"].endswith("我收到并签收了昕昕寄来的礼物：空白迷你灵感便签本")
    assert user["provenance"] == "user_stated"
    assert ai["importance"] == user["importance"] == gift_memory.RECEIVED_IMPORTANCE


@pytest.mark.asyncio
async def test_received_memory_stays_injectable_when_texts_are_long(stored):
    long = "好" * 200
    gift_memory.remember_gift_received(
        "user-1", "ws-1", {**GIFT, "gift_note": long}, agent_name=None, reveal_message=long,
    )
    sides = _by_side(await stored())

    # 超过单条注入上限的记忆检索时会被整条跳过：放不下的补充句宁可不加。
    assert not gift_memory.exceeds_injection_limit(sides["ai"]["content"])
    assert sides["ai"]["content"].count("「") == 1
    assert "对方寄来的礼物" in sides["user"]["content"]


@pytest.mark.asyncio
async def test_thanks_keeps_the_users_words_and_extracts_with_evidence(stored, monkeypatch):
    pipeline = AsyncMock(return_value=[])
    monkeypatch.setattr(gift_memory, "process_memory_pipeline", pipeline)

    gift_memory.remember_gift_thanks(
        "user-1", "ws-1", GIFT, message="好喜欢||正好缺个本子",
        message_id="msg-9", message_content="我收到了「空白迷你灵感便签本」，想说：好喜欢||正好缺个本子",
    )
    [ai] = await stored()

    assert ai["source"] == "ai" and "特意跟我道谢：「好喜欢 正好缺个本子」" in ai["content"]
    args, kwargs = pipeline.await_args
    assert args == ("user-1", "user: 我收到了「空白迷你灵感便签本」，想说：好喜欢||正好缺个本子")
    assert kwargs["side"] == "user" and kwargs["evidence_message_ids"] == ["msg-9"]
    assert kwargs["context_conversation"] == "user: 我签收了你寄来的「空白迷你灵感便签本」"


@pytest.mark.asyncio
async def test_thanks_without_a_chat_message_only_writes_the_ai_side(stored, monkeypatch):
    pipeline = AsyncMock()
    monkeypatch.setattr(gift_memory, "process_memory_pipeline", pipeline)
    gift_memory.remember_gift_thanks(
        "user-1", None, GIFT, message="谢谢", message_id=None, message_content=None,
    )
    assert len(await stored()) == 1
    pipeline.assert_not_awaited()


@pytest.mark.asyncio
async def test_thanks_extraction_failure_is_contained(stored, monkeypatch):
    monkeypatch.setattr(gift_memory, "process_memory_pipeline", AsyncMock(side_effect=RuntimeError("llm")))
    gift_memory.remember_gift_thanks(
        "user-1", "ws-1", GIFT, message="谢谢", message_id="m", message_content="谢谢",
    )
    assert len(await stored()) == 1


@pytest.mark.asyncio
async def test_cancelled_names_when_the_gift_was_sent(stored):
    gift_memory.remember_gift_cancelled("user-1", "ws-1", GIFT)
    gift_memory.remember_gift_cancelled("user-1", "ws-1", {"id": "g2", "ordered_at": None})
    earlier, same_day = await stored()
    assert "我寄给用户的那份礼物（2026-10-15寄出）中途取消了" in earlier["content"]
    assert "便签本" not in earlier["content"]
    assert "寄出）" not in same_day["content"] and "那份礼物中途取消了" in same_day["content"]


@pytest.mark.asyncio
async def test_write_failures_and_duplicates_are_logged_not_raised(monkeypatch, caplog):
    pending = []
    monkeypatch.setattr(gift_memory, "fire_background", pending.append)
    monkeypatch.setattr(gift_memory, "store_memory", AsyncMock(side_effect=[RuntimeError("db"), None]))

    gift_memory.remember_gift_sent("user-1", "ws-1", GIFT)
    gift_memory.remember_gift_sent("user-1", "ws-1", GIFT)
    with caplog.at_level("INFO", logger=gift_memory.__name__):
        for job in pending:
            await job

    outcomes = [r.outcome for r in caplog.records if getattr(r, "event", None) == "gift.memory"]
    assert outcomes == ["failed", "deduplicated"]


def test_dates_are_local_and_tolerate_bad_input():
    assert gift_memory._parse_time("not a date") is None
    assert gift_memory._parse_time(None) is None
    naive = gift_memory._parse_time(datetime(2026, 10, 14, 16, 30))
    assert naive.tzinfo is UTC
    assert gift_memory._parse_time("2026-10-14T16:30:00") == naive
    assert gift_memory._date_prefix(naive) == "2026-10-15 "


@pytest.mark.asyncio
@pytest.mark.parametrize("backlog_mode", [False, True])
async def test_gift_chat_messages_are_context_not_extraction_targets(backlog_mode):
    now = datetime.now(UTC)
    gift_card = {
        "id": "gift-card", "role": "user", "content": "我签收了你寄来的「便签本」",
        "createdAt": now, "metadata": {"real_world_type": "gift", "trigger_type": "gift_received"},
    }
    normal = {"id": "normal", "role": "user", "content": "这个本子我拿来记灵感",
              "createdAt": now + timedelta(seconds=1), "metadata": {}}
    pipeline = AsyncMock(return_value=[])
    set_wm = AsyncMock()
    with patch.object(post_process, "get_watermark", AsyncMock(return_value=now - timedelta(minutes=1) if backlog_mode else None)), \
         patch.object(post_process, "_fetch_new_side_messages", AsyncMock(return_value=[gift_card, normal])), \
         patch.object(post_process, "process_memory_pipeline", pipeline), \
         patch.object(post_process, "set_watermark", set_wm):
        await post_process._pipeline_with_watermark("user-1", [gift_card, normal], "conv", side="user",
                                                     workspace_id="ws-1")

    kwargs = pipeline.await_args.kwargs
    assert kwargs["evidence_message_ids"] == ["normal"]
    assert "签收" not in kwargs["new_conversation"]
    assert "签收" in kwargs["context_conversation"]  # 仍是历史上下文，帮抽取理解「这个本子」
    assert set_wm.await_args.args[2] == normal["createdAt"]

    # 只有礼物消息时不调模型，但水位线照常推进，不会每轮都重新捞到它。
    pipeline.reset_mock()
    with patch.object(post_process, "get_watermark", AsyncMock(return_value=now - timedelta(minutes=1))), \
         patch.object(post_process, "_fetch_new_side_messages", AsyncMock(return_value=[gift_card])), \
         patch.object(post_process, "process_memory_pipeline", pipeline), \
         patch.object(post_process, "set_watermark", set_wm):
        assert await post_process._pipeline_with_watermark("user-1", [gift_card], "conv", side="user") == 0
    pipeline.assert_not_awaited()
    assert set_wm.await_args.args[2] == now
