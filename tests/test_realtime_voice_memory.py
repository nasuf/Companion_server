"""Realtime call memory: rolling instructions block and end-of-call extraction."""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.memory.recording.outcomes import MemorySideOutcome
from app.services.realtime_voice.bridge import VoiceBridge
from app.services.realtime_voice.memory_slot import MemorySlot


def _memory(memory_id: str, text: str, owner: str = "关于用户") -> dict[str, str]:
    return {"id": memory_id, "raw": text, "line": f"{owner}：{text}"}


# ---------------------------------------------------------------------------
# MemorySlot
# ---------------------------------------------------------------------------

def test_slot_appends_a_block_and_skips_core_memories():
    slot = MemorySlot("BASE\n你记得的关于用户的事实\n- 用户叫小明，在杭州上班")
    assert slot.instructions() == slot._base
    changed = slot.merge([
        _memory("m1", "用户叫小明，在杭州上班"),
        _memory("m2", "用户养了一只叫团子的橘猫"),
        _memory("m2", "用户养了一只叫团子的橘猫"),
        {"id": "m3", "raw": "", "line": " "},
    ], turn=1)
    assert changed
    assert slot.lines == ["关于用户：用户养了一只叫团子的橘猫"]
    text = slot.instructions()
    assert text.startswith(slot._base + "\n\n【此刻相关的记忆】")
    assert text.endswith("- 关于用户：用户养了一只叫团子的橘猫")
    assert not slot.merge([_memory("m2", "用户养了一只叫团子的橘猫")], turn=2)


def test_slot_keeps_recent_hits_first_and_expires_stale_ones():
    slot = MemorySlot("BASE", ttl_turns=2)
    slot.merge([_memory("a", "A 事"), _memory("b", "B 事")], turn=1)
    slot.merge([_memory("c", "C 事"), _memory("a", "A 事")], turn=2)
    assert slot.lines == ["关于用户：C 事", "关于用户：A 事", "关于用户：B 事"]
    # Turn 3: b was last hit on turn 1 and drops out; a and c survive.
    assert slot.expire(3)
    assert slot.lines == ["关于用户：C 事", "关于用户：A 事"]
    assert slot.expire(4)
    assert slot.lines == []
    assert slot.instructions() == "BASE"
    assert not slot.expire(5)


def test_slot_bounds_lines_and_characters():
    # Each line is 21 characters: two fit in 45, a third does not.
    slot = MemorySlot("BASE", max_lines=3, char_budget=45)
    slot.merge([_memory(str(i), f"第{i}件事" + "很长" * 6) for i in range(6)], turn=1)
    assert len(slot.lines) == 2
    slot = MemorySlot("BASE", max_lines=3, char_budget=1000)
    slot.merge([_memory(str(i), f"第{i}件事") for i in range(6)], turn=1)
    assert len(slot.lines) == 3
    # A single oversized line is still kept rather than leaving the block empty.
    slot = MemorySlot("BASE", char_budget=5)
    slot.merge([_memory("x", "一条很长很长的记忆")], turn=1)
    assert slot.lines == ["关于用户：一条很长很长的记忆"]


# ---------------------------------------------------------------------------
# VoiceBridge: retrieve on this turn, use from the next
# ---------------------------------------------------------------------------

def _bridge(retrieve, *, instructions: str | None = "BASE", send=None):
    provider: list[dict] = []
    client: list[dict] = []

    async def send_provider(event):
        provider.append(event)

    async def send_client(event):
        client.append(event)

    async def save(*_):
        return "msg"

    bridge = VoiceBridge(
        send_provider=send or send_provider,
        send_client=send_client,
        retrieve_memories=retrieve,
        save_user_line=save,
        save_assistant_line=save,
        retrieval_timeout_s=0.2,
        instructions=instructions,
    )
    return bridge, provider, client


async def _user_says(bridge: VoiceBridge, text: str) -> None:
    await bridge.feed_provider_event({
        "type": "conversation.item.input_audio_transcription.completed",
        "transcript": text,
    })
    if bridge._prefetch_task:
        await bridge._prefetch_task


def _updates(provider: list[dict]) -> list[str]:
    return [e["session"]["instructions"] for e in provider if e["type"] == "session.update"]


async def test_each_turn_rewrites_the_block_without_touching_the_reply():
    calls: list[tuple] = []

    async def retrieve(text, previous):
        calls.append((text, previous))
        return [_memory("m1", "用户养了一只叫团子的橘猫")]

    bridge, provider, _ = _bridge(retrieve)
    await _user_says(bridge, "我家猫最近不吃饭")
    assert calls == [("我家猫最近不吃饭", None)]
    [update] = provider
    assert update["session"] == {"instructions": update["session"]["instructions"]}
    assert "团子" in update["session"]["instructions"]
    # Only instructions are sent: no response.create, no cancel.
    assert {e["type"] for e in provider} == {"session.update"}

    await bridge.feed_provider_event({"type": "response.done", "response": {
        "status": "completed",
        "output": [{"type": "message", "content": [{"type": "audio", "transcript": "团子怎么了？"}]}],
    }})
    await _user_says(bridge, "那后来呢")
    assert calls[-1] == ("那后来呢", "团子怎么了？")
    # Same memory again: the block text is unchanged, so nothing is resent.
    assert len(_updates(provider)) == 1


async def test_block_expires_after_turns_without_hits():
    hits = [[_memory("m1", "用户在准备考研")], [], [], []]

    async def retrieve(text, previous):
        return hits.pop(0)

    bridge, provider, _ = _bridge(retrieve)
    for text in ("最近好累", "嗯", "今天下雨了", "晚饭吃了面"):
        await _user_says(bridge, text)
    instructions = _updates(provider)
    assert "考研" in instructions[0]
    assert instructions[-1] == "BASE"
    assert len(instructions) == 2


async def test_a_retrieval_overtaken_by_the_next_turn_is_dropped():
    release = asyncio.Event()
    calls = 0

    async def retrieve(text, previous):
        nonlocal calls
        calls += 1
        if calls == 1:
            await release.wait()
            return [_memory("old", "上一个话题")]
        return [_memory("new", "这一轮的话题")]

    bridge, provider, _ = _bridge(retrieve)
    await bridge.feed_provider_event({
        "type": "conversation.item.input_audio_transcription.completed", "transcript": "第一句",
    })
    first = bridge._prefetch_task
    bridge._turn += 1  # the next user turn arrives while the first is in flight
    release.set()
    await first
    assert _updates(provider) == []
    await _user_says(bridge, "第二句")
    assert "这一轮的话题" in _updates(provider)[-1]


async def test_slow_or_failing_retrieval_keeps_the_call_going():
    async def slow(text, previous):
        await asyncio.sleep(5)
        return []

    bridge, provider, _ = _bridge(slow)
    await _user_says(bridge, "你还记得吗")
    assert provider == []

    async def broken(text, previous):
        raise RuntimeError("vector store down")

    bridge, provider, _ = _bridge(broken)
    await _user_says(bridge, "你还记得吗")
    assert provider == []

    async def found(text, previous):
        return [_memory("m1", "用户喜欢蓝色")]

    async def send_fails(event):
        raise RuntimeError("socket closed")

    bridge, _, _ = _bridge(found, send=send_fails)
    await _user_says(bridge, "我喜欢什么颜色")
    assert bridge._slot.lines == ["关于用户：用户喜欢蓝色"]


async def test_without_instructions_or_after_close_nothing_is_rewritten():
    async def found(text, previous):
        return [_memory("m1", "用户喜欢蓝色")]

    bridge, provider, _ = _bridge(found, instructions=None)
    await _user_says(bridge, "我喜欢什么颜色")
    assert provider == []

    bridge, provider, _ = _bridge(found)
    await bridge._refresh_slot([_memory("m1", "用户喜欢蓝色")], 0, 0.0)
    assert len(provider) == 1
    await bridge.aclose()
    await bridge._refresh_slot([_memory("m2", "新的")], 0, 0.0)
    assert len(provider) == 1


async def test_bridge_records_stored_message_ids_in_order():
    ids = iter(["u1", "a1", "u2", None])

    async def save(*_):
        return next(ids)

    async def retrieve(text, previous):
        return []

    bridge = VoiceBridge(
        send_provider=AsyncMock(), send_client=AsyncMock(), retrieve_memories=retrieve,
        save_user_line=save, save_assistant_line=save,
    )
    done = {"type": "response.done", "response": {"status": "completed", "output": [
        {"type": "message", "content": [{"type": "audio", "transcript": "嗯嗯"}]}]}}
    await _user_says(bridge, "你好")
    await bridge.feed_provider_event(done)
    await _user_says(bridge, "在吗")
    await bridge.feed_provider_event(done)
    assert bridge.message_ids == ["u1", "a1", "u2"]


# ---------------------------------------------------------------------------
# Retrieval callback built by the endpoint
# ---------------------------------------------------------------------------

async def test_endpoint_retrieval_borrows_the_topic_for_short_follow_ups(monkeypatch):
    from app.api.realtime import voice as endpoint
    from app.services.chat import memory_usage
    from app.services.memory.retrieval import hybrid

    hybrid_retrieve = AsyncMock(return_value={
        "memories": [{"source": "user", "id": "m1", "text": "用户的猫叫团子"}],
        "candidate_ids": ["m1", "m2"],
    })
    usage = []
    monkeypatch.setattr(hybrid, "hybrid_retrieve", hybrid_retrieve)
    monkeypatch.setattr(memory_usage, "submit_memory_usage", lambda **kw: usage.append(kw))
    bridge = endpoint._bridge(
        sock=SimpleNamespace(send_event=AsyncMock()), websocket=SimpleNamespace(send_json=AsyncMock()),
        conversation_id="c1", user_id="u1", workspace_id="w1", model="m", instructions="BASE",
    )

    entries = await bridge._retrieve_memories("那后来呢", "你家团子前两天是不是不太舒服")
    assert entries == [{"id": "m1", "raw": "用户的猫叫团子", "line": "关于用户：用户的猫叫团子"}]
    kwargs = hybrid_retrieve.await_args.kwargs
    assert hybrid_retrieve.await_args.args[:2] == ("那后来呢", "u1")
    assert kwargs["enhanced_query"] == "你家团子前两天是不是不太舒服 那后来呢"
    assert usage == [{"event_id": None, "user_id": "u1", "workspace_id": "w1",
                      "selected_ids": ["m1"], "candidate_ids": ["m1", "m2"]}]

    await bridge._retrieve_memories("我这周末想去爬山，你觉得天气会好吗", "嗯")
    assert hybrid_retrieve.await_args.kwargs["enhanced_query"] is None

    def boom(**_):
        raise RuntimeError("redis down")

    monkeypatch.setattr(memory_usage, "submit_memory_usage", boom)
    hybrid_retrieve.return_value = "unexpected"
    assert await bridge._retrieve_memories("那个呢", None) == []
    hybrid_retrieve.return_value = {"memories": [{"source": "ai", "id": "a1", "text": "我在杭州"}]}
    assert await bridge._retrieve_memories("你在哪", None) == [
        {"id": "a1", "raw": "我在杭州", "line": "关于你自己：我在杭州"}]


# ---------------------------------------------------------------------------
# End-of-call extraction
# ---------------------------------------------------------------------------

def _outcomes(*, status="completed", reason="processed", count=1, has_more=False):
    return [MemorySideOutcome(side, status, reason, count, has_more=has_more) for side in ("user", "ai")]


@pytest.fixture
def pipeline(monkeypatch):
    from contextlib import asynccontextmanager

    from app.services.chat import post_process
    from app.services.llm import usage_tracker

    calls: list[list[str]] = []
    results: list = []
    sessions: list[dict] = []

    async def run(user_id, messages, *, conversation_id, workspace_id, source_message_ids):
        assert (user_id, messages, conversation_id, workspace_id) == ("u1", [], "c1", "w1")
        calls.append(source_message_ids)
        return results.pop(0) if results else _outcomes()

    @asynccontextmanager
    async def session(**kwargs):
        sessions.append(kwargs)
        yield

    monkeypatch.setattr(post_process, "_bg_memory_pipeline", run)
    monkeypatch.setattr(usage_tracker, "usage_session", session)
    return SimpleNamespace(calls=calls, results=results, sessions=sessions)


async def _extract(ids):
    from app.services.realtime_voice import persist

    await persist._run_extraction(
        user_id="u1", workspace_id="w1", conversation_id="c1", agent_id="a1", message_ids=ids,
    )


async def test_call_rows_are_extracted_in_batches(pipeline, caplog):
    caplog.set_level(logging.INFO, logger="app.services.realtime_voice.persist")
    ids = [f"m{i}" for i in range(130)]
    await _extract(ids)
    assert pipeline.calls == [ids[:60], ids[60:120], ids[120:]]
    assert pipeline.sessions == [{"scope": "post_process", "conversation_id": "c1",
                                  "agent_id": "a1", "user_id": "u1"}]
    assert "end-of-call extraction completed conv=c1 rounds=3 memories=6" in caplog.text


async def test_backlog_beyond_the_call_keeps_running_until_drained(pipeline):
    pipeline.results.extend([_outcomes(has_more=True), _outcomes(has_more=True), _outcomes()])
    await _extract(["m1", "m2"])
    assert pipeline.calls == [["m1", "m2"]] * 3


async def test_failure_stops_and_is_reported(pipeline, caplog):
    pipeline.results.append([
        MemorySideOutcome("user", "completed", "processed", 1),
        MemorySideOutcome("ai", "failed", "lock_unavailable"),
    ])
    await _extract([f"m{i}" for i in range(90)])
    assert len(pipeline.calls) == 1
    assert "end-of-call extraction ai:lock_unavailable" in caplog.text


async def test_round_limit_bounds_a_runaway_backlog(pipeline, caplog):
    from app.services.realtime_voice import persist

    pipeline.results.extend([_outcomes(has_more=True)] * persist._MAX_EXTRACTION_ROUNDS)
    await _extract(["m1"])
    assert len(pipeline.calls) == persist._MAX_EXTRACTION_ROUNDS
    assert "end-of-call extraction round_limit" in caplog.text


async def test_pipeline_exception_is_logged(pipeline, monkeypatch, caplog):
    from app.services.chat import post_process

    monkeypatch.setattr(post_process, "_bg_memory_pipeline", AsyncMock(side_effect=RuntimeError("x")))
    await _extract(["m1"])
    assert "end-of-call extraction error:RuntimeError" in caplog.text


def test_schedule_passes_unique_ids_and_skips_empty_calls(monkeypatch):
    from app.services.realtime_voice import persist

    scheduled = []

    def fire(coro):
        scheduled.append(coro.cr_frame.f_locals)
        coro.close()

    monkeypatch.setattr(persist, "fire_background", fire)
    persist.schedule_end_of_call_extraction(
        user_id="u1", workspace_id="w1", conversation_id="c1", agent_id="a1", message_ids=[],
    )
    assert scheduled == []
    persist.schedule_end_of_call_extraction(
        user_id="u1", workspace_id="w1", conversation_id="c1", agent_id="a1",
        message_ids=["m1", "m2", "m1"],
    )
    assert scheduled[0]["message_ids"] == ["m1", "m2"]


# ---------------------------------------------------------------------------
# History seed acknowledgement
# ---------------------------------------------------------------------------

class _Provider:
    def __init__(self, acks):
        self.acks = acks
        self.sent: list[dict] = []
        self.inbox: list[dict] = [{"type": "session.created"}]

    async def send_event(self, event):
        self.sent.append(event)
        if event["type"] == "session.update":
            self.inbox.append({"type": "session.updated"})
        elif event["type"] == "conversation.item.create" and self.acks:
            # Everything the provider says after the first item, in order.
            self.inbox.extend(self.acks)
            self.acks = []

    async def recv_event(self):
        if not self.inbox:
            await asyncio.sleep(3600)
        return self.inbox.pop(0)


async def test_history_is_acknowledged_by_count_not_client_id(caplog):
    from app.services.realtime_voice.provider import configure_session

    history = [{"role": "user", "text": "我叫小明"}, {"role": "assistant", "text": "你好小明"}]
    # StepFun replies with its own ids (observed 2026-11-05).
    sock = _Provider([
        {"type": "conversation.item.created", "item": {"id": "srv-1", "role": "user"}},
        {"type": "rate_limits.updated"},
        {"type": "conversation.item.created", "item": {"id": "srv-2", "role": "assistant"}},
    ])
    await configure_session(sock, instructions="hi", voice="v", history=history)
    assert "history seed" not in caplog.text
    assert [e["item"]["id"] for e in sock.sent if e["type"] == "conversation.item.create"] == [
        "hist_0", "hist_1"]


async def test_history_error_is_reported_without_failing_the_call(caplog):
    from app.services.realtime_voice.provider import configure_session

    sock = _Provider([{"type": "error", "error": {"message": "bad item"}}])
    voice = await configure_session(
        sock, instructions="hi", voice="v", history=[{"role": "user", "text": "你好"}],
    )
    assert voice == "v"
    assert "history seed returned an error: bad item" in caplog.text
    assert "history seed incomplete, remaining=1" in caplog.text


def test_memory_lines_from_classified_memories_keep_raw_text_for_dedup():
    from app.services.memory.retrieval.context_selector import ClassifiedMemory
    from app.services.realtime_voice.instructions import memory_lines

    lines = memory_lines([
        ClassifiedMemory(text="用户养了一只叫团子的橘猫", relevance="strong", score=0.9,
                         id="m1", occur_time="2026-10-01T08:00:00+00:00"),
        ClassifiedMemory(text="我在杭州做整理师", relevance="medium", score=0.6, id="a1", source="ai"),
    ])
    assert [entry["id"] for entry in lines] == ["m1", "a1"]
    assert lines[0]["raw"] == "用户养了一只叫团子的橘猫"
    assert lines[0]["line"].startswith("关于用户：[事件时间 ")
    assert lines[1]["line"].startswith("关于你自己：")
    # The raw text, not the time-labelled line, is what dedups against the core block.
    slot = MemorySlot("- 用户养了一只叫团子的橘猫")
    assert not slot.merge(lines[:1], turn=1)


# ---------------------------------------------------------------------------
# Microphone meter (diagnoses calls where StepFun never hears the user)
# ---------------------------------------------------------------------------

async def test_mic_meter_reports_volume_once_after_three_seconds(caplog):
    import struct

    caplog.set_level(logging.INFO, logger="app.services.realtime_voice.bridge")

    async def retrieve(text, previous):
        return []

    bridge, provider, _ = _bridge(retrieve)
    quiet = struct.pack("<8h", 0, 0, 0, 0, 0, 0, 0, 0) * 600   # 0.2 s of digital silence
    loud = struct.pack("<8h", 16384, 0, 0, 0, 0, 0, 0, 0) * 600
    for _ in range(14):
        await bridge.feed_audio(quiet)
    assert "mic first 3s" not in caplog.text
    await bridge.feed_audio(loud)
    await bridge.feed_audio(loud)
    assert caplog.text.count("mic first 3s") == 1
    assert len([e for e in provider if e["type"] == "input_audio_buffer.append"]) == 16
    await bridge.feed_provider_event({"type": "input_audio_buffer.speech_started"})
    summary = bridge.mic_summary
    assert "chunks=16" in summary and "seconds=3.2" in summary
    assert "peak_dbfs=-6.0" in summary
    assert summary.endswith("speech_turns=1")


def test_mic_meter_with_no_audio_reads_minus_infinity():
    from app.services.realtime_voice.bridge import _MicMeter

    assert _MicMeter().summary() == "bytes=0 chunks=0 seconds=0.0 peak_dbfs=-inf rms_dbfs=-inf"
