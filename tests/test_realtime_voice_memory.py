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

def test_slot_note_holds_the_block_and_skips_core_memories():
    slot = MemorySlot("BASE\n你记得的关于用户的事实\n- 用户叫小明，在杭州上班")
    assert slot.note() == ""
    changed = slot.merge([
        _memory("m1", "用户叫小明，在杭州上班"),
        _memory("m2", "用户养了一只叫团子的橘猫"),
        _memory("m2", "用户养了一只叫团子的橘猫"),
        {"id": "m3", "raw": "", "line": " "},
    ], turn=1)
    assert changed
    assert slot.lines == ["关于用户：用户养了一只叫团子的橘猫"]
    note = slot.note()
    assert note.startswith("【此刻相关的记忆】")
    assert note.endswith("- 关于用户：用户养了一只叫团子的橘猫")
    assert "BASE" not in note
    assert not slot.merge([_memory("m2", "用户养了一只叫团子的橘猫")], turn=2)


def test_slot_keeps_recent_hits_first_and_expires_stale_ones():
    slot = MemorySlot("BASE", ttl_turns=2)
    slot.merge([_memory("a", "A 事"), _memory("b", "B 事")], turn=1)
    slot.merge([_memory("c", "C 事"), _memory("a", "A 事")], turn=2)
    assert slot.lines == ["关于用户：C 事", "关于用户：A 事", "关于用户：B 事"]
    assert slot.expire(3)
    assert slot.lines == ["关于用户：C 事", "关于用户：A 事"]
    assert slot.expire(4)
    assert slot.lines == [] and slot.note() == ""
    assert not slot.expire(5)


def test_slot_bounds_lines_and_characters():
    # Each line is 21 characters: two fit in 45, a third does not.
    slot = MemorySlot("BASE", max_lines=3, char_budget=45)
    slot.merge([_memory(str(i), f"第{i}件事" + "很长" * 6) for i in range(6)], turn=1)
    assert len(slot.lines) == 2
    slot = MemorySlot("BASE", max_lines=3, char_budget=1000)
    slot.merge([_memory(str(i), f"第{i}件事") for i in range(6)], turn=1)
    assert len(slot.lines) == 3
    slot = MemorySlot("BASE", char_budget=5)
    slot.merge([_memory("x", "一条很长很长的记忆")], turn=1)
    assert slot.lines == ["关于用户：一条很长很长的记忆"]


# ---------------------------------------------------------------------------
# VoiceBridge: memories first, then the reply
# ---------------------------------------------------------------------------

def _bridge(retrieve, *, instructions: str | None = "BASE", send=None, memory_wait_s=0.5):
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
        memory_wait_s=memory_wait_s,
        instructions=instructions,
    )
    return bridge, provider, client


_ITEMS = iter(range(10_000))


async def _user_says(bridge: VoiceBridge, text: str, *, live: str | None = None) -> str:
    """One full user turn as DashScope reports it; returns the item id."""
    item = f"item_{next(_ITEMS)}"
    await bridge.feed_provider_event({"type": "input_audio_buffer.speech_started", "item_id": item})
    await bridge.feed_provider_event({
        "type": "conversation.item.input_audio_transcription.delta",
        "item_id": item, "text": live if live is not None else text, "stash": "",
    })
    await bridge.feed_provider_event({"type": "input_audio_buffer.speech_stopped", "item_id": item})
    await asyncio.sleep(0)  # the final transcript lands ~0.2 s later in production
    await bridge.feed_provider_event({
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": item, "transcript": text,
    })
    await _settle(bridge)
    return item


async def _settle(bridge: VoiceBridge) -> None:
    for task in (bridge._prefetch_task, bridge._reply_task, bridge._tool_task):
        if task is not None and not task.done():
            try:
                await task
            except asyncio.CancelledError:
                pass


async def _reply_done(bridge: VoiceBridge, text: str = "", status: str = "completed") -> None:
    output = [{"type": "message", "content": [{"type": "audio", "transcript": text}]}] if text else []
    await bridge.feed_provider_event({"type": "response.created"})
    await bridge.feed_provider_event({"type": "response.done", "response": {"status": status, "output": output}})


def _notes(provider: list[dict]) -> list[str]:
    return [
        e["item"]["content"][0]["text"] for e in provider
        if e["type"] == "conversation.item.create" and e["item"].get("role") == "system"
    ]


async def test_reply_waits_for_this_turns_memories_and_carries_them_as_a_note():
    calls: list[tuple] = []

    async def retrieve(text, previous):
        calls.append((text, previous))
        return [_memory("m1", "用户养了一只叫团子的橘猫")]

    bridge, provider, _ = _bridge(retrieve)
    await _user_says(bridge, "我家猫最近不吃饭")
    # Retrieval started from the live transcript when the user stopped; the
    # matching final transcript did not start a second one.
    assert calls == [("我家猫最近不吃饭", None)]
    kinds = [e["type"] for e in provider]
    assert kinds[-2:] == ["conversation.item.create", "response.create"]
    [note] = _notes(provider)
    assert "团子" in note and note.startswith("【此刻相关的记忆】")
    assert not any(e["type"] == "session.update" for e in provider)

    # The provider echoes the note; it is deleted once the reply is done.
    await bridge.feed_provider_event({"type": "conversation.item.created", "item": {
        "id": "note_1", "role": "system", "type": "message", "content": [{"type": "input_text", "text": note}]}})
    await _reply_done(bridge, "团子怎么了？")
    assert provider[-1] == {"event_id": provider[-1]["event_id"], "type": "conversation.item.delete", "item_id": "note_1"}

    await _user_says(bridge, "那后来呢")
    assert calls[-1] == ("那后来呢", "团子怎么了？")


async def test_a_changed_final_transcript_restarts_retrieval():
    calls: list[str] = []

    async def retrieve(text, previous):
        calls.append(text)
        return []

    bridge, _, _ = _bridge(retrieve)
    await _user_says(bridge, "我下周要去乌鲁木齐出差", live="我下周要去")
    assert calls == ["我下周要去", "我下周要去乌鲁木齐出差"]


async def test_a_memory_hit_once_goes_out_with_two_replies_then_rests():
    hits = [[_memory("m1", "用户在准备考研")], [], [], []]

    async def retrieve(text, previous):
        return hits.pop(0)

    bridge, provider, _ = _bridge(retrieve)
    for text in ("最近好累", "今天天气不错", "今天下雨了", "晚饭吃了面"):
        await _user_says(bridge, text)
        await _reply_done(bridge, f"回{text}")
    notes = _notes(provider)
    assert len(notes) == 2 and all("考研" in n for n in notes)


async def test_slow_or_failing_retrieval_never_holds_the_reply():
    async def slow(text, previous):
        await asyncio.sleep(5)
        return []

    bridge, provider, _ = _bridge(slow, memory_wait_s=0.05)
    await _user_says(bridge, "你还记得吗")
    assert provider[-1]["type"] == "response.create" and _notes(provider) == []

    async def broken(text, previous):
        raise RuntimeError("vector store down")

    bridge, provider, _ = _bridge(broken)
    await _user_says(bridge, "你还记得吗")
    assert provider[-1]["type"] == "response.create"


async def test_without_instructions_or_after_close_no_note_is_sent():
    async def found(text, previous):
        return [_memory("m1", "用户喜欢蓝色")]

    bridge, provider, _ = _bridge(found, instructions=None)
    await _user_says(bridge, "我喜欢什么颜色")
    assert _notes(provider) == [] and provider[-1]["type"] == "response.create"

    bridge, provider, _ = _bridge(found)
    await bridge.aclose()
    await _user_says(bridge, "我喜欢什么颜色")
    assert provider == []


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
    # The provider replies with its own item ids.
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
    await configure_session(
        sock, instructions="hi", voice="v", history=[{"role": "user", "text": "你好"}],
    )
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
# Microphone meter (diagnoses calls where the provider never hears the user)
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
    assert summary.endswith("speech_turns=1 echo_dropped=0 memory_skipped=0")


def test_mic_meter_with_no_audio_reads_minus_infinity():
    from app.services.realtime_voice.bridge import _MicMeter

    assert _MicMeter().summary() == "bytes=0 chunks=0 seconds=0.0 peak_dbfs=-inf rms_dbfs=-inf"


# ---------------------------------------------------------------------------
# Opening greeting: a one-off user cue makes the agent speak first
# ---------------------------------------------------------------------------

async def _greeting_bridge(send=None):
    async def retrieve(text, previous):
        return []

    return _bridge(retrieve, send=send)


async def test_greet_sends_a_cue_then_a_response_once():
    from app.services.realtime_voice.bridge import _GREET_CUE

    bridge, provider, _ = await _greeting_bridge()
    await bridge.feed_client_text('{"type": "greet"}')
    await bridge.feed_client_text('{"type": "greet"}')
    cue, create = provider
    assert cue["type"] == "conversation.item.create"
    assert cue["item"]["role"] == "user"
    assert cue["item"]["content"] == [{"type": "input_text", "text": _GREET_CUE}]
    assert create["type"] == "response.create"


async def test_greeting_cue_is_deleted_after_the_greeting_and_never_stored():
    from app.services.realtime_voice.bridge import _GREET_CUE

    bridge, provider, _ = await _greeting_bridge()
    await bridge.feed_client_text('{"type": "greet"}')
    # The provider echoes the cue with its own id.
    await bridge.feed_provider_event({"type": "conversation.item.created", "item": {
        "id": "srv-cue", "role": "user", "type": "message",
        "content": [{"type": "audio", "audio": "", "transcript": _GREET_CUE}]}})
    # Other user items (history, real speech) are not the cue.
    await bridge.feed_provider_event({"type": "conversation.item.created", "item": {
        "id": "srv-hist", "role": "user", "content": [{"type": "audio", "transcript": "你好"}]}})
    assert bridge._greet_cue_id == "srv-cue"
    await bridge.feed_provider_event({"type": "response.done", "response": {"status": "completed", "output": [
        {"type": "message", "content": [{"type": "audio", "transcript": "喂，怎么突然给我打电话呀？"}]}]}})
    assert provider[-1] == {"event_id": provider[-1]["event_id"], "type": "conversation.item.delete",
                            "item_id": "srv-cue"}
    assert bridge.message_ids == ["msg"]
    # Only once: the next reply deletes nothing.
    await bridge.feed_provider_event({"type": "response.done", "response": {"status": "completed"}})
    assert sum(e["type"] == "conversation.item.delete" for e in provider) == 1


async def test_failed_cue_delete_does_not_break_the_call(caplog):
    from app.services.realtime_voice.bridge import _GREET_CUE

    async def send(event):
        if event["type"] == "conversation.item.delete":
            raise RuntimeError("socket closed")

    bridge, _, client = await _greeting_bridge(send=send)
    await bridge.feed_client_text('{"type": "greet"}')
    await bridge.feed_provider_event({"type": "conversation.item.created", "item": {
        "id": "srv-cue", "role": "user", "content": [{"type": "input_text", "text": _GREET_CUE}]}})
    await bridge.feed_provider_event({"type": "response.done", "response": {"status": "completed"}})
    assert "greeting cue not removed" in caplog.text
    assert client[-1]["type"] == "response.done"


async def test_only_the_greeting_cue_we_sent_is_recognised():
    from app.services.realtime_voice.bridge import _GREET_CUE, _item_text

    bridge, _, _ = await _greeting_bridge()
    # Not sent yet: an identical user item is not taken for the cue.
    for item in (
        None, {"id": "x", "role": "user", "content": [{"text": _GREET_CUE}]},
    ):
        await bridge.feed_provider_event({"type": "conversation.item.created", "item": item})
    assert bridge._greet_cue_id is None
    await bridge.feed_client_text('{"type": "greet"}')
    for item in (
        {"id": "a", "role": "assistant", "content": [{"text": _GREET_CUE}]},
        {"role": "user", "content": [{"text": _GREET_CUE}]},
        {"id": "b", "role": "user", "content": "text"},
        {"id": "c", "role": "user", "content": ["raw"]},
    ):
        await bridge.feed_provider_event({"type": "conversation.item.created", "item": item})
    assert bridge._greet_cue_id is None
    assert _item_text({"content": [{"transcript": "嗯"}]}) == "嗯"
    assert _item_text({"content": "x"}) == ""


# ---------------------------------------------------------------------------
# Call opening knows how long ago the two last talked
# ---------------------------------------------------------------------------

def test_history_turns_merge_text_bubbles_into_one_turn():
    from app.services.realtime_voice.instructions import history_turns

    turns = history_turns([
        {"role": "user", "content": "忙吗"},
        {"role": "assistant", "content": "刚弄完裱花"},
        {"role": "assistant", "content": " 正歇着呢 "},
        {"role": "system", "content": "ignored"},
        {"role": "assistant", "content": "  "},
        {"role": "user", "content": "那聊会儿"},
    ], limit=2, char_limit=8)
    assert turns == [{"role": "assistant", "text": "刚弄完裱花 正…"},
                     {"role": "user", "text": "那聊会儿"}]


@pytest.mark.parametrize("gap,was_call,line,cue", [
    (None, False, "", "简短自然地打个招呼"),
    (90, True, "就在刚才（打电话）", "刚挂断又马上打回来"),
    (90, False, "就在刚才（文字聊天）", "刚才还在文字聊天"),
    (40 * 60, True, "40 分钟前（打电话），不是刚才", "隔了 40 分钟又给你打来电话"),
    (5 * 3600, False, "5 小时前（文字聊天）", "已经 5 小时了"),
])
def test_opening_line_and_cue_follow_the_gap(gap, was_call, line, cue):
    from app.services.realtime_voice.instructions import greeting_cue, last_contact_line

    assert line in last_contact_line(gap, was_call=was_call)
    assert bool(last_contact_line(gap, was_call=was_call)) == bool(line)
    assert cue in greeting_cue(gap, was_call=was_call)


def test_last_contact_line_lands_in_the_opening():
    from app.services.realtime_voice.instructions import PersonaFacts, build_instructions

    persona = PersonaFacts("阿宁", "", "女生", 27, "", "", "", "", "", "", "")
    text = build_instructions(persona, [], [], has_history=True, last_contact="你们上一次说话是 3 小时前（打电话），不是刚才。")
    assert "不要重复你上一句说过的话。\n你们上一次说话是 3 小时前" in text


def test_last_contact_reads_the_newest_message():
    from datetime import datetime, timedelta, timezone

    from app.services.realtime_voice.context import _last_contact

    assert _last_contact([]) == (None, False)
    assert _last_contact([{"createdAt": "2026-11-05"}]) == (None, False)
    naive = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=2)
    gap, was_call = _last_contact([{"createdAt": naive, "channel": "realtime_voice"}])
    assert 100 < gap < 140 and was_call
    gap, was_call = _last_contact([{"createdAt": datetime.now(timezone.utc) - timedelta(hours=3), "channel": None}])
    assert 3 * 3600 - 5 < gap < 3 * 3600 + 5 and not was_call


async def test_recent_messages_skip_cards_and_keep_time_and_channel(monkeypatch):
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from app.services.realtime_voice import context

    at = datetime(2026, 11, 5, 10, 0, tzinfo=timezone.utc)
    rows = [
        SimpleNamespace(role="assistant", content="喂", createdAt=at, metadata={"channel": "realtime_voice"}),
        SimpleNamespace(role="assistant", content="活动卡片：咖啡馆", createdAt=at, metadata={"component_card": {"type": "x"}}),
        SimpleNamespace(role="user", content="  ", createdAt=at, metadata=None),
        SimpleNamespace(role="user", content="在吗", createdAt=at, metadata=None),
    ]

    async def find_many(**kwargs):
        assert kwargs["take"] == 40
        return rows

    monkeypatch.setattr(context.db, "message", SimpleNamespace(find_many=find_many), raising=False)
    assert await context._recent_messages("conv") == [
        {"role": "user", "content": "在吗", "createdAt": at, "channel": None},
        {"role": "assistant", "content": "喂", "createdAt": at, "channel": "realtime_voice"},
    ]


async def test_call_context_returns_a_gap_aware_cue(monkeypatch):
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace

    from app.services.realtime_voice import context
    from app.services.realtime_voice.instructions import DynamicContext

    recent = [{"role": "assistant", "content": "下次聊", "channel": "realtime_voice",
               "createdAt": datetime.now(timezone.utc) - timedelta(seconds=30)}]

    async def gather(**_):
        return [], [], recent, DynamicContext(), ["2026-11-04 推荐「江边旧址」：对方放进了待出行，还没去"]

    monkeypatch.setattr(context, "_gather_all", gather)
    agent = SimpleNamespace(name="阿宁", gender="female")
    _, instructions, history, cue = await context.load_call_context(
        agent=agent, user_id="u", workspace_id=None, conversation_id="c")
    assert history == [{"role": "assistant", "text": "下次聊"}]
    assert "就在刚才（打电话）" in instructions
    assert "刚挂断又马上打回来" in cue
    assert "推荐「江边旧址」" in instructions and "只按这份记录说" in instructions


async def test_bridge_greets_with_the_cue_it_was_given():
    async def retrieve(text, previous):
        return []

    provider: list[dict] = []

    async def send(event):
        provider.append(event)

    async def save(*_):
        return "msg"

    bridge = VoiceBridge(send_provider=send, send_client=send, retrieve_memories=retrieve,
                         save_user_line=save, save_assistant_line=save, greet_cue="（刚挂又打来）")
    await bridge.feed_client_text('{"type": "greet"}')
    assert provider[0]["item"]["content"][0]["text"] == "（刚挂又打来）"
    await bridge.feed_provider_event({"type": "conversation.item.created", "item": {
        "id": "srv-cue", "role": "user", "content": [{"type": "audio", "transcript": "（刚挂又打来）"}]}})
    assert bridge._greet_cue_id == "srv-cue"


@pytest.mark.parametrize("query,platform", [
    ({"platform": "Android"}, "android"),
    ({"platform": "ios"}, "ios"),
    ({"platform": "web"}, "unknown"),
    ({}, "unknown"),
])
def test_client_platform_is_logged_from_the_query(query, platform):
    from types import SimpleNamespace

    from app.api.realtime import voice as endpoint

    assert endpoint._client_platform(SimpleNamespace(query_params=query)) == platform
    assert endpoint._client_platform(SimpleNamespace()) == "unknown"


# ---------------------------------------------------------------------------
# Repetition: sampling temperature, deduped history, repeat guard
# ---------------------------------------------------------------------------

def test_temperature_is_sent_and_clamped():
    from app.services.realtime_voice.protocol import session_update_event

    assert "temperature" not in session_update_event(instructions="t", voice="v")["session"]
    for value, expected in ((0.7, 0.7), (-1, 0.0), (9, 1.99)):
        assert session_update_event(instructions="t", voice="v", temperature=value)["session"]["temperature"] == expected


async def test_configure_session_forwards_temperature():
    from app.services.realtime_voice.provider import configure_session

    sent: list[dict] = []

    class Sock:
        def __init__(self):
            self.events = [{"type": "session.created"}, {"type": "session.updated"}]

        async def send_event(self, event):
            sent.append(event)

        async def recv_event(self):
            return self.events.pop(0)

    await configure_session(Sock(), instructions="t", voice="v", history=[], temperature=0.7)
    assert sent[0]["session"]["temperature"] == 0.7


def test_history_drops_lines_the_agent_already_said():
    from app.services.realtime_voice.instructions import history_turns

    line = "怎么了这是？听你这动静，是遇到什么难事了？"
    turns = history_turns([
        {"role": "assistant", "content": line},
        {"role": "assistant", "content": line.replace("？", "?")},
        {"role": "user", "content": "您看。"},
        {"role": "assistant", "content": "对呀"},
        {"role": "user", "content": "嗯"},
        {"role": "assistant", "content": "对呀"},
        {"role": "assistant", "content": line},
    ])
    assert turns == [
        {"role": "assistant", "text": line},
        {"role": "user", "text": "您看。"},
        {"role": "assistant", "text": "对呀"},
        {"role": "user", "text": "嗯"},
        {"role": "assistant", "text": "对呀"},
    ]


def test_slot_names_avoided_lines_after_the_memory_block():
    slot = MemorySlot("BASE")
    assert slot.note() == ""
    assert slot.avoid(" 第一句 ")
    assert not slot.avoid("第一句")
    assert not slot.avoid("  ")
    slot.merge([{"id": "m", "raw": "喜欢猫", "line": "关于用户：喜欢猫"}], 1)
    assert slot.avoid("第二句") and slot.avoid("第三句")
    text = slot.note()
    assert text.startswith("【此刻相关的记忆】")
    assert text.endswith("【别再重复】你这通电话里已经说过「第二句」、「第三句」，接下来换个说法或换个话题，不要再说这句。")


_REPEAT = "行吧，那我就当是受宠若惊了。不过你这奖励也太含糊了，我这心里正打鼓呢，到底是什么好事呀？"


async def _say(bridge, text):
    await bridge.feed_provider_event({"type": "response.done", "response": {"status": "completed", "output": [
        {"type": "message", "content": [{"type": "audio", "transcript": text}]}]}})


async def _next_note(bridge, provider) -> str:
    """The note sent with the reply to the next user turn ("" when none)."""
    before = len(_notes(provider))
    await _user_says(bridge, "嗯，然后呢")
    notes = _notes(provider)
    return notes[-1] if len(notes) > before else ""


async def test_repeat_guard_names_a_near_identical_reply_once(caplog):
    caplog.set_level(logging.INFO, logger="app.services.realtime_voice.bridge")

    async def retrieve(text, previous):
        return []

    bridge, provider, _ = _bridge(retrieve)
    await _say(bridge, _REPEAT)
    await _say(bridge, "对呀")
    await _say(bridge, "对呀")
    assert "repeat detected" not in caplog.text
    second = _REPEAT.replace("行吧", "行")
    await _say(bridge, second)
    assert "repeat detected" in caplog.text
    assert f"已经说过「{second}」" in await _next_note(bridge, provider)
    await _say(bridge, second)
    assert caplog.text.count("repeat detected") == 1


async def test_repeat_guard_window_and_without_a_slot(caplog):
    caplog.set_level(logging.INFO, logger="app.services.realtime_voice.bridge")

    async def retrieve(text, previous):
        return []

    bridge, provider, _ = _bridge(retrieve)
    for line in (_REPEAT, "今天店里活儿挺多的一直忙到现在", "苏式焖肉面是我们这里的招牌", "你平时都喜欢吃些什么呀告诉我"):
        await _say(bridge, line)
    await _say(bridge, _REPEAT)  # fell out of the last three replies
    assert "repeat detected" not in caplog.text
    assert await _next_note(bridge, provider) == ""

    bridge, provider, _ = _bridge(retrieve, instructions=None)
    await _say(bridge, _REPEAT)
    await _say(bridge, _REPEAT)
    assert "repeat detected" not in caplog.text
    await _user_says(bridge, "嗯")
    assert _notes(provider) == []


# ---------------------------------------------------------------------------
# Review fixes: client interrupt, error matching, note ids, stuck segments
# ---------------------------------------------------------------------------

async def test_client_interrupt_keeps_what_was_said_before_the_cut():
    saved: list[tuple] = []

    async def save_assistant(text, user_msg_id, interrupted):
        saved.append((text, interrupted))
        return "a1"

    bridge, provider, client = _bridge(lambda *_: asyncio.sleep(0, []))
    bridge._save_assistant_line = save_assistant
    await _user_says(bridge, "讲个故事吧")
    await bridge.feed_provider_event({"type": "response.created"})
    await bridge.feed_provider_event({"type": "response.audio_transcript.delta", "delta": "从前有座山"})
    await bridge.feed_client_text('{"type": "interrupt"}')
    assert saved == [("从前有座山", True)]
    assert client[-1] == {"type": "interrupted"}
    # The cancelled reply's own transcript is not stored a second time.
    await bridge.feed_provider_event({"type": "response.done", "response": {"status": "cancelled", "output": [
        {"type": "message", "content": [{"type": "audio", "transcript": "从前有座山，山里有座庙"}]}]}})
    assert saved == [("从前有座山", True)]
    assert any(e["type"] == "response.cancel" for e in provider)


async def test_barge_in_before_the_reply_starts_keeps_the_call_alive():
    """2026-11-07: the user spoke right as a reply was requested. The cancel
    was rejected ("Cannot cancel response before response.created"), the
    error freed the reply and deleted its note, and DashScope closed the call
    ("previous item id does not exist")."""
    async def retrieve(text, previous):
        return [_memory("m1", "用户在学吉他")]

    bridge, provider, client = _bridge(retrieve)
    await _user_says(bridge, "我今天练琴了")
    assert provider[-1]["type"] == "response.create"
    await bridge.feed_provider_event({"type": "conversation.item.created", "item": {
        "id": "item_srv_note", "role": "system", "content": [{"type": "input_text", "text": "x"}]}})
    sent = len(provider)
    await bridge.feed_provider_event({"type": "input_audio_buffer.speech_started", "item_id": "late"})
    # No cancel before response.created; nothing deleted.
    assert provider[sent:] == []
    await bridge.feed_provider_event({"type": "error", "error": {
        "message": "Cannot cancel response before response.created"}})
    assert bridge._responding and bridge._note_id == "item_srv_note"
    assert client[-1]["type"] == "warning"
    # The cancel goes out once the reply exists.
    await bridge.feed_provider_event({"type": "response.created"})
    assert [e["type"] for e in provider[sent:]] == ["response.cancel"]
    # The note is deleted only after the reply is done.
    await bridge.feed_provider_event({"type": "response.done", "response": {"status": "cancelled"}})
    deletes = [e["item_id"] for e in provider[sent:] if e["type"] == "conversation.item.delete"]
    assert deletes == ["item_srv_note"]


async def test_a_reply_that_never_starts_is_released(monkeypatch):
    from app.services.realtime_voice import bridge as bridge_module

    monkeypatch.setattr(bridge_module, "_REPLY_START_TIMEOUT_S", 0.01)

    async def retrieve(text, previous):
        return [_memory("m1", "用户在学吉他")]

    bridge, provider, _ = _bridge(retrieve)
    await _user_says(bridge, "我今天练琴了")
    await bridge.feed_provider_event({"type": "conversation.item.created", "item": {
        "id": "item_srv_note", "role": "system", "content": []}})
    await asyncio.sleep(0.05)
    assert not bridge._responding
    assert provider[-1] == {"event_id": provider[-1]["event_id"],
                            "type": "conversation.item.delete", "item_id": "item_srv_note"}
    # A reply that did start is left alone.
    await _user_says(bridge, "弹了一小时")
    await bridge.feed_provider_event({"type": "response.created"})
    await asyncio.sleep(0.05)
    assert bridge._responding


async def test_each_reply_note_carries_the_current_time():
    async def retrieve(text, previous):
        return []

    provider: list[dict] = []

    async def send(event):
        provider.append(event)

    async def save(*_):
        return "msg"

    async def client(_):
        pass

    bridge = VoiceBridge(
        send_provider=send, send_client=client, retrieve_memories=retrieve,
        save_user_line=save, save_assistant_line=save, retrieval_timeout_s=0.2,
        memory_wait_s=0.05, instructions="BASE", clock_line=lambda: "【现在的时间】07:16",
    )
    await _user_says(bridge, "现在几点了")
    assert _notes(provider) == ["【现在的时间】07:16"]


def test_clock_line_is_to_the_minute(monkeypatch):
    from datetime import datetime
    from app.services.realtime_voice import instructions
    from app.services.time_domain import clock

    now = datetime(2026, 11, 7, 7, 16)
    monkeypatch.setattr(clock, "get_current_time", lambda: clock.TimeInfo(
        now=now, date=now.date(), weekday="星期六", is_weekend=True, timestamp_ms=0))
    line = instructions.clock_line()
    assert line.startswith("【现在的时间】11月7日 星期六 07:16。")


async def test_note_id_is_taken_even_if_the_echoed_text_differs():
    async def retrieve(text, previous):
        return [_memory("m1", "用户喜欢爬山")]

    bridge, provider, _ = _bridge(retrieve)
    await _user_says(bridge, "周末去哪玩")
    await bridge.feed_provider_event({"type": "conversation.item.created", "item": {
        "id": "item_a", "role": "system", "content": [{"type": "input_text", "text": "（规范化后的文本）"}]}})
    # A second system item in the same reply does not replace the first.
    await bridge.feed_provider_event({"type": "conversation.item.created", "item": {
        "id": "item_b", "role": "system", "content": []}})
    await _reply_done(bridge, "去爬山吧")
    deletes = [e["item_id"] for e in provider if e["type"] == "conversation.item.delete"]
    assert deletes == ["item_a"]


async def test_speech_without_a_stop_ends_after_the_cap(monkeypatch):
    from app.services.realtime_voice import bridge as bridge_module

    monkeypatch.setattr(bridge_module, "_SEGMENT_MAX_S", 0.01)
    bridge, provider, _ = _bridge(lambda *_: asyncio.sleep(0, []))
    await bridge.feed_provider_event({"type": "input_audio_buffer.speech_started", "item_id": "stuck"})
    await bridge.feed_provider_event({
        "type": "conversation.item.input_audio_transcription.delta",
        "item_id": "stuck", "text": "我想跟你说件事", "stash": ""})
    assert bridge._user_speaking()
    await asyncio.sleep(0.05)
    await _settle(bridge)
    assert not bridge._user_speaking()
    assert provider[-1]["type"] == "response.create"


# ---------------------------------------------------------------------------
# VOICE01.03: small talk, per-turn cap, memories that keep coming back
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    # Spoken on production calls, 2026-11-04 to 11-07
    "喂喂喂喂。", "嗯。", "哈喽哈喽", "能听到吗？能听到吗？", "我听得清", "测试", "嗯拜拜",
    "对的对的对的", "可以可以可以", "好吧好吧好吧", "Hello在吗", "喂喂喂，你在吗", "嗯，对对对",
])
def test_small_talk_is_recognised(text):
    from app.services.realtime_voice.instructions import is_small_talk

    assert is_small_talk(text)


@pytest.mark.parametrize("text", [
    "", "。", "你在干嘛呢", "哈喽你在干嘛呀", "嗯我还没想好呢", "没呢信号挺好呢", "你推荐过我吗",
    "妈妈", "嗯，然后呢", "测试结果怎么样了", "喂" * 21,
])
def test_turns_with_content_are_not_small_talk(text):
    from app.services.realtime_voice.instructions import is_small_talk

    assert not is_small_talk(text)


def test_slot_takes_at_most_three_new_memories_per_turn():
    slot = MemorySlot("BASE")
    slot.merge([_memory(str(i), f"第{i}件事") for i in range(6)], turn=1)
    assert slot.lines == ["关于用户：第0件事", "关于用户：第1件事", "关于用户：第2件事"]
    slot.merge([_memory(str(i), f"第{i}件事") for i in range(3, 6)], turn=2)
    assert len(slot.lines) == 4 and slot.lines[0] == "关于用户：第3件事"


def test_a_memory_shown_twice_rests_until_it_expires_then_can_return():
    slot = MemorySlot("BASE", ttl_turns=2)
    for turn in (1, 2):
        slot.merge([_memory("t", "用户在测试信号")], turn=turn)
        assert slot.lines == ["关于用户：用户在测试信号"]
        slot.mark_shown()
    # Hit again on turns 3 and 4: still held back, and the hits keep it alive.
    for turn in (3, 4):
        slot.merge([_memory("t", "用户在测试信号"), _memory("c", "用户养了猫")], turn=turn)
        assert slot.lines == ["关于用户：用户养了猫"]
    slot.expire(6)
    slot.merge([_memory("t", "用户在测试信号")], turn=7)
    assert slot.lines == ["关于用户：用户在测试信号"]


def test_note_without_memories_keeps_the_avoided_lines():
    slot = MemorySlot("BASE")
    slot.merge([_memory("m", "用户养了猫")], turn=1)
    slot.avoid("同一句话")
    quiet = slot.note(with_memories=False)
    assert "养了猫" not in quiet and quiet.startswith("【别再重复】")


async def test_small_talk_retrieves_nothing_and_its_reply_carries_no_memories():
    queries = []

    async def retrieve(text, previous):
        queries.append(text)
        return [_memory("m1", "用户在准备考研")]

    bridge, provider, _ = _bridge(retrieve)
    await _user_says(bridge, "最近复习好累")
    await _reply_done(bridge, "辛苦啦")
    await _user_says(bridge, "喂喂喂喂")
    await _reply_done(bridge, "在呢")
    await _user_says(bridge, "嗯")
    await _reply_done(bridge, "怎么啦")
    assert queries == ["最近复习好累"]
    assert len(_notes(provider)) == 1 and "考研" in _notes(provider)[0]
    assert bridge.mic_summary.endswith("memory_skipped=2")


async def test_small_talk_before_the_last_turn_is_answered_keeps_its_memories():
    async def retrieve(text, previous):
        return [_memory("m1", "用户在准备考研")]

    bridge, provider, _ = _bridge(retrieve)
    # The agent is still finishing an earlier reply, so neither turn is answered yet.
    bridge._responding = True
    await _user_says(bridge, "最近复习好累")
    await _user_says(bridge, "嗯")
    assert not _notes(provider)
    bridge._responding = False
    await bridge._maybe_reply()
    assert _notes(provider) and "考研" in _notes(provider)[-1]
    assert bridge.mic_summary.endswith("memory_skipped=0")
