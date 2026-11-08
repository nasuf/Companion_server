"""VOICE01.01: per-call diagnostics counters and summary (no transcript text)."""
import pytest

from app.services.realtime_voice.call_stats import CallStats, percentiles


def test_percentiles_use_nearest_rank_and_report_none_when_empty():
    assert percentiles([]) is None
    assert percentiles([5]) == {"p50": 5, "p90": 5, "max": 5, "n": 1}
    p = percentiles([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    assert p == {"p50": 50, "p90": 90, "max": 100, "n": 10}


def test_first_audio_only_counts_after_a_user_turn_ended():
    s = CallStats()
    s.first_audio(at=5.0)  # greeting before any user turn: ignored
    assert s.first_audio_ms == []
    s.turn_ended(at=10.0)
    s.first_audio(at=10.4)
    s.first_audio(at=10.9)  # same turn, second reply chunk: ignored until next turn_ended
    assert s.first_audio_ms == [400]


def test_retrieval_and_injection_counts_feed_the_summary():
    s = CallStats(platform="ios", model="qwen-omni", vad_threshold=0.5, silence_ms=600)
    s.retrieval(0.25)
    s.retrieval(1.5, timed_out=True)
    s.retrieval_result(candidates=30, selected=4)
    s.retrieval_result(candidates=18, selected=5)
    s.reply_note(4)
    s.reply_note(0)  # small-talk reply carried no memories
    s.barge_ins = 2
    out = s.summary(turns=6, speech_turns=8, echo_dropped=1, memory_skipped=2,
                    mic={"seconds": 40.0, "peak_dbfs": -6.0}, usage={"responses": 6})
    assert out["environment"]["platform"] == "ios" and out["environment"]["vad_threshold"] == 0.5
    assert out["retrieval"]["timeouts"] == 1 and out["retrieval"]["count"] == 2
    assert out["retrieval"]["candidates"]["max"] == 30 and out["retrieval"]["selected"]["max"] == 5
    assert out["retrieval"]["rejected_total"] == (30 - 4) + (18 - 5)
    assert out["injection"]["replies"] == 2 and out["injection"]["replies_with_memories"] == 1
    assert out["injection"]["small_talk_skipped"] == 2
    assert out["turns"] == {"answered": 6, "speech_segments": 8, "echo_dropped": 1,
                            "barge_ins": 2, "client_interrupts": 0, "repeats": 0, "tool_calls": 0}
    assert out["samples"]["retrieval_ms"] == [250, 1500]


def test_metrics_never_collected_report_none_not_zero():
    out = CallStats().summary(turns=0, speech_turns=0, echo_dropped=0, memory_skipped=0, mic=None, usage=None)
    assert out["first_audio_ms"] is None and out["retrieval"]["latency_ms"] is None
    assert out["retrieval"]["candidates"] is None and out["retrieval"]["rejected_total"] is None
    assert out["usage"] is None


def test_samples_are_bounded():
    from app.services.realtime_voice import call_stats as cs

    s = CallStats()
    for i in range(cs._MAX_SAMPLES + 50):
        s.retrieval(0.01)
    assert len(s.retrieval_ms) == cs._MAX_SAMPLES


# ── 落库与基线: 真 PG ─────────────────────────────────────────────────
from datetime import UTC, datetime, timedelta  # noqa: E402

from tests.test_service_health_postgres import world  # noqa: E402,F401


@pytest.fixture
async def calls_db(world, monkeypatch):
    database, conv, space, user, _ = world
    monkeypatch.setattr("app.db.db", database)
    try:
        yield database, conv, space, user
    finally:
        await database.execute_raw("DELETE FROM realtime_call_sessions WHERE user_id=$1", user)


async def _store(database, user, conv, space, *, first_audio, retrieval, started=None, platform="ios"):
    from app.services.realtime_voice import call_stats as cs

    s = CallStats(platform=platform, model="qwen-omni")
    if started:
        s.started_at = started
    for ms in first_audio:
        s.turn_ended(at=0.0)
        s.first_audio(at=ms / 1000)
    for ms in retrieval:
        s.retrieval(ms / 1000)
    s.retrieval_result(candidates=20, selected=3)
    s.reply_note(3)
    summary = s.summary(turns=len(first_audio), speech_turns=len(first_audio), echo_dropped=0,
                        memory_skipped=0, mic=None, usage=None)
    await cs.persist(s, summary, user_id=user, workspace_id=space, conversation_id=conv, agent_id="agent")


async def test_calls_are_stored_and_merged_into_a_baseline(calls_db):
    from app.services.realtime_voice import call_stats as cs

    database, conv, space, user = calls_db
    await _store(database, user, conv, space, first_audio=[400, 600], retrieval=[300])
    await _store(database, user, conv, space, first_audio=[800], retrieval=[500, 900], platform="android")
    await _store(database, user, conv, space, first_audio=[5000], retrieval=[100],
                 started=datetime.now(UTC) - timedelta(days=30))  # outside the 7-day window

    out = await cs.baseline(user_id=user, days=7)
    assert out["call_count"] == 2 and len(out["calls"]) == 2
    assert out["baseline"]["first_audio_ms"] == {"p50": 600, "p90": 800, "max": 800, "n": 3}
    assert out["baseline"]["retrieval_ms"]["n"] == 3 and out["baseline"]["injected"]["max"] == 3
    newest = out["calls"][0]
    assert newest["platform"] == "android" and newest["turns"] == 1
    assert newest["rejected_total"] == 17 and newest["injection"]["replies_with_memories"] == 1
    # no transcript text anywhere in the row
    rows = await database.query_raw("SELECT stats::text AS s FROM realtime_call_sessions WHERE user_id=$1", user)
    assert all("text" not in r["s"] and "transcript" not in r["s"] for r in rows)

    wide = await cs.baseline(user_id=user, days=90, limit=1)
    assert wide["call_count"] == 3 and len(wide["calls"]) == 1
    assert (await cs.baseline(user_id=user, days=9999))["window"]["days"] == 90


async def test_a_failed_insert_never_breaks_the_hang_up(monkeypatch, caplog):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from app.services.realtime_voice import call_stats as cs

    monkeypatch.setattr("app.db.db", SimpleNamespace(execute_raw=AsyncMock(side_effect=RuntimeError("db down"))))
    await cs.persist(CallStats(), {}, user_id="u", workspace_id=None, conversation_id="c", agent_id="a")
    assert "call stats not stored" in caplog.text


async def test_admin_baseline_endpoint_passes_filters(monkeypatch):
    from unittest.mock import AsyncMock

    from app.api.admin import realtime_calls as admin
    from app.services.realtime_voice import call_stats as cs

    fake = AsyncMock(return_value={"calls": []})
    monkeypatch.setattr(cs, "baseline", fake)
    assert await admin.call_baseline(user_id="u", days=30, limit=20, _={}) == {"calls": []}
    fake.assert_awaited_once_with(user_id="u", days=30, limit=20)
