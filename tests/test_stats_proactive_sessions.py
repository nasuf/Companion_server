import json
from datetime import timedelta
from uuid import uuid4
from unittest.mock import patch

from app.services.effect import daily_facts, proactive_stats, weekly
from tests.test_beta_daily_facts_postgres import (
    pg,
    _user,
    _conversation,
    _at,
    _now,
    _msg,
    _seed,
    _metrics,
    DAY,
)


async def test_proactive_event_dedup_push_and_wakeup(pg):
    user = await _user(pg)
    conv, agent = await _conversation(pg, user)
    event = "proactive-" + uuid4().hex
    await _msg(pg, conv, "user", _at(DAY - timedelta(days=3), 10))
    mid = uuid4().hex
    metadata = dict(
        proactive=True, proactive_event_id=event, trigger_type="silence_wakeup"
    )
    await pg.db.execute_raw(
        "INSERT INTO messages(id,conversation_id,role,content,created_at,metadata) VALUES($1,$2,'assistant','合成',$3::timestamp,$4::jsonb)",
        mid,
        conv,
        _at(DAY, 10).isoformat(),
        json.dumps(metadata),
    )
    await _msg(pg, conv, "assistant", _at(DAY, 10, 0, 1), metadata)
    for minute in (1, 3, 5):
        await _msg(pg, conv, "user", _at(DAY, 10, minute))
        await _msg(pg, conv, "assistant", _at(DAY, 10, minute, 1))
    await pg.db.execute_raw(
        "INSERT INTO notification_events(id,user_id,agent_id,conversation_id,message_id,type,title,body,dedupe_key,status,created_at,updated_at) VALUES($1,$2,$3,$4,$5,'agent_message','合成','合成',$5,'sent',$6::timestamp,$6::timestamp)",
        uuid4().hex,
        user,
        agent,
        conv,
        mid,
        _at(DAY, 10).isoformat(),
    )
    now = _now(DAY + timedelta(days=3))
    await daily_facts.compute_day(DAY, now=now)
    with patch.object(
        proactive_stats.defs, "local_today", return_value=DAY + timedelta(days=3)
    ):
        out = await proactive_stats.read(DAY, DAY, "external")
    assert out["total"]["sent"] == 1 and out["total"]["push_accepted"] == 1
    assert out["total"]["answered_3h"] == 1 and out["total"]["three_turns"] == 1
    assert out["total"]["silent_2d"] == 1 and out["total"]["awakened_24h"] == 1
    assert out["total"]["latency"]["≤1分钟"] == 1
    assert "delivered" in out["unavailable"] and "opened" in out["unavailable"]
    # Recomputing remains idempotent, not another conversion.
    await daily_facts.compute_day(DAY, now=now)
    metric = (await _metrics(pg, DAY, "all"))["proactive_sent"]
    assert metric["value"] == 1
    # Prior schema lacking IDs is explicitly not a measured funnel event.
    await _msg(pg, conv, "assistant", _at(DAY, 14), {"proactive": True})
    await daily_facts.compute_day(DAY, now=now)
    assert (await proactive_stats.read(DAY, DAY, "all"))["total"]["legacy"] == 1


async def test_ended_session_depth_bubbles_and_guarded_review_queue(pg):
    user = await _user(pg)
    conv, _ = await _conversation(pg, user)
    # One burst spans midnight. Depth is one, not two user bubbles/two AI bubbles.
    await _msg(pg, conv, "user", _at(DAY - timedelta(days=1), 23, 59))
    await _msg(pg, conv, "user", _at(DAY, 0, 1))
    await _msg(pg, conv, "assistant", _at(DAY, 0, 1, 1))
    await _msg(pg, conv, "assistant", _at(DAY, 0, 1, 2))
    await daily_facts.compute_day(DAY, now=_now(DAY, 2))
    metrics = await _metrics(pg, DAY, "external")
    assert metrics["session_depth"]["value"] == 0
    await daily_facts.compute_day(DAY, now=_now(DAY + timedelta(days=3)))
    metrics = await _metrics(pg, DAY, "external")
    assert metrics["session_depth"]["value"] == 1
    assert (
        metrics["short_session"]["numerator"] == 1
        and metrics["short_session"]["denominator"] == 1
    )
    # Missing-source days do not permit queueing; after a complete day only IDs/counts persist.
    import pytest

    with pytest.raises(ValueError):
        await weekly.enqueue(DAY, DAY + timedelta(days=1), "external", user)
    await pg.db.execute_raw(
        "INSERT INTO stats_review_items(kind,source_id,local_date,user_id,conversation_id,occurred_at,definition_version,computed_at) VALUES('resend_60s',$1,$2::date,$3,$4,$5::timestamp,$6,NOW())",
        uuid4().hex,
        str(DAY),
        user,
        conv,
        _at(DAY, 10).isoformat(),
        daily_facts.DEFINITION_VERSION,
    )
    out = await weekly.enqueue(DAY, DAY, "external", user)
    assert out["added"] == 1
    assert (await weekly.enqueue(DAY, DAY, "external", user))["added"] == 0
    queued = (
        await pg.db.query_raw(
            "SELECT counts FROM stats_review_queue WHERE user_id=$1", user
        )
    )[0]
    assert queued["counts"] == {"resend_60s": 1}


def test_question_and_safe_proactive_labels():
    from app.services.effect.session_stats import is_question

    assert (
        is_question("你去吗？🙂")
        and is_question("觉得如何 ✨")
        and not is_question("那就好")
    )
    assert proactive_stats.safe_label("send_skipped") == "send_skipped"
    assert proactive_stats.safe_label("user phone or message") == "unknown"
