"""Topic scope, statement-clock expiry and opt-out output contract."""

from datetime import datetime, timedelta, timezone
from dataclasses import replace
import pytest

from app.services.time_domain.topic_policy import (
    parse_commands,
    topic_id,
    TopicRestriction,
)
from app.services.time_domain.topic_store import TopicSnapshot
from app.services.time_domain.followup import (
    FollowupPolicy,
    derive_followup_policy,
    guard_optional_questions,
)

NOW = datetime(2026, 11, 1, 10, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "topic",
    ["面试", "升职", "前任", "工资", "考试", "相亲", "买房", "离职", "体重", "失眠"],
)
@pytest.mark.parametrize(
    "template,scope,action",
    [
        ("暂时别问{}了", "topic", "pause"),
        ("以后不要再提{}了", "topic", "stop"),
        ("现在可以问{}了", "topic", "resume"),
        ("今天别问{}了", "topic", "pause"),
        ("{}以后别再问了", "topic", "stop"),
    ],
)
def test_explicit_topic_commands(topic, template, scope, action):
    commands = parse_commands(template.format(topic), statement_at=NOW)
    assert len(commands) == 1
    item = commands[0]
    assert (item.topic, item.scope, item.action) == (topic, scope, action)
    assert item.source_quote in template.format(topic)
    assert (item.expires_at is None) == (action in {"stop", "resume"})


@pytest.mark.parametrize(
    "topic",
    ["面试", "升职", "前任", "工资", "考试", "相亲", "买房", "离职", "体重", "失眠"],
)
@pytest.mark.parametrize(
    "template",
    [
        "同事说“以后不要问{}了”",
        "如果以后我说不要问{}了，你会怎么办",
        "我不是让你别问{}",
        "她说以后别问{}了",
        "解释一下“以后别问{}”是什么意思",
        "我没说以后不要问{}了",
        "假如我说以后别问{}了",
        "不要不问{}了",
        "翻译：以后别问{}了",
    ],
)
def test_non_requests_cannot_change_preferences(topic, template):
    assert parse_commands(template.format(topic), statement_at=NOW) == ()


def test_ambiguous_target_never_becomes_permanent_global_preference():
    command = parse_commands("这件事以后别提了", statement_at=NOW)[0]
    assert (command.scope, command.action, command.topic) == ("session", "pause", "")
    assert command.expires_at == NOW + timedelta(hours=3)


def test_global_preference_and_session_resume_are_distinct_keys():
    a = parse_commands("我不喜欢一直被问", statement_at=NOW)[0]
    b = parse_commands("现在可以问了", statement_at=NOW)[0]
    assert (a.scope, a.action, a.expires_at) == ("global", "stop", None)
    assert b.scope == "session" and a.key != b.key


def test_today_expires_at_business_midnight_not_execution_time():
    command = parse_commands("今天别问面试了", statement_at=NOW)[0]
    assert command.expires_at == datetime(2026, 11, 1, 16, tzinfo=timezone.utc)


def test_topic_identity_is_generation_and_owner_scoped():
    assert (
        len(
            {
                topic_id("w", "g", "topic:面试"),
                topic_id("w2", "g", "topic:面试"),
                topic_id("w", "g2", "topic:面试"),
            }
        )
        == 3
    )


def restriction(topic="面试", scope="topic", action="stop", expires=None):
    return TopicRestriction("topic-id", topic, scope, action, expires, "source", 1)


def policy(*items, status="recorded", context=""):
    return FollowupPolicy(
        topic_snapshot=TopicSnapshot(status, items, sampled_at=NOW),
        topic_context=context,
    )


def test_named_stop_removes_only_named_topic_not_unrelated_question():
    result, removed = guard_optional_questions(
        ["面试怎么样？", "今天吃什么？", "我在做饭。"], policy(restriction())
    )
    assert result == ["今天吃什么？", "我在做饭。"] and removed == 1


def test_named_stop_also_removes_unprompted_remention_and_contextual_question():
    result, removed = guard_optional_questions(
        ["还是聊聊面试吧。", "后来怎么样？", "我在做饭。"],
        policy(restriction(), context="关于面试"),
    )
    assert result == ["我在做饭。"] and removed == 2


def test_global_stop_does_not_remove_declarative_empathy():
    result, removed = guard_optional_questions(
        ["我在呢。", "想聊什么？"], policy(restriction("", "global"))
    )
    assert result == ["我在呢。"] and removed == 1


def test_expired_pause_is_not_effective():
    assert (
        guard_optional_questions(["面试怎么样？"], policy(restriction(expires=NOW)))[1]
        == 0
    )


@pytest.mark.parametrize(
    "status", ["unavailable", "scope_unavailable", "capacity_exceeded"]
)
def test_storage_failure_is_not_consent(status):
    assert guard_optional_questions(
        ["你怎么样？"], policy(status=status), fallback=None
    ) == ([], 1)


def test_named_stop_does_not_activate_legacy_global_stop():
    result = derive_followup_policy([], now=NOW, current_message="以后别问面试了")
    assert result.optional_questions_allowed


def test_diagnostic_bounds_and_unknown_are_explicit():
    p = policy(*(restriction(str(i)) for i in range(30)))
    diag = p.diagnostics()["topic_preferences"]
    assert diag["item_count"] == 30 and len(diag["items"]) == 20
    assert diag["sampled_at"] == NOW.isoformat() and not diag["complete_history"]


@pytest.mark.parametrize(
    "text",
    [
        "朋友让我别再问我了",
        "老板希望我以后不要问工资了",
        "我不希望你不问我",
        "他们讨论别问面试了这句话",
    ],
)
def test_reported_requests_outside_old_said_vocabulary_cannot_persist(text):
    assert parse_commands(text, statement_at=NOW) == ()


@pytest.mark.asyncio
async def test_proactive_generation_storage_failure_is_fail_closed(monkeypatch):
    from app.services.proactive import followup
    from app.services.time_domain import topic_store
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    monkeypatch.setattr(
        topic_store,
        "load_snapshot",
        AsyncMock(side_effect=OSError("owned synthetic outage")),
    )
    render = AsyncMock()
    monkeypatch.setattr(followup, "render_prompt", render)
    state = SimpleNamespace(user_id="synthetic", workspace_id="synthetic")
    turns = [followup._Turn("synthetic", "assistant", "面试怎么样？", False, NOW)]
    assert await followup.generate_followup_message(state, turns, NOW) is None
    render.assert_not_awaited()


@pytest.mark.parametrize(
    "message,expected",
    [
        ("在2026年12月1日之前别问面试了", "2026-11-30T16:00:00+00:00"),
        ("到2026年12月1日为止别问面试了", "2026-12-01T16:00:00+00:00"),
        ("明天前别问面试了", "2026-11-01T16:00:00+00:00"),
    ],
)
def test_explicit_expiry_reuses_canonical_statement_time_query(message, expected):
    command = parse_commands(message, statement_at=NOW)[0]
    assert command.action == "pause" and command.scope == "topic"
    assert command.expires_at == datetime.fromisoformat(expected)
    assert command.source_quote == message


@pytest.mark.parametrize("message", ["月底之前别问面试了", "2025年1月1日前别问面试了"])
def test_ambiguous_or_past_expiry_never_turns_into_permanent_topic_stop(message):
    command = parse_commands(message, statement_at=NOW)[0]
    assert (
        command.scope == "session" and command.reason == "ambiguous_expiry_session_only"
    )


def test_today_expiry_uses_configured_timezone(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "schedule_timezone", "America/New_York")
    anchor = datetime(2026, 11, 1, 4, 30, tzinfo=timezone.utc)  # DST fallback day
    command = parse_commands("今天别问面试了", statement_at=anchor)[0]
    assert command.expires_at == datetime(2026, 11, 2, 5, tzinfo=timezone.utc)


def _frozen_contract_cases():
    import json
    from pathlib import Path

    folder = Path(__file__).resolve().parents[1] / "evals/temporal/topic_preferences"
    return [
        case
        for file in sorted(folder.glob("*.json"))
        for case in json.loads(file.read_text())
    ]


@pytest.mark.parametrize("case", _frozen_contract_cases(), ids=lambda c: c["id"])
def test_frozen_source_contract(case):
    commands = parse_commands(case["message"], statement_at=NOW)
    if case["expected_scope"] is None:
        assert not commands
    else:
        assert len(commands) == 1
        assert commands[0].scope == case["expected_scope"]
        assert commands[0].action == case["expected_action"]
        if case["expected_scope"] == "topic":
            assert commands[0].topic == case["topic"]


@pytest.mark.parametrize(
    "message,label",
    [
        ("以后别问“面试”了", "面试"),
        ("暂时不要提「升职」了", "升职"),
        ("现在可以问“考试”了", "考试"),
    ],
)
def test_quoted_noun_is_a_target_and_original_evidence_is_preserved(message, label):
    result = parse_commands(message, statement_at=NOW)
    assert len(result) == 1 and result[0].scope == "topic" and result[0].topic == label
    assert result[0].source_quote == message


@pytest.mark.parametrize(
    "text",
    [
        "“以后别问面试了”",
        "她说以后别问“面试”了",
        "解释“别问面试”",
        "别问“可以问工资”了",
    ],
)
def test_quoted_instruction_never_becomes_permanent_named_preference(text):
    assert not any(
        c.action == "stop" and c.scope != "session"
        for c in parse_commands(text, statement_at=NOW)
    )


def test_no_active_preference_does_not_change_model_input_or_cache_prefix():
    baseline = FollowupPolicy()
    candidate = replace(
        baseline, topic_snapshot=TopicSnapshot("no_active_restrictions", sampled_at=NOW)
    )
    assert candidate.prompt_value() == baseline.prompt_value()
    assert (
        candidate.diagnostics()["topic_preferences"]["status"]
        == "no_active_restrictions"
    )


@pytest.mark.asyncio
async def test_ingress_captures_before_short_circuit_and_never_drops_required_reply(
    monkeypatch,
):
    from app.services.time_domain import topic_store
    from unittest.mock import AsyncMock

    writer = AsyncMock(return_value=1)
    monkeypatch.setattr(topic_store, "record_commands", writer)
    kwargs = dict(user_id="u", conversation_id="c", message_ids={"m"}, statement_at=NOW)
    assert await topic_store.record_turn_preferences(
        current_message="以后别提面试", **kwargs
    ) == {"status": "recorded", "applied_commands": 1}
    writer.assert_awaited_once()
    writer.side_effect = OSError("synthetic")
    assert (
        await topic_store.record_turn_preferences(
            current_message="以后别提面试", **kwargs
        )
    )["status"] == "unavailable"
    writer.reset_mock()
    assert (
        await topic_store.record_turn_preferences(
            current_message="今天吃了面", **kwargs
        )
    )["status"] == "not_requested"
    writer.assert_not_awaited()


@pytest.mark.parametrize(
    "question",
    ["今天忙啥了呀", "最近忙什么呀", "刚才干啥了", "做什么呀", "进展怎么样呀", "咋了"],
)
def test_global_preference_catches_subjectless_colloquial_questions(question):
    assert guard_optional_questions(
        [question, "我在呢。"], policy(restriction("", "global"))
    )[0] == ["我在呢。"]


@pytest.mark.parametrize(
    "statement",
    ["今天忙什么都可以", "做什么都没关系", "有啥事都可以说", "想聊的时候我都在"],
)
def test_colloquial_guard_preserves_support_statements(statement):
    assert guard_optional_questions([statement], policy(restriction("", "global")))[
        0
    ] == [statement]


@pytest.mark.parametrize("message", ["我们聊聊面试吧", "我想聊聊面试", "继续聊面试吧"])
def test_user_explicitly_reopens_named_topic(message):
    command = parse_commands(message, statement_at=NOW)[0]
    assert (command.scope, command.action, command.topic) == ("topic", "resume", "面试")


@pytest.mark.parametrize("text", ["是忙啥了呀", "是干什么了呀", "今天是忙什么了呢"])
def test_optional_copula_in_colloquial_questions_is_not_a_bypass(text):
    assert guard_optional_questions(
        [text, "我在呢"], policy(restriction("", "global"))
    )[0] == ["我在呢"]


def test_unavailable_policy_does_not_allow_declarative_remention_of_unknown_topics():
    assert guard_optional_questions(
        ["还是聊聊工资吧"], policy(status="unavailable")
    ) == (["好，我听着。"], 1)
    assert guard_optional_questions(
        ["还是聊聊工资吧"], policy(status="capacity_exceeded"), fallback=None
    ) == ([], 1)


@pytest.mark.parametrize(
    "text", ["怎么啦", "今天咋这么累呀", "是忙啥累到了呀", "今天忙啥啦"]
)
def test_subjectless_care_questions_respect_global_stop(text):
    assert guard_optional_questions(
        [text, "我在呢"], policy(restriction("", "global"))
    )[0] == ["我在呢"]


def test_presentation_suffix_does_not_bypass_named_restriction():
    assert guard_optional_questions(
        ["是又加班了吗", "喝点水吧"], policy(restriction("加班安排"))
    )[0] == ["喝点水吧"]
    # A restriction on a specific disagreement is not a ban on the whole subject.
    assert guard_optional_questions(
        ["租房预算多少？"], policy(restriction("租房纠纷"))
    )[0] == ["租房预算多少？"]
    assert guard_optional_questions(
        ["今天的新面试怎么样？"], policy(restriction("昨天的面试"))
    )[0] == ["今天的新面试怎么样？"]


@pytest.mark.parametrize(
    "label,reply",
    [
        ("租房纠纷", "是之前租房的事闹的吗"),
        ("邻居噪音", "是上次邻居那件事吗"),
        ("论文修改", "后来论文的事怎么样了？"),
    ],
)
def test_ambiguous_generated_reference_does_not_reopen_restricted_topic(label, reply):
    assert guard_optional_questions([reply, "我在呢"], policy(restriction(label)))[
        0
    ] == ["我在呢"]


@pytest.mark.parametrize(
    "reply", ["租房预算多少？", "之前的咖啡好喝吗？", "之前租房有点累，我听见了"]
)
def test_partial_reference_does_not_merge_topics_or_remove_unrelated_reply(reply):
    assert guard_optional_questions([reply], policy(restriction("租房纠纷")))[0] == [
        reply
    ]


def test_crowded_prompt_is_bounded_without_weakening_output_guard():
    import json

    items = tuple(restriction("合成限制主题" + str(i) + "长" * 40) for i in range(128))
    p = policy(*items)
    payload = p.prompt_value().split("; topic_preferences=", 1)[1]
    parsed = json.loads(payload)
    assert len(payload.encode("utf-8")) <= 256
    assert parsed["omitted_restriction_count"] > 0
    assert guard_optional_questions([items[-1].topic + "怎么样？", "我在呢"], p)[0] == [
        "我在呢"
    ]


@pytest.mark.parametrize(
    "text",
    [
        "怎么啦🥺",
        "🥺怎么啦",
        "今天忙啥呀✨",
        "是忙啥了呀😮‍💨",
        "吃饭了吗😊",
        "要不要聊聊～",
    ],
)
def test_presentation_emoji_cannot_bypass_optional_question_guard(text):
    assert guard_optional_questions(
        [text, "我在呢"], policy(restriction("", "global"))
    )[0] == ["我在呢"]


@pytest.mark.parametrize("text", ["我在呢🥺", "✨先歇一会儿", "你想说的时候我都在😊"])
def test_emoji_support_statements_are_not_questions(text):
    assert guard_optional_questions([text], policy(restriction("", "global")))[0] == [
        text
    ]


@pytest.mark.parametrize(
    "zone,anchor,expiry",
    [
        ("Asia/Shanghai", "2026-10-31T15:59:00+00:00", "2026-10-31T16:00:00+00:00"),
        ("Asia/Shanghai", "2026-12-31T15:59:00+00:00", "2026-12-31T16:00:00+00:00"),
        ("America/New_York", "2026-11-01T04:30:00+00:00", "2026-11-02T05:00:00+00:00"),
        ("America/New_York", "2026-03-08T05:30:00+00:00", "2026-03-09T04:00:00+00:00"),
    ],
)
@pytest.mark.parametrize("text", ["今天别问面试了", "今晚别问面试了"])
def test_local_day_optout_handles_month_year_and_dst_boundaries(
    monkeypatch, zone, anchor, expiry, text
):
    from app.config import settings

    monkeypatch.setattr(settings, "schedule_timezone", zone)
    item = parse_commands(text, statement_at=datetime.fromisoformat(anchor))[0]
    assert item.expires_at == datetime.fromisoformat(expiry)


@pytest.mark.parametrize("topic", ["11月5日的面试", "上周的体检", "去年12月的年会"])
def test_explicit_deadline_binds_to_request_not_topic_embedded_date(topic):
    command = parse_commands(
        "在2026年11月2日前别问" + topic + "了",
        statement_at=datetime(2026, 11, 1, 2, tzinfo=timezone.utc),
    )[0]
    assert command.scope == "topic" and command.topic == topic
    assert command.expires_at == datetime(2026, 11, 1, 16, tzinfo=timezone.utc)
    assert not TopicRestriction(
        "x", topic, "topic", "pause", command.expires_at, "source", 1
    ).active(datetime(2026, 11, 3, tzinfo=timezone.utc))
