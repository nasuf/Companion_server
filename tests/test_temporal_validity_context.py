"""Bounded query failure, truncation and post-generation source protection."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
import asyncio
import pytest

from app.services.memory.retrieval.context_selector import ClassifiedMemory
from app.services.time_domain.query import resolve_temporal_query
from app.services.time_domain.validity_context import (
    load_validity_context,
    assertions_still_available,
)
from tests.test_memory_validity import source, proposal
from app.services.memory.recording.validity import resolve_validity

NOW = datetime(2026, 11, 1, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def no_replacements_in_source_reader_unit_fixtures(monkeypatch):
    # These fixtures model the source query, not the independent change ledger.
    # Real constraints, source mutation and replacement chains use PostgreSQL.
    from app.services.memory.interaction import change_history

    monkeypatch.setattr(change_history, "read_constraints", AsyncMock(return_value=[]))


def query(text="2026年8月3日当时我住在哪里？"):
    return resolve_temporal_query(text, anchor=NOW, anchor_source="persisted_message")


def memory(**changes):
    return ClassifiedMemory(
        **{
            "id": "m",
            "text": "用户住在旅馆",
            "source": "user",
            "relevance": "strong",
            "score": 1,
            **changes,
        }
    )


def row(index=0):
    text = "我从2026年8月1日到2026年8月6日住在旅馆。"
    value = resolve_validity(
        proposal(
            text,
            status="completed",
            valid_from_quote="2026年8月1日",
            valid_to_quote="2026年8月6日",
        ),
        [source(text)],
        (1,),
    )
    return {
        "memory_id": "m",
        "side": "user",
        "historical_memory_id": "m",
        "assertion_id": str(index),
        "version_id": 1,
        "assertion": value,
        "recorded_at": NOW,
        "retracted_at": None,
        "content": "用户住在旅馆",
        "occur_time": None,
        "statement_time": NOW,
        "chain_truncated": False,
    }


async def test_missing_scope_or_candidates_never_queries_database():
    db = SimpleNamespace(
        query_raw=AsyncMock(side_effect=AssertionError("unexpected read"))
    )
    for memories, uid, wid in [
        ([], "u", "w"),
        ([memory()], "", "w"),
        ([memory(id="")], "u", "w"),
    ]:
        _, records, info = await load_validity_context(
            memories, user_id=uid, workspace_id=wid, query=query(), database=db
        )
        assert not records and info["status"] == "not_requested"
    db.query_raw.assert_not_called()


@pytest.mark.parametrize(
    "error", [ConnectionError("private detail"), TimeoutError("private detail")]
)
async def test_dependency_failure_is_unknown_and_cannot_pass_post_generation_check(
    error,
):
    db = SimpleNamespace(query_raw=AsyncMock(side_effect=error))
    text, records, info = await load_validity_context(
        [memory()], user_id="u", workspace_id="w", query=query(), database=db
    )
    assert not text and not records and info["status"] == "unavailable"
    assert "private" not in str(info)
    assert not await assertions_still_available(
        {("user", "m"): [{"assertion_id": "1"}]},
        [memory()],
        user_id="u",
        workspace_id="w",
        query=query(),
        database=db,
    )


async def test_cancellation_propagates_instead_of_claiming_unknown():
    db = SimpleNamespace(query_raw=AsyncMock(side_effect=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        await load_validity_context(
            [memory()], user_id="u", workspace_id="w", query=query(), database=db
        )


async def test_candidate_and_prompt_budgets_are_explicit_and_scope_is_parameterized():
    db = SimpleNamespace(query_raw=AsyncMock(return_value=[row(i) for i in range(8)]))
    text, records, info = await load_validity_context(
        [memory()] * 21, user_id="owner'", workspace_id="w", query=query(), database=db
    )
    assert len(records[("user", "m")]) == 6 and info["truncated"]
    assert info["estimated_tokens"] <= 400 and 0 < info["shown_count"] < 6
    assert "独立核实" in text and not info["complete_history"]
    args = db.query_raw.await_args.args
    assert "owner'" not in args[0] and args[1] == "owner'"


async def test_native_datetime_rows_can_be_serialized_to_actual_model_payload():
    import json
    from app.services.time_domain.recall import recall_payload

    db = SimpleNamespace(query_raw=AsyncMock(return_value=[row()]))
    _, records, _ = await load_validity_context(
        [memory()], user_id="u", workspace_id="w", query=query(), database=db
    )
    payload, reason = recall_payload(
        "2026年8月3日当时我住在哪里？",
        [memory()],
        query().diagnostics(),
        validity_records=records,
    )
    assert reason == "eligible"
    assert (
        json.loads(json.dumps(payload))["memories"][0]["system_recorded_at"]
        == NOW.isoformat()
    )


async def test_unknown_and_outside_are_not_inside_and_invalid_metadata_not_used():
    valid = row()
    outside = row(1)
    outside["assertion"] = {
        **outside["assertion"],
        "valid_to": "2026-08-02T00:00:00+08:00",
    }
    unknown = row(2)
    unknown["assertion"] = {
        **unknown["assertion"],
        "valid_to": None,
        "to_precision": None,
    }
    corrupt = row(3)
    corrupt["assertion"] = {**corrupt["assertion"], "kind": "invalid"}
    db = SimpleNamespace(
        query_raw=AsyncMock(return_value=[valid, outside, unknown, corrupt])
    )
    _, records, _ = await load_validity_context(
        [memory()], user_id="u", workspace_id="w", query=query(), database=db
    )
    assert {r["membership"] for r in records[("user", "m")]} == {
        "inside",
        "outside",
        "unknown",
    }
    assert len(records[("user", "m")]) == 3
    assert await assertions_still_available(
        {}, [], user_id="u", workspace_id="w", query=query(), database=db
    )


@pytest.mark.parametrize(
    "question,expected",
    [
        ("2026年8月3日当时我住在哪里？", "user"),
        ("2026年8月3日当时你的职业是什么？", "ai"),
        ("你记得2026年8月3日当时我住在哪里？", "user"),
    ],
)
async def test_explicit_historical_subject_cannot_use_other_side_evidence(
    question, expected
):
    from app.services.time_domain.recall import recall_payload
    from dataclasses import replace

    current_query = resolve_temporal_query(
        question, anchor=NOW, anchor_source="persisted_message"
    )
    value = row()["assertion"]
    from app.services.memory.recording.validity import public_validity

    base = dict(
        content="在旅馆住过",
        source_excerpt="我从2026年8月1日到2026年8月6日住在旅馆。",
        validity=public_validity(value),
        membership="inside",
        recorded_at=NOW,
        retracted_at=None,
    )
    memories = [memory(), memory(source="ai")]
    records = {(side, "m"): [{**base, "source": side}] for side in ("user", "ai")}
    payload, reason = recall_payload(
        question, memories, current_query.diagnostics(), validity_records=records
    )
    assert reason == "eligible"
    assert [r["source"] for r in payload["memories"]] == [expected]
    other = "ai" if expected == "user" else "user"
    payload, reason = recall_payload(
        question,
        [memory(source=other)],
        current_query.diagnostics(),
        validity_records=records,
    )
    assert payload is None and reason == "no_as_of_evidence"


async def test_outside_interval_cannot_be_used_as_negative_presence_evidence():
    from app.services.time_domain.recall import recall_payload

    db = SimpleNamespace(query_raw=AsyncMock(return_value=[row()]))
    q = query("2026年8月9日当时我住在哪里？")
    _, records, _ = await load_validity_context(
        [memory()], user_id="u", workspace_id="w", query=q, database=db
    )
    assert records[("user", "m")][0]["membership"] == "outside"
    payload, reason = recall_payload(
        "2026年8月9日当时我住在哪里？",
        [memory()],
        q.diagnostics(),
        validity_records=records,
    )
    assert payload is None and reason == "no_as_of_evidence"


@pytest.mark.parametrize(
    "question",
    [
        "2026年8月3日当时你住在哪里？",
        "2026年8月3日当时你在天津吗？",
    ],
)
async def test_assistant_authored_location_does_not_verify_its_own_trip(question):
    from app.services.time_domain.recall import recall_payload

    q = query(question)
    data = row()
    data.update(side="ai", content="角色住在天津")
    _, records, _ = await load_validity_context(
        [memory(source="ai", text="角色住在天津")],
        user_id="u",
        workspace_id="w",
        query=q,
        database=SimpleNamespace(query_raw=AsyncMock(return_value=[data])),
    )
    assert records
    payload, reason = recall_payload(
        question,
        [memory(source="ai")],
        q.diagnostics(),
        validity_records=records,
    )
    assert payload is None and reason == "agent_location_not_verified"


@pytest.mark.parametrize(
    "change,reason",
    [
        ("ambiguous", "unknown_as_of_time"),
        ("unbound", "unbound_as_of_time"),
        ("long", "input_limit"),
        ("many", "input_limit"),
        ("invalid", "no_as_of_evidence"),
    ],
)
async def test_history_payload_refuses_unqualified_or_over_budget_sources(
    change, reason
):
    from app.services.time_domain.recall import recall_payload

    db = SimpleNamespace(query_raw=AsyncMock(return_value=[row()]))
    _, records, _ = await load_validity_context(
        [memory()], user_id="u", workspace_id="w", query=query(), database=db
    )
    diagnostic = query().diagnostics()
    if change == "ambiguous":
        diagnostic["status"] = "ambiguous"
    if change == "unbound":
        diagnostic["date_basis"]["kind"] = "unknown"
    if change == "long":
        records[("user", "m")][0]["content"] = "x" * 1001
    if change == "many":
        records[("user", "m")] *= 21
    if change == "invalid":
        records[("user", "m")][0]["validity"]["status"] = "invented"
    payload, actual_reason = recall_payload(
        "2026年8月3日当时我住在哪里？", [memory()], diagnostic, validity_records=records
    )
    assert payload is None and actual_reason == reason


async def test_recent_source_scope_limits_failure_and_cancellation():
    at = NOW.isoformat()
    message = {
        "id": "origin",
        "role": "user",
        "content": "我当时住在杭州",
        "createdAt": at,
    }
    db = SimpleNamespace(query_raw=AsyncMock(return_value=[]))
    args = dict(user_id="u", workspace_id="w", query=query(), database=db)
    for invalid in (
        {**message, "role": "assistant"},
        {**message, "id": None},
        {**message, "content": "x" * 801},
        {**message, "createdAt": None},
    ):
        assert not (await load_validity_context([], recent_messages=[invalid], **args))[
            1
        ]
    db.query_raw.assert_not_called()
    many = [{**message, "id": str(i)} for i in range(20)]
    _, _, info = await load_validity_context([], recent_messages=many, **args)
    import json

    assert info["truncated"] and len(json.loads(db.query_raw.await_args.args[3])) == 12
    long = [{**message, "id": str(i), "content": "字" * 800} for i in range(10)]
    _, _, info = await load_validity_context([], recent_messages=long, **args)
    assert info["truncated"] and len(json.loads(db.query_raw.await_args.args[3])) == 5
    db.query_raw.side_effect = ConnectionError("private detail")
    _, records, info = await load_validity_context(
        [], recent_messages=[message], **args
    )
    assert (
        not records and info["status"] == "unavailable" and "private" not in str(info)
    )
    db.query_raw.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await load_validity_context([], recent_messages=[message], **args)


def test_recent_statement_cannot_answer_for_agent_or_bypass_scope_contract():
    from app.services.time_domain.recall import recall_payload

    records = {
        ("dialogue", "s"): [
            {
                "source": "user",
                "content": "我住杭州",
                "evidence_type": "recent_statement",
                "statement_time": NOW,
            }
        ]
    }
    q = query()
    payload, reason = recall_payload(
        "2026年8月3日当时我住在哪里？", [], q.diagnostics(), validity_records=records
    )
    assert reason == "eligible" and payload["memories"][0]["event_date"] is None
    payload, reason = recall_payload(
        "2026年8月3日当时你住在哪里？", [], q.diagnostics(), validity_records=records
    )
    assert payload is None and reason == "agent_location_not_verified"
    records[("dialogue", "s")][0]["source"] = "ai"
    assert (
        recall_payload(
            "2026年8月3日当时我住在哪里？",
            [],
            q.diagnostics(),
            validity_records=records,
        )[1]
        == "invalid_source"
    )


@pytest.mark.parametrize("hedge", ["", "应该", "大概", "估计", "可能"])
async def test_historical_answer_requires_calendar_support_even_if_model_accepts(
    monkeypatch, hedge
):
    from app.services.time_domain import recall

    records = {
        ("dialogue", "s"): [
            {
                "source": "user",
                "content": "我从2026年8月1日起在松林旅馆住过，没说哪天离开。",
                "evidence_type": "recent_statement",
                "statement_time": NOW,
            }
        ]
    }
    model = AsyncMock(
        side_effect=[
            {"memory_indices": [0], "reply": f"你8月15日{hedge}还在松林旅馆"},
            {"grounded": True, "time_support": [{"index": 0, "quote": "2026年8月1日"}]},
        ]
    )
    monkeypatch.setattr(recall, "invoke_json", model)
    monkeypatch.setattr(recall, "get_grounding_model", lambda: None)
    monkeypatch.setattr(recall, "get_chat_model", lambda: "configured-main")
    monkeypatch.setattr(
        recall, "get_prompt_text", AsyncMock(return_value="synthetic rules")
    )
    monkeypatch.setattr(recall, "build_reply_prefix", AsyncMock(return_value=""))
    question = "2026年8月15日当时我住在哪里？"
    diagnostics = {"temporal_context": {"query": query(question).diagnostics()}}
    result = await recall.answer_temporal_recall(
        question, [], diagnostics, max_chars=100, validity_records=records
    )
    assert result == recall._FALLBACK and model.await_count == 2
    assert model.await_args_list[0].args[0] == "configured-main"
    assert (
        diagnostics["temporal_context"]["recall"]["reason"]
        == "unsupported_historical_time"
    )


@pytest.mark.parametrize("hedge", ["", "应该"])
@pytest.mark.parametrize("protocol", ["legacy", "labels"])
@pytest.mark.parametrize(
    "provider", ["ark", "dashscope", "deepseek", "ollama", "claude"]
)
async def test_supported_answer_not_rejected_by_style_word(
    monkeypatch, hedge, provider, protocol
):
    from app.services.time_domain import recall

    monkeypatch.setattr(recall, "provider_name", lambda model: provider)

    text = "我从2026年8月1日到8月16日住在松林旅馆。"
    records = {
        ("dialogue", "s"): [
            {
                "source": "user",
                "content": text,
                "evidence_type": "recent_statement",
                "statement_time": NOW,
            }
        ]
    }
    reply = f"你当时{hedge}住在松林旅馆"
    verdict = (
        {"grounded": True, "calendar_evidence_labels": ["T0"]}
        if protocol == "labels"
        else {
            "grounded": True,
            "time_support": [{"index": 0, "quote": "2026年8月1日到8月16日"}],
        }
    )
    model = AsyncMock(
        side_effect=[
            {"memory_indices": [0], "reply": reply},
            verdict,
        ]
    )
    monkeypatch.setattr(recall, "invoke_json", model)
    monkeypatch.setattr(recall, "get_grounding_model", lambda: None)
    monkeypatch.setattr(recall, "get_chat_model", lambda: "configured-main")
    monkeypatch.setattr(
        recall, "get_prompt_text", AsyncMock(return_value="synthetic rules")
    )
    monkeypatch.setattr(recall, "build_reply_prefix", AsyncMock(return_value=""))
    question = "2026年8月15日当时我住在哪里？"
    diagnostics = {"temporal_context": {"query": query(question).diagnostics()}}
    assert (
        await recall.answer_temporal_recall(
            question, [], diagnostics, max_chars=100, validity_records=records
        )
        == reply
    )

    options = model.await_args_list[0].kwargs
    import json

    checked = json.loads(model.await_args_list[1].args[1][1].content)
    assert checked["input"]["calendar_evidence"] == [
        {"label": "T0", "record_index": 0, "date_index": 0, "relation": "contains"}
    ]
    if provider in {"ark", "dashscope", "deepseek"}:
        assert options["response_format"] == {"type": "json_object"}
    else:
        assert "response_format" not in options


def support_payload(text, question="2026年8月3日当时我住在哪里？"):
    from app.services.time_domain.recall import recall_payload

    records = {
        ("dialogue", "s"): [
            {
                "source": "user",
                "content": text,
                "evidence_type": "recent_statement",
                "statement_time": NOW,
            }
        ]
    }
    return recall_payload(
        question, [], query(question).diagnostics(), validity_records=records
    )[0]


@pytest.mark.parametrize(
    "quote",
    ["2026 年 8 月 1 日到 8 月 6 日", "2026年8月1日到8月6日", "2026\t年8月1日到8月6日"],
)
def test_exact_calendar_quote_allows_only_typographic_whitespace(quote):
    from app.services.time_domain.source_evidence import validate_as_of_support

    payload = support_payload("我从2026年8月1日到8月6日住旅馆。")
    assert validate_as_of_support(
        {"time_support": [{"index": 0, "quote": quote}]}, payload, [{"index": 0}]
    )


@pytest.mark.parametrize(
    "support",
    [
        None,
        {},
        [None],
        [{"index": True, "quote": "2026年8月1日"}],
        [{"index": 1, "quote": "2026年8月1日"}],
        [{"index": 0, "quote": "2026年8月1日到8月6日"}],
        [{"index": 0, "quote": ""}],
        [{"index": 0, "quote": "x" * 151}],
        [{"index": 0, "quote": "2026年8月1日", "override": True}],
        [{"index": 0, "quote": "2026年8月1日"}] * 2,
    ],
)
def test_invalid_or_fabricated_calendar_support_is_rejected(support):
    from app.services.time_domain.source_evidence import validate_as_of_support

    payload = support_payload(
        "我从2026年8月1日起住旅馆。", "2026年8月1日当时我住哪里？"
    )
    with pytest.raises(ValueError):
        validate_as_of_support({"time_support": support}, payload, [{"index": 0}])


def test_noncovering_and_empty_support_cannot_be_used_to_answer():
    from app.services.time_domain.source_evidence import validate_as_of_support

    payload = support_payload("我从2026年8月1日起住旅馆。")
    assert not validate_as_of_support({"time_support": []}, payload, [{"index": 0}])
    assert not validate_as_of_support(
        {"time_support": [{"index": 0, "quote": "2026年8月1日"}]},
        payload,
        [{"index": 0}],
    )
    payload = support_payload("我从8月1日到8月6日住旅馆。")
    assert not validate_as_of_support(
        {"time_support": [{"index": 0, "quote": "8月1日到8月6日"}]},
        payload,
        [{"index": 0}],
    )
    payload = support_payload("我2026年8月1日入住。2026年8月1日又换了一家。")
    with pytest.raises(ValueError):
        validate_as_of_support(
            {"time_support": [{"index": 0, "quote": "2026年8月1日"}]},
            payload,
            [{"index": 0}],
        )


def test_materialized_inside_can_support_without_model_generated_quote():
    from app.services.time_domain.source_evidence import validate_as_of_support

    payload = support_payload("我从2026年8月1日到8月6日住旅馆。")
    payload["memories"][0]["validity_membership"] = "inside"
    assert validate_as_of_support(
        {"time_support": [{"index": 0, "quote": None}]}, payload, [{"index": 0}]
    )
    payload["requested_validity_range"].pop("end_boundary")
    with pytest.raises(ValueError):
        validate_as_of_support(
            {"time_support": [{"index": 0, "quote": None}]}, payload, [{"index": 0}]
        )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2026年8月1日到8月6日", "resolved"),
        ("2026年8月1日至8月6日", "resolved"),
        ("2026年12月31日到1月2日", "unresolved"),
        ("2024年2月28日到2月29日", "resolved"),
        ("2025年2月28日到2月29日", "unresolved"),
        ("2026年9月1日说我8月1日到8月6日住这里", "ambiguous"),
        ("2026年8月1日和8月6日", "ambiguous"),
        ("8月1日到8月6日", "ambiguous"),
    ],
)
def test_adjacent_year_inheritance_does_not_borrow_statement_year_or_rollover(
    text, expected
):
    value = query(text)
    assert value.status == expected
    if expected == "resolved":
        assert value.diagnostics()["ranges"][0]["end_boundary"] in {
            "inclusive_second",
            "exclusive",
            "inclusive_microsecond",
        }


async def test_valid_calendar_cannot_override_semantic_rejection_of_another_claim(
    monkeypatch,
):
    from app.services.time_domain import recall

    question = "2026年8月3日当时我做了什么？"
    records = {
        ("dialogue", "s0"): [
            {
                "source": "user",
                "content": "我2026年8月3日上完绘画课。",
                "evidence_type": "recent_statement",
                "statement_time": NOW,
            }
        ],
        ("dialogue", "s1"): [
            {
                "source": "user",
                "content": "我准备下个月申请结业证。",
                "evidence_type": "recent_statement",
                "statement_time": NOW,
            }
        ],
    }
    model = AsyncMock(
        side_effect=[
            {"memory_indices": [0, 1], "reply": "你上完绘画课，还拿到了结业证"},
            {"grounded": False, "time_support": [{"index": 0, "date_index": 0}]},
        ]
    )
    monkeypatch.setattr(recall, "invoke_json", model)
    monkeypatch.setattr(recall, "get_chat_model", lambda: "main")
    monkeypatch.setattr(
        recall, "get_prompt_text", AsyncMock(return_value="synthetic rules")
    )
    monkeypatch.setattr(recall, "build_reply_prefix", AsyncMock(return_value=""))
    diagnostics = {"temporal_context": {"query": query(question).diagnostics()}}
    result = await recall.answer_temporal_recall(
        question, [], diagnostics, max_chars=100, validity_records=records
    )
    assert result == recall._FALLBACK and model.await_count == 2
    assert diagnostics["temporal_context"]["recall"]["reason"] == "answer_not_grounded"


def test_single_literal_date_can_be_selected_without_generated_quote():
    from app.services.time_domain.source_evidence import validate_as_of_support

    payload = support_payload("我2026年8月3日参加了志愿活动。")
    assert validate_as_of_support(
        {"time_support": [{"index": 0, "quote": None}]}, payload, [{"index": 0}]
    )
    payload = support_payload("我从2026年8月1日起住旅馆。")
    assert not validate_as_of_support(
        {"time_support": [{"index": 0, "quote": None}]}, payload, [{"index": 0}]
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("我2026年8月3日当时的项目是甲，更正为乙", ["2026年8月3日"]),
        ("我从2026年8月1日到8月6日住在旅馆", ["2026年8月1日到8月6日"]),
        ("我8月1日住旅馆", []),
        ("我2026年8月1日说，我8月3日住旅馆", ["2026年8月1日"]),
    ],
)
def test_literal_source_date_spans_are_calendar_syntax_not_truth(text, expected):
    from app.services.time_domain.source_evidence import source_date_spans

    result = source_date_spans(text, anchor=NOW, timezone_name="Asia/Shanghai")
    assert [x["quote"] for x in result] == expected


def test_question_have_or_not_is_not_a_negative_assertion():
    assert query("2026年8月3日当时我有没有参加志愿活动").status == "resolved"
    assert query("2026年8月3日我没有参加志愿活动").status == "ambiguous"
    assert query("更正，我2026年8月3日当时住在甲城").status == "ambiguous"


@pytest.mark.parametrize(
    "question",
    [
        "2026年2月15日当时我是否有每天下午2点到5点复习英语阅读的计划",
        "2026年2月15日当时我有没有计划3月12日参加考试",
    ],
)
def test_explicit_asof_reference_is_separate_from_fact_predicate_dates(question):
    parsed = query(question)
    assert parsed.status == "resolved" and parsed.operation.type == "as_of"
    assert parsed.expressions[0].start.date().isoformat() == "2026-02-15"
    assert parsed.diagnostics()["binding"]["decision"] == "explicit_as_of_reference"
    assert not parsed.allows_event_range()


@pytest.mark.parametrize(
    "question",
    [
        "不是2026年2月15日当时，我想问更早的时候",
        "“2026年2月15日当时”只是他说的日期，我不知道是否真实",
        "2026年2月15日当时和2026年3月12日当时我住哪里",
    ],
)
def test_conflicting_quoted_or_negated_asof_reference_remains_ambiguous(question):
    assert query(question).status != "resolved"


def date_reference_records(prior="我2026年8月3日参加志愿活动", gap=120, side="user"):
    from datetime import timedelta

    return [
        dict(
            index=0,
            source=side,
            source_excerpt=prior,
            statement_at=(NOW - timedelta(seconds=gap)).isoformat(),
            evidence_type="recent_statement",
        ),
        dict(
            index=1,
            source="user",
            source_excerpt="那天我还帮助路人找回手机",
            statement_at=NOW.isoformat(),
            evidence_type="recent_statement",
        ),
    ]


def test_recent_day_reference_keeps_original_date_source_identity():
    from app.services.time_domain.recall import (
        attach_source_date_spans,
        validate_as_of_support,
    )

    records = date_reference_records()
    attach_source_date_spans(records, timezone_name="Asia/Shanghai")
    assert records[1]["literal_date_spans"][0]["origin_index"] == 0
    payload = support_payload("我2026年8月3日参加志愿活动")
    payload["memories"] = records
    assert validate_as_of_support(
        {"time_support": [{"index": 1, "date_index": 0}]}, payload, [{"index": 1}]
    )


@pytest.mark.parametrize(
    "options",
    [
        {"gap": 1801},
        {"gap": 0},
        {"side": "ai"},
        {"prior": "我2026年8月3日和2026年8月5日参加志愿活动"},
        {"prior": "如果2026年8月3日我参加志愿活动"},
        {"prior": "我从2026年8月1日到8月6日参加志愿活动"},
        {"prior": "我8月3日参加志愿活动"},
    ],
)
def test_recent_day_reference_cannot_invent_ambiguous_distant_or_foreign_date(options):
    from app.services.time_domain.source_evidence import attach_source_date_spans

    records = date_reference_records(**options)
    attach_source_date_spans(records, timezone_name="Asia/Shanghai")
    assert records[1]["literal_date_spans"] == []


@pytest.mark.parametrize("index", [True, -1, 1, None, "0", 1.5])
def test_date_support_index_is_server_bounded(index):
    from app.services.time_domain.source_evidence import validate_as_of_support

    payload = support_payload("我2026年8月3日参加志愿活动")
    with pytest.raises(ValueError):
        validate_as_of_support(
            {"time_support": [{"index": 0, "date_index": index}]},
            payload,
            [{"index": 0}],
        )


@pytest.mark.parametrize(
    "text,accepted",
    [
        ("我从2026年8月1日到8月2日住旅馆", True),
        ("我从2026年8月1日到8月6日住旅馆", False),
        ("我从2026年8月1日起住旅馆", False),
    ],
)
def test_closed_episode_endpoint_is_not_an_infinite_absence_proof(text, accepted):
    from app.services.time_domain.source_evidence import validate_as_of_support

    payload = support_payload(text)
    assert (
        validate_as_of_support(
            {
                "time_support": [
                    {"index": 0, "date_index": 0, "relation": "ended_before"}
                ]
            },
            payload,
            [{"index": 0}],
        )
        is accepted
    )


def test_inside_materialization_cannot_skip_time_relation_validation():
    from app.services.time_domain.source_evidence import validate_as_of_support

    payload = support_payload("我2026年8月3日参加志愿活动")
    payload["memories"][0]["validity_membership"] = "inside"
    with pytest.raises(ValueError, match="invalid_time_support_relation"):
        validate_as_of_support(
            {
                "time_support": [
                    {"index": 0, "date_index": None, "relation": "invented"}
                ]
            },
            payload,
            [{"index": 0}],
        )
    assert not validate_as_of_support(
        {
            "time_support": [
                {"index": 0, "date_index": None, "relation": "ended_before"}
            ]
        },
        payload,
        [{"index": 0}],
    )


@pytest.mark.parametrize(
    "text", ["我晚上8点遛狗", "我每天下午2点到5点学习", "我早上7点起床"]
)
def test_clock_without_calendar_does_not_borrow_mention_date(text):
    from app.services.time_domain.source_evidence import source_date_spans

    assert source_date_spans(text, anchor=NOW, timezone_name="Asia/Shanghai") == []


def test_cancellation_deictic_keeps_date_without_becoming_occurred_fact():
    from app.services.time_domain.source_evidence import attach_source_date_spans

    records = date_reference_records("我2026年8月3日计划参加志愿活动")
    records[1]["source_excerpt"] = "我取消了这次志愿活动"
    attach_source_date_spans(records, timezone_name="Asia/Shanghai")
    assert records[1]["literal_date_spans"][0]["origin_index"] == 0
    assert records[1]["source_excerpt"] == "我取消了这次志愿活动"


def test_equivalent_date_mentions_do_not_make_deictic_ambiguous():
    from app.services.time_domain.source_evidence import attach_source_date_spans

    records = date_reference_records("我2026年8月3日说我2026年8月3日参加志愿活动")
    attach_source_date_spans(records, timezone_name="Asia/Shanghai")
    assert records[1]["literal_date_spans"][0]["origin_index"] == 0


def test_calendar_support_does_not_require_dates_for_unused_background():
    from copy import deepcopy
    from app.services.time_domain.source_evidence import validate_as_of_support

    payload = support_payload("我从2026年8月1日到8月6日在桂林培训。")
    background = deepcopy(payload["memories"][0])
    background.update(
        index=1, content="我现在在北海。", source_excerpt="我现在在北海。"
    )
    background["literal_date_spans"] = []
    payload["memories"].append(background)
    matches = [{"index": 0}, {"index": 1}]
    assert validate_as_of_support(
        {"time_support": [{"index": 0, "date_index": 0}]}, payload, matches
    )
    # The semantic check remains mandatory; this helper proves calendar syntax,
    # not that either record's content entails a reply.
    assert not validate_as_of_support({"time_support": []}, payload, matches)
    with pytest.raises(ValueError):
        validate_as_of_support(
            {"time_support": [{"index": 2, "date_index": 0}]}, payload, matches
        )
    with pytest.raises(ValueError):
        validate_as_of_support(
            {"time_support": [{"index": 1, "date_index": 0}]}, payload, matches
        )
    payload["requested_validity_range"]["start"] = "2026-09-03T00:00:00+08:00"
    payload["requested_validity_range"]["end"] = "2026-09-03T23:59:59.999999+08:00"
    assert not validate_as_of_support(
        {"time_support": [{"index": 0, "date_index": 0}]}, payload, matches
    )


@pytest.mark.parametrize(
    "text",
    [
        "截至2026年6月9日当时，我是不是还在照顾猫？",
        "我当时住哪里，指2026年7月13日？",
        "你记得我2026年7月13日当时住哪里吗？",
        "2026年7月13日那时候我在做什么？",
        "当时我在做什么，日期是2026年7月13日？",
    ],
)
def test_explicit_as_of_reference_not_limited_to_sentence_start(text):
    value = query(text)
    assert value.status == "resolved"
    assert value.operation.type == "as_of"
    span = value.binding["spans"][0]
    assert text[span["start"] : span["end"]].startswith("2026")
    assert value.binding["decision"] == "explicit_as_of_reference"


@pytest.mark.parametrize(
    "text",
    [
        "不是2026年7月13日，当时我住哪里？",
        "假如2026年7月13日当时我住这里，会怎样？",
        "你2026年7月13日当时告诉过我住在哪里？",
        "2026年7月13日当时的日记说了什么？",
        "你说的‘2026年7月13日当时’指什么？",
        "2026年7月13日，我当时7月14日也在那吗？",
        "2026年7月13日当时或者2026年7月15日当时我在哪？",
        "前阵子2026年7月13日当时，我住哪里？",
    ],
)
def test_as_of_reference_does_not_override_ambiguous_or_statement_dates(text):
    assert (query(text).binding or {}).get("decision") != "explicit_as_of_reference"


@pytest.mark.parametrize(
    "question",
    [
        "我当时在哪个车间，问的是2025年11月11日？",
        "当时我负责什么，问的是 2024年2月8日。",
        "那时候我有没有上课？问的是2026年4月3日",
    ],
)
def test_terminal_asof_day_clarification_is_not_a_statement_clock(question):
    result = query(question)
    assert result.status == "resolved"
    assert result.operation.type == "as_of"
    assert result.binding["decision"] == "explicit_as_of_reference"
    assert result.date_basis == "event_candidate"


@pytest.mark.parametrize(
    "question",
    [
        "我当时说过在哪个车间，问的是2025年11月11日？",
        "我当时聊的工作是什么，问的是2025年11月11日？",
        "当时我朗读了“问的是2025年11月11日”，那是事实吗？",
        "如果我当时在工厂，问的是2025年11月11日？",
        "我当时在哪，问的是2025年11月11日还是2025年11月12日？",
        "我当时在哪，问的是11月11日？",
        "我当时问的是什么，2025年11月11日？",
    ],
)
def test_terminal_reference_does_not_erase_other_ambiguity(question):
    result = query(question)
    assert result.status != "resolved" or result.date_basis != "event_candidate"


@pytest.mark.parametrize(
    "reply",
    [
        "你那天没在后厨帮忙",
        "你那天不在后厨",
        "你当时未在那工作",
        "你当时并不在那",
        "你没有在那工作",
    ],
)
def test_absence_predicate_cannot_come_from_a_closed_episode(reply):
    from app.services.time_domain.recall import has_unsupported_denial

    evidence = [
        {"content": "我从2025年3月1日至3月4日在后厨帮忙"},
        {"content": "2025年3月5日店里来了新厨师"},
    ]
    assert has_unsupported_denial(reply, evidence, "")
    assert not has_unsupported_denial(reply, [{"content": "我那天明确不在后厨"}], "")


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "2023年1月3日选了布，1月7日裁好，1月9日缝成袋子。",
            ["2023-01-03", "2023-01-07", "2023-01-09"],
        ),
        ("2023年4月30日交稿，5月1日收到回复。", ["2023-04-30", "2023-05-01"]),
        ("2023年12月31日交稿，1月1日收到回复。", ["2023-12-31"]),
        ("2023年4月30日交稿。5月1日收到回复。", ["2023-04-30"]),
        ("2023年4月30日交稿，次年5月1日收到回复。", ["2023-04-30"]),
        (
            "2023年4月30日交稿，2024年4月30日补稿，5月1日收到回复。",
            ["2023-04-30", "2024-04-30"],
        ),
        ("2023年4月30日交稿，别人说‘5月1日收到回复’。", ["2023-04-30"]),
        ("5月1日收到回复。", []),
    ],
)
def test_source_ordered_date_list_shares_only_explicit_same_sentence_year(
    text, expected
):
    from app.services.time_domain.source_evidence import source_date_spans

    spans = source_date_spans(
        text, anchor="2026-11-01T00:00:00Z", timezone_name="Asia/Shanghai"
    )
    assert [s["start"][:10] for s in spans] == expected
    for span in spans:
        assert span["quote"] in text
        if "year_reference" in span:
            assert span["year_reference"] in text and "年" not in span["quote"]


@pytest.mark.parametrize(
    "question",
    [
        "2023年9月10日当时你也在海边吗？",
        "2023年9月10日当时你还住在海边吗？",
        "2023年9月10日当时你确定去海边了吗？",
        "2023年9月10日当时你真的去海边了吗？",
    ],
)
def test_role_location_guard_includes_safe_adverbs(question):
    from app.services.time_domain.query import resolve_temporal_query
    from app.services.time_domain.recall import recall_payload
    from app.services.memory.retrieval.context_selector import ClassifiedMemory

    query = resolve_temporal_query(
        question,
        anchor=datetime(2026, 11, 1, tzinfo=timezone.utc),
        anchor_source="persisted_message",
    )
    memory = ClassifiedMemory(
        id="user-fact",
        text="我2023年9月10日在海边",
        source="user",
        relevance="strong",
        score=1,
    )
    payload, reason = recall_payload(question, [memory], query.diagnostics())
    assert payload is None and reason == "agent_location_not_verified"


def test_calendar_labels_keep_undated_attributes_separate_from_dates():
    from app.services.time_domain.source_evidence import (
        calendar_evidence_options,
        validate_calendar_labels,
    )

    payload = support_payload("我2026年8月3日领到了一个帆布袋。")
    payload["memories"].append(
        {
            "index": 1,
            "source": "user",
            "source_excerpt": "袋子上印着白鲸",
            "literal_date_spans": [],
            "validity_membership": "not_materialized",
        }
    )
    assert calendar_evidence_options(payload) == [
        {"label": "T0", "record_index": 0, "date_index": 0, "relation": "contains"}
    ]
    assert validate_calendar_labels(
        {"calendar_evidence_labels": ["T0"]}, payload, [{"index": 0}, {"index": 1}]
    )
    with pytest.raises(ValueError, match="unknown_calendar_label"):
        validate_calendar_labels(
            {"calendar_evidence_labels": ["T1"]}, payload, [{"index": 0}, {"index": 1}]
        )


@pytest.mark.parametrize(
    "labels",
    [
        None,
        {},
        "T0",
        [0],
        [True],
        [None],
        [["T0"]],
        ["T0", "T0"],
        ["T01"],
        ["0"],
        ["T0"] * 21,
    ],
)
def test_model_cannot_manufacture_calendar_labels(labels):
    from app.services.time_domain.source_evidence import validate_calendar_labels

    payload = support_payload("我2026年8月3日领了钥匙")
    with pytest.raises(ValueError):
        validate_calendar_labels(
            {"calendar_evidence_labels": labels}, payload, [{"index": 0}]
        )


def test_calendar_labels_do_not_trust_relation_or_model_supplied_catalog():
    from app.services.time_domain.source_evidence import (
        calendar_evidence_options,
        validate_calendar_labels,
    )

    payload = support_payload("我2026年8月1日领了钥匙")
    payload["memories"][0]["literal_date_spans"][0]["relation_to_query"] = "contains"
    payload["calendar_evidence"] = [{"label": "T0", "record_index": 0}]
    assert calendar_evidence_options(payload) == []
    with pytest.raises(ValueError, match="unknown_calendar_label"):
        validate_calendar_labels(
            {"calendar_evidence_labels": ["T0"]}, payload, [{"index": 0}]
        )
    assert not validate_calendar_labels(
        {"calendar_evidence_labels": []}, payload, [{"index": 0}]
    )


def test_calendar_label_requires_cited_scoped_source():
    from app.services.time_domain.source_evidence import validate_calendar_labels

    payload = support_payload("我2026年8月3日领了钥匙")
    with pytest.raises(ValueError, match="invalid_time_support"):
        validate_calendar_labels(
            {"calendar_evidence_labels": ["T0"]}, payload, [{"index": 1}]
        )


def test_one_source_can_supply_two_distinct_dated_claims():
    from app.services.time_domain.source_evidence import (
        calendar_evidence_options,
        validate_calendar_labels,
    )

    payload = support_payload("我2026年8月3日领了钥匙，同日也交了材料")
    spans = payload["memories"][0]["literal_date_spans"]
    spans.append(dict(spans[0]))
    assert [x["label"] for x in calendar_evidence_options(payload)] == ["T0", "T1"]
    assert validate_calendar_labels(
        {"calendar_evidence_labels": ["T0", "T1"]}, payload, [{"index": 0}]
    )


@pytest.mark.parametrize(
    "text,relation",
    [
        ("我2026年8月1日到8月2日在后厨帮忙", "ended_before"),
        ("我2026年8月1日到8月6日在后厨帮忙", "contains"),
    ],
)
def test_calendar_labels_preserve_exclusive_end_and_episode_semantics(text, relation):
    from app.services.time_domain.source_evidence import (
        calendar_evidence_options,
        validate_calendar_labels,
    )

    payload = support_payload(text)
    assert calendar_evidence_options(payload)[0]["relation"] == relation
    assert validate_calendar_labels(
        {"calendar_evidence_labels": ["T0"]}, payload, [{"index": 0}]
    )


@pytest.mark.parametrize(
    "question",
    [
        "2022年10月3日当时我在哪个录音棚？",
        "2022年4月19日当时我讲的故事关于什么？",
    ],
)
def test_leading_explicit_reference_clock_survives_mention_like_vocabulary(question):
    result = query(question)
    assert result.status == "resolved"
    assert result.operation.type == "as_of"
    assert result.binding["decision"] == "explicit_as_of_reference"
    assert result.date_basis == "event_candidate"


async def test_explicit_question_clock_does_not_date_the_narrated_event(monkeypatch):
    from app.services.time_domain import recall
    from app.services.time_domain.source_evidence import calendar_evidence_options

    text = "2026年8月3日我提到以前去海边旅行。"
    question = "2026年8月3日当时我在海边旅行吗？"
    payload = support_payload(text, question)
    # Calendar labels certify date syntax only. They cannot override a failed
    # semantic check which distinguishes telling a story from its event.
    assert len(calendar_evidence_options(payload)) == 1
    records = {
        ("dialogue", "s"): [
            {
                "source": "user",
                "content": text,
                "evidence_type": "recent_statement",
                "statement_time": NOW,
            }
        ]
    }
    model = AsyncMock(
        side_effect=[
            {"memory_indices": [0], "reply": "你当时在海边旅行"},
            {"grounded": False, "calendar_evidence_labels": ["T0"]},
        ]
    )
    monkeypatch.setattr(recall, "invoke_json", model)
    monkeypatch.setattr(recall, "get_grounding_model", lambda: None)
    monkeypatch.setattr(recall, "get_chat_model", lambda: "configured-main")
    monkeypatch.setattr(
        recall, "get_prompt_text", AsyncMock(return_value="synthetic rules")
    )
    monkeypatch.setattr(recall, "build_reply_prefix", AsyncMock(return_value=""))
    diagnostics = {"temporal_context": {"query": query(question).diagnostics()}}
    assert (
        await recall.answer_temporal_recall(
            question, [], diagnostics, max_chars=100, validity_records=records
        )
        == recall._FALLBACK
    )
    assert model.await_count == 2


def test_calendar_labels_reject_empty_exclusive_requested_period():
    from app.services.time_domain.source_evidence import calendar_evidence_options

    payload = support_payload("我2026年8月3日领到帆布袋。")
    payload["requested_validity_range"]["end"] = payload["requested_validity_range"][
        "start"
    ]
    payload["requested_validity_range"]["end_boundary"] = "exclusive"
    with pytest.raises(ValueError, match="invalid_requested_period"):
        calendar_evidence_options(payload)


@pytest.mark.parametrize("connector", ["到", "持续到", "一直到", "直到", "直至", "至", "—", "~", "～"])
def test_explicit_duration_connectors_use_shared_range_binding(connector):
    from app.services.time_domain.source_evidence import source_date_spans, calendar_evidence_options
    text = f"那次排练从2022年10月1日{connector}10月5日，我全程在三号厅。"
    question = "2022年10月3日当时我在哪个厅？"
    spans = source_date_spans(text, anchor=NOW, timezone_name="Asia/Shanghai")
    assert len(spans) == 1 and spans[0]["precision"] == "range"
    assert spans[0]["quote"] == f"2022年10月1日{connector}10月5日"
    assert len(calendar_evidence_options(support_payload(text, question))) == 1


@pytest.mark.parametrize("connector", ["去过。", "有排练，而", "结束，另一次从", "可能持续到", "没有持续到"])
def test_range_binding_does_not_bridge_prose_or_uncertainty(connector):
    from app.services.time_domain.source_evidence import source_date_spans
    spans = source_date_spans(
        f"2022年10月1日{connector}2022年10月5日", anchor=NOW, timezone_name="Asia/Shanghai"
    )
    assert not any(s["precision"] == "range" for s in spans)
