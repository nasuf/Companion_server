"""Temporal meaning must be grounded separately from memory text and event dates."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import copy
import pytest

from app.services.memory.evidence import EvidenceSource, content_version
from app.services.memory.recording.temporal_sources import resolve_fact_clock
from app.services.memory.recording.validity import (
    resolve_validity,
    public_validity,
    valid_at,
)

NOW = datetime(2026, 11, 1, 2, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2026年8月3日当时我住在哪里？", "resolved"),
        ("2026-08-03 的当时我住在哪里？", "resolved"),
        ("当时我住在哪里？", "ambiguous"),
        ("2026年8月3日聊过，当时我住在哪里？", "ambiguous"),
        ("2026年8月3日当时我最近住在哪里？", "ambiguous"),
        ("2026年2月30日当时我住在哪里？", "unresolved"),
    ],
)
def test_as_of_deictic_requires_adjacent_explicit_date(text, expected):
    from app.services.time_domain.query import resolve_temporal_query

    query = resolve_temporal_query(text, anchor=NOW, anchor_source="persisted_message")
    assert query.status == expected
    assert query.retrieval_range() is None


def source(text, *, at=NOW, role="user"):
    return EvidenceSource(
        "message",
        "synthetic-message",
        relation="extracted_from",
        expected_role=role,
        expected_version=content_version(text),
        expected_statement_time=at,
        input_text=text,
    )


def test_structured_card_title_cannot_become_a_temporal_claim():
    quote = "我从2026年8月1日到2026年8月6日住在天津"
    rendered = "分享音乐\n用户在这条消息中分享了歌曲《" + quote + "》"
    origin = replace(source(rendered), expected_version=content_version("分享音乐"),
                     expected_card_version="synthetic-card-version")
    fact = {"source_indices": [1], "temporal_assertion": {
        "kind": "state", "status": "completed", "source_index": 1,
        "quote": quote, "valid_from_quote": "2026年8月1日",
        "valid_to_quote": "2026年8月6日",
    }}
    clock = resolve_fact_clock(fact, [origin])
    assert clock.basis["validity"] == {
        "version": 1, "state": "unknown", "reason": "rendered_source_not_plaintext",
    }
    assert "validity" not in clock.bind([origin])[0].temporal_basis


def proposal(text, **changes):
    return {
        "temporal_assertion": {
            "kind": "state",
            "status": "ongoing",
            "source_index": 1,
            "quote": text,
            "valid_from_quote": "昨天",
            "valid_to_quote": None,
            **changes,
        },
        "source_indices": [1],
    }


def test_source_clock_not_processing_clock_and_unknown_end():
    text = "我从昨天起住在海边，现在还在这里。"
    value = resolve_validity(proposal(text), [source(text)], (1,))
    assert value["state"] == "asserted"
    assert datetime.fromisoformat(value["valid_from"]) == NOW.replace(
        hour=16
    ) - timedelta(days=2)
    assert value["valid_to"] is None
    assert valid_at(value, NOW) == "unknown"  # no infinite guarantee
    assert valid_at(value, NOW - timedelta(days=3)) == "outside"
    assert value["truth"] == "speaker_assertion"
    assert value["source_quote"] == text


def test_full_boundary_phrase_and_repeated_date_are_not_guessed():
    text = "我从明天开始住在学校。"
    value = resolve_validity(
        proposal(text, kind="plan", status="planned", valid_from_quote="从明天开始"),
        [source(text)],
        (1,),
    )
    assert value["state"] == "asserted" and value["status"] == "planned"
    ambiguous = "我昨天说从昨天开始住学校，现在还在这儿。"
    assert (
        resolve_validity(proposal(ambiguous), [source(ambiguous)], (1,))["state"]
        == "unknown"
    )


def test_occurrence_and_mention_clock_have_distinct_evidence():
    text = "我昨天说的旅行是2026年9月3日去的。"
    value = resolve_fact_clock(
        {
            **proposal(text, kind="point", status="occurred", valid_from_quote=None),
            "time_quote": "2026年9月3日",
            "time_source_index": 1,
        },
        [source(text)],
    )
    assert value.basis["validity"]["status"] == "occurred"
    assert (
        value.occur_time.astimezone(ZoneInfo("Asia/Shanghai")).date().isoformat()
        == "2026-09-03"
    )
    assert value.statement_time == NOW


def test_half_open_calendar_end_and_no_clock_inferred_completion():
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
    assert value["state"] == "asserted"
    end = datetime.fromisoformat(value["valid_to"])
    local = end.astimezone(ZoneInfo("Asia/Shanghai"))
    assert local.day == 7 and local.hour == 0
    assert valid_at(value, end - timedelta(seconds=1)) == "inside"
    assert valid_at(value, end) == "outside"
    plan = "我计划明天去爬山。"
    value = resolve_validity(
        proposal(plan, kind="plan", status="planned", valid_from_quote=None),
        [source(plan)],
        (1,),
    )
    assert value["status"] == "planned" and value["valid_from"] is None


@pytest.mark.parametrize(
    "text",
    [
        "姐姐说：我从昨天起在医院。",
        "如果我从昨天起在医院。",
        "我可能从昨天起住在学校。",
        "我是不是从昨天起就住在这里？",
        "我妈妈从昨天起在医院。",
        "我不是从昨天起住这里的。",
        "同事说“我从昨天起住学校”。",
        "我朋友从昨天开始在外地。",
    ],
)
def test_source_subject_and_modality_not_cherry_picked(text):
    assert resolve_validity(proposal(text), [source(text)], (1,))["state"] == "unknown"


@pytest.mark.parametrize(
    "patch",
    [
        {"valid_from_quote": "明天"},
        {"valid_from_quote": "前几天"},
        {"valid_from_quote": ["昨天"]},
        {"source_index": True},
        {"source_index": 2},
        {"kind": []},
        {"status": "verified"},
        {"valid_from": "2026-01-01"},
        {"quote": "我从昨天起住在另一个地方。"},
    ],
)
def test_invalid_proposals_cannot_mint_validity(patch):
    text = "我从昨天起住在学校，现在还在这里。"
    assert (
        resolve_validity(proposal(text, **patch), [source(text)], (1,))["state"]
        == "unknown"
    )


def test_point_date_and_statement_date_are_not_validity():
    text = "我昨天参加了培训。"
    value = resolve_validity(
        proposal(text, kind="point", status="occurred"), [source(text)], (1,)
    )
    assert value["state"] == "unknown"
    value = resolve_validity(
        proposal(text, kind="point", status="occurred", valid_from_quote=None),
        [source(text)],
        (1,),
    )
    assert value["state"] == "asserted" and value["valid_from"] is None


def test_frozen_clock_binds_validity_only_to_the_correct_source():
    text = "我从昨天起住在学校，现在还在这里。"
    fact = proposal(text)
    fact["source_indices"] = [1, 2]
    origins = [source(text), replace(source("我今天带了书。"), ref="second")]
    clock = resolve_fact_clock(fact, origins)
    bound = clock.bind(origins)
    assert bound[0].temporal_basis["validity"]["state"] == "asserted"
    assert "validity" not in bound[1].temporal_basis
    assert "source_version" not in public_validity(bound[0].temporal_basis["validity"])


@pytest.mark.parametrize(
    "field,bad",
    [
        ("kind", []),
        ("status", {}),
        ("status", "planned"),
        ("kind", "point"),
        ("valid_from", 123),
        ("timezone", "Unknown/Zone"),
        ("from_precision", []),
        ("statement_at", "2026-01-01"),
    ],
)
def test_corrupt_persisted_metadata_is_unknown_not_exception(field, bad):
    text = "我从昨天起住在学校，现在还在这里。"
    value = resolve_validity(proposal(text), [source(text)], (1,))
    corrupt = copy.deepcopy(value)
    corrupt[field] = bad
    assert public_validity(corrupt)["state"] == "unknown"


def test_invalid_source_index_cannot_read_outside_selected_origins():
    text = "我现在住在学校。"
    assert (
        resolve_validity(proposal(text, source_index=2), [source(text)], (2,))["state"]
        == "unknown"
    )


@pytest.mark.parametrize(
    "text,changes",
    [
        (
            "我现在住在学校。",
            {"kind": "state", "status": "planned", "valid_from_quote": None},
        ),
        ("我现在住在学校。", {"status": "completed", "valid_from_quote": None}),
        (
            "我现在住在学校。",
            {"kind": "plan", "status": "planned", "valid_from_quote": None},
        ),
        ("我从明天开始住学校，现在还在这儿。", {"valid_from_quote": "明天"}),
        ("我从昨天到明天住学校。", {"status": "completed", "valid_to_quote": "明天"}),
        (
            "我从2026年9月8日到2026年9月2日住旅馆。",
            {
                "status": "completed",
                "valid_from_quote": "2026年9月8日",
                "valid_to_quote": "2026年9月2日",
            },
        ),
        ("我从昨天下午开始住学校，现在还在这儿。", {"valid_from_quote": "昨天下午"}),
        ("我从前几天开始住学校，现在还在这儿。", {"valid_from_quote": "前几天"}),
        (
            "我从昨天开始住学校，昨天参加了培训。",
            {"kind": "point", "status": "occurred"},
        ),
    ],
)
def test_unsupported_status_and_imprecise_or_future_bounds_remain_unknown(
    text, changes
):
    assert (
        resolve_validity(proposal(text, **changes), [source(text)], (1,))["state"]
        == "unknown"
    )


def test_missing_assertion_and_unknown_metadata_do_not_imply_validity():
    assert resolve_validity({}, [], ())["reason"] == "not_proposed"
    assert public_validity({"version": 1, "state": "unknown"})["state"] == "unknown"
    assert valid_at(None, NOW) == "unknown"


@pytest.mark.parametrize("kind", ["state", "stable"])
def test_continuing_state_has_no_invented_start_or_infinite_end(kind):
    text = "我一直住在宁波。"
    value = resolve_validity(
        proposal(text, kind=kind, valid_from_quote=None), [source(text)], (1,)
    )
    assert value["state"] == "asserted" and value["status"] == "ongoing"
    assert value["valid_from"] is None and value["valid_to"] is None
    assert valid_at(value, NOW) == "unknown"


@pytest.mark.parametrize(
    "patch",
    [
        {"timezone": "x" * 65},
        {
            "valid_from": "2026-11-02T00:00:00+08:00",
            "valid_to": "2026-11-01T00:00:00+08:00",
        },
        {"from_precision": "arbitrary"},
    ],
)
def test_corrupt_history_interval_cannot_establish_membership(patch):
    text = "我从昨天起住在学校，现在还在这里。"
    value = {**resolve_validity(proposal(text), [source(text)], (1,)), **patch}
    assert public_validity(value)["state"] == "unknown"
    assert valid_at(value, NOW) == "unknown"


@pytest.mark.parametrize('prefix', ['我没说过', '我没提过', '我没讲过', '我没有告诉你', '我否认', '我并未说过'])
def test_negative_attribution_outside_selected_quote_cannot_assert_interval(prefix):
    quote = '我从2026年8月1日到2026年8月6日住在旅馆'
    text = prefix + quote
    value = resolve_validity(proposal(quote, status='completed', valid_from_quote='2026年8月1日', valid_to_quote='2026年8月6日'), [source(text)], (1,))
    assert value['state'] == 'unknown'


@pytest.mark.parametrize('verb', ['没住', '没工作', '没实习', '没参加', '不打算住', '不计划住', '不会住'])
def test_negated_state_or_plan_is_not_a_positive_assertion(verb):
    text = '我从2026年8月1日到2026年8月6日' + verb + '在旅馆'
    value = resolve_validity(proposal(text, status='completed', valid_from_quote='2026年8月1日', valid_to_quote='2026年8月6日'), [source(text)], (1,))
    assert value['state'] == 'unknown'


@pytest.mark.parametrize('start,end,expected', [
    ('2026年8月1日','8月6日','2026-08-06T16:00:00+00:00'),
    ('2024年2月28日','2月29日','2024-02-29T16:00:00+00:00'),
    ('2025年2月28日','2月29日',None),
    ('2025年12月31日','1月2日',None),
])
def test_source_endpoint_inherits_only_adjacent_explicit_year(start,end,expected):
    text=f'我从{start}到{end}住在旅馆'
    value=resolve_validity(proposal(text,status='completed',valid_from_quote=start,valid_to_quote=end),[source(text)],(1,))
    if expected:
        assert value['state']=='asserted' and datetime.fromisoformat(value['valid_to'])==datetime.fromisoformat(expected)
        assert value['valid_to_quote']==end
    else:assert value['state']=='unknown'


@pytest.mark.parametrize("text",[
 "我从昨天起有没有住在海边，现在还在这里吗",
 "我从昨天起会不会住在海边，现在还在这里",
 "我从昨天起住在海边，现在还在这里吗",
])
def test_question_without_question_mark_cannot_create_assertion(text):
    value=resolve_validity(proposal(text),[source(text)],(1,))
    assert value['state']=='unknown' and value['reason']=='non_assertive_source'
