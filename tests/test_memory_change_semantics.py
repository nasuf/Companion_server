from datetime import datetime, timezone

import pytest

from app.services.memory.interaction.change_semantics import (
    resolve_change_semantics,
    select_change_statements,
)

ANCHOR = datetime(2026, 11, 1, 4, tzinfo=timezone.utc)


def row(text, *, at=ANCHOR, ident="original"):
    return {"id": ident, "content": text, "created_at": at}


@pytest.mark.parametrize(
    "text,change,kind",
    [
        ("我现在搬到乙城了", "变化", "transition"),
        ("上次说错了，其实我一直住在乙城", "变化", "correction"),
        ("我打算下个月搬到乙城", "变化", "planned"),
        ("我2027年1月2日搬到乙城", "变化", "planned"),
        ("我喜欢散步", "新增", "addition"),
        ("如果我搬到乙城就好了", "变化", "unknown"),
        ("我说的是朋友搬到乙城", "变化", "unknown"),
        ("我并没有搬到乙城", "变化", "unknown"),
        ("“我搬到乙城了”是电影台词", "变化", "unknown"),
        ("我可能搬到乙城", "变化", "unknown"),
        ("我打算明天搬家，上次说错了", "错误", "unknown"),
        ("我还没确定", "变化", "unknown"),
        ("不是", "变化", "unknown"),
        ("算了不用改", "变化", "unknown"),
    ],
)
def test_confirmation_does_not_turn_plan_quote_or_other_person_into_current_fact(
    text, change, kind
):
    assert resolve_change_semantics(change, [row(text)]).kind == kind


def test_unknown_start_never_uses_message_or_confirmation_clock():
    result = resolve_change_semantics("变化", [row("我已经搬到乙城了")])
    assert result.kind == "transition" and result.statement_at == ANCHOR
    assert result.boundary_start is None and result.boundary_end is None


def test_day_precision_is_a_possible_window_not_midnight_as_exact_transition():
    result = resolve_change_semantics("变化", [row("我昨天搬到乙城了")])
    assert result.kind == "transition" and result.precision == "day"
    assert result.boundary_start == datetime(2026, 10, 30, 16, tzinfo=timezone.utc)
    assert result.boundary_end == datetime(2026, 10, 31, 16, tzinfo=timezone.utc)
    assert result.boundary_source_id == "original"


@pytest.mark.parametrize(
    "text",
    [
        "我昨天说过上周搬到乙城了",
        "我10月2日搬到甲城，10月31日搬到乙城",
        "我前阵子搬到乙城了",
        "我刚刚搬到乙城了",
    ],
)
def test_statement_date_multiple_dates_and_imprecise_date_do_not_become_exact_bounds(
    text,
):
    result = resolve_change_semantics("变化", [row(text)])
    assert result.boundary_start is None and result.boundary_end is None


def test_multi_message_span_is_not_collapsed_into_one_effective_date():
    result = resolve_change_semantics(
        "变化", [row("我昨天搬到乙城了"), row("我今天开始新工作", ident="second")]
    )
    assert result.boundary_start is None


def test_malformed_source_clock_and_unknown_model_label_fail_explicitly():
    assert (
        resolve_change_semantics("变化", [row("我搬家了", at=None)]).reason
        == "source_clock_unavailable"
    )
    assert (
        resolve_change_semantics("delete", [row("我搬家了")]).reason
        == "invalid_contract"
    )


def test_yes_is_not_a_fresh_statement_but_explicit_confirmation_change_is():
    original = [row("我昨天搬到乙城了")]
    assert select_change_statements(original, [row("是的")]) == original
    correction = [row("不是，我只是计划下个月搬")]
    assert select_change_statements(original, correction) == correction
    restatement = [row("我今天已经搬到乙城了")]
    assert select_change_statements(original, restatement) == restatement
    uncertain = [row("我还没确定")]
    assert select_change_statements(original, uncertain) == uncertain
    assert resolve_change_semantics("变化", uncertain).kind == "unknown"


@pytest.mark.parametrize(
    "text,kind",
    [
        ("我不搬到乙城了", "unknown"),
        ("我计划已经完成了", "transition"),
        ("我的准备工作已经结束了", "transition"),
        ("我明天的面试已经取消了", "transition"),
        ("我已经搬到乙城，打算明年再搬去丙城", "unknown"),
        ("我计划下个月搬到乙城", "planned"),
        ("我准备搬到乙城", "planned"),
    ],
)
def test_plan_words_do_not_override_completed_events_or_mix_two_facts(text, kind):
    value = resolve_change_semantics("变化", [row(text)])
    assert value.kind == kind
    if kind == "transition":
        assert value.boundary_start is None


@pytest.mark.parametrize(
    "text,kind",
    [
        ("我搬到上海了，你呢？", "transition"),
        ("我搬到上海了，希望一切顺利", "transition"),
        ("其他同事都在杭州，我搬到上海了", "transition"),
        ("我搬到上海了吗？", "unknown"),
    ],
)
def test_aside_clauses_do_not_make_an_assertion_non_assertive(text, kind):
    assert resolve_change_semantics("变化", [row(text)]).kind == kind


@pytest.mark.parametrize("reply", ["对对对", "是的是的", "嗯嗯~", "OK", "确实", "好的好的！"])
def test_repeated_acknowledgement_never_becomes_the_fact_source(reply):
    original = [{"id": "o", "content": "我昨天搬到乙城了"}]
    assert select_change_statements(original, [{"id": "c", "content": reply}]) == original


@pytest.mark.parametrize(
    "text", ["我马上要搬到上海了", "我要入职了", "我快离职了", "我就要搬到北京了"]
)
def test_imminent_change_with_le_is_a_plan_not_a_done_change(text):
    # 了 here marks something about to happen; archiving the current fact
    # would be the worst possible outcome.
    assert resolve_change_semantics("变化", [row(text)]).kind == "planned"


@pytest.mark.parametrize("reply", ["对呀，谢谢", "是的哈哈", "对👍", "嗯嗯谢谢你"])
def test_confirmation_without_its_own_fact_keeps_the_original_plan(reply):
    original = [row("我下个月搬到上海")]
    rows = select_change_statements(original, [row(reply, ident="c")])
    assert rows == original
    assert resolve_change_semantics("变化", rows).kind == "planned"
