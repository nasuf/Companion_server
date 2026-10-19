from datetime import datetime

import pytest

from app.services.offline.opening_hours import assess_opening_hours, public_hours, visit_context, visit_issues
from app.services.offline.recommendation_content import fallback_detail

MUSEUM_HOURS = "10/08 周四 全天不开放；1/1-12/31 周二–周三, 周五–周日 09:00–17:00开放；01/01-10/07 周四 09:00–17:00开放；10/09-12/31 周四 09:00–17:00开放；1/1-12/31 周一 全天不开放；国庆节 09:00–20:00开放"


def card(hours=MUSEUM_HOURS, **metadata):
    return {"title": "镇江博物馆", "location_name": "镇江博物馆", "address": "伯先路85号", "category": "展览与博物馆",
            "discovery_metadata": {"kind": "place", "opening_hours": hours, "checked_at": "2026-10-18T14:15:15+00:00", **metadata}}


@pytest.mark.parametrize("moment,status,reason", [
    ("2026-10-18T22:15:00+08:00", "closed", "after_closing"),
    ("2026-10-18T08:59:00+08:00", "closed", "before_opening"),
    ("2026-10-18T09:00:00+08:00", "open", "within_hours"),
    ("2026-10-18T16:59:00+08:00", "open", "within_hours"),
    ("2026-10-18T17:00:00+08:00", "closed", "after_closing"),
    ("2026-10-18T14:15:00+00:00", "closed", "after_closing"),
    ("2026-10-19T12:00:00+08:00", "closed", "closed_day"),
    ("2026-10-08T12:00:00+08:00", "closed", "closed_day"),
    ("2026-10-15T12:00:00+08:00", "open", "within_hours"),
    ("2026-10-01T12:00:00+08:00", "unknown", "unresolved_rules"),
])
def test_actual_museum_ranges_exceptions_and_current_clock(moment, status, reason):
    result = assess_opening_hours(card(), datetime.fromisoformat(moment))
    assert (result["status"], result["reason"]) == (status, reason)


@pytest.mark.parametrize("hours,moment,status", [
    ("每天 09:00-17:00", "2026-10-19T12:00:00+08:00", "open"),
    ("周一闭馆，周二至周日9:00-17:00", "2026-10-19T12:00:00+08:00", "closed"),
    ("周五22:00-02:00", "2026-10-17T01:00:00+08:00", "open"),
    ("周五22:00-02:00", "2026-10-17T03:00:00+08:00", "unknown"),
    ("全天营业", "2026-10-19T23:59:00+08:00", "open"),
    ("周一 09:00-17:00；周一全天不开放", "2026-10-19T12:00:00+08:00", "unknown"),
    ("预约开放，以通知为准", "2026-10-19T12:00:00+08:00", "unknown"),
    ("09:00-24:30", "2026-10-19T12:00:00+08:00", "unknown"),
    ("", "2026-10-19T12:00:00+08:00", "unknown"),
    ("节假日09:00-20:00；每天09:00-17:00", "2026-10-19T12:00:00+08:00", "open"),
    ("12/20-01/10 每天09:00-17:00", "2026-01-05T12:00:00+08:00", "open"),
    ("10/18 09:00-17:00", "2027-10-18T12:00:00+08:00", "unknown"),
    ("2026-10-18 09:00-17:00", "2026-10-19T12:00:00+08:00", "unknown"),
])
def test_conservative_supported_rules_and_unknowns(hours, moment, status):
    assert assess_opening_hours(card(hours), datetime.fromisoformat(moment))["status"] == status


def test_fallback_never_reintroduces_old_raw_exceptions():
    now = datetime.fromisoformat("2026-10-18T22:15:00+08:00")
    body = fallback_detail(card(), now=now)
    assert "09:00–17:00" in body and "2026-10-18" in body
    assert "10/08" not in body and "国庆节" not in body and "1/1-12/31" not in body
    assert "当前北京时间 2026-10-18 22:15" in visit_context(card(), now)
    assert "国庆" not in public_hours(card(), now)


@pytest.mark.parametrize("text", ["你这会儿没什么要紧事，要不要去逛逛看展呀", "现在就去看看吧", "今天周日开放，有空可以再去逛逛看展", "今晚就去逛逛吧"])
def test_after_closing_immediate_invitation_is_blocked(text):
    assert visit_issues(text, card(), datetime.fromisoformat("2026-10-18T22:15:00+08:00"))


@pytest.mark.parametrize("text", ["另选开放时段去看看", "今天已经闭馆了，改天再去", "下次有空可以去逛逛", "喜欢历史的话，可以先收藏这里"])
def test_future_and_conditional_recommendations_survive_closure(text):
    assert not visit_issues(text, card(), datetime.fromisoformat("2026-10-18T22:15:00+08:00"))


def test_venue_closure_does_not_override_a_future_event_session():
    c = card(kind="event")
    c["starts_at"] = "2026-10-20T20:00:00+08:00"
    now = datetime.fromisoformat("2026-10-19T22:15:00+08:00")
    assert assess_opening_hours(c, now)["status"] == "not_applicable"
    assert not visit_issues("今晚去看演出", c, now)  # Session date is checked separately.


@pytest.mark.parametrize("hours", ["02/30 09:00-17:00", "10/18与10/19 09:00-17:00", "09:00-09:00", "24:00-25:00", "周一09:00-17:00且全天休息"])
def test_ambiguous_invalid_dates_and_intervals_are_unknown(hours):
    assert assess_opening_hours(card(hours), datetime.fromisoformat("2026-10-19T12:00:00+08:00"))["status"] == "unknown"


def test_invalid_reference_and_unknown_now_open_claim():
    now = datetime.fromisoformat("2026-10-19T12:00:00+08:00")
    assert assess_opening_hours(card(checked_at="invalid"), now)["reason"] == "invalid_reference_date"
    assert visit_issues("现在开放，可以去看看", card("以通知为准"), now)
    assert not visit_issues("先收藏这个地方", card("以通知为准"), now)


def test_holiday_dataset_unavailable_keeps_rules_unknown(monkeypatch):
    import chinese_calendar
    def unavailable(_day):
        raise NotImplementedError()
    monkeypatch.setattr(chinese_calendar, "get_holiday_detail", unavailable)
    assert assess_opening_hours(card("国庆节09:00-20:00；每天09:00-17:00"), datetime.fromisoformat("2026-10-19T12:00:00+08:00"))["status"] == "unknown"
