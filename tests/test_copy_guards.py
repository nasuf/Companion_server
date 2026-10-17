"""Deterministic objections to activity copy, anchored on production drafts.

A false positive turns a good message into the generic fallback, so every
positive case here has a nearby negative that must stay clean.
"""
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from app.services.offline import copy_guards as guards
from app.services.offline.evidence_text import dated, mask_quotes, undated

TZ = ZoneInfo("Asia/Shanghai")
# Tuesday 2026-10-13, the day the concert below was recommended.
NOW = datetime(2026, 10, 13, 23, 47, tzinfo=TZ)
CONCERT = {
    "discovery_metadata": {"kind": "event"},
    "starts_at": "2026-10-17T19:30:00+08:00",
    "ends_at": "2026-10-17T22:00:00+08:00",
}
PLACE = {"discovery_metadata": {"kind": "place"}}


def test_date_relation_is_computed_not_left_to_the_model():
    assert guards.date_relation(CONCERT, NOW) == "2026-10-17（星期六，今天之后第 4 天，本周六）"
    assert guards.date_relation(PLACE, NOW) == "固定地点，不对应某一天"
    tomorrow = {**CONCERT, "starts_at": "2026-10-14T10:00:00+08:00", "ends_at": "2026-10-16T18:00:00+08:00"}
    assert guards.date_relation(tomorrow, NOW) == (
        "2026-10-14（星期三，明天，本周三） 至 2026-10-16（星期五，今天之后第 3 天，本周五）")
    assert guards.today_text(NOW) == "2026-10-13（星期二）"


def test_production_draft_is_rejected_for_wrong_year_and_going_along():
    draft = ("哎刚刷到张韶涵要在镇江开演唱会，立马就想到你了！时间是明年十月中旬的一个晚上，"
             "在体育会展中心那边。到时候要不要一起去凑个热闹？")
    issues = guards.copy_issues(draft, CONCERT, NOW)
    assert any("明年" in item for item in issues)
    assert any("一起去" in item for item in issues)


@pytest.mark.parametrize("text", [
    "这周六晚上在体育会展中心开唱，10月17日，周末去正好。",
    "17号晚上的场，要不要去听听？",
    "你10月10号说压力有点大，约上朋友一起去放松也不错。",
    "我们之前聊过音乐，我们去年也提过演唱会。",
    "在那边见到的舞台应该挺亮眼，可以先看看介绍，一起看看歌单也行。",
    "下周三之前把票的事确认好就行。",  # A deadline before the session, not its date.
])
def test_correct_or_unrelated_dates_and_company_stay_clean(text):
    assert guards.copy_issues(text, CONCERT, NOW) == []


@pytest.mark.parametrize("text,needle", [
    ("这周五晚上去看看？", "这周五"),
    ("下周六的场次别错过。", "下周六"),
    ("明天晚上就开唱。", "明天晚上"),
    ("今晚就在体育场。", "今晚"),
    ("10月18日晚上开唱。", "10月18日"),
    ("明天前往体育场就行。", "明天"),  # "前往" is going, not a deadline.
    ("今年最后一场了。", None),  # Same year: allowed.
])
def test_wrong_session_dates_are_named(text, needle):
    issues = guards.date_issues(text, CONCERT, NOW)
    if needle is None:
        assert issues == []
    else:
        assert any(needle in item for item in issues), issues


def test_weekend_and_cross_year_month_day():
    weekday_show = {**CONCERT, "starts_at": "2026-10-14T19:30:00+08:00", "ends_at": "2026-10-14T21:00:00+08:00"}
    assert any("周末" in item for item in guards.date_issues("周末去刚好。", weekday_show, NOW))
    assert guards.date_issues("周末去刚好。", CONCERT, NOW) == []
    december = datetime(2026, 12, 20, 12, tzinfo=TZ)
    january_show = {**CONCERT, "starts_at": "2027-01-05T19:30:00+08:00", "ends_at": "2027-01-05T21:00:00+08:00"}
    assert guards.date_issues("1月5日晚上，明年第一场。", january_show, december) == []
    assert any("1月6日" in item for item in guards.date_issues("1月6日晚上。", january_show, december))


def test_places_are_never_date_checked():
    assert guards.date_issues("明天或这周五都可以去。", PLACE, NOW) == []


@pytest.mark.parametrize("text", [
    "那边开放时间不太固定，记得确认。",
    "虽然咱们之前没具体聊过追星的事儿，但你应该会喜欢。",
    "具体开放时段素材里没有写明。",
    "开放时段没写太细，出发前确认一下。", "开放时间没写清楚。", "门票价格没有写明。",
    "既然没特别偏好，就单纯分享个去处。", "既然没有你之前的喜好记录，就单纯分享给你。",
    "不过这种地方时间不一定固定，出发前确认。",
])
def test_input_state_disclosure_is_feedback(text):
    assert guards.DISCLOSURE_FEEDBACK in guards.copy_issues(text, PLACE, NOW)


@pytest.mark.parametrize("text", ["有空写写日记也不错。", "你没写过信吧，这里能寄明信片。", "名字写得很有意思。",
                                  "时间安排没写死，随你。",
                                  "没什么偏好也没关系，随便逛逛。", "没什么偏好的话可以试试招牌。",
                                  "停留时间不一定要长，逛逛就好。"])
def test_ordinary_writing_words_are_not_disclosure(text):
    assert guards.DISCLOSURE_FEEDBACK not in guards.copy_issues(text, PLACE, NOW)


@pytest.mark.parametrize("text", [
    "具体的开放时段资料里没细说，出发前确认。", "不过那边没写具体开放时段。", "没给具体的开放时间。",
])
def test_data_state_in_other_words_is_feedback(text):
    assert guards.DISCLOSURE_FEEDBACK in guards.copy_issues(text, PLACE, NOW)


@pytest.mark.parametrize("text,visit", [
    ("刚路过解放路那边，看到个茶室。", True), ("我之前去过一次。", True), ("我也去过。", True),
    ("你要是刚路过可以进去看看。", False), ("要是刚好路过就进去坐坐。", False),
    ("顺路路过的话可以看看。", False), ("这个我去过吗？", False),
])
def test_invented_visits_are_feedback_everywhere(text, visit):
    # No facts and no reference count: card copy gets this check too.
    assert any("没有去过或路过" in item for item in guards.copy_issues(text, PLACE, NOW)) is visit


@pytest.mark.parametrize("text,references,cited", [
    ("之前看你关注过这场，若是有空去听听。", 0, True), ("演唱会今晚开唱。之前提过，想着能换换心情。", 0, True),
    ("你之前说喜欢历史，可以去转转。", 0, True), ("你之前说喜欢历史，可以去转转。", 1, False),
    ("想着你之前挺爱听流行歌的，会喜欢。", 0, True), ("你平时做软件费神，去换换脑子？", 0, True),
    ("你最近要是有空，可以去看看。", 0, False), ("要是你今晚刚好没什么安排，去听听？", 0, False),
    ("想着你之前要是听过她的歌，会喜欢现场。", 0, False), ("我之前提过的那家，可以去看看。", 0, False),
])
def test_citing_the_user_needs_a_verified_quote(text, references, cited):
    issues = guards.copy_issues(text, PLACE, NOW, facts_text="{}", user_references=references)
    assert any("没有用户原文可依据" in item for item in issues) is cited


@pytest.mark.parametrize("text,facts,flagged", [
    ("就当去安静角落待会儿。", '{"category": "寺庙"}', True), ("那边挺清静的。", "{}", True),
    ("馆里很安静。", '{"description": "安静的阅读空间"}', False),
])
def test_environment_claims_need_the_facts(text, facts, flagged):
    _environment_check(text, facts, flagged)


@pytest.mark.parametrize("text,facts,flagged", [
    ("反正就在市区，过去也方便。", '{"location": "镇江市体育会展中心-体育场 团山路33号"}', True),
    ("它就在运河路交叉口南边，位置也方便。", '{"location": "解放路辅路与运河路交叉口南"}', True),
    ("地铁口出来就到，交通挺方便。", '{"description": "地铁2号线直达"}', False),
])
def test_location_ease_needs_the_facts(text, facts, flagged):
    issues = guards.copy_issues(text, PLACE, NOW, facts_text=facts, user_references=1)
    assert any("不能说明位置或交通" in item for item in issues) is flagged


def _environment_check(text, facts, flagged):
    issues = guards.copy_issues(text, PLACE, NOW, facts_text=facts, user_references=1)
    assert any("资料没有写环境" in item for item in issues) is flagged
    # Card copy has no repair round, so it keeps relying on the model review here.
    assert not any("资料没有写环境" in item for item in guards.copy_issues(text, PLACE, NOW, facts_text=facts))


@pytest.mark.parametrize("text", [
    "我们现场见！", "到时候见～", "我陪你一起去逛逛。", "咱们就去这儿吧。", "我们要去镇江赴约了！",
])
def test_going_along_is_feedback(text):
    assert any("线上陪伴" in item for item in guards.copy_issues(text, PLACE, NOW))


@pytest.mark.parametrize("text,facts,expected", [
    ("全天免费参加手作", '{"opening_hours": "10:00—21:00", "price_info": "29.00"}', ["全天", "免费"]),
    ("这里随时可去，24小时都开着。", "{}", ["随时可去"]),
    ("免费开放，全天都能去。", '{"description": "全天免费开放的城市公园"}', []),
    ("门票免费。", '{"verified_source_text": [{"text": "场馆免费，18点到21点开放"}]}', []),
    ("出发前确认开放时间，有空随时跟我说说。", "{}", []),
])
def test_access_claims_need_the_facts_to_state_them(text, facts, expected):
    issues = guards.copy_issues(text, PLACE, NOW, facts_text=facts)
    assert len(issues) == len(expected)
    assert all(word in issue for word, issue in zip(expected, issues))


def test_bounded_issues_are_short_and_few():
    assert guards.bounded_issues(["x" * 300, "", None, "b", "c", "d"]) == ["x" * 120, "b", "c"]
    assert guards.bounded_issues("not a list") == []


def test_evidence_dates_round_trip():
    stamped = dated("用户今天出来走走", datetime(2026, 10, 10, 16, 0), with_time=False)
    assert stamped == "[2026-10-11] 用户今天出来走走"  # Naive DB times are UTC.
    assert dated("吃饭了吗", "2026-10-13T07:18:00+00:00", with_time=True) == "[2026-10-13 15:18] 吃饭了吗"
    assert dated("无时间", None, with_time=True) == "无时间"
    assert undated(stamped) == "用户今天出来走走" and undated("[注意] 不是日期") == "[注意] 不是日期"


def test_malformed_inputs_are_ignored_not_guessed():
    from datetime import date
    assert guards.event_span({"starts_at": "not a date"}) is None
    # Naive storage times are UTC: 11:30Z is 19:30 the same local day.
    assert guards.event_span({"starts_at": datetime(2026, 10, 17, 11, 30)}) == (date(2026, 10, 17), date(2026, 10, 17))
    assert guards.date_issues("2月30日晚上。", CONCERT, NOW) == []
    assert guards.copy_issues("详见 https://x.test", PLACE, NOW) == ["删除网址。"]
    assert dated("x", "not-a-date", with_time=False) == "x"


# --- Review findings (2026-10-17) ---

def test_negated_facts_do_not_license_all_day_claims():
    page_facts = '{"date_time": "固定地点；开放时段未提供，不能推断全天或随时营业"}'
    issues = guards.copy_issues("这家书店全天开放，随时都可以去坐坐", PLACE, NOW, facts_text=page_facts)
    assert any("全天" in item for item in issues)
    assert guards.copy_issues("全天都能去。", PLACE, NOW, facts_text='{"description": "非全天开放"}')


@pytest.mark.parametrize("text", [
    "这周一定要去看看", "约上朋友下周一起去", "上周末你说想出门", "叫上朋友陪你去转转",
    "17号晚上开唱", "坐2号线到3号馆", "下周三之前买好票", "就在团山路33号那边，伯先路12号也近",
])
def test_review_false_positives_stay_clean(text):
    assert guards.copy_issues(text, CONCERT, NOW) == []


@pytest.mark.parametrize("text,needle", [("18号晚上开唱", "18号"), ("陪你去逛逛", "陪你去")])
def test_review_false_negatives_are_caught(text, needle):
    assert any(needle in item for item in guards.copy_issues(text, CONCERT, NOW))


def test_long_runs_across_new_year_and_monday_afternoon():
    exhibition = {"discovery_metadata": {"kind": "event"},
                  "starts_at": "2026-12-20T10:00:00+08:00", "ends_at": "2027-02-28T18:00:00+08:00"}
    assert guards.date_issues("12月20日开展，到2月28日都有，21号也行", exhibition,
                              datetime(2027, 1, 5, tzinfo=TZ)) == []
    monday = {**CONCERT, "starts_at": "2026-10-19T14:00:00+08:00", "ends_at": "2026-10-19T16:00:00+08:00"}
    assert guards.date_issues("下周一下午", monday, NOW) == []


def test_stored_reasons_do_not_repeat_the_users_words():
    reasons = ["“你头疼刚好些”与“[2026-10-10] 用户头疼好一些了”不符", "日期不符"]
    assert mask_quotes(reasons, ["[2026-10-10] 用户头疼好一些了"]) == ["“你头疼刚好些”与“[2026-10-10] 〔用户原文〕”不符", "日期不符"]
    assert mask_quotes(["x"], []) == ["x"]
