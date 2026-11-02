"""TIME12 固定作息: 人设解析、某天安排、作息校验与兜底、此刻描述、补全与生成接线。

起因 (2026-10-30 生产): 人设写"晚上9点下班", 当天作息却排 19:30 下班回家, 两段
同时进主回复, 模型在 19:55 说"我都是九点整下班的"又说"提前溜了十分钟"。
"""
from __future__ import annotations

import json
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from app.services.schedule_domain import routine as routine_mod
from app.services.schedule_domain import schedule as schedule_mod
from app.services.schedule_domain.routine import (
    DAY,
    AgentRoutine,
    check_conformance,
    day_plan,
    describe_now,
    fmt_clock,
    parse_routine_lines,
    template_schedule,
)
from tests.conftest import FakeRedis

_TZ = ZoneInfo("Asia/Shanghai")
# 2026-10-30 是星期五, 11-01 是星期日, 10-01 国庆
_FRIDAY = date(2026, 10, 30)
_SUNDAY = date(2026, 11, 1)

# 生产事故 agent 的人设原句
_INCIDENT_LINES = [
    "晚上9点下班，回家后会整理当天的工作笔记",
    "每周日休息，会在家打扫卫生或和朋友聚会",
    "中午12点到1点休息，吃午饭并和猫咪互动",
    "每天早上7点半起床，8点到店准备开门",
]


def _routine(**kw) -> AgentRoutine:
    base = dict(wake=7 * 60 + 30, work_start=8 * 60, work_end=21 * 60, sleep=23 * 60,
                rest_weekdays=frozenset({6}), holiday_off=False, sources={})
    base.update(kw)
    return AgentRoutine(**base)


# ── 人设解析 ──────────────────────────────────────────────────────────

def test_incident_persona_lines_parse_to_fixed_routine():
    parsed = parse_routine_lines(_INCIDENT_LINES)
    assert {k: fmt_clock(v) for k, v in parsed.values.items()} == {
        "work_end": "21:00", "wake": "07:30", "work_start": "08:00",
    }
    assert parsed.rest_weekdays == {6}
    # "中午12点到1点休息" 是午休, 不是睡觉
    assert "sleep" not in parsed.values


@pytest.mark.parametrize("line, expected", [
    ("凌晨1点左右下班，回家后会整理当天带本的笔记", {"work_end": "次日 01:00"}),
    ("晚上十二点前睡觉，睡前会听一段轻音乐", {"sleep": "次日 00:00"}),
    ("晚上10点半左右休息，睡前整理当天的托管记录", {"sleep": "22:30"}),
    ("每天凌晨3点起床到店准备", {"wake": "03:00"}),
    ("每晚11点半睡觉", {"sleep": "23:30"}),
    ("晚上6点下班，回家后会看1小时书再睡觉", {"work_end": "18:00"}),
    ("6点下班", {"work_end": "18:00"}),
    ("早上9点到晚上6点上班，周末双休", {"work_start": "09:00", "work_end": "18:00"}),
    ("每天早上7点半起床，8点半到邮局上班", {"wake": "07:30", "work_start": "08:30"}),
])
def test_persona_phrasings(line, expected):
    parsed = parse_routine_lines([line])
    assert {k: fmt_clock(v) for k, v in parsed.values.items()} == expected


def test_weekend_rest_and_conflict_keep_first_line():
    parsed = parse_routine_lines(["晚上9点下班", "晚上7点下班", "周末不上班"])
    assert fmt_clock(parsed.values["work_end"]) == "21:00"
    assert parsed.conflicts and parsed.conflicts[0].startswith("work_end")
    assert parsed.rest_weekdays == {5, 6}


def test_routine_json_round_trip():
    routine = _routine(work_end=DAY + 60, sources={"work_end": "persona"})
    restored = AgentRoutine.from_json(json.loads(json.dumps(routine.to_json(derived_at="x"))))
    assert restored == routine
    assert AgentRoutine.from_json({"version": 99}) is None
    assert AgentRoutine.from_json({"version": 1, "wake": "bad", "sleep": "23:00", "rest_weekdays": []}) is None


# ── 某一天 ───────────────────────────────────────────────────────────

def test_day_plan_rest_weekday_holiday_and_makeup():
    routine = _routine(holiday_off=True)
    assert day_plan(routine, _FRIDAY).is_workday
    sunday = day_plan(routine, _SUNDAY)
    assert (sunday.is_workday, sunday.reason) == (False, "rest_weekday")
    assert sunday.wake == routine.wake + 60
    legal = SimpleNamespace(type="legal", name="国庆节")
    assert day_plan(routine, date(2026, 10, 1), holiday=legal).reason == "holiday"
    assert day_plan(routine, _SUNDAY, workday_swap=True).reason == "makeup_workday"
    # 门店不放法定假
    assert day_plan(_routine(holiday_off=False), date(2026, 10, 1), holiday=legal).is_workday
    assert day_plan(_routine(work_start=None, work_end=None), _FRIDAY).reason == "no_work"


# ── 校验与兜底 ──────────────────────────────────────────────────────

_INCIDENT_SCHEDULE = [
    {"start": "07:15", "end": "08:00", "event": "起床洗漱", "status": "忙碌", "kind": "起床"},
    {"start": "08:00", "end": "18:00", "event": "开店营业", "status": "很忙碌", "kind": "工作"},
    {"start": "18:00", "end": "19:30", "event": "打烊收尾", "status": "很忙碌", "kind": "工作"},
    {"start": "19:30", "end": "20:30", "event": "步行回住所", "status": "空闲", "kind": "通勤"},
    {"start": "20:30", "end": "23:30", "event": "整理手账", "status": "空闲", "kind": "休闲"},
    {"start": "23:30", "end": "次日 07:15", "event": "睡觉", "status": "睡眠", "kind": "睡眠"},
]


def test_incident_schedule_is_flagged():
    violations = check_conformance(_INCIDENT_SCHEDULE, day_plan(_routine(), _FRIDAY))
    assert any(v.startswith("work_end 19:30≠21:00") for v in violations)


def test_missing_kind_and_work_on_rest_day():
    legacy = [{k: v for k, v in s.items() if k != "kind"} for s in _INCIDENT_SCHEDULE]
    assert check_conformance(legacy, day_plan(_routine(), _FRIDAY)) == ["missing_kind"]
    assert "work_on_rest_day" in check_conformance(_INCIDENT_SCHEDULE, day_plan(_routine(), _SUNDAY))


@pytest.mark.parametrize("routine", [
    _routine(),
    _routine(work_start=9 * 60, work_end=18 * 60, sleep=DAY + 30),
    # 夜班: 18:00 上班, 凌晨 1 点下班, 2 点睡, 10 点起
    _routine(wake=10 * 60, work_start=18 * 60, work_end=DAY + 60, sleep=DAY + 120),
    _routine(work_start=None, work_end=None),
])
@pytest.mark.parametrize("day", [_FRIDAY, _SUNDAY])
def test_template_always_conforms(routine, day):
    plan = day_plan(routine, day)
    schedule = template_schedule(plan)
    assert check_conformance(schedule, plan) == []
    status = schedule_mod.get_current_status(schedule, datetime.combine(day, datetime.min.time()).replace(hour=15))
    assert status["status"] in {"idle", "busy", "very_busy", "sleep"}


# ── 此刻 ─────────────────────────────────────────────────────────────

def _at(day: date, hhmm: str) -> datetime:
    hour, minute = map(int, hhmm.split(":"))
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=_TZ)


def test_describe_now_incident_moment_is_still_at_work():
    view = describe_now(_routine(), None, _at(_FRIDAY, "19:55"))
    assert view["phase"] == "at_work"
    assert view["now_line"] == "现在19:55，正在上班，离下班还有1小时5分钟"
    assert view["today"] == "今天周五，上班 8:00–21:00"
    assert "21:00下班" in view["usual"] and "周日休息" in view["usual"]
    assert view["work_window_source"] == "routine"


def test_describe_now_uses_schedule_window_when_kinds_present():
    view = describe_now(_routine(), _INCIDENT_SCHEDULE, _at(_FRIDAY, "19:55"))
    assert view["work_window_source"] == "schedule"
    assert view["phase"] == "after_work" and view["now_line"].endswith("已经下班25分钟了")


def test_describe_now_before_work_day_off_and_overnight():
    assert describe_now(_routine(), None, _at(_FRIDAY, "07:40"))["phase"] == "before_work"
    assert describe_now(_routine(), None, _at(_SUNDAY, "10:00"))["phase"] == "day_off"
    night = _routine(wake=10 * 60, work_start=18 * 60, work_end=DAY + 60, sleep=DAY + 120)
    view = describe_now(night, None, _at(_FRIDAY, "00:30"))
    assert view["phase"] == "at_work" and view["now_line"].endswith("还在上昨晚的班，离下班还有30分钟")
    assert describe_now(night, None, _at(_FRIDAY, "02:00"))["phase"] == "before_work"
    # 周日凌晨: 周六晚的班还没下; 周一凌晨: 周日休息, 没有昨晚的班
    assert describe_now(night, None, _at(_SUNDAY, "00:30"))["phase"] == "at_work"
    monday = date(2026, 11, 2)
    assert describe_now(night, None, _at(monday, "00:30"))["phase"] == "before_work"


def test_status_context_carries_routine_for_short_circuit_replies():
    status = {"status": "busy", "activity": "上班", "routine": describe_now(_routine(), None, _at(_FRIDAY, "19:55"))}
    text = schedule_mod.format_schedule_context(status)
    assert text.startswith("你现在正在上班") and "离下班还有1小时5分钟" in text


# ── 补全与存档 ──────────────────────────────────────────────────────

class _FakeDB:
    def __init__(self, *, origin: dict | None, memories: list[str], stored: dict | None = None):
        self.origin, self.memories, self.stored = origin, memories, stored
        self.updates: list[tuple] = []

    async def query_raw(self, sql, *args):
        if "memory_profile_origins" in sql:
            return [{"payload_text": json.dumps(self.origin, ensure_ascii=False)}] if self.origin else []
        if "memories_ai" in sql:
            return [{"content": c} for c in self.memories]
        if "SELECT routine FROM ai_agents" in sql:
            return [{"routine": self.stored}]
        raise AssertionError(sql)

    async def execute_raw(self, sql, *args):
        self.updates.append((sql, args))
        if "routine IS NULL" in sql and self.stored is None:
            self.stored = json.loads(args[0])
        return 1


def _agent():
    return SimpleNamespace(id="agent-1", occupation="猫咖店员", age=25, city="苏州", lifeOverview="店员")


@pytest.fixture
def fake_db(monkeypatch):
    import app.db as app_db

    def install(**kw):
        fake = _FakeDB(**kw)
        monkeypatch.setattr(app_db, "db", fake)
        return fake
    return install


def _llm(monkeypatch, result):
    from app.services.llm import models
    from app.services.prompting import store

    call = AsyncMock(return_value=result)
    monkeypatch.setattr(models, "invoke_json", call)
    monkeypatch.setattr(models, "get_utility_model", lambda: None)
    monkeypatch.setattr(store, "get_prompt_text", AsyncMock(
        side_effect=lambda key: __import__("app.services.prompting.defaults", fromlist=["x"])
        .SCHEDULE_ROUTINE_DERIVE_PROMPT))
    return call


@pytest.mark.asyncio
async def test_persona_values_win_over_model(fake_db, monkeypatch):
    fake = fake_db(origin={"profile": {"lifestyle": {"routine": _INCIDENT_LINES[:2]}}},
                   memories=["每天早上7点半起床，8点到店准备开门"])
    call = _llm(monkeypatch, {"wake": "06:00", "work_start": "10:00", "work_end": "19:30",
                              "sleep": "23:00", "rest_weekdays": [1], "holiday_off": False})
    routine = await routine_mod.ensure_agent_routine(_agent())
    assert (fmt_clock(routine.wake), fmt_clock(routine.work_start), fmt_clock(routine.work_end)) == (
        "07:30", "08:00", "21:00")
    assert routine.rest_weekdays == frozenset({6})
    assert fmt_clock(routine.sleep) == "23:00" and routine.sources["sleep"] == "derived"
    assert routine.sources["work_end"] == "persona"
    rendered = call.await_args.args[1]
    assert "下班：21:00" in rendered and "已确定，必须原样保留" in rendered
    assert fake.stored is not None and fake.stored["work_end"] == "21:00"


@pytest.mark.asyncio
async def test_existing_routine_is_not_rederived(fake_db, monkeypatch):
    stored = _routine().to_json(derived_at="earlier")
    fake_db(origin=None, memories=[], stored=stored)
    call = _llm(monkeypatch, {})
    assert (await routine_mod.ensure_agent_routine(_agent())) == _routine()
    call.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [
    None,
    {"wake": "07:00", "work_start": "09:00", "work_end": "18:00", "sleep": "23:00"},  # 缺休息日
    {"wake": "07:00", "work_start": "09:00", "work_end": "09:30", "sleep": "23:00", "rest_weekdays": []},
    {"wake": "16:00", "work_start": "17:00", "work_end": "22:00", "sleep": "23:00", "rest_weekdays": []},
])
async def test_invalid_derivation_is_not_persisted(fake_db, monkeypatch, bad):
    fake = fake_db(origin=None, memories=[])
    _llm(monkeypatch, bad)
    assert await routine_mod.ensure_agent_routine(_agent()) is None
    assert fake.updates == [] and fake.stored is None


@pytest.mark.asyncio
async def test_no_fixed_work_is_allowed(fake_db, monkeypatch):
    fake_db(origin=None, memories=[])
    _llm(monkeypatch, {"wake": "09:00", "sleep": "01:00", "rest_weekdays": [], "no_fixed_work": True})
    routine = await routine_mod.ensure_agent_routine(_agent())
    assert routine is not None and not routine.has_work and fmt_clock(routine.sleep) == "次日 01:00"


# ── 生成接线 ──────────────────────────────────────────────────────────

@pytest.fixture
def gen_env(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setattr(schedule_mod, "get_redis", AsyncMock(return_value=redis))
    agent = SimpleNamespace(id="agent-1", name="阿恺", age=25, occupation="猫咖店员", city="", lifeOverview="x")
    raw_sql: list[tuple] = []

    async def execute_raw(sql, *args):
        raw_sql.append((sql, args))
        return 1
    monkeypatch.setattr(schedule_mod, "db", SimpleNamespace(
        aidailyschedule=SimpleNamespace(upsert=AsyncMock()),
        aiagent=SimpleNamespace(find_unique=AsyncMock(return_value=agent)),
        execute_raw=execute_raw,
    ))
    from app.services import runtime_config
    monkeypatch.setattr(runtime_config, "bind_agent_context", AsyncMock())
    monkeypatch.setattr(routine_mod, "ensure_agent_routine", AsyncMock(return_value=_routine()))
    monkeypatch.setattr(schedule_mod, "_ground_schedule_result",
                        AsyncMock(side_effect=lambda a, s, **kw: (s, True)))
    monkeypatch.setattr(schedule_mod, "_grounding_policy", AsyncMock(return_value=""))
    from app.services.prompting.registry import PROMPT_DEFINITION_MAP

    async def prompt_text(key):
        return PROMPT_DEFINITION_MAP[key].default_text
    monkeypatch.setattr(schedule_mod, "get_prompt_text", prompt_text)
    return SimpleNamespace(redis=redis, raw_sql=raw_sql)


def _llm_schedule(monkeypatch, *outputs):
    prompts: list[str] = []
    queue = list(outputs)

    async def invoke(model, prompt):
        prompts.append(prompt)
        return queue.pop(0)
    monkeypatch.setattr(schedule_mod, "invoke_json", invoke)
    monkeypatch.setattr(schedule_mod, "get_utility_model", lambda: None)
    return prompts


def _meta(env) -> dict:
    meta_sql = [args for sql, args in env.raw_sql if "generation_meta" in sql]
    assert meta_sql, "generation meta not written"
    return json.loads(meta_sql[-1][0])


def _conforming() -> list[dict]:
    return template_schedule(day_plan(_routine(), _FRIDAY))


@pytest.mark.asyncio
async def test_generation_appends_constraints_and_accepts_conforming(gen_env, monkeypatch):
    prompts = _llm_schedule(monkeypatch, _conforming())
    result = await schedule_mod.generate_daily_schedule(
        "agent-1", "阿恺", None, life_overview="x", date=_at(_FRIDAY, "03:30"))
    assert "【固定作息" in prompts[0] and "21:00 结束工作离开" in prompts[0] and '"kind"' in prompts[0]
    assert result == _conforming()
    assert _meta(gen_env)["conformance"] == "ok"


@pytest.mark.asyncio
async def test_generation_retries_with_violations_then_accepts(gen_env, monkeypatch):
    prompts = _llm_schedule(monkeypatch, _INCIDENT_SCHEDULE, _conforming())
    result = await schedule_mod.generate_daily_schedule(
        "agent-1", "阿恺", None, life_overview="x", date=_at(_FRIDAY, "03:30"))
    assert len(prompts) == 2 and "work_end 19:30≠21:00" in prompts[1]
    assert result == _conforming()
    meta = _meta(gen_env)
    assert meta["conformance"] == "retried" and meta["first_violations"]


@pytest.mark.asyncio
async def test_generation_falls_back_to_routine_template(gen_env, monkeypatch):
    _llm_schedule(monkeypatch, _INCIDENT_SCHEDULE, _INCIDENT_SCHEDULE)
    result = await schedule_mod.generate_daily_schedule(
        "agent-1", "阿恺", None, life_overview="x", date=_at(_FRIDAY, "03:30"))
    assert result == template_schedule(day_plan(_routine(), _FRIDAY))
    assert _meta(gen_env)["conformance"] == "template"


@pytest.mark.asyncio
async def test_generation_without_routine_keeps_old_behaviour(gen_env, monkeypatch):
    monkeypatch.setattr(routine_mod, "ensure_agent_routine", AsyncMock(return_value=None))
    legacy = [{k: v for k, v in s.items() if k != "kind"} for s in _INCIDENT_SCHEDULE]
    prompts = _llm_schedule(monkeypatch, legacy)
    result = await schedule_mod.generate_daily_schedule(
        "agent-1", "阿恺", None, life_overview="x", date=_at(_FRIDAY, "03:30"))
    assert "【固定作息" not in prompts[0] and result == legacy
    assert _meta(gen_env)["routine"] == "unavailable"


@pytest.mark.asyncio
async def test_legacy_schedule_refresh_once_and_never_over_adjustment(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setattr(schedule_mod, "get_redis", AsyncMock(return_value=redis))
    monkeypatch.setattr(routine_mod, "get_agent_routine", AsyncMock(return_value=_routine()))
    legacy = [{k: v for k, v in s.items() if k != "kind"} for s in _INCIDENT_SCHEDULE]
    a, b, c = (SimpleNamespace(id=x) for x in "abc")
    assert await schedule_mod.needs_routine_refresh(a, legacy) is True
    assert await schedule_mod.needs_routine_refresh(a, legacy) is False
    assert await schedule_mod.needs_routine_refresh(b, _INCIDENT_SCHEDULE) is False
    redis._store[schedule_mod._adj_key("c")] = "1"
    assert await schedule_mod.needs_routine_refresh(c, legacy) is False


@pytest.mark.asyncio
async def test_current_status_with_routine(monkeypatch):
    monkeypatch.setattr(routine_mod, "get_agent_routine", AsyncMock(return_value=_routine()))
    status = await schedule_mod.current_status_with_routine("a", _INCIDENT_SCHEDULE, _at(_FRIDAY, "19:55"))
    assert status["activity"] == "步行回住所" and status["routine"]["phase"] == "after_work"
    monkeypatch.setattr(routine_mod, "get_agent_routine", AsyncMock(side_effect=RuntimeError("down")))
    status = await schedule_mod.current_status_with_routine("a", _INCIDENT_SCHEDULE, _at(_FRIDAY, "19:55"))
    assert "routine" not in status


def test_short_work_on_rest_day_and_flexible_jobs_are_allowed():
    sunday = day_plan(_routine(), _SUNDAY)
    schedule = template_schedule(sunday)
    schedule.insert(2, {"start": "14:00", "end": "15:00", "event": "整理下周排班", "status": "忙碌", "kind": "工作"})
    assert check_conformance(schedule, sunday) == []
    # 休息日即使作息里有零碎工作, "今天"仍说不上班
    view = describe_now(_routine(), schedule, _at(_SUNDAY, "14:30"))
    assert view["phase"] == "day_off"
    flexible = day_plan(_routine(work_start=None, work_end=None), _FRIDAY)
    assert check_conformance(_INCIDENT_SCHEDULE, flexible) == []


# ── 审查反例 (2026-11-02 第一轮 review) ───────────────────────────────

@pytest.mark.parametrize("line, expected", [
    ("深夜两点才睡", {"sleep": "次日 02:00"}),
    ("晚上1点睡觉", {"sleep": "次日 01:00"}),
    ("上午8点到11点上班", {"work_start": "08:00", "work_end": "11:00"}),
    ("上午9点到11点上课，下午2点到5点上课", {"work_start": "09:00", "work_end": "17:00"}),
    ("早上6点到10点出摊", {"work_start": "06:00", "work_end": "10:00"}),
    ("晚上6点到10点在市集出摊", {"work_start": "18:00", "work_end": "22:00"}),
    ("早一点睡对身体好", {}),
    ("今天想晚一点下班", {}),
    ("一点也不想起床", {}),
    ("两点一线的上班生活", {}),
    ("中午12点午睡一会儿", {}),
    ("下午3点左右会休息半小时，吃点水果", {}),
    ("每天8点提醒自己喝水", {}),
    # 盲测补充: 说法多样的作息, 以及讲某一次 / 不固定的分句
    ("朝九晚五，偶尔加班", {"work_start": "09:00", "work_end": "17:00"}),
    ("公司早十晚七", {"work_start": "10:00", "work_end": "19:00"}),
    ("花店早八点开门，晚六点关门", {"work_start": "08:00", "work_end": "18:00"}),
    ("晚上8点开播，直播到晚上11点", {"work_start": "20:00", "work_end": "23:00"}),
    ("白班早上8点到下午4点", {"work_start": "08:00", "work_end": "16:00"}),
    ("上午9点上班，中午11点半到下午2点午休，下午5点半下班", {"work_start": "09:00", "work_end": "17:30"}),
    ("白天睡到下午两点才起", {"wake": "14:00"}),
    ("有一次通宵到凌晨四点才睡，平时晚上十点半睡", {"sleep": "22:30"}),
    ("没有固定作息，偶尔晚上十一点前会睡", {}),
    ("五点四十起床", {"wake": "05:40"}),
    ("每天早上9点起营业到晚上10点", {"work_start": "09:00", "work_end": "22:00"}),
    ("每天7点起床偶尔赖床", {"wake": "07:00"}),
    ("早上7点起床（偶尔会睡到8点）", {"wake": "07:00"}),
])
def test_review_counterexamples(line, expected):
    assert {k: fmt_clock(v) for k, v in parse_routine_lines([line]).values.items()} == expected


@pytest.mark.parametrize("line, rest", [
    ("周六周日休息", {5, 6}),
    ("周六、周日不上班", {5, 6}),
    ("星期六和星期天放假", {5, 6}),
    ("周一不休息，周二休息", {1}),
    ("每周日休息", {6}),
    ("周三从不休息", None),
])
def test_rest_day_phrasings(line, rest):
    assert parse_routine_lines([line]).rest_weekdays == rest


def test_validation_requires_wake_before_work_and_sleep_after():
    from app.services.schedule_domain.routine import validate_routine_values
    base = {"wake": 420, "work_start": 540, "work_end": 1080, "sleep": 1380}
    assert validate_routine_values(base, set()) is None
    assert validate_routine_values({**base, "wake": 600}, set()) == "work_before_wake"
    night = {"wake": 600, "work_start": 18 * 60, "work_end": DAY + 120, "sleep": DAY + 30}
    assert validate_routine_values(night, set()) == "sleep_before_work_end"


def test_late_start_template_has_short_breakfast():
    plan = day_plan(_routine(wake=11 * 60, work_start=18 * 60, work_end=DAY + 60, sleep=DAY + 120), _FRIDAY)
    schedule = template_schedule(plan)
    breakfast = [s for s in schedule if s["event"] == "吃早饭"]
    assert breakfast and (breakfast[0]["start"], breakfast[0]["end"]) == ("11:30", "12:10")
    assert check_conformance(schedule, plan) == []


def test_user_break_inside_work_hours_is_not_called_working():
    schedule = template_schedule(day_plan(_routine(), _FRIDAY))
    adjusted = schedule_mod._overlay_window(schedule, 15 * 60, 16 * 60, dict(schedule_mod._AWAKE_FILL))
    view = describe_now(_routine(), adjusted, _at(_FRIDAY, "15:30"))
    assert view["phase"] == "work_break" and view["now_line"].endswith("21:00下班")


@pytest.mark.asyncio
async def test_refresh_needs_a_routine_and_tries_derivation_once_a_day(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setattr(schedule_mod, "get_redis", AsyncMock(return_value=redis))
    monkeypatch.setattr(routine_mod, "get_agent_routine", AsyncMock(return_value=None))
    derive = AsyncMock(return_value=None)
    monkeypatch.setattr(routine_mod, "ensure_agent_routine", derive)
    legacy = [{k: v for k, v in s.items() if k != "kind"} for s in _INCIDENT_SCHEDULE]
    agent = SimpleNamespace(id="x")
    assert await schedule_mod.needs_routine_refresh(agent, legacy) is False
    assert await schedule_mod.needs_routine_refresh(agent, legacy) is False
    assert derive.await_count == 1


@pytest.mark.asyncio
async def test_refresh_does_not_overwrite_adjustment_made_during_generation(gen_env, monkeypatch):
    async def invoke(model, prompt):
        gen_env.redis._store[schedule_mod._adj_key("agent-1")] = "1"  # 用户此刻要求了调整
        return _conforming()
    monkeypatch.setattr(schedule_mod, "invoke_json", invoke)
    monkeypatch.setattr(schedule_mod, "get_utility_model", lambda: None)
    await schedule_mod.generate_daily_schedule(
        "agent-1", "阿恺", None, life_overview="x", date=_at(_FRIDAY, "10:00"), refresh=True)
    assert not any(k.startswith("schedule:agent-1:") for k in gen_env.redis._store)


def test_schedule_query_context_is_stable_across_minutes():
    a = {"status": "busy", "activity": "上班", "routine": describe_now(_routine(), None, _at(_FRIDAY, "19:55"))}
    b = {**a, "routine": describe_now(_routine(), None, _at(_FRIDAY, "19:56"))}
    one = schedule_mod.format_full_schedule_for_query(_INCIDENT_SCHEDULE, "current", a)
    two = schedule_mod.format_full_schedule_for_query(_INCIDENT_SCHEDULE, "current", b)
    assert one == two and "你平时的作息：" in one and "离下班" not in one
