"""话题连续性状态 (《主动聊天机制（新增）》): 会话边界 / B 名额 / 被动承接决策表."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from app.services.interaction import topic_continuity as tc
from app.services.interaction.topic_continuity import (
    ContinuityState,
    TopicVerdict,
    clean_single_line,
    is_dialogue_noise,
    render_dialogue,
    resolve_cue,
)

UTC = timezone.utc
NOW = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)


class FakeRedis:
    """只实现本模块用到的 hash 命令."""

    def __init__(self):
        self.store: dict[str, dict[str, str]] = {}
        self.fail = False

    async def hgetall(self, key):
        if self.fail:
            raise ConnectionError("redis down")
        return dict(self.store.get(key, {}))

    def pipeline(self):
        return _Pipe(self)

    async def hdel(self, key, *fields):
        if self.fail:
            raise ConnectionError("redis down")
        for f in fields:
            self.store.get(key, {}).pop(f, None)


class _Pipe:
    def __init__(self, redis: FakeRedis):
        self.redis = redis
        self.ops: list = []

    def hset(self, key, mapping):
        self.ops.append(("hset", key, mapping))

    def hdel(self, key, *fields):
        self.ops.append(("hdel", key, fields))

    def expire(self, key, ttl):
        self.ops.append(("expire", key, ttl))

    async def execute(self):
        if self.redis.fail:
            raise ConnectionError("redis down")
        for op, key, arg in self.ops:
            bucket = self.redis.store.setdefault(key, {})
            if op == "hset":
                bucket.update(arg)
            elif op == "hdel":
                for f in arg:
                    bucket.pop(f, None)


@pytest.fixture
def redis(monkeypatch):
    fake = FakeRedis()

    async def _get():
        return fake

    monkeypatch.setattr(tc, "get_redis", _get)
    return fake


_BASE_STATE = ContinuityState(
    verdict=TopicVerdict("unfinished", "llm"),
    anchor_message_id="ai-1",
    followup_sent_at=None,
    last_user_at=None,
    session_closed=False,
)


def _state(**kw) -> ContinuityState:
    return replace(_BASE_STATE, **kw)


# ── resolve_cue 决策表 ────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("gap_min", "followup_used", "return_line"),
    [
        (6, False, False),     # <10min: 只考虑跳话题过渡
        (10, False, False),    # spec 是"大于 10 分钟", 恰好 10 分钟不算
        (11, False, True),     # >10min 且没追问过: 先说一句承接短句
        (11, True, False),     # 追问过: 不带承接话术, 只保留跳话题过渡
        (179, False, True),
    ],
)
def test_unfinished_cue_matrix(gap_min, followup_used, return_line):
    state = _state(followup_sent_at=NOW if followup_used else None)
    cue = resolve_cue(state, previous_assistant_id="ai-1", gap_seconds=gap_min * 60)
    assert cue is not None and cue.unfinished is True
    assert cue.return_line is return_line


def test_finished_topic_only_suppresses():
    """spec: 已完结话题无论隔多久都不带承接 → 什么都不补, 但要压掉重逢短档."""
    state = _state(verdict=TopicVerdict("finished", "llm"))
    cue = resolve_cue(state, previous_assistant_id="ai-1", gap_seconds=45 * 60)
    assert cue is not None and cue.unfinished is False and cue.return_line is False


def test_three_hours_or_more_hands_over_to_reengagement():
    cue = resolve_cue(_state(), previous_assistant_id="ai-1", gap_seconds=3 * 3600)
    assert cue is None


def test_stale_anchor_is_ignored():
    """AI 在判定之后又说过话 (anchor 对不上) → 结论作废."""
    assert resolve_cue(_state(), previous_assistant_id="ai-2", gap_seconds=900) is None
    assert resolve_cue(_state(), previous_assistant_id=None, gap_seconds=900) is None


def test_no_verdict_or_unknown_state_keeps_old_behaviour():
    assert resolve_cue(None, previous_assistant_id="ai-1", gap_seconds=900) is None
    assert resolve_cue(_state(verdict=None), previous_assistant_id="ai-1", gap_seconds=900) is None
    assert resolve_cue(_state(), previous_assistant_id="ai-1", gap_seconds=None) is None


def test_replying_to_followup_suppresses_return_phrase():
    """对方在回 B 追问: 追问已经接住了旧话题, 不再叠承接短句."""
    cue = resolve_cue(
        None, previous_assistant_id="b-1", gap_seconds=40 * 60,
        previous_assistant_is_followup=True,
    )
    assert cue is not None and cue.unfinished is False


# ── 对话渲染 / 单句清洗 ───────────────────────────────────────────────

def test_render_dialogue_uses_utc8_stamps():
    text = render_dialogue([
        ("user", "我明天面试", datetime(2026, 9, 29, 6, 0, tzinfo=UTC)),
        ("assistant", "什么岗位呀？", datetime(2026, 9, 29, 6, 1, tzinfo=UTC)),
        ("user", "没时间戳的", None),
    ])
    assert text.splitlines() == [
        "[09-29 14:00] 用户: 我明天面试",
        "[09-29 14:01] AI: 什么岗位呀？",
        "用户: 没时间戳的",
    ]


def test_dialogue_noise():
    assert is_dialogue_noise({"kind": "game_status"})
    assert is_dialogue_noise({"offering_received": True})
    assert not is_dialogue_noise({"proactive": True})
    assert not is_dialogue_noise(None)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("SKIP。", None),
        ("忙完啦||刚说到哪了", "忙完啦，刚说到哪了"),
        ("“回来啦～”", "回来啦～"),
        ("怎么才回呀", None),        # 提示词禁止: 怎么才回
        ("还以为你消失很久了", None),  # 提示词禁止: 消失很久
        ("等你好久啦", None),          # 提示词禁止: 等你好久
        ("嗯", None),                 # 太短
    ],
)
def test_clean_single_line(raw, expected):
    assert clean_single_line(raw) == expected


@pytest.mark.parametrize(
    ("raw", "absent", "present"),
    [
        # 查岗口吻: 对方还没回来 (B 追问) 时冒犯, 对方已经回来 (回归承接) 时正是自然承接
        ("忙完了吗？刚说到面试那儿", None, "忙完了吗？刚说到面试那儿"),
        ("是不是去忙啦～", None, "是不是去忙啦～"),
        # 催促 / 抱怨 / 质问: 任何时候都不说
        ("怎么才回呀", None, None),
        ("等你好久啦", None, None),
        ("你怎么不回我", None, None),
    ],
)
def test_check_in_phrasing_is_fine_once_user_is_back(raw, absent, present):
    assert clean_single_line(raw) == absent
    assert clean_single_line(raw, user_present=True) == present


# ── 会话边界 / B 名额 ────────────────────────────────────────────────

async def test_followup_budget_survives_user_replies_within_session(redis):
    await tc.note_user_message("c1", now=NOW)
    await tc.reserve_followup("c1", now=NOW + timedelta(minutes=5))
    # 用户 20 分钟后回来, 同一会话: B 名额仍是已用
    await tc.note_user_message("c1", now=NOW + timedelta(minutes=25))
    state = await tc.load_continuity("c1")
    assert state.followup_available is False


async def test_long_gap_starts_new_session_and_restores_followup(redis):
    await tc.note_user_message("c1", now=NOW)
    await tc.reserve_followup("c1", now=NOW + timedelta(minutes=5))
    await tc.note_user_message("c1", now=NOW + timedelta(hours=3, minutes=1))
    state = await tc.load_continuity("c1")
    assert state.followup_available is True


async def test_closed_session_rolls_over_on_next_user_message(redis):
    """A 模式新开场 / 告别之后, 用户的下一条消息开新会话."""
    await tc.note_user_message("c1", now=NOW)
    await tc.reserve_followup("c1", now=NOW)
    await tc.close_session("c1")
    await tc.note_user_message("c1", now=NOW + timedelta(minutes=2))
    state = await tc.load_continuity("c1")
    assert state.followup_available is True
    assert state.session_closed is False


async def test_session_rollover_keeps_verdict(redis):
    """「晚安」隔两小时回来: 新会话, 但"上一轮已完结"仍要能压掉重逢寒暄."""
    await tc.note_user_message("c1", now=NOW)
    await tc.record_verdict(
        "c1", TopicVerdict("finished", "farewell"), anchor_message_id="ai-9", now=NOW,
    )
    await tc.close_session("c1")
    await tc.note_user_message("c1", now=NOW + timedelta(hours=2))
    state = await tc.load_continuity("c1")
    assert state.verdict == TopicVerdict("finished", "farewell")
    assert state.anchor_message_id == "ai-9"


async def test_redis_failure_reads_as_unknown_not_empty(redis):
    """Redis 挂了 = 不知道 B 用没用过, 调用方必须能区分 (不能当成"没用过")."""
    redis.fail = True
    assert await tc.load_continuity("c1") is None
    await tc.note_user_message("c1", now=NOW)  # 不抛


async def test_reserve_and_release_followup(redis):
    assert await tc.reserve_followup("c1", now=NOW) is True
    assert (await tc.load_continuity("c1")).followup_available is False
    await tc.release_followup("c1")  # 没发出去, 名额还回去
    assert (await tc.load_continuity("c1")).followup_available is True

    redis.fail = True
    assert await tc.reserve_followup("c1", now=NOW) is False  # 记不上账就别发


async def test_verdict_roundtrip(redis):
    verdict = TopicVerdict("unfinished", "llm")
    await tc.record_verdict("c1", verdict, anchor_message_id="ai-3", now=NOW)
    state = await tc.load_continuity("c1")
    assert state.verdict == verdict
    assert state.anchor_message_id == "ai-3"
