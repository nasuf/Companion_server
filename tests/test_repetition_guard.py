"""Not restating the previous reply: acknowledgements, overlapping turns, bubble backstop."""
from __future__ import annotations

import pytest

from app.services.chat.repetition_guard import (
    drop_restated_bubbles,
    is_acknowledgement,
    previous_ai_turn,
    previous_reply_for_prompt,
    previous_reply_situation,
)


def _row(id_, role, content, at):
    return {"id": id_, "role": role, "content": content, "createdAt": f"2026-11-02T09:{at}+00:00"}


SUNSET = [
    _row("u0", "user", "还没发现，不过有的时候抬头看天就会发现日落很好看", "51:59"),
    _row("a1", "assistant", "对啊 抬头撞见好看的日落👌", "52:10"),
    _row("a2", "assistant", "那种感觉特别惊喜吧", "52:10"),
    _row("a3", "assistant", "像突然收到小礼物似的", "52:10"),
    _row("u1", "user", "是呀", "52:21"),
]


@pytest.mark.parametrize("text, ack", [
    ("是呀", True), ("对啊！", True), ("嗯嗯", True), ("好的~", True), ("哈哈哈", True), ("确实", True),
    ("是吗", False), ("好看", False), ("喜欢这种感觉", False), ("", False),
    # 带问号是没听懂, 不是附和
    ("啊？", False), ("嗯?", False), ("是吧？", False),
])
def test_acknowledgement(text, ack):
    assert is_acknowledgement(text) is ack


def test_acknowledgement_gets_the_previous_reply():
    situation, previous = previous_reply_situation(SUNSET, {"u1"}, "是呀")
    assert situation == "ack"
    assert previous == "对啊 抬头撞见好看的日落👌，那种感觉特别惊喜吧，像突然收到小礼物似的"
    # 有新内容的消息不注入
    assert previous_reply_situation(SUNSET, {"u1"}, "我也想去转塘看看") is None


def test_ordinary_turns_quote_the_previous_reply_in_the_prompt_only():
    # 对方有新内容: 重写与兜底的判定仍不管 (None), 提示段照样引用上一轮
    assert previous_reply_situation(SUNSET, {"u1"}, "我也想去转塘看看") is None
    situation, previous = previous_reply_for_prompt(SUNSET, {"u1"}, "我也想去转塘看看")
    assert situation == "other" and previous.startswith("对啊 抬头撞见好看的日落")
    # 附和 / 交叠沿用原来的判定; 没有上一轮 AI 回复时不注入
    assert previous_reply_for_prompt(SUNSET, {"u1"}, "是呀")[0] == "ack"
    rows = [_row("a1", "assistant", "早呀", "40:00"), _row("u0", "user", "早", "41:00"),
            _row("u1", "user", "今天好冷", "42:00")]
    assert previous_reply_for_prompt(rows, {"u1"}, "今天好冷") is None


def test_reply_after_the_current_message_is_an_overlap():
    rows = [
        _row("u1", "user", "转塘是哪里啊", "51:16"),
        _row("u2", "user", "不过听起来是看日落的好地方", "51:20"),
        _row("a1", "assistant", "是杭州西湖区的一个街道呀", "51:29"),
        _row("a2", "assistant", "有好多好看的日落点", "51:29"),
    ]
    bubbles, after = previous_ai_turn(rows, {"u2"})
    assert bubbles == ["是杭州西湖区的一个街道呀", "有好多好看的日落点"] and after
    assert previous_reply_situation(rows, {"u2"}, "不过听起来是看日落的好地方")[0] == "overlap"


def test_no_previous_turn_when_a_user_message_sits_between():
    rows = [_row("a1", "assistant", "早呀", "40:00"), _row("u0", "user", "早", "41:00"),
            _row("u1", "user", "是呀", "42:00")]
    assert previous_ai_turn(rows, {"u1"}) == ([], False)


def test_stale_ai_messages_are_not_the_previous_turn():
    rows = [
        _row("a0", "assistant", "晚安呀 明天见", "00:00"),
        _row("a1", "assistant", "在忙吗", "30:00"),        # 半小时后的主动消息, 不同一轮
        _row("a2", "assistant", "最近天气变冷了", "31:00"),
        _row("u1", "user", "嗯嗯", "32:00"),
    ]
    assert previous_ai_turn(rows, {"u1"})[0] == ["在忙吗", "最近天气变冷了"]
    # 没有文字的用户行 (卡片) 也是边界
    rows = [_row("a0", "assistant", "早呀", "00:00"), _row("c1", "user", "", "01:00"),
            _row("a1", "assistant", "收到啦", "02:00"), _row("u1", "user", "嗯", "03:00")]
    assert previous_ai_turn(rows, {"u1"})[0] == ["收到啦"]


def test_long_previous_reply_keeps_the_most_recent_part():
    rows = [_row("a1", "assistant", "前面" * 80, "00:00"), _row("a2", "assistant", "最后一句", "00:01"),
            _row("u1", "user", "是呀", "00:30")]
    situation, previous = previous_reply_situation(rows, {"u1"}, "是呀")
    assert previous.startswith("…") and previous.endswith("最后一句")


def test_restated_bubbles_are_dropped_but_never_all():
    previous = ["那你要是喝到好喝的 记得跟我说一声啊", "我也想试试新口味"]
    kept, dropped, all_restated = drop_restated_bubbles(
        ["哈哈 那说定了", "那你要是喝到好喝的 一定要跟我说啊"], previous, "好")
    assert kept == ["哈哈 那说定了"] and dropped == 1 and not all_restated
    # 全是复述: 不发空回复, 报告出来
    kept, dropped, all_restated = drop_restated_bubbles(
        ["那你要是喝到好喝的 一定要跟我说啊"], previous, "好")
    assert kept == ["那你要是喝到好喝的 一定要跟我说啊"] and dropped == 0 and all_restated
    # 短气泡、对方让你再说一遍或在提问时不判
    assert drop_restated_bubbles(["对啊"], ["对啊 是这样"], "")[1] == 0
    assert drop_restated_bubbles(["那你要是喝到好喝的 一定要跟我说啊"], previous, "你刚才说啥")[1] == 0
    assert drop_restated_bubbles(["在杭州西湖区转塘街道呀", "你没看到吗"], ["我在杭州西湖区转塘街道"],
                                 "你说你在哪来着")[1] == 0
    # 删完剩下的开头是连接词, 读不通, 不删
    kept, dropped, _ = drop_restated_bubbles(
        ["那你要是喝到好喝的 一定要跟我说啊", "而且要拍照给我看"], previous, "嗯")
    assert dropped == 0 and len(kept) == 2


async def test_previous_reply_section_sits_after_the_current_message():
    from app.services.chat import prompt_builder as pb
    from tests.test_prompt_layout_cache import _system_prompt

    prompt = await _system_prompt(previous_reply=("ack", "对啊 抬头撞见好看的日落"))
    messages = pb.build_chat_messages(prompt, SUNSET)
    assert messages[-2]["role"] == "user" and messages[-2]["content"].endswith("是呀")
    tail = messages[-1]["content"]
    assert "## 刚说过的话" in tail and "「对啊 抬头撞见好看的日落」" in tail

    overlap = await _system_prompt(previous_reply=("overlap", "有好多好看的日落点"))
    _, turn = pb.split_system_prompt(overlap)
    assert "## 上一轮已经说过" in turn and "## 刚说过的话" not in turn
    assert "## 刚说过的话" not in await _system_prompt(previous_reply=None)
    other = await _system_prompt(previous_reply=("other", "对啊 抬头撞见好看的日落"))
    _, turn = pb.split_system_prompt(other)
    assert "## 上一轮说过的话" in turn and "上一轮问过、对方没接的问题也别再问" in turn
    assert "## 刚说过的话" not in turn and "## 上一轮已经说过" not in turn
    # 和附和 / 交叠两段一样放在当前消息之后、紧贴生成 (放在消息之前时回放效果不到一半)
    placed = pb.build_chat_messages(other, SUNSET)
    assert placed[-2]["role"] == "user" and "## 上一轮说过的话" in placed[-1]["content"]


class _Ctx:
    def __init__(self, replies, messages, user_message="嘿嘿"):
        self.replies = replies
        self.reply_is_fallback = False
        self.contradiction_inquiry = None
        self.crisis_care_turn = False
        self.reply_emotion_pre = None
        self.messages_dicts = messages
        self.current_turn_ids = {"u1"}
        self.user_message = user_message
        self.response_diagnostics = {}

        async def build():
            return [{"role": "system", "content": "base"}, {"role": "user", "content": user_message}]

        self._build_main_chat_messages = build


SNOW = [
    _row("u1", "user", "嘿嘿", "02:44"),
    _row("a1", "assistant", "嫩江居然都下雪了啊", "02:48"),
    _row("a2", "assistant", "我这边还二十度左右呢", "02:48"),
    _row("a3", "assistant", "温差也太大了", "02:48"),
]


async def test_fully_restated_draft_is_rewritten_once(monkeypatch):
    from app.services.chat import graph_phases, repetition_guard

    calls = []

    async def fake_generate(**kwargs):
        messages = await kwargs["chat_messages_factory"]()
        calls.append(messages)
        return ["嫩江的雪看着好有氛围", "你那边下雪会冷吗"], "", False, None

    monkeypatch.setattr(repetition_guard, "restated_retry_note",
                        lambda previous: _async("重写这一轮"))
    monkeypatch.setattr(graph_phases, "restated_retry_note", repetition_guard.restated_retry_note)
    services = type("S", (), {"_generate_reply": staticmethod(fake_generate)})
    ctx = _Ctx(["嫩江都下雪了", "我这边还二十度左右呢", "温差也太大了"], SNOW)
    await graph_phases._rewrite_if_restated(ctx, services, {"chat_messages_factory": None})
    assert ctx.replies == ["嫩江的雪看着好有氛围", "你那边下雪会冷吗"]
    assert ctx.response_diagnostics["restated_retry"] == "rewritten"
    assert calls[0][-1] == {"role": "system", "content": "重写这一轮"}

    # 不是复述就不重写
    calls.clear()
    ctx = _Ctx(["哈哈 你那边好冷啊"], SNOW)
    await graph_phases._rewrite_if_restated(ctx, services, {"chat_messages_factory": None})
    assert not calls and "restated_retry" not in ctx.response_diagnostics


async def _async(value):
    return value


async def test_retry_outcomes_disabled_and_still_restated(monkeypatch):
    from app.services.chat import graph_phases

    draft = ["嫩江都下雪了", "我这边还二十度左右呢", "温差也太大了"]
    # 后台停用重写提示: 不重写, 记录下来
    monkeypatch.setattr(graph_phases, "restated_retry_note", lambda previous: _async(None))
    ctx = _Ctx(list(draft), SNOW)
    await graph_phases._rewrite_if_restated(ctx, object(), {})
    assert ctx.replies == draft and ctx.response_diagnostics["restated_retry"] == "disabled"

    # 重写后还是复述: 保留第一版
    async def same_again(**kwargs):
        return list(draft), "", False, None

    monkeypatch.setattr(graph_phases, "restated_retry_note", lambda previous: _async("重写"))
    services = type("S", (), {"_generate_reply": staticmethod(same_again)})
    ctx = _Ctx(list(draft), SNOW)
    await graph_phases._rewrite_if_restated(ctx, services, {})
    assert ctx.replies == draft and ctx.response_diagnostics["restated_retry"] == "still_restated"


async def test_retry_note_quotes_the_recent_end_of_the_previous_turn(monkeypatch):
    from app.services.chat import prompt_builder, repetition_guard

    monkeypatch.setattr(prompt_builder, "_get_optional_prompt", lambda key: _async("上一轮：「{previous}」"))
    note = await repetition_guard.restated_retry_note(["前面" * 80, "最后一句"])
    assert note.startswith("上一轮：「…") and note.endswith("最后一句」")
    monkeypatch.setattr(prompt_builder, "_get_optional_prompt", lambda key: _async(None))
    assert await repetition_guard.restated_retry_note(["随便"]) is None


def test_backstop_keeps_the_continuation_line_and_records_diagnostics():
    from app.services.chat import graph_phases

    previous = [_row("u0", "user", "我明天面试", "00:00"),
                _row("a1", "assistant", "你那个面试准备得怎么样了", "01:00"),
                _row("a2", "assistant", "那你要是喝到好喝的 记得跟我说一声啊", "01:00"),
                _row("u1", "user", "好", "02:00")]
    ctx = _Ctx(["对了 面试准备得怎么样了呀", "哈哈 那说定了", "那你要是喝到好喝的 一定要跟我说啊"],
               previous, user_message="好")
    ctx.continuation_lines = ["对了 面试准备得怎么样了呀"]
    ctx.reply_emotion_pre = {"emotion": "高兴"}
    graph_phases._drop_restated_bubbles(ctx)
    assert ctx.replies == ["对了 面试准备得怎么样了呀", "哈哈 那说定了"]
    assert ctx.reply_emotion_pre is None
    diag = ctx.response_diagnostics["restated_bubbles"]
    assert diag["situation"] == "ack" and diag["dropped"] == 1 and diag["previous_bubbles"] == 2

    # 对方说了新内容: 只拦几乎原样的句子, 换个说法的不动
    ctx = _Ctx(["那你要是喝到好喝的 一定要跟我说啊"], previous, user_message="我明天去买")
    ctx.continuation_lines = []
    graph_phases._drop_restated_bubbles(ctx)
    assert ctx.replies == ["那你要是喝到好喝的 一定要跟我说啊"]
    assert ctx.response_diagnostics["restated_bubbles"]["situation"] == "other"


def test_ordinary_turn_drops_a_bubble_copied_from_the_previous_turn():
    """2026-10-30 生产: 用户说「现在又没啥朋友」, 回复又原样带上上一轮的「我刚还在跟导演走位呢」。"""
    from app.services.chat import graph_phases

    rows = [_row("u0", "user", "唉", "26:50"),
            _row("a1", "assistant", "唉 那你这一年是不是都挺难受的啊", "27:05"),
            _row("a2", "assistant", "我刚还在跟导演走位呢", "27:05"),
            _row("a3", "assistant", "都没敢一直摸鱼回你", "27:05"),
            _row("u1", "user", "现在又没啥朋友", "27:37")]
    ctx = _Ctx(["那也太孤单了吧", "我刚还在跟导演走位呢", "都没敢一直摸鱼回你"], rows,
               user_message="现在又没啥朋友")
    ctx.continuation_lines = []
    graph_phases._drop_restated_bubbles(ctx)
    assert ctx.replies == ["那也太孤单了吧"]
    assert ctx.response_diagnostics["restated_bubbles"]["situation"] == "other"
    # 对方在问: 再答一遍是对的
    ctx = _Ctx(["我刚还在跟导演走位呢"], rows, user_message="你在干嘛")
    ctx.continuation_lines = []
    graph_phases._drop_restated_bubbles(ctx)
    assert ctx.replies == ["我刚还在跟导演走位呢"]


async def test_short_delays_get_no_delay_explanation_section():
    from tests.test_prompt_layout_cache import _system_prompt

    delay = {"received_at": "2026-10-30T11:27:37", "activity": "化妆间准备演出", "status": "busy",
             "delay_reason": ""}
    assert "## 回复时机说明" not in await _system_prompt(delay_context={**delay, "delay_seconds": 6})
    assert "## 回复时机说明" in await _system_prompt(delay_context={**delay, "delay_seconds": 35})


def test_empty_assistant_rows_are_skipped():
    rows = [_row("a1", "assistant", "收到啦", "00:00"), _row("a2", "assistant", "", "00:01"),
            _row("u1", "user", "嗯", "01:00")]
    assert previous_ai_turn(rows, {"u1"})[0] == ["收到啦"]
