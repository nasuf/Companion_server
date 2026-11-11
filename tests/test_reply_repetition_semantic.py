"""CHAT01.10 A 方案: 语义复读检查 —— 换说法的重复气泡发出前删掉, 开关 + 灰度, 超时放行。"""
import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.services.chat.reply_repetition import semantic
from app.services.chat.reply_repetition.policy import resolve

NOW = datetime.now(timezone.utc)
ON = resolve({"semantic_enabled": True, "semantic_rollout_pct": 100})
OFF = resolve({})


def _row(id_, role, content, minutes_ago, **meta):
    return {"id": id_, "role": role, "content": content, "metadata": meta or None,
            "createdAt": (NOW - timedelta(minutes=minutes_ago)).isoformat()}


HISTORY = [
    _row("u0", "user", "我今天加班到好晚", 30),
    _row("a1", "assistant", "辛苦啦，加班到这么晚肯定累坏了", 29),
    _row("a2", "assistant", "晚饭有没有好好吃呀", 29),
    _row("u1", "user", "吃了点面", 5),
]

# 合成向量: 同一组的句子方向相同 (相似度 1), 不同组正交 (相似度 0)
GROUPS = {"累": [1.0, 0.0, 0.0], "饭": [0.0, 1.0, 0.0]}


async def fake_embed(text):
    await asyncio.sleep(0)
    for key, vector in GROUPS.items():
        if key in text:
            return vector
    return [0.0, 0.0, 1.0]


def _ctx(replies, *, user="吃了点面", history=HISTORY, user_id="u-1", **over):
    return SimpleNamespace(user_id=user_id, user_message=user, messages_dicts=list(history),
                           current_turn_ids={"u1"}, replies=list(replies), reply_is_fallback=False,
                           contradiction_inquiry=None, crisis_care_turn=False, response_diagnostics={},
                           continuation_lines=[], reply_emotion_pre={"emotion": "高兴"},
                           semantic_repetition_task=None, **over)


@pytest.fixture(autouse=True)
def embeddings(monkeypatch):
    monkeypatch.setattr(semantic, "_embed", fake_embed)


async def _run(ctx, policy=ON):
    semantic.start(ctx, policy)
    await semantic.apply(ctx, policy)
    return ctx.response_diagnostics.get("reply_repetition", {}).get("semantic", {})


async def test_a_paraphrased_bubble_is_dropped_and_recorded():
    # 「今天真是累坏你了」和上一轮「加班到这么晚肯定累坏了」意思相同, 删; 新内容保留
    ctx = _ctx(["面也挺好的，暖胃", "今天真是累坏你了呀"])
    record = await _run(ctx)
    assert ctx.replies == ["面也挺好的，暖胃"]
    assert record["status"] == "dropped" and record["dropped"] == 1
    assert record["dropped_texts"] == ["今天真是累坏你了呀"]
    assert record["max_similarity"] == 1.0 and record["threshold"] == 0.9
    assert ctx.reply_emotion_pre is None  # 情绪要跟实际发出的文字走


async def test_off_by_default_and_outside_the_rollout():
    ctx = _ctx(["面也挺好的，暖胃", "今天真是累坏你了呀"])
    assert await _run(ctx, OFF) == {}
    assert len(ctx.replies) == 2
    partial = resolve({"semantic_enabled": True, "semantic_rollout_pct": 1})
    outside = next(f"user-{i}" for i in range(500) if not semantic.in_rollout(f"user-{i}", 1))
    ctx = _ctx(["今天真是累坏你了呀", "面也挺好的"], user_id=outside)
    assert (await _run(ctx, partial))["status"] == "not_in_rollout" and len(ctx.replies) == 2


def test_rollout_buckets_are_stable_and_bounded():
    users = [f"user-{i}" for i in range(2000)]
    share = sum(semantic.in_rollout(u, 20) for u in users) / len(users)
    assert 0.15 < share < 0.25
    assert all(semantic.in_rollout(u, 20) == semantic.in_rollout(u, 20) for u in users[:50])
    assert not semantic.in_rollout("x", 0) and semantic.in_rollout("x", 100) and not semantic.in_rollout(None, 50)


@pytest.mark.parametrize("user", ["你刚才说什么来着", "你觉得我累吗？"])
async def test_never_judged_when_the_user_asks_or_wants_it_again(user):
    ctx = _ctx(["今天真是累坏你了呀", "面也挺好的"], user=user)
    assert (await _run(ctx))["status"] == "user_asked" and len(ctx.replies) == 2


async def test_never_empties_or_breaks_the_reply():
    ctx = _ctx(["今天真是累坏你了呀", "晚饭吃得还行吧这次"])
    assert (await _run(ctx))["status"] == "all_similar" and len(ctx.replies) == 2
    # 删完只剩问句 / 连接词开头: 读不通, 不删
    ctx = _ctx(["今天真是累坏你了呀", "那你明天还要早起吗"])
    assert (await _run(ctx))["status"] == "would_break_reply" and len(ctx.replies) == 2
    ctx = _ctx(["今天真是累坏你了呀", "而且面也挺暖胃的"])
    assert (await _run(ctx))["status"] == "would_break_reply"


async def test_continuation_lead_line_is_never_judged():
    ctx = _ctx(["对了 加班那事我还记着呢累不累", "面也挺好的，暖胃"])
    ctx.continuation_lines = ["对了 加班那事我还记着呢累不累"]
    assert (await _run(ctx))["status"] == "kept_all"
    assert ctx.replies[0].startswith("对了")


async def test_slow_embeddings_never_hold_the_reply(monkeypatch):
    async def slow(text):
        await asyncio.sleep(2)
        return [1.0, 0.0, 0.0]

    monkeypatch.setattr(semantic, "_embed", slow)
    monkeypatch.setattr(semantic, "_WAIT_S", 0.01)
    ctx = _ctx(["今天真是累坏你了呀", "面也挺好的"])
    record = await _run(ctx)
    assert record["status"] == "timeout" and len(ctx.replies) == 2 and "elapsed_ms" in record


async def test_embedding_errors_let_the_reply_through(monkeypatch):
    async def boom(text):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(semantic, "_embed", boom)
    ctx = _ctx(["今天真是累坏你了呀", "面也挺好的"])
    record = await _run(ctx)
    assert record == {"status": "error", "error": "RuntimeError"} and len(ctx.replies) == 2


async def test_skipped_turns_and_no_previous_turn():
    ctx = _ctx(["今天真是累坏你了呀"])
    ctx.reply_is_fallback = True
    assert (await _run(ctx))["status"] == "not_applicable"
    ctx = _ctx(["今天真是累坏你了呀"], history=[_row("u1", "user", "吃了点面", 5)])
    assert (await _run(ctx))["status"] == "no_previous_turn"
    assert await semantic.apply(_ctx(["x"]), ON) is None  # 没有启动任务: 什么都不做


def test_previous_bubbles_skip_cards_short_lines_old_turns_and_keep_three_turns():
    rows = [_row("u0", "user", "在吗", 3000), _row("old", "assistant", "这是一天多以前说的长句子", 2000)]
    for i in range(4):
        rows += [_row(f"u{i}x", "user", f"第{i}句", 40 - i * 5),
                 _row(f"a{i}", "assistant", f"第{i}轮的回复内容足够长", 39 - i * 5),
                 _row(f"c{i}", "assistant", "活动卡片：西湖", 39 - i * 5, component_card={"type": "offline_activity"}),
                 _row(f"s{i}", "assistant", "嗯嗯", 39 - i * 5)]
    rows.append(_row("u1", "user", "好", 1))
    previous = semantic.previous_bubbles(rows, {"u1"}, now=NOW)
    assert previous == ["第1轮的回复内容足够长", "第2轮的回复内容足够长", "第3轮的回复内容足够长"]


def test_previous_bubbles_window_naive_times_and_non_dialogue_rows():
    naive = (NOW - timedelta(minutes=10)).replace(tzinfo=None).isoformat()
    rows = [
        _row("u0", "user", "早", 3000),
        _row("a0", "assistant", "一天多以前那一轮的回复内容", 2999),  # 在最近三轮里, 但超过 24 小时
        _row("u1", "user", "   ", 20),  # 空用户行不算轮次边界
        {"id": "sys", "role": "system", "content": "系统提示", "metadata": None, "createdAt": naive},
        {"id": "a1", "role": "assistant", "content": "这是没有时区的时间写的回复", "metadata": None,
         "createdAt": naive},
        _row("u2", "user", "嗯", 1),
    ]
    assert semantic.previous_bubbles(rows, {"u2"}, now=NOW) == ["这是没有时区的时间写的回复"]


async def test_embed_uses_the_shared_local_embedding(monkeypatch):
    from app.services.memory.storage import embedding

    calls = []

    async def fake_generate(text):
        calls.append(text)
        return [0.5, 0.5]

    monkeypatch.setattr(embedding, "generate_embedding", fake_generate)
    monkeypatch.undo()  # 撤掉 autouse 的 _embed 替身, 只保留下面这一处
    monkeypatch.setattr(embedding, "generate_embedding", fake_generate)
    assert await semantic._embed("句子") == [0.5, 0.5] and calls == ["句子"]
