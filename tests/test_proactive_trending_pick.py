"""A 模式「全网热点内容搭话」(《主动交流提示词（新增）》4-1 / 4-2 / 4-3).

筛选 (trending_pick) → ctx["trending_pick"] → 4-3 生成; 卡片与消息同源。
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.proactive import sender, trending_pick
from app.services.proactive.trending_pick import TrendingPick, pick_trending

CANDIDATES = [
    {"title": "秋天第一杯奶茶", "snippet": "奶茶店排起长队", "platform": "微博"},
    {"title": "某明星去世", "snippet": "", "platform": "微博"},        # 规则黑名单
    {"title": "猫咪学会开冰箱", "snippet": "主人哭笑不得", "platform": "抖音"},
    {"title": "", "snippet": "没有标题的", "platform": "知乎"},
]


class FakePicker:
    """按 prompt key 返回模型输出, 记录参数."""

    def __init__(self, **outputs):
        self.outputs = outputs
        self.calls: list[tuple[str, dict]] = []

    async def __call__(self, key, params):
        self.calls.append((key, params))
        return self.outputs.get(key)


@pytest.fixture
def picker(monkeypatch):
    fake = FakePicker()
    monkeypatch.setattr(trending_pick, "_run_pick", fake)
    monkeypatch.setattr(trending_pick.random, "shuffle", lambda _pool: None)  # 保持顺序可断言
    return fake


def _hobbies(monkeypatch, hobbies):
    monkeypatch.setattr(trending_pick, "_load_user_hobbies", AsyncMock(return_value=hobbies))


async def test_blocklist_and_untitled_candidates_never_reach_the_model(monkeypatch, picker):
    _hobbies(monkeypatch, [])
    picker.outputs["proactive.trending_pick_random"] = {"index": 1, "summary": "猫咪学会自己开冰箱"}
    pick = await pick_trending(CANDIDATES, user_id="u", workspace_id="w")
    formatted = picker.calls[0][1]["candidates"]
    assert formatted.splitlines() == [
        "0. [微博] 秋天第一杯奶茶 —— 奶茶店排起长队",
        "1. [抖音] 猫咪学会开冰箱 —— 主人哭笑不得",
    ]
    # 序号对应过滤后的候选, 卡片挂的就是模型选中的那一条
    assert pick == TrendingPick("猫咪学会自己开冰箱", CANDIDATES[2], "random")


async def test_interest_first_then_random_fallback(monkeypatch, picker):
    _hobbies(monkeypatch, ["喜欢喝奶茶"])
    monkeypatch.setattr(trending_pick.random, "random", lambda: 0.1)  # 命中爱好优先
    picker.outputs.update({
        "proactive.trending_pick_interest": {"index": -1, "summary": ""},  # 无匹配
        "proactive.trending_pick_random": {"index": 0, "summary": "奶茶店排长队"},
    })
    pick = await pick_trending(CANDIDATES, user_id="u", workspace_id="w")
    assert [key for key, _ in picker.calls] == [
        "proactive.trending_pick_interest", "proactive.trending_pick_random",
    ]
    assert picker.calls[0][1]["user_hobbies"] == "- 喜欢喝奶茶"
    assert pick.mode == "random" and pick.item is CANDIDATES[0]


async def test_interest_match_wins(monkeypatch, picker):
    _hobbies(monkeypatch, ["养了两只猫"])
    monkeypatch.setattr(trending_pick.random, "random", lambda: 0.1)
    picker.outputs["proactive.trending_pick_interest"] = {"index": 1, "summary": "猫会开冰箱了"}
    pick = await pick_trending(CANDIDATES, user_id="u", workspace_id="w")
    assert pick.mode == "interest" and pick.item is CANDIDATES[2]
    assert len(picker.calls) == 1


async def test_no_hobbies_goes_straight_to_random(monkeypatch, picker):
    _hobbies(monkeypatch, [])
    monkeypatch.setattr(trending_pick.random, "random", lambda: 0.0)
    await pick_trending(CANDIDATES, user_id="u", workspace_id="w")
    assert [key for key, _ in picker.calls] == ["proactive.trending_pick_random"]


async def test_recently_featured_titles_are_excluded(monkeypatch, picker):
    _hobbies(monkeypatch, [])
    await pick_trending(
        CANDIDATES, user_id="u", workspace_id="w", exclude_titles={"秋天第一杯奶茶"},
    )
    assert "奶茶" not in picker.calls[0][1]["candidates"]

    picker.calls.clear()
    everything = {"秋天第一杯奶茶", "猫咪学会开冰箱"}
    assert await pick_trending(CANDIDATES, user_id="u", workspace_id="w", exclude_titles=everything) is None
    assert picker.calls == []  # 一条能聊的都没有就别调模型


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "not json",
        {"index": -1, "summary": ""},
        {"index": 9, "summary": "越界"},
        {"index": "x", "summary": "不是数字"},
        {"index": 0, "summary": ""},  # 选了但没摘要, 4-3 无从生成
    ],
)
def test_parse_pick_rejects_bad_output(raw):
    assert trending_pick._parse_pick(raw, CANDIDATES[:1], "random") is None


def test_parse_pick_caps_summary_at_100_chars():
    pick = trending_pick._parse_pick({"index": 0, "summary": "长" * 150}, CANDIDATES[:1], "random")
    assert len(pick.summary) == 100


async def test_hobbies_are_preferences_not_dislikes(monkeypatch):
    rows = [
        SimpleNamespace(content="喜欢喝奶茶", subCategory="饮食喜好"),
        SimpleNamespace(content="讨厌香菜", subCategory="饮食厌恶"),
        SimpleNamespace(content="周末爱爬山", subCategory="生活习惯"),
        SimpleNamespace(content="", subCategory="审美爱好"),
    ]
    find = AsyncMock(return_value=rows)
    monkeypatch.setattr(trending_pick.memory_repo, "find_many", find)
    assert await trending_pick._load_user_hobbies("u", "w") == ["喜欢喝奶茶", "周末爱爬山"]
    assert find.await_args.kwargs["where"]["mainCategory"] == "偏好"

    monkeypatch.setattr(trending_pick.memory_repo, "find_many", AsyncMock(side_effect=RuntimeError))
    assert await trending_pick._load_user_hobbies("u", "w") == []


# ── sender: 4-3 生成 + 卡片同源 ─────────────────────────────────────────

async def test_trending_pick_routes_to_trending_chat_prompt(monkeypatch):
    from app.services.prompting.registry import PROMPT_DEFINITION_MAP

    monkeypatch.setattr(
        sender, "get_prompt_text",
        AsyncMock(return_value=PROMPT_DEFINITION_MAP["proactive.trending_chat"].default_text),
    )
    invoke = AsyncMock(return_value="刷到有只猫会自己开冰箱，笑死")
    monkeypatch.setattr(sender, "invoke_text", invoke)
    monkeypatch.setattr(sender, "get_chat_model", lambda: None)
    ctx = {
        "agent": SimpleNamespace(mbti=None, currentMbti=None),
        "trigger_type": "silence_wakeup",
        "source": "greeting",
        "trending_pick": TrendingPick("猫咪学会自己开冰箱", CANDIDATES[2], "random"),
        "_admin_test": True,
    }
    assert await sender._generate_message(ctx) == "刷到有只猫会自己开冰箱，笑死"
    prompt = invoke.await_args.args[1]
    assert "筛选后的热点内容：猫咪学会自己开冰箱" in prompt and "{" not in prompt


async def test_attach_trending_stores_pick(monkeypatch):
    meta = SimpleNamespace(candidates=tuple(CANDIDATES))
    monkeypatch.setattr(
        "app.services.proactive.trending_context.resolve_trending_context",
        AsyncMock(return_value=("text", True, meta)),
    )
    monkeypatch.setattr(
        "app.services.proactive.featured_topics.get_recent_featured", AsyncMock(return_value=set()),
    )
    pick = TrendingPick("猫会开冰箱", CANDIDATES[2], "random")
    picker = AsyncMock(return_value=pick)
    monkeypatch.setattr(trending_pick, "pick_trending", picker)
    ctx: dict = {"topic_theme": "日常"}
    state = SimpleNamespace(user_id="u", workspace_id="w")
    assert await sender._attach_trending(ctx, state, "silence_wakeup", None) is True
    assert ctx["trending_pick"] is pick
    assert picker.await_args.kwargs == {"user_id": "u", "workspace_id": "w", "exclude_titles": set()}

    # 一条都不合适: 不带热点, 按原来源正常发
    picker.return_value = None
    ctx = {"topic_theme": "日常"}
    await sender._attach_trending(ctx, state, "silence_wakeup", None)
    assert "trending_pick" not in ctx


async def test_decay_final_never_pulls_trending(monkeypatch):
    """衰减最后一次有专属 prompt: 抓热榜只会让卡片与消息不同源."""
    resolve = AsyncMock(return_value=("text", True, SimpleNamespace(candidates=tuple(CANDIDATES))))
    monkeypatch.setattr("app.services.proactive.trending_context.resolve_trending_context", resolve)
    ctx: dict = {"topic_theme": "日常", "is_decay_final": True}
    state = SimpleNamespace(user_id="u", workspace_id="w")
    assert await sender._attach_trending(ctx, state, "silence_wakeup", None) is False
    resolve.assert_not_awaited()
    assert "trending_pick" not in ctx


async def test_music_source_never_pulls_trending(monkeypatch):
    """音乐推荐挂的是音乐卡, 消息再讲热点就不同源了."""
    resolve = AsyncMock(return_value=("text", True, SimpleNamespace(candidates=tuple(CANDIDATES))))
    monkeypatch.setattr("app.services.proactive.trending_context.resolve_trending_context", resolve)
    ctx: dict = {"topic_theme": "日常", "source": "music"}
    assert await sender._attach_trending(ctx, SimpleNamespace(user_id="u", workspace_id="w"),
                                         "silence_wakeup", None) is False
    resolve.assert_not_awaited()


@pytest.mark.parametrize(
    ("pick", "forced"),
    [
        (TrendingPick("猫会开冰箱", CANDIDATES[2], "random"), True),
        # 抓了热榜但一条没选中: 消息与热点无关, 不能强挂一张独立搜来的卡
        (None, False),
    ],
)
async def test_link_card_is_forced_only_for_the_picked_item(monkeypatch, pick, forced):
    link = AsyncMock(return_value=(None, "e2e"))
    monkeypatch.setattr("app.services.chat_links.maybe_prepare_proactive_link_recommendation", link)
    monkeypatch.setattr(
        "app.services.proactive.trending_gate.should_attach_trending_link_card", lambda **_k: True,
    )
    ctx = {"source": "greeting", "stage": "warming", "topic_theme": "日常", "trending_pick": pick}
    await sender._prepare_attachments(
        ctx, SimpleNamespace(user_id="u"), SimpleNamespace(conversation_id="c"),
        trigger_type="silence_wakeup", message="刷到一只猫会开冰箱", trending_attached=True,
        admin_test_options=None,
    )
    kwargs = link.await_args.kwargs
    assert kwargs["force"] is forced
    assert kwargs["preselected_item"] == (pick.item if pick else None)
