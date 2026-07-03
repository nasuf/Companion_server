"""W2 回归: 中期记忆 MVP — 重逢时「上次聊到」摘要."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.chat.session_recap import (
    RECAP_GAP_SECONDS,
    get_or_build_session_recap,
)

P = "app.services.chat.session_recap"


class _FakeRedis:
    def __init__(self, preset: dict | None = None):
        self.store: dict[str, str] = dict(preset or {})

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value


def _msgs(n=6):
    return [
        {"id": f"m{i}", "role": "user" if i % 2 == 0 else "assistant",
         "content": f"消息{i}"}
        for i in range(n)
    ]


@pytest.mark.asyncio
class TestRecapGate:
    async def test_below_gap_threshold_returns_none_without_llm(self):
        with patch(f"{P}._summarize", AsyncMock()) as summarize:
            out = await get_or_build_session_recap(
                "c1", _msgs(), gap_seconds=RECAP_GAP_SECONDS - 1,
            )
        assert out is None
        summarize.assert_not_called()

    async def test_none_gap_returns_none(self):
        assert await get_or_build_session_recap("c1", _msgs(), gap_seconds=None) is None

    async def test_too_short_pre_gap_history_skips(self):
        with patch(f"{P}.get_redis", AsyncMock(return_value=_FakeRedis())):
            out = await get_or_build_session_recap(
                "c1", [{"id": "m1", "role": "user", "content": "嗨"}],
                gap_seconds=RECAP_GAP_SECONDS + 1,
            )
        assert out is None


@pytest.mark.asyncio
class TestRecapBuildAndCache:
    async def test_generates_and_caches(self):
        fake = _FakeRedis()
        with (
            patch(f"{P}.get_redis", AsyncMock(return_value=fake)),
            patch(f"{P}._summarize", AsyncMock(return_value="聊了面试准备，用户有点紧张")) as summarize,
        ):
            out1 = await get_or_build_session_recap(
                "c1", _msgs(), gap_seconds=RECAP_GAP_SECONDS + 1,
            )
            out2 = await get_or_build_session_recap(
                "c1", _msgs(), gap_seconds=RECAP_GAP_SECONDS + 1,
            )
        assert out1 == out2 == "聊了面试准备，用户有点紧张"
        summarize.assert_awaited_once()  # 第二次命中缓存, 同一次重逢只调一次 LLM

    async def test_excludes_current_turn_from_source(self):
        captured: list[list[dict]] = []

        async def fake_summarize(msgs):
            captured.append(msgs)
            return "摘要"

        with (
            patch(f"{P}.get_redis", AsyncMock(return_value=_FakeRedis())),
            patch(f"{P}._summarize", side_effect=fake_summarize),
        ):
            await get_or_build_session_recap(
                "c1", _msgs() + [{"id": "cur", "role": "user", "content": "我回来了"}],
                gap_seconds=RECAP_GAP_SECONDS + 1,
                exclude_ids={"cur"},
            )
        assert all(m.get("id") != "cur" for m in captured[0])

    async def test_llm_failure_returns_none(self):
        with (
            patch(f"{P}.get_redis", AsyncMock(return_value=_FakeRedis())),
            patch(f"{P}._summarize", AsyncMock(return_value=None)),
        ):
            out = await get_or_build_session_recap(
                "c1", _msgs(), gap_seconds=RECAP_GAP_SECONDS + 1,
            )
        assert out is None

    async def test_redis_down_still_generates(self):
        with (
            patch(f"{P}.get_redis", AsyncMock(side_effect=RuntimeError("down"))),
            patch(f"{P}._summarize", AsyncMock(return_value="摘要内容也够长")),
        ):
            out = await get_or_build_session_recap(
                "c1", _msgs(), gap_seconds=RECAP_GAP_SECONDS + 1,
            )
        assert out == "摘要内容也够长"


@pytest.mark.asyncio
async def test_prompt_builder_renders_recap_section():
    from app.services.chat.prompt_builder import _build_session_recap_section
    from app.services.prompting import defaults as d

    with patch(
        "app.services.chat.prompt_builder._get_optional_prompt",
        AsyncMock(return_value=d.CHAT_SESSION_RECAP_SECTION_PROMPT),
    ):
        section = await _build_session_recap_section("聊了面试准备")
    assert section is not None
    assert section.prompt_key == "chat.session_recap_section"
    assert "聊了面试准备" in section.body
    assert "不要硬拉回去" in section.body

    assert await _build_session_recap_section(None) is None
    assert await _build_session_recap_section("") is None
