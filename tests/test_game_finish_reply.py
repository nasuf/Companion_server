"""完局伴聊的 LLM 生成.

原来是硬编码 if/else (`if count <= 2: return "《X》才刚开头…"`), 用户说它"机械"是
字面意义上的准确。而讽刺的是最丰富的数据喂给了最贫瘠的生成路径 —— 库里每一步 AI
决策都存着 reason / top_candidates / simulations, agent 明明"想"过 220 种可能,
嘴上只会说"先停在这里吧"。

这套测试守三件事: 素材整理不造措辞、模型跑偏时能退回兜底、以及 worth_remembering
**允许为空且大多数时候就该为空** —— 不给这个权限的话, 我们只是把便宜的模板换成了
贵的模板。
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.services.games.finish_reply import generate_finish_reply
from app.services.games.narrative import GameNarrative, build_narrative, render_material


def _rich(**kw) -> GameNarrative:
    base = dict(
        title="围棋", outcome="lose", is_cooperative=False, action_count=40,
        duration_seconds=300, moment_texts=["有一次直接将军"],
        snapshots=[{
            "event_type": "ai_move_decided", "reason": "territory_and_influence",
            "move": {"coordinate": "C17"},
        }],
        rarity_notes=[],
    )
    base.update(kw)
    return build_narrative(**base)


class TestNarrative:
    def test_ai_reasons_are_passed_as_codes_not_prewritten_phrases(self):
        """代码只传信号, 措辞在 prompt 的词表里 —— 后台才改得动.

        预写短语还会让模型照抄, 反倒失去结合上下文措辞的机会。
        """
        n = _rich()
        assert n.decisions
        assert n.decisions[0].startswith("territory_and_influence")
        assert "C17" in n.decisions[0]

    def test_repeated_reasons_are_collapsed(self):
        """下十步都是"想围地"讲一次就够, 重复只会让模型平铺直叙."""
        snaps = [{"event_type": "ai_move_decided", "reason": "territory_and_influence"}] * 5
        assert len(_rich(snapshots=snaps).decisions) == 1

    def test_unknown_reason_codes_are_dropped(self):
        """引擎会加新 reason, 没映射的直接丢 —— 把 raw code 塞进 prompt 模型看不懂."""
        snaps = [{"event_type": "ai_move_decided", "reason": "some_new_heuristic"}]
        assert _rich(snapshots=snaps).decisions == []

    def test_non_decision_snapshots_are_ignored(self):
        snaps = [{"event_type": "analysis_snapshot", "reason": "defend"}]
        assert _rich(snapshots=snaps).decisions == []

    def test_empty_game_has_no_substance(self):
        """没有高光、没有 AI 决策、也不稀有 —— 不值得为它调模型."""
        n = _rich(moment_texts=[], snapshots=[], rarity_notes=[])
        assert n.has_substance is False

    def test_rarity_alone_is_substance(self):
        n = _rich(moment_texts=[], snapshots=[], rarity_notes=["这是用户第一次赢"])
        assert n.has_substance is True

    def test_material_gives_facts_not_phrasing(self):
        """在这里拼句子等于又造一个模板 —— 逐局记忆的模板化就是这么来的."""
        text = render_material(_rich())
        for canned in ("真棒", "下次再来", "好可惜", "我们一起"):
            assert canned not in text

    def test_reason_wording_lives_in_the_prompt_not_the_code(self):
        """守住这条: 面向用户的措辞进 registry, 代码只留信号白名单.

        写死在代码里的话后台调不了, 而这些词恰恰是这个功能最该打磨的部分。
        """
        import inspect

        from app.services.games import narrative as mod
        from app.services.prompting.defaults import GAME_FINISH_REPLY_PROMPT

        src = inspect.getsource(mod)
        for phrase in ("想先把那块地围起来", "看到能吃子", "想抢中间的位置"):
            assert phrase not in src, f"{phrase!r} 应该在 prompt 里, 不在代码里"
            assert phrase in GAME_FINISH_REPLY_PROMPT, f"prompt 词表缺 {phrase!r}"

    def test_every_known_reason_has_a_glossary_entry(self):
        """白名单里放行的代号, prompt 词表里必须解释 —— 否则模型收到看不懂的串."""
        from app.services.games.narrative import _KNOWN_REASONS
        from app.services.prompting.defaults import GAME_FINISH_REPLY_PROMPT

        missing = [r for r in _KNOWN_REASONS if r not in GAME_FINISH_REPLY_PROMPT]
        assert not missing, f"这些代号会被传给模型但词表没解释: {missing}"

    def test_cooperative_outcome_wording_differs(self):
        """合作游戏没有"我赢了" —— 说错会让 agent 显得没在同一条船上."""
        text = render_material(_rich(is_cooperative=True, outcome="win"))
        assert "一起过关" in text
        assert "用户赢了" not in text


@pytest.mark.asyncio
class TestGeneration:
    async def _run(self, monkeypatch, llm_result, narrative=None):
        monkeypatch.setattr(
            "app.services.prompting.store.get_prompt_text",
            AsyncMock(return_value="{material}|{agent_state}"),
        )
        monkeypatch.setattr(
            "app.services.llm.models.invoke_json",
            AsyncMock(return_value=llm_result),
        )
        monkeypatch.setattr(
            "app.services.llm.models.get_utility_model", lambda *a, **k: object(),
        )
        return await generate_finish_reply(
            narrative or _rich(), agent_state="在上班摸鱼", fallback="这局结束了。",
        )

    async def test_happy_path(self, monkeypatch):
        out = await self._run(monkeypatch, {
            "reply": "你后面几步走得比我稳，我本来还想抢右边那块",
            "worth_remembering": None,
        })
        assert out.source == "llm"
        assert "抢右边" in out.text
        assert out.worth_remembering is None

    async def test_memory_is_captured_when_present(self, monkeypatch):
        out = await self._run(monkeypatch, {
            "reply": "这局真险", "worth_remembering": "他第一次在围棋上赢我",
        })
        assert out.worth_remembering == "他第一次在围棋上赢我"

    async def test_string_null_is_treated_as_empty(self, monkeypatch):
        """模型常用 "null"/"无" 表达空 —— 那些不是记忆."""
        for junk in ("null", "None", "无", "没有", "  "):
            out = await self._run(monkeypatch, {"reply": "嗯", "worth_remembering": junk})
            assert out.worth_remembering is None, junk

    async def test_overlong_reply_falls_back(self, monkeypatch):
        """伴聊是一两句话; 冒出赛后总结说明 prompt 没压住, 兜底至少是短的."""
        out = await self._run(monkeypatch, {"reply": "复盘：" + "这一步" * 60})
        assert out.source == "rejected"
        assert out.text == "这局结束了。"

    async def test_empty_reply_falls_back(self, monkeypatch):
        out = await self._run(monkeypatch, {"reply": ""})
        assert out.text == "这局结束了。"

    async def test_bad_shape_falls_back(self, monkeypatch):
        out = await self._run(monkeypatch, ["not", "a", "dict"])
        assert out.source == "bad_shape"

    async def test_llm_failure_falls_back(self, monkeypatch):
        monkeypatch.setattr(
            "app.services.prompting.store.get_prompt_text",
            AsyncMock(side_effect=RuntimeError("down")),
        )
        out = await generate_finish_reply(
            _rich(), agent_state="", fallback="这局结束了。",
        )
        assert out.source == "llm_failed"
        assert out.text == "这局结束了。"

    async def test_no_substance_skips_the_model(self, monkeypatch):
        """素材全空时模型拿不到任何独特信息, 产出跟硬编码没区别, 白花一次调用."""
        invoke = AsyncMock()
        monkeypatch.setattr("app.services.llm.models.invoke_json", invoke)
        out = await generate_finish_reply(
            _rich(moment_texts=[], snapshots=[], rarity_notes=[]),
            agent_state="", fallback="这局结束了。",
        )
        assert out.source == "no_substance"
        invoke.assert_not_awaited()

    async def test_overlong_memory_is_dropped_but_reply_kept(self, monkeypatch):
        out = await self._run(monkeypatch, {
            "reply": "打得不错", "worth_remembering": "很长的回忆" * 30,
        })
        assert out.text == "打得不错"
        assert out.worth_remembering is None
