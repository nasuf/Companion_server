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
        snapshots=[
            {"event_type": "ai_move_decided", "score": -80, "depth": 5,
             "move": {"coordinate": "C17"}},
            {"event_type": "ai_move_decided", "score": 90, "depth": 6,
             "move": {"coordinate": "F6"}},
        ],
        rarity_notes=[],
    )
    base.update(kw)
    return build_narrative(**base)


class TestNarrative:
    def test_score_swing_becomes_material(self):
        """局面从吃紧翻到占优, 那一刻才有故事 —— 每步都报评分会变成流水账.

        第一版按一条围棋样本假设快照里有 reason 代号, 实测 761 条里 740 条为 None,
        按代号做白名单只能覆盖 2/60 局。score + depth 才是所有对弈引擎的通用输出。
        """
        n = _rich()
        assert n.decisions
        assert "稳了" in n.decisions[0]
        # 刻意不给坐标: 实测模型会把 AI 自己的落子说成用户的 ("你 c8b6 那步"),
        # 而且棋谱坐标对用户没有意义 —— 没人聊天时这么说话。
        assert "F6" not in n.decisions[0]

    def test_steady_game_yields_no_decisions(self):
        """一路顺风没有转折就没什么可讲 —— 硬凑"我一直占优"是废话."""
        snaps = [{"event_type": "ai_move_decided", "score": 90, "depth": 5}] * 6
        assert _rich(snapshots=snaps).decisions == []

    def test_scores_without_sign_change_are_ignored(self):
        """小幅波动 (在阈值带内) 不算转折."""
        snaps = [{"event_type": "ai_move_decided", "score": s} for s in (5, -8, 12, -3)]
        assert _rich(snapshots=snaps).decisions == []

    def test_snapshots_without_score_are_skipped(self):
        """实测 761 条决策快照里字段差异很大, 缺 score 的直接跳过而不是猜."""
        snaps = [{"event_type": "ai_move_decided", "algorithm": "mcts"}]
        assert _rich(snapshots=snaps).decisions == []

    def test_non_decision_snapshots_are_ignored(self):
        snaps = [{"event_type": "analysis_snapshot", "score": 99}]
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

    def test_prompt_explains_how_to_use_the_judgement_line(self):
        """代码只出信号 (占优/吃紧/算了几步), 怎么说完全由 prompt 决定 —— 后台才改得动."""
        from app.services.prompting.defaults import GAME_FINISH_REPLY_PROMPT

        assert "我当时的判断" in GAME_FINISH_REPLY_PROMPT
        assert "别报数字" in GAME_FINISH_REPLY_PROMPT

    def test_prompt_guards_against_swapping_who_won(self):
        """实测模型会把"用户第一次赢"安到自己头上, 说反胜负比少说一句严重得多."""
        from app.services.prompting.defaults import GAME_FINISH_REPLY_PROMPT

        assert "不是你的" in GAME_FINISH_REPLY_PROMPT

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


class TestOutcomeContradictionGuard:
    """模型会把胜负说反, 而说反的记忆会长期污染人设.

    prompt 里已经写了"「结果」说的是用户的输赢, 不是你的", 实测仍会写出
    「这是我第一次在象棋里赢下用户」—— 而那局是用户赢的。所以做代码侧兜底。
    """

    async def _run(self, monkeypatch, memory: str, outcome: str):
        monkeypatch.setattr(
            "app.services.prompting.store.get_prompt_text",
            AsyncMock(return_value="{material}|{agent_state}"),
        )
        monkeypatch.setattr(
            "app.services.llm.models.invoke_json",
            AsyncMock(return_value={
                "reply": "这局挺有意思", "worth_remembering": memory,
            }),
        )
        monkeypatch.setattr(
            "app.services.llm.models.get_utility_model", lambda *a, **k: object(),
        )
        return await generate_finish_reply(
            _rich(outcome=outcome), agent_state="", fallback="兜底",
        )

    async def test_ai_claiming_a_win_the_user_actually_won_is_dropped(self, monkeypatch):
        out = await self._run(monkeypatch, "这是我第一次在象棋里赢下用户", "win")
        assert out.worth_remembering is None
        # 回复保留: 回复往往是对的, 一起丢损失更大
        assert out.text == "这局挺有意思"

    async def test_user_claiming_a_win_they_actually_lost_is_dropped(self, monkeypatch):
        out = await self._run(monkeypatch, "用户第一次赢了我", "lose")
        assert out.worth_remembering is None

    async def test_correct_attribution_survives(self, monkeypatch):
        out = await self._run(monkeypatch, "用户第一次在象棋里赢了我", "win")
        assert out.worth_remembering == "用户第一次在象棋里赢了我"

    @pytest.mark.parametrize("memory", [
        "我没赢，但这局很激烈",
        "我差一点就赢了",
        "这次我没能赢下来",
        "我输给了他",
    ])
    async def test_ai_admitting_it_lost_is_not_a_contradiction(
        self, monkeypatch, memory,
    ):
        """用户赢的局里, AI 说"我没赢"/"我差点赢"是**正确**表述.

        只看"主语 + 赢"会把这些当成说反了而丢掉 —— 跟守卫的初衷正相反。
        """
        out = await self._run(monkeypatch, memory, "win")
        assert out.worth_remembering == memory

    async def test_cooperative_we_won_is_not_an_ai_claim(self, monkeypatch):
        """合作局的「我们赢了」主语是双方, 不是 AI 自称独赢."""
        out = await self._run(monkeypatch, "我们一起赢下了这局", "win")
        assert out.worth_remembering == "我们一起赢下了这局"

    async def test_negation_elsewhere_does_not_excuse_a_real_swap(self, monkeypatch):
        """否定判定只看匹配到的那一段 —— 后半句的"不"不该给前半句开脱."""
        out = await self._run(
            monkeypatch, "这是我第一次赢下用户，他下次不会再输了", "win",
        )
        assert out.worth_remembering is None

    async def test_memory_without_any_win_claim_survives(self, monkeypatch):
        """大多数记忆不提胜负, 不该被误伤."""
        out = await self._run(monkeypatch, "那次关键交换改变了整个节奏", "win")
        assert out.worth_remembering == "那次关键交换改变了整个节奏"

    async def test_draw_and_abort_are_not_checked(self, monkeypatch):
        """平局/中断没有明确胜负, 无从判断矛盾, 不做拦截."""
        out = await self._run(monkeypatch, "我赢得挺险", "draw")
        assert out.worth_remembering == "我赢得挺险"
