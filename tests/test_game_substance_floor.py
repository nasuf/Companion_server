""""这局有没有真的玩起来"的地板.

守的是一条生产事故: 一局 **4 步 18 秒**的国际象棋产出了

    刚忙完一摊子客诉，这盘国际象棋倒成了个意外的小插曲。本来以为几步就能定局，
    结果那步关键的交换把局势彻底搅活了，直到最后才敢松口气。

—— 一场不存在的对局。三个原因叠加:

1. 中途退出也给用户判负, 所以那局在库里是 `status='settled'`, 跟真下完一局无法
   区分。实测象棋/围棋的 settled 步数**中位数是 0**, 一局真棋都没下完过。
2. 稀有性的比较池同样全是这些 2 秒退出局, 于是 18 秒被评成"这是玩得最久的一局"。
3. 第 4 手吃掉一个兵被渲染成"关键交换改变了棋子的力量对比"(象棋类 131 个 capture
   里 59 个是吃兵), 模型就在这句话上继续加戏。
"""

from __future__ import annotations

import inspect

import pytest

from app.services.games.native import _has_heavy_capture, _memory_moment
from app.services.games.quick_exit import quick_exit_reply
from app.services.games.substance import (
    _ACTION_FLOOR,
    action_floor,
    played_enough,
)


class TestFloor:
    def test_the_production_case_is_rejected(self):
        """4 步的国际象棋不算玩过一局."""
        assert played_enough("chess", 4) is False

    def test_zero_action_settled_games_are_rejected(self):
        """实测象棋/围棋 settled 的步数中位数是 0 —— 全是开局就走."""
        for key in ("chess", "xiangqi", "go"):
            assert played_enough(key, 0) is False, key

    def test_real_games_pass(self):
        """实测真局: 五子棋 20-77 步, 黑白棋 58-60 步, 象棋 33-104 步."""
        for key, acts in (("gomoku", 20), ("reversi", 58), ("xiangqi", 33)):
            assert played_enough(key, acts) is True, key

    def test_gomoku_floor_is_the_theoretical_minimum(self):
        """先手第 5 子成五 = 用户 5 手 + AI 4 手 = 9, 恰好等于实测最小值.

        再高一点就会误伤真正的快胜局。
        """
        assert action_floor("gomoku") == 9
        assert played_enough("gomoku", 9) is True
        assert played_enough("gomoku", 8) is False

    def test_puzzle_games_keep_low_floors(self):
        """扫雷 3 步扫完小盘是真的赢 —— 不能套用棋类的门槛."""
        assert played_enough("minesweeper", 3) is True
        assert played_enough("match3", 4) is True

    def test_none_counts_as_nothing_played(self):
        assert played_enough("chess", None) is False

    def test_unknown_games_get_a_default(self):
        """新游戏上线时忘了配阈值也不能放过 0 步的局."""
        assert played_enough("some_new_game", 0) is False
        assert action_floor("some_new_game") > 0

    def test_every_shipped_game_has_a_floor(self):
        """漏配的游戏会退到默认值 —— 那对棋类太松, 对解谜类太严."""
        from app.services.games.native import _GAME_DEFINITIONS

        missing = sorted(set(_GAME_DEFINITIONS) - set(_ACTION_FLOOR))
        assert not missing, f"这些游戏没配地板: {missing}"


class TestWiring:
    def test_quick_exit_branch_comes_before_the_recap_branch(self):
        """没玩起来的局根本不该走到复盘那条路 —— 否则白花一次 LLM 调用."""
        from app.services.games.native import _persist_chat_side_effects

        src = inspect.getsource(_persist_chat_side_effects)
        assert src.index("_is_quick_exit") < src.index("_llm_finish_reply")

    def test_quick_exit_covers_aborts_too(self):
        """中途退出也判负, 所以 settled 和 aborted 是同一件事, 不能只挡一边."""
        from app.services.games.native import _persist_chat_side_effects

        src = inspect.getsource(_persist_chat_side_effects)
        condition = src[src.index("if _is_quick_exit") : src.index("quick_exit_reply()")]
        assert "game_finished" not in condition, "quick-exit 分支不该只对完局生效"

    def test_rarity_pool_excludes_trivial_games(self):
        """不筛的话比较池全是 2 秒的退出局, 18 秒就成了"玩得最久"."""
        from app.services.games.rarity import compute_rarity

        src = inspect.getsource(compute_rarity)
        assert "action_floor(game_key)" in src
        assert "action_count')::int," in src


class TestOverstatedCapture:
    """吃一个兵不是"关键交换改变了棋子的力量对比"."""

    @staticmethod
    def _capture(piece):
        return [{"type": "capture", "actor": "agent", "captured_piece": piece}]

    def test_pawn_capture_is_not_a_key_exchange(self):
        for pawn in ("P", "p"):
            text = _memory_moment(self._capture(pawn), "chess")
            assert "关键交换" not in text, pawn

    def test_capturing_a_real_piece_still_counts(self):
        """车马炮士象后被吃是真的改变力量对比."""
        for piece in ("R", "N", "C", "B", "A", "q"):
            text = _memory_moment(self._capture(piece), "xiangqi")
            assert "关键交换" in text, piece

    def test_a_pawn_plus_a_rook_still_counts(self):
        """一局里既吃兵又吃车, 该讲的是车."""
        moments = [
            {"type": "capture", "captured_piece": "P"},
            {"type": "capture", "captured_piece": "R"},
        ]
        assert "关键交换" in _memory_moment(moments, "chess")

    def test_missing_piece_field_is_kept(self):
        """旧数据可能没有 captured_piece —— 无法判断时宁可保留, 让模型自己拿捏."""
        assert _has_heavy_capture([{"type": "capture"}]) is True

    def test_other_moment_types_are_untouched(self):
        """只收紧 capture, 将军/制胜手的措辞不动."""
        text = _memory_moment([{"type": "check"}], "chess")
        assert "将军" in text

    def test_pawn_capture_alone_leaves_no_moment(self):
        """只吃了个兵的局没有可讲的高光 —— 空串会让上游跳过记忆写入."""
        assert _memory_moment(self._capture("P"), "chess") == ""


@pytest.mark.parametrize("game_key,acts,expected", [
    # 生产实测的边界样本
    ("chess", 4, False),      # 事故那一局
    ("chess", 0, False),
    ("xiangqi", 33, True),
    ("xiangqi", 2, False),
    ("gomoku", 9, True),
    ("reversi", 60, True),
    ("reversi", 12, False),
    ("minesweeper", 5, True),
])
def test_production_samples(game_key, acts, expected):
    assert played_enough(game_key, acts) is expected


class TestFallbackWording:
    """兜底文案也不能编.

    地板拦下 LLM 之后, 那些局落到硬编码文案上, 而那些文案假定真下过一局 ——
    实测一局 **0 步**的棋回的是「你后面已经追得很近了，再开一局？」和
    「不是碰巧，是你后面几步真的走得比我稳」, 同样是编造。
    """

    @staticmethod
    def _generic(game_key, acts, outcome):
        from app.services.games.native import _definition, _generic_finish_reply

        return _generic_finish_reply(
            None,  # session 只在个别游戏分支里用到, 这里走不到
            _definition(game_key),
            {
                "user_outcome": outcome,
                "process": {game_key: {"action_count": acts, "key_moments": []}},
            },
        )

    def test_zero_action_loss_does_not_praise_the_user(self):
        text = self._generic("chess", 0, "lose")
        for fabricated in ("追得很近", "走得比我稳", "差点", "转折"):
            assert fabricated not in text, fabricated

    def test_zero_action_win_does_not_describe_play(self):
        text = self._generic("chess", 0, "win")
        assert "后面几步" not in text

    def test_it_does_not_announce_a_result(self):
        """系统判了用户负, 但用户的体感是"我没玩" —— 宣布输赢只会让人莫名其妙."""
        text = self._generic("chess", 0, "lose")
        for verdict in ("我赢", "你输", "我先收下", "你拿下"):
            assert verdict not in text, verdict

    def test_real_games_keep_their_wording(self):
        """只拦没玩起来的局, 真局的文案一个字不动."""
        text = self._generic("xiangqi", 40, "lose")
        assert text != quick_exit_reply()

    def test_gomoku_path_is_covered_too(self):
        """五子棋走独立的 _finish_reply, 漏了它等于半个修复."""
        from app.services.games.native import _finish_reply

        text = _finish_reply(None, {"user_outcome": "lose", "gomoku": {"move_count": 2}})
        assert text == quick_exit_reply()

    def test_gomoku_real_game_still_describes_the_line(self):
        from app.services.games.native import _finish_reply

        text = _finish_reply(
            None,
            {"user_outcome": "win", "gomoku": {"move_count": 30, "winning_line": []}},
        )
        assert text != quick_exit_reply()




class TestQuickExitStaysSilent:
    """点开又关时 AI 什么都不说.

    这条守的是一次真实的骚扰: 上一版做了 10 分钟冷却 + 3 档 × 4 句轮换, 生产上
    3 天发出 **84 条**, 时间戳精确落在 11:00 / 11:10 / 11:20 / 11:30 ——
    **限流只是限流, 它从不停止**。用户持续摆弄界面就一整天每小时收 6 条。

    真朋友看你反复点开关掉, 说一次就不再提了。而"说一次"本身价值也很低: 进出游戏
    在聊天里已经有系统卡片, AI 再评论一句是复述。
    """

    def test_no_message_is_produced(self):
        assert quick_exit_reply() == ""

    def test_takes_no_arguments_so_there_is_nothing_to_tune(self):
        """无参数是刻意的 —— 有了"第几次"/"几步"这些入参就会有人想按它们分档,
        而那正是上一版发出 84 条的起点。"""
        import inspect

        assert not inspect.signature(quick_exit_reply).parameters

    def test_no_canned_lines_survive_in_the_module(self):
        """模块里不该再留句子池: 留着就会被接回去用."""
        import inspect

        from app.services.games import quick_exit

        src = inspect.getsource(quick_exit)
        body = src[src.index('"""', src.index('"""') + 3):]  # 跳过模块 docstring
        for canned in ("棋盘", "不玩了", "撤啦", "摆熟"):
            assert canned not in body, f"句子池残留: {canned}"

    def test_no_rate_limiting_machinery_survives(self):
        """冷却 = 限流 = 最终还是会发。不留 Redis 计数就不会有人重新打开这条路."""
        import inspect

        from app.services.games import quick_exit

        src = inspect.getsource(quick_exit)
        for machinery in ("get_redis", "COOLDOWN", "incr", "pipeline"):
            assert machinery not in src, f"限流机制残留: {machinery}"

    def test_the_pattern_still_reaches_the_ai_elsewhere(self):
        """沉默不等于丢信息: "老是点开又关"该沉淀成印象, 不是逐次评论."""
        import inspect

        from app.services.games.daily_digest import render_digest
        from app.services.memory.behaviour_signals import _game_fact

        assert "没下完就走了" in inspect.getsource(_game_fact)
        assert inspect.getsource(render_digest)

    def test_both_sync_paths_use_the_same_exit(self):
        """五子棋和其他游戏各有一条同步文案路径 —— 漏一条就又开始编造."""
        import inspect

        from app.services.games.native import _finish_reply, _generic_finish_reply

        for fn in (_finish_reply, _generic_finish_reply):
            assert "quick_exit_reply()" in inspect.getsource(fn), fn.__name__
