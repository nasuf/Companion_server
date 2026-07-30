"""存量超长记忆拆分脚本的行为约束.

这个脚本要改 606 行生产数据、新增 717 行, 出错的代价是记忆内容损坏且难以察觉
(超长记忆本来就检索不到, 拆坏了也没人会立刻发现)。所以下面锁的是几条"错了就是
数据损坏"的性质, 而不是输出格式。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "split_oversized_memories.py"
_spec = importlib.util.spec_from_file_location("split_oversized_memories", _SCRIPT)
sos = importlib.util.module_from_spec(_spec)
sys.modules["split_oversized_memories"] = sos
_spec.loader.exec_module(sos)

from app.services.memory.retrieval.context_selector import (  # noqa: E402
    MAX_MEMORY_TOKENS_PER_ITEM,
    estimate_tokens,
)

_OVERSIZED = (
    "入职第一天的“手抖”：第一天正式接听电话，虽然培训时已滚瓜烂熟，但听到电话铃声"
    "响起的一瞬间，她还是紧张得手抖。接起电话后，是一位语气温和的阿姨咨询怎么修改"
    "AI伴侣的发型。我因为紧张，说话有点结巴，但阿姨反而安慰她：“小姑娘，别急，慢慢"
    "来，我不赶时间。”在对方的鼓励下，她顺利完成了第一通服务。挂断电话后，她长长地"
    "舒了一口气。"
)
_UNSPLITTABLE = "那个下着冬雨的黄昏她在公司楼下等了我很久很久始终没有离开" * 8


def _row(content: str, rid: str = "m1") -> dict:
    base = {"id": rid, "content": content, "mention_count": 3, "current_score": 0.7}
    base.update(dict.fromkeys(sos._INHERITED, None))
    base["user_id"] = "u1"
    base["workspace_id"] = "ws1"
    base["level"] = 2
    base["importance"] = 0.72
    return base


class TestPlanning:
    def test_within_limit_rows_are_not_planned(self):
        stats = sos.Stats()
        plans = sos.build_plans("memories_ai", [_row("短记忆")], stats)
        assert plans == []
        assert stats.oversized == 0

    def test_oversized_row_is_planned_into_multiple_pieces(self):
        stats = sos.Stats()
        plans = sos.build_plans("memories_ai", [_row(_OVERSIZED)], stats)
        assert len(plans) == 1
        assert len(plans[0].pieces) >= 2
        assert stats.oversized == 1
        assert stats.planned == 1

    def test_unsplittable_row_is_skipped_not_butchered(self):
        """拆不动就别动.

        硬切会切在句子中间, 用户读到半句话比读不到更糟 —— 这条记忆现在已经检索不到,
        保持现状不会让任何事变差。
        """
        stats = sos.Stats()
        plans = sos.build_plans("memories_ai", [_row(_UNSPLITTABLE)], stats)
        assert plans == []
        assert stats.oversized == 1
        assert stats.skipped_unsplittable == 1

    def test_every_planned_piece_fits_the_limit(self):
        """拆了还超限就是白拆 —— 那条记忆依然进不了注入集."""
        stats = sos.Stats()
        for plan in sos.build_plans("memories_ai", [_row(_OVERSIZED)], stats):
            for piece in plan.pieces:
                assert estimate_tokens(piece) <= MAX_MEMORY_TOKENS_PER_ITEM

    def test_no_content_is_lost_in_the_plan(self):
        import re

        stats = sos.Stats()
        plan = sos.build_plans("memories_ai", [_row(_OVERSIZED)], stats)[0]
        strip = lambda s: re.sub(r"[\s。；;，,、：:“”「」]", "", s)  # noqa: E731
        assert set(strip(plan.original)) <= set(strip("".join(plan.pieces)))

    def test_first_piece_replaces_the_original_row(self):
        """原行保留 id: changelog / access log / 实体引用都挂在它上面, 换 id 会断链."""
        stats = sos.Stats()
        plan = sos.build_plans("memories_ai", [_row(_OVERSIZED, "keep-me")], stats)[0]
        assert plan.memory_id == "keep-me"
        assert plan.new_pieces == plan.pieces[1:]


class TestApply:
    @pytest.mark.asyncio
    async def test_apply_updates_original_and_inserts_the_rest(self, monkeypatch):
        executed: list[tuple[str, tuple]] = []
        embedded: list[str] = []

        async def fake_exec(sql, *args):
            executed.append((sql, args))
            return 1  # 受影响行数: 0 表示内容被并发改过, 见 TestConcurrencySafety

        async def fake_embed(text):
            embedded.append(text)
            return [0.0] * 8

        async def fake_store(mem_id, vec):
            pass

        monkeypatch.setattr(sos.db, "execute_raw", fake_exec)
        monkeypatch.setattr(sos, "generate_embedding", fake_embed)
        monkeypatch.setattr(sos, "store_embedding", fake_store)

        stats = sos.Stats()
        plan = sos.build_plans("memories_ai", [_row(_OVERSIZED)], stats)[0]
        new_ids = [f"new-{i}" for i in range(len(plan.new_pieces))]
        await sos.apply_plan(plan, new_ids, stats)

        updates = [s for s, _ in executed if s.strip().startswith("UPDATE")]
        inserts = [s for s, _ in executed if "INSERT INTO memories_ai" in s]
        assert len(updates) == 1, "原行应该被改写而不是新增一条"
        assert len(inserts) == len(plan.new_pieces)
        assert stats.rows_created == len(plan.new_pieces)

    @pytest.mark.asyncio
    async def test_all_pieces_including_the_original_get_reembedded(self, monkeypatch):
        """原行内容变了, 它的旧向量指向的是已经不存在的文本 —— 不重嵌等于检索错配."""
        embedded: list[str] = []

        async def fake_exec(sql, *args):
            return 1

        async def fake_embed(text):
            embedded.append(text)
            return [0.0] * 8

        monkeypatch.setattr(sos.db, "execute_raw", fake_exec)
        monkeypatch.setattr(sos, "generate_embedding", fake_embed)
        monkeypatch.setattr(sos, "store_embedding", lambda *a: _noop())

        stats = sos.Stats()
        plan = sos.build_plans("memories_ai", [_row(_OVERSIZED)], stats)[0]
        await sos.apply_plan(plan, [f"n{i}" for i in plan.new_pieces], stats)
        assert embedded == plan.pieces, "重嵌的文本必须与写入的分片一一对应"

    @pytest.mark.asyncio
    async def test_embedding_failure_does_not_roll_back_written_rows(self, monkeypatch):
        """行本身是对的, 缺向量只是暂时检索不到, 补跑嵌入即可; 回滚反而留下半修状态."""
        async def fake_exec(sql, *args):
            return 1

        async def boom(_text):
            raise RuntimeError("ollama down")

        monkeypatch.setattr(sos.db, "execute_raw", fake_exec)
        monkeypatch.setattr(sos, "generate_embedding", boom)

        stats = sos.Stats()
        plan = sos.build_plans("memories_ai", [_row(_OVERSIZED)], stats)[0]
        await sos.apply_plan(plan, [f"n{i}" for i in plan.new_pieces], stats)
        assert stats.rows_created == len(plan.new_pieces)
        assert len(stats.embed_failed) == len(plan.pieces)

    @pytest.mark.asyncio
    async def test_new_rows_do_not_inherit_usage_counters(self, monkeypatch):
        """mention_count 是"这条被用过多少次"的历史.

        复制到 N 个新行会把统计放大 N 倍, 进而污染 L2 动态分数的频率因子。
        """
        captured: list[tuple[str, tuple]] = []

        async def fake_exec(sql, *args):
            captured.append((sql, args))
            return 1

        monkeypatch.setattr(sos.db, "execute_raw", fake_exec)
        monkeypatch.setattr(sos, "generate_embedding", lambda t: _val([0.0] * 8))
        monkeypatch.setattr(sos, "store_embedding", lambda *a: _noop())

        stats = sos.Stats()
        plan = sos.build_plans("memories_ai", [_row(_OVERSIZED)], stats)[0]
        await sos.apply_plan(plan, ["n0"], stats)

        for sql, args in captured:
            if "INSERT INTO memories_ai" in sql:
                cols = sql.split("(")[1].split(")")[0].split(", ")
                # 绑定参数必须是列清单的前缀, 后面才是 NOW() 这类字面量。
                # 顺序一旦被打乱, 下面的按名取值就会静默取到别的列的值。
                assert len(args) <= len(cols)
                assert cols[len(args):] == ["created_at", "updated_at"]
                assert args[cols.index("mention_count")] == 0
                assert "current_score" not in cols

    @pytest.mark.asyncio
    async def test_new_rows_inherit_scope_and_tier(self, monkeypatch):
        """workspace / level / 类目必须继承, 否则拆出来的记忆会跑到别人的库里或错层."""
        captured: list[tuple[str, tuple]] = []

        async def fake_exec(sql, *args):
            captured.append((sql, args))
            return 1

        monkeypatch.setattr(sos.db, "execute_raw", fake_exec)
        monkeypatch.setattr(sos, "generate_embedding", lambda t: _val([0.0] * 8))
        monkeypatch.setattr(sos, "store_embedding", lambda *a: _noop())

        stats = sos.Stats()
        plan = sos.build_plans("memories_ai", [_row(_OVERSIZED)], stats)[0]
        await sos.apply_plan(plan, ["n0"], stats)

        for sql, args in captured:
            if "INSERT INTO memories_ai" in sql:
                cols = sql.split("(")[1].split(")")[0].split(", ")
                assert args[cols.index("workspace_id")] == "ws1"
                assert args[cols.index("user_id")] == "u1"
                assert args[cols.index("level")] == 2


class TestConcurrencySafety:
    """脚本要跑好几分钟, 期间线上是活的.

    hygiene 合并或新一轮抽取都可能改到同一条记忆。无条件覆盖会把它们的结果静默冲掉。
    """

    @pytest.mark.asyncio
    async def test_update_is_conditional_on_the_original_content(self, monkeypatch):
        captured: list[tuple[str, tuple]] = []

        async def fake_exec(sql, *args):
            captured.append((sql, args))
            return 1

        monkeypatch.setattr(sos.db, "execute_raw", fake_exec)
        monkeypatch.setattr(sos, "generate_embedding", lambda t: _val([0.0] * 8))
        monkeypatch.setattr(sos, "store_embedding", lambda *a: _noop())

        stats = sos.Stats()
        plan = sos.build_plans("memories_ai", [_row(_OVERSIZED)], stats)[0]
        await sos.apply_plan(plan, ["n0"], stats)

        upd = next(s for s, _ in captured if s.strip().startswith("UPDATE"))
        assert "AND content = $3" in upd, "UPDATE 没带原文条件, 会覆盖并发修改"

    @pytest.mark.asyncio
    async def test_nothing_is_written_when_the_row_changed(self, monkeypatch):
        """条件不满足时不能继续插分片 —— 别人改过的内容我们没重新拆过."""
        calls: list[str] = []

        async def fake_exec(sql, *args):
            calls.append(sql)
            return 0 if sql.strip().startswith("UPDATE") else 1

        monkeypatch.setattr(sos.db, "execute_raw", fake_exec)
        monkeypatch.setattr(sos, "generate_embedding", lambda t: _val([0.0] * 8))
        monkeypatch.setattr(sos, "store_embedding", lambda *a: _noop())

        stats = sos.Stats()
        plan = sos.build_plans("memories_ai", [_row(_OVERSIZED)], stats)[0]
        ok = await sos.apply_plan(plan, ["n0"], stats)

        assert ok is False
        assert stats.skipped_changed == 1
        assert stats.rows_created == 0
        assert not any("INSERT INTO memories_ai" in c for c in calls)


class TestIdempotence:
    def test_second_run_finds_nothing(self):
        """修完的条目不再超限, 重跑扫不到 —— 否则会越拆越碎."""
        stats1 = sos.Stats()
        plan = sos.build_plans("memories_ai", [_row(_OVERSIZED)], stats1)[0]
        rows = [_row(p, f"r{i}") for i, p in enumerate(plan.pieces)]
        stats2 = sos.Stats()
        assert sos.build_plans("memories_ai", rows, stats2) == []


async def _noop():
    return None


async def _val(v):
    return v
