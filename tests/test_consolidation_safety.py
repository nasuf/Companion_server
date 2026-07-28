"""L3 簇压缩的安全性守卫 (Phase 3).

整合是所有记忆维护任务里唯一会**归档原始记忆**的一个, 所以它出错的代价和别的任务
不是一个量级。这些测试盯的是三件让它敢开启的事:

    半失败不留残局   摘要建了但原行没归档 → 下一轮同一簇再压一次 → 重复摘要
    摘要不被改层     有损压缩的前提是"留在 L3", 被 hygiene 吸走就破坏了这个前提
    动作可追溯       归档前先写 changelog, 否则出事时没有回滚依据
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.memory.lifecycle import consolidation
from app.services.memory.provenance import (
    COMPRESSION_EXEMPT,
    CONSOLIDATED,
    REFLECTED,
)


def _row(mid: str, content: str = "早上七点起床") -> dict:
    return {
        "id": mid, "content": content, "importance": 0.3,
        "main_category": "生活", "sub_category": "日常",
        "occur_time": None, "created_at": None, "provenance": "daily_summary",
        "_vec": [0.1] * 8,
    }


class TestArchivalIsAllOrNothing:
    @pytest.mark.asyncio
    async def test_partial_archive_rolls_back_the_digest(self):
        """归档少了几行就必须撤销摘要。

        旧实现逐条 update 且吞异常, 中途失败会留下"摘要已生成 + 原行还在"的状态,
        于是下一轮同一簇被再压一次, 产出重复摘要。
        """
        rows = [_row(f"m{i}") for i in range(5)]
        rolled_back: list[str] = []

        async def _fake_execute(sql, *args):
            if "INSERT INTO memory_changelogs" in sql:
                return len(rows)
            return len(rows) - 1  # 少归档一行

        with patch.object(consolidation, "store_memory",
                          AsyncMock(return_value="digest-1")), \
             patch.object(consolidation, "get_prompt_text",
                          AsyncMock(return_value="{memory_items}")), \
             patch.object(consolidation, "invoke_json",
                          AsyncMock(return_value={"summary": "那段时间作息很规律"})), \
             patch.object(consolidation, "get_utility_model", lambda: object()), \
             patch.object(consolidation.db, "execute_raw", _fake_execute), \
             patch.object(
                 consolidation.memory_repo, "update",
                 AsyncMock(side_effect=lambda mid, **kw: rolled_back.append(mid)),
             ):
            result = await consolidation._compress_cluster(
                source="ai", user_id="u1", workspace_id="w1", cluster=rows,
            )

        assert result is None, "半失败却返回了摘要 ID"
        assert rolled_back == ["digest-1"], "摘要没有被撤销"

    @pytest.mark.asyncio
    async def test_changelog_is_written_before_archiving(self):
        """顺序反了的话, 归档成功而 changelog 失败就产生无从追溯的孤儿行。"""
        rows = [_row(f"m{i}") for i in range(5)]
        order: list[str] = []

        async def _fake_execute(sql, *args):
            order.append("changelog" if "memory_changelogs" in sql else "archive")
            return len(rows)

        with patch.object(consolidation, "store_memory",
                          AsyncMock(return_value="digest-1")), \
             patch.object(consolidation, "get_prompt_text",
                          AsyncMock(return_value="{memory_items}")), \
             patch.object(consolidation, "invoke_json",
                          AsyncMock(return_value={"summary": "作息规律的一段日子"})), \
             patch.object(consolidation, "get_utility_model", lambda: object()), \
             patch.object(consolidation.db, "execute_raw", _fake_execute):
            await consolidation._compress_cluster(
                source="ai", user_id="u1", workspace_id="w1", cluster=rows,
            )

        assert order == ["changelog", "archive"]

    @pytest.mark.asyncio
    async def test_changelog_carries_the_original_text(self):
        """撤销脚本要靠它直接显示"这次整合吞掉了什么"。

        原行只是归档不是删除, 所以数据不会丢 —— 但少了快照, 运维得先去两张表里
        捞原行才看得懂自己在撤什么, 那这份审计就等于没有。
        """
        rows = [_row(f"m{i}", f"第 {i} 天午休听播客") for i in range(5)]
        captured: dict = {}

        async def _fake_execute(sql, *args):
            if "memory_changelogs" in sql:
                captured["sql"] = sql
                captured["args"] = args
            return len(rows)

        with patch.object(consolidation, "store_memory",
                          AsyncMock(return_value="digest-1")), \
             patch.object(consolidation, "get_prompt_text",
                          AsyncMock(return_value="{memory_items}")), \
             patch.object(consolidation, "invoke_json",
                          AsyncMock(return_value={"summary": "那阵子作息一直很规律"})), \
             patch.object(consolidation, "get_utility_model", lambda: object()), \
             patch.object(consolidation.db, "execute_raw", _fake_execute):
            await consolidation._compress_cluster(
                source="ai", user_id="u1", workspace_id="w1", cluster=rows,
            )

        assert "old_value" in captured["sql"]
        assert "第 0 天午休听播客" in captured["args"]

    @pytest.mark.asyncio
    async def test_archive_is_a_single_statement(self):
        """逐行归档就有"归档到一半"的中间态。批量语句天然原子, 没有这个态。"""
        rows = [_row(f"m{i}") for i in range(5)]
        archive_calls: list[str] = []

        async def _fake_execute(sql, *args):
            if "is_archived = true" in sql:
                archive_calls.append(sql)
            return len(rows)

        with patch.object(consolidation, "store_memory",
                          AsyncMock(return_value="digest-1")), \
             patch.object(consolidation, "get_prompt_text",
                          AsyncMock(return_value="{memory_items}")), \
             patch.object(consolidation, "invoke_json",
                          AsyncMock(return_value={"summary": "那阵子作息一直很规律"})), \
             patch.object(consolidation, "get_utility_model", lambda: object()), \
             patch.object(consolidation.db, "execute_raw", _fake_execute):
            await consolidation._compress_cluster(
                source="ai", user_id="u1", workspace_id="w1", cluster=rows,
            )

        assert len(archive_calls) == 1
        assert "ANY($1::text[])" in archive_calls[0]

    @pytest.mark.asyncio
    async def test_rollback_failure_does_not_mask_the_original_error(self):
        """撤销失败只记日志 —— 再抛异常会把真正的错因盖掉。"""
        rows = [_row(f"m{i}") for i in range(5)]

        with patch.object(consolidation, "store_memory",
                          AsyncMock(return_value="digest-1")), \
             patch.object(consolidation, "get_prompt_text",
                          AsyncMock(return_value="{memory_items}")), \
             patch.object(consolidation, "invoke_json",
                          AsyncMock(return_value={"summary": "那阵子作息一直很规律"})), \
             patch.object(consolidation, "get_utility_model", lambda: object()), \
             patch.object(consolidation.db, "execute_raw",
                          AsyncMock(side_effect=RuntimeError("db down"))), \
             patch.object(consolidation.memory_repo, "update",
                          AsyncMock(side_effect=RuntimeError("also down"))):
            result = await consolidation._compress_cluster(
                source="ai", user_id="u1", workspace_id="w1", cluster=rows,
            )

        assert result is None


class TestDigestsStayInTheColdTier:
    def test_consolidation_candidates_exclude_the_exempt_provenances(self):
        """候选白名单目前本就把它们挡在外面, 显式排除是为了以后放宽白名单时
        不会顺手把它们放进来。"""
        import inspect

        source = inspect.getsource(consolidation._load_candidates)
        assert "COALESCE(m.provenance, '') <> ALL($4::text[])" in source
        assert "sorted(COMPRESSION_EXEMPT)" in source

    def test_hygiene_excludes_the_compression_exempt_provenances(self):
        """摘要是有损产物, 被 hygiene 合并进 L2 就等于让它占据比原始记忆更强的
        位置 —— 而原始行已经归档、找不回来了。反思判断同理: 合并会抹掉它的证据
        边界, 而没有证据的推断无法复核。"""
        import inspect

        from app.services.memory.lifecycle import hygiene

        source = inspect.getsource(hygiene._scope_memories)
        assert "COMPRESSION_EXEMPT" in source
        assert '"OR"' in source, "必须用显式 OR"

    @pytest.mark.asyncio
    async def test_hygiene_filter_keeps_rows_with_null_provenance(self):
        """SQL 的 != 不匹配 NULL, 而历史记忆的 provenance 大多是 NULL。

        直接写 {"not": CONSOLIDATED} 会把它们一并排除 —— 生产实测 82 条里会漏掉
        25 条, 静默少处理三成数据。
        """
        from app.services.memory.lifecycle import hygiene

        with patch.object(
            hygiene.memory_repo, "find_many", new=AsyncMock(return_value=[]),
        ) as find_many:
            await hygiene._scope_memories(
                source="user", user_id="u1", workspace_id="w1", limit=10,
            )

        where = find_many.await_args.kwargs["where"]
        assert {"provenance": None} in where["OR"], "NULL 行会被静默排除"
        exempt = next(
            clause["provenance"]["notIn"] for clause in where["OR"]
            # 另一个分支的 provenance 是 None (匹配 NULL 行), 不能直接下标
            if isinstance(clause.get("provenance"), dict)
            and "notIn" in clause["provenance"]
        )
        assert set(exempt) == set(COMPRESSION_EXEMPT)
        assert CONSOLIDATED in exempt and REFLECTED in exempt

    def test_digests_are_clamped_to_the_cold_tier(self):
        import inspect

        source = inspect.getsource(consolidation._compress_cluster)
        assert "level=3" in source
        assert "min(0.49" in source


class TestEnabledByDefault:
    def test_consolidation_is_on(self):
        """2026-07-28 起默认开启。这条不是为了锁死状态, 是为了让关掉它成为一个
        需要解释的动作 —— 它曾经默认关了很久, 期间因为聚类阈值没跟着换模型一起
        重标而完全空转, 没人发现。"""
        from app.config import Settings

        assert Settings.model_fields["memory_consolidation_enabled"].default is True

    def test_canary_allowlist_exists_for_rollback(self):
        """全量开之后仍要留缩回单 workspace 的手段 —— 它是唯一会归档原始数据的
        维护任务, 出问题时"先缩小范围"比"直接关掉"更有价值。"""
        from app.config import Settings

        assert "memory_consolidation_workspaces" in Settings.model_fields
        assert Settings.model_fields["memory_consolidation_workspaces"].default == ""


class TestRunAudit:
    @pytest.mark.asyncio
    async def test_dry_run_touches_nothing(self):
        """开 flag 前先看它会动多少东西, 比开了再后悔便宜。"""
        with patch.object(
            consolidation, "_load_candidates",
            AsyncMock(return_value=[_row(f"m{i}") for i in range(6)]),
        ), patch.object(consolidation, "_compress_cluster", AsyncMock()) as compress, \
           patch.object(consolidation, "_record_run", AsyncMock()) as record:
            stats = await consolidation.compress_l3_clusters_for_workspace(
                user_id="u1", workspace_id="w1", dry_run=True,
            )

        compress.assert_not_awaited()
        record.assert_not_awaited()
        assert stats["clusters"] > 0, "dry-run 应当仍然报告会动多少簇"
        assert stats["digests"] == 0

    @pytest.mark.asyncio
    async def test_run_is_recorded_with_its_job_tag(self):
        """这张表此前只有 hygiene 在写。簇压缩才是会归档原行的那个, 更需要留痕。"""
        captured: dict = {}

        async def _capture(sql, *args):
            captured["sql"] = sql
            captured["args"] = args
            return 1

        with patch.object(consolidation.db, "execute_raw", _capture):
            await consolidation._record_run(
                {"clusters": 2, "compressed_rows": 11, "digests": 2,
                 "failed": 0, "digest_ids": ["d1", "d2"]},
                user_id="u1", workspace_id="w1",
            )

        assert "l3_compression" in captured["sql"]
        assert "d1" in captured["args"][-1]

    @pytest.mark.asyncio
    async def test_audit_failure_does_not_fail_the_run(self):
        """记不上账不该让已经完成的整合报错。"""
        with patch.object(consolidation.db, "execute_raw",
                          AsyncMock(side_effect=RuntimeError("audit table gone"))):
            await consolidation._record_run(
                {"clusters": 1, "digests": 1, "digest_ids": ["d1"]},
                user_id="u1", workspace_id="w1",
            )
