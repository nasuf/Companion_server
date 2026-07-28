"""Phase 2 L3 cluster compression tests (no LLM / no DB)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.memory.lifecycle import consolidation as cons


def _cand(mid: str, vec: list[float], main="生活", sub="其他", imp=0.3, summary=None):
    return {
        "id": mid,
        "content": summary or f"记忆{mid}",
        "importance": imp,
        "main_category": main,
        "sub_category": sub,
        "occur_time": None,
        "provenance": "daily_summary",
        "_vec": vec,
    }


class TestClustering:
    def test_similar_rows_same_sub_form_cluster(self):
        rows = [_cand(f"m{i}", [1.0, 0.0]) for i in range(5)]
        clusters = cons._cluster(rows)
        assert len(clusters) == 1
        assert len(clusters[0]) == 5

    def test_below_min_cluster_size_ignored(self):
        rows = [_cand(f"m{i}", [1.0, 0.0]) for i in range(4)]  # < 5
        assert cons._cluster(rows) == []

    def test_dissimilar_rows_not_clustered(self):
        rows = [_cand(f"m{i}", [1.0, 0.0]) for i in range(4)]
        rows.append(_cand("far", [0.0, 1.0]))  # orthogonal → sim 0
        assert cons._cluster(rows) == []

    def test_different_subcategories_never_mix(self):
        rows = [_cand(f"a{i}", [1.0, 0.0], sub="工作") for i in range(3)]
        rows += [_cand(f"b{i}", [1.0, 0.0], sub="健康") for i in range(3)]
        # Same vectors but split 3/3 across subs → neither reaches 5.
        assert cons._cluster(rows) == []


@pytest.mark.asyncio
class TestCompression:
    async def test_cluster_compressed_and_originals_archived(self):
        rows = [_cand(f"m{i}", [1.0, 0.0], imp=0.3 + i * 0.02) for i in range(6)]
        store_mock = AsyncMock(return_value="digest-1")
        executed: list[str] = []

        async def _execute(sql, *args):
            executed.append(sql)
            return 6  # 归档行数与簇大小一致

        with (
            patch.object(cons, "_load_candidates", AsyncMock(side_effect=[rows, []])),
            patch.object(cons, "get_prompt_text", AsyncMock(return_value="{owner}{main_category}{sub_category}{memory_items}")),
            patch.object(cons, "invoke_json", AsyncMock(return_value={"summary": "十月里我常在早晨散步、买咖啡，偶尔去河边拍照"})),
            patch.object(cons, "get_utility_model", lambda: object()),
            patch.object(cons, "store_memory", store_mock),
            patch.object(cons.db, "execute_raw", _execute),
        ):
            stats = await cons.compress_l3_clusters_for_workspace(
                user_id="u1", workspace_id="ws1",
            )

        assert stats["digests"] == 1
        assert stats["compressed_rows"] == 6
        # Digest stored as L3 consolidated with clamped importance.
        store_kwargs = store_mock.await_args.kwargs
        assert store_kwargs["level"] == 3
        assert store_kwargs["provenance"] == "consolidated"
        assert store_kwargs["importance"] <= 0.49
        # 2026-07-20: digest 必须跳过 reconciliation, 否则可能被 update_existing
        # 并进一条非簇同类记忆并连带覆盖它.
        assert store_kwargs["skip_reconciliation"] is True
        # 归档与审计各一条批量语句 (外加一条 run 级审计)。
        # 2026-07-28: 原先是逐行 update + 逐行 changelog, 中途失败会留下"摘要已生成
        # 但原行还在"的半失败态, 下一轮同一簇再被压一次产出重复摘要。批量语句天然
        # 原子, 没有这个中间态。
        changelog_sql = [s for s in executed if "memory_changelogs" in s]
        archive_sql = [s for s in executed if "is_archived = true" in s]
        assert len(changelog_sql) == 1
        assert len(archive_sql) == 1
        assert "consolidated_into" in changelog_sql[0]
        assert "ANY($1::text[])" in archive_sql[0]

    async def test_short_digest_rejected_nothing_archived(self):
        rows = [_cand(f"m{i}", [1.0, 0.0]) for i in range(5)]
        update_mock = AsyncMock()
        with (
            patch.object(cons, "_load_candidates", AsyncMock(side_effect=[rows, []])),
            patch.object(cons, "get_prompt_text", AsyncMock(return_value="{owner}{main_category}{sub_category}{memory_items}")),
            patch.object(cons, "invoke_json", AsyncMock(return_value={"summary": "好"})),
            patch.object(cons, "get_utility_model", lambda: object()),
            patch.object(cons, "store_memory", AsyncMock()) as store_mock,
            patch.object(cons.memory_repo, "update", update_mock),
        ):
            stats = await cons.compress_l3_clusters_for_workspace(
                user_id="u1", workspace_id="ws1",
            )

        assert stats["digests"] == 0
        store_mock.assert_not_awaited()
        update_mock.assert_not_awaited()

    async def test_store_failure_keeps_originals(self):
        """digest 入库失败 (dedup/异常) → 原始行绝不归档."""
        rows = [_cand(f"m{i}", [1.0, 0.0]) for i in range(5)]
        update_mock = AsyncMock()
        with (
            patch.object(cons, "_load_candidates", AsyncMock(side_effect=[rows, []])),
            patch.object(cons, "get_prompt_text", AsyncMock(return_value="{owner}{main_category}{sub_category}{memory_items}")),
            patch.object(cons, "invoke_json", AsyncMock(return_value={"summary": "十月里我常在早晨散步、买咖啡，偶尔去河边拍照"})),
            patch.object(cons, "get_utility_model", lambda: object()),
            patch.object(cons, "store_memory", AsyncMock(return_value=None)),
            patch.object(cons.memory_repo, "update", update_mock),
        ):
            stats = await cons.compress_l3_clusters_for_workspace(
                user_id="u1", workspace_id="ws1",
            )

        assert stats["digests"] == 0
        update_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduler_job_noops_when_flag_disabled(monkeypatch):
    from jobs import scheduler as sched

    monkeypatch.setattr(sched, "_run_distributed_job", AsyncMock())
    from app.config import settings
    monkeypatch.setattr(settings, "memory_consolidation_enabled", False)

    await sched._run_memory_consolidation()
    sched._run_distributed_job.assert_not_awaited()
