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


def test_candidate_query_skips_already_consolidated_rows():
    """批量归档是原子的, 但"摘要已建、还没走到归档"之间进程崩掉的话, 原行仍未
    归档 —— 下一轮会把同一簇再压一次产出重复摘要。changelog 先于归档写入, 用它
    覆盖这个崩溃窗口。"""
    import inspect

    source = inspect.getsource(cons._load_candidates)
    assert "operation = 'consolidated_into'" in source


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "prepare", "model", "embedding", "commit", "short", "invalid", "long", "identity", "model_config"])
async def test_compression_prepares_before_model_and_commits_once(monkeypatch, failure):
    rows = [_cand(f"m{i}", [1.0, 0.0], summary=f"第{i}天午休听了一段音乐") for i in range(5)]
    prepared = {"parents": rows, "side": "ai"}
    prepare = AsyncMock(return_value=prepared, side_effect=RuntimeError("prepare") if failure=="prepare" else None)
    model = AsyncMock(return_value={"summary": "那段时间常在午休听一段音乐"}, side_effect=RuntimeError("model") if failure=="model" else None)
    if failure=="short": model.return_value={"summary":"好"}
    if failure=="invalid": model.return_value={"summary":["wrong type"]}
    if failure=="long": model.return_value={"summary":"音乐"*1000}
    embedding = AsyncMock(return_value=[.1]*1024, side_effect=RuntimeError("embedding") if failure=="embedding" else None)
    commit = AsyncMock(return_value="digest", side_effect=RuntimeError("commit") if failure=="commit" else None)
    for name,value in {"prepare_cluster":prepare,"invoke_json":model,"generate_embedding":embedding,
                       "commit_cluster":commit,"_generated_self_memory_allowed":AsyncMock(return_value=failure!="identity"),
                       "get_prompt_text":AsyncMock(return_value="{owner}{main_category}{sub_category}{memory_items}")}.items():
        monkeypatch.setattr(cons,name,value)
    def utility():
        if failure=="model_config":raise ValueError("synthetic invalid model configuration")
        return object()
    monkeypatch.setattr(cons,"get_utility_model",utility)
    result=await cons._compress_cluster(source="ai",user_id="u",workspace_id="w",cluster=rows)
    assert result==("digest" if failure is None else None)
    if failure=="prepare":model.assert_not_awaited()
    if failure is None:
        assert commit.await_count==1
        assert commit.await_args.kwargs["inputs"] is prepared
        assert commit.await_args.kwargs["digest"]=="那段时间常在午休听一段音乐"
        assert all(r["content"] in commit.await_args.kwargs["prompt"] for r in rows)
    elif failure!="commit":commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduler_job_noops_when_flag_disabled(monkeypatch):
    from jobs import scheduler as sched

    monkeypatch.setattr(sched, "_run_distributed_job", AsyncMock())
    from app.config import settings
    monkeypatch.setattr(settings, "memory_consolidation_enabled", False)

    await sched._run_memory_consolidation()
    sched._run_distributed_job.assert_not_awaited()
