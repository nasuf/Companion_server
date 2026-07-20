"""CI smoke for the memory recall eval machinery (no Ollama / no DB).

Uses a fake embedder with hand-assigned vectors so the pipeline mechanics
(threshold gate → ranking → selection → metrics) are asserted deterministically.
Semantic quality is measured by the real runner (evals/memory_recall/run_eval.py)
with bge-m3 embeddings — not in CI.
"""

from __future__ import annotations

import pytest

from evals.memory_recall.cases import CASES, SEED_BANK, RecallCase, SeedMemory
from evals.memory_recall.run_eval import SIMILARITY_THRESHOLD, evaluate_cases

_BANK = (
    SeedMemory("ai-color", "我喜欢雾霾蓝和燕麦色", "偏好", "审美爱好", "ai"),
    SeedMemory("user-color", "用户喜欢黑色", "偏好", "审美爱好", "user"),
    SeedMemory("noise", "我周末喜欢整理多肉", "偏好", "生活习惯", "ai"),
)

_CASES = (
    RecallCase("smoke-hit", "smoke", "你喜欢什么颜色", ("ai-color",), ("noise",)),
    RecallCase("smoke-threshold", "smoke", "完全无关的话题", (), ()),
)

# Hand-assigned vectors: query1 ≈ ai-color > user-color > noise; query2 ⊥ all.
_VECS = {
    "我喜欢雾霾蓝和燕麦色": [1.0, 0.0, 0.0],
    "用户喜欢黑色": [0.8, 0.6, 0.0],
    "我周末喜欢整理多肉": [0.2, 0.0, 0.98],
    "你喜欢什么颜色": [1.0, 0.1, 0.0],
    "完全无关的话题": [0.0, 0.0, 0.0],
}


async def _fake_embed(texts: list[str]) -> list[list[float]]:
    return [_VECS[t] for t in texts]


@pytest.mark.asyncio
async def test_eval_pipeline_mechanics():
    metrics = await evaluate_cases(_fake_embed, cases=_CASES, seed_bank=_BANK)

    assert metrics["total_cases"] == 2
    # Case 1: ai-color similar → selected; noise below usefulness but tracked.
    # Case 2: zero-vector query → nothing passes the threshold → trivially "hit"
    # (no expectations), exercising the empty-result path.
    assert metrics["recall_rate"] == 1.0
    smoke = metrics["groups"]["smoke"]
    assert smoke["total"] == 2 and smoke["hits"] == 2


@pytest.mark.asyncio
async def test_eval_reports_missing_hits():
    cases = (RecallCase("must-fail", "smoke", "完全无关的话题", ("ai-color",)),)
    metrics = await evaluate_cases(_fake_embed, cases=cases, seed_bank=_BANK)
    assert metrics["recall_rate"] == 0.0
    assert metrics["failures"][0]["missing"] == ["ai-color"]


def test_threshold_stays_in_sync_with_hybrid():
    """The eval must gate candidates exactly like production hybrid retrieval."""
    from app.services.memory.retrieval.hybrid import _SIMILARITY_THRESHOLD

    assert SIMILARITY_THRESHOLD == _SIMILARITY_THRESHOLD


def test_case_bank_integrity():
    """Every referenced memory id exists; case ids unique; bank non-trivial."""
    seed_ids = {s.id for s in SEED_BANK}
    case_ids = [c.id for c in CASES]
    assert len(case_ids) == len(set(case_ids))
    assert len(CASES) >= 20
    for case in CASES:
        for sid in case.expect_hit + case.expect_miss + case.seeds:
            assert sid in seed_ids, f"case {case.id} references unknown seed {sid}"
