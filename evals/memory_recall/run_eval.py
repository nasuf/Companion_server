"""Memory recall eval runner.

Runs the REAL ranking + selection code (`rank_memory_candidate` +
`select_context`) over the case bank with real embeddings, no DB.

Usage (from Companion_server/):
    .venv/bin/python -m evals.memory_recall.run_eval                 # full run
    .venv/bin/python -m evals.memory_recall.run_eval --json out.json
    .venv/bin/python -m evals.memory_recall.run_eval --baseline old.json

Requires local Ollama with the configured embedding model (bge-m3). Query and
seed embeddings are cached to .emb_cache.json for fast re-runs while tuning.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Protocol

from app.services.memory.normalization import cosine_similarity
from app.services.memory.retrieval.context_selector import estimate_tokens, select_context
from app.services.memory.retrieval.ranking import rank_memory_candidate

from evals.memory_recall.cases import CASES, SEED_BANK, RecallCase, SeedMemory

# Mirrors hybrid.py's vector-arm gate — keep in sync (guard test exists).
SIMILARITY_THRESHOLD = 0.35

_CACHE_PATH = Path(__file__).parent / ".emb_cache.json"


class Embedder(Protocol):
    def __call__(self, texts: list[str]) -> Awaitable[list[list[float]]]: ...


@dataclass
class CaseResult:
    case_id: str
    group: str
    hit: bool
    missing: list[str]
    contamination: list[str]
    selected_ids: list[str]
    tokens: int


def _candidate_from_seed(seed: SeedMemory, similarity: float) -> dict:
    return {
        "id": seed.id,
        "content": seed.text,
        "summary": seed.text,
        "level": seed.level,
        "importance": seed.importance,
        "similarity": similarity,
        "source": seed.source,
        "main_category": seed.main,
        "sub_category": seed.sub,
        "created_at": "2026-06-01T00:00:00+00:00",
        "last_accessed_at": "2026-07-01T00:00:00+00:00",
    }


async def evaluate_cases(
    embed: Embedder,
    cases: tuple[RecallCase, ...] = CASES,
    seed_bank: tuple[SeedMemory, ...] = SEED_BANK,
) -> dict:
    """Run all cases; returns the metrics dict (also used by the CI smoke test)."""
    seeds_by_id = {s.id: s for s in seed_bank}
    seed_texts = [s.text for s in seed_bank]
    seed_vecs = await embed(seed_texts)
    vec_by_id = {s.id: v for s, v in zip(seed_bank, seed_vecs)}

    query_texts = [c.enhanced_query or c.query for c in cases]
    query_vecs = await embed(query_texts)

    results: list[CaseResult] = []
    for case, q_vec in zip(cases, query_vecs):
        bank = (
            [seeds_by_id[sid] for sid in case.seeds] if case.seeds else list(seed_bank)
        )
        candidates = []
        for seed in bank:
            sim = cosine_similarity(q_vec, vec_by_id[seed.id])
            if sim < SIMILARITY_THRESHOLD:
                continue
            candidates.append(_candidate_from_seed(seed, sim))

        # Real ranking layer (query = the user's literal message, as in prod).
        for cand in candidates:
            score, reasons = rank_memory_candidate(cand, case.query)
            cand["rank_score"] = score
            cand["rank_reasons"] = reasons
        candidates.sort(key=lambda c: float(c.get("rank_score", 0)), reverse=True)

        selected = select_context(candidates, 800, query=case.query)
        selected_ids = [m.id for m in selected]
        missing = [h for h in case.expect_hit if h not in selected_ids]
        contamination = [m for m in case.expect_miss if m in selected_ids]
        results.append(CaseResult(
            case_id=case.id,
            group=case.group,
            hit=not missing,
            missing=missing,
            contamination=contamination,
            selected_ids=selected_ids,
            tokens=sum(estimate_tokens(m.text) for m in selected),
        ))

    by_group: dict[str, dict] = {}
    for r in results:
        g = by_group.setdefault(r.group, {"total": 0, "hits": 0, "contaminated": 0})
        g["total"] += 1
        g["hits"] += int(r.hit)
        g["contaminated"] += int(bool(r.contamination))

    total = len(results)
    hits = sum(int(r.hit) for r in results)
    return {
        "total_cases": total,
        "recall_rate": round(hits / total, 4) if total else 0.0,
        "contamination_rate": round(
            sum(int(bool(r.contamination)) for r in results) / total, 4,
        ) if total else 0.0,
        "avg_tokens": round(sum(r.tokens for r in results) / total, 1) if total else 0.0,
        "groups": {
            g: {**v, "recall": round(v["hits"] / v["total"], 4)}
            for g, v in sorted(by_group.items())
        },
        "failures": [
            {
                "case": r.case_id, "group": r.group,
                "missing": r.missing, "contamination": r.contamination,
                "selected": r.selected_ids,
            }
            for r in results
            if not r.hit or r.contamination
        ],
    }


async def _embed_via_raw_endpoint(texts: list[str]) -> list[list[float]]:
    """Fallback: per-text POST to Ollama's legacy /api/embeddings endpoint.

    Some local Ollama builds 503 on the batch /api/embed route that langchain
    uses while the legacy per-text route works fine; the eval is offline
    tooling, so slow-but-working beats failing.
    """
    import httpx

    from app.config import settings

    import asyncio as _asyncio

    base = (getattr(settings, "ollama_base_url", None) or "http://localhost:11434").rstrip("/")
    model_name = getattr(settings, "embedding_model", None) or "bge-m3"
    out: list[list[float]] = []
    # trust_env=False: localhost tooling must not route through a system proxy
    # (a configured proxy turns every request into a 503 from the proxy).
    async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
        for text in texts:
            # Local Ollama 503s while (re)loading the model or when busy —
            # retry with backoff; this is offline tooling, patience is fine.
            for attempt in range(6):
                resp = await client.post(
                    f"{base}/api/embeddings",
                    json={"model": model_name, "prompt": text},
                )
                if resp.status_code == 503 and attempt < 5:
                    await _asyncio.sleep(1.5 * (attempt + 1))
                    continue
                resp.raise_for_status()
                out.append(resp.json()["embedding"])
                break
    return out


def _cached_ollama_embedder() -> Callable[[list[str]], Awaitable[list[list[float]]]]:
    """Real embeddings with an on-disk cache keyed by model **and** text.

    The model belongs in the key. Keyed by text alone, switching embedding
    models would silently serve the old model's vectors and the eval would
    report numbers for a model it is not running.
    """
    from app.config import settings
    from app.services.llm.models import get_embedding_model

    cache: dict[str, list[float]] = {}
    if _CACHE_PATH.exists():
        try:
            cache = json.loads(_CACHE_PATH.read_text())
        except Exception:
            cache = {}

    async def embed(texts: list[str]) -> list[list[float]]:
        model_name = settings.embedding_model
        keys = [
            hashlib.md5(f"{model_name}\x00{t}".encode()).hexdigest() for t in texts
        ]
        pending = [t for t, k in zip(texts, keys) if k not in cache]
        if pending:
            try:
                model = get_embedding_model()
                vecs = await model.aembed_documents(pending)
            except Exception as e:
                print(f"batch embed failed ({e}); falling back to raw endpoint…")
                vecs = await _embed_via_raw_endpoint(pending)
            for t, v in zip(pending, vecs):
                cache[hashlib.md5(t.encode()).hexdigest()] = v
            _CACHE_PATH.write_text(json.dumps(cache))
        return [cache[k] for k in keys]

    return embed


def _print_report(metrics: dict, baseline: dict | None) -> None:
    print(f"cases: {metrics['total_cases']}  "
          f"recall: {metrics['recall_rate']:.1%}  "
          f"contamination: {metrics['contamination_rate']:.1%}  "
          f"avg_tokens: {metrics['avg_tokens']}")
    for group, g in metrics["groups"].items():
        print(f"  {group:<18} {g['hits']}/{g['total']}  recall={g['recall']:.1%}")
    if metrics["failures"]:
        print("\nfailures:")
        for f in metrics["failures"]:
            print(f"  [{f['group']}] {f['case']} missing={f['missing']} "
                  f"contamination={f['contamination']}")
    if baseline:
        delta = metrics["recall_rate"] - baseline.get("recall_rate", 0)
        print(f"\nvs baseline: recall {delta:+.1%}, "
              f"contamination {metrics['contamination_rate'] - baseline.get('contamination_rate', 0):+.1%}")


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="write metrics JSON to this path")
    parser.add_argument("--baseline", help="compare against a previous metrics JSON")
    args = parser.parse_args()

    metrics = await evaluate_cases(_cached_ollama_embedder())

    baseline = None
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text())
    _print_report(metrics, baseline)

    if args.json:
        Path(args.json).write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
        print(f"\nmetrics written to {args.json}")


if __name__ == "__main__":
    asyncio.run(_main())
