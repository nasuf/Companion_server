"""Actual hybrid retrieval on migrated disposable PG/Redis and local embeddings.

Both arms initially run the same implementation. This is a repeatability and
measurement baseline, not evidence of an algorithm improvement. Embeddings are
frozen before timing; latency is retrieval-only, not end-to-end chat latency.
"""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from datetime import datetime
import json
import math
from pathlib import Path
import time
from unittest.mock import patch
from uuid import uuid4

from .dataset import fingerprint
from .manifest import conditions


class CountedDatabase:
    def __init__(self, db):
        self.client, self.calls = db, 0
        self.failures = []

    def __getattr__(self, key):
        return getattr(self.client, key)

    async def query_raw(self, *args, **kwargs):
        self.calls += 1
        try:
            return await self.client.query_raw(*args, **kwargs)
        except Exception as exc:
            self.failures.append(type(exc).__name__)
            raise


async def fixture(db, case, vectors):
    """Unique owners/workspaces; no existing rows or production IDs are reused."""
    owners, agents, spaces, memory_keys = [], [], {}, {}
    try:
        for scope in ("target", "other_agent", "other_user"):
            if scope != "other_agent":
                user = await db.user.create(
                    data={"username": "memory-eval-" + uuid4().hex}
                )
                owners.append(user.id)
            owner = owners[-1]
            agent = await db.aiagent.create(data={"userId": owner, "name": "Synthetic"})
            agents.append(agent.id)
            # Ordinary users have one active companion. A previous companion
            # remains in an archived workspace belonging to the same owner.
            space = await db.chatworkspace.create(
                data={
                    "userId": owner,
                    "agentId": agent.id,
                    "status": "archived" if scope == "other_agent" else "active",
                }
            )
            spaces[scope] = (owner, space.id)
        for m in case["memories"]:
            owner, workspace = spaces[m["scope"]]
            model = db.aimemory if m["source"] == "ai" else db.usermemory
            data = {
                "userId": owner,
                "workspaceId": workspace,
                "content": m["text"],
                "level": m["level"],
                "importance": (
                    0.9 if m["level"] == 1 else 0.7 if m["level"] == 2 else 0.4
                ),
                "mainCategory": m["main"],
                "subCategory": m["sub"],
                "isArchived": m.get("archived", False),
                "createdAt": datetime.fromisoformat(
                    m.get("created_at", "2026-10-01T12:00:00+08:00")
                ),
                "updatedAt": datetime.fromisoformat(
                    m.get("created_at", "2026-10-01T12:00:00+08:00")
                ),
            }
            if m.get("occur_time"):
                data["occurTime"] = datetime.fromisoformat(m["occur_time"])
            row = await model.create(data=data)
            memory_keys[row.id] = m["key"]
            await db.execute_raw(
                "INSERT INTO memory_embeddings(memory_id,embedding) VALUES($1,$2::extensions.vector)",
                row.id,
                json.dumps(vectors[m["text"]]),
            )
        return owners, agents, spaces, memory_keys
    except BaseException:
        await cleanup(db, owners, agents, spaces, memory_keys)
        raise


async def cleanup(db, owners, agents, spaces, memory_keys):
    if memory_keys:
        await db.execute_raw(
            "DELETE FROM memory_embeddings WHERE memory_id=ANY($1::text[])",
            list(memory_keys),
        )
    if owners:
        for model in (db.aimemory, db.usermemory):
            await model.delete_many(where={"userId": {"in": owners}})
    for _, wid in spaces.values():
        await db.chatworkspace.delete_many(where={"id": wid})
    if agents:
        await db.aiagent.delete_many(where={"id": {"in": agents}})
    if owners:
        await db.user.delete_many(where={"id": {"in": owners}})


async def embeddings(cases, model, output: Path):
    """Content/model/digest-bound disk cache, never reuse a different model's vectors."""
    import httpx
    from app.config import settings

    async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
        tags = (
            await client.get(settings.ollama_base_url.rstrip("/") + "/api/tags")
        ).json()["models"]
    matches = [
        m
        for m in tags
        if m["name"] == settings.embedding_model
        or m["name"] == settings.embedding_model + ":latest"
    ]
    if len(matches) != 1:
        raise ValueError("Exact local embedding model/digest unavailable")
    digest = matches[0]["digest"]
    texts = sorted(
        {m["text"] for c in cases for m in c["memories"]} | {c["query"] for c in cases}
    )
    identity = fingerprint([settings.embedding_model, digest, texts])
    cache_path = output / "embeddings.private.json"
    cached = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if cached.get("identity") == identity:
        vectors = cached["vectors"]
    else:
        vectors = {}
        # aembed_query matches generate_embedding(), including any provider-specific
        # query prefix. The original seed embedding service uses that same API.
        for index, text in enumerate(texts):
            vectors[text] = await model.aembed_query(text)
            if index % 40 == 0:
                print(
                    json.dumps({"embedding_progress": index, "total": len(texts)}),
                    flush=True,
                )
        cache_path.write_text(json.dumps({"identity": identity, "vectors": vectors}))
        cache_path.chmod(0o600)
    if set(vectors) != set(texts) or any(
        len(v) != settings.embedding_dimensions
        or any(
            not isinstance(x, (int, float))
            or isinstance(x, bool)
            or not math.isfinite(x)
            for x in v
        )
        for v in vectors.values()
    ):
        raise ValueError("Incomplete embedding cache or wrong dimensions")
    return vectors, digest


async def run(cases, manifest, output: Path, samples=1):
    from app.config import settings
    from app.db import db
    from app.redis_client import get_redis, close_redis
    from app.services.llm.models import get_embedding_model
    from app.services.memory.retrieval import hybrid, vector_search, relevance, ranking
    from app.services.schedule_domain import time_parser
    from app.services.runtime import cache
    from .metrics import summarize

    if (
        settings.embedding_model != manifest["embedding"]["model"]
        or settings.embedding_dimensions != manifest["embedding"]["dimensions"]
    ):
        raise ValueError("Effective embedding model differs from frozen baseline")
    output.mkdir(parents=True, exist_ok=True)
    vectors, digest = await embeddings(cases, get_embedding_model(), output)
    if digest != manifest["embedding"]["digest"]:
        raise ValueError(
            "Local embedding digest differs from qualified production model"
        )
    redis = None
    counted = CountedDatabase(db)
    rows = []
    prefix = "memory-eval:" + uuid4().hex + ":"
    clock = datetime.fromisoformat(manifest["clock"])

    class FrozenDateTime:
        def now(self, tz=None):
            return clock.astimezone(tz) if tz else clock.replace(tzinfo=None)

        def __getattr__(self, name):
            return getattr(datetime, name)

        def __instancecheck__(self, obj):
            return isinstance(obj, datetime)

    async def vector(text):
        return vectors[text]

    try:
        await db.connect()
        redis = await get_redis()
        await redis.ping()
        with ExitStack() as stack:
            # Patch only IO measurement, frozen time and precomputed model vectors;
            # production SQL, scope filters, thresholds, fusion/ranking/selection
            # and Redis serialization/invalidation execute unchanged.
            stack.enter_context(patch.object(vector_search, "db", counted))
            from app.services.memory.storage import entity_repo

            stack.enter_context(patch.object(entity_repo, "db", counted))
            stack.enter_context(
                patch.object(vector_search, "generate_embedding", vector)
            )
            stack.enter_context(patch.object(cache, "CACHE_PREFIX", prefix + "cache:"))
            stack.enter_context(
                patch.object(cache, "VERSION_PREFIX", prefix + "version:")
            )
            for module in (relevance, ranking):
                stack.enter_context(patch.object(module, "datetime", FrozenDateTime()))
            stack.enter_context(
                patch.object(time_parser, "_now_corrected", lambda: clock)
            )
            for ci, case in enumerate(cases):
                owners, agents, spaces, keys = await fixture(db, case, vectors)
                owner, wid = spaces["target"]
                frozen = conditions(manifest, case, digest)
                try:
                    scope_violations = []
                    if case["operation"] == "parallel":
                        other_owner, other_wid = spaces["other_agent"]
                        a, b = await asyncio.gather(
                            hybrid.hybrid_retrieve(
                                case["query"], owner, workspace_id=wid
                            ),
                            hybrid.hybrid_retrieve(
                                case["query"], other_owner, workspace_id=other_wid
                            ),
                        )
                        for result, scope in ((a, "target"), (b, "other_agent")):
                            allowed = {
                                m["key"]
                                for m in case["memories"]
                                if m["scope"] == scope and not m.get("archived")
                            }
                            if any(
                                keys.get(m.id) not in allowed
                                for m in result.get("memories") or []
                            ):
                                scope_violations.append("parallel_scope_contamination")
                    for sample in range(samples):
                        # Alternate order to avoid always rewarding one arm for
                        # process/DB warmup. Retrieval caches reset per arm.
                        arms = (
                            ("baseline", "candidate")
                            if (ci + sample) % 2 == 0
                            else ("candidate", "baseline")
                        )
                        for arm in arms:
                            await cache.bump_cache_version(owner, wid)
                            for mode in ("cold", "warm"):
                                candidates = []

                                def capture(**kwargs):
                                    candidates.extend(
                                        m["id"] for m in kwargs.get("candidates", [])
                                    )

                                start = time.perf_counter()
                                before = counted.calls
                                before_failures = len(counted.failures)
                                error = None
                                result = {}
                                violations = list(scope_violations)
                                try:
                                    with patch.object(
                                        hybrid, "record_retrieval_session", capture
                                    ):
                                        result = await hybrid.hybrid_retrieve(
                                            case["query"],
                                            owner,
                                            workspace_id=wid,
                                            token_budget=manifest["retrieval"][
                                                "token_budget"
                                            ],
                                        )
                                except Exception as exc:
                                    error = type(exc).__name__
                                if len(counted.failures) > before_failures:
                                    error = "retrieval_database_failure"
                                selected = [
                                    keys.get(m.id, "unknown:" + m.id)
                                    for m in result.get("memories") or []
                                ]
                                candidate_ids = (
                                    result.get("candidate_ids") or candidates
                                )
                                row = {
                                    "case": case["id"],
                                    "sample": sample,
                                    "arm": arm,
                                    "cache": mode,
                                    "conditions": frozen,
                                    "stored": None,
                                    "candidates": list(
                                        dict.fromkeys(
                                            keys.get(mid, "unknown:" + mid)
                                            for mid in candidate_ids
                                        )
                                    ),
                                    "code_tree_sha256": manifest["code_tree_sha256"],
                                    "selected": selected,
                                    "used": None,
                                    "latency_ms": (time.perf_counter() - start) * 1000,
                                    "db_calls": counted.calls - before,
                                    "llm_calls": 0,
                                    "llm_cost_cny": 0.0,
                                    "error": error,
                                    "network_violations": violations,
                                    "embedding_inference_in_latency": False,
                                    "fixture_seeded": True,
                                }
                                row["fixture_ids"] = [
                                    m["key"]
                                    for m in case["memories"]
                                    if m["scope"] == "target" and not m.get("archived")
                                ]
                                rows.append(row)
                                with (output / "observations.jsonl").open("a") as file:
                                    file.write(
                                        json.dumps(row, ensure_ascii=False) + "\n"
                                    )
                finally:
                    await cleanup(db, owners, agents, spaces, keys)
                if ci % 20 == 0:
                    print(
                        json.dumps({"retrieval_progress": ci + 1, "total": len(cases)}),
                        flush=True,
                    )
        report = summarize(cases, rows, samples=samples)
        if counted.failures:
            report["instrumentation_passed"] = False
            report["complete"] = False
            report["errors"].append("retrieval_database_failure")
        report.update(
            embedding_digest=digest,
            scope="fixture-seeded actual hybrid PG/Redis retrieval; no extraction or chat send",
            cache_invalidation_verified=all(
                r["db_calls"] > 0 for r in rows if r["cache"] == "cold"
            )
            and all(r["db_calls"] == 0 for r in rows if r["cache"] == "warm"),
            source_manifest_sha256=fingerprint(manifest),
        )
        report["embedding_vectors_sha256"] = fingerprint(vectors)
        if not report["cache_invalidation_verified"]:
            report["complete"] = False
            report["instrumentation_passed"] = False
            report["errors"].append("cache_invalidation_failure")
        (output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        return report
    finally:
        # Only this run's keys; never FLUSHDB/FLUSHALL, never a shared prefix.
        try:
            if redis is not None:
                async for key in redis.scan_iter(match=prefix + "*", count=100):
                    await redis.delete(key)
        finally:
            await close_redis()
            if db.is_connected():
                await db.disconnect()
