"""Freeze code, prompt policies, dataset membership and comparable conditions."""

from __future__ import annotations

from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import subprocess

from .dataset import fingerprint, split_cases, SEED

ROOT = Path(__file__).resolve().parents[2]
NOW = "2026-10-10T12:00:00+08:00"


def source_files() -> dict:
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in sorted(set(names))
        if name.startswith(("app/", "jobs/", "prisma/", "evals/"))
        and (ROOT / name).is_file()
    }


def verify_source(manifest: dict) -> None:
    if manifest["code_files"] != source_files() or manifest[
        "code_tree_sha256"
    ] != fingerprint(manifest["code_files"]):
        raise ValueError("Source drift: freeze a new manifest before running")


def freeze(cases: list[dict], production: dict) -> dict:
    """The input is a private native read-only snapshot, never ORM test access."""
    if production.get("business_mutations") != 0 or production.get("read_only") != "on":
        raise ValueError("Qualified read-only baseline snapshot required")
    source = source_files()
    from app.services.memory.retrieval import hybrid, context_selector
    from app.services.memory.lifecycle import value
    from app.config import settings

    env = production["environment"]
    model_name = env.get("EMBEDDING_MODEL", settings.embedding_model)
    embedding_models = [
        m
        for m in production.get("embedding_runtime", {}).get("models", [])
        if m["name"] in {model_name, model_name + ":latest"}
    ]
    if len(embedding_models) != 1 or not embedding_models[0].get("digest"):
        raise ValueError("Qualified native production embedding digest is required")
    prompts = {
        p["key"]: {
            "content_sha256": hashlib.sha256(p["content"].encode()).hexdigest(),
            "enabled": p["enabled"],
            "revision": p["revision"],
            "history_sha256": fingerprint(p["history"]),
        }
        for p in production["prompts"]
    }
    return {
        "schema_version": 1,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        ),
        "code_files": source,
        "code_tree_sha256": fingerprint(source),
        "production_image": production["image"],
        "production_observed_at": production["verified_at"],
        "dataset_sha256": fingerprint(cases),
        "split": split_cases(cases),
        "seed": SEED,
        "clock": NOW,
        "timezone": "Asia/Shanghai",
        "cache_modes": ["cold", "warm"],
        "evaluation_model_call": {
            "temperature": 0,
            "max_tokens": 512,
            "prompt_key": "memory.strong_reply",
            "n": 1,
            "max_per": 60,
            "max_total": 150,
            "judge_thinking": False,
        },
        "embedding": {
            "model": model_name,
            "digest": embedding_models[0]["digest"],
            "provider": env.get("EMBEDDING_PROVIDER", settings.embedding_provider),
            "dimensions": int(
                env.get("EMBEDDING_DIMENSIONS", settings.embedding_dimensions)
            ),
        },
        "models": production["system_config"],
        "agent_override_count": production["agent_override_count"],
        "prices": production["models"],
        "prompts": prompts,
        "migrations": production["migrations"],
        "retrieval": {
            "token_budget": 800,
            "similarity_threshold": hybrid._SIMILARITY_THRESHOLD,
            "l3_threshold": hybrid.WARM_SAMPLE_THRESHOLD,
            "l3_budget": hybrid.WARM_SAMPLE_BUDGET,
            "per_source_limit": context_selector.MAX_MEMORIES_PER_SOURCE,
            "per_item_token_limit": context_selector.MAX_MEMORY_TOKENS_PER_ITEM,
        },
        "lifecycle": {
            "half_life_days": value.HALF_LIFE_DAYS,
            "access_reward": value.ACCESS_REWARD,
            "contribution_reward": value.CONTRIBUTION_REWARD,
        },
        "packages": {
            n: version(n)
            for n in ("prisma", "redis", "langchain-core", "langgraph", "httpx")
        },
        "execution_scope": "synthetic global model policy; per-agent production overrides are not simulated",
        "external_comparators": {
            "mem0_oss": {"status": "not_run", "commit": None},
            "mem0_platform": {"status": "not_run", "uploaded_records": 0},
        },
    }


def conditions(manifest: dict, case: dict, embedding_digest: str) -> str:
    """Common conditions; implementation identity is retained separately per arm."""
    return fingerprint(
        {
            "dataset": manifest["dataset_sha256"],
            "case": case["id"],
            "clock": manifest["clock"],
            "retrieval": manifest["retrieval"],
            "models": manifest["models"],
            "prompts": manifest["prompts"],
            "evaluation_model_call": manifest["evaluation_model_call"],
            "embedding": manifest["embedding"],
            "embedding_digest": embedding_digest,
        }
    )
