"""M02.01 CLI. Validation needs no services; replay requires explicit local URLs."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
import os
from pathlib import Path

from .dataset import fingerprint, load_cases, split_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--validate-only", action="store_true")
    action.add_argument(
        "--freeze", type=Path, help="private native production policy snapshot"
    )
    action.add_argument("--replay", action="store_true")
    action.add_argument("--adapters", action="store_true")
    action.add_argument(
        "--grade", type=Path, help="observations JSONL, never raw production chats"
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--database-url")
    parser.add_argument("--redis-url")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=1)
    args = parser.parse_args()
    cases = load_cases()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.validate_only:
        split = split_cases(cases)
        print(
            json.dumps(
                {
                    "valid": True,
                    "cases": len(cases),
                    "groups": dict(Counter(c["group"] for c in cases)),
                    "split": dict(Counter(split.values())),
                    "dataset_sha256": fingerprint(cases),
                }
            )
        )
        return
    if not args.output:
        parser.error("--output required")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.freeze:
        from .safety import configure_isolation

        configure_isolation(
            "postgresql://synthetic:synthetic@127.0.0.1:1/companion_memory_eval_manifest",
            "redis://127.0.0.1:1/15",
        )
        from .manifest import freeze

        result = freeze(cases, json.loads(args.freeze.read_text()))
        (args.output / "manifest.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )
        print(
            json.dumps(
                {
                    "frozen": True,
                    "cases": len(cases),
                    "models": result["models"],
                    "embedding": result["embedding"],
                }
            )
        )
        return
    if not args.manifest:
        parser.error("--manifest required")
    manifest = json.loads(args.manifest.read_text())
    if manifest["dataset_sha256"] != fingerprint(cases) or manifest[
        "split"
    ] != split_cases(cases):
        raise ValueError("Dataset/split differs from frozen manifest")
    if args.grade:
        from .metrics import summarize

        rows = [
            json.loads(line)
            for line in args.grade.read_text().splitlines()
            if line.strip()
        ]
        result = summarize(cases, rows, samples=args.samples)
    elif args.adapters:
        from .manifest import verify_source
        from .safety import configure_isolation, loopback_network_fence

        verify_source(manifest)
        configure_isolation(
            "postgresql://synthetic:synthetic@127.0.0.1:1/companion_memory_eval_adapters",
            "redis://127.0.0.1:1/15",
        )
        os.environ.update(
            OLLAMA_BASE_URL="http://127.0.0.1:11434",
            EMBEDDING_MODEL=manifest["embedding"]["model"],
            EMBEDDING_DIMENSIONS=str(manifest["embedding"]["dimensions"]),
        )
        from .adapters import run

        with loopback_network_fence() as violations:
            result = asyncio.run(run(args.output, manifest))
        if violations:
            raise RuntimeError("Adapter attempted external IO")
        print(
            json.dumps(
                {
                    "adapters_completed": True,
                    "recall_cases": result["memory_recall"]["total_cases"],
                    "temporal_cases": result["temporal_recall"]["cases"],
                }
            )
        )
        return
    else:
        from .manifest import verify_source

        verify_source(manifest)
        from .safety import configure_isolation, loopback_network_fence

        if not args.database_url or not args.redis_url:
            parser.error("Explicit --database-url and --redis-url required")
        if (args.output / "observations.jsonl").exists():
            raise ValueError("Use a fresh run directory; never mix observations")
        from urllib.parse import urlsplit

        url = urlsplit(args.ollama_url)
        if (
            url.scheme != "http"
            or url.hostname not in {"127.0.0.1", "::1"}
            or url.port != 11434
            or url.path not in ("", "/")
            or url.query
            or url.username
        ):
            raise ValueError(
                "Only the explicitly selected local Ollama endpoint is allowed"
            )
        configure_isolation(args.database_url, args.redis_url)
        os.environ.update(
            OLLAMA_BASE_URL=args.ollama_url,
            EMBEDDING_MODEL=manifest["embedding"]["model"],
            EMBEDDING_PROVIDER="ollama",
            EMBEDDING_DIMENSIONS=str(manifest["embedding"]["dimensions"]),
        )
        from .replay import run

        with loopback_network_fence() as violations:
            result = asyncio.run(
                run(cases, manifest, args.output, samples=args.samples)
            )
        result["network_violations"] = violations
        if violations:
            result["instrumentation_passed"] = False
            result["complete"] = False
    (args.output / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "instrumentation_passed": result["instrumentation_passed"],
                "hard_failures": len(result["hard_failures"]),
                "model_quality": result["model_quality"],
            }
        )
    )
    if not result["instrumentation_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
