"""Opt-in paired reply smoke test for the frozen TIME02D regression cases."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evals.temporal_dialogue.run import source_digest


async def run(args):
    if args.output.exists():
        raise ValueError("Preserve every run; choose a fresh output path")
    corpus = args.cases
    harness = ROOT / "evals/temporal_awareness/run_incident_eval.py"
    identity = {
        "source_sha256": source_digest(args.source_root),
        "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(harness.read_bytes() + Path(__file__).read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(args.prompt_snapshot.read_bytes()).hexdigest(),
        "samples": args.samples, "model": args.model, "judge_model": args.judge_model,
        "temperature": .7, "max_tokens": 700, "guard_model": "qwen3.5-flash",
        "recall_model": "qwen3.5-plus", "recall_check_model": "qwen3.5-plus",
    }
    sys.path.insert(0, str(harness.parent))
    spec = importlib.util.spec_from_file_location("binding_reply_harness", harness)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = json.loads(corpus.read_text())
    await module.run(args, cases)
    report = json.loads(args.output.read_text())
    report["identity"] = identity
    report["expected"] = len(cases) * args.samples
    report["source_unchanged"] = source_digest(args.source_root) == identity["source_sha256"]
    report["complete"] = len(report["runs"]) == report["expected"] and report["source_unchanged"]
    report["full_temporal_qualification"] = False
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    args.output.chmod(0o600)
    if not report["complete"]:
        raise RuntimeError("Incomplete or stale evidence")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.json"),
                        help="Explicit frozen corpus; defaults to the binding regression cases")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--prompt-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--judge-model", default="qwen3.5-plus")
    parser.add_argument("--samples", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--live", action="store_true", required=True)
    asyncio.run(run(parser.parse_args()))
