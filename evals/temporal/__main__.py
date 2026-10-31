"""One explicit entry point for temporal truth, replay and model evaluation.

Native reports retain their own denominators and qualification rules. Successful
collection is never relabelled as a model-quality pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SUITES = {
    "contract": {"layer": "truth", "module": "evals.temporal.contract.run_eval"},
    "replay": {
        "layer": "replay",
        "tests": [
            "tests/test_temporal_context_contract.py",
            "tests/test_temporal_anchor_evidence.py",
            "tests/test_temporal_evidence_prompt.py",
            "tests/test_temporal_recall.py",
            "tests/test_followup_preference_order.py",
            "tests/test_reminder_confirmation_clock.py",
            "tests/test_temporal_domain_convergence.py",
            "tests/test_direct_reminder_clock.py",
        ],
    },
    "binding": {"layer": "model", "module": "evals.temporal.binding.run"},
    "dialogue": {"layer": "model", "module": "evals.temporal.dialogue.run"},
    "incident": {
        "layer": "model",
        "module": "evals.temporal.awareness.run_incident_eval",
    },
    # Historical DB-backed evaluators remain available through their original
    # explicit CLI, not silently executed by the isolated release gate.
    "awareness": {
        "layer": "model",
        "module": "evals.temporal.awareness.run_eval",
        "requires_domain_io": True,
    },
    "recall": {
        "layer": "model",
        "module": "evals.temporal.recall.run_eval",
        "requires_domain_io": True,
    },
}


def inventory() -> dict:
    manifest = json.loads(Path(__file__).with_name("corpus_manifest.json").read_text())
    rows = []
    for row in manifest["files"]:
        path = ROOT / row["new"]
        digest = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
        rows.append({**row, "actual_sha256": digest, "ok": digest == row["sha256"]})
    return {
        "version": 1,
        "ok": all(row["ok"] for row in rows),
        "files": rows,
        "suites": SUITES,
        "evidence_corpus": "evals/temporal/evidence",
        "counting": "Keep original IDs, partitions and suite denominators; never sum repeated scenarios.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--suite", choices=SUITES)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--live", action="store_true", help="Explicit model API cost opt-in"
    )
    args, native = parser.parse_known_args()
    if native[:1] == ["--"]:
        native = native[1:]
    data = inventory()
    if args.inventory:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0 if data["ok"] else 1
    if not args.suite or args.output_dir is None:
        parser.error("--suite and a fresh --output-dir are required")
    if not data["ok"]:
        parser.error("Frozen corpus integrity failed")
    suite = SUITES[args.suite]
    if suite.get("requires_domain_io"):
        parser.error(
            "This historical evaluator requires explicit isolated domain IO; use its documented CLI"
        )
    if suite["layer"] == "model" and not args.live:
        parser.error(
            "Model evaluation requires --live; collection is not qualification"
        )
    if args.output_dir.exists():
        parser.error("Preserve prior evidence; choose a new output directory")
    # Never allow native flags to replace controlled output paths.
    if any(
        x == "--output" or x.startswith("--output=") or x.startswith("--junitxml")
        for x in native
    ):
        parser.error("Output is owned by --output-dir")
    args.output_dir.mkdir(parents=True, mode=0o700)
    if "tests" in suite:
        if native:
            parser.error(
                "Replay selection is fixed; use pytest directly for exploratory subsets"
            )
        command = [
            sys.executable,
            "-m",
            "pytest",
            *suite["tests"],
            "-q",
            "--junitxml=" + str(args.output_dir.resolve() / "replay.xml"),
        ]
    else:
        output = args.output_dir.resolve() / (
            "native.jsonl" if args.suite == "dialogue" else "native.json"
        )
        command = [
            sys.executable,
            "-m",
            suite["module"],
            *native,
            "--output",
            str(output),
        ]
        if suite["layer"] == "model":
            command.append("--live")
    with (args.output_dir / "execution.log").open("x") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    status = (
        "failed"
        if result.returncode
        else (
            "collected_pending_native_quality_review"
            if suite["layer"] == "model"
            else "passed"
        )
    )
    report = {
        "version": 1,
        "suite": args.suite,
        "layer": suite["layer"],
        "status": status,
        "exit_code": result.returncode,
        "corpus_integrity": data,
        "qualification": "Model native report and original gates required; no aggregate quality claim",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"status": status, "output_dir": str(args.output_dir)}))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
