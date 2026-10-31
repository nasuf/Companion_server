"""Exercise the files actually selected by Docker COPY, not the full checkout."""

from pathlib import Path
import os
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_production_image_contains_canonical_temporal_evaluation_dependencies(tmp_path):
    image = tmp_path / "image"
    image.mkdir()
    for line in (ROOT / "Dockerfile").read_text().splitlines():
        if not line.startswith("COPY "):
            continue
        *patterns, target = shlex.split(line)[1:]
        destination = image / target
        for pattern in patterns:
            sources = sorted(ROOT.glob(pattern))
            assert sources, pattern
            for source in sources:
                if source.is_dir():
                    shutil.copytree(
                        source,
                        destination,
                        dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"),
                    )
                else:
                    destination.mkdir(parents=True, exist_ok=True)
                    # Preserve symlinks to detect dangling canonical corpus paths.
                    copied = destination / source.name
                    if source.is_symlink():
                        copied.symlink_to(os.readlink(source))
                    else:
                        shutil.copy2(source, copied)
    script = """
import importlib, pathlib, sys
sys.path.insert(0, sys.argv[1])
for name in ('cases', 'incident_cases', 'judge', 'standard', 'run_eval', 'run_incident_eval'):
    old = importlib.import_module('evals.temporal_awareness.' + name)
    new = importlib.import_module('evals.temporal.awareness.' + name)
    assert old is new, name
for name in ('binding/cases.json', 'evidence/cases.json', 'dialogue/cases.jsonl'):
    old = pathlib.Path(sys.argv[1], 'evals/temporal_' + name)
    new = pathlib.Path(sys.argv[1], 'evals/temporal', name)
    assert old.is_file() and old.read_bytes() == new.read_bytes(), name
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(image)],
        cwd=image,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
