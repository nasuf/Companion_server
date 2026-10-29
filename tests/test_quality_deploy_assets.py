"""Build inputs must survive the production upload allowlist, not just local COPY."""

from pathlib import Path
import shlex
import shutil
import subprocess


def test_production_rsync_supplies_every_offline_renderer_copy_input(tmp_path):
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/deploy.yml").read_text()
    block = workflow.split("rsync -azvcOR --stats", 1)[1].split(
        '"$VPS_USER@$VPS_HOST:/app/companion-server/"', 1
    )[0]
    patterns = [
        token
        for token in shlex.split(block.replace("\\\n", " "))
        if token.startswith(("evals/", "tests/"))
    ]
    sources = sorted(
        {
            str(path.relative_to(root))
            for pattern in patterns
            for path in root.glob(pattern)
        }
    )
    assert sources
    # Use the actual local rsync operation, with no server or credentials.
    subprocess.run(
        [shutil.which("rsync") or "rsync", "-aR", *sources, str(tmp_path)],
        cwd=root,
        check=True,
        capture_output=True,
    )
    for line in (root / "Dockerfile").read_text().splitlines():
        if not line.startswith("COPY "):
            continue
        for pattern in shlex.split(line)[1:-1]:
            if pattern.startswith(("evals/", "tests/")):
                files = list(root.glob(pattern))
                assert files, pattern
                for source in files:
                    assert (
                        tmp_path / source.relative_to(root)
                    ).read_bytes() == source.read_bytes()
    assert not list(tmp_path.rglob(".env"))
    assert not (tmp_path / "evals/results").exists()
    assert "EVAL_ARK_API_KEY" not in workflow
    assert "EVAL_DASHSCOPE_API_KEY" not in workflow
