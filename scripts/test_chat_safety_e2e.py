"""Run chat security/reconnect E2E against a local candidate Docker image.

Usage: python scripts/test_chat_safety_e2e.py --image companion-server:candidate
No host ports, production credentials, real DB, LLM calls or persistent volumes.
All containers use a new internal network and are removed in finally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import uuid


def run(args, *, check=True, timeout=120):
    result = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(f"Docker command failed: {result.stderr[-3000:]}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--graph", action="store_true", help="Qualify the G01 graph stream adapter")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    # Do not pull implicitly: use the image that has actually been built/reviewed.
    image = run(["image", "inspect", args.image, "--format", "{{.Id}}"]).stdout.strip()
    run(["image", "inspect", "redis:7-alpine"])
    run(["image", "inspect", "nginx:1.24-alpine"])
    root = Path(__file__).resolve().parents[1]
    harness = root / "tests" / "integration"
    suffix = uuid.uuid4().hex[:10]
    network = f"companion-chat-test-{suffix}"
    names = [f"{network}-{kind}" for kind in ("redis", "a", "b", "probe", "nginx")]
    env = {
        "APP_ENV": "test", "PYTHON_DOTENV_DISABLED": "1", "PYTHONPATH": "/verification:/tests:/app",
        "DATABASE_URL": "postgresql://synthetic:synthetic@unavailable/synthetic",
        "DIRECT_DATABASE_URL": "postgresql://synthetic:synthetic@unavailable/synthetic",
        "REDIS_URL": "redis://test-redis:6379/0", "TRACE_BACKEND": "off",
        "ONLINE_MODEL": "false", "LANGSMITH_TRACING": "false",
        "CORS_ALLOWED_ORIGINS": "https://banshengcomp.com,https://www.banshengcomp.com",
        "JWT_SECRET": "isolated-chat-e2e-secret-at-least-32-characters",
    }
    if args.graph:
        env["CHAT_GRAPH_E2E"] = "1"
    common = ["--network", network, "--read-only", "--tmpfs", "/tmp", "--cap-drop", "ALL",
              "--security-opt", "no-new-privileges", "-w", "/tmp",
              "-v", f"{harness}:/verification:ro", "-v", f"{root / 'tests'}:/tests:ro"]
    for key, value in env.items():
        common.extend(["-e", f"{key}={value}"])
    output = args.output or Path(tempfile.mkdtemp(prefix="companion-chat-e2e-"))
    output.mkdir(parents=True, exist_ok=True)
    try:
        run(["network", "create", "--internal", network])
        run(["run", "-d", "--name", names[0], "--network", network, "--network-alias", "test-redis",
             "redis:7-alpine", "redis-server", "--save", "", "--appendonly", "no"])
        for name, alias in zip(names[1:3], ("worker-a", "worker-b")):
            run(["run", "-d", "--name", name, "--network-alias", alias, *common, image,
                 "python", "-m", "uvicorn", "chat_safety_harness:app", "--host", "0.0.0.0", "--port", "8000"])
        run(["run", "-d", "--name", names[4], "--network", network,
             "--network-alias", "test-entry", "--read-only", "--tmpfs", "/tmp",
             "--user", "101:101", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
             "-v", f"{harness}:/verification:ro", "--entrypoint", "nginx", "nginx:1.24-alpine",
             "-c", "/verification/chat_safety_nginx.conf", "-g", "daemon off;"])
        result = run(["run", "--name", names[3], *common, image,
                      "python", "/verification/graph_chat_probe.py" if args.graph else "/verification/chat_safety_probe.py"], check=False)
        (output / "probe.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"E2E failed; diagnostics in {output}")
        summary = json.loads(result.stdout.strip().splitlines()[-1])
        summary.update(image=image, internal_network=True, production_credentials=False)
        (output / "result.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, ensure_ascii=False))
    finally:
        for name in names:
            state = run(["inspect", "--format", "{{json .State}}", name], check=False)
            (output / f"{name}.state.json").write_text(state.stdout or state.stderr)
            log = run(["logs", name], check=False)
            (output / f"{name}.log").write_text(log.stdout + log.stderr)
            run(["rm", "-fv", name], check=False)
        run(["network", "rm", network], check=False)


if __name__ == "__main__":
    main()
