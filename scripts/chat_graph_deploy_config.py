"""Write fixed graph configuration before restart for image rollback compatibility."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile


def apply_rollout(env_file: Path, config_file: Path | None = None) -> dict:
    """Write fixed compatibility flags for qualified older rollback images.

    Current application code has one graph executor and ignores these flags.
    The retired private cohort file is intentionally neither read nor changed.
    config_file remains accepted for an older deployment command only.
    """
    keys = ("CHAT_EXECUTOR", "CHAT_GRAPH_CONVERSATION_ALLOWLIST", "CHAT_GRAPH_ALL_CONVERSATIONS")
    lines = env_file.read_text(encoding="utf-8").splitlines()
    retained = [line for line in lines if line.split("=", 1)[0].strip().removeprefix("export ").strip() not in keys]
    retained.extend((f"{keys[0]}=langgraph", f"{keys[1]}=", f"{keys[2]}=true"))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=env_file.parent,
                                         prefix=".graph-env-", delete=False) as output:
            temporary = Path(output.name)
            os.chmod(temporary, 0o600)
            output.write("\n".join(retained) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, env_file)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"executor": "langgraph", "cohort_size": 0,
            "all_conversations": True, "checkpoint_enabled": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, help="Deprecated; ignored after executor retirement")
    args = parser.parse_args()
    try:
        report = apply_rollout(args.env_file, args.config_file)
    except (OSError, ValueError, TypeError):
        # Never echo private config, environment values or IDs into CI logs.
        parser.exit(2, "Cannot write fixed graph deployment configuration; server not stopped.\n")
    print("CHAT_GRAPH_DEPLOY " + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
