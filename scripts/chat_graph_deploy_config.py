"""Apply private, persistent graph rollout configuration before server restart.

The host file lives outside the checkout/image. Missing configuration keeps all
traffic on legacy; invalid configuration aborts deployment before server stop.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from uuid import UUID


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate configuration field")
        result[key] = value
    return result


def read_rollout(path: Path) -> tuple[str, list[str], bool]:
    if not path.exists() and not path.is_symlink():
        return "legacy", [], False
    with path.open("rb") as source:
        raw = source.read(16385)
    if len(raw) > 16384:
        raise ValueError("rollout configuration exceeds size limit")
    data = json.loads(raw, object_pairs_hook=_unique_object)
    required = {"executor", "conversation_ids"}
    if not isinstance(data, dict) or not required <= set(data) or set(data) - (required | {"all_conversations"}):
        raise ValueError("rollout configuration requires executor and conversation_ids")
    executor, ids = data["executor"], data["conversation_ids"]
    all_conversations = data.get("all_conversations", False)
    if type(all_conversations) is not bool:
        raise ValueError("all_conversations must be an explicit boolean")
    if executor not in ("legacy", "langgraph"):
        raise ValueError("unsupported executor")
    if not isinstance(ids, list) or len(ids) > 100:
        raise ValueError("conversation_ids must be a list of at most 100 IDs")
    for item in ids:
        if not isinstance(item, str) or str(UUID(item)) != item:
            raise ValueError("conversation IDs must be canonical UUIDs")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate conversation IDs")
    if all_conversations and (executor != "langgraph" or ids):
        raise ValueError("full rollout requires langgraph and an empty cohort")
    if executor == "langgraph" and not ids and not all_conversations:
        raise ValueError("langgraph requires an explicit nonempty cohort")
    if executor == "legacy" and ids:
        raise ValueError("legacy rollback requires an empty cohort")
    return executor, ids, all_conversations


def apply_rollout(env_file: Path, config_file: Path) -> dict:
    executor, ids, all_conversations = read_rollout(config_file)
    keys = ("CHAT_EXECUTOR", "CHAT_GRAPH_CONVERSATION_ALLOWLIST", "CHAT_GRAPH_ALL_CONVERSATIONS")
    lines = env_file.read_text(encoding="utf-8").splitlines()
    retained = [line for line in lines if line.split("=", 1)[0].strip() not in keys]
    retained.extend((f"{keys[0]}={executor}", f"{keys[1]}={','.join(ids)}",
                     f"{keys[2]}={str(all_conversations).lower()}"))
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
    return {"executor": executor, "cohort_size": len(ids),
            "all_conversations": all_conversations, "checkpoint_enabled": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = apply_rollout(args.env_file, args.config_file)
    except (OSError, ValueError, TypeError):
        # Never echo private config, environment values or IDs into CI logs.
        parser.exit(2, "Invalid or unreadable chat graph rollout configuration; server not stopped.\n")
    print("CHAT_GRAPH_DEPLOY " + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
