"""Apply persistent Redis alert recipients without exposing IDs or secrets in logs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from uuid import UUID


def apply_config(env_file: Path, config_file: Path) -> int:
    recipients = []
    if config_file.exists() or config_file.is_symlink():
        with config_file.open("rb") as source:
            raw = source.read(4097)
        if len(raw) > 4096:
            raise ValueError("configuration exceeds size limit")
        recipients = json.loads(raw)
        if not isinstance(recipients, list) or len(recipients) > 20:
            raise ValueError("expected at most 20 recipients")
        for item in recipients:
            if not isinstance(item, str) or str(UUID(item)) != item:
                raise ValueError("recipient must be a canonical UUID")
        if len(set(recipients)) != len(recipients):
            raise ValueError("duplicate recipients")
    key = "REDIS_MEMORY_ALERT_USER_IDS"
    lines = [line for line in env_file.read_text().splitlines() if line.split("=", 1)[0].strip() != key]
    lines.append(f"{key}={','.join(recipients)}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=env_file.parent, prefix=".redis-alert-env-", delete=False) as output:
            temporary = Path(output.name)
            os.chmod(temporary, 0o600)
            output.write("\n".join(lines) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, env_file)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return len(recipients)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    args = parser.parse_args()
    try:
        count = apply_config(args.env_file, args.config_file)
    except (OSError, ValueError, TypeError):
        parser.exit(2, "Invalid or unreadable Redis alert configuration; server not stopped.\n")
    print(f"REDIS_ALERT_DEPLOY recipient_count={count}")
