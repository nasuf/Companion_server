"""One-shot backfill for ai_agents.source_template_id.

clone_template_agent_for_user records source_template_id via best-effort raw
SQL; clones created before the column existed (or whose write failed) carry
NULL and are invisible to the template knowledge sync (they would silently
miss published knowledge memories).

Conservative matching — a wrong link is worse than a missing one:

- candidate templates = agents owned by the template system user
  (__companion_template_system__)
- a NULL-source agent is linked only when its (name, gender, background)
  matches EXACTLY ONE template; ambiguous keys are reported and skipped
- rows owned by the template system user itself are never touched

Usage (from Companion_server/):
    .venv/bin/python scripts/backfill_source_template_id.py           # dry-run
    .venv/bin/python scripts/backfill_source_template_id.py --apply   # write
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import db, ensure_connected  # noqa: E402
from app.services.agent_template.registry import TEMPLATE_SYSTEM_USERNAME  # noqa: E402


def _match_key(name: object, gender: object, background: object) -> tuple[str, str, str]:
    return (
        str(name or "").strip(),
        str(gender or "").strip(),
        str(background or "").strip(),
    )


async def run(apply: bool) -> None:
    await ensure_connected()

    owner = await db.user.find_unique(where={"username": TEMPLATE_SYSTEM_USERNAME})
    if owner is None:
        print("No template system user exists — nothing to backfill.")
        return

    templates = await db.aiagent.find_many(where={"userId": owner.id})
    if not templates:
        print("Template system user owns no templates — nothing to backfill.")
        return

    # Build the persona-fingerprint index; keys shared by 2+ templates are
    # ambiguous and must never be used for linking.
    key_to_templates: dict[tuple[str, str, str], list[str]] = {}
    for template in templates:
        key = _match_key(template.name, template.gender, template.background)
        key_to_templates.setdefault(key, []).append(template.id)
    ambiguous = {key for key, ids in key_to_templates.items() if len(ids) > 1}
    if ambiguous:
        print(f"WARNING: {len(ambiguous)} ambiguous template fingerprint(s) skipped:")
        for key in ambiguous:
            print(f"  name={key[0]!r} gender={key[1]!r} (background elided)")

    # Raw SQL: the generated Prisma client may predate source_template_id.
    candidates = await db.query_raw(
        """
        SELECT id, name, gender, background
        FROM ai_agents
        WHERE source_template_id IS NULL AND user_id <> $1
        """,
        owner.id,
    )
    print(f"Templates: {len(templates)}  |  NULL-source candidate agents: {len(candidates)}")

    linked = 0
    skipped_no_match = 0
    skipped_ambiguous = 0
    for row in candidates:
        key = _match_key(row.get("name"), row.get("gender"), row.get("background"))
        if key in ambiguous:
            skipped_ambiguous += 1
            continue
        template_ids = key_to_templates.get(key)
        if not template_ids:
            skipped_no_match += 1
            continue
        template_id = template_ids[0]
        if apply:
            await db.execute_raw(
                "UPDATE ai_agents SET source_template_id = $1 WHERE id = $2",
                template_id,
                row["id"],
            )
        linked += 1
        if linked <= 10:
            print(f"  link agent {row['id'][:8]} -> template {template_id[:8]} ({key[0]})")

    mode = "APPLIED" if apply else "DRY-RUN (use --apply to write)"
    print(
        f"[{mode}] linked={linked}  no_match={skipped_no_match}  "
        f"ambiguous={skipped_ambiguous}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write updates (default: dry-run)")
    args = parser.parse_args()
    asyncio.run(run(apply=args.apply))


if __name__ == "__main__":
    main()
