"""Backfill speech_usage from historical Axiom "transient chat transcription" logs.

Voice-to-text transcriptions (display_mode='text') were never persisted before
the speech_usage table existed — the only historical trace is the INFO log line:

    [speech-to-text] transient chat transcription user_id=... conversation_id=...
        duration_seconds=... model=... request_id=...

This script reconstructs those rows into speech_usage (source='backfill') so the
admin media-usage overview can report historical voice-to-text durations.

Pure-voice (display_mode='voice') is NOT backfilled here — it already lives in
chat_message_attachments and is counted from there.

Two input sources:
  --axiom      Query Axiom directly. Needs a QUERY-capable token (the server's
               AXIOM_TOKEN is ingest-only and will 403). Provide it via
               AXIOM_QUERY_TOKEN; dataset via AXIOM_QUERY_DATASET (default
               'companion-dev'). Optional AXIOM_QUERY_URL override.
  --file PATH  Read events exported from the Axiom UI. Accepts a JSON array or
               NDJSON; each item may be a raw log object or Axiom's
               {"_time": ..., "data": {"message": ...}} shape.

Idempotent: existing backfill rows are keyed by request_id (or, when absent, by
user_id+conversation_id+created_at) and skipped on re-run.

Usage (from Companion_server/):
  # dry-run (prints what would be inserted)
  AXIOM_QUERY_TOKEN=xaqt-... .venv/bin/python scripts/backfill_speech_usage.py \
      --axiom --start 2026-04-01 --end 2026-07-24
  # write
  AXIOM_QUERY_TOKEN=xaqt-... .venv/bin/python scripts/backfill_speech_usage.py \
      --axiom --start 2026-04-01 --end 2026-07-24 --apply
  # from an exported file
  .venv/bin/python scripts/backfill_speech_usage.py --file export.json --apply
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import db, ensure_connected  # noqa: E402

_DEFAULT_AXIOM_URL = "https://api.axiom.co/v1/datasets/_apl"
_DEFAULT_DATASET = "companion-dev"

# Matches the text-mode log line. request_id / model are logged via %s so they
# may be the literal "None"; duration is always an integer.
_LOG_RE = re.compile(
    r"transient chat transcription "
    r"user_id=(?P<user_id>\S+) "
    r"conversation_id=(?P<conversation_id>\S+) "
    r"duration_seconds=(?P<duration>\d+) "
    r"model=(?P<model>\S+) "
    r"request_id=(?P<request_id>\S+)"
)


@dataclass(frozen=True)
class UsageRow:
    user_id: str
    conversation_id: str
    duration_seconds: int
    model: str | None
    request_id: str | None
    created_at: datetime

    def dedup_key(self) -> str:
        if self.request_id:
            return f"rid:{self.request_id}"
        return f"combo:{self.user_id}|{self.conversation_id}|{self.created_at.isoformat()}"


def _clean(value: str | None) -> str | None:
    """Normalize logged %s placeholders: 'None'/'' → None."""
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped == "None":
        return None
    return stripped


def _parse_time(raw: object) -> datetime | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _row_from_message(message: str, when: datetime | None) -> UsageRow | None:
    match = _LOG_RE.search(message)
    if not match:
        return None
    return UsageRow(
        user_id=match.group("user_id"),
        conversation_id=match.group("conversation_id"),
        duration_seconds=int(match.group("duration")),
        model=_clean(match.group("model")),
        request_id=_clean(match.group("request_id")),
        created_at=when or datetime.now(timezone.utc),
    )


def _extract_message_and_time(item: object) -> tuple[str, datetime | None]:
    """Pull (message, _time) from an Axiom match or a raw exported log object."""
    if not isinstance(item, dict):
        return "", None
    when = _parse_time(item.get("_time"))
    data = item.get("data")
    if isinstance(data, dict):
        msg = data.get("message")
        if isinstance(msg, str):
            return msg, when or _parse_time(data.get("_time"))
    msg = item.get("message")
    if isinstance(msg, str):
        return msg, when
    return "", when


# ── Sources ───────────────────────────────────────────────────────


async def _rows_from_axiom(start: str, end: str) -> list[UsageRow]:
    token = os.environ.get("AXIOM_QUERY_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "AXIOM_QUERY_TOKEN is required for --axiom (the server's ingest-only "
            "AXIOM_TOKEN cannot query). Create a query-capable token in Axiom."
        )
    dataset = os.environ.get("AXIOM_QUERY_DATASET", _DEFAULT_DATASET).strip()
    url = os.environ.get("AXIOM_QUERY_URL", _DEFAULT_AXIOM_URL).strip()
    apl = (
        f"['{dataset}'] "
        "| where message contains 'transient chat transcription' "
        "| project _time, message "
        "| limit 500000"
    )
    body = {
        "apl": apl,
        "startTime": _to_iso(start, end_of_day=False),
        "endTime": _to_iso(end, end_of_day=True),
    }
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=body,
        )
    if resp.status_code != 200:
        raise SystemExit(f"Axiom query failed: {resp.status_code} {resp.text[:400]}")
    payload = resp.json()
    matches = payload.get("matches")
    if not isinstance(matches, list):
        raise SystemExit(
            "Unexpected Axiom response shape (no 'matches'). "
            f"Keys: {list(payload)[:10]}"
        )
    rows: list[UsageRow] = []
    for item in matches:
        message, when = _extract_message_and_time(item)
        if message:
            row = _row_from_message(message, when)
            if row:
                rows.append(row)
    return rows


def _rows_from_file(path: str) -> list[UsageRow]:
    raw = Path(path).read_text(encoding="utf-8").strip()
    items: list[object]
    if raw.startswith("["):
        items = json.loads(raw)
    else:  # NDJSON
        items = [json.loads(line) for line in raw.splitlines() if line.strip()]
    rows: list[UsageRow] = []
    for item in items:
        message, when = _extract_message_and_time(item)
        if message:
            row = _row_from_message(message, when)
            if row:
                rows.append(row)
    return rows


def _to_iso(day: str, *, end_of_day: bool) -> str:
    """Accept 'YYYY-MM-DD' or a full ISO timestamp; return RFC3339 UTC."""
    text = day.strip()
    if len(text) == 10:  # date only
        suffix = "T23:59:59Z" if end_of_day else "T00:00:00Z"
        return f"{text}{suffix}"
    return text.replace("+00:00", "Z")


# ── Write ─────────────────────────────────────────────────────────


async def _existing_backfill_keys() -> set[str]:
    rows = await db.query_raw(
        """
        SELECT request_id, user_id, conversation_id, created_at
        FROM speech_usage
        WHERE source = 'backfill'
        """
    )
    keys: set[str] = set()
    for r in rows or []:
        rid = r.get("request_id")
        if rid:
            keys.add(f"rid:{rid}")
        else:
            created = r.get("created_at")
            created_iso = (
                created.isoformat() if isinstance(created, datetime) else str(created)
            )
            keys.add(f"combo:{r['user_id']}|{r['conversation_id']}|{created_iso}")
    return keys


async def _insert(row: UsageRow) -> None:
    await db.execute_raw(
        """
        INSERT INTO speech_usage (
            user_id, conversation_id, display_mode,
            duration_seconds, model, request_id, source, created_at
        )
        VALUES ($1, $2, 'text', $3, $4, $5, 'backfill', $6::timestamptz)
        """,
        row.user_id,
        row.conversation_id,
        row.duration_seconds,
        row.model,
        row.request_id,
        row.created_at.isoformat(),
    )


async def main(args: argparse.Namespace) -> None:
    if args.axiom:
        if not args.start or not args.end:
            raise SystemExit("--axiom requires --start and --end (YYYY-MM-DD).")
        rows = await _rows_from_axiom(args.start, args.end)
    else:
        rows = _rows_from_file(args.file)

    # In-run dedup (same request_id can appear if logs were duplicated).
    unique: dict[str, UsageRow] = {}
    for row in rows:
        unique.setdefault(row.dedup_key(), row)
    parsed = list(unique.values())

    parsed_seconds = sum(r.duration_seconds for r in parsed)
    print(f"parsed rows            : {len(rows)}")
    print(f"unique rows            : {len(parsed)}")
    print(f"  → total_seconds      : {parsed_seconds} ({parsed_seconds / 60:.1f} min)")

    # Dry-run stays fully offline so parsing can be validated before the
    # speech_usage table exists (i.e. before the migration is deployed).
    if not args.apply:
        print("\nDry-run only (no DB). Re-run with --apply to write.")
        return

    await ensure_connected()
    existing = await _existing_backfill_keys()
    to_insert = [r for r in parsed if r.dedup_key() not in existing]
    print(f"already backfilled     : {len(parsed) - len(to_insert)}")
    print(f"to insert              : {len(to_insert)}")

    inserted = 0
    for row in to_insert:
        await _insert(row)
        inserted += 1
    print(f"\nApplied: inserted {inserted} rows into speech_usage (source='backfill').")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--axiom", action="store_true", help="query Axiom directly")
    source.add_argument("--file", help="path to Axiom-exported JSON/NDJSON")
    parser.add_argument("--start", help="window start YYYY-MM-DD (Axiom mode)")
    parser.add_argument("--end", help="window end YYYY-MM-DD (Axiom mode)")
    parser.add_argument("--apply", action="store_true", help="write (default dry-run)")
    asyncio.run(main(parser.parse_args()))
