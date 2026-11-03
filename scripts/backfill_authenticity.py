"""One-off backfill of "真人判定" signals from nginx access logs on stdin.

Usage on the server (logs stay on the host; the raw address never reaches disk):
    zcat -f /var/log/nginx/access.log* | docker exec -i companion-server \
        python scripts/backfill_authenticity.py [--dry-run]

- Network: user-side chat requests (POST typing / ws-ticket) carry the
  conversation ID; the owner's IP is stored as hash + /24 + region, like live.
  Admin console traffic (referer /admin) and admin accounts are skipped.
- Compose timing: first typing ping after the previous user message, to the
  message itself (second precision), for messages that have no signal yet.
Re-runnable: existing rows are left untouched.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

_LINE = re.compile(
    r'^(?P<ip>\S+) \S+ \S+ \[(?P<at>[^\]]+)\] "(?P<method>[A-Z]+) (?P<path>\S+)[^"]*" \d+ \S+ '
    r'"(?P<referer>[^"]*)" "(?P<agent>[^"]*)"')
_PATH = re.compile(
    r"^(?:/api)?/(?:conversations/(?P<typing>[0-9a-f-]{36})/typing|chat/(?P<ticket>[0-9a-f-]{36})/ws-ticket)(?:\?|$)")


def parse(lines):
    """Yield (conversation_id, kind, ip, user_agent, utc datetime) for user-side requests."""
    for line in lines:
        match = _LINE.match(line)
        if not match or match["method"] != "POST" or "/admin" in match["referer"]:
            continue
        path = _PATH.match(match["path"])
        if not path:
            continue
        try:
            at = datetime.strptime(match["at"], "%d/%b/%Y:%H:%M:%S %z").astimezone(timezone.utc)
        except ValueError:
            continue
        kind = "typing" if path["typing"] else "ticket"
        yield path["typing"] or path["ticket"], kind, match["ip"], match["agent"], at


def compose_times(messages, pings, max_seconds=600):
    """message_id -> compose ms: first ping after the previous user message, to the send."""
    pings = sorted(pings)
    out, previous, index = {}, None, 0
    for message_id, at in messages:
        while index < len(pings) and previous is not None and pings[index] <= previous:
            index += 1
        if index < len(pings) and pings[index] <= at and (previous is None or pings[index] > previous):
            elapsed = (at - pings[index]).total_seconds()
            if elapsed <= max_seconds:
                out[message_id] = int(elapsed * 1000)
        previous = at
    return out


async def main(lines, dry_run: bool) -> dict:
    from app.db import db
    from app.services.authenticity import signals

    networks = {}
    pings = defaultdict(list)
    for conversation, kind, ip, agent, at in parse(lines):
        if kind == "typing":
            pings[conversation].append(at)
        ip = signals.client_ip({}, ip)
        if ip is None:
            continue
        key = (conversation, ip, agent)
        first, last, hits = networks.get(key, (at, at, 0))
        networks[key] = (min(first, at), max(last, at), hits + 1)
    conversations = sorted({key[0] for key in networks} | set(pings))
    owned = not db.is_connected()
    if owned:
        await db.connect()
    try:
        owners = {r["id"]: r["user_id"] for r in await db.query_raw(
            """SELECT c.id, c.user_id FROM conversations c JOIN users u ON u.id=c.user_id
               WHERE c.id=ANY($1::text[]) AND u.role<>'admin' AND u.status='active'""",
            conversations)}
        merged = {}
        for (conversation, ip, agent), (first, last, hits) in networks.items():
            user = owners.get(conversation)
            if not user:
                continue
            key = (user, ip)
            if key in merged:
                f, l, h, a = merged[key]
                merged[key] = (min(f, first), max(l, last), h + hits, agent if last > l else a)
            else:
                merged[key] = (first, last, hits, agent)
        inputs = []
        for conversation, times in pings.items():
            user = owners.get(conversation)
            if not user or not times:
                continue
            since = (min(times) - timedelta(minutes=10)).replace(tzinfo=None).isoformat()
            rows = await db.query_raw(
                """SELECT m.id, m.created_at, char_length(btrim(m.content)) AS chars,
                          s.message_id IS NOT NULL AS has_signal
                   FROM messages m LEFT JOIN user_input_signals s ON s.message_id=m.id
                   WHERE m.conversation_id=$1 AND m.role='user' AND m.created_at >= $2::timestamp
                   ORDER BY m.created_at""",
                conversation, since)
            ordered = [(r["id"], _utc(r["created_at"])) for r in rows]
            timed = compose_times(ordered, times)
            for r in rows:
                if r["id"] in timed and not r["has_signal"] and r["chars"]:
                    inputs.append((r["id"], user, _utc(r["created_at"]), r["chars"], timed[r["id"]]))
        summary = {"lines_users": len({u for u, _ in merged}), "networks": len(merged),
                   "inputs": len(inputs), "dry_run": dry_run}
        if dry_run:
            return summary
        for (user, ip), (first, last, hits, agent) in merged.items():
            await signals.record_network(user, ip, agent, source="nginx_backfill", seen_at=last,
                                         first_seen_at=first, hits=hits, merge=False)
        for message_id, user, at, chars, ms in inputs:
            await db.execute_raw(
                """INSERT INTO user_input_signals (message_id, user_id, created_at, chars,
                     server_compose_ms, source)
                   SELECT $1, id, $3::timestamp, $4, $5, 'nginx_backfill'
                   FROM users WHERE id=$2 AND status<>'deleting'
                   ON CONFLICT (message_id) DO NOTHING""",
                message_id, user, at.replace(tzinfo=None).isoformat(), chars, ms)
        return summary
    finally:
        if owned:
            await db.disconnect()


def _utc(value) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(main(sys.stdin, args.dry_run)), ensure_ascii=False))
