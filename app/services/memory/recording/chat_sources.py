"""Resolve chat extraction inputs from persisted identities, never matching text.

This is a bounded source snapshot, not a durable job or permission to replay.
The extraction/write path still validates the source versions before storing.
"""
from app.db import db
from app.services.memory.recording.outcomes import MemoryProgressError

MAX_SOURCES = 300  # 100 user fragments + 100 reply bubbles + 100 context rows


def _valid_id(value):
    return isinstance(value, str) and 0 < len(value) <= 200


def excluded_from_chat_extraction(message: dict) -> bool:
    metadata = message.get("metadata") or {}
    if metadata.get("human_takeover") or metadata.get("memory_confirmation"):
        return True
    if metadata.get("real_world_type") == "gift":
        return True
    if message.get("role") == "assistant":
        # Offline-activity copy is business text about the user's outing in the
        # user's city ("头疼好些了，去清凉山走走"). Self-memory extraction turned it
        # into first-person persona facts, so it never becomes an AI memory.
        if str(metadata.get("trigger_type") or "").startswith("offline_activity_"):
            return True
        # The explicit marker is written with each reply. Older replies already
        # carry the same server-owned intent in their response diagnostics.
        kind = metadata.get("memory_extraction_exclusion")
        diagnostics = metadata.get("response_diagnostics") or {}
        return kind in {"schedule_query", "current_state"} or (
            isinstance(diagnostics, dict)
            and diagnostics.get("short_circuit_kind") in {"schedule_query", "current_state"}
        )
    return False


def has_music_card(message: dict) -> bool:
    """Only persisted user music cards provide this structured source."""
    card = (message.get("metadata") or {}).get("component_card")
    return message.get("role") == "user" and isinstance(card, dict) and card.get("type") == "music_track"


def extraction_text(message: dict) -> str:
    """Render recorded card data without changing the user-visible message."""
    content = message.get("content") or ""
    if not has_music_card(message):
        return content
    card = message["metadata"]["component_card"]
    payload = card.get("payload")
    track = payload.get("track") if isinstance(payload, dict) else None
    if not isinstance(track, dict) or not isinstance(track.get("title"), str) or not track["title"].strip():
        raise MemoryProgressError("source_invalid")
    title = track["title"].strip()[:300]
    artist = track.get("artist")
    artist = artist.strip()[:300] if isinstance(artist, str) else ""
    line = f"用户在这条消息中分享了歌曲《{title}》" + (f"，歌手：{artist}" if artist else "")
    return content + "\n" + line if content.strip() else line


async def load_chat_sources(*, conversation_id: str, user_id: str, workspace_id: str,
                            messages: list[dict], source_message_ids: list[str] | None = None,
                            reply_turn_id: str | None = None) -> list[dict]:
    """Fetch the exact current turn plus a small historical context window.

    IDs and the parent turn identity come from server persistence. Missing IDs
    never fall back to a latest row, text similarity, or a synthetic timestamp.
    """
    explicit = source_message_ids or []
    if (not isinstance(explicit, list) or len(explicit) > 100
            or any(not _valid_id(i) for i in explicit)
            or (reply_turn_id is not None and not _valid_id(reply_turn_id))):
        raise MemoryProgressError("source_identity_invalid")
    from app.services.memory.recording.context_window import history_rows
    recent = history_rows(messages)
    # Expanding disambiguation must not admit older rows as fresh facts on a
    # first run / expired watermark. Preserve the original six-row eligibility.
    eligible_ids = {m.get("id") for m in messages[-6:]} | set(explicit)
    if not explicit and not reply_turn_id and any(
        m.get("role") in {"user", "assistant"} and not _valid_id(m.get("id")) for m in recent
    ):
        raise MemoryProgressError("source_identity_missing")
    ids = list(dict.fromkeys([m["id"] for m in recent if _valid_id(m.get("id"))] + explicit))
    if not ids:
        raise MemoryProgressError("source_identity_missing")
    try:
        rows = await db.query_raw("""SELECT m.id,m.role,m.content,m.metadata,m.created_at AS "createdAt"
            FROM messages m JOIN conversations c ON c.id=m.conversation_id
            JOIN chat_workspaces w ON w.id=c.workspace_id AND w.user_id=c.user_id AND w.agent_id=c.agent_id
            JOIN ai_agents a ON a.id=c.agent_id AND a.user_id=c.user_id
            WHERE c.id=$1 AND c.user_id=$2 AND c.workspace_id=$3 AND NOT c.is_deleted
              AND w.status='active' AND m.role IN ('user','assistant')
              AND (m.id=ANY($4::text[]) OR ($5::text IS NOT NULL AND m.role='assistant'
                   AND m.metadata->>'achievement_turn_id'=$5))
            ORDER BY m.created_at,m.id LIMIT 301""",
            conversation_id, user_id, workspace_id, ids, reply_turn_id)
    except Exception as exc:
        raise MemoryProgressError("source_read_unavailable") from exc
    if (len(rows) > MAX_SOURCES or not set(ids).issubset({r['id'] for r in rows})
            or (reply_turn_id is not None and not any(
                r['role'] == 'assistant' and (r.get('metadata') or {}).get('achievement_turn_id') == reply_turn_id
                for r in rows))):
        raise MemoryProgressError("source_invalid")
    for row in rows:
        is_reply = (reply_turn_id is not None and row['role'] == 'assistant'
                    and (row.get('metadata') or {}).get('achievement_turn_id') == reply_turn_id)
        row['_memory_context_only'] = row['id'] not in eligible_ids and not is_reply
    if any(sum(r['role'] == role and not r['_memory_context_only'] for r in rows) > 100
           for role in ('user', 'assistant')):
        raise MemoryProgressError("source_batch_limit")
    return rows
