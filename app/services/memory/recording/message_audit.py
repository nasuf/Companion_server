"""Rotating, bounded discovery of messages missed by the fast Redis cursor.

Audit position is not a completion claim. An unclaimed page is held until SQL
source receipts exist. End-of-window wraps to the admission floor, so a late
commit behind either cursor is revisited on subsequent chat triggers.
"""

SCAN_PAGE_SIZE = 200


async def unrecorded_since_admission(conversation_id, side, *, database, limit=31):
    from app.services.memory.recording.receipts import MemoryReceiptError

    if side not in {"user", "ai"} or type(limit) is not int or not 1 <= limit <= 31:
        raise MemoryReceiptError("memory_plan_invalid")
    async with database.tx() as tx:
        await tx.execute_raw("SET LOCAL statement_timeout = '3s'")
        first = await tx.query_raw("""SELECT id,
            (SELECT min((s->>'expected_statement_time')::timestamptz)
             FROM jsonb_array_elements(sources) s) AS floor_at
            FROM memory_extraction_batches
            WHERE conversation_id=$1 AND side=$2 AND pipeline_version IS NOT NULL
            ORDER BY prepared_at,id LIMIT 1""", conversation_id, side)
        if not first or first[0]["floor_at"] is None:
            return []  # No admission is not permission to backfill old history.
        batch = first[0]
        await tx.execute_raw("""INSERT INTO memory_message_audit_cursors(batch_id)
            VALUES($1) ON CONFLICT DO NOTHING""", batch["id"])
        cursors = await tx.query_raw("""SELECT after_at,after_id
            FROM memory_message_audit_cursors WHERE batch_id=$1 FOR UPDATE""", batch["id"])
        cursor = cursors[0]
        # LIMIT is applied before the source anti-join, across ALL roles. A
        # fully processed conversation therefore scans at most 200 rows too.
        page = await tx.query_raw("""WITH page AS MATERIALIZED (
            SELECT id,(created_at AT TIME ZONE 'UTC') AS created_at,role FROM messages
            WHERE conversation_id=$1 AND created_at >= ($2::timestamptz AT TIME ZONE 'UTC')
              AND ($3::timestamptz IS NULL OR (created_at,id)>(($3::timestamptz AT TIME ZONE 'UTC'),$4::text))
            ORDER BY messages.created_at,messages.id LIMIT $5
        ) SELECT p.*, (s.message_id IS NULL OR b.status IN ('preparing','processing'))
            AS needs_processing FROM page p
          LEFT JOIN memory_extraction_sources s ON s.message_id=p.id AND s.side=$6
          LEFT JOIN memory_extraction_batches b ON b.id=s.batch_id
          ORDER BY p.created_at,p.id""", conversation_id, batch["floor_at"],
            cursor["after_at"], cursor["after_id"], SCAN_PAGE_SIZE, side)
        role = "user" if side == "user" else "assistant"
        missing = [r["id"] for r in page if r["role"] == role and r["needs_processing"]]
        if missing:
            # Do not advance past failed/unclaimed messages, even if more than
            # one worker discovers this page. Admission serializes the claims.
            return await tx.query_raw('''SELECT id,role,content,metadata,
                created_at AS "createdAt" FROM messages
                WHERE conversation_id=$1 AND id=ANY($2::text[]) AND role=$3
                ORDER BY created_at,id''', conversation_id, missing[:limit], role)
        last = page[-1] if len(page) == SCAN_PAGE_SIZE else None
        await tx.execute_raw("""UPDATE memory_message_audit_cursors
            SET after_at=$2::timestamptz,after_id=$3 WHERE batch_id=$1""", batch["id"],
            last["created_at"] if last else None, last["id"] if last else None)
        return []
