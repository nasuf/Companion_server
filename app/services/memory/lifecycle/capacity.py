"""Bound optional derived memories and reclaim rebuildable cold embeddings.

Factual chat/seed memories and original compressed text are never deleted by
this policy. Deleted vectors are regenerated before an archive is restored.
"""
from app.db import db

DAILY_SUMMARY_ROW_BUDGET = 2000  # includes archived originals; a lifetime scope budget


async def reclaim_consolidated_embeddings(*, limit: int = 500) -> dict:
    if not 1 <= limit <= 2000:
        raise ValueError("Invalid cold-vector cleanup batch")
    stats = {"user": 0, "ai": 0}
    for side in ("user", "ai"):
        table = "memories_ai" if side == "ai" else "memories_user"
        # Row locks are shared with restore's transaction. Reclamation cannot
        # race between its vector rebuild and unarchive commit.
        stats[side] = await db.execute_raw(
            f"""WITH cold AS (
                SELECT m.id FROM {table} m WHERE m.is_archived AND m.level=3
                  AND m.provenance IS DISTINCT FROM 'profile_seed'
                  AND m.provenance IS DISTINCT FROM 'knowledge_seed'
                  AND EXISTS (SELECT 1 FROM memory_embeddings e WHERE e.memory_id=m.id)
                  AND EXISTS (
                    SELECT 1 FROM memory_changelogs cl JOIN {table} d ON d.id=cl.new_value
                    WHERE cl.memory_id=m.id AND cl.operation='consolidated_into'
                      AND cl.user_id=m.user_id AND cl.workspace_id IS NOT DISTINCT FROM m.workspace_id
                      AND d.user_id=m.user_id AND d.workspace_id IS NOT DISTINCT FROM m.workspace_id
                      AND NOT d.is_archived AND d.provenance='consolidated' AND d.level=3
                  ) ORDER BY m.created_at,m.id LIMIT {limit} FOR UPDATE OF m SKIP LOCKED
              ) DELETE FROM memory_embeddings WHERE memory_id IN (SELECT id FROM cold)""",
        )
        # A crash may leave an unpublished staged summary. It has no original
        # rows pointing to it. Reclaim this failed derived output, never source
        # memories or a published/restored digest with an audit trail.
        from datetime import timedelta
        async with db.tx(timeout=timedelta(seconds=15)) as tx:
            abandoned = await tx.query_raw(
                f"""DELETE FROM {table} WHERE id IN (
                    SELECT m.id FROM {table} m WHERE m.is_archived AND m.level=3
                      AND m.provenance='consolidated' AND m.created_at < NOW()-INTERVAL '1 day'
                      AND NOT EXISTS (SELECT 1 FROM memory_changelogs cl
                        WHERE cl.operation='consolidated_into' AND cl.new_value=m.id)
                    ORDER BY m.created_at,m.id LIMIT {limit} FOR UPDATE OF m SKIP LOCKED
                  ) RETURNING id""",
            )
            if abandoned:
                await tx.execute_raw("DELETE FROM memory_embeddings WHERE memory_id=ANY($1::text[])",
                                     [r["id"] for r in abandoned])
        stats[side + "_abandoned_digests"] = len(abandoned)
    await db.execute_raw(
        """DELETE FROM memory_daily_reviews WHERE id IN (
             SELECT id FROM memory_daily_reviews WHERE created_at < NOW()-INTERVAL '30 days'
             ORDER BY created_at LIMIT 1000)""",
    )
    return stats


async def restore_consolidated_digest(digest_id: str) -> dict:
    """Rebuild all vectors first; restore text, digest state and audit atomically.

    A provider failure leaves all originals archived. Concurrent cleanup skips
    rows locked by the final restore transaction.
    """
    import uuid
    from datetime import timedelta
    from app.services.memory.storage.embedding import generate_embedding

    for source, table in (("user", "memories_user"), ("ai", "memories_ai")):
        digests = await db.query_raw(
            f"SELECT id,user_id,workspace_id FROM {table} WHERE id=$1 AND provenance='consolidated' AND NOT is_archived",
            digest_id,
        )
        if not digests:
            continue
        digest = digests[0]
        originals = await db.query_raw(
            f"""SELECT m.id,m.content FROM {table} m WHERE m.is_archived AND m.level=3
                AND m.user_id=$2 AND m.workspace_id IS NOT DISTINCT FROM $3::text
                AND EXISTS (SELECT 1 FROM memory_changelogs cl WHERE cl.memory_id=m.id
                  AND cl.operation='consolidated_into' AND cl.new_value=$1
                  AND cl.user_id=m.user_id AND cl.workspace_id IS NOT DISTINCT FROM m.workspace_id)
                ORDER BY m.id LIMIT 301""",
            digest_id, digest["user_id"], digest["workspace_id"],
        )
        if not originals:
            return {"found": 0, "restored": 0}
        if len(originals) > 300:
            raise ValueError("Archive exceeds consolidation cluster bound")
        vectors = [await generate_embedding(row["content"]) for row in originals]
        async with db.tx(timeout=timedelta(seconds=15)) as tx:
            live = await tx.query_raw(f"SELECT id FROM {table} WHERE id=$1 AND NOT is_archived FOR UPDATE", digest_id)
            locked = await tx.query_raw(
                f"SELECT id,content FROM {table} WHERE id=ANY($1::text[]) AND is_archived ORDER BY id FOR UPDATE",
                [row["id"] for row in originals],
            )
            if not live or locked != originals:
                raise RuntimeError("Archive changed during vector generation")
            for row, vector in zip(originals, vectors):
                await tx.execute_raw(
                    "INSERT INTO memory_embeddings(memory_id,embedding) VALUES($1,$2::extensions.vector) "
                    "ON CONFLICT(memory_id) DO UPDATE SET embedding=EXCLUDED.embedding",
                    row["id"], "[" + ",".join(str(float(v)) for v in vector) + "]",
                )
            ids = [r["id"] for r in originals]
            await tx.execute_raw(f"UPDATE {table} SET is_archived=false WHERE id=ANY($1::text[])", ids)
            await tx.execute_raw(f"UPDATE {table} SET is_archived=true WHERE id=$1", digest_id)
            await tx.execute_raw(
                """INSERT INTO memory_changelogs(id,user_id,workspace_id,memory_id,operation,new_value)
                     SELECT unnest($1::text[]),$2,$3,unnest($4::text[]),'consolidation_undone',$5""",
                [str(uuid.uuid4()) for _ in ids], digest["user_id"], digest["workspace_id"], ids, digest_id,
            )
        from app.services.memory.storage.repo import invalidate_scope
        await invalidate_scope(digest["user_id"], digest["workspace_id"])
        return {"found": len(originals), "restored": len(originals), "source": source}
    return {"found": 0, "restored": 0}
