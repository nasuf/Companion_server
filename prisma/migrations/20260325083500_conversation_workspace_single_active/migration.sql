-- Ensure each workspace has at most one active conversation.
-- Prisma tracks execution in _prisma_migrations; this SQL runs only once per database.

WITH ranked_conversations AS (
    SELECT
        "id",
        ROW_NUMBER() OVER (
            PARTITION BY "workspace_id"
            ORDER BY "updated_at" DESC, "created_at" DESC, "id" DESC
        ) AS rn
    FROM "conversations"
    WHERE "workspace_id" IS NOT NULL
      AND "is_deleted" = FALSE
)
UPDATE "conversations" c
SET
    "is_deleted" = TRUE,
    "archived_at" = COALESCE(c."archived_at", CURRENT_TIMESTAMP)
FROM ranked_conversations r
WHERE c."id" = r."id"
  AND r.rn > 1;

CREATE UNIQUE INDEX IF NOT EXISTS "conversations_workspace_id_active_key"
    ON "conversations"("workspace_id")
    WHERE "workspace_id" IS NOT NULL
      AND "is_deleted" = FALSE;
