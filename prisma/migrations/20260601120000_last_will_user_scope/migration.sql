ALTER TABLE "last_wills"
    DROP CONSTRAINT IF EXISTS "last_wills_agent_id_fkey";

ALTER TABLE "last_wills"
    ALTER COLUMN "agent_id" DROP NOT NULL;

WITH ranked AS (
    SELECT
        id,
        ROW_NUMBER() OVER (
            PARTITION BY user_id
            ORDER BY
                CASE
                    WHEN status IN ('active', 'triggered', 'paused') AND btrim(content) <> '' THEN 0
                    WHEN status = 'draft' AND btrim(content) <> '' THEN 1
                    WHEN btrim(content) <> '' THEN 2
                    ELSE 3
                END,
                updated_at DESC,
                created_at DESC,
                id DESC
        ) AS rn
    FROM "last_wills"
)
DELETE FROM "last_wills" lw
USING ranked
WHERE lw.id = ranked.id
  AND ranked.rn > 1;

DROP INDEX IF EXISTS "last_wills_user_agent_unique";
DROP INDEX IF EXISTS "last_wills_user_agent_status_idx";

CREATE UNIQUE INDEX IF NOT EXISTS "last_wills_user_unique"
    ON "last_wills"("user_id");

CREATE INDEX IF NOT EXISTS "last_wills_user_status_idx"
    ON "last_wills"("user_id", "status");

ALTER TABLE "last_wills"
    ADD CONSTRAINT "last_wills_agent_id_fkey"
    FOREIGN KEY ("agent_id") REFERENCES "ai_agents"("id") ON DELETE SET NULL ON UPDATE CASCADE;
