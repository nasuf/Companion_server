BEGIN;

ALTER TABLE "offline_activity_recommendations"
    ADD COLUMN "focus_condition_id" TEXT,
    ADD COLUMN "next_companion_at" TIMESTAMPTZ(3),
    ADD COLUMN "last_companion_at" TIMESTAMPTZ(3),
    ADD COLUMN "companion_claim_token" TEXT,
    ADD COLUMN "companion_claimed_at" TIMESTAMPTZ(3),
    ADD COLUMN "companion_state" JSONB NOT NULL DEFAULT '{}';

ALTER TABLE "offline_shooting_conditions"
    ADD COLUMN "guidance_profile" JSONB NOT NULL DEFAULT '{}';

DELETE FROM "offline_prewritten_fragments"
WHERE "id" IN (
    SELECT "id"
    FROM (
        SELECT "id",
               ROW_NUMBER() OVER (
                   PARTITION BY "condition_id", "tier"
                   ORDER BY "created_at", "id"
               ) AS duplicate_rank
        FROM "offline_prewritten_fragments"
    ) ranked
    WHERE duplicate_rank > 1
);

DROP INDEX IF EXISTS "offline_prewritten_fragment_condition_tier_idx";
CREATE UNIQUE INDEX "offline_prewritten_fragment_condition_tier_key"
    ON "offline_prewritten_fragments" ("condition_id", "tier");

CREATE INDEX "offline_activity_companion_due_idx"
    ON "offline_activity_recommendations" ("status", "reached", "next_companion_at");

WITH ranked_active AS (
    SELECT "id",
           ROW_NUMBER() OVER (
               PARTITION BY COALESCE("workspace_id", "user_id")
               ORDER BY "arrival_confirmed_at" DESC NULLS LAST, "created_at" DESC
           ) AS current_rank
    FROM "offline_activity_recommendations"
    WHERE "status" = 'accepted' AND "reached" = TRUE
)
UPDATE "offline_activity_recommendations" AS activity
SET "status" = 'completed',
    "completed_at" = COALESCE("completed_at", CURRENT_TIMESTAMP),
    "archived_at" = COALESCE("archived_at", CURRENT_TIMESTAMP),
    "auto_archived" = TRUE,
    "auto_archive_at" = NULL,
    "companion_state" = '{"mode":"stopped","migration_reason":"duplicate_reached"}'::jsonb
WHERE activity."id" IN (
    SELECT "id" FROM ranked_active WHERE current_rank > 1
);

CREATE UNIQUE INDEX "offline_activity_one_reached_workspace_key"
    ON "offline_activity_recommendations" ("workspace_id")
    WHERE "workspace_id" IS NOT NULL
      AND "status" = 'accepted'
      AND "reached" = TRUE;

CREATE UNIQUE INDEX "offline_activity_one_reached_legacy_user_key"
    ON "offline_activity_recommendations" ("user_id")
    WHERE "workspace_id" IS NULL
      AND "status" = 'accepted'
      AND "reached" = TRUE;

WITH ranked_activities AS (
    SELECT "id",
           ROW_NUMBER() OVER (
               PARTITION BY COALESCE("workspace_id", "user_id")
               ORDER BY "arrival_confirmed_at" DESC NULLS LAST, "created_at" DESC
           ) AS current_rank
    FROM "offline_activity_recommendations"
    WHERE "status" = 'accepted' AND "reached" = TRUE
)
UPDATE "offline_activity_recommendations" AS activity
SET "focus_condition_id" = (
        SELECT condition."id"
        FROM "offline_shooting_conditions" AS condition
        WHERE condition."recommendation_id" = activity."id"
          AND condition."triggered" = FALSE
        ORDER BY condition."sort_order"
        LIMIT 1
    ),
    "next_companion_at" = CURRENT_TIMESTAMP
        + ((10 + FLOOR(RANDOM() * 11)) * INTERVAL '1 minute'),
    "companion_state" = '{"mode":"guided","unanswered_count":0,"recent_modes":[]}'::jsonb
WHERE activity."status" = 'accepted'
  AND activity."reached" = TRUE
  AND activity."conditions_ready_at" IS NOT NULL
  AND activity."id" IN (
      SELECT "id" FROM ranked_activities WHERE current_rank = 1
  )
  AND EXISTS (
      SELECT 1
      FROM "offline_shooting_conditions" AS condition
      WHERE condition."recommendation_id" = activity."id"
        AND condition."triggered" = FALSE
  );

COMMIT;
