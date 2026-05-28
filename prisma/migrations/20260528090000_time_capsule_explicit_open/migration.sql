ALTER TABLE "time_capsules"
    ADD COLUMN "opened_at" TIMESTAMP(3);

UPDATE "time_capsules"
SET "opened_at" = COALESCE("sealed_at", "updated_at", CURRENT_TIMESTAMP)
WHERE "status" = 'sealed'
  AND "open_date"::date <= (CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Shanghai')::date;

CREATE INDEX "time_capsules_user_agent_opened_idx"
    ON "time_capsules"("user_id", "agent_id", "opened_at");
