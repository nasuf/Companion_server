-- Additive event idempotency; old applications can run against this schema.
CREATE TABLE "memory_usage_receipts" (
  "event_id" TEXT NOT NULL REFERENCES "messages"("id") ON DELETE CASCADE,
  "memory_side" TEXT NOT NULL CHECK ("memory_side" IN ('user','ai')),
  "memory_id" TEXT NOT NULL,
  "contributed" BOOLEAN NOT NULL,
  "reward" DOUBLE PRECISION NOT NULL CHECK ("reward" >= 0 AND "reward" <= 0.12),
  "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY ("event_id","memory_side","memory_id")
);
CREATE INDEX "memory_usage_receipts_retention_idx" ON "memory_usage_receipts"("created_at");

-- Newer provisioning paths had omitted the clock. Preserve current scores and
-- levels; do not retroactively apply months of decay on first unified sweep.
UPDATE "memories_user" SET "value_updated_at"=CURRENT_TIMESTAMP WHERE "value_updated_at" IS NULL;
UPDATE "memories_ai" SET "value_updated_at"=CURRENT_TIMESTAMP WHERE "value_updated_at" IS NULL;
CREATE INDEX "memories_user_decay_due_idx" ON "memories_user"
  (COALESCE("value_updated_at","created_at"),"id")
  WHERE NOT "is_archived" AND "sub_category" IS DISTINCT FROM '提醒';
CREATE INDEX "memories_ai_decay_due_idx" ON "memories_ai"
  (COALESCE("value_updated_at","created_at"),"id")
  WHERE NOT "is_archived" AND "sub_category" IS DISTINCT FROM '提醒';
