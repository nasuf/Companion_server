BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
-- Additive rollout: existing clients ignore these fields. Default pauses emoji-only replies.
ALTER TABLE "system_config" ADD COLUMN "filler_emoji_probability" DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE "system_config" ADD CONSTRAINT "system_config_filler_emoji_probability_check"
    CHECK ("filler_emoji_probability" >= 0 AND "filler_emoji_probability" <= 1);

CREATE TABLE "runtime_config_changes" (
    "id" TEXT PRIMARY KEY,
    "admin_id" TEXT,
    "key" TEXT NOT NULL,
    "before" JSONB NOT NULL,
    "after" JSONB NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX "runtime_config_changes_key_created_at_idx"
    ON "runtime_config_changes" ("key", "created_at" DESC);

COMMIT;
