ALTER TABLE "time_capsules"
    ADD COLUMN IF NOT EXISTS "media" JSONB;

ALTER TABLE "time_capsules"
    ADD COLUMN IF NOT EXISTS "skin" TEXT NOT NULL DEFAULT 'paper';
