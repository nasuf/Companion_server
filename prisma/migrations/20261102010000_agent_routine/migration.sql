-- TIME12: agent fixed routine and per-day routine conformance.
-- Both columns are nullable and additive; NULL routine means "not derived yet".
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
ALTER TABLE "ai_agents" ADD COLUMN IF NOT EXISTS "routine" JSONB;
ALTER TABLE "ai_daily_schedules" ADD COLUMN IF NOT EXISTS "generation_meta" JSONB;
COMMIT;
