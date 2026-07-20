-- Phase 2: memory provenance + changelog operational indexes.
--
-- provenance: where a memory came from (profile_seed / user_stated /
-- ai_authored / daily_summary / consolidated). Turns the "guess origin from
-- category" defenses into lookups: seed rows are write-protected, AI-authored
-- rows never become persona facts, daily-summary rows are consolidation food.
-- NULL = legacy rows (a one-shot backfill script exists in scripts/).
ALTER TABLE "memories_user" ADD COLUMN "provenance" TEXT;
ALTER TABLE "memories_ai" ADD COLUMN "provenance" TEXT;

-- L2 dynamics aggregates access counts (1y window) + all-time MAX(created_at)
-- per memory; retention purges access rows by age. Both were sequential scans.
CREATE INDEX "memory_changelogs_memory_op_created_idx"
    ON "memory_changelogs"("memory_id", "operation", "created_at");
CREATE INDEX "memory_changelogs_op_created_idx"
    ON "memory_changelogs"("operation", "created_at");
