-- memory_consolidation_runs 此前只有 hygiene 在写, L3 簇压缩完全没有审计记录。
-- 两者都叫"整合"但做的事不同 (前者合并近重复, 后者压缩同题簇并归档原行), 混在
-- 一张表里会让"这条记忆去哪了"无从查起。
ALTER TABLE "memory_consolidation_runs"
  ADD COLUMN IF NOT EXISTS "job" TEXT NOT NULL DEFAULT 'hygiene';

CREATE INDEX IF NOT EXISTS "memory_consolidation_runs_job_idx"
  ON "memory_consolidation_runs" ("job", "created_at" DESC);
