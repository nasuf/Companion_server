-- L1 晋升闸门按记忆查它在多少个自然日被注入过 (lifecycle/lazy_update.py)。
-- 主键以 event_id 打头, 按 memory_id 查只能全表扫。表有 30 天保留期, 体量随
-- 对话量线性增长, 上线时只有几千行, 建索引不需要 CONCURRENTLY。
CREATE INDEX IF NOT EXISTS "memory_usage_receipts_contributed_memory_idx"
  ON "memory_usage_receipts" ("memory_side", "memory_id", "created_at")
  WHERE "contributed";
