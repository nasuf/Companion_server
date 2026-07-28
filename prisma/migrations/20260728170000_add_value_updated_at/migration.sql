-- 惰性衰减需要知道"距上次算值多久"。不能复用 updated_at: 那个字段被访问打点、
-- admin 编辑、矛盾处理等多处刷新, 每刷一次 Δt 归零, 记忆就永远衰减不下去。
ALTER TABLE "memories_user" ADD COLUMN IF NOT EXISTS "value_updated_at" TIMESTAMP(3);
ALTER TABLE "memories_ai"   ADD COLUMN IF NOT EXISTS "value_updated_at" TIMESTAMP(3);

-- 兜底扫描要找"最久没算过的", 这个索引让它不必全表排序。
-- NULL 排在最前 (从未算过的优先), 与 lifecycle 里 NULLS FIRST 的取值一致。
CREATE INDEX IF NOT EXISTS "memories_user_value_updated_idx"
  ON "memories_user" ("value_updated_at") WHERE "is_archived" = false;
CREATE INDEX IF NOT EXISTS "memories_ai_value_updated_idx"
  ON "memories_ai" ("value_updated_at") WHERE "is_archived" = false;
