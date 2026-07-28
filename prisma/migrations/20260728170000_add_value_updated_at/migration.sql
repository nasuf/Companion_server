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

-- 存量行的衰减时钟从**上线时刻**起算, 而不是从创建时刻补算。
--
-- 不回填的话, COALESCE(value_updated_at, created_at) 会取到两年前的 created_at,
-- Δt=730 天, 首次扫描一跑就把近 5900 条建号人设一次性打出 L1 —— 那是个没有观察
-- 窗口的大规模重排, 而分层改造的前提是每一步都能单独验证、单独回滚。
--
-- 语义上这是"既往不咎": 存量记忆得到一次重新开始的机会, 之后按真实使用情况该降
-- 就降。是否要主动重排存量人设是**另一个决定**, 需要单独的数据支撑, 不该由一次
-- 加列迁移顺带做掉。
UPDATE "memories_user" SET "value_updated_at" = CURRENT_TIMESTAMP
  WHERE "value_updated_at" IS NULL;
UPDATE "memories_ai" SET "value_updated_at" = CURRENT_TIMESTAMP
  WHERE "value_updated_at" IS NULL;
