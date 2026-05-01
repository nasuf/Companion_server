-- Part 5 §4.2 提醒记忆 recurrence 字段 (once=默认 / yearly / monthly / weekly / daily)
-- 向后兼容: 旧行 recurrence=NULL, 代码读到 None 走 once 语义.
ALTER TABLE "memories_user" ADD COLUMN "recurrence" TEXT;
ALTER TABLE "memories_ai" ADD COLUMN "recurrence" TEXT;

-- 部分索引: 仅覆盖未归档的提醒条目 (大头闲聊/职业等不进索引, B-Tree 体积小).
-- special_dates 扫描热路径按 (user_id, sub_category="提醒") 过滤后再用 recurrence 字段匹配.
CREATE INDEX "memories_user_reminder_idx"
  ON "memories_user" ("user_id", "sub_category", "recurrence")
  WHERE "sub_category" = '提醒' AND "is_archived" = FALSE;
CREATE INDEX "memories_ai_reminder_idx"
  ON "memories_ai" ("user_id", "sub_category", "recurrence")
  WHERE "sub_category" = '提醒' AND "is_archived" = FALSE;
