-- Admin 数据监控聚合的支撑索引.
--
-- 监控面板的分时段/每日活跃/句子区间/DAU-WAU-MAU 查询都以 role='user' +
-- created_at 范围为条件对 messages 全表扫 (原表只有 (conversation_id, created_at)
-- 索引, 帮不上). 新增注册走势按 users.created_at 分桶, users 也无 created_at 索引.
CREATE INDEX "messages_role_created_idx" ON "messages"("role", "created_at");
CREATE INDEX "users_created_idx" ON "users"("created_at");
