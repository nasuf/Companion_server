-- Part 5 §3.1: 为记忆表增加 statement_time 字段.
--
-- 双时间维度按 spec §3.1 定义:
--   statement_time = 用户说出这句话的时间 (消息接收时刻)
--   event_time     = 用户描述的事件实际发生时间 (时间解析器输出)
--
-- spec §6.1 协同规则: 时间解析器输出的 event_time 写入 memories.occur_time 字段.
-- 因此本迁移**只**新增 statement_time, 不新增 event_time —— event_time 在 Python
-- 层 (TimeExtract dataclass) 表达, 落库时映射到既有的 occur_time.

ALTER TABLE "memories_user"
    ADD COLUMN IF NOT EXISTS "statement_time" TIMESTAMP(3);

ALTER TABLE "memories_ai"
    ADD COLUMN IF NOT EXISTS "statement_time" TIMESTAMP(3);

-- occur_time 已有索引 (历史 migration 创建), 此处不重复.
