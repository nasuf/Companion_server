-- DB-backed snapshot for Redis-primary proactive daily counters.
-- Redis 挂时 can_send_proactive / can_send_proactive_2day 降级读取此表, 防止计数丢失。
-- 2day 滑动窗口计数通过 sum(today+yesterday) 从此表推导, 不独立持久化。

CREATE TABLE IF NOT EXISTS "proactive_counters" (
    "id" TEXT NOT NULL,
    "agent_id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "date" TEXT NOT NULL,
    "count" INTEGER NOT NULL DEFAULT 0,
    "updated_at" TIMESTAMP(3) NOT NULL,
    CONSTRAINT "proactive_counters_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX IF NOT EXISTS "proactive_counters_agent_id_user_id_date_key"
    ON "proactive_counters"("agent_id", "user_id", "date");

CREATE INDEX IF NOT EXISTS "proactive_counters_agent_id_user_id_idx"
    ON "proactive_counters"("agent_id", "user_id");
