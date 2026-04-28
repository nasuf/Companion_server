-- 增加 scope 字段区分非会话场景 (agent_creation / schedule_cron),
-- 同时把 conversation_id 改为可空 (这两类场景没有 conversation 归属).

ALTER TABLE "llm_usage" ALTER COLUMN "conversation_id" DROP NOT NULL;
ALTER TABLE "llm_usage" ADD COLUMN "scope" TEXT NOT NULL DEFAULT 'chat';
