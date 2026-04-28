-- 每条 chat 请求结束时写一行的 token 用量明细 (后台 统计概览 用).
CREATE TABLE "llm_usage" (
    "id" TEXT NOT NULL,
    "conversation_id" TEXT NOT NULL,
    "agent_id" TEXT,
    "user_id" TEXT,
    "trace_id" TEXT,
    "input_tokens" INTEGER NOT NULL,
    "output_tokens" INTEGER NOT NULL,
    "tokens_by_model" JSONB NOT NULL,
    "cost_cny" DOUBLE PRECISION NOT NULL,
    "call_count" INTEGER NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "llm_usage_pkey" PRIMARY KEY ("id")
);

CREATE INDEX "llm_usage_agent_id_created_at_idx" ON "llm_usage"("agent_id", "created_at" DESC);
CREATE INDEX "llm_usage_created_at_idx" ON "llm_usage"("created_at" DESC);
