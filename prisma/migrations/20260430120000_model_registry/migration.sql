-- 模型库: admin 维护可选模型 + 单价. system_config/agent_config_overrides
-- 的 chat/small 字段从这里选 identifier; pricing 从这里取价格.
CREATE TABLE "model_registry" (
    "id" TEXT NOT NULL,
    "identifier" TEXT NOT NULL,
    "display_name" TEXT,
    "provider" TEXT NOT NULL,
    "enabled" BOOLEAN NOT NULL DEFAULT true,
    "context_window" INTEGER,
    "input_cost_per_million" DOUBLE PRECISION,
    "output_cost_per_million" DOUBLE PRECISION,
    "notes" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "model_registry_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX "model_registry_identifier_key" ON "model_registry"("identifier");
CREATE INDEX "model_registry_provider_enabled_idx" ON "model_registry"("provider", "enabled");

-- Seed: 当前在用的 4 个模型. 远程价格来自 app/services/llm/pricing.py
-- (元/1M tokens, 0-128K tier). 本地价格 = 0 (自托管).
INSERT INTO "model_registry" (
    "id", "identifier", "display_name", "provider", "enabled",
    "context_window", "input_cost_per_million", "output_cost_per_million", "updated_at"
) VALUES
    (gen_random_uuid(), 'qwen2.5:14b',   'Qwen 2.5 14B (本地)',  'ollama',    true, 32768,  0,   0,   CURRENT_TIMESTAMP),
    (gen_random_uuid(), 'qwen2.5:7b',    'Qwen 2.5 7B (本地)',   'ollama',    true, 32768,  0,   0,   CURRENT_TIMESTAMP),
    (gen_random_uuid(), 'qwen3.5-plus',  'Qwen 3.5 Plus (远程)', 'dashscope', true, 131072, 0.8, 4.8, CURRENT_TIMESTAMP),
    (gen_random_uuid(), 'qwen3.5-flash', 'Qwen 3.5 Flash (远程)','dashscope', true, 131072, 0.2, 2.0, CURRENT_TIMESTAMP);
