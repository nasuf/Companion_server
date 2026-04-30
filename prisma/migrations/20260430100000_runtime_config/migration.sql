-- 全局运行时配置 (单行, id=1)
CREATE TABLE "system_config" (
    "id" INTEGER NOT NULL DEFAULT 1,
    "online_model" BOOLEAN,
    "local_chat_model" TEXT,
    "local_small_model" TEXT,
    "remote_chat_model" TEXT,
    "remote_small_model" TEXT,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "system_config_pkey" PRIMARY KEY ("id")
);

-- 单 agent 模型 override
CREATE TABLE "agent_config_overrides" (
    "agent_id" TEXT NOT NULL,
    "online_model" BOOLEAN,
    "local_chat_model" TEXT,
    "local_small_model" TEXT,
    "remote_chat_model" TEXT,
    "remote_small_model" TEXT,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "agent_config_overrides_pkey" PRIMARY KEY ("agent_id")
);

ALTER TABLE "agent_config_overrides" ADD CONSTRAINT "agent_config_overrides_agent_id_fkey"
    FOREIGN KEY ("agent_id") REFERENCES "ai_agents"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- Seed: system_config 单行默认值 (跟 app/config.py defaults 对齐).
-- 设计: DB 是模型配置真源, env 仅在 DB load 失败时兜底. 任何 admin
-- 修改写 DB, 重启不丢. 这一行保证 fresh deploy 跟现有 deploy (经
-- load_caches auto-seed) 行为一致.
INSERT INTO "system_config" (
    "id", "online_model",
    "local_chat_model", "local_small_model",
    "remote_chat_model", "remote_small_model",
    "updated_at"
) VALUES (
    1, false,
    'qwen2.5:14b', 'qwen2.5:7b',
    'qwen3.5-plus', 'qwen3.5-flash',
    CURRENT_TIMESTAMP
);
