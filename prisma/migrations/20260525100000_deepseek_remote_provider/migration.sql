ALTER TABLE "system_config"
    ADD COLUMN "remote_provider" TEXT;

ALTER TABLE "agent_config_overrides"
    ADD COLUMN "remote_provider" TEXT;

UPDATE "system_config"
SET "remote_provider" = 'dashscope'
WHERE "remote_provider" IS NULL;

INSERT INTO "model_registry" (
    "id", "identifier", "display_name", "provider", "enabled",
    "context_window", "input_cost_per_million", "output_cost_per_million", "notes", "updated_at"
) VALUES
    (gen_random_uuid(), 'deepseek-v4-pro', 'DeepSeek V4 Pro', 'deepseek', true, 1000000, NULL, NULL, 'Direct DeepSeek API; fill pricing manually in admin.', CURRENT_TIMESTAMP),
    (gen_random_uuid(), 'deepseek-v4-flash', 'DeepSeek V4 Flash', 'deepseek', true, 1000000, NULL, NULL, 'Direct DeepSeek API; fill pricing manually in admin.', CURRENT_TIMESTAMP)
ON CONFLICT ("identifier") DO NOTHING;
