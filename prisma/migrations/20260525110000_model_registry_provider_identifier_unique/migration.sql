DROP INDEX IF EXISTS "model_registry_identifier_key";

CREATE UNIQUE INDEX IF NOT EXISTS "model_registry_provider_identifier_key"
    ON "model_registry"("provider", "identifier");

INSERT INTO "model_registry" (
    "id", "identifier", "display_name", "provider", "enabled",
    "context_window", "input_cost_per_million", "output_cost_per_million", "notes", "updated_at"
) VALUES
    (gen_random_uuid(), 'deepseek-v4-pro', 'DeepSeek V4 Pro', 'deepseek', true, 1000000, NULL, NULL, 'Direct DeepSeek API; fill pricing manually in admin.', CURRENT_TIMESTAMP),
    (gen_random_uuid(), 'deepseek-v4-flash', 'DeepSeek V4 Flash', 'deepseek', true, 1000000, NULL, NULL, 'Direct DeepSeek API; fill pricing manually in admin.', CURRENT_TIMESTAMP)
ON CONFLICT ("provider", "identifier") DO NOTHING;
