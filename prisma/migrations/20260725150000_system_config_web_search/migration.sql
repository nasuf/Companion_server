-- Web search switch for main replies (Ark Responses API web_search tool).
-- Global only; NULL = fall back to env WEB_SEARCH_ENABLED (default false).
ALTER TABLE "system_config" ADD COLUMN IF NOT EXISTS "web_search_enabled" BOOLEAN;
