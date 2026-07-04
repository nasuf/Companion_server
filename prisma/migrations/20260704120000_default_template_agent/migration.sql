-- Add the default template agent pointer to the singleton system_config row.
-- New users (e.g. WeChat Mini Program first login) are cloned from this agent so
-- they can chat immediately without waiting for LLM provisioning.
ALTER TABLE system_config
    ADD COLUMN IF NOT EXISTS default_template_agent_id TEXT;
