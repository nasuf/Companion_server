-- Track which template an agent was cloned from, so the admin 模板管理 tab can
-- show how many in-use agents came from each template.
ALTER TABLE ai_agents
    ADD COLUMN IF NOT EXISTS source_template_id TEXT;

CREATE INDEX IF NOT EXISTS ai_agents_source_template_idx
    ON ai_agents(source_template_id);
