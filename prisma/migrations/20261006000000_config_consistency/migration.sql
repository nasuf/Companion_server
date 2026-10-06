-- Additive only. Historical content, enabled states and version rows remain intact.
ALTER TABLE prompt_templates ADD COLUMN revision INTEGER NOT NULL DEFAULT 1,
  ADD COLUMN web_managed BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE prompt_template_versions ADD COLUMN revision INTEGER;
-- Missing/unknown history and unrecorded customizations are protected too.
UPDATE prompt_templates p SET web_managed = true
WHERE p.content IS DISTINCT FROM p.default_content
   OR NOT EXISTS (SELECT 1 FROM prompt_template_versions v WHERE v.prompt_id=p.id
                  AND v.change_type='bootstrap'
                  AND v.created_at BETWEEN p.created_at - INTERVAL '1 minute' AND p.created_at + INTERVAL '1 minute')
   OR EXISTS (SELECT 1 FROM prompt_template_versions v WHERE v.prompt_key=p.key
              AND v.change_type NOT IN ('bootstrap','code_sync','enable','disable'));
CREATE TABLE configuration_revisions (scope TEXT PRIMARY KEY, revision BIGINT NOT NULL DEFAULT 1);
INSERT INTO configuration_revisions(scope) VALUES ('models');
CREATE FUNCTION bump_prompt_revision() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  NEW.revision := OLD.revision + 1;
  -- Ownership is sticky, including a later reset to default.
  NEW.web_managed := OLD.web_managed OR NEW.web_managed;
  RETURN NEW;
END $$;
CREATE TRIGGER prompt_revision BEFORE UPDATE ON prompt_templates
FOR EACH ROW EXECUTE FUNCTION bump_prompt_revision();
CREATE FUNCTION bump_configuration_revision() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 UPDATE configuration_revisions SET revision=revision+1 WHERE scope='models';
 RETURN NULL;
END $$;
-- Covers every writer, including scripts and old API workers during rollback.
CREATE TRIGGER system_config_revision AFTER INSERT OR UPDATE OR DELETE ON system_config
FOR EACH STATEMENT EXECUTE FUNCTION bump_configuration_revision();
CREATE TRIGGER agent_config_revision AFTER INSERT OR UPDATE OR DELETE ON agent_config_overrides
FOR EACH STATEMENT EXECUTE FUNCTION bump_configuration_revision();
CREATE TRIGGER model_registry_revision AFTER INSERT OR UPDATE OR DELETE ON model_registry
FOR EACH STATEMENT EXECUTE FUNCTION bump_configuration_revision();
