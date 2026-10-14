-- Additive compression input receipts; no historic origin backfill.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
ALTER TABLE memory_input_origins DROP CONSTRAINT memory_input_origins_source_status_check;
ALTER TABLE memory_input_origins ADD CONSTRAINT memory_input_origins_source_status_check
 CHECK(source_status IN ('knowledge_document','admin_edit','user_edit','admin_repair','contradiction_confirmation','daily_summary','consolidation'));
ALTER TABLE memory_input_origins ADD CONSTRAINT memory_consolidation_origin_check CHECK (
 source_status <> 'consolidation' OR (
 payload_text::jsonb->'input'->>'format'='l3-compression-v1'
 AND payload_text::jsonb->'input'->>'side' IN ('user','ai')
 AND payload_text::jsonb->'input'->>'cluster_key' ~ '^[a-f0-9]{64}$'
 AND jsonb_typeof(payload_text::jsonb->'input'->'inputs'->'parents')='array'
 AND jsonb_array_length(payload_text::jsonb->'input'->'inputs'->'parents') BETWEEN 5 AND 300
 AND payload_text::jsonb->'input'->'inputs'->>'side'=payload_text::jsonb->'input'->>'side'
 ) IS TRUE);
CREATE UNIQUE INDEX memory_consolidation_origin_cluster_key ON memory_input_origins
 (workspace_id,(payload_text::jsonb->'input'->>'side'),(payload_text::jsonb->'input'->>'cluster_key'))
 WHERE source_status='consolidation';
CREATE FUNCTION validate_consolidation_evidence() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE parent_side TEXT;
BEGIN
 IF NEW.source_kind='input' AND NEW.source_status='consolidation' THEN
   SELECT payload_text::jsonb->'input'->>'side' INTO parent_side FROM memory_input_origins WHERE id=NEW.source_ref;
   IF parent_side IS DISTINCT FROM NEW.memory_source OR NOT EXISTS (
      SELECT 1 FROM memories_ai WHERE NEW.memory_source='ai' AND id=NEW.memory_id AND provenance='consolidated' AND level=3
      UNION ALL
      SELECT 1 FROM memories_user WHERE NEW.memory_source='user' AND id=NEW.memory_id AND provenance='consolidated' AND level=3
   ) THEN RAISE EXCEPTION 'consolidation_invalid_target' USING ERRCODE='23514'; END IF;
 END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER memory_consolidation_evidence_validate BEFORE INSERT ON memory_evidence_links
 FOR EACH ROW EXECUTE FUNCTION validate_consolidation_evidence();
COMMIT;
