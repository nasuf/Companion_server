-- Additive immutable daily-review receipts. No historical memory backfill.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
ALTER TABLE memory_input_origins DROP CONSTRAINT memory_input_origins_source_status_check;
ALTER TABLE memory_input_origins ADD CONSTRAINT memory_input_origins_source_status_check
 CHECK(source_status IN ('knowledge_document','admin_edit','user_edit','admin_repair','contradiction_confirmation','daily_summary'));
ALTER TABLE memory_input_origins ADD CONSTRAINT memory_daily_review_origin_check CHECK (
 source_status <> 'daily_summary' OR ((payload_text::jsonb->'input'->>'local_date') ~ '^20[0-9]{2}-[0-9]{2}-[0-9]{2}$'
 AND payload_text::jsonb->'input'->'inputs'->>'format'='daily-review-v1'
 AND char_length(payload_text::jsonb->'input'->'summary'->>'timezone') BETWEEN 1 AND 64
 AND payload_text::jsonb->'input'->'summary'->>'timezone'=payload_text::jsonb->'input'->'inputs'->>'timezone'
 AND payload_text::jsonb->'input'->>'local_date'=payload_text::jsonb->'input'->'inputs'->>'local_date') IS TRUE);
CREATE UNIQUE INDEX memory_daily_review_origin_day_key ON memory_input_origins
 (workspace_id,(payload_text::jsonb->'input'->>'local_date')) WHERE source_status='daily_summary';
CREATE INDEX schedule_adjust_logs_agent_created_idx ON schedule_adjust_logs(agent_id,created_at);
CREATE INDEX game_sessions_workspace_review_day_idx ON game_sessions(workspace_id,(COALESCE(ended_at,created_at)));
CREATE FUNCTION validate_daily_review_evidence() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF NEW.source_kind='input' AND NEW.source_status='daily_summary' AND (
   NEW.memory_source<>'ai' OR NOT EXISTS(SELECT 1 FROM memories_ai WHERE id=NEW.memory_id
   AND provenance='daily_summary' FOR SHARE)) THEN
   RAISE EXCEPTION 'daily_review_invalid_target' USING ERRCODE='23514';
 END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER memory_daily_review_evidence_validate BEFORE INSERT ON memory_evidence_links
 FOR EACH ROW EXECUTE FUNCTION validate_daily_review_evidence();
COMMIT;
