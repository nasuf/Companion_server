-- Append-only semantic history. No backfill: legacy rows get a labelled baseline
-- at their first semantic update; recorded_at is never their original event time.
CREATE TABLE memory_temporal_versions (
 id BIGSERIAL PRIMARY KEY,
 memory_id TEXT NOT NULL,
 side TEXT NOT NULL CHECK(side IN ('user','ai')),
 user_memory_id TEXT REFERENCES memories_user(id) ON DELETE CASCADE,
 ai_memory_id TEXT REFERENCES memories_ai(id) ON DELETE CASCADE,
 user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 workspace_id TEXT REFERENCES chat_workspaces(id) ON DELETE CASCADE,
 content TEXT NOT NULL,
 content_version TEXT NOT NULL,
 occur_time TIMESTAMPTZ,
 statement_time TIMESTAMPTZ,
 is_archived BOOLEAN NOT NULL,
 record_kind TEXT NOT NULL CHECK(record_kind IN ('insert','baseline','update')),
 recorded_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 CHECK((side='user' AND user_memory_id IS NOT NULL AND user_memory_id=memory_id AND ai_memory_id IS NULL)
    OR (side='ai' AND ai_memory_id IS NOT NULL AND ai_memory_id=memory_id AND user_memory_id IS NULL))
);
CREATE INDEX memory_temporal_versions_page_idx ON memory_temporal_versions(side,memory_id,id DESC);
-- PostgreSQL does not index referencing FK columns automatically. Cascaded
-- erasure must use these indexes, including large legacy/workspace deletions.
CREATE INDEX memory_temporal_versions_user_memory_idx ON memory_temporal_versions(user_memory_id) WHERE user_memory_id IS NOT NULL;
CREATE INDEX memory_temporal_versions_ai_memory_idx ON memory_temporal_versions(ai_memory_id) WHERE ai_memory_id IS NOT NULL;
CREATE INDEX memory_temporal_versions_user_idx ON memory_temporal_versions(user_id);
CREATE INDEX memory_temporal_versions_workspace_idx ON memory_temporal_versions(workspace_id);
CREATE FUNCTION capture_memory_temporal_version() RETURNS TRIGGER LANGUAGE plpgsql AS $$
DECLARE owner_side TEXT; previous_exists BOOLEAN;
BEGIN
 owner_side:=CASE TG_TABLE_NAME WHEN 'memories_user' THEN 'user' ELSE 'ai' END;
 IF TG_OP='UPDATE' THEN
  IF (NEW.user_id,NEW.workspace_id) IS DISTINCT FROM (OLD.user_id,OLD.workspace_id) THEN
   -- History must never be transferred to a different owner or workspace.
   DELETE FROM memory_temporal_versions WHERE side=owner_side AND memory_id=OLD.id;
  ELSIF (NEW.content,NEW.occur_time,NEW.statement_time,NEW.is_archived)
      IS NOT DISTINCT FROM (OLD.content,OLD.occur_time,OLD.statement_time,OLD.is_archived) THEN
   RETURN NEW;
  ELSE
   previous_exists:=EXISTS(SELECT 1 FROM memory_temporal_versions WHERE side=owner_side AND memory_id=OLD.id);
   IF NOT previous_exists THEN
    INSERT INTO memory_temporal_versions(memory_id,side,user_memory_id,ai_memory_id,user_id,workspace_id,
      content,content_version,occur_time,statement_time,is_archived,record_kind)
    VALUES(OLD.id,owner_side,CASE WHEN owner_side='user' THEN OLD.id END,CASE WHEN owner_side='ai' THEN OLD.id END,
      OLD.user_id,OLD.workspace_id,OLD.content,encode(sha256(convert_to(OLD.content,'UTF8')),'hex'),
      OLD.occur_time AT TIME ZONE 'UTC',OLD.statement_time AT TIME ZONE 'UTC',OLD.is_archived,'baseline');
   END IF;
  END IF;
 END IF;
 INSERT INTO memory_temporal_versions(memory_id,side,user_memory_id,ai_memory_id,user_id,workspace_id,
   content,content_version,occur_time,statement_time,is_archived,record_kind)
 VALUES(NEW.id,owner_side,CASE WHEN owner_side='user' THEN NEW.id END,CASE WHEN owner_side='ai' THEN NEW.id END,
   NEW.user_id,NEW.workspace_id,NEW.content,encode(sha256(convert_to(NEW.content,'UTF8')),'hex'),
   NEW.occur_time AT TIME ZONE 'UTC',NEW.statement_time AT TIME ZONE 'UTC',NEW.is_archived,
   CASE TG_OP WHEN 'INSERT' THEN 'insert' ELSE 'update' END);
 RETURN NEW;
END $$;
CREATE TRIGGER memory_user_temporal_history AFTER INSERT OR UPDATE ON memories_user
 FOR EACH ROW EXECUTE FUNCTION capture_memory_temporal_version();
CREATE TRIGGER memory_ai_temporal_history AFTER INSERT OR UPDATE ON memories_ai
 FOR EACH ROW EXECUTE FUNCTION capture_memory_temporal_version();
-- New evidence is pinned to a semantic revision, not just equal content text.
ALTER TABLE memory_evidence_links ADD COLUMN temporal_version_id BIGINT
 REFERENCES memory_temporal_versions(id) ON DELETE SET NULL;
CREATE INDEX memory_evidence_temporal_version_idx ON memory_evidence_links(temporal_version_id)
 WHERE temporal_version_id IS NOT NULL;
-- A receipt pins semantic identity as well as text. Timestamps can be corrected
-- without changing content; that must still invalidate old pending effects.
ALTER TABLE memory_write_receipts ADD COLUMN memory_revision BIGINT;
CREATE FUNCTION validate_memory_temporal_binding() RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
 IF NEW.temporal_version_id IS NOT NULL AND NOT EXISTS (
   SELECT 1 FROM memory_temporal_versions v WHERE v.id=NEW.temporal_version_id
     AND v.memory_id=NEW.memory_id AND v.side=NEW.memory_source
     AND v.user_id=NEW.user_id AND v.workspace_id IS NOT DISTINCT FROM NEW.workspace_id
     AND v.content_version=NEW.content_version
 ) THEN RAISE EXCEPTION 'memory_temporal_binding_scope' USING ERRCODE='23514'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER memory_evidence_temporal_validate BEFORE INSERT ON memory_evidence_links
 FOR EACH ROW EXECUTE FUNCTION validate_memory_temporal_binding();

-- A deleted semantic revision behaves like the existing live source pointers:
-- only detachment is allowed; snapshots and non-null retargeting stay immutable.
CREATE OR REPLACE FUNCTION preserve_memory_evidence_snapshot() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF (to_jsonb(NEW)-'source_message_id'-'parent_user_memory_id'-'parent_ai_memory_id'-'source_profile_id'-'source_input_id'-'temporal_version_id')
       IS DISTINCT FROM (to_jsonb(OLD)-'source_message_id'-'parent_user_memory_id'-'parent_ai_memory_id'-'source_profile_id'-'source_input_id'-'temporal_version_id')
       OR (NEW.source_message_id IS NOT NULL AND NEW.source_message_id IS DISTINCT FROM OLD.source_message_id)
       OR (NEW.parent_user_memory_id IS NOT NULL AND NEW.parent_user_memory_id IS DISTINCT FROM OLD.parent_user_memory_id)
       OR (NEW.parent_ai_memory_id IS NOT NULL AND NEW.parent_ai_memory_id IS DISTINCT FROM OLD.parent_ai_memory_id)
       OR (NEW.source_profile_id IS NOT NULL AND NEW.source_profile_id IS DISTINCT FROM OLD.source_profile_id)
       OR (NEW.source_input_id IS NOT NULL AND NEW.source_input_id IS DISTINCT FROM OLD.source_input_id)
       OR (NEW.temporal_version_id IS NOT NULL AND NEW.temporal_version_id IS DISTINCT FROM OLD.temporal_version_id) THEN
        RAISE EXCEPTION 'memory_evidence_immutable' USING ERRCODE='23514';
    END IF;
    RETURN NEW;
END $$;
