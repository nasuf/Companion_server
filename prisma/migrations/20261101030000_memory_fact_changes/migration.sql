-- Confirmed change metadata supplements the existing quality-state replacement
-- chain. No historical inference, backfill, or new replacement authority.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
CREATE TABLE memory_fact_changes (
 id TEXT PRIMARY KEY REFERENCES memory_input_origins(id) ON DELETE CASCADE,
 user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 workspace_id TEXT NOT NULL REFERENCES chat_workspaces(id) ON DELETE CASCADE,
 old_memory_id TEXT NOT NULL REFERENCES memories_user(id) ON DELETE CASCADE,
 new_memory_id TEXT NOT NULL REFERENCES memories_user(id) ON DELETE CASCADE,
 old_version_id BIGINT NOT NULL REFERENCES memory_temporal_versions(id) ON DELETE CASCADE,
 new_version_id BIGINT NOT NULL REFERENCES memory_temporal_versions(id) ON DELETE CASCADE,
 kind TEXT NOT NULL CHECK(kind IN ('transition','correction','addition','planned')),
 semantics JSONB NOT NULL CHECK(jsonb_typeof(semantics)='object'),
 statement_at TIMESTAMPTZ NOT NULL,
 boundary_start TIMESTAMPTZ,
 boundary_end TIMESTAMPTZ,
 -- Keep the committed retraction when the source is later erased. This opaque
 -- pin is never a source link without live hash/scope/time revalidation. The
 -- entire record still cascades with either memory, input, user or workspace.
 boundary_source_id TEXT,
 recorded_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
 CHECK(old_memory_id<>new_memory_id),
 CHECK((boundary_start IS NULL AND boundary_end IS NULL AND boundary_source_id IS NULL)
    OR (boundary_start IS NOT NULL AND boundary_end>boundary_start AND boundary_source_id IS NOT NULL)),
 UNIQUE(new_memory_id)
);
CREATE INDEX memory_fact_changes_old_idx ON memory_fact_changes(user_id,workspace_id,old_memory_id,recorded_at DESC);
CREATE INDEX memory_fact_changes_new_idx ON memory_fact_changes(user_id,workspace_id,new_memory_id,recorded_at DESC);
CREATE INDEX memory_fact_changes_boundary_source_idx ON memory_fact_changes(boundary_source_id);
-- Cascaded deletes must not scan: index every referencing column.
CREATE INDEX memory_fact_changes_old_memory_idx ON memory_fact_changes(old_memory_id);
CREATE INDEX memory_fact_changes_old_version_idx ON memory_fact_changes(old_version_id);
CREATE INDEX memory_fact_changes_new_version_idx ON memory_fact_changes(new_version_id);
CREATE INDEX memory_fact_changes_workspace_idx ON memory_fact_changes(workspace_id);
CREATE UNIQUE INDEX memory_fact_changes_replacement_idx ON memory_fact_changes(old_memory_id)
 WHERE kind IN ('transition','correction');
CREATE FUNCTION validate_memory_fact_change() RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
 IF TG_OP='UPDATE' THEN RAISE EXCEPTION 'memory_fact_change_immutable' USING ERRCODE='23514'; END IF;
 IF NOT EXISTS (
  SELECT 1 FROM memory_input_origins i
  JOIN memories_user o ON o.id=NEW.old_memory_id AND o.user_id=i.user_id AND o.workspace_id=i.workspace_id
  JOIN memories_user n ON n.id=NEW.new_memory_id AND n.user_id=i.user_id AND n.workspace_id=i.workspace_id
  JOIN memory_temporal_versions ov ON ov.id=NEW.old_version_id AND ov.side='user' AND ov.memory_id=o.id
    AND ov.user_id=i.user_id AND ov.workspace_id=i.workspace_id
  JOIN memory_temporal_versions nv ON nv.id=NEW.new_version_id AND nv.side='user' AND nv.memory_id=n.id
    AND nv.user_id=i.user_id AND nv.workspace_id=i.workspace_id
  WHERE i.id=NEW.id AND i.user_id=NEW.user_id AND i.workspace_id=NEW.workspace_id
    AND i.source_status='contradiction_confirmation'
    AND i.payload_text::jsonb->'input'->>'old_memory_id'=o.id
    AND i.payload_text::jsonb->'input'->>'proposed_content'=n.content
    AND i.payload_text::jsonb->'input'->'change_semantics'=NEW.semantics
    AND NEW.semantics->>'kind'=NEW.kind
    AND (NEW.semantics->>'statement_at')::timestamptz=NEW.statement_at
    AND (NEW.semantics->'boundary'->>'start')::timestamptz IS NOT DISTINCT FROM NEW.boundary_start
    AND (NEW.semantics->'boundary'->>'end_exclusive')::timestamptz IS NOT DISTINCT FROM NEW.boundary_end
    AND NEW.semantics->'boundary'->>'source_message_id' IS NOT DISTINCT FROM NEW.boundary_source_id
    AND jsonb_typeof(NEW.semantics->'fact_source_ids')='array'
    AND jsonb_array_length(NEW.semantics->'fact_source_ids') BETWEEN 1 AND 40
    AND NOT EXISTS (
      SELECT 1 FROM jsonb_array_elements_text(NEW.semantics->'fact_source_ids') fact_id
      WHERE NOT EXISTS (SELECT 1 FROM jsonb_array_elements(i.payload_text::jsonb->'input'->'sources') s
        WHERE s->>'kind'='message' AND s->>'ref'=fact_id))
    AND ov.content_version=encode(sha256(convert_to(o.content,'UTF8')),'hex')
    AND nv.content_version=encode(sha256(convert_to(n.content,'UTF8')),'hex')
    AND NOT n.is_archived
    AND (NEW.kind NOT IN ('transition','correction') OR o.is_archived)
  FOR SHARE OF i,o,n,ov,nv
 ) THEN RAISE EXCEPTION 'memory_fact_change_scope_or_version' USING ERRCODE='23514'; END IF;
 IF NEW.boundary_source_id IS NOT NULL AND NOT EXISTS (
  SELECT 1 FROM messages m JOIN conversations c ON c.id=m.conversation_id
  JOIN memory_input_origins i ON i.id=NEW.id
  CROSS JOIN LATERAL jsonb_array_elements(i.payload_text::jsonb->'input'->'sources') s
  WHERE m.id=NEW.boundary_source_id AND c.user_id=NEW.user_id AND c.workspace_id=NEW.workspace_id
    AND m.role='user' AND NOT c.is_deleted AND c.archived_at IS NULL
    AND s->>'kind'='message' AND s->>'ref'=m.id
    AND s->>'expected_version'=encode(sha256(convert_to(m.content,'UTF8')),'hex')
    AND (s->>'expected_statement_time')::timestamptz=m.created_at AT TIME ZONE 'UTC'
 ) THEN RAISE EXCEPTION 'memory_fact_change_boundary_source' USING ERRCODE='23514'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER memory_fact_change_validate BEFORE INSERT OR UPDATE ON memory_fact_changes
 FOR EACH ROW EXECUTE FUNCTION validate_memory_fact_change();
-- A retracted source must not become valid again when its replacement memory
-- or input history is erased. Store only the original message pin; deleting
-- that message, its workspace or owner erases this fence as well. Only the
-- corrected memory's fact sources are fenced (not other batch input), and
-- memory_id is an opaque pin of the corrected memory: reads fence that memory's
-- evidence only, so other facts stated in the same message keep theirs.
CREATE TABLE memory_source_retractions (
 source_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
 source_version TEXT NOT NULL CHECK(length(source_version)=64),
 source_statement_time TIMESTAMP NOT NULL,
 user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 workspace_id TEXT NOT NULL REFERENCES chat_workspaces(id) ON DELETE CASCADE,
 memory_id TEXT NOT NULL,
 origin_id TEXT NOT NULL,
 recorded_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
 PRIMARY KEY(source_message_id,source_version,memory_id)
);
CREATE INDEX memory_source_retractions_scope_idx ON memory_source_retractions(user_id,workspace_id);
CREATE INDEX memory_source_retractions_workspace_idx ON memory_source_retractions(workspace_id);
CREATE INDEX memory_source_retractions_memory_idx ON memory_source_retractions(memory_id);
CREATE FUNCTION validate_memory_source_retraction() RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
 IF TG_OP='UPDATE' THEN RAISE EXCEPTION 'memory_source_retraction_immutable' USING ERRCODE='23514'; END IF;
 IF NOT EXISTS (
   SELECT 1 FROM memory_fact_changes f
   JOIN memory_evidence_links e ON e.memory_id=f.old_memory_id AND e.memory_source='user'
     AND e.user_id=f.user_id AND e.workspace_id=f.workspace_id
   JOIN messages m ON m.id=e.source_message_id
   JOIN conversations c ON c.id=m.conversation_id
   WHERE f.id=NEW.origin_id AND f.kind='correction' AND f.user_id=NEW.user_id
     AND f.workspace_id=NEW.workspace_id AND m.id=NEW.source_message_id
     AND f.old_memory_id=NEW.memory_id
     AND COALESCE(e.temporal_basis->>'source_role','fact')='fact'
     AND c.user_id=NEW.user_id AND c.workspace_id=NEW.workspace_id AND m.role='user'
     AND e.relation IN ('extracted_from','recorded_from')
     AND e.source_version=NEW.source_version
     AND e.source_statement_time=NEW.source_statement_time
     AND m.created_at=NEW.source_statement_time
     AND encode(sha256(convert_to(m.content,'UTF8')),'hex')=NEW.source_version
   FOR SHARE OF m,c
 ) THEN RAISE EXCEPTION 'memory_source_retraction_invalid' USING ERRCODE='23514'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER memory_source_retraction_validate BEFORE INSERT OR UPDATE ON memory_source_retractions
 FOR EACH ROW EXECUTE FUNCTION validate_memory_source_retraction();
COMMIT;
