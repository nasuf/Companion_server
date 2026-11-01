-- Additive source assertions, not a second memory store. No legacy backfill.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';
CREATE TABLE memory_temporal_assertions (
  id TEXT PRIMARY KEY REFERENCES memory_evidence_links(id) ON DELETE CASCADE,
  temporal_version_id BIGINT NOT NULL REFERENCES memory_temporal_versions(id) ON DELETE CASCADE,
  source_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  workspace_id TEXT NOT NULL REFERENCES chat_workspaces(id) ON DELETE CASCADE,
  side TEXT NOT NULL CHECK(side IN ('user','ai')),
  memory_id TEXT NOT NULL,
  assertion JSONB NOT NULL CHECK(jsonb_typeof(assertion)='object' AND octet_length(assertion::text)<=4096),
  statement_at TIMESTAMPTZ NOT NULL,
  valid_from TIMESTAMPTZ,
  valid_to TIMESTAMPTZ,
  recorded_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  retracted_at TIMESTAMPTZ,
  CHECK(valid_from IS NULL OR valid_to IS NULL OR valid_from<valid_to),
  CHECK(retracted_at IS NULL OR retracted_at>=recorded_at)
);
CREATE INDEX memory_temporal_assertions_page_idx ON memory_temporal_assertions
  (workspace_id,side,memory_id,statement_at DESC,id);
CREATE INDEX memory_temporal_assertions_version_idx ON memory_temporal_assertions(temporal_version_id);
CREATE INDEX memory_temporal_assertions_source_idx ON memory_temporal_assertions(source_message_id);
CREATE INDEX memory_temporal_assertions_user_idx ON memory_temporal_assertions(user_id);

CREATE FUNCTION validate_memory_temporal_assertion() RETURNS TRIGGER LANGUAGE plpgsql AS $$
DECLARE evidence RECORD;
BEGIN
  IF TG_OP='UPDATE' THEN
    IF (to_jsonb(NEW)-'retracted_at') IS DISTINCT FROM (to_jsonb(OLD)-'retracted_at')
       OR OLD.retracted_at IS NOT NULL OR NEW.retracted_at IS NULL THEN
      RAISE EXCEPTION 'temporal_assertion_immutable' USING ERRCODE='23514';
    END IF;
    RETURN NEW;
  END IF;
  SELECT e.*,m.created_at AS live_statement_at,m.content AS live_text INTO evidence FROM memory_evidence_links e
    JOIN messages m ON m.id=e.source_message_id
    JOIN conversations c ON c.id=m.conversation_id
    JOIN memory_temporal_versions v ON v.id=e.temporal_version_id
    WHERE e.id=NEW.id AND e.source_kind='message' AND NOT c.is_deleted
      AND e.user_id=c.user_id AND e.workspace_id=c.workspace_id AND e.agent_id=c.agent_id
      AND e.source_version=encode(sha256(convert_to(m.content,'UTF8')),'hex')
      AND e.source_role=m.role AND e.temporal_basis->>'source_role'='fact'
    FOR SHARE OF e,m,c,v;
  IF NOT FOUND OR (evidence.temporal_version_id,evidence.source_message_id,evidence.user_id,
       evidence.workspace_id,evidence.memory_source,evidence.memory_id)
       IS DISTINCT FROM (NEW.temporal_version_id,NEW.source_message_id,NEW.user_id,
       NEW.workspace_id,NEW.side,NEW.memory_id)
     OR evidence.source_role IS DISTINCT FROM (CASE NEW.side WHEN 'user' THEN 'user' ELSE 'assistant' END)
     OR evidence.source_statement_time IS DISTINCT FROM evidence.live_statement_at
     OR NEW.statement_at IS DISTINCT FROM (evidence.live_statement_at AT TIME ZONE 'UTC')
     OR evidence.temporal_basis->'validity'->>'state' IS DISTINCT FROM 'asserted'
     OR NEW.assertion IS DISTINCT FROM evidence.temporal_basis->'validity'
     OR NEW.assertion->>'version' IS DISTINCT FROM '1'
     OR NEW.assertion->>'truth' IS DISTINCT FROM 'speaker_assertion'
     OR NEW.assertion->>'end_boundary' IS DISTINCT FROM 'exclusive'
     OR jsonb_typeof(NEW.assertion->'source_quote') IS DISTINCT FROM 'string'
     OR length(NEW.assertion->>'source_quote') NOT BETWEEN 1 AND 500
     OR position(NEW.assertion->>'source_quote' IN evidence.live_text)=0
     OR COALESCE(NEW.assertion->>'kind','') NOT IN ('point','state','stable','plan','recurring')
     OR COALESCE(NEW.assertion->>'status','') NOT IN ('planned','occurred','ongoing','completed','cancelled','uncertain')
     OR NEW.valid_from IS DISTINCT FROM (NEW.assertion->>'valid_from')::timestamptz
     OR NEW.valid_to IS DISTINCT FROM (NEW.assertion->>'valid_to')::timestamptz
     OR NEW.statement_at IS DISTINCT FROM (NEW.assertion->>'statement_at')::timestamptz
  THEN RAISE EXCEPTION 'temporal_assertion_scope_or_source' USING ERRCODE='23514'; END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER memory_temporal_assertion_validate BEFORE INSERT OR UPDATE ON memory_temporal_assertions
  FOR EACH ROW EXECUTE FUNCTION validate_memory_temporal_assertion();

-- Retraction is a system observation, NOT the end of the fact's real validity.
CREATE FUNCTION retract_memory_temporal_assertions() RETURNS TRIGGER LANGUAGE plpgsql AS $$
DECLARE owner_side TEXT;
BEGIN
  IF (NEW.content,NEW.occur_time,NEW.statement_time,NEW.is_archived)
     IS NOT DISTINCT FROM (OLD.content,OLD.occur_time,OLD.statement_time,OLD.is_archived) THEN RETURN NEW; END IF;
  owner_side:=CASE TG_TABLE_NAME WHEN 'memories_user' THEN 'user' ELSE 'ai' END;
  -- A transaction may have started before another writer inserted the
  -- assertion. Its transaction timestamp is not the retraction observation.
  UPDATE memory_temporal_assertions SET retracted_at=GREATEST(clock_timestamp(),recorded_at)
    WHERE side=owner_side AND memory_id=OLD.id AND retracted_at IS NULL;
  RETURN NEW;
END $$;
CREATE TRIGGER zz_memory_user_assertion_retract AFTER UPDATE ON memories_user
  FOR EACH ROW EXECUTE FUNCTION retract_memory_temporal_assertions();
CREATE TRIGGER zz_memory_ai_assertion_retract AFTER UPDATE ON memories_ai
  FOR EACH ROW EXECUTE FUNCTION retract_memory_temporal_assertions();

-- New replacement edges are scoped and acyclic. Historical edges are not
-- rewritten; readers still validate every hop and refuse ambiguous branches.
CREATE FUNCTION validate_memory_replacement_edge() RETURNS TRIGGER LANGUAGE plpgsql AS $$
DECLARE node TEXT; visited TEXT[]; next_id TEXT; table_name TEXT; found_owner RECORD;
BEGIN
  IF NEW.superseded_by_memory_id IS NULL THEN RETURN NEW; END IF;
  IF TG_OP='UPDATE' AND (NEW.superseded_by_memory_id,NEW.user_id,NEW.workspace_id,NEW.memory_source,NEW.memory_id)
    IS NOT DISTINCT FROM (OLD.superseded_by_memory_id,OLD.user_id,OLD.workspace_id,OLD.memory_source,OLD.memory_id)
    THEN RETURN NEW; END IF;
  IF NEW.memory_source NOT IN ('user','ai') OR NEW.workspace_id IS NULL THEN
    RAISE EXCEPTION 'memory_replacement_scope' USING ERRCODE='23514'; END IF;
  IF current_setting('transaction_isolation')='repeatable read' THEN
    RAISE EXCEPTION 'memory_replacement_requires_fresh_snapshot' USING ERRCODE='23514'; END IF;
  PERFORM pg_advisory_xact_lock(hashtextextended('memory-replacement:'||NEW.workspace_id,0));
  table_name:=CASE NEW.memory_source WHEN 'user' THEN 'memories_user' ELSE 'memories_ai' END;
  EXECUTE format('SELECT user_id,workspace_id FROM %I WHERE id=$1',table_name)
    INTO found_owner USING NEW.memory_id;
  IF found_owner.user_id IS DISTINCT FROM NEW.user_id OR found_owner.workspace_id IS DISTINCT FROM NEW.workspace_id THEN
    RAISE EXCEPTION 'memory_replacement_scope' USING ERRCODE='23514'; END IF;
  visited:=ARRAY[NEW.memory_id]; node:=NEW.superseded_by_memory_id;
  LOOP
    IF node=ANY(visited) OR cardinality(visited)>=64 THEN
      RAISE EXCEPTION 'memory_replacement_cycle_or_limit' USING ERRCODE='23514'; END IF;
    EXECUTE format('SELECT user_id,workspace_id FROM %I WHERE id=$1',table_name)
      INTO found_owner USING node;
    IF found_owner.user_id IS DISTINCT FROM NEW.user_id OR found_owner.workspace_id IS DISTINCT FROM NEW.workspace_id THEN
      RAISE EXCEPTION 'memory_replacement_scope' USING ERRCODE='23514'; END IF;
    visited:=visited||node;
    SELECT q.superseded_by_memory_id INTO next_id FROM memory_quality_states q
      WHERE q.memory_id=node AND q.memory_source=NEW.memory_source
        AND q.user_id=NEW.user_id AND q.workspace_id=NEW.workspace_id;
    EXIT WHEN next_id IS NULL;
    node:=next_id;
  END LOOP;
  RETURN NEW;
END $$;
CREATE TRIGGER memory_replacement_edge_validate BEFORE INSERT OR UPDATE ON memory_quality_states
  FOR EACH ROW EXECUTE FUNCTION validate_memory_replacement_edge();
COMMIT;
