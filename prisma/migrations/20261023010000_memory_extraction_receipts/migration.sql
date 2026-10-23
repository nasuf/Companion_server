-- New conversations are admitted by the application; no historical backfill.
CREATE TABLE memory_extraction_batches (
 id TEXT PRIMARY KEY,
 user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 workspace_id TEXT NOT NULL REFERENCES chat_workspaces(id) ON DELETE CASCADE,
 conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
 side TEXT NOT NULL CHECK (side IN ('user','ai')),
 scope JSONB NOT NULL,
 sources JSONB NOT NULL CHECK (jsonb_typeof(sources)='array' AND jsonb_array_length(sources) BETWEEN 1 AND 100),
 extraction JSONB,
 fact_count INTEGER CHECK(fact_count BETWEEN 0 AND 100),
 prepared_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 status TEXT NOT NULL DEFAULT 'preparing' CHECK (status IN ('preparing','processing','completed','blocked')),
 result_ids JSONB NOT NULL DEFAULT '[]',
 last_error TEXT,
 updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 CHECK (extraction IS NULL OR (jsonb_typeof(extraction)='object' AND octet_length(extraction::text)<=262144))
);
CREATE INDEX memory_extraction_batches_scope_idx ON memory_extraction_batches(conversation_id,side,prepared_at,id);
CREATE INDEX memory_extraction_batches_recent_idx ON memory_extraction_batches(prepared_at DESC,id DESC);
CREATE INDEX memory_extraction_batches_user_idx ON memory_extraction_batches(user_id);
CREATE INDEX memory_extraction_batches_workspace_idx ON memory_extraction_batches(workspace_id,side);
CREATE TABLE memory_extraction_sources (
 message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
 side TEXT NOT NULL CHECK (side IN ('user','ai')),
 batch_id TEXT NOT NULL REFERENCES memory_extraction_batches(id) ON DELETE CASCADE,
 PRIMARY KEY(message_id,side)
);
CREATE INDEX memory_extraction_sources_batch_idx ON memory_extraction_sources(batch_id);
CREATE TABLE memory_write_receipts (
 batch_id TEXT NOT NULL REFERENCES memory_extraction_batches(id) ON DELETE CASCADE,
 fact_index INTEGER NOT NULL CHECK(fact_index BETWEEN 0 AND 99),
 piece_index INTEGER NOT NULL CHECK(piece_index BETWEEN -1 AND 99),
 input_hash TEXT NOT NULL CHECK(length(input_hash)=64),
 status TEXT NOT NULL CHECK(status IN ('written','skipped','erased')),
 memory_id TEXT,
 memory_version TEXT,
 followup_done BOOLEAN NOT NULL DEFAULT false,
 followup_outcome TEXT CHECK(followup_outcome IN ('applied','superseded','erased')),
 created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 PRIMARY KEY(batch_id,fact_index,piece_index)
);
CREATE INDEX memory_write_receipts_memory_idx ON memory_write_receipts(memory_id) WHERE memory_id IS NOT NULL;
CREATE TABLE memory_erasure_epochs (
 workspace_id TEXT PRIMARY KEY REFERENCES chat_workspaces(id) ON DELETE CASCADE,
 generation BIGINT NOT NULL DEFAULT 0 CHECK(generation>=0)
);
CREATE TABLE memory_erased_sources (
 workspace_id TEXT NOT NULL REFERENCES chat_workspaces(id) ON DELETE CASCADE,
 side TEXT NOT NULL CHECK(side IN ('user','ai')),
 message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
 source_version TEXT NOT NULL,
 erased_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 PRIMARY KEY(workspace_id,side,message_id,source_version)
);
CREATE INDEX memory_erased_sources_message_idx ON memory_erased_sources(message_id);
-- Physical erasure is distinct from archival. Keep only source hashes/IDs,
-- never the erased text. Writers take SHARE on the epoch in their short commit.
CREATE FUNCTION fence_deleted_memory_sources() RETURNS TRIGGER LANGUAGE plpgsql AS $$
DECLARE owner_side TEXT; b RECORD; src JSONB; i INTEGER; indices INTEGER[];
        facts JSONB; fact JSONB; affected BOOLEAN; tracked BOOLEAN; fact_no INTEGER;
BEGIN
 owner_side := CASE TG_TABLE_NAME WHEN 'memories_user' THEN 'user' ELSE 'ai' END;
 IF OLD.workspace_id IS NULL OR NOT EXISTS(SELECT 1 FROM chat_workspaces WHERE id=OLD.workspace_id) THEN RETURN OLD; END IF;
 INSERT INTO memory_erasure_epochs(workspace_id,generation) VALUES(OLD.workspace_id,1)
 ON CONFLICT(workspace_id) DO UPDATE SET generation=memory_erasure_epochs.generation+1;
 INSERT INTO memory_erased_sources(workspace_id,side,message_id,source_version)
 SELECT OLD.workspace_id,owner_side,e.source_message_id,e.source_version
 FROM memory_evidence_links e JOIN messages m ON m.id=e.source_message_id
 WHERE e.memory_id=OLD.id AND e.memory_source=owner_side AND e.source_version IS NOT NULL
   AND COALESCE(e.temporal_basis->>'source_role','fact')='fact'
 ON CONFLICT DO NOTHING;
 tracked := EXISTS(SELECT 1 FROM memory_write_receipts WHERE memory_id=OLD.id AND batch_id IN (SELECT id FROM memory_extraction_batches WHERE side=owner_side AND workspace_id=OLD.workspace_id));
 UPDATE memory_write_receipts SET status='erased',memory_id=NULL,memory_version=NULL,followup_done=true,followup_outcome='erased'
 WHERE memory_id=OLD.id AND batch_id IN (SELECT id FROM memory_extraction_batches WHERE side=owner_side AND workspace_id=OLD.workspace_id);
 -- Redact affected frozen facts while retaining ordinal positions and unrelated
 -- facts. Aggregated hints are optional; clear them rather than retain erased text.
 FOR b IN SELECT * FROM memory_extraction_batches WHERE workspace_id=OLD.workspace_id AND side=owner_side
   AND extraction IS NOT NULL AND (EXISTS(SELECT 1 FROM memory_write_receipts r WHERE r.batch_id=memory_extraction_batches.id AND r.status='erased') OR EXISTS(SELECT 1 FROM memory_extraction_sources s
      JOIN memory_erased_sources e ON e.message_id=s.message_id AND e.side=s.side
      WHERE s.batch_id=memory_extraction_batches.id AND e.workspace_id=OLD.workspace_id)) FOR UPDATE
 LOOP
  indices := ARRAY[]::INTEGER[]; i:=0;
  FOR src IN SELECT value FROM jsonb_array_elements(b.sources) LOOP
   i:=i+1;
   IF EXISTS(SELECT 1 FROM memory_erased_sources WHERE workspace_id=OLD.workspace_id AND side=owner_side
      AND message_id=src->>'ref' AND source_version=src->>'expected_version') THEN indices:=array_append(indices,i); END IF;
  END LOOP;
  facts:='[]'::JSONB; affected:=false; fact_no:=-1;
  FOR fact IN SELECT value FROM jsonb_array_elements(b.extraction->'memories') LOOP
   fact_no:=fact_no+1;
   IF EXISTS(SELECT 1 FROM memory_write_receipts WHERE batch_id=b.id AND fact_index=fact_no AND status='erased')
      OR (NOT tracked AND (jsonb_typeof(fact->'source_indices') IS DISTINCT FROM 'array'
      OR jsonb_array_length(fact->'source_indices')=0
      OR EXISTS(SELECT 1 FROM jsonb_array_elements(fact->'source_indices') v WHERE v.value=ANY(SELECT to_jsonb(n) FROM unnest(indices) n))))
   THEN facts:=facts||jsonb_build_array(jsonb_build_object('_erased',true)); affected:=true;
   ELSE facts:=facts||jsonb_build_array(fact); END IF;
  END LOOP;
  IF affected THEN UPDATE memory_extraction_batches SET extraction=jsonb_build_object('plan_version',b.extraction->'plan_version','memories',facts,'entities','[]'::JSONB,'preferences','[]'::JSONB,'topics','[]'::JSONB),updated_at=CURRENT_TIMESTAMP WHERE id=b.id; END IF;
 END LOOP;
 RETURN OLD;
END $$;
CREATE TRIGGER memory_user_erasure_fence BEFORE DELETE ON memories_user FOR EACH ROW EXECUTE FUNCTION fence_deleted_memory_sources();
CREATE TRIGGER memory_ai_erasure_fence BEFORE DELETE ON memories_ai FOR EACH ROW EXECUTE FUNCTION fence_deleted_memory_sources();

-- Deleting a source cancels its incomplete plan and removes derived payloads.
-- Already committed memories retain their independent deletion policy.
CREATE FUNCTION fence_deleted_memory_input() RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
 UPDATE memory_extraction_batches SET status='blocked',extraction=NULL,last_error='source_deleted',updated_at=CURRENT_TIMESTAMP
 WHERE id IN (SELECT batch_id FROM memory_extraction_sources WHERE message_id=OLD.id)
   AND status IN ('preparing','processing');
 RETURN OLD;
END $$;
CREATE TRIGGER memory_input_erasure_fence BEFORE DELETE ON messages FOR EACH ROW EXECUTE FUNCTION fence_deleted_memory_input();
