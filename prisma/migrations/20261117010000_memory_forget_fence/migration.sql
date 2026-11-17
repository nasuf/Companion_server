-- User forgetting applies the erasure fence whether the row is deleted or
-- archived. Archive-based deletes (app bulk delete, workspace wipe) call this
-- in the same transaction as the archive update; the delete trigger calls it
-- for one row. One call bumps the epoch once and redacts each batch once.
CREATE FUNCTION fence_forgotten_memories(p_side TEXT, p_workspace_id TEXT, p_ids TEXT[])
RETURNS VOID LANGUAGE plpgsql AS $$
DECLARE b RECORD; src JSONB; i INTEGER; indices INTEGER[];
        facts JSONB; fact JSONB; affected BOOLEAN; untracked BOOLEAN; fact_no INTEGER;
BEGIN
 IF p_side NOT IN ('user','ai') THEN RAISE EXCEPTION 'invalid memory side'; END IF;
 IF p_workspace_id IS NULL OR coalesce(cardinality(p_ids),0)=0
    OR NOT EXISTS(SELECT 1 FROM chat_workspaces WHERE id=p_workspace_id) THEN RETURN; END IF;
 INSERT INTO memory_erasure_epochs(workspace_id,generation) VALUES(p_workspace_id,1)
 ON CONFLICT(workspace_id) DO UPDATE SET generation=memory_erasure_epochs.generation+1;
 INSERT INTO memory_erased_sources(workspace_id,side,message_id,source_version)
 SELECT p_workspace_id,p_side,e.source_message_id,e.source_version
 FROM memory_evidence_links e JOIN messages m ON m.id=e.source_message_id
 WHERE e.memory_id=ANY(p_ids) AND e.memory_source=p_side AND e.source_version IS NOT NULL
   AND COALESCE(e.temporal_basis->>'source_role','fact')='fact'
 ON CONFLICT DO NOTHING;
 -- A memory written before receipts existed is redacted by source indices.
 untracked := EXISTS(SELECT 1 FROM unnest(p_ids) AS t(id) WHERE NOT EXISTS(
   SELECT 1 FROM memory_write_receipts WHERE memory_id=t.id AND batch_id IN (
     SELECT id FROM memory_extraction_batches WHERE side=p_side AND workspace_id=p_workspace_id)));
 UPDATE memory_write_receipts SET status='erased',memory_id=NULL,memory_version=NULL,followup_done=true,followup_outcome='erased'
 WHERE memory_id=ANY(p_ids) AND batch_id IN (SELECT id FROM memory_extraction_batches WHERE side=p_side AND workspace_id=p_workspace_id);
 FOR b IN SELECT * FROM memory_extraction_batches WHERE workspace_id=p_workspace_id AND side=p_side
   AND extraction IS NOT NULL AND (EXISTS(SELECT 1 FROM memory_write_receipts r WHERE r.batch_id=memory_extraction_batches.id AND r.status='erased') OR EXISTS(SELECT 1 FROM memory_extraction_sources s
      JOIN memory_erased_sources e ON e.message_id=s.message_id AND e.side=s.side
      WHERE s.batch_id=memory_extraction_batches.id AND e.workspace_id=p_workspace_id)) FOR UPDATE
 LOOP
  indices := ARRAY[]::INTEGER[]; i:=0;
  FOR src IN SELECT value FROM jsonb_array_elements(b.sources) LOOP
   i:=i+1;
   IF EXISTS(SELECT 1 FROM memory_erased_sources WHERE workspace_id=p_workspace_id AND side=p_side
      AND message_id=src->>'ref' AND source_version=src->>'expected_version') THEN indices:=array_append(indices,i); END IF;
  END LOOP;
  facts:='[]'::JSONB; affected:=false; fact_no:=-1;
  FOR fact IN SELECT value FROM jsonb_array_elements(b.extraction->'memories') LOOP
   fact_no:=fact_no+1;
   IF EXISTS(SELECT 1 FROM memory_write_receipts WHERE batch_id=b.id AND fact_index=fact_no AND status='erased')
      OR (untracked AND (jsonb_typeof(fact->'source_indices') IS DISTINCT FROM 'array'
      OR jsonb_array_length(fact->'source_indices')=0
      OR EXISTS(SELECT 1 FROM jsonb_array_elements(fact->'source_indices') v WHERE v.value=ANY(SELECT to_jsonb(n) FROM unnest(indices) n))))
   THEN facts:=facts||jsonb_build_array(jsonb_build_object('_erased',true)); affected:=true;
   ELSE facts:=facts||jsonb_build_array(fact); END IF;
  END LOOP;
  IF affected THEN UPDATE memory_extraction_batches SET extraction=jsonb_build_object('plan_version',b.extraction->'plan_version','memories',facts,'entities','[]'::JSONB,'preferences','[]'::JSONB,'topics','[]'::JSONB),updated_at=CURRENT_TIMESTAMP WHERE id=b.id; END IF;
 END LOOP;
END $$;

CREATE OR REPLACE FUNCTION fence_deleted_memory_sources() RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
 PERFORM fence_forgotten_memories(
   CASE TG_TABLE_NAME WHEN 'memories_user' THEN 'user' ELSE 'ai' END, OLD.workspace_id, ARRAY[OLD.id]);
 RETURN OLD;
END $$;
