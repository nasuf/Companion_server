-- Digest cascade follows only originals that are still archived. After an
-- undo the originals are live again while the consolidated_into audit stays,
-- so deleting the old digest must not forget them.
CREATE OR REPLACE FUNCTION fence_forgotten_memories(p_side TEXT, p_workspace_id TEXT, p_ids TEXT[])
RETURNS VOID LANGUAGE plpgsql AS $$
DECLARE b RECORD; requested TEXT[]; src JSONB; i INTEGER; indices INTEGER[];
        facts JSONB; fact JSONB; affected BOOLEAN; untracked BOOLEAN; fact_no INTEGER;
BEGIN
 IF p_side NOT IN ('user','ai') THEN RAISE EXCEPTION 'invalid memory side'; END IF;
 IF p_workspace_id IS NULL OR coalesce(cardinality(p_ids),0)=0
    OR NOT EXISTS(SELECT 1 FROM chat_workspaces WHERE id=p_workspace_id) THEN RETURN; END IF;
 -- Originals still archived into a forgotten digest are forgotten with it
 -- (an undone consolidation leaves them live and untouched): their
 -- sources are fenced, their text is scrubbed and their vectors dropped. The
 -- archived rows themselves stay, like any archive-forgotten memory.
 requested := p_ids;
 WITH RECURSIVE tree(id) AS (
   SELECT unnest(p_ids)
   UNION SELECT c.memory_id FROM memory_changelogs c JOIN tree t ON c.new_value=t.id
   WHERE c.operation='consolidated_into' AND c.workspace_id=p_workspace_id
     AND (EXISTS(SELECT 1 FROM memories_user m WHERE p_side='user' AND m.id=c.memory_id AND m.is_archived)
       OR EXISTS(SELECT 1 FROM memories_ai m WHERE p_side='ai' AND m.id=c.memory_id AND m.is_archived)))
 SELECT array_agg(id) INTO p_ids FROM tree;
 DELETE FROM memory_embeddings WHERE memory_id=ANY(p_ids) AND NOT memory_id=ANY(requested);
 -- Usage receipts only link events to memories; nothing should keep pointing
 -- at a forgotten one.
 DELETE FROM memory_usage_receipts WHERE memory_side=p_side AND memory_id=ANY(p_ids);
 INSERT INTO memory_erasure_epochs(workspace_id,generation) VALUES(p_workspace_id,1)
 ON CONFLICT(workspace_id) DO UPDATE SET generation=memory_erasure_epochs.generation+1;
 INSERT INTO memory_erased_sources(workspace_id,side,message_id,source_version)
 SELECT p_workspace_id,p_side,e.source_message_id,e.source_version
 FROM memory_evidence_links e JOIN messages m ON m.id=e.source_message_id
 WHERE e.memory_id=ANY(p_ids) AND e.memory_source=p_side AND e.source_version IS NOT NULL
   AND COALESCE(e.temporal_basis->>'source_role','fact')='fact'
 ON CONFLICT DO NOTHING;
 -- Forgotten text is copied once into a short-lived queue; the scrub job
 -- removes it from traces, message metadata and pending states, then deletes
 -- the queue row. Nothing else keeps the text after forgetting.
 IF p_side='user' THEN
  INSERT INTO memory_forget_scrubs(workspace_id,side,memory_id,content)
  SELECT p_workspace_id,'user',id,content FROM memories_user WHERE id=ANY(p_ids) AND content IS NOT NULL
  ON CONFLICT(workspace_id,side,memory_id) DO NOTHING;
 ELSE
  INSERT INTO memory_forget_scrubs(workspace_id,side,memory_id,content)
  SELECT p_workspace_id,'ai',id,content FROM memories_ai WHERE id=ANY(p_ids) AND content IS NOT NULL
  ON CONFLICT(workspace_id,side,memory_id) DO NOTHING;
 END IF;
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

