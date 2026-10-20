-- NOT VALID preserves unrelated historical orphan rows. New writes are fenced,
-- and deleting an existing owner still cascades to all of that owner's rows.
ALTER TABLE "speech_usage" ADD CONSTRAINT "speech_usage_erasure_owner_fkey" FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE NOT VALID;
ALTER TABLE "tts_usage" ADD CONSTRAINT "tts_usage_erasure_owner_fkey" FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE NOT VALID;
ALTER TABLE "memories_user" ADD CONSTRAINT "memories_user_erasure_owner_fkey" FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE NOT VALID;
ALTER TABLE "memories_ai" ADD CONSTRAINT "memories_ai_erasure_owner_fkey" FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE NOT VALID;
ALTER TABLE "memory_repair_items" ADD CONSTRAINT "memory_repair_items_erasure_owner_fkey" FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE NOT VALID;
ALTER TABLE "memory_quality_states" ADD CONSTRAINT "memory_quality_states_erasure_owner_fkey" FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE NOT VALID;
ALTER TABLE "memory_consolidation_runs" ADD CONSTRAINT "memory_consolidation_runs_erasure_owner_fkey" FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE NOT VALID;
ALTER TABLE "memory_visible_use_events" ADD CONSTRAINT "memory_visible_use_events_erasure_owner_fkey" FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE NOT VALID;
ALTER TABLE "crisis_events" ADD CONSTRAINT "crisis_events_erasure_owner_fkey" FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE NOT VALID;
ALTER TABLE "proactive_chat_logs" ADD CONSTRAINT "proactive_chat_logs_erasure_owner_fkey" FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE NOT VALID;
ALTER TABLE "proactive_counters" ADD CONSTRAINT "proactive_counters_erasure_owner_fkey" FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE NOT VALID;
ALTER TABLE "memory_entities" ADD CONSTRAINT "memory_entities_erasure_owner_fkey" FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE NOT VALID;
ALTER TABLE "memory_mentions" ADD CONSTRAINT "memory_mentions_erasure_owner_fkey" FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE NOT VALID;
ALTER TABLE "llm_usage" ADD CONSTRAINT "llm_usage_erasure_owner_fkey" FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE NOT VALID;

-- A late background child must not recreate prompts after its root is erased.
ALTER TABLE trace_runs ADD CONSTRAINT trace_runs_erasure_root_fkey
  FOREIGN KEY (trace_id) REFERENCES trace_runs(run_id) ON DELETE CASCADE ON UPDATE CASCADE NOT VALID;

-- Chat roots use actual immutable conversation IDs. Pseudo IDs used by global
-- admin diagnostics stay outside this guard and are inventoried separately.
CREATE FUNCTION guard_trace_erasure_root() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF NEW.run_id=NEW.trace_id AND NEW.extra_json->'metadata'->>'owner_user_id' IS NOT NULL THEN
    PERFORM 1 FROM users WHERE id=NEW.extra_json->'metadata'->>'owner_user_id' AND status<>'deleting' FOR SHARE;
    IF NOT FOUND THEN RAISE EXCEPTION 'Trace owner unavailable' USING ERRCODE='23503'; END IF;
  END IF;
  IF NEW.run_id=NEW.trace_id AND NEW.name='chat_request'
     AND COALESCE(NEW.inputs_json->>'conversation_id','') ~ '^[0-9a-fA-F-]{32,36}$' THEN
    PERFORM 1 FROM conversations c JOIN users u ON u.id=c.user_id
      WHERE c.id=NEW.inputs_json->>'conversation_id' AND u.status='active'
      FOR SHARE OF c,u;
    IF NOT FOUND THEN RAISE EXCEPTION 'Trace owner unavailable' USING ERRCODE='23503'; END IF;
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER trace_erasure_root_guard BEFORE INSERT ON trace_runs
  FOR EACH ROW EXECUTE FUNCTION guard_trace_erasure_root();

-- Serialize new side-table writes against deletion's lifecycle barrier, without
-- holding an application/LLM lock. Nullable system-wide measurements stay valid.
CREATE FUNCTION guard_erasure_owner_active() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE binding record; resource text; bound text;
BEGIN
  IF TG_OP='UPDATE' AND OLD.user_id IS DISTINCT FROM NEW.user_id AND OLD.user_id IS NOT NULL THEN
    PERFORM 1 FROM users WHERE id=OLD.user_id AND status='deleting' FOR SHARE;
    IF FOUND THEN RAISE EXCEPTION 'Account unavailable' USING ERRCODE='23503'; END IF;
  END IF;
  IF NEW.user_id IS NOT NULL THEN
    PERFORM 1 FROM users WHERE id=NEW.user_id AND status<>'deleting' FOR SHARE;
    IF NOT FOUND THEN RAISE EXCEPTION 'Account unavailable' USING ERRCODE='23503'; END IF;
  ELSE
    -- Nullable system measurements may still contain a private resource ID.
    -- Real UUID/hex resources must exist; global pseudo-conversation labels
    -- remain valid, as they do for the Trace root guard above.
    FOR binding IN SELECT * FROM (VALUES ('conversation_id','conversations'),
        ('workspace_id','chat_workspaces'),('agent_id','ai_agents')) AS v(field_name,table_name) LOOP
      resource := to_jsonb(NEW)->>binding.field_name;
      IF resource ~ '^[0-9a-fA-F-]{32,36}$' THEN
        EXECUTE format('SELECT r.id FROM public.%I r JOIN users u ON u.id=r.user_id WHERE r.id=$1 AND u.status<>$2 FOR SHARE OF r,u',binding.table_name)
          INTO bound USING resource,'deleting';
        IF bound IS NULL THEN RAISE EXCEPTION 'Account resource unavailable' USING ERRCODE='23503'; END IF;
      END IF;
    END LOOP;
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER erasure_owner_active BEFORE INSERT OR UPDATE OF user_id ON "speech_usage" FOR EACH ROW EXECUTE FUNCTION guard_erasure_owner_active();
CREATE TRIGGER erasure_owner_active BEFORE INSERT OR UPDATE OF user_id ON "tts_usage" FOR EACH ROW EXECUTE FUNCTION guard_erasure_owner_active();
CREATE TRIGGER erasure_owner_active BEFORE INSERT OR UPDATE OF user_id ON "llm_usage" FOR EACH ROW EXECUTE FUNCTION guard_erasure_owner_active();
CREATE TRIGGER erasure_owner_active BEFORE INSERT OR UPDATE OF user_id ON "memories_user" FOR EACH ROW EXECUTE FUNCTION guard_erasure_owner_active();
CREATE TRIGGER erasure_owner_active BEFORE INSERT OR UPDATE OF user_id ON "memories_ai" FOR EACH ROW EXECUTE FUNCTION guard_erasure_owner_active();
CREATE TRIGGER erasure_owner_active BEFORE INSERT OR UPDATE OF user_id ON "memory_consolidation_runs" FOR EACH ROW EXECUTE FUNCTION guard_erasure_owner_active();
CREATE TRIGGER erasure_owner_active BEFORE INSERT OR UPDATE OF user_id ON "memory_entities" FOR EACH ROW EXECUTE FUNCTION guard_erasure_owner_active();
CREATE TRIGGER erasure_owner_active BEFORE INSERT OR UPDATE OF user_id ON "memory_mentions" FOR EACH ROW EXECUTE FUNCTION guard_erasure_owner_active();
CREATE TRIGGER erasure_owner_active BEFORE INSERT OR UPDATE OF user_id ON "memory_quality_states" FOR EACH ROW EXECUTE FUNCTION guard_erasure_owner_active();
CREATE TRIGGER erasure_owner_active BEFORE INSERT OR UPDATE OF user_id ON "memory_repair_items" FOR EACH ROW EXECUTE FUNCTION guard_erasure_owner_active();
CREATE TRIGGER erasure_owner_active BEFORE INSERT OR UPDATE OF user_id ON "memory_visible_use_events" FOR EACH ROW EXECUTE FUNCTION guard_erasure_owner_active();
CREATE TRIGGER erasure_owner_active BEFORE INSERT OR UPDATE OF user_id ON "crisis_events" FOR EACH ROW EXECUTE FUNCTION guard_erasure_owner_active();
CREATE TRIGGER erasure_owner_active BEFORE INSERT OR UPDATE OF user_id ON "proactive_chat_logs" FOR EACH ROW EXECUTE FUNCTION guard_erasure_owner_active();
CREATE TRIGGER erasure_owner_active BEFORE INSERT OR UPDATE OF user_id ON "proactive_counters" FOR EACH ROW EXECUTE FUNCTION guard_erasure_owner_active();

-- Mirrors are written asynchronously after a Trace viewer request.
ALTER TABLE message_traces ADD CONSTRAINT message_traces_erasure_conversation_fkey
  FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE ON UPDATE CASCADE NOT VALID;

-- Existing cascade tables must also refuse late inserts and owner rebindings.
DO $$ DECLARE entry record; guarded_columns text; BEGIN
  FOR entry IN SELECT c.table_name FROM information_schema.columns c
      JOIN information_schema.tables t ON t.table_schema=c.table_schema AND t.table_name=c.table_name
      WHERE c.table_schema='public' AND c.column_name='user_id' AND t.table_type='BASE TABLE' LOOP
    SELECT string_agg(format('%I',column_name),',' ORDER BY column_name) INTO guarded_columns
      FROM information_schema.columns WHERE table_schema='public' AND table_name=entry.table_name
        AND column_name IN ('user_id','conversation_id','workspace_id','agent_id');
    EXECUTE format('DROP TRIGGER IF EXISTS erasure_owner_active ON public.%I',entry.table_name);
    EXECUTE format('CREATE TRIGGER erasure_owner_active BEFORE INSERT OR UPDATE OF %s ON public.%I FOR EACH ROW EXECUTE FUNCTION guard_erasure_owner_active()',guarded_columns,entry.table_name);
  END LOOP;
END $$;

-- Embeddings are polymorphic across the two memory tables, so a conventional
-- foreign key cannot enforce this owner/resource barrier.
CREATE FUNCTION guard_erasure_embedding_owner() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  PERFORM m.id FROM memories_user m JOIN users u ON u.id=m.user_id
    WHERE m.id=NEW.memory_id AND u.status<>'deleting' FOR SHARE OF m,u;
  IF NOT FOUND THEN
    PERFORM m.id FROM memories_ai m JOIN users u ON u.id=m.user_id
      WHERE m.id=NEW.memory_id AND u.status<>'deleting' FOR SHARE OF m,u;
    IF NOT FOUND THEN RAISE EXCEPTION 'Memory owner unavailable' USING ERRCODE='23503'; END IF;
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER erasure_embedding_owner BEFORE INSERT OR UPDATE ON memory_embeddings
  FOR EACH ROW EXECUTE FUNCTION guard_erasure_embedding_owner();
