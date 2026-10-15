-- New location shares only. Preserve old unknown origins and residence rows.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
ALTER TABLE memory_input_origins DROP CONSTRAINT memory_input_origins_source_status_check;
ALTER TABLE memory_input_origins ADD CONSTRAINT memory_input_origins_source_status_check
 CHECK(source_status IN ('knowledge_document','admin_edit','user_edit','admin_repair','contradiction_confirmation','daily_summary','consolidation','location_share'));

CREATE TABLE location_memory_events (
 message_id TEXT PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE ON UPDATE RESTRICT,
 scope JSONB NOT NULL,
 source_version TEXT NOT NULL CHECK(source_version ~ '^[a-f0-9]{64}$'),
 status TEXT NOT NULL DEFAULT 'pending_dispatch' CHECK(status IN ('pending_dispatch','queued','completed','cancelled')),
 job_id TEXT,
 next_dispatch_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 dispatch_attempts INT NOT NULL DEFAULT 0 CHECK(dispatch_attempts>=0),
 last_error TEXT,
 memory_ids JSONB NOT NULL DEFAULT '[]' CHECK(jsonb_typeof(memory_ids)='array' AND jsonb_array_length(memory_ids)<=2),
 created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
 updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX location_memory_dispatch_idx ON location_memory_events(next_dispatch_at,message_id)
 WHERE status='pending_dispatch';
CREATE INDEX location_memory_recent_idx ON location_memory_events(created_at DESC,message_id DESC);

CREATE FUNCTION location_message_version(content TEXT, metadata JSONB, created_at TIMESTAMP)
 RETURNS TEXT LANGUAGE SQL IMMUTABLE STRICT AS $$
 SELECT encode(sha256(convert_to(jsonb_build_object('content',content,
 'component_card',metadata->'component_card','accepted_at',created_at)::text,'UTF8')),'hex')
$$;
CREATE FUNCTION record_location_memory_event() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF NEW.role='user' AND NEW.metadata->'component_card'->>'type'='location'
 AND COALESCE(NEW.metadata->>'human_takeover','false')<>'true'
 AND EXISTS(SELECT 1 FROM conversations c JOIN chat_workspaces w ON w.id=c.workspace_id
   AND w.user_id=c.user_id AND w.agent_id=c.agent_id
   WHERE c.id=NEW.conversation_id AND NOT c.is_deleted
   AND NOT EXISTS(SELECT 1 FROM human_takeovers h WHERE h.user_id=c.user_id AND h.status='active')) THEN
   INSERT INTO location_memory_events(message_id,source_version,scope)
     SELECT NEW.id,location_message_version(NEW.content,NEW.metadata,NEW.created_at),
       jsonb_build_object('user_id',c.user_id,'workspace_id',c.workspace_id,'agent_id',c.agent_id,'conversation_id',c.id)
     FROM conversations c WHERE c.id=NEW.conversation_id;
 END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER message_location_memory_event AFTER INSERT ON messages
 FOR EACH ROW EXECUTE FUNCTION record_location_memory_event();

ALTER TABLE memory_input_origins ADD CONSTRAINT memory_location_input_check CHECK (
 source_status<>'location_share' OR (
 payload_text::jsonb->'input'->>'format'='location-share-v1'
 AND length(payload_text::jsonb->'input'->>'message_id') BETWEEN 1 AND 200
 AND payload_text::jsonb->'input'->>'message_version' ~ '^[a-f0-9]{64}$'
 AND payload_text::jsonb->'input'->'component_card'->>'type'='location'
 AND jsonb_typeof(payload_text::jsonb->'input'->'component_card'->'payload')='object'
 AND octet_length(payload_text)<=32768
 ) IS TRUE);
CREATE UNIQUE INDEX memory_location_origin_message_key ON memory_input_origins
 ((payload_text::jsonb->'input'->>'message_id')) WHERE source_status='location_share';

CREATE INDEX messages_location_client_idx ON messages(conversation_id,(metadata->>'client_id'))
 WHERE role='user' AND metadata->'component_card'->>'type'='location';
CREATE FUNCTION preserve_location_event_identity() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF NEW.message_id IS DISTINCT FROM OLD.message_id OR NEW.source_version IS DISTINCT FROM OLD.source_version
 OR NEW.scope IS DISTINCT FROM OLD.scope OR NEW.created_at IS DISTINCT FROM OLD.created_at
 OR (OLD.status IN ('completed','cancelled') AND NEW IS DISTINCT FROM OLD) THEN
   RAISE EXCEPTION 'location_event_identity_immutable' USING ERRCODE='23514';
 END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER location_event_identity BEFORE UPDATE ON location_memory_events
 FOR EACH ROW EXECUTE FUNCTION preserve_location_event_identity();

CREATE FUNCTION validate_location_input() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE data JSONB;
BEGIN
 IF NEW.source_status='location_share' THEN
   data=NEW.payload_text::jsonb->'input';
   PERFORM 1 FROM messages m JOIN conversations c ON c.id=m.conversation_id
     JOIN location_memory_events e ON e.message_id=m.id
     WHERE m.id=data->>'message_id' AND m.role='user' AND NOT c.is_deleted
     AND COALESCE(m.metadata->>'human_takeover','false')<>'true'
     AND c.user_id=NEW.user_id AND c.workspace_id=NEW.workspace_id AND c.agent_id=NEW.agent_id
     AND e.scope=jsonb_build_object('user_id',c.user_id,'workspace_id',c.workspace_id,'agent_id',c.agent_id,'conversation_id',c.id)
     AND e.status IN ('pending_dispatch','queued') AND e.source_version=data->>'message_version'
     AND e.source_version=location_message_version(m.content,m.metadata,m.created_at)
     AND m.metadata->'component_card'=data->'component_card' AND m.content=data->>'content'
     AND c.id=data->>'conversation_id' AND m.created_at=(data->>'accepted_at')::timestamptz AT TIME ZONE 'UTC'
     FOR SHARE OF m,c,e;
   IF NOT FOUND THEN RAISE EXCEPTION 'location_input_source_mismatch' USING ERRCODE='23514'; END IF;
 END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER memory_location_input_validate BEFORE INSERT ON memory_input_origins
 FOR EACH ROW EXECUTE FUNCTION validate_location_input();
COMMIT;
