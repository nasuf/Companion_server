-- No historical backfill. Registration commits or rolls back with the message.
CREATE TABLE message_followup_events (
  message_id TEXT PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE ON UPDATE RESTRICT,
  scope JSONB NOT NULL,
  source_version TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending_dispatch'
    CHECK (status IN ('pending_dispatch','queued','failed','completed','cancelled')),
  generation INTEGER NOT NULL DEFAULT 0 CHECK (generation >= 0),
  job_id TEXT,
  dispatch_attempts INTEGER NOT NULL DEFAULT 0 CHECK (dispatch_attempts >= 0),
  last_error TEXT,
  outcome TEXT,
  next_dispatch_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX message_followup_dispatch_idx ON message_followup_events(next_dispatch_at,message_id)
  WHERE status IN ('pending_dispatch','queued');
CREATE INDEX message_followup_recent_idx ON message_followup_events(created_at DESC,message_id DESC);
CREATE INDEX message_followup_status_recent_idx ON message_followup_events(status,created_at DESC,message_id DESC);
CREATE INDEX message_followup_retention_idx ON message_followup_events(updated_at,message_id)
  WHERE status IN ('completed','cancelled');

-- Ignore subsequent audio binding; retain the fields used by message rules.
CREATE FUNCTION message_followup_version(body TEXT, role TEXT, metadata JSONB, happened TIMESTAMP)
RETURNS TEXT LANGUAGE SQL IMMUTABLE AS $$
 SELECT encode(sha256(convert_to(jsonb_build_array(body,role,happened,
   metadata->'component_card',metadata->'human_takeover',metadata->'proactive',
   metadata->'trigger_type',metadata->'achievement_turn_id',metadata->'sticker_url',metadata->'red_packet',metadata->'gift')::text,'UTF8')),'hex');
$$;
CREATE FUNCTION register_message_followup() RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
 IF NEW.role IN ('user','assistant') THEN
   INSERT INTO message_followup_events(message_id,scope,source_version)
   SELECT NEW.id,jsonb_build_object('user_id',c.user_id,'agent_id',c.agent_id,
     'workspace_id',c.workspace_id,'conversation_id',c.id,'role',NEW.role),
     message_followup_version(NEW.content,NEW.role,NEW.metadata,NEW.created_at)
   FROM conversations c JOIN ai_agents a ON a.id=c.agent_id AND a.user_id=c.user_id
   LEFT JOIN chat_workspaces w ON w.id=c.workspace_id
   WHERE c.id=NEW.conversation_id AND NOT c.is_deleted
     AND (c.workspace_id IS NULL OR (w.status='active' AND w.user_id=c.user_id AND w.agent_id=c.agent_id))
   ON CONFLICT DO NOTHING;
 END IF;
 RETURN NEW;
END;
$$;
CREATE TRIGGER message_followup_register AFTER INSERT ON messages
 FOR EACH ROW EXECUTE FUNCTION register_message_followup();
