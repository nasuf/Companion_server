-- No historical preference inference or backfill. Source deletion removes its
-- current policy rather than reviving an older instruction from event history.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
CREATE TABLE conversation_topics (
 id TEXT PRIMARY KEY,
 user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 agent_id TEXT NOT NULL REFERENCES ai_agents(id) ON DELETE CASCADE,
 workspace_id TEXT NOT NULL REFERENCES chat_workspaces(id) ON DELETE CASCADE,
 workspace_generation TEXT NOT NULL,
 agent_generation TEXT NOT NULL,
 owner_generation TEXT NOT NULL,
 topic_key TEXT NOT NULL CHECK(length(topic_key) BETWEEN 1 AND 100),
 label TEXT NOT NULL CHECK(length(label)<=64),
 scope TEXT NOT NULL CHECK(scope IN ('session','topic','global')),
 state TEXT NOT NULL CHECK(state IN ('open','awaiting_user','paused','closed')),
 action TEXT NOT NULL CHECK(action IN ('pause','stop','resume')),
 expires_at TIMESTAMPTZ,
 source_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
 source_version TEXT NOT NULL CHECK(length(source_version)=64),
 source_statement_at TIMESTAMPTZ NOT NULL,
 decision_at TIMESTAMPTZ NOT NULL,
 source_ordinal INTEGER NOT NULL CHECK(source_ordinal BETWEEN 0 AND 12000),
 revision INTEGER NOT NULL DEFAULT 1 CHECK(revision>0),
 policy_version TEXT NOT NULL,
 last_user_mention TIMESTAMPTZ,
 last_agent_question TIMESTAMPTZ,
 answered_at TIMESTAMPTZ,
 asked_count INTEGER NOT NULL DEFAULT 0 CHECK(asked_count>=0),
 created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
 updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
 UNIQUE(workspace_id,workspace_generation,topic_key)
);
CREATE INDEX conversation_topics_scope_idx ON conversation_topics(workspace_id,workspace_generation,updated_at DESC,id);
CREATE INDEX conversation_topics_user_idx ON conversation_topics(user_id);
CREATE INDEX conversation_topics_source_idx ON conversation_topics(source_message_id);
-- A payload-free watermark survives deletion/retention of the projection.
-- Without it an old job could recreate a preference after FK cascades remove
-- both the latest policy and its per-message audit/idempotency rows.
CREATE TABLE conversation_topic_fences (
 topic_id TEXT PRIMARY KEY,
 workspace_id TEXT NOT NULL REFERENCES chat_workspaces(id) ON DELETE CASCADE,
 decision_at TIMESTAMPTZ NOT NULL,
 deleted BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX conversation_topic_fences_workspace_idx ON conversation_topic_fences(workspace_id);
CREATE FUNCTION fence_conversation_topic() RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
 IF TG_OP='DELETE' THEN
   UPDATE conversation_topic_fences SET deleted=TRUE WHERE topic_id=OLD.id;
   RETURN OLD;
 END IF;
 INSERT INTO conversation_topic_fences(topic_id,workspace_id,decision_at,deleted)
 VALUES (NEW.id,NEW.workspace_id,NEW.decision_at,FALSE)
 ON CONFLICT (topic_id) DO UPDATE SET decision_at=GREATEST(conversation_topic_fences.decision_at,EXCLUDED.decision_at),deleted=FALSE;
 RETURN NEW;
END $$;
CREATE TRIGGER conversation_topic_fence AFTER INSERT OR UPDATE OR DELETE ON conversation_topics
 FOR EACH ROW EXECUTE FUNCTION fence_conversation_topic();
CREATE TABLE conversation_topic_events (
 id TEXT PRIMARY KEY,
 topic_id TEXT NOT NULL REFERENCES conversation_topics(id) ON DELETE CASCADE,
 user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 source_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
 actor_user_id TEXT REFERENCES users(id) ON DELETE SET NULL,
 source_type TEXT NOT NULL DEFAULT 'user' CHECK(source_type IN ('user','admin')),
 action TEXT NOT NULL CHECK(action IN ('pause','stop','resume')),
 expires_at TIMESTAMPTZ,
 source_version TEXT NOT NULL CHECK(length(source_version)=64),
 statement_at TIMESTAMPTZ NOT NULL,
 ordinal INTEGER NOT NULL,
 reason TEXT NOT NULL CHECK(length(reason)<=500),
 policy_version TEXT NOT NULL,
 recorded_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX conversation_topic_events_topic_idx ON conversation_topic_events(topic_id,recorded_at DESC,id);
CREATE INDEX conversation_topic_events_user_idx ON conversation_topic_events(user_id);
CREATE INDEX conversation_topic_events_source_idx ON conversation_topic_events(source_message_id);
CREATE INDEX conversation_topic_events_actor_idx ON conversation_topic_events(actor_user_id);
CREATE FUNCTION validate_conversation_topic() RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
 -- Scope writes and capacity admission share one transaction lock. The limit
 -- includes deleted source-free tombstones; removing one must never allow an
 -- old worker to recreate a forgotten preference. No automatic preference eviction.
 IF TG_OP='INSERT' THEN
  PERFORM pg_advisory_xact_lock(hashtextextended('topic-policy:'||NEW.workspace_id,0));
  IF NOT EXISTS(SELECT 1 FROM conversation_topic_fences WHERE topic_id=NEW.id)
   AND EXISTS(SELECT 1 FROM conversation_topic_fences WHERE workspace_id=NEW.workspace_id
      ORDER BY topic_id OFFSET 4095 LIMIT 1) THEN
   RAISE EXCEPTION 'topic_identity_capacity_exhausted' USING ERRCODE='23514';
  END IF;
 END IF;
 IF TG_OP='UPDATE' AND (NEW.user_id,NEW.agent_id,NEW.workspace_id,NEW.workspace_generation,NEW.agent_generation,NEW.owner_generation,NEW.scope,NEW.label,NEW.topic_key,NEW.action,NEW.expires_at,NEW.source_message_id,NEW.source_version,NEW.source_statement_at,NEW.decision_at,NEW.revision) IS NOT DISTINCT FROM (OLD.user_id,OLD.agent_id,OLD.workspace_id,OLD.workspace_generation,OLD.agent_generation,OLD.owner_generation,OLD.scope,OLD.label,OLD.topic_key,OLD.action,OLD.expires_at,OLD.source_message_id,OLD.source_version,OLD.source_statement_at,OLD.decision_at,OLD.revision) THEN RETURN NEW; END IF;
 IF NOT EXISTS (
  SELECT 1 FROM messages m JOIN conversations c ON c.id=m.conversation_id
  JOIN users u ON u.id=c.user_id JOIN chat_workspaces w ON w.id=c.workspace_id
  JOIN ai_agents a ON a.id=c.agent_id
  WHERE m.id=NEW.source_message_id AND m.role='user'
   AND encode(sha256(convert_to(m.content,'UTF8')),'hex')=NEW.source_version
   AND m.created_at AT TIME ZONE 'UTC'=NEW.source_statement_at
   AND COALESCE(m.metadata->>'human_takeover','false')<>'true'
   AND NOT COALESCE(m.metadata,'{}'::jsonb) ? 'component_card'
   AND c.user_id=NEW.user_id AND c.workspace_id=NEW.workspace_id AND c.agent_id=NEW.agent_id
   AND w.user_id=u.id AND w.agent_id=a.id AND a.user_id=u.id
   AND u.execution_generation=NEW.owner_generation AND w.execution_generation=NEW.workspace_generation
   AND a.execution_generation=NEW.agent_generation
   AND u.status='active' AND u.archived_at IS NULL AND a.status='active' AND a.archived_at IS NULL
   AND w.status='active' AND w.archived_at IS NULL AND NOT c.is_deleted AND c.archived_at IS NULL
  FOR SHARE OF m,c,u,w,a
 ) THEN RAISE EXCEPTION 'topic_scope_or_source_unavailable' USING ERRCODE='23514'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER conversation_topic_validate BEFORE INSERT OR UPDATE ON conversation_topics
 FOR EACH ROW EXECUTE FUNCTION validate_conversation_topic();
CREATE TABLE conversation_topic_observations (
 topic_id TEXT NOT NULL REFERENCES conversation_topics(id) ON DELETE CASCADE,
 source_message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
 user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 kind TEXT NOT NULL CHECK(kind IN ('question','mention')),
 statement_at TIMESTAMPTZ NOT NULL,
 source_version TEXT NOT NULL,
 PRIMARY KEY(topic_id,source_message_id)
);
CREATE INDEX conversation_topic_observations_source_idx ON conversation_topic_observations(source_message_id);
CREATE INDEX conversation_topic_observations_user_idx ON conversation_topic_observations(user_id);
CREATE FUNCTION remove_topic_observation() RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
 UPDATE conversation_topics SET
   asked_count=GREATEST(0,asked_count-CASE WHEN OLD.kind='question' THEN 1 ELSE 0 END),
   last_agent_question=CASE WHEN OLD.kind='question' AND last_agent_question=OLD.statement_at THEN NULL ELSE last_agent_question END,
   last_user_mention=CASE WHEN OLD.kind='mention' AND last_user_mention=OLD.statement_at THEN NULL ELSE last_user_mention END,
   answered_at=CASE WHEN OLD.kind='mention' AND answered_at=OLD.statement_at THEN NULL ELSE answered_at END
 WHERE id=OLD.topic_id;
 RETURN OLD;
END $$;
CREATE TRIGGER topic_observation_remove AFTER DELETE ON conversation_topic_observations
 FOR EACH ROW EXECUTE FUNCTION remove_topic_observation();
CREATE FUNCTION invalidate_topic_observations() RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
 IF (NEW.content,NEW.created_at,NEW.role,NEW.metadata->'human_takeover',NEW.metadata->'component_card') IS DISTINCT FROM (OLD.content,OLD.created_at,OLD.role,OLD.metadata->'human_takeover',OLD.metadata->'component_card') THEN
   DELETE FROM conversation_topic_observations WHERE source_message_id=OLD.id;
 END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER topic_observations_invalidate AFTER UPDATE ON messages
 FOR EACH ROW EXECUTE FUNCTION invalidate_topic_observations();
CREATE FUNCTION validate_topic_event() RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
 IF TG_OP='UPDATE' THEN
   IF OLD.actor_user_id IS NOT NULL AND NEW.actor_user_id IS NULL
      AND (to_jsonb(OLD)-'actor_user_id') IS NOT DISTINCT FROM (to_jsonb(NEW)-'actor_user_id')
      AND (current_setting('app.account_erasure_user',true)=OLD.actor_user_id OR NOT EXISTS(SELECT 1 FROM users WHERE id=OLD.actor_user_id)) THEN RETURN NEW; END IF;
   RAISE EXCEPTION 'topic_event_immutable' USING ERRCODE='23514';
 END IF;
 IF NOT EXISTS (SELECT 1 FROM conversation_topics t JOIN messages m ON m.id=NEW.source_message_id
   JOIN conversations c ON c.id=m.conversation_id
   WHERE t.id=NEW.topic_id AND t.user_id=NEW.user_id AND c.user_id=t.user_id
     AND c.workspace_id=t.workspace_id AND c.agent_id=t.agent_id AND NOT c.is_deleted
     AND c.archived_at IS NULL AND m.role='user'
     AND m.created_at AT TIME ZONE 'UTC'=NEW.statement_at
     AND encode(sha256(convert_to(m.content,'UTF8')),'hex')=NEW.source_version
     AND COALESCE(m.metadata->>'human_takeover','false')<>'true'
     AND NOT COALESCE(m.metadata,'{}'::jsonb) ? 'component_card'
     AND ((NEW.source_type='user' AND NEW.actor_user_id IS NULL) OR (NEW.source_type='admin' AND EXISTS (SELECT 1 FROM users u WHERE u.id=NEW.actor_user_id AND u.role='admin' AND u.status='active')))
   FOR SHARE OF t,m,c) THEN
   RAISE EXCEPTION 'topic_event_scope_or_source' USING ERRCODE='23514';
 END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER conversation_topic_event_validate BEFORE INSERT OR UPDATE ON conversation_topic_events
 FOR EACH ROW EXECUTE FUNCTION validate_topic_event();
CREATE FUNCTION validate_topic_observation() RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
 IF TG_OP='UPDATE' THEN RAISE EXCEPTION 'topic_observation_immutable' USING ERRCODE='23514'; END IF;
 IF NOT EXISTS (SELECT 1 FROM conversation_topics t JOIN messages m ON m.id=NEW.source_message_id
   JOIN conversations c ON c.id=m.conversation_id
   WHERE t.id=NEW.topic_id AND t.user_id=NEW.user_id AND c.user_id=t.user_id
     AND c.workspace_id=t.workspace_id AND c.agent_id=t.agent_id AND NOT c.is_deleted
     AND c.archived_at IS NULL AND m.role=CASE NEW.kind WHEN 'question' THEN 'assistant' ELSE 'user' END
     AND m.created_at AT TIME ZONE 'UTC'=NEW.statement_at
     AND encode(sha256(convert_to(m.content,'UTF8')),'hex')=NEW.source_version
     AND COALESCE(m.metadata->>'human_takeover','false')<>'true'
     AND NOT COALESCE(m.metadata,'{}'::jsonb) ? 'component_card'
   FOR SHARE OF t,m,c) THEN
   RAISE EXCEPTION 'topic_observation_scope_or_source' USING ERRCODE='23514';
 END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER conversation_topic_observation_validate BEFORE INSERT OR UPDATE ON conversation_topic_observations
 FOR EACH ROW EXECUTE FUNCTION validate_topic_observation();
COMMIT;
