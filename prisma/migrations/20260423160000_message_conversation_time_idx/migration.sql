-- Composite index on messages(conversation_id, created_at DESC) for
-- _fetch_intent_context: every user message pulls the recent N messages
-- for the same conversation ordered by created_at desc.
--
-- Production rollout note: on a large messages table this statement takes
-- an ACCESS EXCLUSIVE lock during index build. For zero-downtime deploy,
-- run the CONCURRENTLY variant out-of-band before applying the migration:
--   CREATE INDEX CONCURRENTLY "messages_conversation_id_created_at_idx"
--     ON "messages" ("conversation_id", "created_at" DESC);
-- The IF NOT EXISTS below then no-ops on prod.
CREATE INDEX IF NOT EXISTS "messages_conversation_id_created_at_idx"
ON "messages" ("conversation_id", "created_at" DESC);
