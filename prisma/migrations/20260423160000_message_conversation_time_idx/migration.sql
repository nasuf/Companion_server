-- Add composite index on messages(conversation_id, created_at DESC) to
-- support _fetch_intent_context hot path (orchestrator D4): every user
-- message pulls the recent N messages for the same conversation ordered
-- by created_at desc.
CREATE INDEX IF NOT EXISTS "messages_conversation_id_created_at_idx"
ON "messages" ("conversation_id", "created_at" DESC);
