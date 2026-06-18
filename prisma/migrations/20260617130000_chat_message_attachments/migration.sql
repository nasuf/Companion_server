CREATE TABLE IF NOT EXISTS chat_message_attachments (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    message_id TEXT REFERENCES messages(id) ON DELETE CASCADE,
    kind TEXT NOT NULL DEFAULT 'image',
    name TEXT,
    mime TEXT NOT NULL,
    size INTEGER NOT NULL,
    width INTEGER,
    height INTEGER,
    storage_key TEXT NOT NULL UNIQUE,
    url TEXT NOT NULL,
    vision_status TEXT NOT NULL DEFAULT 'pending',
    vision_summary TEXT,
    vision_error TEXT,
    created_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS chat_message_attachments_conversation_created_idx
    ON chat_message_attachments(conversation_id, created_at DESC);

CREATE INDEX IF NOT EXISTS chat_message_attachments_message_idx
    ON chat_message_attachments(message_id);

CREATE INDEX IF NOT EXISTS chat_message_attachments_user_message_created_idx
    ON chat_message_attachments(user_id, message_id, created_at);
