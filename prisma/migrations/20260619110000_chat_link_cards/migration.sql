CREATE TABLE IF NOT EXISTS chat_link_cards (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    message_id TEXT REFERENCES messages(id) ON DELETE CASCADE,
    role TEXT NOT NULL DEFAULT 'user',
    source_app TEXT,
    source_url TEXT NOT NULL,
    final_url TEXT NOT NULL,
    platform TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    author TEXT,
    image_url TEXT,
    content_text TEXT NOT NULL DEFAULT '',
    original_text TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'ready',
    error TEXT,
    metadata JSONB,
    created_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS chat_link_cards_conversation_created_idx
    ON chat_link_cards(conversation_id, created_at DESC);

CREATE INDEX IF NOT EXISTS chat_link_cards_user_created_idx
    ON chat_link_cards(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS chat_link_cards_message_idx
    ON chat_link_cards(message_id);

CREATE INDEX IF NOT EXISTS chat_link_cards_platform_created_idx
    ON chat_link_cards(platform, created_at DESC);

CREATE UNIQUE INDEX IF NOT EXISTS chat_link_cards_user_conv_final_role_unique
    ON chat_link_cards(user_id, conversation_id, final_url, role);
