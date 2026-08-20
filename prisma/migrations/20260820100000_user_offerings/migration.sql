CREATE TABLE IF NOT EXISTS user_offerings (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    conversation_id TEXT,
    message_id TEXT,
    kind TEXT NOT NULL,
    ticket_amount INTEGER NOT NULL,
    agent_value_yuan INTEGER NOT NULL,
    status TEXT NOT NULL,
    blessing TEXT,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    received_at TIMESTAMP(3),

    CONSTRAINT user_offerings_user_id_fkey
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT user_offerings_agent_id_fkey
        FOREIGN KEY (agent_id) REFERENCES ai_agents(id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT user_offerings_conversation_id_fkey
        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE SET NULL ON UPDATE CASCADE,
    CONSTRAINT user_offerings_kind_check
        CHECK (kind IN ('red_packet', 'gift')),
    CONSTRAINT user_offerings_status_check
        CHECK (status IN ('sent', 'received')),
    CONSTRAINT user_offerings_ticket_amount_positive
        CHECK (ticket_amount > 0),
    CONSTRAINT user_offerings_agent_value_positive
        CHECK (agent_value_yuan > 0)
);

CREATE INDEX IF NOT EXISTS user_offerings_user_agent_time_idx
    ON user_offerings(user_id, agent_id, created_at DESC);

CREATE INDEX IF NOT EXISTS user_offerings_conversation_idx
    ON user_offerings(conversation_id);

CREATE UNIQUE INDEX IF NOT EXISTS user_offerings_message_unique
    ON user_offerings(message_id);

CREATE TABLE IF NOT EXISTS agent_wallets (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    agent_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    received_tickets INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT agent_wallets_agent_id_key UNIQUE (agent_id),
    CONSTRAINT agent_wallets_agent_id_fkey
        FOREIGN KEY (agent_id) REFERENCES ai_agents(id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT agent_wallets_user_id_fkey
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT agent_wallets_received_tickets_nonnegative
        CHECK (received_tickets >= 0)
);

CREATE INDEX IF NOT EXISTS agent_wallets_user_idx
    ON agent_wallets(user_id);
