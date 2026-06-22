CREATE TABLE IF NOT EXISTS offline_activity_recommendations (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    agent_id TEXT NOT NULL REFERENCES ai_agents(id) ON DELETE CASCADE,
    workspace_id TEXT REFERENCES chat_workspaces(id) ON DELETE SET NULL,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    source TEXT NOT NULL DEFAULT 'scheduled',
    title TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    category TEXT,
    city TEXT,
    location_name TEXT,
    address TEXT,
    starts_at TIMESTAMPTZ,
    ends_at TIMESTAMPTZ,
    official_url TEXT,
    image_urls JSONB NOT NULL DEFAULT '[]'::jsonb,
    search_sources JSONB NOT NULL DEFAULT '[]'::jsonb,
    easter_egg_task JSONB,
    task_hint TEXT,
    accepted_at TIMESTAMPTZ,
    ignored_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS offline_activity_feedback (
    id TEXT PRIMARY KEY,
    recommendation_id TEXT NOT NULL REFERENCES offline_activity_recommendations(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    text TEXT NOT NULL DEFAULT '',
    photo_attachment_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS gift_addresses (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    recipient_name TEXT NOT NULL,
    phone TEXT NOT NULL,
    province TEXT NOT NULL DEFAULT '',
    city TEXT NOT NULL,
    district TEXT NOT NULL DEFAULT '',
    detail TEXT NOT NULL,
    is_default BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS real_world_gifts (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    agent_id TEXT NOT NULL REFERENCES ai_agents(id) ON DELETE CASCADE,
    workspace_id TEXT REFERENCES chat_workspaces(id) ON DELETE SET NULL,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
    status TEXT NOT NULL DEFAULT 'pending_address',
    trigger_type TEXT NOT NULL DEFAULT 'daily_probability',
    gift_name TEXT NOT NULL DEFAULT '',
    gift_reason TEXT,
    gift_note TEXT,
    product_image_url TEXT,
    target_amount_cents INTEGER NOT NULL DEFAULT 0,
    paid_amount_cents INTEGER NOT NULL DEFAULT 0,
    provider TEXT NOT NULL DEFAULT 'mock',
    provider_order_id TEXT,
    tracking_number TEXT,
    address_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    failure_reason TEXT,
    thanks_message TEXT,
    thanks_sent_at TIMESTAMPTZ,
    ordered_at TIMESTAMPTZ,
    shipped_at TIMESTAMPTZ,
    delivered_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS gift_tracking_events (
    id TEXT PRIMARY KEY,
    gift_id TEXT NOT NULL REFERENCES real_world_gifts(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    location TEXT,
    occurred_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS real_world_trigger_states (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    agent_id TEXT NOT NULL REFERENCES ai_agents(id) ON DELETE CASCADE,
    workspace_id TEXT REFERENCES chat_workspaces(id) ON DELETE CASCADE,
    next_activity_recommendation_at TIMESTAMPTZ,
    last_activity_recommendation_at TIMESTAMPTZ,
    last_gift_paid_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, agent_id)
);

CREATE TABLE IF NOT EXISTS real_world_recharge_ledger (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    amount_cents INTEGER NOT NULL,
    source TEXT NOT NULL DEFAULT 'store_recharge',
    source_id TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS offline_activity_user_status_idx
    ON offline_activity_recommendations(user_id, status, created_at DESC);

CREATE INDEX IF NOT EXISTS offline_activity_workspace_idx
    ON offline_activity_recommendations(workspace_id, created_at DESC);

CREATE INDEX IF NOT EXISTS offline_activity_feedback_recommendation_id_idx
    ON offline_activity_feedback(recommendation_id);

CREATE INDEX IF NOT EXISTS gift_addresses_user_default_idx
    ON gift_addresses(user_id, is_default);

CREATE INDEX IF NOT EXISTS real_world_gifts_user_status_idx
    ON real_world_gifts(user_id, status, created_at DESC);

CREATE INDEX IF NOT EXISTS gift_tracking_events_gift_time_idx
    ON gift_tracking_events(gift_id, occurred_at ASC);

CREATE UNIQUE INDEX IF NOT EXISTS real_world_trigger_states_workspace_id_key
    ON real_world_trigger_states(workspace_id);

CREATE INDEX IF NOT EXISTS real_world_trigger_states_due_idx
    ON real_world_trigger_states(next_activity_recommendation_at);

CREATE INDEX IF NOT EXISTS real_world_recharge_ledger_user_idx
    ON real_world_recharge_ledger(user_id, created_at DESC);
