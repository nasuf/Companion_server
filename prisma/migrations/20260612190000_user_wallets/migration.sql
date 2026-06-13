CREATE TABLE IF NOT EXISTS user_wallets (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id TEXT NOT NULL,
    ticket_balance INTEGER NOT NULL DEFAULT 0,
    point_balance INTEGER NOT NULL DEFAULT 0,
    achievement_points_synced INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT user_wallets_user_id_key UNIQUE (user_id),
    CONSTRAINT user_wallets_user_id_fkey
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT user_wallets_ticket_balance_nonnegative CHECK (ticket_balance >= 0),
    CONSTRAINT user_wallets_point_balance_nonnegative CHECK (point_balance >= 0),
    CONSTRAINT user_wallets_achievement_points_synced_nonnegative CHECK (achievement_points_synced >= 0)
);

CREATE TABLE IF NOT EXISTS wallet_ledger (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id TEXT NOT NULL,
    currency TEXT NOT NULL,
    delta INTEGER NOT NULL,
    balance_after INTEGER NOT NULL,
    source TEXT NOT NULL,
    source_id TEXT,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT wallet_ledger_user_id_fkey
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT wallet_ledger_currency_check CHECK (currency IN ('ticket', 'point')),
    CONSTRAINT wallet_ledger_balance_after_nonnegative CHECK (balance_after >= 0)
);

CREATE INDEX IF NOT EXISTS wallet_ledger_user_currency_time_idx
    ON wallet_ledger(user_id, currency, created_at DESC);

CREATE INDEX IF NOT EXISTS wallet_ledger_source_idx
    ON wallet_ledger(source, source_id);
