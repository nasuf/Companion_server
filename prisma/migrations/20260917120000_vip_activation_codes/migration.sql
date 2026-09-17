-- VIP activation codes: admin-generated redeemable codes + per-user redemption audit.

CREATE TABLE vip_activation_codes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    code TEXT NOT NULL,
    duration_days INT NOT NULL CHECK (duration_days > 0),
    max_redemptions INT NULL CHECK (max_redemptions IS NULL OR max_redemptions > 0),
    redemption_count INT NOT NULL DEFAULT 0 CHECK (redemption_count >= 0),
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    valid_from TIMESTAMP NULL,
    valid_until TIMESTAMP NULL,
    note TEXT NULL,
    created_by UUID NULL REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX vip_activation_codes_code_idx ON vip_activation_codes (code);
CREATE INDEX vip_activation_codes_enabled_created_idx
    ON vip_activation_codes (enabled, created_at DESC);

CREATE TABLE vip_code_redemptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    code_id UUID NOT NULL REFERENCES vip_activation_codes(id) ON DELETE RESTRICT,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    duration_days INT NOT NULL CHECK (duration_days > 0),
    status TEXT NOT NULL DEFAULT 'granted'
        CHECK (status IN ('granted', 'revoked')),
    redeemed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    effective_start TIMESTAMP NULL,
    effective_end TIMESTAMP NULL,
    revoked_at TIMESTAMP NULL,
    revoked_by UUID NULL REFERENCES users(id) ON DELETE SET NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE UNIQUE INDEX vip_code_redemptions_code_user_granted_idx
    ON vip_code_redemptions (code_id, user_id)
    WHERE status = 'granted';

CREATE INDEX vip_code_redemptions_user_redeemed_idx
    ON vip_code_redemptions (user_id, redeemed_at DESC);

CREATE INDEX vip_code_redemptions_code_redeemed_idx
    ON vip_code_redemptions (code_id, redeemed_at DESC);
