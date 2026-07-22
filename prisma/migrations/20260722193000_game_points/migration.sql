-- Game points system: per-user spendable game-point wallet, audit ledger,
-- admin-configurable level ladder, and per-game scoring rules.
--
-- Design notes:
--   * `user_game_wallets.balance` is the spendable game-point balance. It gates
--     play (balance <= 0 => cannot start a game today) and drives the level.
--   * `last_grant_date` (UTC+8 date) records the last day the daily grant ran so
--     the +20 top-up happens at most once per day (only when balance < 20).
--   * Levels are derived from `lifetime_earned` (points actually earned by
--     winning / reaching milestones) against `game_level_tiers`
--     (cumulative_points thresholds). Daily grants, losses, quits and shop
--     conversions never change `lifetime_earned`, so the level only ever climbs.
--   * `game_point_ledger` gives an audit trail and enforces idempotency for
--     daily grants (source_id = date) and per-session settlement
--     (source_id = game session id) via a partial unique index.

CREATE TABLE IF NOT EXISTS user_game_wallets (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id TEXT NOT NULL,
    balance INTEGER NOT NULL DEFAULT 0,
    -- Monotonic total of points earned purely by winning / reaching milestones.
    -- Drives the level; never decreased by losses/quits/grants/conversions.
    lifetime_earned INTEGER NOT NULL DEFAULT 0,
    last_grant_date DATE,
    created_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT user_game_wallets_user_id_key UNIQUE (user_id),
    CONSTRAINT user_game_wallets_user_id_fkey
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT user_game_wallets_balance_nonnegative CHECK (balance >= 0),
    CONSTRAINT user_game_wallets_lifetime_earned_nonnegative CHECK (lifetime_earned >= 0)
);

CREATE TABLE IF NOT EXISTS game_point_ledger (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id TEXT NOT NULL,
    delta INTEGER NOT NULL,
    balance_after INTEGER NOT NULL,
    source TEXT NOT NULL,
    source_id TEXT,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT game_point_ledger_user_id_fkey
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT game_point_ledger_balance_after_nonnegative CHECK (balance_after >= 0)
);

CREATE INDEX IF NOT EXISTS game_point_ledger_user_time_idx
    ON game_point_ledger(user_id, created_at DESC);

-- One grant per (user, day) and one settlement per (user, session): a partial
-- unique index guards double-credit even if the caller retries.
CREATE UNIQUE INDEX IF NOT EXISTS game_point_ledger_source_key
    ON game_point_ledger(user_id, source, source_id)
    WHERE source_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS game_level_tiers (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    sort_order INTEGER NOT NULL,
    stage_name TEXT NOT NULL,
    tier_name TEXT NOT NULL,
    upgrade_points INTEGER NOT NULL DEFAULT 0,
    cumulative_points INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT game_level_tiers_sort_order_key UNIQUE (sort_order),
    CONSTRAINT game_level_tiers_upgrade_points_nonnegative CHECK (upgrade_points >= 0),
    CONSTRAINT game_level_tiers_cumulative_points_nonnegative CHECK (cumulative_points >= 0)
);

CREATE TABLE IF NOT EXISTS game_point_rules (
    game_key TEXT PRIMARY KEY,
    rules JSONB NOT NULL,
    updated_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Seed the default level ladder (PM spec 2026-07). Admin can edit later.
INSERT INTO game_level_tiers (sort_order, stage_name, tier_name, upgrade_points, cumulative_points) VALUES
    (1,  '白手套（新手）', '白手套・1 阶', 0,    0),
    (2,  '白手套（新手）', '白手套・2 阶', 50,   50),
    (3,  '白手套（新手）', '白手套・3 阶', 100,  150),
    (4,  '白手套（新手）', '白手套・4 阶', 150,  300),
    (5,  '白手套（新手）', '白手套・5 阶', 200,  500),
    (6,  '蓝手套（进阶）', '蓝手套・1 阶', 250,  750),
    (7,  '蓝手套（进阶）', '蓝手套・2 阶', 300,  1050),
    (8,  '蓝手套（进阶）', '蓝手套・3 阶', 350,  1400),
    (9,  '蓝手套（进阶）', '蓝手套・4 阶', 400,  1800),
    (10, '蓝手套（进阶）', '蓝手套・5 阶', 400,  2200),
    (11, '紫手套（熟练）', '紫手套・1 阶', 450,  2650),
    (12, '紫手套（熟练）', '紫手套・2 阶', 500,  3150),
    (13, '紫手套（熟练）', '紫手套・3 阶', 550,  3700),
    (14, '紫手套（熟练）', '紫手套・4 阶', 600,  4300),
    (15, '紫手套（熟练）', '紫手套・5 阶', 600,  4900),
    (16, '橙手套（高阶）', '橙手套・1 阶', 650,  5550),
    (17, '橙手套（高阶）', '橙手套・2 阶', 700,  6250),
    (18, '橙手套（高阶）', '橙手套・3 阶', 750,  7000),
    (19, '橙手套（高阶）', '橙手套・4 阶', 800,  7800),
    (20, '橙手套（高阶）', '橙手套・5 阶', 800,  8600),
    (21, '彩手套（满级）', '彩手套・1 阶', 850,  9450),
    (22, '彩手套（满级）', '彩手套・2 阶', 900,  10350),
    (23, '彩手套（满级）', '彩手套・3 阶', 1000, 11350),
    (24, '彩手套（满级）', '彩手套・4 阶', 1000, 12350),
    (25, '彩手套（满级）', '彩手套・5 阶', 1100, 13450)
ON CONFLICT (sort_order) DO NOTHING;

-- Seed per-game scoring rules (PM spec 2026-07). `quit` = 中途退出.
-- match3 (怪物消消乐) and tetris_duel (双人方块竞速) have no PM spec yet, so they
-- carry placeholder values flagged with "pending_pm" for the admin UI to surface.
INSERT INTO game_point_rules (game_key, rules) VALUES
    ('reversi',          '{"type":"outcome","win":4,"lose":-3,"draw":0,"quit":-3}'::jsonb),
    ('gomoku',           '{"type":"outcome","win":3,"lose":-2,"draw":0,"quit":-2}'::jsonb),
    ('xiangqi',          '{"type":"outcome","win":4,"lose":-3,"draw":0,"quit":-3}'::jsonb),
    ('go',               '{"type":"outcome","win":5,"lose":-4,"draw":0,"quit":-4}'::jsonb),
    ('chinese_checkers', '{"type":"outcome","win":5,"lose":-4,"draw":0,"quit":-4}'::jsonb),
    ('chess',            '{"type":"outcome","win":4,"lose":-3,"draw":0,"quit":-3}'::jsonb),
    ('minesweeper',      '{"type":"outcome","win":3,"lose":-2,"draw":0,"quit":-2}'::jsonb),
    ('match3',           '{"type":"outcome","win":3,"lose":-2,"draw":0,"quit":-2,"pending_pm":true}'::jsonb),
    ('tetris_duel',      '{"type":"outcome","win":4,"lose":-3,"draw":0,"quit":-3,"pending_pm":true}'::jsonb),
    ('number_merge',     '{"type":"milestone","milestones":[{"tile":128,"points":2},{"tile":256,"points":5},{"tile":512,"points":6},{"tile":1024,"points":15},{"tile":2048,"points":25}],"quit_below_threshold":{"threshold":128,"below":-2,"at_or_above":0}}'::jsonb)
ON CONFLICT (game_key) DO NOTHING;
