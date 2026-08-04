-- "你的回合" turn-banner timing (ms), per-game and admin-tunable. Delivered in
-- engine_config; the client derives the pop-in / hold / fade-out phases from
-- these three values. Defaults match the previously hard-coded 200/600/200
-- cadence so existing rows are visually identical after the migration.
ALTER TABLE "native_game_configs"
    ADD COLUMN IF NOT EXISTS "banner_in_ms" INTEGER NOT NULL DEFAULT 200,
    ADD COLUMN IF NOT EXISTS "banner_hold_ms" INTEGER NOT NULL DEFAULT 600,
    ADD COLUMN IF NOT EXISTS "banner_out_ms" INTEGER NOT NULL DEFAULT 200;
