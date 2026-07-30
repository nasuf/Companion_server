-- Per-game visibility switch for the Flutter game hub. Disabled games keep all
-- their code/config but are hidden from the client catalog. Defaults to true so
-- every existing game stays visible after the migration.
ALTER TABLE "native_game_configs"
    ADD COLUMN IF NOT EXISTS "enabled" BOOLEAN NOT NULL DEFAULT true;
