-- Upper bound of the AI reaction-time range. Together with min_response_ms it
-- defines [min, max]; the client picks a random delay in that range per move so
-- the AI never responds at a fixed cadence. Defaults to 1600 (> the 900 min) so
-- existing rows get a sensible spread immediately.
ALTER TABLE "native_game_configs"
    ADD COLUMN IF NOT EXISTS "max_response_ms" INTEGER NOT NULL DEFAULT 1600;
