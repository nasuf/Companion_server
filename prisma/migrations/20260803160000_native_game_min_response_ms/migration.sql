-- Per-game minimum AI response time (ms). Guarantees each AI move takes at
-- least this long wall-clock so the opponent feels human instead of instant;
-- if the engine already spent longer thinking, no extra delay is added. The
-- client (Flutter) reads it from engine_config and pads the remainder. Applies
-- to every game except tetris_duel, which paces via its own agent_move_ms.
-- Defaults to 900 so existing games immediately gain human-like pacing.
ALTER TABLE "native_game_configs"
    ADD COLUMN IF NOT EXISTS "min_response_ms" INTEGER NOT NULL DEFAULT 900;
