CREATE TABLE "native_game_configs" (
    "game_key" TEXT NOT NULL,
    "mode" TEXT NOT NULL DEFAULT 'adaptive',
    "base_strength" INTEGER NOT NULL DEFAULT 50,
    "min_strength" INTEGER NOT NULL DEFAULT 20,
    "max_strength" INTEGER NOT NULL DEFAULT 85,
    "target_user_rate" DOUBLE PRECISION NOT NULL DEFAULT 0.55,
    "adjustment_window" INTEGER NOT NULL DEFAULT 10,
    "minimum_games" INTEGER NOT NULL DEFAULT 3,
    "maximum_step" INTEGER NOT NULL DEFAULT 5,
    "algorithm_overrides" JSONB NOT NULL DEFAULT '{}'::jsonb,
    "version" INTEGER NOT NULL DEFAULT 1,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "native_game_configs_pkey" PRIMARY KEY ("game_key"),
    CONSTRAINT "native_game_configs_mode_check"
        CHECK ("mode" IN ('fixed', 'adaptive')),
    CONSTRAINT "native_game_configs_strength_check"
        CHECK (
            "min_strength" BETWEEN 0 AND 100
            AND "base_strength" BETWEEN "min_strength" AND "max_strength"
            AND "max_strength" BETWEEN 0 AND 100
        ),
    CONSTRAINT "native_game_configs_rate_check"
        CHECK ("target_user_rate" BETWEEN 0.05 AND 0.95),
    CONSTRAINT "native_game_configs_window_check"
        CHECK ("adjustment_window" BETWEEN 2 AND 50),
    CONSTRAINT "native_game_configs_minimum_games_check"
        CHECK ("minimum_games" BETWEEN 1 AND 20),
    CONSTRAINT "native_game_configs_maximum_step_check"
        CHECK ("maximum_step" BETWEEN 1 AND 15)
);

CREATE TABLE "native_game_config_versions" (
    "id" TEXT NOT NULL DEFAULT gen_random_uuid()::text,
    "game_key" TEXT NOT NULL,
    "version" INTEGER NOT NULL,
    "config" JSONB NOT NULL,
    "published_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "native_game_config_versions_pkey" PRIMARY KEY ("id"),
    CONSTRAINT "native_game_config_versions_game_key_fkey"
        FOREIGN KEY ("game_key") REFERENCES "native_game_configs"("game_key")
        ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE UNIQUE INDEX "native_game_config_versions_game_version_key"
    ON "native_game_config_versions"("game_key", "version");
CREATE INDEX "native_game_config_versions_game_published_idx"
    ON "native_game_config_versions"("game_key", "published_at" DESC);

CREATE TABLE "native_game_skill_states" (
    "id" TEXT NOT NULL DEFAULT gen_random_uuid()::text,
    "user_id" TEXT NOT NULL,
    "agent_id" TEXT NOT NULL,
    "game_key" TEXT NOT NULL,
    "effective_strength" INTEGER NOT NULL DEFAULT 50,
    "completed_games" INTEGER NOT NULL DEFAULT 0,
    "ewma_user_rate" DOUBLE PRECISION,
    "wins" INTEGER NOT NULL DEFAULT 0,
    "losses" INTEGER NOT NULL DEFAULT 0,
    "draws" INTEGER NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "native_game_skill_states_pkey" PRIMARY KEY ("id"),
    CONSTRAINT "native_game_skill_states_user_id_fkey"
        FOREIGN KEY ("user_id") REFERENCES "users"("id")
        ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "native_game_skill_states_agent_id_fkey"
        FOREIGN KEY ("agent_id") REFERENCES "ai_agents"("id")
        ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "native_game_skill_states_strength_check"
        CHECK ("effective_strength" BETWEEN 0 AND 100),
    CONSTRAINT "native_game_skill_states_counts_check"
        CHECK (
            "completed_games" >= 0 AND "wins" >= 0
            AND "losses" >= 0 AND "draws" >= 0
        ),
    CONSTRAINT "native_game_skill_states_rate_check"
        CHECK ("ewma_user_rate" IS NULL OR "ewma_user_rate" BETWEEN 0 AND 1)
);

CREATE UNIQUE INDEX "native_game_skill_states_pair_game_key"
    ON "native_game_skill_states"("user_id", "agent_id", "game_key");
CREATE INDEX "native_game_skill_states_game_updated_idx"
    ON "native_game_skill_states"("game_key", "updated_at" DESC);

INSERT INTO "native_game_configs" (
    "game_key", "mode", "base_strength", "min_strength", "max_strength",
    "target_user_rate", "adjustment_window", "minimum_games", "maximum_step"
) VALUES
    ('go', 'adaptive', 50, 20, 85, 0.55, 10, 3, 5),
    ('reversi', 'adaptive', 50, 20, 85, 0.55, 10, 3, 5),
    ('gomoku', 'adaptive', 50, 20, 85, 0.55, 10, 3, 5),
    ('xiangqi', 'adaptive', 50, 20, 85, 0.55, 10, 3, 5),
    ('chess', 'adaptive', 50, 20, 85, 0.55, 10, 3, 5),
    ('chinese_checkers', 'adaptive', 50, 20, 85, 0.55, 10, 3, 5),
    ('match3', 'adaptive', 50, 20, 85, 0.70, 10, 3, 5),
    ('minesweeper', 'adaptive', 50, 20, 85, 0.70, 10, 3, 5),
    ('number_merge', 'adaptive', 50, 20, 85, 0.70, 10, 3, 5),
    ('tetris_duel', 'adaptive', 50, 20, 85, 0.55, 10, 3, 5);

INSERT INTO "native_game_config_versions" ("game_key", "version", "config")
SELECT
    "game_key",
    "version",
    jsonb_build_object(
        'game_key', "game_key",
        'mode', "mode",
        'base_strength', "base_strength",
        'min_strength', "min_strength",
        'max_strength', "max_strength",
        'target_user_rate', "target_user_rate",
        'adjustment_window', "adjustment_window",
        'minimum_games', "minimum_games",
        'maximum_step', "maximum_step",
        'algorithm_overrides', "algorithm_overrides",
        'version', "version"
    )
FROM "native_game_configs";
