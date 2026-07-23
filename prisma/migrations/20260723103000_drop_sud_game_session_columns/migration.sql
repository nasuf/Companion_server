-- Drop the residual SUD columns from game_sessions now that the SUD provider
-- integration is fully removed. Native games never used these (they were only
-- ever written empty/NULL), and no index depends on them, so these are
-- metadata-only ALTERs with no table rewrite and no data-loss risk for the
-- native game flow.
ALTER TABLE "game_sessions"
    DROP COLUMN IF EXISTS "mg_id",
    DROP COLUMN IF EXISTS "sud_code",
    DROP COLUMN IF EXISTS "sud_code_expires_at",
    DROP COLUMN IF EXISTS "sdk_enabled";

-- Native is now the only game provider, so make it the column default too.
ALTER TABLE "game_sessions" ALTER COLUMN "provider" SET DEFAULT 'native';
