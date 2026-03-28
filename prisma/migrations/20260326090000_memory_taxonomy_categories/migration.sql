ALTER TABLE "memories_user"
    ADD COLUMN IF NOT EXISTS "main_category" TEXT,
    ADD COLUMN IF NOT EXISTS "sub_category" TEXT;

ALTER TABLE "memories_ai"
    ADD COLUMN IF NOT EXISTS "main_category" TEXT,
    ADD COLUMN IF NOT EXISTS "sub_category" TEXT;

CREATE INDEX IF NOT EXISTS "memories_user_main_category_idx"
    ON "memories_user"("main_category");
CREATE INDEX IF NOT EXISTS "memories_user_sub_category_idx"
    ON "memories_user"("sub_category");
CREATE INDEX IF NOT EXISTS "memories_ai_main_category_idx"
    ON "memories_ai"("main_category");
CREATE INDEX IF NOT EXISTS "memories_ai_sub_category_idx"
    ON "memories_ai"("sub_category");

UPDATE "memories_user"
SET
    "main_category" = CASE
        WHEN "type" = 'identity' THEN '身份'
        WHEN "type" = 'preference' THEN '偏好'
        WHEN "type" = 'emotion' THEN '情绪'
        WHEN "type" = 'thought' THEN '思维'
        ELSE '生活'
    END,
    "sub_category" = COALESCE("sub_category", '其他')
WHERE "main_category" IS NULL;

UPDATE "memories_ai"
SET
    "main_category" = CASE
        WHEN "type" = 'identity' THEN '身份'
        WHEN "type" = 'preference' THEN '偏好'
        WHEN "type" = 'emotion' THEN '情绪'
        WHEN "type" = 'thought' THEN '思维'
        ELSE '生活'
    END,
    "sub_category" = COALESCE("sub_category", '其他')
WHERE "main_category" IS NULL;
