CREATE TABLE "agent_avatar_cache" (
    "key" TEXT NOT NULL,
    "gender" TEXT,
    "content_type" TEXT NOT NULL,
    "image_bytes" BYTEA NOT NULL,
    "source_url" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "agent_avatar_cache_pkey" PRIMARY KEY ("key")
);

CREATE INDEX "agent_avatar_cache_gender_idx" ON "agent_avatar_cache"("gender");
