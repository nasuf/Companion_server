CREATE TABLE "user_feedback" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "content" TEXT NOT NULL,
    "contact" TEXT NOT NULL,
    "occurred_at" TEXT,
    "image_keys" TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    "status" TEXT NOT NULL DEFAULT 'open',
    "app_version" TEXT,
    "platform" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "user_feedback_pkey" PRIMARY KEY ("id")
);

CREATE INDEX "user_feedback_user_id_created_at_idx" ON "user_feedback"("user_id", "created_at" DESC);
CREATE INDEX "user_feedback_status_created_at_idx" ON "user_feedback"("status", "created_at" DESC);

ALTER TABLE "user_feedback" ADD CONSTRAINT "user_feedback_user_id_fkey"
    FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
