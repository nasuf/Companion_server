-- 后台「聊天管理 → 主动交流」可调参数: 只存与默认不同的项
ALTER TABLE "system_config" ADD COLUMN "proactive_policy" JSONB;

-- 参数修改记录
CREATE TABLE "proactive_tuning_changes" (
    "id" TEXT NOT NULL,
    "admin_id" TEXT,
    "changes" JSONB NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "proactive_tuning_changes_pkey" PRIMARY KEY ("id")
);

CREATE INDEX "proactive_tuning_changes_created_at_idx" ON "proactive_tuning_changes"("created_at" DESC);
