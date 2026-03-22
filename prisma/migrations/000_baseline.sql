-- CreateExtension
CREATE EXTENSION IF NOT EXISTS "vector" WITH SCHEMA "extensions";

-- CreateTable
CREATE TABLE "users" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "email" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "users_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ai_agents" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "personality" JSONB,
    "seven_dim_traits" JSONB,
    "current_traits" JSONB,
    "traits_history" JSONB,
    "background" TEXT,
    "values" JSONB,
    "life_overview" TEXT,
    "life_overview_data" JSONB,
    "age" INTEGER,
    "occupation" TEXT,
    "city" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "ai_agents_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "conversations" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "agent_id" TEXT NOT NULL,
    "title" TEXT,
    "is_deleted" BOOLEAN NOT NULL DEFAULT false,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "conversations_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "messages" (
    "id" TEXT NOT NULL,
    "conversation_id" TEXT NOT NULL,
    "role" TEXT NOT NULL,
    "content" TEXT NOT NULL,
    "metadata" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "messages_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "prompt_templates" (
    "id" TEXT NOT NULL,
    "key" TEXT NOT NULL,
    "stage" TEXT NOT NULL,
    "category" TEXT NOT NULL,
    "title" TEXT NOT NULL,
    "description" TEXT,
    "content" TEXT NOT NULL,
    "default_content" TEXT NOT NULL,
    "is_enabled" BOOLEAN NOT NULL DEFAULT true,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "prompt_templates_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "memories_user" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "type" TEXT,
    "level" INTEGER NOT NULL DEFAULT 3,
    "content" TEXT NOT NULL,
    "summary" TEXT,
    "importance" DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    "mention_count" INTEGER NOT NULL DEFAULT 0,
    "is_archived" BOOLEAN NOT NULL DEFAULT false,
    "occur_time" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "memories_user_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "memories_ai" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "type" TEXT,
    "level" INTEGER NOT NULL DEFAULT 3,
    "content" TEXT NOT NULL,
    "summary" TEXT,
    "importance" DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    "mention_count" INTEGER NOT NULL DEFAULT 0,
    "is_archived" BOOLEAN NOT NULL DEFAULT false,
    "occur_time" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "memories_ai_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "user_profiles" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "traits" JSONB,
    "interests" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "user_profiles_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ai_emotion_states" (
    "id" TEXT NOT NULL,
    "agent_id" TEXT NOT NULL,
    "pleasure" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "arousal" DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    "dominance" DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "ai_emotion_states_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "user_portraits" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "agent_id" TEXT NOT NULL,
    "version" INTEGER NOT NULL DEFAULT 1,
    "content" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "user_portraits_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "memory_changelogs" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "memory_id" TEXT NOT NULL,
    "operation" TEXT NOT NULL,
    "old_value" TEXT,
    "new_value" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "memory_changelogs_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "memory_embeddings" (
    "memory_id" TEXT NOT NULL,
    "embedding" extensions.vector(768) NOT NULL,

    CONSTRAINT "memory_embeddings_pkey" PRIMARY KEY ("memory_id")
);

-- CreateTable
CREATE TABLE "intimacies" (
    "id" TEXT NOT NULL,
    "agent_id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "topic_intimacy" INTEGER NOT NULL DEFAULT 50,
    "topic_level" TEXT NOT NULL DEFAULT 'L3',
    "topic_updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "growth_intimacy" INTEGER NOT NULL DEFAULT 500,
    "growth_level" TEXT NOT NULL DEFAULT 'G5',
    "growth_updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "intimacies_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ai_daily_schedules" (
    "id" TEXT NOT NULL,
    "agent_id" TEXT NOT NULL,
    "date" DATE NOT NULL,
    "schedule_data" JSONB NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "ai_daily_schedules_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "schedule_adjust_logs" (
    "id" TEXT NOT NULL,
    "agent_id" TEXT NOT NULL,
    "adjust_type" TEXT NOT NULL,
    "old_value" TEXT,
    "new_value" TEXT,
    "reason" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "schedule_adjust_logs_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "proactive_chat_logs" (
    "id" TEXT NOT NULL,
    "agent_id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "message" TEXT NOT NULL,
    "event_type" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "proactive_chat_logs_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "trait_feedback_logs" (
    "id" TEXT NOT NULL,
    "agent_id" TEXT NOT NULL,
    "dimension" TEXT NOT NULL,
    "delta" DOUBLE PRECISION NOT NULL,
    "source" TEXT NOT NULL,
    "reason" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "trait_feedback_logs_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "stickers" (
    "id" SERIAL NOT NULL,
    "url" VARCHAR(255) NOT NULL,
    "emotion_tags" JSONB NOT NULL,
    "intensity" INTEGER NOT NULL DEFAULT 3,
    "style" VARCHAR(50),
    "tags" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "stickers_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "time_triggers" (
    "id" TEXT NOT NULL,
    "ai_agent_id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "trigger_time" TIMESTAMP(3) NOT NULL,
    "repeat_rule" TEXT,
    "action_type" TEXT NOT NULL,
    "action_data" JSONB,
    "is_active" BOOLEAN NOT NULL DEFAULT true,
    "last_fired" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "time_triggers_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "users_email_key" ON "users"("email");

-- CreateIndex
CREATE UNIQUE INDEX "prompt_templates_key_key" ON "prompt_templates"("key");

-- CreateIndex
CREATE UNIQUE INDEX "user_profiles_user_id_key" ON "user_profiles"("user_id");

-- CreateIndex
CREATE UNIQUE INDEX "ai_emotion_states_agent_id_key" ON "ai_emotion_states"("agent_id");

-- CreateIndex
CREATE UNIQUE INDEX "intimacies_agent_id_user_id_key" ON "intimacies"("agent_id", "user_id");

-- CreateIndex
CREATE UNIQUE INDEX "ai_daily_schedules_agent_id_date_key" ON "ai_daily_schedules"("agent_id", "date");

-- CreateIndex
CREATE INDEX "idx_user_portraits_user_id" ON "user_portraits"("user_id");

-- CreateIndex
CREATE INDEX "idx_memory_changelogs_user_id" ON "memory_changelogs"("user_id");

-- CreateIndex
CREATE INDEX "idx_memory_changelogs_created_at" ON "memory_changelogs"("created_at");

-- CreateIndex
CREATE INDEX "idx_memory_embeddings_vector" ON "memory_embeddings" USING ivfflat ("embedding" extensions.vector_cosine_ops) WITH (lists = 100);

-- CreateIndex
CREATE INDEX "idx_intimacies_agent_user" ON "intimacies"("agent_id", "user_id");

-- CreateIndex
CREATE INDEX "idx_ai_daily_schedules_agent_date" ON "ai_daily_schedules"("agent_id", "date");

-- CreateIndex
CREATE INDEX "idx_proactive_chat_logs_agent" ON "proactive_chat_logs"("agent_id", "created_at");

-- AddForeignKey
ALTER TABLE "ai_agents" ADD CONSTRAINT "ai_agents_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "conversations" ADD CONSTRAINT "conversations_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "conversations" ADD CONSTRAINT "conversations_agent_id_fkey" FOREIGN KEY ("agent_id") REFERENCES "ai_agents"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "messages" ADD CONSTRAINT "messages_conversation_id_fkey" FOREIGN KEY ("conversation_id") REFERENCES "conversations"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "user_profiles" ADD CONSTRAINT "user_profiles_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ai_emotion_states" ADD CONSTRAINT "ai_emotion_states_agent_id_fkey" FOREIGN KEY ("agent_id") REFERENCES "ai_agents"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "user_portraits" ADD CONSTRAINT "user_portraits_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "memory_changelogs" ADD CONSTRAINT "memory_changelogs_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "intimacies" ADD CONSTRAINT "intimacies_agent_id_fkey" FOREIGN KEY ("agent_id") REFERENCES "ai_agents"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "intimacies" ADD CONSTRAINT "intimacies_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ai_daily_schedules" ADD CONSTRAINT "ai_daily_schedules_agent_id_fkey" FOREIGN KEY ("agent_id") REFERENCES "ai_agents"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "proactive_chat_logs" ADD CONSTRAINT "proactive_chat_logs_agent_id_fkey" FOREIGN KEY ("agent_id") REFERENCES "ai_agents"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "trait_feedback_logs" ADD CONSTRAINT "trait_feedback_logs_agent_id_fkey" FOREIGN KEY ("agent_id") REFERENCES "ai_agents"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "time_triggers" ADD CONSTRAINT "time_triggers_ai_agent_id_fkey" FOREIGN KEY ("ai_agent_id") REFERENCES "ai_agents"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "time_triggers" ADD CONSTRAINT "time_triggers_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
