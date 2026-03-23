-- Migration: Add authentication fields to users and gender to ai_agents
-- WARNING: This migration deletes ALL existing test data to add NOT NULL constraints

-- Delete in correct foreign key order
DELETE FROM "messages";
DELETE FROM "proactive_chat_logs";
DELETE FROM "trait_feedback_logs";
DELETE FROM "time_triggers";
DELETE FROM "schedule_adjust_logs";
DELETE FROM "ai_daily_schedules";
DELETE FROM "ai_emotion_states";
DELETE FROM "memory_changelogs";
DELETE FROM "user_portraits";
DELETE FROM "user_profiles";
DELETE FROM "intimacies";
DELETE FROM "memories_user";
DELETE FROM "memories_ai";
DELETE FROM "conversations";
DELETE FROM "ai_agents";
DELETE FROM "users";

-- Add auth fields to users
ALTER TABLE "users" ADD COLUMN "username" TEXT NOT NULL DEFAULT '';
ALTER TABLE "users" ADD COLUMN "hashed_password" TEXT NOT NULL DEFAULT '';
ALTER TABLE "users" ADD COLUMN "role" TEXT NOT NULL DEFAULT 'user';

-- Remove defaults (they were only for migration safety)
ALTER TABLE "users" ALTER COLUMN "username" DROP DEFAULT;
ALTER TABLE "users" ALTER COLUMN "hashed_password" DROP DEFAULT;

-- Add unique constraint on username
ALTER TABLE "users" ADD CONSTRAINT "users_username_key" UNIQUE ("username");

-- Add gender to ai_agents
ALTER TABLE "ai_agents" ADD COLUMN "gender" TEXT;
