-- Add mbti field on ai_agents per spec《产品手册·背景信息》§1.2.
-- JSON shape: { "EI": int, "NS": int, "TF": int, "JP": int,
--               "type": "ENFP" (4-letter code),
--               "summary": "..." (LLM-generated 1-2 sentence flavor) }

ALTER TABLE "ai_agents" ADD COLUMN "mbti" JSONB;
