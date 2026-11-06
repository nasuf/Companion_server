BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

-- Voice calls and voice messages move to DashScope qwen3.8-omni-flash-realtime;
-- voice messages are the same model reading the reply aloud, so one voice id
-- (clones included) serves both. Every system voice of the model, each
-- verified on 2026-11-06: accepted, read the test line verbatim, gender by
-- listening (pitch check agreed except 东厂-李公公, a male character).
-- Existing rows keep their enabled flag (admins may have disabled one).
INSERT INTO tts_voice_profiles (
  id, display_name, provider, model, voice_id, gender, source, enabled, updated_at
) VALUES
  (gen_random_uuid(), '甜甜', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Tina', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '林欣宜', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Cindy', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '清欢', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Liora Mira', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '林川野', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Raymond', 'male', 'system', true, NOW()),
  (gen_random_uuid(), '泽恩', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Zane', 'male', 'system', true, NOW()),
  (gen_random_uuid(), '卡捷琳娜', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Katerina', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '甜茶', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Ryan', 'male', 'system', true, NOW()),
  (gen_random_uuid(), '温柔生活博主-舒然', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Mia', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '绵绵', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Cici', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '予安', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Theo Calm', 'male', 'system', true, NOW()),
  (gen_random_uuid(), '苏瑶', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Serena', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '四月', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Maia', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '江晨', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Evan', 'male', 'system', true, NOW()),
  (gen_random_uuid(), '小乔妹', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Qiao', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '茉兔', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Momo', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '伟伦', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Wil', 'male', 'system', true, NOW()),
  (gen_random_uuid(), '台普-安琪', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Angel', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '东厂-李公公', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Li Cassian', 'male', 'system', true, NOW()),
  (gen_random_uuid(), '喜剧担当-阿逗', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Joyner', 'male', 'system', true, NOW()),
  (gen_random_uuid(), '金爷', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Gold', 'male', 'system', true, NOW()),
  (gen_random_uuid(), '詹妮弗', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Jennifer', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '艾登', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Aiden', 'male', 'system', true, NOW()),
  (gen_random_uuid(), '敏儿', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Mione', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '四川-晴儿', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Sunny', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '北京-晓东', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Dylan', 'male', 'system', true, NOW()),
  (gen_random_uuid(), '四川-程川', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Eric', 'male', 'system', true, NOW()),
  (gen_random_uuid(), '天津-李彼得', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Peter', 'male', 'system', true, NOW()),
  (gen_random_uuid(), '阿樸伯', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Joseph Chen', 'male', 'system', true, NOW()),
  (gen_random_uuid(), '陕西-秦川', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Marcus', 'male', 'system', true, NOW()),
  (gen_random_uuid(), '南京-老李', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Li', 'male', 'system', true, NOW()),
  (gen_random_uuid(), '粤语-阿强', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Rocky', 'male', 'system', true, NOW()),
  (gen_random_uuid(), '粤语-阿清', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Kiki', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '素熙', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Sohee', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '艾莉卡', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Eliška', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '阿列克', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Alek', 'male', 'system', true, NOW()),
  (gen_random_uuid(), '阿尔达', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Arda', 'male', 'system', true, NOW()),
  (gen_random_uuid(), '多尔切', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Dolce', 'male', 'system', true, NOW()),
  (gen_random_uuid(), '莱恩', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Lenn', 'male', 'system', true, NOW()),
  (gen_random_uuid(), '小野杏', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Ono Anna', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '索尼莎', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Sonrisa', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '博德加', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Bodega', 'male', 'system', true, NOW()),
  (gen_random_uuid(), '安德雷', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Andre', 'male', 'system', true, NOW()),
  (gen_random_uuid(), '拉迪奥·戈尔', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Radio Gol', 'male', 'system', true, NOW()),
  (gen_random_uuid(), '阿力', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Rizky', 'male', 'system', true, NOW()),
  (gen_random_uuid(), '萝雅', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Roya', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '阿幸', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Hana', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '雅克', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Jakub', 'male', 'system', true, NOW()),
  (gen_random_uuid(), '海娜（Griet）', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Griet', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '玛丽娜', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Marina', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '西芮', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Siiri', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '林恩', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Ingrid', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '海娜（Sigga）', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Sigga', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '雅娜', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Bea', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '思怡', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Chloe', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '埃米尔安', 'dashscope', 'qwen3.8-omni-flash-realtime', 'Emilien', 'male', 'system', true, NOW()),
  (gen_random_uuid(), '龙安灵心', 'dashscope', 'qwen3.8-omni-flash-realtime', 'longanlingxin', 'female', 'system', true, NOW())
ON CONFLICT (provider, model, voice_id) DO NOTHING;

-- Defaults: 苏瑶 (Serena, "温柔小姐姐") and 予安 (Theo Calm, calm and soothing),
-- closest to the previous 温柔女声 / 温柔男声. Admins can change them later.
UPDATE tts_voice_profiles
SET enabled = true, updated_at = NOW()
WHERE provider = 'dashscope' AND model = 'qwen3.8-omni-flash-realtime'
  AND voice_id IN ('Serena', 'Theo Calm') AND NOT enabled;
UPDATE tts_voice_profiles
SET is_default = false, updated_at = NOW()
WHERE provider = 'dashscope' AND model = 'qwen3.8-omni-flash-realtime'
  AND is_default AND voice_id NOT IN ('Serena', 'Theo Calm');
UPDATE tts_voice_profiles
SET is_default = true, updated_at = NOW()
WHERE provider = 'dashscope' AND model = 'qwen3.8-omni-flash-realtime'
  AND ((voice_id = 'Serena' AND gender = 'female')
    OR (voice_id = 'Theo Calm' AND gender = 'male'))
  AND NOT is_default;

-- Every agent moves to its gender default: StepFun voice ids are unknown to
-- DashScope (an unknown voice closes the call socket on the first reply).
UPDATE ai_agents
SET tts_voice_id = CASE WHEN lower(coalesce(gender, '')) = 'male'
                        THEN 'Theo Calm' ELSE 'Serena' END,
    updated_at = NOW()
WHERE tts_voice_id IS DISTINCT FROM
      CASE WHEN lower(coalesce(gender, '')) = 'male'
           THEN 'Theo Calm' ELSE 'Serena' END;

-- Prices (CNY per million tokens, DashScope list price, Beijing, 2026-11-06).
-- Text and audio tokens are priced separately; the ':audio' row carries the
-- audio prices and is never a selectable model (realtime_voice/billing.py).
INSERT INTO model_registry (
  id, identifier, display_name, provider, enabled, model_kind, billing_unit,
  input_cost_per_million, output_cost_per_million, notes, created_at, updated_at
) VALUES
  (gen_random_uuid(), 'qwen3.8-omni-flash-realtime', 'Qwen3.8 Omni Flash Realtime（语音通话与语音消息）',
   'dashscope', true, 'realtime', 'per_million_tokens', 1.5, 4.5,
   '文本 token 价格；音频 token 价格见 :audio 行', NOW(), NOW()),
  (gen_random_uuid(), 'qwen3.8-omni-flash-realtime:audio', 'Qwen3.8 Omni Flash Realtime 音频 token（仅计价）',
   'dashscope', true, 'realtime', 'per_million_tokens', 6, 12,
   '音频输入 / 输出 token 价格，不可选为模型', NOW(), NOW())
ON CONFLICT (provider, identifier) DO NOTHING;

-- StepFun rows stay for pricing historical usage, but can no longer be picked.
UPDATE model_registry SET enabled = false, updated_at = NOW()
WHERE provider = 'stepfun' AND enabled;
UPDATE system_config SET tts_model = NULL WHERE id = 1 AND tts_model IS NOT NULL;
UPDATE system_config SET realtime_model = NULL
WHERE id = 1 AND realtime_model IS NOT NULL AND realtime_model <> 'qwen3.8-omni-flash-realtime';
COMMIT;
