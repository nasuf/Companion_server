BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
-- Every StepFun official voice accepted by both stepaudio-2.5-tts and
-- stepaudio-2.5-realtime (verified one by one on 2026-11-04), so an agent's
-- voice messages and calls keep one voice. Excluded: TTS-only voices the
-- realtime endpoint rejects, and English-only voices.
-- Existing rows keep their enabled flag (admins may have disabled one).
INSERT INTO tts_voice_profiles (
  id, display_name, provider, model, voice_id, gender, source, enabled, updated_at
) VALUES
  (gen_random_uuid(), '气质温婉', 'stepfun', 'stepaudio-2.5-tts', 'elegantgentle-female', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '机灵少女', 'stepfun', 'stepaudio-2.5-tts', 'jilingshaonv', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '经典女声', 'stepfun', 'stepaudio-2.5-tts', 'jingdiannvsheng', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '冷艳御姐', 'stepfun', 'stepaudio-2.5-tts', 'lengyanyujie', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '邻家姐姐', 'stepfun', 'stepaudio-2.5-tts', 'linjiajiejie', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '邻家妹妹', 'stepfun', 'stepaudio-2.5-tts', 'linjiameimei', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '活力轻快', 'stepfun', 'stepaudio-2.5-tts', 'livelybreezy-female', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '清纯少女', 'stepfun', 'stepaudio-2.5-tts', 'qingchunshaonv', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '亲切女声', 'stepfun', 'stepaudio-2.5-tts', 'qinqienvsheng', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '软萌女声', 'stepfun', 'stepaudio-2.5-tts', 'ruanmengnvsheng', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '爽快姐姐', 'stepfun', 'stepaudio-2.5-tts', 'shuangkuaijiejie', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '文静学姐', 'stepfun', 'stepaudio-2.5-tts', 'wenjingxuejie', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '温柔女声', 'stepfun', 'stepaudio-2.5-tts', 'wenrounvsheng', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '温柔熟女', 'stepfun', 'stepaudio-2.5-tts', 'wenroushunv', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '优雅女声', 'stepfun', 'stepaudio-2.5-tts', 'youyanvsheng', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '元气少女', 'stepfun', 'stepaudio-2.5-tts', 'yuanqishaonv', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '知性姐姐', 'stepfun', 'stepaudio-2.5-tts', 'zhixingjiejie', 'female', 'system', true, NOW()),
  (gen_random_uuid(), '播音男声', 'stepfun', 'stepaudio-2.5-tts', 'boyinnansheng', 'male', 'system', true, NOW()),
  (gen_random_uuid(), '磁性男声', 'stepfun', 'stepaudio-2.5-tts', 'cixingnansheng', 'male', 'system', true, NOW()),
  (gen_random_uuid(), '青年大学生', 'stepfun', 'stepaudio-2.5-tts', 'qingniandaxuesheng', 'male', 'system', true, NOW()),
  (gen_random_uuid(), '儒雅男士', 'stepfun', 'stepaudio-2.5-tts', 'ruyananshi', 'male', 'system', true, NOW()),
  (gen_random_uuid(), '深沉男音', 'stepfun', 'stepaudio-2.5-tts', 'shenchennanyin', 'male', 'system', true, NOW()),
  (gen_random_uuid(), '温柔男声', 'stepfun', 'stepaudio-2.5-tts', 'wenrounansheng', 'male', 'system', true, NOW()),
  (gen_random_uuid(), '元气男声', 'stepfun', 'stepaudio-2.5-tts', 'yuanqinansheng', 'male', 'system', true, NOW()),
  (gen_random_uuid(), '正派青年', 'stepfun', 'stepaudio-2.5-tts', 'zhengpaiqingnian', 'male', 'system', true, NOW())
ON CONFLICT (provider, model, voice_id) DO NOTHING;
COMMIT;
