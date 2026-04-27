-- 删除 chat.emotion_instruction / chat.memory_instruction 两个工程冗余 prompt.
-- 审计认定: spec §4 仅要求注入 PAD/关系阶段数据本身, 不要求加 "让情绪影响语气"
-- 或 "记忆 token 预算" 这两句明示文案——LLM 看到 PAD 数值已会自然反应,
-- 看到 token 数字也不改变行为.
DELETE FROM "prompt_templates" WHERE "key" IN (
    'chat.emotion_instruction',
    'chat.memory_instruction'
);
