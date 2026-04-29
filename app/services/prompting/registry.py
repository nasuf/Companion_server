"""Prompt registry for admin management and runtime lookup."""

from __future__ import annotations

from dataclasses import dataclass

from app.services.prompting.defaults import (
    AI_PAD_PROMPT,
    AI_REPLY_EMOTION_PROMPT,
    ANTI_HALLUCINATION_HARD_RULE_PROMPT,
    APOLOGY_PROMPT,
    APOLOGY_REPLY_PROMPT,
    ATTACK_LEVEL_PROMPT,
    ATTACK_TARGET_PROMPT,
    BANNED_WORD_PROMPT,
    BLACKLIST_REPLY_PROMPT,
    BOUNDARY_FINAL_WARNING_PROMPT,
    CHARACTER_GENERATION_PROMPT,
    CHARACTER_REPAIR_MISSING_FIELDS_PROMPT,
    CHAT_AI_STATE_CONSTRAINT_PROMPT,
    CONSISTENCY_RULES_PROMPT,
    CURRENT_STATE_REPLY_PROMPT,
    DAILY_SCHEDULE_PROMPT,
    DELAY_EXPLANATION_PROMPT,
    DELETION_CONFIRM_PROMPT,
    DELETION_INTENT_PROMPT,
    DELETION_REPLY_PROMPT,
    EMOTION_EXTRACTION_PROMPT,
    END_REPLY_PROMPT,
    PERSONALITY_SCORING_PROMPT,
    INTENT_SPLIT_PROMPT,
    INTENT_UNIFIED_PROMPT,
    L3_MEMORY_REPLY_PROMPT,
    L3_TRIGGER_PROMPT,
    LIFE_OVERVIEW_PROMPT,
    LIGHT_ATTACK_REPLY_PROMPT,
    LOW_PATIENCE_REPLY_PROMPT,
    MEDIUM_ATTACK_REPLY_PROMPT,
    MEDIUM_MEMORY_REPLY_PROMPT,
    MEDIUM_PATIENCE_REPLY_PROMPT,
    MEMORY_CONTRADICTION_ANALYSIS_PROMPT,
    MEMORY_CONTRADICTION_DETECTION_PROMPT,
    MEMORY_CONTRADICTION_INQUIRY_PROMPT,
    MEMORY_CONTRADICTION_REPLY_PROMPT,
    MEMORY_EXTRACTION_AI_PROMPT,
    MEMORY_EXTRACTION_USER_PROMPT,
    MEMORY_PAIRWISE_CONTRADICTION_PROMPT,
    MEMORY_RELEVANCE_PROMPT,
    PERSONALITY_RULES_PROMPT,
    PORTRAIT_GENERATION_PROMPT,
    POSITIVE_INTERACTION_PROMPT,
    PORTRAIT_UPDATE_PROMPT,
    PROACTIVE_SILENCE_PLAIN_PROMPT,
    PROACTIVE_SILENCE_AI_MEMORY_PROMPT,
    PROACTIVE_SILENCE_USER_MEMORY_PROMPT,
    PROACTIVE_SILENCE_SCHEDULE_PROMPT,
    PROACTIVE_MEMORY_AI_PROMPT,
    PROACTIVE_MEMORY_USER_PROMPT,
    PROACTIVE_SCHEDULED_SCENE_PROMPT,
    PROACTIVE_DECAY_FINAL_PROMPT,
    PROACTIVE_FIRST_GREETING_PROMPT,
    PROACTIVE_SPECIAL_HOLIDAY_PROMPT,
    PROACTIVE_SPECIAL_BIRTHDAY_PROMPT,
    PROACTIVE_SPECIAL_REMINDER_PROMPT,
    PROACTIVE_SPECIAL_COMBINED_PROMPT,
    REPLY_SPLIT_2_PROMPT,
    REPLY_SPLIT_3_PROMPT,
    RESPONSE_INSTRUCTION_PROMPT,
    DAILY_SCHEDULE_WITH_USER_MEMORY_PROMPT,
    MEMORY_JUDGEMENT_AI_PROMPT,
    MEMORY_JUDGEMENT_USER_PROMPT,
    SCHEDULE_ADJUST_REPLY_PROMPT,
    SCHEDULE_DAILY_SUMMARY_MEMORIES_PROMPT,
    SCHEDULE_DAILY_SUMMARY_PROMPT,
    SCHEDULE_QUERY_REPLY_PROMPT,
    SEVERE_ATTACK_REPLY_PROMPT,
    STRONG_MEMORY_REPLY_PROMPT,
    SYSTEM_BASE_PROMPT,
    WEAK_MEMORY_REPLY_PROMPT,
)


@dataclass(frozen=True)
class PromptDefinition:
    key: str
    title: str
    stage: str
    category: str
    description: str
    default_text: str


# spec 文档来源标记约定 (description 内统一格式):
#   《记忆部分产品手册》       — 背景 + 记忆模块 (录入/检索/矛盾/删除)
#   《终稿·第一部分》         — 生活画像 / AI 作息 / 昨日总结
#   《终稿·第三部分 交互系统》 — 聊天热路径 / 边界 / 意图 / 回复加工 / 异步回复
#   《终稿·第四部分 主动交流》 — 沉默唤醒 / 衰减 / 特殊日期 / 开场首句
#   《终稿·第五部分 时间日期系统》— 节日 / 提醒 / 作息时间映射
#   《终稿·指令模版 4.19》    — 各 prompt 的页编号 (P20/P23/P25/P26 等)
# 工程项以 【工程补丁】/【工程附加】/【工程扩展】/【工程主导】 前缀标识, 不带 spec 引用.

PROMPT_DEFINITIONS = [
    # ── 聊天热路径 (Part 3) ──
    PromptDefinition(
        "chat.system_base", "聊天核心规则", "聊天热路径", "聊天",
        "【工程主导】无字面 spec 章节. AI 不暴露 AI 身份 / 像真人聊天的 system 锚点 (《记忆部分产品手册》整体精神, 但具体文案由工程主导). 拼接在每条消息 system prompt 顶部.",
        SYSTEM_BASE_PROMPT,
    ),
    PromptDefinition(
        "chat.response_instruction", "聊天回复规则", "聊天热路径", "聊天",
        "《终稿·第三部分 交互系统》§5.5: 拆分 1-3 条 (均匀随机) + 每条 ≤ 60 字 + 总 ≤ 150 字. 占位符 {n}/{max_per}/{total}.",
        RESPONSE_INSTRUCTION_PROMPT,
    ),
    PromptDefinition(
        "chat.personality_rules", "人格一致性规则", "聊天热路径", "聊天",
        "【工程补丁】无字面 spec 章节. 防 LLM 长对话漂移到客服 / AI 助手语气, 强制保持人设.",
        PERSONALITY_RULES_PROMPT,
    ),
    PromptDefinition(
        "chat.consistency_rules", "对话一致性规则", "聊天热路径", "聊天",
        "【工程补丁】无字面 spec 章节. 防 LLM 重复提问 / 与上文矛盾 / 短情绪话立刻给大道理.",
        CONSISTENCY_RULES_PROMPT,
    ),
    PromptDefinition(
        "chat.anti_hallucination_hard_rule", "反幻觉硬约束", "聊天热路径", "聊天",
        "【工程补丁】无字面 spec 章节, 但呼应《记忆部分产品手册》§3.4 \"若联想记忆为无, 则明确告诉用户不记得\"."
        " v3 升级: 三分判断 (有 X / 有内容但无 X / 段空) + 顶部位置 + 警示用户问句不是记忆,"
        " 防 LLM 顺承预设性问句编造过往. 详见 prompt 顶部注释.",
        ANTI_HALLUCINATION_HARD_RULE_PROMPT,
    ),
    PromptDefinition(
        "chat.ai_state_constraint", "AI 自洽性隐性约束", "聊天热路径", "聊天",
        "【工程补丁】spec §4 不含 AI 当前作息, 但 ≥1min 延迟主回复路径下 LLM 可能编造"
        "跟真实状态矛盾的活动 (e.g. AI 睡眠中却说\"刚去爬山\"). 这段以约束式注入:"
        "告诉 LLM 状态但禁止主动展开, 避免跟 §3.4.3 询问当前状态分支撞主题. "
        "占位符: {activity} {status}.",
        CHAT_AI_STATE_CONSTRAINT_PROMPT,
    ),

    # ── 初始化 (背景生成) ──
    PromptDefinition(
        "agent.personality_scoring", "AI 性格打分", "初始化", "初始化",
        "《记忆部分产品手册》§1.2-1.3 + 《终稿·指令模版 4.19》P25-26「AI性格打分」: "
        "7 维 (活泼/理性/感性/计划/随性/脑洞/幽默) → MBTI 4 轴 (EI/NS/TF/JP) + 性格画像 summary. "
        "agent 创建时后台异步调用, 不阻塞 API 响应.",
        PERSONALITY_SCORING_PROMPT,
    ),
    PromptDefinition(
        "character.generation", "AI 背景单步生成", "初始化", "初始化",
        "《记忆部分产品手册》§1.4「AI背景生成」单步 LLM 调用: 输入姓名/性别/年龄/职业/MBTI/7 维 → "
        "输出全 5 维 L1 记忆 JSON (身份/偏好/生活/情绪/思维). "
        "占位符: {name} {gender_zh} {age} {career_title} {career_duties} {career_clients} "
        "{career_income} {career_social_value} {mbti_type} {mbti_summary} "
        "{ei}/{inv_ei}/{ns}/{inv_ns}/{tf}/{inv_tf}/{jp}/{inv_jp} "
        "{liveliness}/{rationality}/{sensitivity}/{planning}/{spontaneity}/{imagination}/{humor} "
        "—— 编辑时不要删除任何 {} 占位符, 也不要破坏 JSON 结构示意.",
        CHARACTER_GENERATION_PROMPT,
    ),
    PromptDefinition(
        "character.repair_missing_fields", "背景生成·缺字段补齐", "初始化", "初始化",
        "【工程附加】无 spec 出处. character profile 主路径输出 JSON 偶尔被 max_tokens 截断 → "
        "末尾字段缺失. 本 prompt 注入已生成的角色概要 + 缺字段清单, 让 LLM 只补缺. "
        "{persona_summary} / {missing_fields} 运行时填充.",
        CHARACTER_REPAIR_MISSING_FIELDS_PROMPT,
    ),

    # ── 记忆: 录入 (Memory §2) ──
    PromptDefinition(
        "memory.judgement_user", "记忆判断(用户)", "异步记忆", "记忆",
        "《记忆部分产品手册》§2.1.2: 小模型预筛, 二分类判断用户消息是否值得记忆 (记/不记).",
        MEMORY_JUDGEMENT_USER_PROMPT,
    ),
    PromptDefinition(
        "memory.judgement_ai", "AI 信息记忆判断", "异步记忆", "记忆",
        "《记忆部分产品手册》§2.2.2: 小模型预筛, 二分类判断 AI 自身消息是否值得进入自我记忆 (记/不记).",
        MEMORY_JUDGEMENT_AI_PROMPT,
    ),
    PromptDefinition(
        "memory.extraction_user", "用户长期记忆提取", "异步记忆", "记忆",
        "《记忆部分产品手册》§2.1.3: 拆分 + 五类分类 + importance 打分合并一步, "
        "仅抽用户侧, 存 memories_user (B 库).",
        MEMORY_EXTRACTION_USER_PROMPT,
    ),
    PromptDefinition(
        "memory.extraction_ai", "AI 自我长期记忆提取", "异步记忆", "记忆",
        "《记忆部分产品手册》§2.2.3: 拆分 + 五类分类 + importance 打分合并一步, "
        "仅抽 AI 自我记忆, 存 memories_ai (A 库).",
        MEMORY_EXTRACTION_AI_PROMPT,
    ),

    # ── 记忆: 矛盾处理 (Memory §4) ──
    PromptDefinition(
        "memory.contradiction_detection", "矛盾检测(交互)", "交互矛盾", "记忆",
        "《记忆部分产品手册》§4.1: 用户消息 vs L1 核心记忆的热路径矛盾判断, 中/强 relevance 时触发.",
        MEMORY_CONTRADICTION_DETECTION_PROMPT,
    ),
    PromptDefinition(
        "memory.contradiction_inquiry", "矛盾询问", "交互矛盾", "记忆",
        "《记忆部分产品手册》§4.2: 友好追问用户确认矛盾原因 (例:「我记得你之前说住在苏州, 是搬家了吗?」).",
        MEMORY_CONTRADICTION_INQUIRY_PROMPT,
    ),
    PromptDefinition(
        "memory.contradiction_analysis", "矛盾判断(回复分析)", "交互矛盾", "记忆",
        "《记忆部分产品手册》§4.3: 分析用户对追问的回答, 输出 变化 / 新增 / 错误 + 调整方案.",
        MEMORY_CONTRADICTION_ANALYSIS_PROMPT,
    ),
    PromptDefinition(
        "memory.contradiction_reply", "矛盾回复", "交互矛盾", "记忆",
        "《记忆部分产品手册》§4.5: 用户解释后 AI 自然把话题拉回正轨.",
        MEMORY_CONTRADICTION_REPLY_PROMPT,
    ),
    PromptDefinition(
        "memory.pairwise_contradiction", "L1 一致性扫描", "交互矛盾", "记忆",
        "《背景信息》§1.4: agent 创建期 L1 记忆两两扫描矛盾对, 自动 drop 低 importance 那条, 防止人设自相矛盾.",
        MEMORY_PAIRWISE_CONTRADICTION_PROMPT,
    ),

    # ── 记忆: 删除 (Memory §5) ──
    PromptDefinition(
        "memory.deletion_intent", "删除记忆意图识别", "异步记忆", "记忆",
        "《记忆部分产品手册》§5.1: 小模型识别用户是否在要求 AI 忘记/删除某条记忆 "
        "(「忘了吧」/「别记了」等关键词 + 语义).",
        DELETION_INTENT_PROMPT,
    ),

    # ── 情绪 (PAD) ──
    PromptDefinition(
        "emotion.extraction", "用户 PAD 值判断", "异步情绪", "情绪",
        "《记忆部分产品手册》§3.3 + 《终稿·指令模版 4.19》P26: 从用户消息抽 "
        "PAD (Pleasure / Arousal / Dominance) 三维值.",
        EMOTION_EXTRACTION_PROMPT,
    ),
    PromptDefinition(
        "emotion.ai_pad", "AI PAD 值判断", "前置情感", "情绪",
        "《终稿·第三部分 交互系统》§5.3 / §5.4 / §6.2: AI PAD 决定 emoji / sticker 概率与回复时机. "
        "本 prompt 从 AI 当前作息 + 上下文推 PAD 值.",
        AI_PAD_PROMPT,
    ),

    # ── 作息系统 (Part 1) ──
    PromptDefinition(
        "schedule.life_overview", "生活画像生成", "作息系统", "作息",
        "《终稿·第一部分》§2.1 + 《终稿·指令模版 4.19》P20「AI 生活画像」: "
        "200 字纯文本生活画像, 7 个中文性格维度由 MBTI 4 维反推近似值.",
        LIFE_OVERVIEW_PROMPT,
    ),
    PromptDefinition(
        "schedule.daily_schedule", "每日作息生成", "作息系统", "作息",
        "《终稿·第一部分》§2.1: 70% 概率使用, 生成当天作息 (空闲 / 忙碌 / 很忙碌 / 睡眠 4 状态).",
        DAILY_SCHEDULE_PROMPT,
    ),
    PromptDefinition(
        "schedule.daily_schedule_with_memory", "每日作息生成(带用户记忆)", "作息系统", "作息",
        "《终稿·第一部分》§2.1: 30% 概率使用, 作息中包含一项与用户记忆有关的事 "
        "(用户感知「AI 在意我」).",
        DAILY_SCHEDULE_WITH_USER_MEMORY_PROMPT,
    ),
    PromptDefinition(
        "schedule.daily_summary", "昨日生活总结(文本)", "作息系统", "作息",
        "《终稿·第一部分》§2.2: 昨日作息 + 调整 + 主动日志 → 200 字第一人称总结文本.",
        SCHEDULE_DAILY_SUMMARY_PROMPT,
    ),
    PromptDefinition(
        "schedule.daily_summary_memories", "昨日总结拆分打分", "作息系统", "记忆",
        "《终稿·第一部分》§2.3: 把总结文本拆分为记忆条目 + 五类分类 + importance 打分 (0-100).",
        SCHEDULE_DAILY_SUMMARY_MEMORIES_PROMPT,
    ),

    # ── 用户画像 ──
    PromptDefinition(
        "portrait.generation", "用户画像生成", "异步画像", "画像",
        "《记忆部分产品手册》§1.6 + 《终稿·指令模版 4.19》P23: "
        "200 字五段式用户画像 (基础认知 / 偏好 / 思维 / 情感 / 生活).",
        PORTRAIT_GENERATION_PROMPT,
    ),
    PromptDefinition(
        "portrait.update", "用户画像更新", "异步画像", "画像",
        "《记忆部分产品手册》§1.6 + 《终稿·指令模版 4.19》P23: "
        "根据记忆变化增量更新画像 (覆盖式重写, 不追加).",
        PORTRAIT_UPDATE_PROMPT,
    ),

    # ── 主动交流 (Part 4) ──
    PromptDefinition(
        "proactive.silence_plain", "沉默唤醒(无记忆)", "主动交流", "主动消息",
        "《终稿·第四部分 主动交流》§4.1 沉默唤醒: 不涉及记忆的轻打招呼. "
        "cold-start (亲密度 P1/P2) 阶段权重最高.",
        PROACTIVE_SILENCE_PLAIN_PROMPT,
    ),
    PromptDefinition(
        "proactive.silence_ai_memory", "沉默唤醒(AI记忆)", "主动交流", "主动消息",
        "《终稿·第四部分 主动交流》§4.1 沉默唤醒: 基于 AI 自身 L1/L2 记忆轻带一句.",
        PROACTIVE_SILENCE_AI_MEMORY_PROMPT,
    ),
    PromptDefinition(
        "proactive.silence_user_memory", "沉默唤醒(用户记忆)", "主动交流", "主动消息",
        "《终稿·第四部分 主动交流》§4.1 沉默唤醒: 基于用户 L1/L2 记忆轻带一句.",
        PROACTIVE_SILENCE_USER_MEMORY_PROMPT,
    ),
    PromptDefinition(
        "proactive.silence_schedule", "沉默唤醒(作息)", "主动交流", "主动消息",
        "《终稿·第四部分 主动交流》§4.1 沉默唤醒: 基于 AI 当前作息状态分享一句.",
        PROACTIVE_SILENCE_SCHEDULE_PROMPT,
    ),
    PromptDefinition(
        "proactive.memory_ai", "记忆主动(AI记忆)", "主动交流", "主动消息",
        "《终稿·第四部分 主动交流》§4.2 记忆主动: AI 有感而发分享自己的经历.",
        PROACTIVE_MEMORY_AI_PROMPT,
    ),
    PromptDefinition(
        "proactive.memory_user", "记忆主动(用户记忆)", "主动交流", "主动消息",
        "《终稿·第四部分 主动交流》§4.2 记忆主动: AI 忽然想起用户曾说过的事.",
        PROACTIVE_MEMORY_USER_PROMPT,
    ),
    PromptDefinition(
        "proactive.scheduled_scene", "定时情景(AI作息)", "主动交流", "主动消息",
        "《终稿·第四部分 主动交流》§1.3: 40% 概率定时情景, 基于 AI 当前作息状态分享一句.",
        PROACTIVE_SCHEDULED_SCENE_PROMPT,
    ),
    PromptDefinition(
        "proactive.decay_final", "衰减最后一次回复", "主动交流", "主动消息",
        "《终稿·第四部分 主动交流》§8 三级阶梯衰减: "
        "第三阶段 (n=6, 30 天内 1 次) 唯一一次温和告别, n≥7 后永久停止.",
        PROACTIVE_DECAY_FINAL_PROMPT,
    ),
    PromptDefinition(
        "proactive.first_greeting", "AI首次打招呼", "主动交流", "主动消息",
        "《终稿·第四部分 主动交流》§12 开场主动第一句话: "
        "用户首次进入聊天 / WS 建联且历史为空时 AI 的开场白.",
        PROACTIVE_FIRST_GREETING_PROMPT,
    ),
    PromptDefinition(
        "proactive.special_holiday", "特殊日期(节日)", "主动交流", "主动消息",
        "《终稿·第四部分 主动交流》§10 + 《终稿·第五部分 时间日期系统》§4.1: "
        "春节 / 元旦 / 静态节假日表命中时主动祝福.",
        PROACTIVE_SPECIAL_HOLIDAY_PROMPT,
    ),
    PromptDefinition(
        "proactive.special_birthday", "特殊日期(生日)", "主动交流", "主动消息",
        "《终稿·第四部分 主动交流》§10 + 《终稿·第五部分 时间日期系统》§4.1: "
        "用户 / AI 生日主动祝福 (从 memories 身份/生日 提取).",
        PROACTIVE_SPECIAL_BIRTHDAY_PROMPT,
    ),
    PromptDefinition(
        "proactive.special_reminder", "特殊日期(提醒)", "主动交流", "主动消息",
        "《终稿·第四部分 主动交流》§10 + 《终稿·第五部分 时间日期系统》§4.2: "
        "用户 / AI 提醒日期主动提醒 (memories 生活/提醒 子类 + occur_time 命中).",
        PROACTIVE_SPECIAL_REMINDER_PROMPT,
    ),
    PromptDefinition(
        "proactive.special_combined", "特殊日期(合并)", "主动交流", "主动消息",
        "《终稿·第四部分 主动交流》§10.3: 同日多个特殊日期合并到一条消息 (避免一日多次打扰).",
        PROACTIVE_SPECIAL_COMBINED_PROMPT,
    ),

    # ── 边界系统 (Part 3 §2) ──
    PromptDefinition(
        "boundary.attack_target", "攻击目标识别", "边界系统", "边界",
        "《终稿·第三部分 交互系统》§2.6 + 《终稿·指令模版 4.19》: "
        "攻击目标四分类 (攻击 AI / 攻击第三方 / 无负面 / 无目标脏话).",
        ATTACK_TARGET_PROMPT,
    ),
    PromptDefinition(
        "boundary.attack_level", "攻击级别识别", "边界系统", "边界",
        "《终稿·第三部分 交互系统》§2.4-§2.6: 攻击级别判定 K1 (轻) / K2 (中) / K3 (重).",
        ATTACK_LEVEL_PROMPT,
    ),
    PromptDefinition(
        "boundary.banned_word", "违禁词判断", "边界系统", "边界",
        "《终稿·第三部分 交互系统》§2.6: 判断消息是否含违禁内容 (谐音 / 缩写 / 涉黄涉暴等), "
        "语义级兜底关键词漏判.",
        BANNED_WORD_PROMPT,
    ),
    PromptDefinition(
        "boundary.positive_interaction", "正向互动判断", "边界系统", "边界",
        "《终稿·第三部分 交互系统》§2.5: 判断消息是否构成正向互动 (感谢/善意/积极反馈/正向情绪), "
        "用于门控 +20 耐心恢复, 防中性应答与普通问询滥发.",
        POSITIVE_INTERACTION_PROMPT,
    ),
    PromptDefinition(
        "boundary.light_attack_reply", "轻度攻击回复(K1)", "边界系统", "边界",
        "《终稿·第三部分 交互系统》§5.4: K1 级别回复——微微不舒服, 表达「以后别这么说」.",
        LIGHT_ATTACK_REPLY_PROMPT,
    ),
    PromptDefinition(
        "boundary.medium_attack_reply", "中度攻击回复(K2)", "边界系统", "边界",
        "《终稿·第三部分 交互系统》§5.4: K2 级别回复——认真但不过激, 让对方感受到情绪.",
        MEDIUM_ATTACK_REPLY_PROMPT,
    ),
    PromptDefinition(
        "boundary.severe_attack_reply", "重度攻击回复(K3)", "边界系统", "边界",
        "《终稿·第三部分 交互系统》§5.4: K3 级别回复——坚定明确表达底线,「再这样我们没法聊」.",
        SEVERE_ATTACK_REPLY_PROMPT,
    ),
    PromptDefinition(
        "boundary.medium_patience_reply", "中耐心回复", "边界系统", "边界",
        "《终稿·第三部分 交互系统》§2.3: 耐心 30-69 (medium) 区间的正常对话, 自然带出情绪余波.",
        MEDIUM_PATIENCE_REPLY_PROMPT,
    ),
    PromptDefinition(
        "boundary.low_patience_reply", "低耐心回复", "边界系统", "边界",
        "《终稿·第三部分 交互系统》§2.3: 耐心 1-29 (low) 区间的简短冷淡回复.",
        LOW_PATIENCE_REPLY_PROMPT,
    ),
    PromptDefinition(
        "boundary.final_warning", "最终警告", "边界系统", "边界",
        "【PM 后期补丁】《终稿·指令模版 4.19》【最终警告】段: "
        "攻击 AI 扣分后 patience<20 时覆写 K1/K2/K3. "
        "《终稿·第三部分 交互系统》§2.6 步骤 5.4 未列此档, 是 spec 定稿后追加的 PM 规则.",
        BOUNDARY_FINAL_WARNING_PROMPT,
    ),
    PromptDefinition(
        "boundary.blacklist_reply", "拉黑回复", "边界系统", "边界",
        "《终稿·第三部分 交互系统》§2.6: 耐心 ≤ 0 (blocked) 状态下的固定拒绝话术.",
        BLACKLIST_REPLY_PROMPT,
    ),
    PromptDefinition(
        "boundary.apology_reply", "道歉/承诺回复", "边界系统", "边界",
        "《终稿·第三部分 交互系统》§2.6.2.1: "
        "接受用户道歉 / 承诺后的和解回复 (拉黑态可解除, 耐心恢复至 70).",
        APOLOGY_REPLY_PROMPT,
    ),
    PromptDefinition(
        "boundary.apology", "道歉真诚度检测", "边界系统", "边界",
        "【工程扩展】拉黑态防 spam-unblock: sincerity ≥ 0.5 阈值闸门用户是否真心道歉. "
        "《终稿·第三部分 交互系统》§2.6.2.1 的 intent.unified 仅做意图分类, 不评估真诚度——"
        "本 prompt 是工程补充, 防止反复短道歉刷解封.",
        APOLOGY_PROMPT,
    ),

    # ── 意图识别 + 短路回复 (Part 3 §3) ──
    PromptDefinition(
        "intent.unified", "统一意图识别", "意图识别", "意图",
        "《终稿·第三部分 交互系统》§3.3 step 1: 多选意图识别 "
        "(8 类: 终结 / 计划查询 / 作息调整 / 当前状态 / 道歉承诺 / 删除 / 久远记忆 / 日常交流). "
        "用户消息 > 4 字符时使用.",
        INTENT_UNIFIED_PROMPT,
    ),
    PromptDefinition(
        "intent.split", "用户意图拆分", "意图识别", "意图",
        "《终稿·第三部分 交互系统》§3.3 step 2-3: "
        "多意图消息拆成 {label: fragment} 子意图片段, 主意图先回复, 其余依次 sub-intent 递归.",
        INTENT_SPLIT_PROMPT,
    ),
    PromptDefinition(
        "intent.end_reply", "终结意图回复", "意图处理", "意图",
        "《终稿·第三部分 交互系统》§3 终结意图: 用户表达想结束对话时的告别回复.",
        END_REPLY_PROMPT,
    ),
    PromptDefinition(
        "intent.schedule_query_reply", "计划查询回复", "意图处理", "意图",
        "《终稿·第三部分 交互系统》§3.2: 询问当前计划意图——回答 AI 当前 / 未来日程 (基于 daily_schedule 数据).",
        SCHEDULE_QUERY_REPLY_PROMPT,
    ),
    PromptDefinition(
        "intent.schedule_adjust_reply", "作息调整回复", "意图处理", "意图",
        "《终稿·第三部分 交互系统》§3.2: 作息调整意图——同意用户调整作息并输出 adjustment 写库.",
        SCHEDULE_ADJUST_REPLY_PROMPT,
    ),
    PromptDefinition(
        "intent.current_state_reply", "当前状态回复", "意图处理", "意图",
        "《终稿·第三部分 交互系统》§3.2: 询问当前状态意图——回答 AI 现在在做什么 (调用 get_current_status).",
        CURRENT_STATE_REPLY_PROMPT,
    ),
    PromptDefinition(
        "intent.deletion_confirm", "删除意图确认", "意图处理", "意图",
        "《记忆部分产品手册》§5: 用户表达删除意图后, 询问其具体想删除的内容.",
        DELETION_CONFIRM_PROMPT,
    ),
    PromptDefinition(
        "intent.deletion_reply", "删除回复", "意图处理", "意图",
        "《记忆部分产品手册》§5: 删除执行后表示已忘记并自然带过.",
        DELETION_REPLY_PROMPT,
    ),

    # ── 记忆: 检索 + 分级回复 (Memory §3) ──
    PromptDefinition(
        "memory.relevance", "回忆相关度判断", "日常交流", "记忆",
        "《记忆部分产品手册》§3.1: 判断当前消息与记忆的强 / 中 / 弱相关度. "
        "占位符 {message} (用户当前消息) + {context} (最近几轮对话). "
        "{context} 为 spec 偏离 (spec 仅输入单条用户消息), 工程加入用于解析省略式追问 "
        "(例「颜色呢?」), 见 CLAUDE.md 偏离表 #9.",
        MEMORY_RELEVANCE_PROMPT,
    ),
    PromptDefinition(
        "memory.l3_trigger", "L3 唤醒判断", "日常交流", "记忆",
        "《记忆部分产品手册》§3.2 step 2-3: 是否需要调用 L3 久远记忆 "
        "(输出: 不满纠正 / 请求更久 / 无).",
        L3_TRIGGER_PROMPT,
    ),
    PromptDefinition(
        "memory.weak_reply", "弱记忆回复", "日常交流", "记忆",
        "《记忆部分产品手册》§3.4: 弱相关度回复——不显式引用记忆, 只保持自然.",
        WEAK_MEMORY_REPLY_PROMPT,
    ),
    PromptDefinition(
        "memory.medium_reply", "中记忆回复", "日常交流", "记忆",
        "《记忆部分产品手册》§3.4: 中相关度回复——自然带过记忆中的相关事实.",
        MEDIUM_MEMORY_REPLY_PROMPT,
    ),
    PromptDefinition(
        "memory.strong_reply", "强记忆回复", "日常交流", "记忆",
        "《记忆部分产品手册》§3.4: 强相关度回复——直接基于记忆作出关心式或事实呼应回复.",
        STRONG_MEMORY_REPLY_PROMPT,
    ),
    PromptDefinition(
        "memory.l3_reply", "久远记忆回复", "日常交流", "记忆",
        "《记忆部分产品手册》§3.2 step 4: L3 久远记忆的模糊回复, 语气「我好像记得...」.",
        L3_MEMORY_REPLY_PROMPT,
    ),

    # ── 回复加工 (Part 3 §5 / §6) ──
    PromptDefinition(
        "reply.delay_explanation", "延迟解释回复", "异步回复", "回复加工",
        "《终稿·第三部分 交互系统》§6.5: "
        "实际延迟 ≥ 1 分钟时, 在主回复之前独立推送的延迟解释消息 (例「刚在忙...」).",
        DELAY_EXPLANATION_PROMPT,
    ),
    PromptDefinition(
        "reply.emotion_detection", "AI 语句情绪识别", "回复加工", "回复加工",
        "《终稿·第三部分 交互系统》§5 step 1: "
        "识别 AI 自身回复的情绪标签 (12 类) + 强度 (0-100), 用于 §5 emoji / sticker 概率计算.",
        AI_REPLY_EMOTION_PROMPT,
    ),
    PromptDefinition(
        "reply.split_2", "AI 语句拆分(2句)", "回复加工", "回复加工",
        "《终稿·第三部分 交互系统》§5.5: n=2 时把回复拆为 2 条自然语句. n 由均匀随机 1-3 决定.",
        REPLY_SPLIT_2_PROMPT,
    ),
    PromptDefinition(
        "reply.split_3", "AI 语句拆分(3句)", "回复加工", "回复加工",
        "《终稿·第三部分 交互系统》§5.5: n=3 时把回复拆为 3 条自然语句.",
        REPLY_SPLIT_3_PROMPT,
    ),
]

PROMPT_DEFINITION_MAP = {definition.key: definition for definition in PROMPT_DEFINITIONS}
