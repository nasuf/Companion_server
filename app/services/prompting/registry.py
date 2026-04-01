"""Prompt registry for admin management and runtime lookup."""

from __future__ import annotations

from dataclasses import dataclass

from app.services.prompting.defaults import (
    APOLOGY_PROMPT,
    ATTACK_INTENT_PROMPT,
    COMPRESS_PROMPT,
    CONFLICT_DETECTION_PROMPT,
    CONSISTENCY_RULES_PROMPT,
    CONSOLIDATE_SUMMARIZE_PROMPT,
    DAILY_SCHEDULE_PROMPT,
    DELETION_INTENT_PROMPT,
    EMOTION_EXTRACTION_PROMPT,
    EMOTION_INSTRUCTION_PROMPT,
    FAST_FACT_PROMPT,
    LAYER1_REVIEW_PROMPT,
    LAYER2_DISTILLATION_PROMPT,
    LAYER3_STATE_PROMPT,
    LIFE_OVERVIEW_PROMPT,
    MEMORY_EXTRACTION_PROMPT,
    MEMORY_INSTRUCTION_PROMPT,
    MEMORY_SCORING_PROMPT,
    PERSONALITY_RULES_PROMPT,
    PORTRAIT_GENERATION_PROMPT,
    PORTRAIT_UPDATE_PROMPT,
    PROACTIVE_PROMPT,
    QUERY_ANALYZER_PROMPT,
    RESPONSE_INSTRUCTION_PROMPT,
    SCHEDULE_REVIEW_PROMPT,
    SELF_MEMORY_PROMPT,
    SEVERITY_PROMPT,
    SYSTEM_BASE_PROMPT,
)


@dataclass(frozen=True)
class PromptDefinition:
    key: str
    title: str
    stage: str
    category: str
    description: str
    default_text: str


PROMPT_DEFINITIONS = [
    PromptDefinition("chat.system_base", "聊天核心规则", "聊天热路径", "聊天", "系统人格锚点。", SYSTEM_BASE_PROMPT),
    PromptDefinition("chat.response_instruction", "聊天回复规则", "聊天热路径", "聊天", "条数、字数、语气与回复方式。", RESPONSE_INSTRUCTION_PROMPT),
    PromptDefinition("chat.personality_rules", "人格一致性规则", "聊天热路径", "聊天", "约束人格和表达风格。", PERSONALITY_RULES_PROMPT),
    PromptDefinition("chat.consistency_rules", "对话一致性规则", "聊天热路径", "聊天", "防止重复提问和自相矛盾。", CONSISTENCY_RULES_PROMPT),
    PromptDefinition("chat.emotion_instruction", "情绪注入规则", "聊天热路径", "聊天", "控制情绪如何影响语气。", EMOTION_INSTRUCTION_PROMPT),
    PromptDefinition("chat.memory_instruction", "记忆注入规则", "聊天热路径", "聊天", "控制记忆上下文提示语。", MEMORY_INSTRUCTION_PROMPT),
    PromptDefinition("memory.fast_fact", "热路径事实提取", "聊天热路径", "记忆", "同步提取 working memory。", FAST_FACT_PROMPT),
    PromptDefinition("memory.extraction", "长期记忆提取", "异步记忆", "记忆", "对话后抽取 L1/L2/L3 记忆。", MEMORY_EXTRACTION_PROMPT),
    PromptDefinition("memory.scoring", "记忆相关性评分", "检索重排", "记忆", "LLM 辅助记忆相关性和重要性评分。", MEMORY_SCORING_PROMPT),
    PromptDefinition("memory.conflict_detection", "记忆冲突检测", "异步记忆", "记忆", "检测新旧核心记忆冲突。", CONFLICT_DETECTION_PROMPT),
    PromptDefinition("memory.deletion_intent", "删除记忆意图识别", "异步记忆", "记忆", "识别用户删除记忆请求。", DELETION_INTENT_PROMPT),
    PromptDefinition("memory.query_analyzer", "检索意图分析", "检索前置", "记忆", "判断检索类型与实体。", QUERY_ANALYZER_PROMPT),
    PromptDefinition("memory.compression", "记忆压缩总结", "定时压缩", "记忆", "多条记忆压缩为总结。", COMPRESS_PROMPT),
    PromptDefinition("memory.consolidation", "记忆整合总结", "定时整合", "记忆", "相似记忆整合为长期记忆。", CONSOLIDATE_SUMMARIZE_PROMPT),
    PromptDefinition("emotion.extraction", "情绪提取", "异步情绪", "情绪", "用户消息 PAD 提取。", EMOTION_EXTRACTION_PROMPT),
    PromptDefinition("summarizer.layer1_review", "摘要层1-对话回顾", "异步摘要", "摘要", "30 条消息回顾总结。", LAYER1_REVIEW_PROMPT),
    PromptDefinition("summarizer.layer2_distillation", "摘要层2-记忆提炼", "异步摘要", "摘要", "记忆与当前消息提炼。", LAYER2_DISTILLATION_PROMPT),
    PromptDefinition("summarizer.layer3_state", "摘要层3-状态分析", "异步摘要", "摘要", "当前情绪/话题/意图分析。", LAYER3_STATE_PROMPT),
    PromptDefinition("schedule.life_overview", "生活画像生成", "作息系统", "作息", "生成 AI 长期生活画像。", LIFE_OVERVIEW_PROMPT),
    PromptDefinition("schedule.daily_schedule", "每日作息生成", "作息系统", "作息", "生成当天具体作息安排。", DAILY_SCHEDULE_PROMPT),
    PromptDefinition("schedule.review", "作息回顾总结", "作息系统", "作息", "把一天经历沉淀成 AI 自我记忆。", SCHEDULE_REVIEW_PROMPT),
    PromptDefinition("portrait.generation", "用户画像生成", "异步画像", "画像", "从记忆生成用户画像。", PORTRAIT_GENERATION_PROMPT),
    PromptDefinition("portrait.update", "用户画像更新", "异步画像", "画像", "根据变化增量更新画像。", PORTRAIT_UPDATE_PROMPT),
    PromptDefinition("proactive.message", "主动消息生成", "主动交流", "主动消息", "生成 AI 主动发起的消息。", PROACTIVE_PROMPT),
    PromptDefinition("boundary.attack_intent", "攻击意图识别", "边界系统", "边界", "判断攻击意图类别。", ATTACK_INTENT_PROMPT),
    PromptDefinition("boundary.severity", "攻击严重度评估", "边界系统", "边界", "评估攻击消息严重度。", SEVERITY_PROMPT),
    PromptDefinition("boundary.apology", "道歉识别", "边界系统", "边界", "识别是否为有效道歉。", APOLOGY_PROMPT),
    PromptDefinition("self_memory.daily", "AI 自我记忆生成", "AI 自我记忆", "自我记忆", "从每日对话生成 AI 视角记忆。", SELF_MEMORY_PROMPT),
]

PROMPT_DEFINITION_MAP = {definition.key: definition for definition in PROMPT_DEFINITIONS}
