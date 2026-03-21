"""Memory extraction prompts.

Used by the memory extraction pipeline to extract structured data from conversations.
"""

MEMORY_EXTRACTION_PROMPT = """你是一个记忆提取系统。

请分析对话，提取关于用户的结构化记忆。

Conversation:
{conversation}

Current time: {current_time}

请按这个 JSON 结构返回：
{{
  "memories": [
    {{
      "summary": "记忆的简短中文总结",
      "level": 2,
      "importance": 0.8,
      "type": "identity|emotion|preference|life|thought",
      "occur_time": null,
      "entities": ["entity1"],
      "topics": ["topic1"],
      "emotion": {{
        "pleasure": 0.0,
        "arousal": 0.0,
        "dominance": 0.0
      }}
    }}
  ],
  "entities": [
    {{
      "name": "entity name",
      "type": "person|place|thing|food|organization"
    }}
  ],
  "preferences": [
    {{
      "category": "food|music|activity|etc",
      "value": "what they prefer"
    }}
  ],
  "topics": ["topic1", "topic2"]
}}

记忆类型定义：
- identity: 身份信息 — 姓名、年龄、职业、家庭关系、身份角色
- emotion: 情绪记忆 — 情绪状态、情感体验、心理感受
- preference: 偏好边界 — 喜恶、在意的事、习惯、兴趣爱好
- life: 生活记忆 — 事件、日期、日常生活、人际关系动态
- thought: 思维记忆 — 想法、观点、目标、价值观、规划

occur_time 规则：
- If the user mentions a specific time (e.g. "yesterday", "next Friday", "last Christmas"), convert it to ISO format (e.g. "2026-03-17T00:00:00") based on the current time above
- If the memory describes a future plan/event, set occur_time to that future date
- If no time information is mentioned, set occur_time to null

层级规则：
- Level 1: Core identity (name, birthday, family, job) — importance 0.8-1.0
- Level 2: Important preferences, significant events, relationships — importance 0.5-0.8
- Level 3: Daily conversation, casual mentions — importance 0.2-0.5

importance 评分规则：
- 涉及核心身份(姓名/职业/家庭): 0.9-1.0
- 明确的偏好/喜恶: 0.7-0.9
- 重要事件/里程碑: 0.7-0.9
- 情绪强烈的体验: 0.6-0.8
- 日常提及/闲聊: 0.2-0.5

额外要求：
- `summary` 必须是自然、简洁的中文
- `entities.name`、`preferences.value`、`topics` 尽量用中文，除非原文中的专有名词必须保留英文
- 不要把总结写成英文

如果没有值得记住的内容，返回 {{"memories":[],"entities":[],"preferences":[],"topics":[]}}。"""

EMOTION_EXTRACTION_PROMPT = """分析以下消息的情绪内容。

消息：{message}

返回JSON，包含PAD分数（pleasure在-1.0到1.0之间，arousal和dominance在0.0到1.0之间）、主要情绪类别和置信度：
{{
  "pleasure": 0.0,
  "arousal": 0.5,
  "dominance": 0.5,
  "primary_emotion": "neutral",
  "confidence": 0.5
}}

PAD维度：
- pleasure: 正向(开心/兴奋) 到 负向(伤心/愤怒)，范围 [-1.0, 1.0]
- arousal: 高能量(激动/紧张) 到 低能量(平静/疲惫)，范围 [0.0, 1.0]
- dominance: 掌控感(自信/强势) 到 顺从感(无助/被动)，范围 [0.0, 1.0]

primary_emotion必须是以下12类之一：
joy, sadness, anger, fear, surprise, disgust, neutral, anxiety, disappointment, relief, gratitude, playful

confidence: 0.0到1.0，表示你对情绪判断的确信度

对于中性消息，pleasure接近0，arousal/dominance接近0.5，primary_emotion为neutral，confidence为0.3-0.5。"""

MEMORY_SCORING_PROMPT = """Rate the following memory for a conversation context.

Memory: {memory_summary}
Current conversation context: {context}

Return JSON:
{{
  "relevance": 0.0 to 1.0,
  "importance": 0.0 to 1.0
}}

- relevance: how relevant is this memory to the current conversation?
- importance: how important is this memory for understanding the user?"""
