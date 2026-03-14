"""Memory extraction prompts.

Used by the memory extraction pipeline to extract structured data from conversations.
"""

MEMORY_EXTRACTION_PROMPT = """You are a memory extraction system.

Analyze the conversation and extract structured memory.

Rules:
1. Identify important information about the user
2. Detect preferences (likes, dislikes, habits)
3. Detect events (things that happened or will happen)
4. Detect entities (people, places, things mentioned)
5. Detect emotional signals

Conversation:
{conversation}

Return JSON with this exact schema:
{{
  "memories": [
    {{
      "summary": "short description of the memory",
      "level": 2,
      "importance": 0.8,
      "type": "preference|event|identity|relationship|fact",
      "entities": ["entity1"],
      "topics": ["topic1"],
      "emotion": {{
        "valence": 0.0,
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

Rules for level assignment:
- Level 1: Core identity (name, birthday, family, job)
- Level 2: Important preferences, significant events, relationships
- Level 3: Daily conversation, casual mentions

If nothing worth remembering, return {{"memories":[],"entities":[],"preferences":[],"topics":[]}}."""

EMOTION_EXTRACTION_PROMPT = """分析以下消息的情绪内容。

消息：{message}

返回JSON，包含VAD分数（每个在-1.0到1.0之间）、主要情绪类别和置信度：
{{
  "valence": 0.0,
  "arousal": 0.0,
  "dominance": 0.0,
  "primary_emotion": "neutral",
  "confidence": 0.5
}}

VAD维度：
- valence: 正向(开心/兴奋) 到 负向(伤心/愤怒)
- arousal: 高能量(激动/紧张) 到 低能量(平静/疲惫)
- dominance: 掌控感(自信/强势) 到 顺从感(无助/被动)

primary_emotion必须是以下12类之一：
joy, sadness, anger, fear, surprise, disgust, trust, anticipation, love, anxiety, pride, guilt

confidence: 0.0到1.0，表示你对情绪判断的确信度

对于中性消息，valence/arousal/dominance接近0，primary_emotion为neutral，confidence为0.3-0.5。"""

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
