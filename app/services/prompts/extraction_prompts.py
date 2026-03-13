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

EMOTION_EXTRACTION_PROMPT = """Analyze the emotional content of the following message.

Message: {message}

Return JSON with VAD (Valence-Arousal-Dominance) scores, each between -1.0 and 1.0:
{{
  "valence": 0.0,
  "arousal": 0.0,
  "dominance": 0.0
}}

- valence: positive (happy, excited) to negative (sad, angry)
- arousal: high energy to low energy
- dominance: feeling in control to feeling submissive

For neutral messages, return values near 0."""

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
