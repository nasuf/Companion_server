"""Summarizer prompt templates.

Separated from summarizer.py for maintainability and reuse.
"""

LAYER1_REVIEW_PROMPT = """You are a conversation summarizer.

Summarize the following conversation history in 200-300 words.
Focus on: key topics discussed, important information shared, relationship dynamics.

Conversation:
{conversation}

Summary:"""

LAYER2_DISTILLATION_PROMPT = """You are a memory distillation system.

Given the user's stored memories and the current message, extract the most relevant points in 150-200 words.
Focus on: what the user cares about, relevant past context, useful background.

Stored memories:
{memories}

Current message: {current_message}

Key points:"""

LAYER3_STATE_PROMPT = """You are a conversation state analyzer.

Analyze the recent messages and current message. In 100-150 words, describe:
1. Current emotional tone
2. Active topic/subject
3. User's likely intent
4. Suggested response approach

Recent messages:
{recent}

Current message: {current_message}

Analysis:"""
