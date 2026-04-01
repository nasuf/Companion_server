"""Emotion API routes.

Provides endpoints for querying AI emotion state and emotion timeline.
"""

from fastapi import APIRouter, HTTPException

from app.db import db
from app.services.relationship.emotion import get_ai_emotion, emotion_to_tone
from app.services.emoji import recommend_emoji

router = APIRouter(prefix="/emotions", tags=["emotions"])


@router.get("/{agent_id}/current")
async def get_current_emotion(agent_id: str):
    """Get the current emotion state for an AI agent."""
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent or getattr(agent, "status", "active") != "active":
        raise HTTPException(status_code=404, detail="Agent not found")

    emotion = await get_ai_emotion(agent_id)
    tone = emotion_to_tone(emotion)

    return {
        "agent_id": agent_id,
        "pleasure": emotion["pleasure"],
        "arousal": emotion["arousal"],
        "dominance": emotion["dominance"],
        "tone": tone,
    }


@router.get("/{agent_id}/timeline")
async def get_emotion_timeline(agent_id: str, limit: int = 50):
    """Get emotion history from message metadata for an agent.

    Reconstructs timeline from messages that have emotion metadata.
    """
    agent = await db.aiagent.find_unique(where={"id": agent_id})
    if not agent or getattr(agent, "status", "active") != "active":
        raise HTTPException(status_code=404, detail="Agent not found")

    # Get conversations for this agent
    conversations = await db.conversation.find_many(
        where={"agentId": agent_id, "isDeleted": False},
    )
    conv_ids = [c.id for c in conversations]

    if not conv_ids:
        return []

    # Get messages with metadata (emotion data stored during chat)
    messages = await db.message.find_many(
        where={
            "conversationId": {"in": conv_ids},
            "role": "user",
        },
        order={"createdAt": "desc"},
        take=limit,
    )

    timeline = []
    for msg in messages:
        metadata = msg.metadata if msg.metadata else {}
        if isinstance(metadata, dict) and "emotion" in metadata:
            emo = metadata["emotion"]
            timeline.append({
                "timestamp": str(msg.createdAt),
                "pleasure": emo.get("pleasure", 0.0),
                "arousal": emo.get("arousal", 0.0),
                "dominance": emo.get("dominance", 0.0),
                "message_preview": msg.content[:80],
            })

    return timeline


@router.post("/emoji/recommend")
async def recommend_emoji_api(
    pleasure: float = 0.0,
    arousal: float = 0.0,
    primary_emotion: str | None = None,
    count: int = 3,
):
    """推荐表情。"""
    emojis = recommend_emoji(pleasure, arousal, primary_emotion, count)
    return {"emojis": emojis}
