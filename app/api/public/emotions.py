"""Emotion API routes."""

from fastapi import APIRouter, Depends

from app.api.ownership import require_agent_owner
from app.db import db
from app.services.relationship.emotion import (
    emotion_to_tone,
    neutral_emotion,
    normalize_emotion_result,
)
from app.services.emoji import recommend_emoji

router = APIRouter(prefix="/emotions", tags=["emotions"])


async def _latest_assistant_reply_emotion(agent) -> dict:
    """Derive current display emotion from latest assistant reply metadata."""
    msg = await db.message.find_first(
        where={
            "role": "assistant",
            "conversation": {
                "agentId": agent.id,
                "userId": agent.userId,
            },
        },
        order={"createdAt": "desc"},
    )
    metadata = getattr(msg, "metadata", None) if msg else None
    if not isinstance(metadata, dict):
        return neutral_emotion(source="current_default")
    if not metadata.get("ai_emotion"):
        return neutral_emotion(source="current_default")
    return normalize_emotion_result(
        {
            "emotion": metadata.get("ai_emotion"),
            "intensity": metadata.get("emotion_intensity", 0),
            "confidence": metadata.get("emotion_confidence", 0.8),
        },
        source="latest_reply",
    )


@router.get("/{agent_id}/current")
async def get_current_emotion(agent=Depends(require_agent_owner)):
    """Return current coarse emotion state for compatibility.

    Runtime AI emotion-vector cache has been removed; current mood is derived
    from the latest assistant reply label when available.
    """
    emotion = await _latest_assistant_reply_emotion(agent)
    tone = emotion_to_tone(emotion)

    return {
        "agent_id": agent.id,
        "emotion": emotion["emotion"],
        "intensity": emotion["intensity"],
        "confidence": emotion["confidence"],
        "tone": tone,
    }


@router.get("/{agent_id}/timeline")
async def get_emotion_timeline(
    limit: int = 50,
    agent=Depends(require_agent_owner),
):
    """Get emotion history from message metadata for an agent.

    Reconstructs timeline from messages that have emotion metadata.
    """
    # Get conversations for this agent
    conversations = await db.conversation.find_many(
        where={"agentId": agent.id, "isDeleted": False},
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
                "emotion": emo.get("emotion", "中性"),
                "intensity": emo.get("intensity", 0),
                "confidence": emo.get("confidence", 0.0),
                "message_preview": msg.content[:80],
            })

    return timeline


@router.post("/emoji/recommend")
async def recommend_emoji_api(
    emotion: str | None = None,
    primary_emotion: str | None = None,
    count: int = 3,
):
    """推荐表情。"""
    emojis = recommend_emoji(primary_emotion or emotion, count)
    return {"emojis": emojis}
