from pydantic import BaseModel
from fastapi import APIRouter

from app.services.sticker import recommend_sticker

router = APIRouter(prefix="/stickers", tags=["stickers"])


class StickerRecommendRequest(BaseModel):
    user_id: str
    target_emotion: dict  # {"pleasure": 0.6, "arousal": 0.6, "dominance": 0.5, "primary_emotion": "高兴"}


class StickerRecommendResponse(BaseModel):
    code: int = 0
    data: dict | None = None


@router.post("/recommend", response_model=StickerRecommendResponse)
async def recommend(req: StickerRecommendRequest):
    """PRD §5.7.2.4: 表情包推荐接口。"""
    te = req.target_emotion
    result = await recommend_sticker(
        pleasure=te.get("pleasure", 0.0),
        arousal=te.get("arousal", 0.0),
        dominance=te.get("dominance", 0.5),
        primary_emotion=te.get("primary_emotion"),
    )
    if result:
        return StickerRecommendResponse(data={
            "emoji_id": str(result["id"]),
            "emoji_url": result["url"],
            "match_score": result["match_score"],
        })
    return StickerRecommendResponse(data=None)
