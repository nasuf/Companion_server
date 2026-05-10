from pydantic import BaseModel
from fastapi import APIRouter

from app.services.sticker import recommend_sticker

router = APIRouter(prefix="/stickers", tags=["stickers"])


class StickerRecommendRequest(BaseModel):
    target_emotion: dict  # {"emotion": "高兴", "intensity": 70}


class StickerRecommendResponse(BaseModel):
    code: int = 0
    data: dict | None = None


@router.post("/recommend", response_model=StickerRecommendResponse)
async def recommend(req: StickerRecommendRequest):
    """PRD §5.7.2.4: 表情包推荐接口。"""
    te = req.target_emotion
    result = await recommend_sticker(
        primary_emotion=te.get("emotion") or te.get("primary_emotion"),
        intensity=te.get("intensity", 50),
    )
    if result:
        return StickerRecommendResponse(data={
            "emoji_id": str(result["id"]),
            "emoji_url": result["url"],
            "match_score": result["match_score"],
        })
    return StickerRecommendResponse(data=None)
