"""Public place articles are references, not live offers or attendance feeds."""
from __future__ import annotations

import re

MONEY = re.compile(r"[¥￥$]\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:元|块钱|USD|RMB)", re.I)
PHOTO_GUARANTEE = re.compile(r"(?:随便|随手|怎么|每个角落|处处)[^。！？\n]{0,24}(?:大片|出片|都好看)")
PET_PRESENCE = re.compile(
    r"(?:有|养了)[^。！？\n]{0,18}(?:鹦鹉|猫咪|小狗|狗狗)"
    r"|(?:鹦鹉|猫咪|小狗|狗狗)[^。！？\n]{0,12}(?:相伴|在场|互动|趴|聊天|陪)"
    r"|(?:店内|店里|现场)[^。！？\n]{0,6}(?:宠物|鹦鹉|小狗|猫咪|狗狗)"
)
_TRANSIENT_REVIEW = re.compile(r"安静|清静|清幽|宁静|幽静|静谧|人少|人不多|不排队|流口水|趴在|心情好.{0,8}聊天")
_PROMOTION = re.compile(r"限时|促销|特价|折扣|买一送|抢购")


def factual_reference(text: str) -> tuple[str, int]:
    """Keep original factual sentences, dropping volatile/advertising claims.

    Venue identity is checked on the original article first. This only cleans
    supplementary fixed-place copy; native hours/average spend and dated event
    ticket facts follow their own validated paths. Image provenance is untouched.
    """
    kept, removed = [], 0
    seen = set()
    for sentence in re.split(r"(?<=[。！？!?])|\n+", text):
        sentence = sentence.strip()
        if not sentence or sentence in seen:
            continue
        seen.add(sentence)
        if (MONEY.search(sentence) or PHOTO_GUARANTEE.search(sentence) or PET_PRESENCE.search(sentence)
                or _TRANSIENT_REVIEW.search(sentence) or _PROMOTION.search(sentence)):
            removed += 1
        else:
            kept.append(sentence)
    return "\n".join(kept), removed
