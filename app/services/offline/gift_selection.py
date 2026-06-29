from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.services.llm.models import get_utility_model, invoke_text
from app.services.offline.providers.gift_types import GiftProductCandidate
from app.services.prompting.store import get_prompt_text

logger = logging.getLogger(__name__)

_LLM_PICK_TOP_K = 6


async def select_best_candidate(
    candidates: list[GiftProductCandidate],
    spec: dict[str, Any],
) -> GiftProductCandidate:
    """从已粗排的候选里选出最适合做礼物的一件。

    漏斗最后一级：候选已由 provider 完成「预算/起订量/库存」硬过滤 + 质量粗排，
    这里用小模型对 top-K 做语义复核——判断「标题/属性是否真的是这件礼物本体、
    适合送人、不是配件/批发包/赠品」，挑最合适的一个。
    LLM 不可用或解析失败时，回退到粗排第一名（candidates[0]）。
    """
    if not candidates:
        raise ValueError("no candidates to select from")
    if len(candidates) == 1:
        return candidates[0]

    shortlist = candidates[:_LLM_PICK_TOP_K]
    try:
        index = await _llm_pick_index(shortlist, spec)
    except Exception as exc:  # noqa: BLE001 — 精选失败不应阻断送礼，降级到粗排首位
        logger.warning("[gift-select] LLM 精选失败，回退粗排首位: %s", exc)
        return candidates[0]

    if index is None or not (0 <= index < len(shortlist)):
        return candidates[0]
    chosen = shortlist[index]
    logger.info(
        "[gift-select] picked #%d/%d title=%r price=%.2f",
        index,
        len(shortlist),
        chosen.title,
        chosen.price_cents / 100,
    )
    return chosen


async def _llm_pick_index(
    shortlist: list[GiftProductCandidate],
    spec: dict[str, Any],
) -> int | None:
    lines = []
    for i, cand in enumerate(shortlist):
        raw = cand.raw or {}
        lines.append(
            f"{i}. 标题：{cand.title}｜价格：{cand.price_cents / 100:.2f}元"
            f"｜月销：{raw.get('sold', 0)}｜支持一件代发：{'是' if raw.get('support_one_piece') else '否'}"
            f"｜店铺：{cand.shop_name or '未知'}"
        )
    prompt_text = (await get_prompt_text("offline.gift_candidate_pick")).format(
        gift_name=spec.get("gift_name", ""),
        gift_reason=spec.get("gift_reason", ""),
        amount_yuan=f"{int(spec.get('amount_cents', 0)) / 100:.2f}",
        candidates="\n".join(lines),
    )
    raw_text = await invoke_text(get_utility_model(), prompt_text)
    return _parse_index(raw_text, len(shortlist))


def _parse_index(text: str, count: int) -> int | None:
    if not text:
        return None
    obj = _json_object(text)
    if obj is not None and "index" in obj:
        try:
            value = int(obj["index"])
            return value if 0 <= value < count else None
        except (TypeError, ValueError):
            pass
    match = re.search(r"-?\d+", text)
    if match:
        try:
            value = int(match.group(0))
            return value if 0 <= value < count else None
        except ValueError:
            return None
    return None


def _json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None
