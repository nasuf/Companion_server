"""后台礼物推荐：基于用户画像标签与记忆，一次生成多个礼物候选供运营挑选。

可追溯是这里的核心：prompt 里每条标签 / 记忆都带编号（T1 / M1），模型必须用编号
引用依据；服务端逐条校验编号并回填原文。引用不存在的编号会被丢弃，一个有效依据都
没有的候选标记为 grounded=False —— 运营看到的「依据」都是真实存在的用户信息。
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import HTTPException

from app.observability.events import EVT_GIFT_RECOMMENDATIONS
from app.services.llm.models import get_chat_model, invoke_text
from app.services.offline import gift_admin_repository as admin_repo
from app.services.offline import repository as repo
from app.services.offline.chat_emit import offline_trace
from app.services.offline.gift_selection import parse_json_object
from app.services.prompting.store import get_prompt_text

logger = logging.getLogger(__name__)

_TAG_LIMIT = 12
_MEMORY_TEXT_LIMIT = 120
_EXCLUDE_LIMIT = 60
_DEFAULT_BUDGET_LABEL = "未指定，按 30-150 元的小礼物把握"
_REF_PATTERN = re.compile(r"^\s*([MT])\s*(\d{1,3})\s*$", re.IGNORECASE)


async def generate_recommendations(
    *,
    user_id: str,
    workspace: dict[str, Any] | None,
    budget_min_yuan: int | None,
    budget_max_yuan: int | None,
    exclude_names: list[str],
    count: int,
) -> dict[str, Any]:
    workspace_id = workspace.get("workspace_id") if workspace else None
    agent_id = workspace.get("agent_id") if workspace else None
    tags = await repo.list_user_tags(user_id, workspace_id, agent_id=agent_id, limit=_TAG_LIMIT)
    memories = await admin_repo.user_memories_for_gift(user_id, workspace_id)
    if not tags and not memories:
        # 真实空数据：不能拿一份「什么依据都没有」的推荐冒充分析结果。
        raise HTTPException(status_code=422, detail="该用户还没有可用于分析的画像标签或记忆，暂时无法推荐")

    sent_names = await admin_repo.gift_names_for_user(user_id)
    exclude = _dedupe_names([*sent_names, *exclude_names])[:_EXCLUDE_LIMIT]
    budget_label = _budget_label(budget_min_yuan, budget_max_yuan)
    prompt = (await get_prompt_text("offline.gift_admin_recommendations")).format(
        count=count,
        tags="\n".join(f"T{i}. {tag}" for i, tag in enumerate(tags, 1)) or "暂无",
        memories="\n".join(_memory_line(i, m) for i, m in enumerate(memories, 1)) or "暂无",
        budget=budget_label,
        exclude="\n".join(f"- {name}" for name in exclude) or "暂无",
    )

    async with offline_trace(
        "gift_admin_recommendations",
        conversation_id=workspace.get("conversation_id") if workspace else None,
        agent_id=agent_id,
        user_id=user_id,
    ) as tracer:
        try:
            raw = await invoke_text(get_chat_model(), prompt)
        except Exception as exc:
            logger.warning("[gift-admin] recommendation LLM failed user_id=%s: %s", user_id, exc)
            raise HTTPException(status_code=502, detail="礼物推荐生成失败，请稍后重试") from exc
        trace_id = tracer.safe_trace_id

    parsed = parse_json_object(raw) or {}
    items = parsed.get("candidates")
    if not isinstance(items, list):
        raise HTTPException(status_code=502, detail="模型没有返回可解析的推荐结果，请重试")

    candidates, dropped = _build_candidates(
        items,
        tags=tags,
        memories=memories,
        exclude=exclude,
        count=count,
        budget=(budget_min_yuan, budget_max_yuan),
    )
    logger.info(
        "[gift-admin] recommendations user_id=%s returned=%d dropped=%d",
        user_id,
        len(candidates),
        dropped,
        extra={
            "event": EVT_GIFT_RECOMMENDATIONS,
            "user_id": user_id,
            "returned": len(candidates),
            "dropped": dropped,
            "grounded": sum(1 for c in candidates if c["grounded"]),
        },
    )
    if not candidates:
        raise HTTPException(status_code=502, detail="这一批推荐都与已送礼物重复或格式不完整，请再换一批")
    return {
        "batch_id": uuid4().hex,
        "generated_at": datetime.now(UTC).isoformat(),
        "workspace_id": workspace_id,
        "budget_label": budget_label,
        "tags": tags,
        "memory_count": len(memories),
        "candidates": candidates,
        "dropped_count": dropped,
        "trace_id": trace_id,
    }


def _build_candidates(
    items: list[Any],
    *,
    tags: list[str],
    memories: list[dict[str, Any]],
    exclude: list[str],
    count: int,
    budget: tuple[int | None, int | None],
) -> tuple[list[dict[str, Any]], int]:
    seen = {_name_key(name) for name in exclude}
    candidates: list[dict[str, Any]] = []
    dropped = 0
    for item in items:
        if len(candidates) >= count:
            break
        candidate = _candidate(item, tags=tags, memories=memories, budget=budget)
        if candidate is None or _name_key(candidate["gift_name"]) in seen:
            dropped += 1
            continue
        seen.add(_name_key(candidate["gift_name"]))
        candidates.append(candidate)
    return candidates, dropped


def _candidate(
    item: Any,
    *,
    tags: list[str],
    memories: list[dict[str, Any]],
    budget: tuple[int | None, int | None],
) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    name = _text(item.get("gift_name"), 40)
    reason = _text(item.get("reason"), 160)
    note = _text(item.get("gift_note"), 180)
    if not name or not reason:
        return None
    evidence = _resolve_evidence(item.get("evidence"), tags=tags, memories=memories)
    price_min, price_max = _price_range(item.get("price_min_yuan"), item.get("price_max_yuan"), budget)
    keywords = item.get("search_keywords")
    return {
        "gift_name": name,
        "search_keywords": [
            kw for kw in (_text(k, 30) for k in (keywords if isinstance(keywords, list) else [])) if kw
        ][:4] or [name],
        "price_min_yuan": price_min,
        "price_max_yuan": price_max,
        "reason": reason,
        "gift_note": note,
        "evidence": evidence,
        "grounded": bool(evidence),
    }


def _resolve_evidence(
    raw: Any,
    *,
    tags: list[str],
    memories: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    used: set[str] = set()
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict):
            continue
        match = _REF_PATTERN.match(str(entry.get("ref") or ""))
        if not match:
            continue
        kind, index = match.group(1).upper(), int(match.group(2))
        ref = f"{kind}{index}"
        if ref in used:
            continue
        feature = _text(entry.get("feature"), 80)
        if kind == "T" and 1 <= index <= len(tags):
            resolved.append({
                "ref": ref,
                "feature": feature or tags[index - 1],
                "source_type": "tag",
                "source_text": tags[index - 1],
                "source_category": "画像标签",
                "source_date": None,
            })
        elif kind == "M" and 1 <= index <= len(memories):
            memory = memories[index - 1]
            resolved.append({
                "ref": ref,
                "feature": feature or memory["content"][:80],
                "source_type": "memory",
                "source_text": memory["content"],
                "source_category": memory.get("category"),
                "source_date": memory.get("updated_at"),
            })
        else:
            continue
        used.add(ref)
    return resolved[:5]


def _price_range(
    raw_min: Any,
    raw_max: Any,
    budget: tuple[int | None, int | None],
) -> tuple[int | None, int | None]:
    low, high = _positive_int(raw_min), _positive_int(raw_max)
    if low and high and low > high:
        low, high = high, low
    budget_min, budget_max = budget
    if budget_max and high and high > budget_max:
        high = budget_max
    if budget_min and low and low < budget_min:
        low = budget_min
    if low and high and low > high:
        low = high
    return low, high


def _budget_label(budget_min: int | None, budget_max: int | None) -> str:
    if budget_min and budget_max:
        low, high = sorted((budget_min, budget_max))
        return f"{low}-{high} 元"
    if budget_max:
        return f"不超过 {budget_max} 元"
    if budget_min:
        return f"不低于 {budget_min} 元，控制在合理的小礼物范围"
    return _DEFAULT_BUDGET_LABEL


def _memory_line(index: int, memory: dict[str, Any]) -> str:
    meta = " · ".join(
        part for part in (memory.get("category"), (memory.get("updated_at") or "")[:10]) if part
    )
    text = memory["content"].replace("\n", " ")[:_MEMORY_TEXT_LIMIT]
    return f"M{index}. [{meta}] {text}" if meta else f"M{index}. {text}"


def _dedupe_names(names: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        key = _name_key(name)
        if key and key not in seen:
            seen.add(key)
            result.append(name.strip())
    return result


def _name_key(name: str) -> str:
    return re.sub(r"\s+", "", name or "").lower()


def _text(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit] if isinstance(value, (str, int, float)) else ""


def _positive_int(value: Any) -> int | None:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None
