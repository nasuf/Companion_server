"""语义复读检查 (CHAT01.10 A 方案): 换了说法、意思几乎相同的气泡, 发出前删掉。

字面重合度的兜底 (review.drop_restated) 只拦几乎原样的句子。2026-11-04~10 的 7 天观察里
夜间评分判出 189 轮复读, 其中 142 轮是换说法的复述, 运行时只处理了 9 轮。这里用本地向量
(和记忆检索同一个模型, 不出网) 比这一轮每条气泡和前三轮 AI 的气泡, 余弦相似度到界值就删。

界值按生产 938 轮夜间标签校准 (见 policy 里的说明)。为了不拖慢回复: 主回复一生成完就在后台
算向量 (start), 和回复规范化并行; 发出前 (apply) 最多再等 _WAIT_S, 没算完就放行, 不删。

保护规则与字面兜底相同: 不删光; 对方在问 / 让你再说一遍时不删; 删完读不通的不删。
后台开关 + 按用户固定分桶的灰度比例, 默认关闭。
"""
from __future__ import annotations

import asyncio
import hashlib
import math
from datetime import datetime, timedelta, timezone

from app.services.chat.reply_repetition.policy import Policy, current
from app.services.chat.reply_repetition.rules import (
    _ASKED_TO_REPEAT,
    _DANGLING_START,
    _QUESTION,
    _USER_QUESTION,
    _key,
)

PREVIOUS_TURNS = 3
PREVIOUS_WINDOW = timedelta(hours=24)
# 太短的气泡 ("那可不") 向量相似度不稳, 不判
MIN_KEY_CHARS = 6
# 发出前最多再等这么久; 向量在主回复生成后就开始算, 通常此时已经算完
_WAIT_S = 0.3
_TEXT_CHARS = 120


async def _embed(text: str) -> list[float]:
    from app.services.memory.storage.embedding import generate_embedding

    return await generate_embedding(text)


def in_rollout(user_id: str | None, pct: int) -> bool:
    """按用户固定分桶: 同一个用户始终在或不在灰度里。"""
    if pct <= 0 or not user_id:
        return False
    if pct >= 100:
        return True
    return int(hashlib.sha256(user_id.encode()).hexdigest()[:8], 16) % 100 < pct


def _created(message: dict) -> datetime | None:
    from app.services.chat.message_utils import _parse_message_created_at

    value = _parse_message_created_at(message.get("createdAt"))
    if value is not None and value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


def _is_system_line(message: dict) -> bool:
    meta = message.get("metadata")
    return isinstance(meta, dict) and bool(
        meta.get("kind") or meta.get("component_card") or meta.get("offering_received"))


def previous_bubbles(messages: list[dict], current_turn_ids: set[str],
                     now: datetime | None = None) -> list[str]:
    """本轮之前最近三轮 AI 回复的气泡 (24 小时内, 不含程序写的卡片 / 状态行)。"""
    from app.services.chat.message_utils import _same_assistant_turn

    now = now or datetime.now(timezone.utc)
    turns: list[list[dict]] = []
    for message in messages:
        if message.get("id") in (current_turn_ids or set()):
            continue
        if message.get("role") != "assistant":
            if message.get("role") == "user" and str(message.get("content") or "").strip():
                turns.append([])  # 用户行是轮次边界
            continue
        if not str(message.get("content") or "").strip() or _is_system_line(message):
            continue
        if turns and turns[-1] and _same_assistant_turn(turns[-1][-1], message):
            turns[-1].append(message)
        else:
            turns.append([message])
    ai_turns = [t for t in turns if t]
    recent = []
    for turn in ai_turns[-PREVIOUS_TURNS:]:
        at = _created(turn[-1])
        if at is not None and now - at > PREVIOUS_WINDOW:
            continue
        recent.extend(str(m["content"]).strip() for m in turn)
    return [b for b in recent if len(_key(b)) >= MIN_KEY_CHARS]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _record(ctx) -> dict:
    return ctx.response_diagnostics.setdefault("reply_repetition", {}).setdefault("semantic", {})


def _skipped(ctx) -> bool:
    return bool(ctx.reply_is_fallback or getattr(ctx, "contradiction_inquiry", None)
                or getattr(ctx, "crisis_care_turn", False))


def start(ctx, policy: Policy | None = None) -> None:
    """主回复刚生成完: 判断这一轮要不要查, 要查就在后台开始算向量。"""
    policy = current() if policy is None else policy
    ctx.semantic_repetition_task = None
    if not policy.semantic_enabled:
        return
    record = _record(ctx)
    if not in_rollout(ctx.user_id, policy.semantic_rollout_pct):
        record["status"] = "not_in_rollout"
        return
    if _skipped(ctx):
        record["status"] = "not_applicable"
        return
    user = ctx.user_message or ""
    if _ASKED_TO_REPEAT.search(user) or _USER_QUESTION.search(user):
        record["status"] = "user_asked"
        return
    previous = previous_bubbles(ctx.messages_dicts, ctx.current_turn_ids)
    if not previous:
        record["status"] = "no_previous_turn"
        return
    texts = list(dict.fromkeys([*previous, *(r for r in ctx.replies if len(_key(r)) >= MIN_KEY_CHARS)]))
    ctx.semantic_repetition_previous = previous
    ctx.semantic_repetition_started = asyncio.get_running_loop().time()

    async def embed_all() -> dict[str, list[float]]:
        vectors = await asyncio.gather(*(_embed(t) for t in texts))
        return dict(zip(texts, vectors))

    ctx.semantic_repetition_task = asyncio.ensure_future(embed_all())


async def apply(ctx, policy: Policy | None = None) -> None:
    """发出前: 向量算完了就删掉和前三轮意思几乎相同的气泡; 没算完就放行。"""
    task = getattr(ctx, "semantic_repetition_task", None)
    if task is None:
        return
    policy = current() if policy is None else policy
    record = _record(ctx)
    loop = asyncio.get_running_loop()
    try:
        vectors = await asyncio.wait_for(asyncio.shield(task), timeout=_WAIT_S)
    except TimeoutError:
        task.cancel()
        record["status"] = "timeout"
        record["elapsed_ms"] = round((loop.time() - ctx.semantic_repetition_started) * 1000)
        return
    except Exception as exc:  # 向量服务出错: 放行, 只记下来
        record["status"] = "error"
        record["error"] = type(exc).__name__
        return
    record["elapsed_ms"] = round((loop.time() - ctx.semantic_repetition_started) * 1000)
    previous = [vectors[p] for p in ctx.semantic_repetition_previous if p in vectors]
    lead = list(ctx.continuation_lines or [])
    lead = lead if ctx.replies[: len(lead)] == lead else []
    body = ctx.replies[len(lead):]
    scores = [max((_cosine(vectors[r], p) for p in previous), default=0.0) if r in vectors else 0.0
              for r in body]
    record["max_similarity"] = round(max(scores, default=0.0), 3)
    record["threshold"] = policy.semantic_similarity
    kept = [r for r, score in zip(body, scores) if score < policy.semantic_similarity]
    dropped = [r for r, score in zip(body, scores) if score >= policy.semantic_similarity]
    if not dropped:
        record["status"] = "kept_all"
        return
    if not kept:
        record["status"] = "all_similar"  # 不能发空回复; 整段都像就交给提示段和夜间统计
        return
    if _DANGLING_START.match(kept[0].strip()) or all(_QUESTION.search(r) for r in kept):
        record["status"] = "would_break_reply"
        return
    ctx.replies = [*lead, *kept]
    ctx.reply_emotion_pre = None  # 情绪 / 表情要跟实际发出的文字走
    record["status"] = "dropped"
    record["dropped"] = len(dropped)
    record["dropped_texts"] = [t[:_TEXT_CHARS] for t in dropped[:4]]
