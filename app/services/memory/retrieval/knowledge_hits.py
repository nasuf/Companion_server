"""Literal-hit probe for admin-published knowledge memories.

Why this exists (2026-07-23, production canary of the template-knowledge
feature): the relevance gate (spec §3.4) classifies world-knowledge questions
("西甲联赛什么时候开始？") as weak — the small model cannot know the memory
bank happens to contain admin-published facts about that exact topic — so
retrieval is skipped and the AI answers "不知道" despite having the memory.
And even when the gate passes ("公司最近有活动吗"), generic wording can rank
the knowledge rows below the 0.50 vector threshold, crowding them out.

The probe is deterministic and cheap: knowledge rows (provenance =
'knowledge_seed') are few (≤200 per workspace, Redis-cached, and the vast
majority of workspaces have none → one cached-empty GET), and a CJK n-gram
containment check ("西甲" / "门票" / "活动" appearing inside a row's content)
is a reliable signal that the row IS what the user is asking about.

Consumption (data_fetch_phase):
- weak relevance + hits → inject the hit rows directly (no embedding, no
  vector search) and escalate to "medium" so the tier/main prompts actually
  carry the text (the weak tier prompt has no memory placeholder at all).
- medium/strong → union the hits into the already-selected set, so literal
  topic matches can never be dropped by vector ranking.

Context fallback (2026-07-23, second canary): elliptical follow-ups ("啥时候
开始？" right after the AI described 西甲) carry no topic tokens themselves,
and enhanced_query restoration by the relevance LLM is not reliable — that
turn retrieved the WRONG time row (伴生App 上线时间) and the reply conflated
the product launch with the event date. When the current message + enhanced
query produce no hits AND the message looks like a short follow-up question,
the probe re-runs on the last few conversation turns' raw text: the context
names the topic (西甲/球队/比赛), the injection cap is raised so the whole
topic block (名称/时间/地点/票务…) rides along, and the reply LLM picks the
attribute the user asked about.

False positives (e.g. the user mentions 公司 about THEIR OWN company) cost a
few extra AI-slot lines rendered under 【你自己的相关经历 / 人设】 — bounded
by the hit caps and harmless to grounding. False negatives fall back to the
pre-existing behavior.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable

from app.db import db
from app.redis_client import get_redis
from app.services.memory.provenance import KNOWLEDGE_SEED
from app.services.memory.retrieval.context_selector import ClassifiedMemory

logger = logging.getLogger(__name__)

_CACHE_TTL_S = 180
_CACHE_KEY_PREFIX = "knowledge_rows"
_MAX_ROWS = 200
_MAX_HITS = 4
# Context-fallback hits are topic-block injections (the current message names
# no attribute we can rank by), so the cap is wider to fit a whole section.
_CONTEXT_MAX_HITS = 8
_MIN_GRAM_LEN = 2
_MAX_GRAM_LEN = 4

# A message qualifies for the context fallback only when it reads like a
# short follow-up question — long or non-interrogative messages must carry
# their own topic tokens (primary path) or stay out.
_CONTINUATION_MAX_LEN = 24
_QUESTION_MARKER_RE = re.compile(r"[?？]|啥|什么|哪|几|多少|多久|谁|怎|吗$|呢$")

# Keep in sync with agent_template.knowledge.KNOWLEDGE_IMPORTANCE (not
# imported: pulling the template service into the chat hot path would drag
# clone/store_memory import chains along for a display-only constant).
_KNOWLEDGE_IMPORTANCE = 0.86
# Synthetic ranking scores for injected hits: high enough to be kept, low
# enough to never outrank a genuine strong vector match (~0.7+).
_HIT_SCORE = 0.55

_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
# ≥3 chars: 2-char alnum tokens are systematic false-hit sources — "ai" shows
# up both in enhanced queries ("AI知道的那个东西") and in persona/knowledge
# copy ("打造「有生命的AI」"), and a single junk hit used to suppress the
# context fallback entirely (2026-07-24 trace: 那是啥 → 西班牙足球甲级联赛).
# Real topic tokens (app / 2026 / vip) are ≥3.
_ALNUM_RUN_RE = re.compile(r"[a-z0-9]{3,}")

# Structural / conversational grams that appear in questions about anything
# and therefore carry no topical signal. Topic nouns (公司/活动/门票/比赛…)
# deliberately stay hit-capable — recall is the point of this probe.
_STOP_GRAMS = frozenset({
    "什么", "怎么", "为何", "时候", "时间", "可以", "知道", "没有", "现在",
    "今天", "明天", "昨天", "我们", "你们", "他们", "一下", "一个", "这个",
    "那个", "还是", "就是", "但是", "因为", "所以", "如果", "感觉", "觉得",
    "真的", "好的", "不是", "有点", "非常", "开始", "结束", "最近", "地方",
    "东西", "事情", "问题", "告诉", "记得", "喜欢", "想去", "不贵",
})


def extract_topic_grams(text: str | None) -> set[str]:
    """CJK n-grams (2-4 chars) + alphanumeric tokens from *text*, minus
    structural stop-grams. Case-folded so "App"/"app" match."""
    if not text:
        return set()
    folded = text.casefold()
    grams: set[str] = set()
    for run in _CJK_RUN_RE.findall(folded):
        run_len = len(run)
        for n in range(_MIN_GRAM_LEN, _MAX_GRAM_LEN + 1):
            if run_len < n:
                break
            for i in range(run_len - n + 1):
                grams.add(run[i : i + n])
    for token in _ALNUM_RUN_RE.findall(folded):
        grams.add(token)
    return grams - _STOP_GRAMS


def find_literal_hits(
    grams: set[str],
    rows: Iterable[dict[str, Any]],
    *,
    max_hits: int = _MAX_HITS,
) -> list[dict[str, Any]]:
    """Rows whose content contains any topic gram, best matches first.

    Ranking: longest matched gram first (a 4-gram like "西甲联赛" is a much
    stronger topical signal than a lone 2-gram), then number of distinct
    matched grams.
    """
    scored: list[tuple[int, int, int, dict[str, Any]]] = []
    for order, row in enumerate(rows):
        content = (row.get("content") or "").casefold()
        if not content:
            continue
        matched = [g for g in grams if g in content]
        if not matched:
            continue
        scored.append((max(len(g) for g in matched), len(matched), -order, row))
    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [row for _, _, _, row in scored[:max_hits]]


def knowledge_rows_cache_key(workspace_id: str) -> str:
    """Public so the append/sync pipeline can bust the cache after writes."""
    return f"{_CACHE_KEY_PREFIX}:{workspace_id}"


async def load_knowledge_rows(workspace_id: str) -> list[dict[str, Any]]:
    """``[{id, content}]`` knowledge rows of a workspace, Redis-cached.

    The empty result is cached too, so the per-message cost for the vast
    majority of workspaces (no knowledge) is a single Redis GET. Cache
    staleness after a knowledge sync is bounded by the TTL.
    """
    try:
        redis = await get_redis()
        raw = await redis.get(knowledge_rows_cache_key(workspace_id))
        if raw is not None:
            return json.loads(raw)
    except Exception:
        pass  # cache miss path below; never break the chat hot path

    try:
        db_rows = await db.aimemory.find_many(
            where={
                "workspaceId": workspace_id,
                "provenance": KNOWLEDGE_SEED,
                "isArchived": False,
            },
            # Stable document order so equal-score hits resolve deterministically.
            order={"createdAt": "asc"},
            take=_MAX_ROWS,
        )
        rows = [
            {"id": row.id, "content": row.content}
            for row in db_rows
            if row.content
        ]
    except Exception as exc:
        logger.warning("[KNOWLEDGE-HIT] row load failed: %s", exc)
        return []

    try:
        redis = await get_redis()
        await redis.set(
            knowledge_rows_cache_key(workspace_id),
            json.dumps(rows, ensure_ascii=False),
            ex=_CACHE_TTL_S,
        )
    except Exception:
        pass
    return rows


def is_continuation_question(message: str | None) -> bool:
    """Short interrogative follow-up ("啥时候开始？" / "在哪办呀") — the only
    shape allowed to borrow topic grams from prior turns."""
    text = (message or "").strip()
    if not text or len(text) > _CONTINUATION_MAX_LEN:
        return False
    return bool(_QUESTION_MARKER_RE.search(text))


async def probe_knowledge_memories(
    *,
    user_message: str | None,
    enhanced_query: str | None = "",
    context_texts: tuple[str, ...] | list[str] = (),
    workspace_id: str | None,
    exclude_texts: set[str] | frozenset[str] = frozenset(),
    max_hits: int = _MAX_HITS,
) -> list[ClassifiedMemory]:
    """Literal-hit knowledge rows as ready-to-inject AI-slot memories.

    Gram sources (2026-07-24 rework — UNION, no suppression):
    - primary: current message + enhanced_query (the relevance LLM's ellipsis
      restoration when it worked: "那门票贵不贵" → "西甲联赛的门票价格")
    - context: for short follow-up questions, the last few turns' raw text is
      ALWAYS added. Suppressing context whenever the primary path had any hit
      proved fragile — a junk restoration ("AI知道的那个东西") once produced a
      single false hit that blocked the 西甲 topic block, and the reply fell
      back to world knowledge (西班牙足球甲级联赛). With a union the two
      sources rank together: a good restoration dominates naturally, a junk
      one is out-voted by longer topical grams (足球联赛) from the context.

    Rows matched by a primary gram carry rank_reason "knowledge_literal_hit";
    context-only matches carry "knowledge_context_hit" (diagnostics).
    """
    if not workspace_id:
        return []
    primary_grams = extract_topic_grams(user_message) | extract_topic_grams(enhanced_query)
    context_grams: set[str] = set()
    if context_texts and is_continuation_question(user_message):
        for text in context_texts:
            context_grams |= extract_topic_grams(text)
        context_grams -= primary_grams
    all_grams = primary_grams | context_grams
    if not all_grams:
        return []
    rows = await load_knowledge_rows(workspace_id)
    if not rows:
        return []

    # Context involvement means the question names its topic only in prior
    # turns — inject the whole topic block so the reply LLM can pick the
    # attribute (time/venue/tickets) being asked about.
    cap = _CONTEXT_MAX_HITS if context_grams else max_hits
    memories: list[ClassifiedMemory] = []
    for row in find_literal_hits(all_grams, rows, max_hits=cap + len(exclude_texts)):
        content = row.get("content") or ""
        if not content or content in exclude_texts:
            continue
        folded = content.casefold()
        reason = (
            "knowledge_literal_hit"
            if any(gram in folded for gram in primary_grams)
            else "knowledge_context_hit"
        )
        memories.append(
            ClassifiedMemory(
                text=content,
                relevance="medium",
                score=_HIT_SCORE,
                id=str(row.get("id") or ""),
                importance=_KNOWLEDGE_IMPORTANCE,
                similarity=_HIT_SCORE,
                main_category="生活",
                sub_category="工作",
                display_score=_HIT_SCORE,
                rank_reasons=[reason],
                source="ai",
            )
        )
        if len(memories) >= cap:
            break
    return memories
