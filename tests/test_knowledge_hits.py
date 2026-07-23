"""Literal-hit probe for admin-published knowledge memories (knowledge_hits.py)
and its merge/escalation semantics in data_fetch_phase.

Pinned against the production canary failure (2026-07-23): "西甲联赛什么时候
开始？" was classified weak → retrieval skipped → AI answered 不知道 despite
the knowledge rows being present and embedded.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.chat.data_fetch_phase import _merge_knowledge_hits
from app.services.memory.retrieval import knowledge_hits as kh
from app.services.memory.retrieval.context_selector import ClassifiedMemory

ROWS = [
    {"id": "k1", "content": "公司名称：伴生"},
    {"id": "k2", "content": "伴生App的产品上线时间：预计2026年9月"},
    {"id": "k3", "content": "2026年恒洁杯第二十届佛山“西甲”足球联赛的赛事时间：2026年7月10日至8月23日"},
    {"id": "k4", "content": "2026年恒洁杯第二十届佛山“西甲”足球联赛的赛事地点：佛山三水云秀山体育场"},
    {"id": "k5", "content": "2026年恒洁杯第二十届佛山“西甲”足球联赛的票务信息：首次推行收费，开闭幕式门票18.8元；小组赛普通区免费"},
    {"id": "k6", "content": "2026年恒洁杯第二十届佛山“西甲”足球联赛的活动亮点：开幕式含全息投影、无人机表演及艺人助阵"},
]


# ── extract_topic_grams ────────────────────────────────────────────────


def test_grams_extract_topic_words_and_drop_stop_grams():
    grams = kh.extract_topic_grams("西甲联赛什么时候开始？")
    assert "西甲" in grams
    assert "西甲联赛" in grams
    assert "什么" not in grams  # stop gram
    assert "时候" not in grams  # stop gram


def test_grams_include_alnum_tokens_casefolded():
    grams = kh.extract_topic_grams("伴生APP好用吗")
    assert "app" in grams
    assert "伴生" in grams


def test_grams_empty_for_noise():
    assert kh.extract_topic_grams("") == set()
    assert kh.extract_topic_grams(None) == set()
    assert kh.extract_topic_grams("??!!") == set()


# ── find_literal_hits ──────────────────────────────────────────────────


def test_hits_rank_longer_gram_matches_first():
    grams = kh.extract_topic_grams("西甲联赛什么时候开始？在哪里比赛呀？")
    hits = kh.find_literal_hits(grams, ROWS)
    contents = [r["content"] for r in hits]
    # All hits are 西甲 event rows; the company/product rows don't match.
    assert contents and all("西甲" in c for c in contents)
    assert not any("公司名称" in c for c in contents)


def test_hits_for_ticket_question_via_enhanced_query_grams():
    grams = kh.extract_topic_grams("那门票贵不贵？我想去看") | kh.extract_topic_grams(
        "西甲联赛的门票价格"
    )
    hits = kh.find_literal_hits(grams, ROWS)
    assert any("票务信息" in r["content"] for r in hits)


def test_hits_for_generic_activity_question():
    grams = kh.extract_topic_grams("对了，你们公司最近有没有搞什么活动呀？")
    hits = kh.find_literal_hits(grams, ROWS)
    contents = [r["content"] for r in hits]
    assert any("活动亮点" in c for c in contents)  # 活动 gram
    assert any("公司名称" in c for c in contents)  # 公司 gram


def test_hits_cap_and_no_match():
    grams = kh.extract_topic_grams("西甲")
    assert len(kh.find_literal_hits(grams, ROWS, max_hits=2)) == 2
    assert kh.find_literal_hits(kh.extract_topic_grams("晚饭吃了炸酱面"), ROWS) == []


# ── probe_knowledge_memories ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_probe_builds_ai_slot_memories():
    with patch.object(kh, "load_knowledge_rows", AsyncMock(return_value=ROWS)):
        memories = await kh.probe_knowledge_memories(
            user_message="西甲联赛什么时候开始？",
            workspace_id="ws-1",
        )
    assert memories
    first = memories[0]
    assert isinstance(first, ClassifiedMemory)
    assert first.source == "ai"
    assert first.rank_reasons == ["knowledge_literal_hit"]
    assert "西甲" in first.text


@pytest.mark.asyncio
async def test_probe_excludes_already_selected_texts():
    with patch.object(kh, "load_knowledge_rows", AsyncMock(return_value=ROWS)):
        memories = await kh.probe_knowledge_memories(
            user_message="西甲联赛的赛事时间和地点",
            workspace_id="ws-1",
            exclude_texts={ROWS[2]["content"]},
        )
    assert all(m.text != ROWS[2]["content"] for m in memories)


@pytest.mark.asyncio
async def test_probe_skips_loader_when_no_topic_grams():
    # Single CJK chars / punctuation yield no 2+ grams → the row loader is
    # never touched. (2-char fillers like "嗯嗯" DO yield one gram and cost a
    # single cached Redis GET — accepted, matching rows are impossible.)
    loader = AsyncMock(return_value=ROWS)
    with patch.object(kh, "load_knowledge_rows", loader):
        assert await kh.probe_knowledge_memories(
            user_message="嗯", workspace_id="ws-1",
        ) == []
    loader.assert_not_awaited()


@pytest.mark.asyncio
async def test_probe_without_workspace_is_noop():
    assert await kh.probe_knowledge_memories(
        user_message="西甲联赛", workspace_id=None,
    ) == []


# ── context fallback (elliptical follow-up questions) ──────────────────

# The second-canary regression: after the AI described 西甲 for a few turns,
# the user asked "啥时候开始？" — no topic tokens, enhanced_query came back
# empty, and vector search grounded the WRONG time row (App 上线时间).
XIJIA_CONTEXT = [
    "西甲你知道吗",
    "有四十八支球队参赛哦",
    "覆盖了粤港澳好多城市",
    "那你想了解哪方面的细节呀？",
]


def test_is_continuation_question():
    assert kh.is_continuation_question("啥时候开始？")
    assert kh.is_continuation_question("在哪里办呀?")
    assert kh.is_continuation_question("门票多少钱")
    assert not kh.is_continuation_question("好饿呀")  # not interrogative
    assert not kh.is_continuation_question("")  # empty
    assert not kh.is_continuation_question(
        "你觉得我今天应该穿什么颜色的衣服出门比较好看呢我很纠结"
    )  # over the short-follow-up length cap


@pytest.mark.asyncio
async def test_context_fallback_recovers_elliptical_question():
    with patch.object(kh, "load_knowledge_rows", AsyncMock(return_value=ROWS)):
        memories = await kh.probe_knowledge_memories(
            user_message="啥时候开始？",
            enhanced_query="",
            context_texts=XIJIA_CONTEXT,
            workspace_id="ws-1",
        )
    texts = [m.text for m in memories]
    # The whole 西甲 topic block rides along — crucially the 赛事时间 row.
    assert any("赛事时间" in t for t in texts)
    assert all(m.rank_reasons == ["knowledge_context_hit"] for m in memories)
    # The product row must NOT hit (context never mentions 伴生/App).
    assert not any("伴生App" in t for t in texts)


@pytest.mark.asyncio
async def test_primary_hits_suppress_context_fallback():
    with patch.object(kh, "load_knowledge_rows", AsyncMock(return_value=ROWS)):
        memories = await kh.probe_knowledge_memories(
            user_message="那门票贵不贵？",
            context_texts=XIJIA_CONTEXT,
            workspace_id="ws-1",
        )
    assert memories
    assert all(m.rank_reasons == ["knowledge_literal_hit"] for m in memories)


@pytest.mark.asyncio
async def test_non_question_short_message_never_borrows_context():
    with patch.object(kh, "load_knowledge_rows", AsyncMock(return_value=ROWS)):
        assert await kh.probe_knowledge_memories(
            user_message="好饿呀",
            context_texts=XIJIA_CONTEXT,
            workspace_id="ws-1",
        ) == []


# ── data_fetch_phase._merge_knowledge_hits ─────────────────────────────


def _hit(text: str) -> ClassifiedMemory:
    return ClassifiedMemory(text=text, relevance="medium", score=0.55, source="ai")


def test_merge_escalates_weak_to_medium():
    relevance, memories, strings = _merge_knowledge_hits(
        "weak", None, None, [_hit("知识A")],
    )
    assert relevance == "medium"
    assert [m.text for m in memories] == ["知识A"]
    assert strings == ["知识A"]


def test_merge_unions_into_existing_selection():
    existing = [_hit("人设记忆")]
    relevance, memories, strings = _merge_knowledge_hits(
        "strong", existing, ["人设记忆"], [_hit("知识A")],
    )
    assert relevance == "strong"  # never downgraded / re-labeled
    assert [m.text for m in memories] == ["人设记忆", "知识A"]
    assert strings == ["人设记忆", "知识A"]


def test_merge_without_hits_is_identity():
    relevance, memories, strings = _merge_knowledge_hits("weak", None, None, [])
    assert (relevance, memories, strings) == ("weak", None, None)
