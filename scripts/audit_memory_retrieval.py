"""Read-only memory retrieval audit for one user/agent/workspace.

The script deliberately avoids `/memories/search` because that endpoint updates
mention_count. It fetches memories through admin read APIs, embeds them locally
with the configured embedding model, then simulates vector/ranker/selector
behavior without touching the database.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from app.services.auth import create_jwt
from app.services.chat.prompt_builder import _build_memory_section
from app.services.llm.models import get_embedding_model
from app.services.memory.retrieval.context_selector import select_context
from app.services.memory.retrieval.ranking import rank_memory_candidate


DEFAULT_BASE_URL = "http://127.0.0.1:8000"


@dataclass
class QueryCase:
    label: str
    query: str
    expected_ids: set[str]
    kind: str


def _text(memory: dict[str, Any]) -> str:
    return str(memory.get("summary") or memory.get("content") or "").strip()


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _safe_preview(text: str, n: int = 80) -> str:
    return text.replace("\n", " ")[:n]


def _as_rank_row(memory: dict[str, Any], similarity: float) -> dict[str, Any]:
    return {
        "id": memory["id"],
        "content": memory.get("content") or "",
        "summary": memory.get("summary") or "",
        "level": memory.get("level"),
        "importance": memory.get("importance"),
        "mention_count": memory.get("mention_count", 0),
        "type": memory.get("type"),
        "main_category": memory.get("main_category"),
        "sub_category": memory.get("sub_category"),
        "created_at": memory.get("created_at"),
        "updated_at": memory.get("updated_at"),
        "last_accessed_at": memory.get("updated_at") or memory.get("created_at"),
        "source": memory.get("source", "user"),
        "similarity": similarity,
    }


async def _fetch_json(
    client: httpx.AsyncClient,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str],
) -> Any:
    resp = await client.get(path, params=params, headers=headers)
    resp.raise_for_status()
    return resp.json()


async def fetch_memories(
    *,
    base_url: str,
    agent_id: str,
    user_id: str | None,
    conversation_id: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    token = create_jwt(user_id or "audit-admin", "admin")
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=60,
        trust_env=False,
    ) as client:
        health = await _fetch_json(client, "/health", headers=headers)
        agents = await _fetch_json(client, "/admin-api/agents", headers=headers)
        agent_info = next((a for a in agents if a.get("id") == agent_id), None)
        resolved_user_id = user_id or (str(agent_info.get("user_id")) if agent_info else None)
        conversations = await _fetch_json(
            client,
            f"/admin-api/agents/{agent_id}/conversations",
            headers=headers,
        )
        if conversation_id:
            current_conversation = next(
                (c for c in conversations if c.get("id") == conversation_id),
                None,
            )
        else:
            current_conversation = conversations[0] if conversations else None
            conversation_id = (
                str(current_conversation.get("id"))
                if current_conversation
                else None
            )

        all_memories: list[dict[str, Any]] = []
        for source in ("user", "ai"):
            offset = 0
            while True:
                page = await _fetch_json(
                    client,
                    f"/admin-api/agents/{agent_id}/memories",
                    params={"source": source, "limit": 200, "offset": offset},
                    headers=headers,
                )
                if not page:
                    break
                for item in page:
                    item["source"] = source
                all_memories.extend(page)
                if len(page) < 200:
                    break
                offset += 200

    meta = {
        "base_url": base_url,
        "health": health,
        "agent_id": agent_id,
        "user_id": resolved_user_id,
        "conversation_id": conversation_id,
        "agent_found": agent_info is not None,
        "agent": agent_info,
        "conversation_found": current_conversation is not None,
        "conversation": current_conversation,
    }
    unique_memories: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in all_memories:
        key = (str(item.get("source") or ""), str(item.get("id") or ""))
        if key in seen:
            continue
        seen.add(key)
        unique_memories.append(item)
    meta["raw_memory_count"] = len(all_memories)
    meta["deduped_memory_count"] = len(unique_memories)
    return meta, unique_memories


def build_query_cases(memories: list[dict[str, Any]]) -> list[QueryCase]:
    cases: list[QueryCase] = []
    seen: set[tuple[str, str]] = set()
    positive_preference_terms = ("喜欢", "爱吃", "最爱", "偏爱", "欣赏")
    negative_preference_terms = ("不喜欢", "讨厌", "不吃", "过敏", "害怕", "雷区")
    relation_terms = ("直属领导", "老板", "上司", "主管", "经理")
    named_terms = ("叫", "名字", "姓名")
    negative_emotion_subcategories = {"悲伤", "恐惧", "焦虑", "失望", "孤独"}
    safety_terms = (
        "轻生", "想死", "自伤", "自残", "不想活", "活不下去", "空落落",
        "难过", "低落", "心情不好", "告别",
    )

    def has_positive_preference(text: str) -> bool:
        positive = any(term in text for term in positive_preference_terms)
        # "不喜欢" contains "喜欢" but should not create a likes expectation.
        return positive and not any(term in text for term in negative_preference_terms)

    def is_named_relation(text: str) -> bool:
        return any(term in text for term in relation_terms) and any(
            term in text for term in named_terms
        )

    def is_negative_safety_memory(memory: dict[str, Any], text: str) -> bool:
        return (
            memory.get("source") == "user"
            and (
                memory.get("sub_category") in negative_emotion_subcategories
                or any(term in text for term in safety_terms)
            )
        )

    def add(label: str, query: str, expected_ids: set[str], kind: str) -> None:
        query = query.strip()
        if not query or not expected_ids:
            return
        key = (kind, query)
        if key in seen:
            return
        seen.add(key)
        cases.append(QueryCase(label=label, query=query, expected_ids=expected_ids, kind=kind))

    for m in memories:
        mid = str(m["id"])
        summary = str(m.get("summary") or "").strip()
        content = str(m.get("content") or "").strip()
        if summary:
            add(f"exact_summary:{mid}", summary, {mid}, "exact_summary")
        if content and content != summary:
            add(f"content:{mid}", content[:120], {mid}, "content")

        text = _text(m)
        main = str(m.get("main_category") or "")
        sub = str(m.get("sub_category") or "")
        if is_named_relation(text):
            add("relation_name:boss_name", "我的老板叫什么名字", {mid}, "relation_name")
            add("relation_name:she_name", "她叫什么", {mid}, "relation_name")
            add("relation_name:leader", "你还记得我的直属领导是谁吗", {mid}, "relation_name")
        if main == "偏好" or any(term in text for term in positive_preference_terms + negative_preference_terms):
            if has_positive_preference(text):
                add("preference:likes", "我喜欢什么", {mid}, "preference")
            if any(term in text for term in negative_preference_terms):
                add("preference:dislikes", "我不喜欢什么", {mid}, "preference")
        if main == "偏好" and not any(
            term in text for term in positive_preference_terms + negative_preference_terms
        ):
            add("preference:likes", "我喜欢什么", {mid}, "preference")
            add("preference:dislikes", "我不喜欢什么", {mid}, "preference")
        if is_negative_safety_memory(m, text):
            add("safety:recent_state", "我刚才状态不好，你记得吗", {mid}, "safety")
            add("safety:crisis_followup", "我现在好多了，但还是有点空", {mid}, "safety")
        if sub == "提醒" or "提醒" in text:
            add("reminder:review", "我最近有什么提醒事项", {mid}, "reminder")

    boss_ids = {
        str(m["id"]) for m in memories
        if any(term in _text(m) for term in ("直属领导", "老板", "上司", "主管", "经理"))
        and any(term in _text(m) for term in ("叫", "名字", "姓名"))
    }
    add("global:boss_exact", "我的老板的名字叫什么", boss_ids, "global_relation")
    add("global:boss_pronoun", "我是说她的名字", boss_ids, "global_relation")

    identity_ids = {
        str(m["id"]) for m in memories
        if m.get("source") == "user"
        and (m.get("sub_category") in {"姓名", "年龄", "生日", "现居地"} or m.get("main_category") == "身份")
    }
    add("global:identity", "你记得我的基本信息吗", identity_ids, "identity")

    return cases


async def embed_texts(texts: list[str], batch_size: int = 8) -> dict[str, list[float]]:
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
    os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)
    os.environ.pop("ALL_PROXY", None)
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)
    os.environ.pop("all_proxy", None)
    model = get_embedding_model()
    result: dict[str, list[float]] = {}
    unique = list(dict.fromkeys(texts))
    for idx in range(0, len(unique), batch_size):
        batch = unique[idx : idx + batch_size]
        vectors = await model.aembed_documents(batch)
        for text, vec in zip(batch, vectors):
            result[text] = vec
    return result


def evaluate_case(
    case: QueryCase,
    memories: list[dict[str, Any]],
    memory_vectors: dict[str, list[float]],
    query_vector: list[float],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for memory in memories:
        text = _text(memory)
        vec = memory_vectors.get(memory["id"])
        if not text or vec is None:
            continue
        similarity = _cosine(query_vector, vec)
        row = _as_rank_row(memory, similarity)
        score, reasons = rank_memory_candidate(row, case.query)
        row["rank_score"] = score
        row["rank_reasons"] = reasons
        rows.append(row)

    vector_ranked = sorted(rows, key=lambda r: float(r["similarity"]), reverse=True)
    hybrid_ranked = sorted(rows, key=lambda r: float(r["rank_score"]), reverse=True)
    selected = select_context(hybrid_ranked, token_budget=800, max_items=10, query=case.query)
    selected_ids = [m.id for m in selected]

    def rank_of(ranked: list[dict[str, Any]]) -> int | None:
        for i, row in enumerate(ranked, 1):
            if row["id"] in case.expected_ids:
                return i
        return None

    vector_rank = rank_of(vector_ranked)
    hybrid_rank = rank_of(hybrid_ranked)
    return {
        "label": case.label,
        "kind": case.kind,
        "query": case.query,
        "expected_ids": sorted(case.expected_ids),
        "vector_rank": vector_rank,
        "hybrid_rank": hybrid_rank,
        "selected": any(mid in selected_ids for mid in case.expected_ids),
        "top_vector": [
            {
                "id": r["id"],
                "source": r["source"],
                "similarity": round(float(r["similarity"]), 4),
                "score": round(float(r["rank_score"]), 4),
                "text": _safe_preview(r.get("summary") or r.get("content") or ""),
            }
            for r in vector_ranked[:5]
        ],
        "top_hybrid": [
            {
                "id": r["id"],
                "source": r["source"],
                "similarity": round(float(r["similarity"]), 4),
                "score": round(float(r["rank_score"]), 4),
                "reasons": r.get("rank_reasons") or [],
                "text": _safe_preview(r.get("summary") or r.get("content") or ""),
            }
            for r in hybrid_ranked[:5]
        ],
        "selected_ids": selected_ids,
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_kind[row["kind"]].append(row)

    def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        if not n:
            return {}
        return {
            "n": n,
            "vector_recall_at_1": round(sum((r["vector_rank"] or 999) <= 1 for r in rows) / n, 3),
            "vector_recall_at_5": round(sum((r["vector_rank"] or 999) <= 5 for r in rows) / n, 3),
            "vector_recall_at_10": round(sum((r["vector_rank"] or 999) <= 10 for r in rows) / n, 3),
            "hybrid_recall_at_1": round(sum((r["hybrid_rank"] or 999) <= 1 for r in rows) / n, 3),
            "hybrid_recall_at_5": round(sum((r["hybrid_rank"] or 999) <= 5 for r in rows) / n, 3),
            "hybrid_recall_at_10": round(sum((r["hybrid_rank"] or 999) <= 10 for r in rows) / n, 3),
            "selector_recall": round(sum(bool(r["selected"]) for r in rows) / n, 3),
        }

    return {
        "overall": metrics(results),
        "by_kind": {kind: metrics(rows) for kind, rows in sorted(by_kind.items())},
    }


def analyze_memory_inventory(memories: list[dict[str, Any]]) -> dict[str, Any]:
    by_source = Counter(m.get("source") for m in memories)
    by_level = Counter(f"L{m.get('level')}" for m in memories)
    by_category = Counter(
        f"{m.get('source')}:{m.get('main_category') or '未分类'} / {m.get('sub_category') or '其他'}"
        for m in memories
    )
    duplicate_groups: dict[str, list[str]] = defaultdict(list)
    for m in memories:
        duplicate_groups[_norm(_text(m))].append(m["id"])
    duplicates = [
        {"norm": key[:80], "ids": ids}
        for key, ids in duplicate_groups.items()
        if key and len(ids) > 1
    ]
    relation_named = [
        m for m in memories
        if any(term in _text(m) for term in ("直属领导", "老板", "上司", "主管", "经理"))
        and any(term in _text(m) for term in ("叫", "名字", "姓名"))
    ]
    safety = [
        m for m in memories
        if m.get("main_category") == "情绪"
        or any(term in _text(m) for term in ("轻生", "想死", "自伤", "自残", "空落落"))
    ]
    return {
        "total": len(memories),
        "by_source": dict(by_source),
        "by_level": dict(by_level),
        "top_categories": by_category.most_common(20),
        "duplicate_groups": duplicates[:30],
        "relation_named_count": len(relation_named),
        "relation_named_samples": [
            {"id": m["id"], "source": m["source"], "text": _safe_preview(_text(m), 120)}
            for m in relation_named[:20]
        ],
        "safety_count": len(safety),
        "safety_samples": [
            {"id": m["id"], "source": m["source"], "text": _safe_preview(_text(m), 120)}
            for m in safety[:20]
        ],
    }


async def render_prompt_samples(results: list[dict[str, Any]], memories_by_id: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    samples = []
    interesting = [
        r for r in results
        if r["kind"] in {"global_relation", "relation_name", "safety"} and r["selected_ids"]
    ][:8]
    for row in interesting:
        selected_rows = []
        for mid in row["selected_ids"]:
            mem = memories_by_id.get(mid)
            if not mem:
                continue
            selected_rows.append(_as_rank_row(mem, 0.9))
        for item in selected_rows:
            score, reasons = rank_memory_candidate(item, row["query"])
            item["rank_score"] = score
            item["rank_reasons"] = reasons
        classified = select_context(selected_rows, query=row["query"])
        section = await _build_memory_section(classified, include_empty_anchor=True)
        samples.append({
            "query": row["query"],
            "kind": row["kind"],
            "memory_section": section or "",
        })
    return samples


def write_report(
    path: Path,
    *,
    meta: dict[str, Any],
    inventory: dict[str, Any],
    summary: dict[str, Any],
    failures: list[dict[str, Any]],
    rank_warnings: list[dict[str, Any]],
    prompt_samples: list[dict[str, str]],
) -> None:
    lines = [
        "# Memory Retrieval Audit",
        "",
        f"- generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"- agent_id: `{meta['agent_id']}`",
        f"- user_id: `{meta['user_id']}`",
        f"- conversation_id: `{meta['conversation_id']}`",
        f"- agent_found: `{meta.get('agent_found')}`",
        f"- conversation_found: `{meta['conversation_found']}`",
        f"- workspace_id: `{(meta.get('conversation') or {}).get('workspace_id')}`",
        f"- raw_memory_count: `{meta.get('raw_memory_count')}`",
        f"- deduped_memory_count: `{meta.get('deduped_memory_count')}`",
        "",
        "## Inventory",
        "",
        f"- total memories: {inventory['total']}",
        f"- by source: `{inventory['by_source']}`",
        f"- by level: `{inventory['by_level']}`",
        f"- relation named memories: {inventory['relation_named_count']}",
        f"- safety memories: {inventory['safety_count']}",
        "",
        "### Top Categories",
        "",
    ]
    for category, count in inventory["top_categories"]:
        lines.append(f"- {category}: {count}")

    lines.extend(["", "## Metrics", "", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", ""])

    lines.extend(["## Prompt Injection Failures", ""])
    if not failures:
        lines.append("- none")
        lines.append("")
    for row in failures[:80]:
        lines.extend([
            f"### {row['kind']} / {row['label']}",
            f"- query: `{row['query']}`",
            f"- expected_ids: `{row['expected_ids']}`",
            f"- vector_rank: `{row['vector_rank']}`; hybrid_rank: `{row['hybrid_rank']}`; selected: `{row['selected']}`",
            "- top_hybrid:",
        ])
        for item in row["top_hybrid"]:
            lines.append(
                f"  - {item['id']} [{item['source']}] sim={item['similarity']} "
                f"score={item['score']} reasons={item['reasons']} text={item['text']}"
            )
        lines.append("")

    lines.extend([
        "## Rank Warnings",
        "",
        "These cases were injected into the final selected context, but the expected memory was outside hybrid top 10.",
        "",
    ])
    if not rank_warnings:
        lines.append("- none")
        lines.append("")
    for row in rank_warnings[:80]:
        lines.extend([
            f"### {row['kind']} / {row['label']}",
            f"- query: `{row['query']}`",
            f"- expected_ids: `{row['expected_ids']}`",
            f"- vector_rank: `{row['vector_rank']}`; hybrid_rank: `{row['hybrid_rank']}`; selected: `{row['selected']}`",
            "- top_hybrid:",
        ])
        for item in row["top_hybrid"]:
            lines.append(
                f"  - {item['id']} [{item['source']}] sim={item['similarity']} "
                f"score={item['score']} reasons={item['reasons']} text={item['text']}"
            )
        lines.append("")

    lines.extend(["## Prompt Section Samples", ""])
    for sample in prompt_samples:
        lines.extend([
            f"### {sample['kind']} / {sample['query']}",
            "```",
            sample["memory_section"],
            "```",
            "",
        ])

    lines.extend(["## Inventory Details", "", "### Relation Named Samples", ""])
    for m in inventory["relation_named_samples"]:
        lines.append(f"- `{m['id']}` [{m['source']}] {m['text']}")
    lines.extend(["", "### Safety Samples", ""])
    for m in inventory["safety_samples"]:
        lines.append(f"- `{m['id']}` [{m['source']}] {m['text']}")
    lines.extend(["", "### Duplicate Groups", ""])
    if inventory["duplicate_groups"]:
        for group in inventory["duplicate_groups"]:
            lines.append(f"- ids={group['ids']} norm={group['norm']}")
    else:
        lines.append("- none")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--user-id", default="")
    parser.add_argument("--conversation-id", default="")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    meta, memories = await fetch_memories(
        base_url=args.base_url,
        agent_id=args.agent_id,
        user_id=args.user_id or None,
        conversation_id=args.conversation_id or None,
    )
    memories = [m for m in memories if _text(m)]
    cases = build_query_cases(memories)

    memory_texts = [_text(m) for m in memories]
    query_texts = [case.query for case in cases]
    vectors = await embed_texts(memory_texts + query_texts)
    memory_vectors = {m["id"]: vectors[_text(m)] for m in memories if _text(m) in vectors}

    results = [
        evaluate_case(case, memories, memory_vectors, vectors[case.query])
        for case in cases
        if case.query in vectors
    ]
    failures = [r for r in results if not r["selected"]]
    rank_warnings = [
        r for r in results
        if r["selected"] and (r["hybrid_rank"] or 999) > 10
    ]
    failures.sort(key=lambda r: (
        0 if r["kind"] in {"global_relation", "relation_name", "safety"} else 1,
        r["hybrid_rank"] or 999,
    ))
    rank_warnings.sort(key=lambda r: (
        0 if r["kind"] in {"global_relation", "relation_name", "safety"} else 1,
        r["hybrid_rank"] or 999,
    ))

    inventory = analyze_memory_inventory(memories)
    summary = summarize_results(results)
    prompt_samples = await render_prompt_samples(
        results,
        {m["id"]: m for m in memories},
    )

    out = Path(args.out) if args.out else Path("reports") / (
        f"memory_retrieval_audit_{args.agent_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    write_report(
        out,
        meta=meta,
        inventory=inventory,
        summary=summary,
        failures=failures,
        rank_warnings=rank_warnings,
        prompt_samples=prompt_samples,
    )

    print(json.dumps({
        "report": str(out),
        "memories": len(memories),
        "cases": len(cases),
        "failures": len(failures),
        "rank_warnings": len(rank_warnings),
        "overall": summary.get("overall"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
