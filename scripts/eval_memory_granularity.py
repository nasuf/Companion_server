"""长叙事记忆 vs 拆成原子事实: 检索质量到底差多少?

## 结论 (2026-08-30 实测): 不值得改写。差异是 0。

    池               注入条数  注入token  有用率   每轮有用记忆
    A 现状              10.0      301     90.0%      4.50
    B 拆成原子事实       10.0      292     90.0%      4.50   (+0%)

    被测: 一个真实克隆 agent (482 条记忆, 其中 46 条长记忆), 30 条真实用户消息,
    LLM 拆分质量已人工抽查 (中位 3 条 / 25 字, 无流水账式碎片)。

**机制**: 两个池都恰好注入 10 条, 而 token 用量 (301 / 292) 离 900 的聚合预算
差得很远 —— 真正的绑定约束是 `MAX_MEMORIES_PER_SOURCE = 10` 这个**条数**上限,
不是 token 预算。所以"长记忆挤占预算, 顶掉两三条原子事实"这个直觉是错的: 无论
拆不拆, 你都只拿到 10 条, 而按相关度排出来的前 10 条有用程度一样。

拆分确实让候选池从 482 涨到 587 (更多、更细的匹配单元), 但那些多出来的候选没能
挤进前 10 —— 说明长记忆并没有因为"话题混合"而被排到不该有的位置。

**第一版跑出 +2% 是假信号**: 那次 LLM 把一件事拆成了流水账 ("我找遍了所有楼道" /
"我找遍了车库" / "我找遍了绿化带" 三条), 平均拆出 9 条。修 prompt 后重跑归零。
所以这里的负面结论是在**拆分质量合格**的前提下得到的, 不是"拆得不好所以没用"。

**这个结论会失效的条件**: 如果以后 `MAX_MEMORIES_PER_SOURCE` 调大、或聚合预算
调小到会真正绑定, 绑定约束就从条数变成 token, 那时长记忆的成本才会显现, 该重跑。

---

背景: 生产 AI 记忆里 11.9% 落在 135-180 token (长但仍能注入), 集中来自两个
模板 agent 的人设, 经克隆放大到 48 个 agent。这些记忆**没有坏**(巡检 0 超限),
问题是粒度: 一段"大橘是流浪猫 + 我带它回家 + 它很聪明"的叙事只有一个 embedding,
它是三个话题的混合向量, 对其中任一话题的 query 只能弱匹配 —— 这个推理听起来
成立, 但实测不成立, 原因见上面的「机制」。

这个脚本回答"拆了到底值不值", 而不是假设它值。只读, 不改任何库数据。

方法 (刻意控制两个混淆):
  混淆1 短文本的相似度分布跟长文本不同 → 不比较绝对相似度, 比较**同一候选池内
        的排名**与最终注入集;
  混淆2 拆分本身会让候选变多 → 两个池都过真实的 rank_memory_candidate +
        select_context (含 900 token 聚合预算), 比的是"预算花完之后拿到了什么";

  池A (现状)  agent 的全部记忆原样
  池B (拆分)  长记忆替换成它拆出来的原子事实, 其余不变
  对同一批**真实用户消息**各跑一遍, 用同一份 judge prompt (跟标定检索阈值用的
  是同一个) 判注入集里每条"对这轮回复有没有用", 比有用率与有用条数。

用法:
    .venv/bin/python -m scripts.eval_memory_granularity --queries 30 --json /tmp/gran.json
"""

from __future__ import annotations

import argparse
import asyncio
import json

import random
from pathlib import Path

from app.db import connect_db, disconnect_db, db
from app.services.memory.normalization import cosine_similarity
from app.services.memory.retrieval.context_selector import (
    estimate_tokens,
    select_context,
)
from app.services.memory.retrieval.hybrid import _SIMILARITY_THRESHOLD, _is_trivial_message
from app.services.memory.retrieval.ranking import rank_memory_candidate

from evals.retrieval_threshold.judge import JUDGE_PROMPT, parse_verdict
from evals.utility_model.run_eval import build_model

# 只拆这个长度以上的。135 token 是"占掉单条预算 75%"的线, 也是生产分布里那
# 11.9% 的下沿。
LONG_TOKEN_THRESHOLD = 135

_SPLIT_CACHE = Path("/tmp/.granularity_split_cache.json")
_EMB_CACHE = Path("/tmp/.granularity_emb_cache.json")

# 拆分 prompt 直接沿用聊天抽取侧已验证的粒度措辞 (defaults.py:853 的规则 6):
# 那条规则是全链路唯一写了"每条只含一个独立事实"并给出反例的地方, 而聊天侧实测
# 长记忆率 0.0% —— 它是被生产数据验证过管用的表述, 不另造一套。
SPLIT_PROMPT = """把下面这条 AI 角色的自我记忆拆成若干条**可独立检索的事实**。

原文：
{content}

拆分依据是"用户可能**分别问起**的不同事情"，不是句子或短语边界。

要求：
1. **一件事就是一条**。同一件事的起因、经过、结果属于同一条，不要拆成流水账。
   反例（错误拆法）：「我找遍了所有楼道」「我找遍了车库」「我找遍了绿化带」——
   这是一件事的三个细节，应合成一条「我找遍了楼道、车库和绿化带」。
   正例：原文既讲了"这只猫怎么来的"又讲了"它什么性格"，那是两件事，拆两条。
2. **每条 15-40 字**。通常拆出 2-4 条；如果你拆出超过 5 条，几乎可以肯定是拆碎了，
   请合并回去。
3. 保持第一人称、保持原文的语气和称谓，不要改写成第三人称或书面语。
4. 不要新增原文没有的信息，也不要丢掉原文的关键信息。
5. 每条能独立看懂 —— 不能出现"这件事""那次"这种脱离上下文就不知所指的指代。

只输出 JSON 数组，形如 ["事实一", "事实二"]，不要任何其他文字。"""


def _load_cache(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


async def _split_one(model, content: str, sem: asyncio.Semaphore) -> list[str] | None:
    async with sem:
        for _ in range(2):
            try:
                resp = await asyncio.wait_for(
                    model.ainvoke(SPLIT_PROMPT.format(content=content)), timeout=90
                )
            except Exception:
                continue
            text = (getattr(resp, "content", "") or "").strip()
            if "[" in text and "]" in text:
                text = text[text.index("["):text.rindex("]") + 1]
            try:
                arr = json.loads(text)
            except Exception:
                continue
            facts = [str(x).strip() for x in arr if str(x).strip()]
            if len(facts) >= 2:
                return facts
    return None


async def _embed_all(texts: list[str]) -> dict[str, list[float]]:
    """Ollama embeddings with an on-disk cache (same model as production)."""
    from app.config import settings
    from app.services.llm.models import get_embedding_model

    cache = _load_cache(_EMB_CACHE)
    model_name = settings.embedding_model
    pending = [t for t in dict.fromkeys(texts) if f"{model_name}\x00{t}" not in cache]
    if pending:
        model = get_embedding_model()
        for i in range(0, len(pending), 32):
            batch = pending[i:i + 32]
            vecs = await model.aembed_documents(batch)
            for t, v in zip(batch, vecs):
                cache[f"{model_name}\x00{t}"] = v
        _EMB_CACHE.write_text(json.dumps(cache))
    return {t: cache[f"{model_name}\x00{t}"] for t in dict.fromkeys(texts)}


def _candidate(mem: dict, sim: float) -> dict:
    return {
        "id": mem["id"], "content": mem["content"], "level": mem.get("level", 2),
        "importance": mem.get("importance", 0.7), "similarity": sim,
        "source": "ai", "main_category": mem.get("main_category"),
        "sub_category": mem.get("sub_category"),
        "created_at": "2026-06-01T00:00:00+00:00",
        "last_accessed_at": "2026-07-01T00:00:00+00:00",
    }


def _run_retrieval(pool: list[dict], qvec: list[float], vecs: dict, query: str) -> list:
    """真实生产链路: 阈值门 → rank_memory_candidate → select_context."""
    cands = []
    for mem in pool:
        sim = cosine_similarity(qvec, vecs[mem["content"]])
        if sim < _SIMILARITY_THRESHOLD:
            continue
        c = _candidate(mem, sim)
        score, reasons = rank_memory_candidate(c, query)
        c["rank_score"], c["rank_reasons"] = score, reasons
        cands.append(c)
    cands.sort(key=lambda c: -float(c.get("rank_score", 0)))
    # ai_max_items: 这批池子全是 AI 侧人设记忆, 给它整个双槽额度, 免得 user 槽
    # 空着反而把 AI 侧压到 10 条以下, 使两个池都受同一个无关约束。
    return select_context(cands, 800, query=query, ai_max_items=10, user_max_items=0)


async def _judge(model, query: str, memory: str, sem: asyncio.Semaphore) -> str | None:
    prompt = JUDGE_PROMPT.format(message=query, memory=memory, owner="AI 自己")
    async with sem:
        for _ in range(2):
            try:
                resp = await asyncio.wait_for(model.ainvoke(prompt), timeout=90)
            except Exception:
                continue
            v = parse_verdict(getattr(resp, "content", "") or "")
            if v:
                return v
    return None


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", type=int, default=30)
    ap.add_argument("--model", default="dashscope:qwen3.5-flash")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json")
    args = ap.parse_args()
    random.Random(args.seed)  # 固定种子: DB 侧用 ORDER BY RANDOM(), 这里保证下游可复现

    await connect_db()
    try:
        # 取一个真实的、聊天量最大的克隆 agent 作为被测对象 —— 用它自己的记忆库和
        # 它自己用户说过的话, 避免拿模板 agent (没人跟它聊过) 编造 query。
        ws = (await db.query_raw("""
            SELECT c.workspace_id, COUNT(*) AS n
            FROM messages m JOIN conversations c ON c.id = m.conversation_id
            WHERE m.role='user' AND c.workspace_id IN (
                SELECT w.id FROM chat_workspaces w
                JOIN ai_agents a ON a.id = w.agent_id
                WHERE a.source_template_id IS NOT NULL
            )
            GROUP BY c.workspace_id ORDER BY n DESC LIMIT 1
        """))[0]
        workspace_id = ws["workspace_id"]
        print(f"被测 workspace {workspace_id[:8]} (该用户发过 {ws['n']} 条消息)")

        mems = await db.query_raw("""
            SELECT id, content, level, importance, main_category, sub_category
            FROM memories_ai WHERE workspace_id=$1 AND is_archived=false
        """, workspace_id)
        longs = [m for m in mems if estimate_tokens(m["content"]) > LONG_TOKEN_THRESHOLD]
        print(f"该 agent {len(mems)} 条记忆, 其中长记忆 {len(longs)} 条 "
              f"({len(longs)/len(mems)*100:.1f}%)")

        msgs = await db.query_raw("""
            SELECT m.content FROM messages m JOIN conversations c ON c.id=m.conversation_id
            WHERE c.workspace_id=$1 AND m.role='user' ORDER BY RANDOM() LIMIT $2
        """, workspace_id, args.queries * 6)
        queries: list[str] = []
        seen = set()
        for r in msgs:
            t = str(r["content"] or "").strip()
            if 3 <= len(t) <= 30 and not _is_trivial_message(t) and t not in seen:
                seen.add(t); queries.append(t)
            if len(queries) >= args.queries:
                break
        print(f"抽到 {len(queries)} 条真实用户消息做 query\n")

        model = build_model(args.model)
        sem = asyncio.Semaphore(args.concurrency)

        # ── 拆分长记忆 (带缓存) ──
        cache = _load_cache(_SPLIT_CACHE)
        todo = [m for m in longs if m["content"] not in cache]
        if todo:
            print(f"用 LLM 拆分 {len(todo)} 条长记忆…")
            results = await asyncio.gather(*(_split_one(model, m["content"], sem) for m in todo))
            for m, facts in zip(todo, results):
                if facts:
                    cache[m["content"]] = facts
            _SPLIT_CACHE.write_text(json.dumps(cache, ensure_ascii=False))
        split_map = {m["content"]: cache.get(m["content"]) for m in longs}
        ok_splits = {k: v for k, v in split_map.items() if v}
        print(f"成功拆分 {len(ok_splits)}/{len(longs)} 条, "
              f"平均拆出 {sum(len(v) for v in ok_splits.values())/max(1,len(ok_splits)):.1f} 条原子事实\n")

        # ── 构造两个池 ──
        pool_a = list(mems)
        pool_b: list[dict] = []
        for m in mems:
            facts = ok_splits.get(m["content"])
            if facts:
                for i, f in enumerate(facts):
                    pool_b.append({**m, "id": f"{m['id']}#a{i}", "content": f})
            else:
                pool_b.append(m)
        print(f"池A(现状) {len(pool_a)} 条 / 池B(拆分后) {len(pool_b)} 条")

        vecs = await _embed_all(
            [m["content"] for m in pool_a] + [m["content"] for m in pool_b] + queries
        )

        # ── 跑检索, 判有用率 ──
        rows: list[dict] = []
        for q in queries:
            qv = vecs[q]
            sel_a = _run_retrieval(pool_a, qv, vecs, q)
            sel_b = _run_retrieval(pool_b, qv, vecs, q)
            rows.append({"query": q, "a": sel_a, "b": sel_b})

        pairs: list[tuple[str, str, str]] = []  # (pool, query, memory_text)
        for r in rows:
            for m in r["a"][:5]:
                pairs.append(("A", r["query"], m.text))
            for m in r["b"][:5]:
                pairs.append(("B", r["query"], m.text))
        print(f"送 LLM 评审 {len(pairs)} 条 (每个池取注入集前 5)…")
        verdicts = await asyncio.gather(*(_judge(model, q, t, sem) for _, q, t in pairs))

        stat = {"A": {"useful": 0, "judged": 0}, "B": {"useful": 0, "judged": 0}}
        useful_per_query = {"A": {}, "B": {}}
        for (pool, q, _t), v in zip(pairs, verdicts):
            if not v:
                continue
            stat[pool]["judged"] += 1
            if v == "有用":
                stat[pool]["useful"] += 1
                useful_per_query[pool][q] = useful_per_query[pool].get(q, 0) + 1

        print("\n" + "=" * 62)
        print(f"{'':<10}{'注入条数':>10}{'注入token':>11}{'判有用':>9}{'有用率':>9}")
        for pool, key in (("A 现状", "a"), ("B 拆分后", "b")):
            p = "A" if key == "a" else "B"
            n_inj = sum(len(r[key]) for r in rows) / len(rows)
            tok = sum(sum(estimate_tokens(m.text) for m in r[key]) for r in rows) / len(rows)
            s = stat[p]
            rate = s["useful"] / s["judged"] * 100 if s["judged"] else 0
            print(f"{pool:<10}{n_inj:>10.1f}{tok:>11.0f}{s['useful']:>9}{rate:>8.1f}%")

        ua = sum(useful_per_query["A"].values()) / len(rows)
        ub = sum(useful_per_query["B"].values()) / len(rows)
        print(f"\n每轮平均拿到的有用记忆条数:  A={ua:.2f}  B={ub:.2f}  "
              f"({'+' if ub>=ua else ''}{(ub-ua)/max(ua,1e-9)*100:.0f}%)")
        print("=" * 62)
        print("\n判据: B 的『每轮有用记忆条数』显著高于 A 才值得改写。只看有用率会被"
              "\n注入条数变化误导 (拆分后条数变多, 率不变也是净赚)。")

        if args.json:
            Path(args.json).write_text(json.dumps({
                "workspace_id": workspace_id,
                "n_memories": len(mems), "n_long": len(longs),
                "n_split_ok": len(ok_splits),
                "pool_a_size": len(pool_a), "pool_b_size": len(pool_b),
                "stat": stat,
                "useful_per_turn": {"A": ua, "B": ub},
                "queries": queries,
            }, ensure_ascii=False, indent=2))
            print(f"\n结果写入 {args.json}")
    finally:
        await disconnect_db()


if __name__ == "__main__":
    asyncio.run(main())
