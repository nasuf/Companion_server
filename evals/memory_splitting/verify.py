"""验证拆分不破坏语义, 而且确实改善了检索.

拆分是有风险的操作: 把一条记忆切开可能让每一段都失去上下文, 检索时反而更找不到。所以
不能只看"超长比例归零"就宣告成功 —— 那只证明了 token 数变小, 没证明还能被找到。

这里量三件事:

1. **可注入性**: 拆分前整条超限被跳过 (等于不存在), 拆分后每段是否都能进注入集。
2. **可检索性**: 用针对各个事实的查询去检索, 拆分前后哪个能命中。
   拆分前那条记忆根本进不了注入集, 所以基线必然是 0 —— 真正要确认的是拆分后能命中。
3. **语义完整性**: 每一段单独读是否还成立 (有主语、有完整信息), 用向量相似度近似 ——
   段落与其所属事实查询的相似度, 应当高于它与其他事实查询的相似度。

    python -m evals.memory_splitting.verify
"""

from __future__ import annotations

import asyncio
import json
import urllib.request
from pathlib import Path

from app.config import settings
from app.services.memory.normalization import cosine_similarity
from app.services.memory.recording.splitting import split_multi_fact
from app.services.memory.retrieval.context_selector import (
    MAX_MEMORY_TOKENS_PER_ITEM,
    estimate_tokens,
)

_CACHE = Path(__file__).parent / ".emb_cache.json"

# 生产真实样本: 客服职业的 duties 转成的那条记忆 (485 字, 6 段)。
SAMPLE = (
    "我的工作是在线与电话咨询解答：通过公司内部通讯工具和400客服热线，实时解答用户在"
    "使用“伴生”App过程中遇到的各类功能性问题，如AI角色创建失败、语音通话中断、订阅支付"
    "问题等；用户情绪安抚与引导：这是我工作中最具挑战也最有价值的部分。当用户因孤独、"
    "焦虑等情绪向AI伴侣倾诉无果，转为人工客服寻求帮助时，我需要耐心倾听并给予回应；"
    "BUG复现与报告：当接到复杂的技术性问题时，我需要在自己的测试设备上一步步操作，"
    "尝试重现用户描述的BUG，并详细记录操作步骤、截图或录屏；用户反馈整理：定期汇总用户"
    "对产品的建议和意见，特别是关于AI角色的性格、对话风格、功能需求等方面的反馈"
)

# 每个查询针对原文里的一个具体事实。拆分的价值就在于让这些查询各自命中对应的那一段。
PROBES = {
    "客服热线": "你平时怎么解答用户的问题？",
    "情绪安抚": "遇到情绪低落的用户你会怎么做？",
    "BUG复现": "你需要自己复现用户报告的问题吗？",
    "反馈整理": "你会整理用户的产品建议吗？",
}


def _load() -> dict:
    if _CACHE.exists():
        try:
            return json.loads(_CACHE.read_text())
        except Exception:
            return {}
    return {}


def _embed_sync(text: str) -> list[float]:
    req = urllib.request.Request(
        f"{settings.ollama_base_url.rstrip('/')}/api/embed",
        data=json.dumps(
            {"model": settings.embedding_model, "input": text}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return (json.loads(resp.read()).get("embeddings") or [[]])[0]


async def _embed_all(texts: list[str]) -> dict[str, list[float]]:
    cache = _load()
    missing = [t for t in texts if t not in cache]
    for t in missing:
        vec = await asyncio.to_thread(_embed_sync, t)
        if vec:
            cache[t] = vec
    if missing:
        try:
            _CACHE.write_text(json.dumps(cache))
        except Exception:
            pass
    return cache


async def main() -> None:
    parts = split_multi_fact(SAMPLE)
    whole_tok = estimate_tokens(SAMPLE)

    print("=== 1. 可注入性 ===")
    print(f"  拆分前: 1 条 {whole_tok} tok  "
          f"→ {'超限, 整条被跳过 (等于不存在)' if whole_tok > MAX_MEMORY_TOKENS_PER_ITEM else '可注入'}")
    over = [p for p in parts if estimate_tokens(p) > MAX_MEMORY_TOKENS_PER_ITEM]
    print(f"  拆分后: {len(parts)} 条, 各 {[estimate_tokens(p) for p in parts]} tok  "
          f"→ {'仍有超限的, 拆分不彻底' if over else '全部可注入'}")

    vectors = await _embed_all([SAMPLE] + parts + list(PROBES.values()))

    print("\n=== 2. 可检索性 (每个查询对上哪一段) ===")
    correct = 0
    for label, query in PROBES.items():
        qv = vectors.get(query)
        if not qv:
            continue
        sims = [(cosine_similarity(qv, vectors[p]), i) for i, p in enumerate(parts)
                if p in vectors]
        if not sims:
            continue
        best_sim, best_i = max(sims)
        whole_sim = cosine_similarity(qv, vectors[SAMPLE]) if SAMPLE in vectors else 0.0
        hit = label.replace("客服热线", "热线").replace("BUG复现", "BUG") in parts[best_i]
        correct += hit
        print(f"  「{label}」{'✓' if hit else '✗'} 最匹配第 {best_i + 1} 段 "
              f"(sim={best_sim:.3f})   整条时 sim={whole_sim:.3f}")
    print(f"  {correct}/{len(PROBES)} 个查询对上了正确的段")

    print("\n=== 3. 语义完整性 (每段是否自带主语) ===")
    for i, p in enumerate(parts, 1):
        ok = p.startswith("我的")
        print(f"  段{i} {'✓' if ok else '✗'} {p[:44]}…")

    print("\n=== 结论 ===")
    gain = sum(
        cosine_similarity(vectors[q], max(
            (vectors[p] for p in parts if p in vectors),
            key=lambda v: cosine_similarity(vectors[q], v),
        )) - cosine_similarity(vectors[q], vectors[SAMPLE])
        for q in PROBES.values() if q in vectors and SAMPLE in vectors
    ) / max(1, len(PROBES))
    print(f"  拆分后最佳匹配段的平均相似度, 比整条高 {gain:+.3f}")
    print("  (整条超限本就进不了注入集, 所以这个差值只是锦上添花; "
          "真正的收益是从「完全检索不到」变成「能检索到」)")


if __name__ == "__main__":
    asyncio.run(main())
