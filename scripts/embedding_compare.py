"""Compare bge-m3 vs quentinz/bge-large-zh-v1.5 on Chinese short-text retrieval.

Targets the actual failure mode from production trace 2026-05-06:
user query "你还记得我上次跟你说喜欢看一部电视剧呢" did not retrieve stored
L2 memory "用户喜欢追剧" via bge-m3 (sim < 0.50 threshold). This script measures
whether bge-large-zh-v1.5 gives better separation on real query/memory shapes.

Usage:
    cd Companion_server && python3 scripts/embedding_compare.py
"""

import asyncio
import math
from typing import Sequence

import httpx

OLLAMA_URL = "http://localhost:11434/api/embed"
MODELS = ["bge-m3", "qwen3-embedding:0.6b-fp16"]

# 测试集设计 — 每个 case 是 (label, query, memory, expected_relevance)
# expected_relevance: True = 应高相似度 (相关), False = 应低相似度 (不相关)
# 类别覆盖生产真实失败模式:
#   A) 同义改写: 用户用同义词问已记的偏好
#   B) 指代/省略: 用户用"那个X"/"我说的X"指代过去聊过的内容
#   C) 时间叙述 vs 事实: "我之前X过Y" 对应 "用户曾经Y"
#   D) 泛化 vs 具体: "我家人" 对应具体亲属条目
#   E) 不相关 (sanity): 完全无关的 query/memory 对
TEST_CASES = [
    # A) 同义改写
    ("A1 追剧/电视剧", "你还记得我跟你说我喜欢看电视剧吗", "用户喜欢追剧", True),
    ("A2 追剧/看剧", "我跟你提过我喜欢追剧", "用户喜欢看剧", True),
    ("A3 食物喜好", "我喜欢吃什么水果你记得吗", "用户喜欢吃西瓜", True),
    ("A4 食物厌恶", "我讨厌吃什么", "用户讨厌吃榴莲", True),
    # B) 指代/省略 (生产失败模式)
    ("B1 那部剧", "你还记得我上次跟你说喜欢看一部电视剧呢", "用户喜欢追剧", True),
    ("B2 我妈", "你还记得我妈叫啥吗", "用户的妈妈叫黑色丝袜", True),
    ("B3 我爬过的山", "我之前不是爬过那座山吗", "用户曾经爬过华山", True),
    ("B4 我的妹妹", "你还记得我妹妹叫什么吗", "用户有个妹妹叫辽能", True),
    # C) 时间叙述 vs 静态事实
    ("C1 经历对事实", "我之前跟你说过我去爬山", "用户曾经爬过华山", True),
    ("C2 时间陈述", "我前几天过的生日呢", "用户的生日是5月4日", True),
    # D) 泛化 vs 具体
    ("D1 家人", "你还记得我家人的事吗", "用户的爸爸叫李飞龙", True),
    ("D2 性别", "我是男生", "用户是男性", True),
    # E) 不相关 (期望低相似度, 测 discrimination)
    ("E1 剧 vs 爸爸", "你还记得我喜欢看一部电视剧呢", "用户的爸爸叫李飞龙", False),
    ("E2 食物 vs 性别", "我喜欢吃辣", "用户是男性", False),
    ("E3 工作 vs 生日", "我今天加班好累", "用户的生日是5月4日", False),
    ("E4 天气 vs 妹妹", "今天天气真好", "用户有个妹妹叫辽能", False),
]


async def embed(client: httpx.AsyncClient, model: str, text: str) -> list[float]:
    r = await client.post(
        OLLAMA_URL,
        json={"model": model, "input": text},
        timeout=30.0,
    )
    r.raise_for_status()
    return r.json()["embeddings"][0]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


async def main() -> None:
    async with httpx.AsyncClient() as client:
        # 先 warmup, 把模型加载进显存避免首次冷启动失真
        for m in MODELS:
            await embed(client, m, "warmup")

        results: dict[str, list[tuple[str, str, str, bool, float]]] = {m: [] for m in MODELS}

        for model in MODELS:
            print(f"\n=== Computing embeddings with {model} ===")
            for label, query, memory, expected in TEST_CASES:
                q_emb = await embed(client, model, query)
                m_emb = await embed(client, model, memory)
                sim = cosine(q_emb, m_emb)
                results[model].append((label, query, memory, expected, sim))

        # 打印对比表
        print("\n" + "=" * 110)
        print(f"{'Case':<25} {'bge-m3':>10} {'Qwen3-0.6B':>14} {'Δ':>8} {'Expected':>10}")
        print("-" * 110)
        improvements = []
        for i, (label, q, m, exp, _) in enumerate(results[MODELS[0]]):
            sim_old = results[MODELS[0]][i][4]
            sim_new = results[MODELS[1]][i][4]
            delta = sim_new - sim_old
            mark = "✓相关" if exp else "✗无关"
            print(f"{label:<25} {sim_old:>10.4f} {sim_new:>14.4f} {delta:>+8.4f} {mark:>10}")
            if exp:
                improvements.append(delta)

        # 汇总指标
        print("\n" + "=" * 110)
        print("汇总:")
        for model in MODELS:
            rel_sims = [r[4] for r in results[model] if r[3]]
            irr_sims = [r[4] for r in results[model] if not r[3]]
            avg_rel = sum(rel_sims) / len(rel_sims)
            avg_irr = sum(irr_sims) / len(irr_sims)
            sep = avg_rel - avg_irr
            # 0.50 阈值下相关 case 的召回 (生产配置)
            recall_at_50 = sum(1 for s in rel_sims if s >= 0.50) / len(rel_sims)
            recall_at_60 = sum(1 for s in rel_sims if s >= 0.60) / len(rel_sims)
            # 阈值下无关 case 的误召
            fp_at_50 = sum(1 for s in irr_sims if s >= 0.50) / len(irr_sims)
            print(f"\n  {model}:")
            print(f"    平均相似度  相关={avg_rel:.4f}  无关={avg_irr:.4f}  分离度={sep:.4f}")
            print(f"    阈值 0.50  recall={recall_at_50:.0%}  误召率={fp_at_50:.0%}")
            print(f"    阈值 0.60  recall={recall_at_60:.0%}")

        print(f"\n相关 case 上 Qwen3-0.6B vs bge-m3 平均提升: {sum(improvements)/len(improvements):+.4f}")
        print(f"提升的 case: {sum(1 for d in improvements if d > 0)}/{len(improvements)}")


if __name__ == "__main__":
    asyncio.run(main())
