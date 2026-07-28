"""换 embedding 模型时, 把所有相似度阈值按百分位平移到新模型的尺度上.

换模型不是改一个配置项. 代码里有 9 个阈值挂在余弦相似度上, 而每个模型的相似度
分布位置和宽窄都不同. 照搬旧数字的后果不是"稍微偏一点", 而是整道闸门失效:

    检索 0.50   照搬 → 几乎全部召回被砍掉
    去重 0.85   照搬 → 永不判重, 重复记忆堆积
    删除 0.85   照搬 → 用户说"忘掉那件事"永远匹配不上

标定方式是**保持百分位**: 旧阈值在旧模型分布里卡在第 P 百分位, 新阈值就取新
模型分布的第 P 百分位. 这样"这道闸门放行多少比例"的语义不变 —— 比重新拍一个
数字有依据, 也比线性缩放稳 (两个分布不同形).

不同阈值面对的文本对不一样, 分布必须分开采, 否则标定的是错的东西:

    query_memory  用户消息 ↔ 记忆      检索 / L3 / 高分保底 / 关系召回
    memory_memory 记忆 ↔ 记忆          去重 / 删除匹配
    label_label   子类名 ↔ 子类名      类目归一化

分两步跑, 计算不放在生产容器里 —— 两两余弦是数万次纯 Python 浮点运算, 塞进
线上 app 容器会跟请求抢 CPU:

    # 1. 服务器上导出文本样本 (轻)
    python calibrate_embedding_thresholds.py export /tmp/cal_texts.json
    # 2. 用 embed_with_ollama.py 对每个模型产出向量, 拷回本机
    # 3. 本机算 (重)
    python calibrate_embedding_thresholds.py calibrate /tmp/cal_texts.json \\
        bge-m3=/tmp/v_bge.json qwen3-embedding:0.6b=/tmp/v_qwen.json
"""

from __future__ import annotations

import asyncio
import itertools
import json
import random
import sys

SAMPLE_MEMORIES = 220
SAMPLE_MESSAGES = 120
QUERY_PAIRS_PER_MESSAGE = 40

# (常量位置, 当前值, 面对的文本对类型)
THRESHOLDS = [
    ("retrieval/hybrid.py:_SIMILARITY_THRESHOLD", 0.50, "query_memory"),
    ("retrieval/hybrid.py:_RELATIONSHIP_RECALL_THRESHOLD", 0.35, "query_memory"),
    ("retrieval/ranking.py:_HIGH_SIMILARITY_THRESHOLD", 0.86, "query_memory"),
    ("retrieval/context_selector.py:_HIGH_SIMILARITY_THRESHOLD", 0.86, "query_memory"),
    ("retrieval/legacy.py:L3 cutoff", 0.60, "query_memory"),
    ("config.py:DEDUP_THRESHOLD", 0.85, "memory_memory"),
    ("config.py:DELETION_SIMILARITY_THRESHOLD", 0.85, "memory_memory"),
    ("normalization.py:SIMILARITY_THRESHOLD", 0.55, "label_label"),
]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


async def export(out_path: str) -> None:
    from app.db import db
    from app.services.memory.taxonomy import TAXONOMY

    await db.connect()
    memories = [
        r["content"] for r in await db.query_raw(
            "SELECT content FROM memories_ai WHERE is_archived = false "
            "UNION ALL SELECT content FROM memories_user WHERE is_archived = false"
        ) if r["content"]
    ]
    messages = [
        r["content"] for r in await db.query_raw(
            "SELECT DISTINCT content FROM messages WHERE role = $1 "
            "AND char_length(content) BETWEEN 2 AND 80", "user"
        ) if r["content"]
    ]
    await db.disconnect()

    rng = random.Random(0)
    sample = {
        "memories": rng.sample(memories, min(SAMPLE_MEMORIES, len(memories))),
        "messages": rng.sample(messages, min(SAMPLE_MESSAGES, len(messages))),
        "labels": sorted({sub for subs in TAXONOMY.values() for sub in subs}),
    }
    open(out_path, "w").write(json.dumps(sample, ensure_ascii=False))
    print(f"exported memories={len(sample['memories'])} "
          f"messages={len(sample['messages'])} labels={len(sample['labels'])} "
          f"→ {out_path}")
    # embed_with_ollama.py 吃的是一个扁平数组
    flat = out_path.replace(".json", "_flat.json")
    open(flat, "w").write(json.dumps(
        sample["memories"] + sample["messages"] + sample["labels"], ensure_ascii=False
    ))
    print(f"flat text list → {flat}")


def _populations(sample: dict, vectors: dict[str, list[float]]) -> dict[str, list[float]]:
    rng = random.Random(0)
    mem = [vectors[t] for t in sample["memories"] if t in vectors]
    msg = [vectors[t] for t in sample["messages"] if t in vectors]
    lab = [vectors[t] for t in sample["labels"] if t in vectors]
    return {
        "query_memory": [
            _cosine(q, m) for q in msg
            for m in rng.sample(mem, min(QUERY_PAIRS_PER_MESSAGE, len(mem)))
        ],
        "memory_memory": [_cosine(a, b) for a, b in itertools.combinations(mem, 2)],
        "label_label": [_cosine(a, b) for a, b in itertools.combinations(lab, 2)],
    }


def _percentile_of(value: float, population: list[float]) -> float:
    return sum(1 for x in population if x <= value) / len(population)


def _value_at(percentile: float, ordered: list[float]) -> float:
    index = min(len(ordered) - 1, max(0, int(percentile * len(ordered))))
    return ordered[index]


def calibrate(sample_path: str, specs: list[str]) -> None:
    sample = json.loads(open(sample_path).read())
    models: dict[str, dict[str, list[float]]] = {}
    for spec in specs:
        name, _, path = spec.partition("=")
        raw = json.loads(open(path).read())
        # embed_with_ollama.py 产出的是 {model: {text: vector}}
        vectors = raw.get(name) or next(iter(raw.values()))
        models[name] = _populations(sample, vectors)
    old_name, new_name = list(models)[0], list(models)[1]

    print(f"\n分布对比 (旧={old_name}  新={new_name})")
    print(f"  {'文本对':<15}{'模型':<24}{'均值':>8}{'标准差':>9}{'p50':>8}{'p95':>8}{'p99':>8}")
    for kind in ("query_memory", "memory_memory", "label_label"):
        for name in (old_name, new_name):
            pop = sorted(models[name][kind])
            mean = sum(pop) / len(pop)
            sd = (sum((x - mean) ** 2 for x in pop) / len(pop)) ** 0.5
            print(f"  {kind:<15}{name:<24}{mean:>8.3f}{sd:>9.3f}"
                  f"{_value_at(0.50, pop):>8.3f}{_value_at(0.95, pop):>8.3f}"
                  f"{_value_at(0.99, pop):>8.3f}")

    print(f"\n阈值平移 (保持百分位)")
    print(f"  {'常量':<52}{'现值':>7}{'百分位':>9}{'建议':>8}")
    suggestions = {}
    for name, current, kind in THRESHOLDS:
        old_pop = models[old_name][kind]
        new_pop = sorted(models[new_name][kind])
        pct = _percentile_of(current, old_pop)
        suggested = round(_value_at(pct, new_pop), 2)
        flag = "  ⚠ 落在旧分布之外" if pct <= 0.001 or pct >= 0.999 else ""
        suggestions[name] = suggested
        print(f"  {name:<52}{current:>7.2f}{pct:>8.1%}{suggested:>8.2f}{flag}")

    print("\n⚠ 标记的项落在旧分布边缘, 没有可对齐的百分位, 建议值不可信 ——")
    print("  这类阈值要按业务语义单独定, 或者用它实际服务的那批数据另测.")
    open("/tmp/threshold_calibration.json", "w").write(
        json.dumps(suggestions, ensure_ascii=False, indent=2))
    print("\nwrote /tmp/threshold_calibration.json")


if __name__ == "__main__":
    if sys.argv[1] == "export":
        asyncio.run(export(sys.argv[2]))
    else:
        calibrate(sys.argv[2], sys.argv[3:])
