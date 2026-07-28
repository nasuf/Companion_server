"""用配对相似度直接标定阈值, 补百分位对齐在分布尾部失效的那几项.

百分位对齐比的是两个模型各自的**边缘分布**. 落在分布中段的阈值这样对齐没问题,
但去重 (0.85) 和高分保底 (0.86) 卡在旧分布的 99.9% / 100% 分位 —— 那里几乎没有
样本, 对齐出来的数字是拿噪声在外推.

配对映射用的是同一对文本在两个模型下的相似度. 要找的是这样一个 T_new: 用它在
新模型上划分, 结果跟旧模型用 T_old 划分尽量一致. 这利用了配对结构, 在尾部比
边缘对齐稳得多 —— 只要那一带确实有样本.

若某个阈值附近连一对样本都没有, 脚本会说明而不是硬给一个数. 这种情况必须去找
它实际服务的那类文本对 (例如去重要找真的近重复记忆), 而不是靠采样外推.

用法:
    python calibrate_paired.py /tmp/cal_texts.json \\
        bge-m3=/tmp/v_old.json qwen3-embedding:0.6b=/tmp/v_new.json
"""

from __future__ import annotations

import itertools
import json
import random
import sys

QUERY_PAIRS_PER_MESSAGE = 40

# (常量, 现值, 文本对类型)
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


def _paired(sample: dict, old: dict, new: dict) -> dict[str, list[tuple[float, float]]]:
    rng = random.Random(0)
    out: dict[str, list[tuple[float, float]]] = {}
    mem = [t for t in sample["memories"] if t in old and t in new]
    msg = [t for t in sample["messages"] if t in old and t in new]
    lab = [t for t in sample["labels"] if t in old and t in new]

    out["query_memory"] = [
        (_cosine(old[q], old[m]), _cosine(new[q], new[m]))
        for q in msg for m in rng.sample(mem, min(QUERY_PAIRS_PER_MESSAGE, len(mem)))
    ]
    out["memory_memory"] = [
        (_cosine(old[a], old[b]), _cosine(new[a], new[b]))
        for a, b in itertools.combinations(mem, 2)
    ]
    out["label_label"] = [
        (_cosine(old[a], old[b]), _cosine(new[a], new[b]))
        for a, b in itertools.combinations(lab, 2)
    ]
    return out


def _best_cut(pairs: list[tuple[float, float]], old_cut: float) -> tuple[float, float, int]:
    """找让新旧划分最一致的 T_new; 同时返回一致率和阈值邻域内的样本数.

    邻域样本数是这个结果可不可信的关键 —— 一致率再高, 若阈值附近根本没有点,
    那只是在两侧的大片区域上"都判对了", 对切点位置毫无约束.
    """
    above = [(o, n) for o, n in pairs if o >= old_cut]
    below = [(o, n) for o, n in pairs if o < old_cut]
    if not above or not below:
        return 0.0, 0.0, 0
    candidates = sorted({round(n, 3) for _, n in pairs})
    best, best_acc = candidates[0], -1.0
    for cut in candidates:
        correct = sum(1 for _, n in above if n >= cut) + sum(1 for _, n in below if n < cut)
        acc = correct / len(pairs)
        if acc > best_acc:
            best, best_acc = cut, acc
    # 旧尺度上贴近切点的样本数 (±0.03)
    near = sum(1 for o, _ in pairs if abs(o - old_cut) <= 0.03)
    return best, best_acc, near


def main() -> None:
    sample = json.loads(open(sys.argv[1]).read())
    specs = sys.argv[2:]
    loaded = []
    for spec in specs:
        name, _, path = spec.partition("=")
        raw = json.loads(open(path).read())
        loaded.append((name, raw.get(name) or next(iter(raw.values()))))
    (old_name, old_vecs), (new_name, new_vecs) = loaded

    pairs_by_kind = _paired(sample, old_vecs, new_vecs)

    print(f"配对标定 (旧={old_name}  新={new_name})\n")
    print(f"  {'常量':<52}{'现值':>7}{'建议':>7}{'划分一致':>10}{'切点邻域样本':>14}")
    results = {}
    for name, current, kind in THRESHOLDS:
        pairs = pairs_by_kind[kind]
        cut, acc, near = _best_cut(pairs, current)
        if near == 0:
            print(f"  {name:<52}{current:>7.2f}{'—':>7}{'—':>10}{near:>14}"
                  f"   ← 阈值附近无样本, 拒绝给建议")
            continue
        results[name] = cut
        print(f"  {name:<52}{current:>7.2f}{cut:>7.2f}{acc:>9.1%}{near:>14}")

    print("\n切点邻域样本数小于约 30 时, 建议值只能当参考 —— 样本没覆盖到那一带,")
    print("落在旧分布尾部的阈值 (去重/高分保底) 需要用真实的近重复文本对另测.")
    open("/tmp/threshold_paired.json", "w").write(
        json.dumps(results, ensure_ascii=False, indent=2))
    print("\nwrote /tmp/threshold_paired.json")


if __name__ == "__main__":
    main()
