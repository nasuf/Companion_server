"""生产的重排公式到底有没有把有用的记忆排到前面.

按原始相似度排时, 有用记忆在 top-10 里近乎均匀分布 (第 1 名只占 19%), 说明
相似度对"有不有用"几乎没有区分力. 但生产不是按相似度排的, 是按

    display_score = importance × 时间新鲜度 × 相似度   (relevance.compute_display_score)

所以那个结论只否定了一个生产没在用的排序. 这里复用已判定的结果 (按消息+记忆
配对, 不重新花钱), 换成真实公式再算一次.

这个问题决定下一步该动哪里: 排序有信号, 就该收紧注入条数把尾巴砍掉; 排序没
信号, 那收条数和抬阈值一样是纯粹的随机丢弃, 该修的是打分本身.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from app.services.memory.retrieval.relevance import compute_display_score


def _rank_profile(ranked: dict[str, list[dict]], total_useful: int) -> list[float]:
    """有用记忆的累计占比, 按名次."""
    at_rank: dict[int, int] = defaultdict(int)
    for items in ranked.values():
        for rank, pair in enumerate(items, 1):
            if pair.get("verdict") == "有用":
                at_rank[rank] += 1
    running, out = 0, []
    for rank in range(1, 11):
        running += at_rank.get(rank, 0)
        out.append(running / total_useful if total_useful else 0.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="带 importance/updated_at 的完整导出")
    ap.add_argument("--judged", required=True, help="已判定结果, 用于复用 verdict")
    args = ap.parse_args()

    judged = json.loads(Path(args.judged).read_text())
    verdict_of = {(p["message"], p["memory"]): p.get("verdict") for p in judged}

    pairs = [p for p in json.loads(Path(args.pairs).read_text())
             if (p["message"], p["memory"]) in verdict_of]
    for pair in pairs:
        pair["verdict"] = verdict_of[(pair["message"], pair["memory"])]
        pair["display_score"] = compute_display_score(
            importance=pair.get("importance") or 0.0,
            last_accessed_at=pair.get("updated_at"),
            similarity=pair["sim"],
        )

    by_message: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        by_message[pair["message"]].append(pair)
    by_message = {m: v for m, v in by_message.items() if len(v) >= 4}

    total_useful = sum(
        1 for items in by_message.values() for p in items if p["verdict"] == "有用"
    )
    print(f"{len(by_message)} 条消息, {sum(len(v) for v in by_message.values())} 条已判定候选, "
          f"其中有用 {total_useful} 条\n")

    orderings = {
        "原始相似度": lambda p: -p["sim"],
        "display_score (生产)": lambda p: -p["display_score"],
        "importance 单独": lambda p: -(p.get("importance") or 0.0),
    }
    print(f"{'排序方式':<24}" + "".join(f"top{k:<6}" for k in (1, 2, 3, 5, 10)))
    for name, key in orderings.items():
        ranked = {m: sorted(v, key=key) for m, v in by_message.items()}
        profile = _rank_profile(ranked, total_useful)
        cells = "".join(f"{profile[k-1]:>6.0%}    " for k in (1, 2, 3, 5, 10))
        print(f"{name:<24}{cells}")

    # 随机基线不是"随便挑一条命中的概率" —— 要问的是随机排序时 top-k 能捕获
    # 全部有用记忆的多大比例. 一条消息有 c 个候选 u 条有用, 随机 top-k 期望捕到
    # u*k/c 条; 对所有消息求和再除以总数.
    def _random_at(k: int) -> float:
        expected = sum(
            sum(1 for p in v if p["verdict"] == "有用") * min(k, len(v)) / len(v)
            for v in by_message.values()
        )
        return expected / total_useful if total_useful else 0.0

    # 完美排序: 每条消息的有用记忆全部挤在最前面.
    def _perfect_at(k: int) -> float:
        best = sum(
            min(k, sum(1 for p in v if p["verdict"] == "有用"))
            for v in by_message.values()
        )
        return best / total_useful if total_useful else 0.0

    for name, fn in (("随机排序", _random_at), ("完美排序", _perfect_at)):
        cells = "".join(f"{fn(k):>6.0%}    " for k in (1, 2, 3, 5, 10))
        print(f"{name:<24}{cells}")

    print("\n跟「随机排序」那行比 —— 高出越多说明排序越有用. 贴着随机就意味着\n"
          "收紧注入条数等于随机丢弃, 该修的是打分而不是条数.")


if __name__ == "__main__":
    main()
