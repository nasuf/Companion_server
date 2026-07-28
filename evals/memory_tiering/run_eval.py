"""分层策略到底有没有用 —— 拿"这条记忆对回复有没有用"的判定去检验.

L1/L2/L3 的分层, importance 打分, 以及"建号时生成的人设全部进 L1"这条规则,
一直是按 spec 和直觉定的, 从没被数据检验过. 这个脚本用同一份人工/模型判定,
一次回答三个平时只能靠争论的问题:

  分层有意义吗        不同 level 的记忆, 被检索到之后的有用率差多少
  importance 是信号吗  分数越高是不是真的越有用
  人设该进 L1 吗       建号生成的人设 vs 聊天学到的事实, 哪个更常用得上

2026-07 首次跑出来的结果推翻了三条既有假设 (仅 AI 侧, 已排除来源混淆):

    L1 · 建号人设   n=279   有用率 11% / 20%   ← 检索池里占比最大, 命中最低
    L1 · 其他       n= 21   有用率 24% / 24%
    L2              n= 87   有用率 29% / 28%
    L2 比人设高 18 个百分点, 置换检验 p=0.0001

    importance ≥0.85   有用率 13%
    importance 0.70-0.85  有用率 52%     ← 高分段反而更差

也就是说: 唯一永不衰减、永不被淘汰的那一类记忆, 恰好是最派不上用场的一类;
而 importance 在高分段与有用性反相关 —— 这正是"按 importance 排序还不如随机
洗牌"的来源.

**这个脚本本身不下结论要怎么改.** 降级人设有真实风险: 对聊天学到的事实, 遗忘
是特性; 对人设, 遗忘是缺陷 —— 一条颜色偏好因为没人问过而沉到 L3, AI 就答不上
关于自己的问题. 脚本的用处是: 任何分层策略的改动, 都能立刻拿同一把尺子复测.

数据来自两步 (判定很贵, 但与分层策略无关, 所以改策略后可以复用):
    scripts/export_retrieval_pairs.py       真实 (消息, 记忆, 层级, 来源) 配对
    evals/retrieval_threshold/budget.py     给每个配对判"有用/沾边/无用"

用法:
    python -m evals.memory_tiering.run_eval --judged /tmp/budget.json \\
        --pairs /tmp/pairs_full.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

BANDS = ((0.85, 1.01, "≥0.85 (L1 档)"), (0.70, 0.85, "0.70-0.85"),
         (0.50, 0.70, "0.50-0.70"), (0.0, 0.50, "<0.50 (L3 档)"))
GROUPS = (
    ("L1 · 建号人设", lambda m: m["level"] == 1 and m.get("provenance") == "init"),
    ("L1 · 其他来源", lambda m: m["level"] == 1 and m.get("provenance") != "init"),
    ("L2", lambda m: m["level"] == 2),
    ("L3", lambda m: m["level"] == 3),
)


def _useful_rate(rows: list[dict], field: str) -> float:
    return sum(1 for r in rows if r.get(field) == "有用") / len(rows) if rows else 0.0


def _permutation_p(a: list[dict], b: list[dict], field: str, seed: int = 0) -> float:
    """b 的有用率高于 a 是否显著. 打散标签重采样, 看观测差值有多容易被偶然复现."""
    if not a or not b:
        return 1.0
    observed = _useful_rate(b, field) - _useful_rate(a, field)
    labels = [r.get(field) == "有用" for r in a + b]
    rng = random.Random(seed)
    hits = 0
    trials = 20000
    for _ in range(trials):
        rng.shuffle(labels)
        diff = sum(labels[len(a):]) / len(b) - sum(labels[:len(a)]) / len(a)
        if diff >= observed:
            hits += 1
    return hits / trials


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", required=True, help="带 verdict 的配对 (budget.py 产出)")
    ap.add_argument("--pairs", help="带 level/provenance 的完整导出; 判定文件已含则可省")
    ap.add_argument("--source", default="ai", choices=("ai", "user", "all"),
                    help="限定归属方 —— 建号人设全是 AI 侧, 混着比会把来源差异算进分层差异")
    args = ap.parse_args()

    judged = json.loads(Path(args.judged).read_text())
    if args.pairs:
        meta_by_text = {
            p["memory"]: p for p in json.loads(Path(args.pairs).read_text())
        }
        for row in judged:
            row.update({
                k: v for k, v in meta_by_text.get(row["memory"], {}).items()
                if k in ("level", "provenance", "importance", "source")
            })

    graded = [r for r in judged if r.get("verdict")]
    rows = [r for r in graded if r.get("level") is not None and r.get("provenance")]
    dropped = len(graded) - len(rows)
    if args.source != "all":
        rows = [r for r in rows if r.get("source") == args.source]
    if not rows:
        raise SystemExit(
            "没有可分析的行 —— 判定文件缺 level/provenance, 用 --pairs 传入导出文件"
        )

    fields = [f for f in ("verdict", "verdict2") if any(f in r for r in rows)]
    print(f"归属={args.source}  可分析配对 {len(rows)} 条  评审 {len(fields)} 位")
    if dropped:
        # 缺元数据的行直接排除而不是当成 init —— 猜一个来源会把结论算歪
        print(f"  (另有 {dropped} 条判定缺 level/provenance, 已排除)")
    print()

    print(f"  {'分组':<22}{'n':>6}" + "".join(f"{'评审' + str(i + 1):>10}" for i in range(len(fields))))
    grouped: dict[str, list[dict]] = {}
    for name, select in GROUPS:
        sub = [r for r in rows if select(r)]
        if not sub:
            continue
        grouped[name] = sub
        cells = "".join(f"{_useful_rate(sub, f):>9.0%} " for f in fields)
        print(f"  {name:<20}{len(sub):>6}{cells}")

    init = grouped.get("L1 · 建号人设")
    l2 = grouped.get("L2")
    if init and l2:
        delta = _useful_rate(l2, "verdict") - _useful_rate(init, "verdict")
        p = _permutation_p(init, l2, "verdict")
        verdict = "显著" if p < 0.05 else "不显著"
        print(f"\n  L2 比建号人设高 {delta:+.0%}   置换检验 p={p:.4f}  {verdict}")
        if p < 0.05 and delta > 0:
            print("  → 最没用的一类记忆, 恰好是唯一永不衰减、永不被淘汰的那一类")

    print(f"\n  {'importance 区间':<22}{'n':>6}{'有用率':>10}")
    for lo, hi, label in BANDS:
        sub = [r for r in rows if lo <= float(r.get("importance") or 0) < hi]
        if sub:
            print(f"  {label:<20}{len(sub):>6}{_useful_rate(sub, 'verdict'):>9.0%}")
    print("\n  分数越高有用率反而越低的话, 说明 importance 不能用来排序 —— "
          "生产曾因此\n  排得比随机洗牌还差 (见 relevance.compute_display_score 注释).")


if __name__ == "__main__":
    main()
