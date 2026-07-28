"""把记忆生命周期规则在真实快照上向前推演, 回答"改了会不会变差".

分层规则的收益和代价都是延迟的 —— 一条被错误衰减掉的记忆, 要几个月后用户问起
才会暴露。所以规则改动必须先离线推演: 拿生产的记忆快照, 按规则模拟时间前进,
看**被人工/模型判定为"有用"的那些记忆, 还留不留在可检索集合里**。

两个指标, 缺一不可:

    有用记忆留存率   判定为有用的记忆仍可被检索到的比例  ← 这是闸门指标
    可检索集合规模   每次检索要扫多少条                  ← 这是成本

只看留存率会得出"什么都别衰减"; 只看规模会得出"全砍掉最省"。分层的意义正是在
两者之间取舍, 所以要一起看。

判定数据 (evals/retrieval_threshold 产出) 描述的是"这条记忆对那句话有没有用",
跟它在哪一层无关 —— 所以改完分层规则可以直接复用, 不必重新标注。

**这个推演测不到什么**, 用它下结论前必须知道:

1. 留存率只统计已判定的那 72 条有用记忆 (来自 33 条消息)。对**未来的、没见过的**
   查询才有用的记忆不在其中 —— 所以推演能证明"没把已知有用的弄丢", 不能证明
   "对以后的问题也够用"。可检索集合规模那一列就是给这个风险留的对照。
2. 访问模式是假设的。默认假设有用的记忆被用到更频繁 —— 那本身就有利于任何基于
   使用信号的策略。所以务必同时跑对照组 (--access-useful 等于 --access-other,
   以及两者都设 0), 确认结论不是被这个假设撑起来的。
3. 推演里的层级迁移是规则的**完整应用**。实际上线会更保守 (比如存量人设的层级
   先不动), 所以推演给的是上界, 每阶段落地后要用真实规则重跑。

用法:
    python -m evals.memory_lifecycle.run_eval \\
        --snapshot /tmp/lifecycle_snapshot.json --judged /tmp/agreement.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from evals.memory_lifecycle.policy import AmvlPolicy, CurrentPolicy, Policy

CHECKPOINTS_DAYS = (0, 30, 90, 180, 365, 730)


def _days_idle(memory: dict, now: datetime) -> float:
    stamp = memory.get("last_access") or memory.get("created_at")
    if not stamp:
        return 90.0
    try:
        when = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return 90.0
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (now - when).total_seconds() / 86400)


def warm_sample_wins(memory_id: str, pairs_by_message: dict, level_of: dict,
                     budget: int) -> bool:
    """一条 L3 记忆能否挤进 warm 采样的 κ 个名额。

    采样不是随机的: 生产实现会对 L3 也跑向量检索, 只是限制最多放 κ 条进候选集。
    所以竞争是**按相似度**的 —— 在同一条消息的所有 L3 候选里排进前 κ 才算留住。

    早先把它当成"预算大于零就都留得住", 那让闸门永远通过。
    """
    if budget <= 0:
        return False
    for pairs in pairs_by_message.values():
        mine = next((p for p in pairs if p.get("_id") == memory_id), None)
        if mine is None:
            continue
        cold = sorted(
            (p for p in pairs if level_of.get(p.get("_id")) == 3),
            key=lambda p: -p["sim"],
        )
        if any(p.get("_id") == memory_id for p in cold[:budget]):
            return True
    return False


def simulate(memories: list[dict], policy: Policy, useful_ids: set[str],
             access_rate_useful: float, access_rate_other: float,
             pairs_by_message: dict | None = None,
             warm_budget: int = 0, seed: int = 0) -> list[dict]:
    """按 policy 推演到各检查点, 返回每个检查点的指标.

    访问模式是这个推演里唯一的假设, 所以拆成两个可调的比率: 有用的记忆被用到的
    频率应当高于无用的。把它设成一样就等于假设"使用与有用性无关", 那种情况下任何
    基于使用信号的策略都不可能有效 —— 那本身就是个值得知道的对照。

    随机数走显式 seed。初版用 `hash((mid, ...))` 决定某步是否命中, 而 Python 的
    字符串哈希每个进程都不一样 —— 同一份代码两次运行会给出不同的闸门结论 (实测
    365 天那一档在 +4% 和 -1.4% 之间跳)。一个结论会漂的闸门, 跟一个不会失败的闸门
    一样没用。
    """
    # 每条记忆一条独立随机流, 而不是共享一个 —— 共享的话抽签结果会依赖字典遍历
    # 顺序, 快照里多一条记忆就会让所有后续记忆的命中序列错位, 两次推演不可比。
    # 种子拼成字符串: Random 不接受 tuple。
    hit_stream = {
        m["id"]: random.Random(f"{seed}:{m['id']}") for m in memories
    }
    from app.services.memory.taxonomy import L1_SINGLETON_SUBS

    now = datetime.now(timezone.utc)
    states = {
        m["id"]: policy.initial(
            importance=m["importance"], level=m["level"],
            days_idle=_days_idle(m, now),
            # 身份事实豁免降级 —— 见 policy.MemoryState.protected
            protected=m.get("sub_category") in L1_SINGLETON_SUBS,
        )
        for m in memories
    }

    out: list[dict] = []
    previous = 0
    for checkpoint in CHECKPOINTS_DAYS:
        elapsed = checkpoint - previous
        if elapsed > 0:
            # 按 30 天为一步推进, 让"连续低于阈值 N 天"这类规则能正确累积
            remaining = elapsed
            while remaining > 0:
                step_days = min(30.0, remaining)
                for mid, state in states.items():
                    rate = access_rate_useful if mid in useful_ids else access_rate_other
                    # rate 是"每 30 天被用到的期望次数", 折算成本步的命中概率
                    probability = rate * (step_days / 30.0)
                    hit = probability >= 1.0 or (
                        probability > 0 and hit_stream[mid].random() < probability
                    )
                    states[mid] = policy.step(
                        state, step_days, accessed=hit, contributed=hit,
                    )
                remaining -= step_days
        previous = checkpoint

        levels = Counter(s.level for s in states.values())
        retrievable = {mid for mid, s in states.items() if policy.is_retrievable(s)}
        # 掉到 L3 的有用记忆还有一次机会: 在同消息的 L3 候选里排进前 κ
        if warm_budget > 0 and pairs_by_message:
            level_of = {mid: st.level for mid, st in states.items()}
            for mid in useful_ids - retrievable:
                if warm_sample_wins(mid, pairs_by_message, level_of, warm_budget):
                    retrievable.add(mid)
        useful_present = len(useful_ids & retrievable)
        out.append({
            "day": checkpoint,
            "retrievable": len(retrievable),
            "useful_retained": useful_present,
            "useful_total": len(useful_ids),
            "useful_rate": useful_present / len(useful_ids) if useful_ids else 0.0,
            "levels": dict(sorted(levels.items())),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--judged", required=True, help="带 verdict 的检索判定")
    ap.add_argument("--pairs", help="带 sim 的完整配对导出; 缺省则 warm 采样按不命中处理")
    ap.add_argument("--access-useful", type=float, default=0.5,
                    help="有用记忆每 30 天被用到的期望次数")
    ap.add_argument("--access-other", type=float, default=0.05,
                    help="其余记忆每 30 天被用到的期望次数")
    ap.add_argument("--warm-budget", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=8,
                    help="跑多少个随机种子; 判定取最坏的那个")
    ap.add_argument("--json")
    args = ap.parse_args()

    snapshot = json.loads(Path(args.snapshot).read_text())
    memories = snapshot["memories"]
    by_content = {m["content"]: m["id"] for m in memories}

    judged = json.loads(Path(args.judged).read_text())
    useful_ids = {
        by_content[j["memory"]] for j in judged
        if j.get("verdict") == "有用" and j["memory"] in by_content
    }
    print(f"快照 {len(memories)} 条记忆; 其中 {len(useful_ids)} 条被判定为有用\n")
    if not useful_ids:
        raise SystemExit("判定集与快照对不上 —— 无法计算闸门指标")

    policies = [
        CurrentPolicy(),
        AmvlPolicy(warm_sample_budget=args.warm_budget),
    ]
    # 判定配对带相似度, 用于判断 L3 能否挤进 warm 采样名额
    pairs_by_message: dict[str, list[dict]] = {}
    if args.pairs:
        for pair in json.loads(Path(args.pairs).read_text()):
            mid = by_content.get(pair["memory"])
            if mid:
                pairs_by_message.setdefault(pair["message"], []).append(
                    {**pair, "_id": mid}
                )

    # 跑多个种子取分布: 单次抽样的结论会随访问模式的运气上下摆动, 拿它做 go/no-go
    # 等于让闸门看天吃饭。
    seeds = list(range(args.seeds))
    runs: dict[str, list[list[dict]]] = {}
    for policy in policies:
        budget = getattr(policy, "warm_sample_budget", 0)
        runs[policy.name] = [
            simulate(memories, policy, useful_ids,
                     args.access_useful, args.access_other,
                     pairs_by_message=pairs_by_message, warm_budget=budget,
                     seed=seed)
            for seed in seeds
        ]

    results = {name: rows[0] for name, rows in runs.items()}
    for policy in policies:
        rows = results[policy.name]
        print(f"[{policy.name}]  (seed 0; 闸门看 {len(seeds)} 个 seed 的均值)")
        print(f"  {'天':>5}{'可检索':>9}{'有用留存':>11}{'留存率':>9}   层级分布")
        for r in rows:
            levels = " ".join(f"L{k}={v}" for k, v in r["levels"].items())
            print(f"  {r['day']:>5}{r['retrievable']:>9}"
                  f"{r['useful_retained']:>9}/{r['useful_total']:<3}"
                  f"{r['useful_rate']:>8.0%}   {levels}")
        print()

    def _rates(name: str, index: int) -> list[float]:
        return [run[index]["useful_rate"] for run in runs[name]]

    print(f"闸门: 新策略的有用记忆留存率不得低于现行 ({len(seeds)} 个 seed)")
    worst = None
    for index, checkpoint in enumerate(CHECKPOINTS_DAYS):
        base_rates = _rates(policies[0].name, index)
        new_rates = _rates(policies[1].name, index)
        delta = st.mean(new_rates) - st.mean(base_rates)
        spread = st.pstdev([n - b for n, b in zip(new_rates, base_rates)])
        if worst is None or delta < worst[1]:
            worst = (checkpoint, delta)
        # 差值小于种子间波动时, 这一档说明不了什么 —— 标出来, 免得把噪声读成结论。
        if delta < -0.005:
            verdict = "劣化"
        elif spread > 0 and abs(delta) < spread:
            verdict = "OK (在噪声内)"
        else:
            verdict = "OK"
        print(f"  第 {checkpoint:>4} 天  {st.mean(base_rates):>5.0%} → "
              f"{st.mean(new_rates):>5.0%}  ({delta:+.1%} ±{spread:.1%})  {verdict}")
    print(f"\n最差点: 第 {worst[0]} 天 {worst[1]:+.1%}")
    print("通过" if worst[1] >= -0.005 else "不通过 —— 按计划应回滚")

    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
