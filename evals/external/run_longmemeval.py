#!/usr/bin/env python
"""跑 LongMemEval 时间推理子集, 量我们检索层的召回.

    PYTHONPATH=. python evals/external/run_longmemeval.py \
        --data /tmp/lme/s_cleaned.json --limit 30

复用生产的 `rank_memory_candidate` 而不是自己另写打分 —— 否则测的是评测脚本,
不是线上行为。时间过滤走 `has_explicit_time` + `parse_time_expressions`, 跟
`hybrid.py` 里的判定条件一致。

对照口径 (论文 Table 4, LongMemEval_M 的 temporal 子集, Value = Round):

    K=V              Recall@5 = 0.421   Recall@10 = 0.499
    K=V + 事实扩展    Recall@5 = 0.489   Recall@10 = 0.550
    + GPT-4o 时间扩展 Recall@5 = 0.526   Recall@10 = 0.722

注意我们默认跑的是 S/oracle 而不是 M, haystack 更小, 数字天然偏高, 不能直接
跟上表比绝对值 —— 它的用途是**我们自己改动前后的对照**。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.external.longmemeval import (  # noqa: E402
    TemporalCase,
    all_evidence_at_k,
    load_temporal_cases,
    recall_at_k,
)

_CACHE = Path(__file__).parent / ".lme_emb_cache.jsonl"


class EmbedCache:
    """按内容哈希缓存向量.

    LongMemEval 的 haystack session 在题目之间大量复用, 不缓存会把同一段文本嵌
    几十次 —— 实测能省掉 70% 以上的调用。

    存 JSONL 追加而不是整份 JSON 重写: 跑满 133 题约 3.3 万条向量 ≈ 400MB, 每 200
    条重写一次整份是 O(n²), 后半程光写盘就比嵌入还慢。追加是 O(n), 而且中途被杀
    也不会丢已算的部分。
    """

    def __init__(self) -> None:
        self._data: dict[str, list[float]] = {}
        self._fh = None
        if _CACHE.exists():
            with _CACHE.open() as f:
                for line in f:
                    try:
                        k, v = json.loads(line)
                        self._data[k] = v
                    except Exception:
                        continue  # 半行 (上次被杀在写盘中途) 丢掉即可
        self._dirty = 0

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha1(text.encode()).hexdigest()

    async def get(self, text: str) -> list[float]:
        """取向量; 失败时逐级砍短重试.

        本地 embedder 会被超长输入打挂 (实测 llama-server 直接 EOF)。跑几千条时
        中途挂一次就白跑, 所以这里砍短重试而不是抛 —— 拿到一个稍短文本的向量,
        比整轮评测中断有用得多。
        """
        from app.services.memory.storage.embedding import generate_embedding

        k = self.key(text)
        if k in self._data:
            return self._data[k]
        last: Exception | None = None
        for cut in (len(text), 600, 300, 150):
            try:
                vec = await generate_embedding(text[:cut])
                self._data[k] = vec
                self._append(k, vec)
                return vec
            except Exception as e:  # noqa: PERF203
                last = e
        raise RuntimeError(f"embedding 连续失败, 最后一次: {last}")

    def _append(self, key: str, vec: list[float]) -> None:
        try:
            if self._fh is None:
                self._fh = _CACHE.open("a")
            self._fh.write(json.dumps([key, vec]) + "\n")
            self._dirty += 1
            if self._dirty >= 200:
                self.flush()
        except Exception:
            pass

    def flush(self) -> None:
        try:
            if self._fh is not None:
                self._fh.flush()
            self._dirty = 0
        except Exception:
            pass

    def close(self) -> None:
        self.flush()
        if self._fh is not None:
            self._fh.close()
            self._fh = None


# 时间窗命中的候选整体上浮多少。取 0.5 是因为生产打分主体落在 0.3-0.9 区间, 半档
# 足以让命中时间窗的候选越过普通语义近邻, 又不至于把窗内的无关内容全顶上来。
# 这是个可调参数, 消融时用 --time-boost 覆盖。
_DEFAULT_TIME_BOOST = 0.5


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _oracle_time_window(case: TemporalCase, pad_days: int) -> tuple | None:
    """用标注证据的日期反推一个时间窗 —— 这是**上界实验**, 不是可上线的东西.

    任何时间过滤技术 (规则解析 / LLM 抽时间范围 / 事件日历索引) 能做到的最好情况,
    就是准确圈出证据所在的时间区间。先量这个上界: 如果连它都提升有限, 说明失败的
    根因是语义匹配而不是时间, 那整条"补时间感知"的路线就不该投入。

    pad_days 是给窗口两边留的余量 —— 现实中不可能圈得严丝合缝。
    """
    from datetime import timedelta

    dates = [r.at for r in case.rounds if r.is_evidence and r.at]
    if not dates:
        return None
    return (min(dates) - timedelta(days=pad_days), max(dates) + timedelta(days=pad_days))


async def rank_case(
    case: TemporalCase,
    cache: EmbedCache,
    use_time_filter: bool,
    time_boost: float = _DEFAULT_TIME_BOOST,
    oracle_window_pad: int | None = None,
    hard_filter: bool = False,
) -> list[str]:
    """返回按我们生产排序打分排好的 round id.

    有一点必须记在这里: `compute_display_score` 的**新鲜度因子在这个数据集上是
    完全失效的**。它按 created_at 算距今天数, 而 LongMemEval 的 session 全在
    2023 年 —— 相对现在都落进 ">365 天" 那一档, 拿到同一个 0.4。也就是说所有
    候选的时间因子一模一样, 排序里实际只有语义相似度在起作用。

    这不是评测的缺陷, 恰恰是被测系统的性质: 我们的时间因子衡量的是"这条记忆行多久
    没被碰过", 不是"这件事什么时候发生的"。对时间推理题来说前者没有信息量。
    """
    from app.services.memory.retrieval.ranking import rank_memory_candidate

    qv = await cache.get(case.question)
    time_range = None
    if oracle_window_pad is not None:
        time_range = _oracle_time_window(case, oracle_window_pad)
    elif use_time_filter:
        from app.services.memory.retrieval.hybrid import has_explicit_time
        from app.services.schedule_domain.time_parser import parse_time_expressions

        if has_explicit_time(case.question):
            parsed = parse_time_expressions(case.question)
            if parsed and not parsed[0].is_future:
                time_range = (parsed[0].start, parsed[0].end)

    scored: list[tuple[float, str]] = []
    for r in case.rounds:
        if hard_filter and time_range and not (r.at and time_range[0] <= r.at <= time_range[1]):
            # 硬过滤模式: 窗外的候选直接不进池。加权只是"上浮", 窗内其他候选同样
            # 上浮, 相对次序不变 —— 要量时间过滤的真实天花板必须整个剔除。
            continue
        sim = _cosine(qv, await cache.get(r.text))
        mem = {
            "content": r.text,
            "similarity": sim,
            "importance": 0.6,
            "main_category": "生活",
            "sub_category": "其他",
            "occur_time": r.at,
            "created_at": r.at,
            "source": "user",
            "level": 2,
        }
        score, _reasons = rank_memory_candidate(mem, case.question)
        # 时间窗命中的候选整体上浮, 与 hybrid 把 time 通路结果并进候选池同义。
        if time_range and r.at and time_range[0] <= r.at <= time_range[1]:
            score += time_boost
        scored.append((score, r.id))
    scored.sort(key=lambda x: -x[0])
    return [rid for _s, rid in scored]


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="longmemeval_oracle.json / _s_cleaned.json")
    ap.add_argument("--limit", type=int, help="只跑前 N 题")
    ap.add_argument("--no-time-filter", action="store_true", help="关掉时间窗过滤做消融")
    ap.add_argument("--time-boost", type=float, default=_DEFAULT_TIME_BOOST,
                    help="命中时间窗的候选加多少分 (消融用)")
    ap.add_argument("--oracle-window", type=int, metavar="PAD_DAYS",
                    help="上界实验: 用标注证据日期反推时间窗, 量时间过滤的天花板")
    ap.add_argument("--granularity", choices=("round", "chunk"), default="round",
                    help="检索单位: round=论文基线, chunk=近似我们的抽取粒度")
    ap.add_argument("--hard-filter", action="store_true",
                    help="窗外候选直接剔除而非降权 (真正的过滤上界)")
    ap.add_argument("--show-failures", type=int, default=5)
    args = ap.parse_args()

    cases = load_temporal_cases(Path(args.data), limit=args.limit,
                                granularity=args.granularity)
    mode = ("上界(oracle 时间窗 ±%dd)" % args.oracle_window) if args.oracle_window is not None \
        else ("无时间过滤" if args.no_time_filter else "生产时间过滤")
    print(f"时间推理题 {len(cases)} 道 (数据: {Path(args.data).name}, 模式: {mode}, 粒度: {args.granularity})")
    if not cases:
        return 1
    pool = sum(len(c.rounds) for c in cases) / len(cases)
    ev = sum(len(c.evidence_ids) for c in cases) / len(cases)
    print(f"平均候选 {pool:.0f} 轮/题, 证据 {ev:.1f} 轮/题\n")

    cache = EmbedCache()
    rows = []
    for i, case in enumerate(cases, 1):
        ranked = await rank_case(
            case, cache,
            use_time_filter=not args.no_time_filter,
            time_boost=args.time_boost,
            oracle_window_pad=args.oracle_window,
            hard_filter=args.hard_filter,
        )
        rows.append((case, ranked))
        if i % 10 == 0:
            print(f"  … {i}/{len(cases)}")
            cache.flush()
    cache.close()

    n = len(rows)
    for k in (5, 10, 20):
        cov = sum(recall_at_k(r, c.evidence_ids, k) for c, r in rows) / n
        allin = sum(all_evidence_at_k(r, c.evidence_ids, k) for c, r in rows) / n
        print(f"  Recall@{k:<3} {cov:.3f}    证据全进 top{k}: {allin:.3f}")

    fails = [(c, r) for c, r in rows if not all_evidence_at_k(r, c.evidence_ids, 10)]
    print(f"\n证据没能全进 top10 的题: {len(fails)}/{n}")
    for c, r in fails[: args.show_failures]:
        got = len(set(r[:10]) & c.evidence_ids)
        print(f"\n  [{got}/{len(c.evidence_ids)} 证据命中] {c.question[:76]}")
        for eid in list(c.evidence_ids)[:3]:
            rank = r.index(eid) + 1 if eid in r else -1
            print(f"      证据 {eid.split(':', 1)[1]:<12} 排名 {rank if rank > 0 else '未召回'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
