"""跑时间推理召回评测.

复用生产的排序与裁剪逻辑 (rank_memory_candidate / select_context) 而不是自己另写
一套 —— 否则测出来的是评测脚本的行为, 不是线上的行为。跟 memory_recall 那套同构。

    python -m evals.temporal_recall.run_eval
    python -m evals.temporal_recall.run_eval --isolate-ranking   # 绕过 select_context 保护槽
    python -m evals.temporal_recall.run_eval --ab-recency        # P3 求近排序 on/off A/B

输出按题型分组。真正要看的是 needs_time=True 那些题的命中率: 对照组 (needs_time=
False) 只用来确认检索本身没坏 —— 如果连纯语义题都错, 时间题的失败就不能归因于时间
能力。

--isolate-ranking 的用意: select_context 有若干"保护槽" (安全 / 关系 / AI 自我 /
当前事实, 见 context_selector.py::select_context) 会在通用排序之前先占位。若一道
求近题恰好命中"我怕什么"级别的安全语义, 保护槽会把 fear_height 类记忆钉在 top-1,
即使 rank_memory_candidate 已经把正确的近期记忆排到了通用序的第一位。这个开关跳过
保护槽, 只看排序层本身的判决 —— 用来跟 P3 (feat 674c3b4) 的度量绑定, 因为 P3 就
活在排序层。

--ab-recency 的用意: 上线时 temporal_recall 聚合指标是平的, 无法证明 P3 的价值。
本 A/B 把 _RECENCY_BOOST_WEIGHT 拨到 0 跑一遍, 再拨回默认跑一遍, 只报告命中"求近"
路径的那几个用例的 delta —— 其它不触发 P3 的题拿进平均是纯噪声, 之前正是被这样
稀释了。判据用 --isolate-ranking 更干净, 否则保护槽会把两组都盖住。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from app.services.memory.normalization import cosine_similarity
from app.services.memory.retrieval import ranking as _ranking_mod
from app.services.memory.retrieval.context_selector import select_context
from app.services.memory.retrieval.ranking import rank_memory_candidate

from evals.temporal_recall.cases import CASES, NOW, SEED_BANK, TemporalCase, TemporalSeed

SIMILARITY_THRESHOLD = 0.35
_CACHE_PATH = Path(__file__).parent / ".emb_cache.json"

# NOW (cases.NOW) 是无时区的 2026-07-29 20:00 本地时间。ranking._occur_recency_factor
# 拿 datetime.now(timezone.utc) 做 age, 因此评测里必须让"当下"= NOW(UTC), 否则真实
# 当下 (可能已过了几周) 会把所有种子都吹得比它们应有的旧, 求近 boost 被无差别削弱。
NOW_UTC = NOW.replace(tzinfo=timezone.utc)


class _FrozenNow:
    """只替换 datetime.now, 保留 datetime 类型本身 —— 关键: ranking 里有
    `isinstance(raw, datetime)` 检查, 换掉 datetime 类会让真实 datetime 实例不再
    是它的子类, 静默把 _occur_recency_factor 短路成 None (=P3 全程不生效).

    做法: 用一个只有 .now 属性的哑对象顶替 ranking.datetime, 别的属性 (fromisoformat
    / timezone 之类) 都通过 __getattr__ fallback 到真 datetime 模块.
    """

    def now(self, tz=None):
        return NOW_UTC if tz is not None else NOW

    def __getattr__(self, name):
        return getattr(datetime, name)

    def __instancecheck__(self, obj):
        return isinstance(obj, datetime)


@contextmanager
def _freeze_time_for_ranking():
    with patch.object(_ranking_mod, "datetime", _FrozenNow()):
        yield


@contextmanager
def _recency_boost(weight: float):
    """临时改 ranking._RECENCY_BOOST_WEIGHT (A/B 用). 常量, 直接 setattr 即可。"""
    original = _ranking_mod._RECENCY_BOOST_WEIGHT
    _ranking_mod._RECENCY_BOOST_WEIGHT = weight
    try:
        yield
    finally:
        _ranking_mod._RECENCY_BOOST_WEIGHT = original


@dataclass
class CaseResult:
    case_id: str
    kind: str
    needs_time: bool
    hit: bool          # 期望的记忆是否排在最前 (见 _judge 的说明)
    recalled: bool     # 期望的记忆是否出现在注入集里 (宽松判据)
    expected: list[str]
    got: list[str]
    recency_triggered: bool  # 该 query 是否命中了 P3 求近路径 (决定它进不进 A/B)
    note: str


def _judge(expected: list[str], got: list[str]) -> tuple[bool, bool]:
    """严格判据看排名, 宽松判据看是否召回.

    为什么必须看排名: 注入上限约 10 条, 而评测种子库只有十几条 —— 用"出现在结果里"
    当判据的话, 几乎所有题都会"通过", 测出来的是池子够小而不是排序对。第一版就是
    这么得到 8/8 的, 而对照组同时暴露了问题: 问"我喜欢喝什么"返回了
    ['like_coffee', 'gym_1', 'guitar_now'] —— 后两条毫不相关却也在里面。

    严格判据: 期望的每一条都要落在结果的前 len(expected) 名内。这才对应真实场景 ——
    prompt 里注入的记忆有限且靠前的权重更大, 把正确答案排到第 8 位跟没找到差不多。
    """
    top = got[: max(1, len(expected))]
    return (all(e in top for e in expected), all(e in got for e in expected))


def _load_cache() -> dict[str, list[float]]:
    if _CACHE_PATH.exists():
        try:
            return json.loads(_CACHE_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict[str, list[float]]) -> None:
    try:
        _CACHE_PATH.write_text(json.dumps(cache))
    except Exception:
        pass


async def _embed_all(texts: list[str]) -> dict[str, list[float]]:
    """向量走真实 embedding 模型, 带磁盘缓存 (反复跑评测不必反复调用).

    直接打 Ollama 的 /api/embed 而不走 storage.embedding.generate_embedding:
    后者带 Redis 缓存层, 而评测常在没有 Redis 的机器上跑。模型和 base_url 都取自
    同一份 settings, 所以算出来的向量跟线上一致。
    """
    import json as _json
    import urllib.request

    from app.config import settings

    cache = _load_cache()
    missing = [t for t in texts if t not in cache]
    if not missing:
        return cache

    url = f"{settings.ollama_base_url.rstrip('/')}/api/embed"

    def _one(text: str) -> list[float]:
        """走标准库而不是 httpx.

        实测同一个请求 curl / urllib 返回 200, httpx 返回空 body 的 503 —— 是本机
        httpx 与 Ollama 之间某层协商的问题, 跟评测逻辑无关。评测只需要拿到向量,
        没必要为此去调查客户端库。
        """
        req = urllib.request.Request(
            url,
            data=_json.dumps(
                {"model": settings.embedding_model, "input": text}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return (_json.loads(resp.read()).get("embeddings") or [[]])[0]

    for text in missing:
        vec = await asyncio.to_thread(_one, text)
        if vec:
            cache[text] = vec
    _save_cache(cache)
    return cache


def _candidate(seed: TemporalSeed, similarity: float) -> dict:
    return {
        "id": seed.id,
        "content": seed.text,
        "level": seed.level,
        "importance": seed.importance,
        "similarity": similarity,
        "source": seed.source,
        "main_category": seed.main,
        "sub_category": seed.sub,
        "occur_time": seed.occur_time,
        "statement_time": seed.statement_time,
        # 排序函数按"多久以前记下的"算新鲜度, 用 statement_time 更贴近它的语义。
        "created_at": seed.statement_time,
        "updated_at": seed.statement_time,
    }


async def run(*, isolate_ranking: bool = False) -> list[CaseResult]:
    """跑一遍所有用例.

    isolate_ranking=True 跳过 select_context 的保护槽 (安全/关系/AI 自我/当前事实),
    直接用 rank_memory_candidate 的排序名次判命中 —— 这才是排序层的成绩。默认走
    完整生产流水线 (含保护槽), 对齐用户体感。
    """
    texts = [s.text for s in SEED_BANK] + [c.query for c in CASES]
    vectors = await _embed_all(texts)

    results: list[CaseResult] = []
    for case in CASES:
        qv = vectors.get(case.query)
        if not qv:
            continue
        candidates = []
        for seed in SEED_BANK:
            sv = vectors.get(seed.text)
            if not sv:
                continue
            sim = cosine_similarity(qv, sv)
            if sim < SIMILARITY_THRESHOLD:
                continue
            cand = _candidate(seed, sim)
            with _freeze_time_for_ranking():
                score, _reasons = rank_memory_candidate(cand, case.query)
            cand["display_score"] = score
            candidates.append(cand)

        candidates.sort(key=lambda c: c["display_score"], reverse=True)
        if isolate_ranking:
            got = [c["id"] for c in candidates]
        else:
            # select_context 返回 ClassifiedMemory dataclass, 不是 dict。
            selected = select_context(candidates, query=case.query)
            got = [m.id for m in selected]
        strict, loose = _judge(list(case.expect_hit), got)
        results.append(CaseResult(
            case_id=case.id,
            kind=case.kind,
            needs_time=case.needs_time,
            hit=strict,
            recalled=loose,
            expected=list(case.expect_hit),
            got=got[:6],
            recency_triggered=_ranking_mod._is_recency_seeking_query(case.query),
            note=case.note,
        ))
    return results


def report(results: list[CaseResult], *, mode_label: str = "生产管线") -> None:
    time_cases = [r for r in results if r.needs_time]
    control = [r for r in results if not r.needs_time]

    print(f"=== [{mode_label}] 对照组 (纯语义, 用来确认检索本身没坏) ===")
    for r in control:
        print(f"  {'✓' if r.hit else '✗'} {r.case_id:<22} 期望 {r.expected} 实际 {r.got[:3]}")
    ctrl_ok = sum(r.hit for r in control)
    print(f"  {ctrl_ok}/{len(control)} 通过")
    if control and ctrl_ok < len(control):
        print("  ⚠ 对照组就有失败 —— 下面时间题的失败不能全归因于时间能力")

    print(f"\n=== [{mode_label}] 时间推理题 ===")
    by_kind: dict[str, list[CaseResult]] = defaultdict(list)
    for r in time_cases:
        by_kind[r.kind].append(r)
    for kind in sorted(by_kind):
        rows = by_kind[kind]
        ok = sum(r.hit for r in rows)
        print(f"\n  [{kind}] {ok}/{len(rows)}")
        for r in rows:
            marker = "✓" if r.hit else "✗"
            recency_tag = " (求近)" if r.recency_triggered else ""
            print(f"    {marker} {r.case_id}{recency_tag}")
            if not r.hit:
                tag = "召回了但排名靠后" if r.recalled else "根本没召回"
                print(f"        期望 {r.expected}   ({tag})")
                print(f"        实际 {r.got}")
                if r.note:
                    print(f"        —— {r.note}")

    total_ok = sum(r.hit for r in time_cases)
    total_recall = sum(r.recalled for r in time_cases)
    n = max(1, len(time_cases))
    print(f"\n  时间题 排名正确 {total_ok}/{len(time_cases)} = {100 * total_ok / n:.0f}%")
    print(f"         召回到即可 {total_recall}/{len(time_cases)} = {100 * total_recall / n:.0f}%")
    if total_recall > total_ok:
        print("  两者差距 = 找得到但排不对: 注入位置有限, 排到后面等于没找到")


def _diff_hits(a: list[CaseResult], b: list[CaseResult]) -> list[tuple[str, bool, bool]]:
    """[(case_id, a.hit, b.hit)] for cases whose hit changed, in case order."""
    a_by_id = {r.case_id: r for r in a}
    changed = []
    for r_b in b:
        r_a = a_by_id.get(r_b.case_id)
        if r_a is not None and r_a.hit != r_b.hit:
            changed.append((r_b.case_id, r_a.hit, r_b.hit))
    return changed


_DEFAULT_RECENCY_WEIGHT = _ranking_mod._RECENCY_BOOST_WEIGHT


async def _run_ab(*, isolate_ranking: bool) -> None:
    """A/B: P3 求近排序 off vs on. 只在命中求近路径的用例上算 delta.

    默认建议配 --isolate-ranking, 否则保护槽会同时压住 off/on 两组, 让 delta 稀释成
    "看起来没变" —— 这正是 P3 上线时 temporal_recall 聚合指标平掉的原因。
    """
    label_off = f"P3 off{' (裸排序)' if isolate_ranking else ''}"
    label_on = f"P3 on (w={_DEFAULT_RECENCY_WEIGHT}){' (裸排序)' if isolate_ranking else ''}"

    with _recency_boost(0.0):
        off = await run(isolate_ranking=isolate_ranking)
    with _recency_boost(_DEFAULT_RECENCY_WEIGHT):
        on = await run(isolate_ranking=isolate_ranking)

    report(off, mode_label=label_off)
    print()
    report(on, mode_label=label_on)

    off_r = [r for r in off if r.recency_triggered and r.needs_time]
    on_r = [r for r in on if r.recency_triggered and r.needs_time]
    off_ok = sum(r.hit for r in off_r)
    on_ok = sum(r.hit for r in on_r)

    print("\n" + "=" * 60)
    print("A/B 结论 (仅命中求近路径的用例, 即 P3 真正影响到的那些)")
    print("=" * 60)
    print(f"  P3 关闭  {off_ok}/{len(off_r)}")
    print(f"  P3 开启  {on_ok}/{len(on_r)}")
    delta = on_ok - off_ok
    tag = "提升" if delta > 0 else ("回退" if delta < 0 else "持平")
    print(f"  Delta   {delta:+d}   →   {tag}")
    changes = _diff_hits(off_r, on_r)
    if changes:
        print("\n  变动明细:")
        for cid, a_hit, b_hit in changes:
            arrow = "变对" if b_hit else "变错"
            print(f"    [{arrow}] {cid}   off={a_hit}  on={b_hit}")
    if not isolate_ranking:
        print("\n  注: 未加 --isolate-ranking, 保护槽仍在,"
              " off/on 都会被安全/关系槽先占. 建议再跑一次带 --isolate-ranking.")


async def _run_sweep(*, isolate_ranking: bool, weights: list[float]) -> None:
    """扫 P3 权重: 求近题目命中率 + 副作用 (非求近题目 / 对照组) 命中率.

    找 P3 的甜蜜点: 权重要足以让"求近题目的正确答案排到 top-1", 又不能挤到不该
    受影响的非求近查询或对照组 (纯语义题). 默认建议加 --isolate-ranking, 只看排序层。

    对比基线是**权重 0** (即 P3 关闭时的表现), 不是当前生产权重 —— 生产权重本身
    可能已经引入了当时没被覆盖的回退, 拿它当基线会把已有回退当"没发生".
    """
    print(f"{'weight':>7}  {'求近题目':<10}  {'非求近时间题':<14}  {'对照组':<10}"
          f"  {'总时间题':<10}  {'回退'}")
    print("-" * 68)
    off_results: dict[str, bool] = {}
    for w in weights:
        with _recency_boost(w):
            results = await run(isolate_ranking=isolate_ranking)
        time_cases = [r for r in results if r.needs_time]
        recency = [r for r in time_cases if r.recency_triggered]
        non_recency = [r for r in time_cases if not r.recency_triggered]
        control = [r for r in results if not r.needs_time]
        r_ok = sum(r.hit for r in recency)
        nr_ok = sum(r.hit for r in non_recency)
        c_ok = sum(r.hit for r in control)
        t_ok = sum(r.hit for r in time_cases)
        # Baseline = weight 0 (P3 off). Track which cases got worse.
        if w == 0.0 or not off_results:
            off_results = {r.case_id: r.hit for r in results}
        regressions = [
            r.case_id for r in results
            if off_results.get(r.case_id) and not r.hit
        ]
        regress_tag = f" ⚠ {','.join(regressions)}" if regressions else ""
        print(f"{w:>7.2f}  {r_ok:>4}/{len(recency):<4}   {nr_ok:>4}/{len(non_recency):<8}"
              f"   {c_ok:>3}/{len(control):<4}   {t_ok:>4}/{len(time_cases):<4}"
              f"{regress_tag}")
    print(f"\n(基线 = 权重 0 时的命中集合. 生产当前权重 = {_DEFAULT_RECENCY_WEIGHT})")


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--isolate-ranking", action="store_true",
                    help="绕过 select_context 保护槽, 只看排序层名次")
    ap.add_argument("--ab-recency", action="store_true",
                    help="P3 求近排序 on/off A/B, 只在命中求近路径的用例上算 delta")
    ap.add_argument("--sweep-recency", action="store_true",
                    help="扫 P3 权重 (0/0.5/1.0/1.5/2.0), 看甜蜜点与副作用")
    args = ap.parse_args()

    if args.sweep_recency:
        await _run_sweep(isolate_ranking=args.isolate_ranking,
                         weights=[0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
    elif args.ab_recency:
        await _run_ab(isolate_ranking=args.isolate_ranking)
    else:
        results = await run(isolate_ranking=args.isolate_ranking)
        label = "裸排序" if args.isolate_ranking else "生产管线"
        report(results, mode_label=label)


if __name__ == "__main__":
    asyncio.run(main())
