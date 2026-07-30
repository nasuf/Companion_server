"""跑时间推理召回评测.

复用生产的排序与裁剪逻辑 (rank_memory_candidate / select_context) 而不是自己另写
一套 —— 否则测出来的是评测脚本的行为, 不是线上的行为。跟 memory_recall 那套同构。

    python -m evals.temporal_recall.run_eval

输出按题型分组。真正要看的是 needs_time=True 那些题的命中率: 对照组 (needs_time=
False) 只用来确认检索本身没坏 —— 如果连纯语义题都错, 时间题的失败就不能归因于时间
能力。
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from app.services.memory.normalization import cosine_similarity
from app.services.memory.retrieval.context_selector import select_context
from app.services.memory.retrieval.ranking import rank_memory_candidate

from evals.temporal_recall.cases import CASES, NOW, SEED_BANK, TemporalCase, TemporalSeed

SIMILARITY_THRESHOLD = 0.35
_CACHE_PATH = Path(__file__).parent / ".emb_cache.json"


@dataclass
class CaseResult:
    case_id: str
    kind: str
    needs_time: bool
    hit: bool          # 期望的记忆是否排在最前 (见 _judge 的说明)
    recalled: bool     # 期望的记忆是否出现在注入集里 (宽松判据)
    expected: list[str]
    got: list[str]
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


async def run() -> list[CaseResult]:
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
            score, _reasons = rank_memory_candidate(cand, case.query)
            cand["display_score"] = score
            candidates.append(cand)

        candidates.sort(key=lambda c: c["display_score"], reverse=True)
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
            note=case.note,
        ))
    return results


def report(results: list[CaseResult]) -> None:
    time_cases = [r for r in results if r.needs_time]
    control = [r for r in results if not r.needs_time]

    print("=== 对照组 (纯语义, 用来确认检索本身没坏) ===")
    for r in control:
        print(f"  {'✓' if r.hit else '✗'} {r.case_id:<22} 期望 {r.expected} 实际 {r.got[:3]}")
    ctrl_ok = sum(r.hit for r in control)
    print(f"  {ctrl_ok}/{len(control)} 通过")
    if control and ctrl_ok < len(control):
        print("  ⚠ 对照组就有失败 —— 下面时间题的失败不能全归因于时间能力")

    print("\n=== 时间推理题 ===")
    by_kind: dict[str, list[CaseResult]] = defaultdict(list)
    for r in time_cases:
        by_kind[r.kind].append(r)
    for kind in sorted(by_kind):
        rows = by_kind[kind]
        ok = sum(r.hit for r in rows)
        print(f"\n  [{kind}] {ok}/{len(rows)}")
        for r in rows:
            print(f"    {'✓' if r.hit else '✗'} {r.case_id}")
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


async def main() -> None:
    report(await run())


if __name__ == "__main__":
    asyncio.run(main())
