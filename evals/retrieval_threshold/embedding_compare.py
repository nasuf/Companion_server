"""换个 embedding 模型能不能把检索排序抬上去.

去掉 importance 之后, 排序完全由相似度决定, 而纯相似度的天花板实测只有
top3 44% (完美排序是 76%). 阈值也一样: spec 的 0.7 对 bge-m3 太严, 不得不降到
0.5 才不漏召. 两件事指向同一个嫌疑 —— bge-m3 对中文短文本的区分度不够.

这里复用已判定的 (消息, 记忆, 有用/沾边/无用), 只把相似度换成别的模型重算,
所以**不需要重新标注**. 判定与模型无关, 这正是能便宜地比模型的原因.

看两件事:
  排序能力  按新相似度排, 有用记忆进 top-k 的比例. 这是能不能收紧注入的前提.
  可分性    有用与无用两组的相似度分布隔多开. 隔得开才谈得上把阈值提回去.

dashscope 的 v3/v4 都是 1024 维, 跟 bge-m3 一致, 所以换模型不动 pgvector 的列
定义, 只要重算存量向量.

用法:
    python -m evals.retrieval_threshold.embedding_compare --judged /tmp/budget.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from collections import defaultdict
from pathlib import Path

import httpx

from app.config import settings

DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
CANDIDATES = ("text-embedding-v3", "text-embedding-v4", "qwen3.7-text-embedding")
# dashscope 单批上限: v3/v4 是 10, qwen3.7 是 20 —— 取小的通用值.
_BATCH = 10


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


async def _embed_all(model: str, texts: list[str]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    async with httpx.AsyncClient(timeout=60) as client:
        for i in range(0, len(texts), _BATCH):
            chunk = texts[i:i + _BATCH]
            for attempt in range(3):
                try:
                    response = await client.post(
                        DASHSCOPE_URL,
                        headers={"Authorization": f"Bearer {settings.dashscope_api_key}"},
                        json={"model": model, "input": chunk},
                    )
                    response.raise_for_status()
                    data = sorted(response.json()["data"], key=lambda d: d["index"])
                    for text, item in zip(chunk, data):
                        out[text] = item["embedding"]
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    await asyncio.sleep(2 * (attempt + 1))
    return out


def _capture_profile(by_message: dict[str, list[dict]], key: str,
                     total_useful: int) -> list[float]:
    at_rank: dict[int, int] = defaultdict(int)
    for items in by_message.values():
        for rank, pair in enumerate(sorted(items, key=lambda p: -p[key]), 1):
            if pair["verdict"] == "有用":
                at_rank[rank] += 1
    running, out = 0, []
    for rank in range(1, 11):
        running += at_rank.get(rank, 0)
        out.append(running / total_useful if total_useful else 0.0)
    return out


def _separation(pairs: list[dict], key: str) -> tuple[float, float, float]:
    """有用组与无用组的相似度均值, 以及两者之差 —— 差越大越能靠阈值切开."""
    useful = [p[key] for p in pairs if p["verdict"] == "有用"]
    useless = [p[key] for p in pairs if p["verdict"] == "无用"]
    mu = sum(useful) / len(useful) if useful else 0.0
    mn = sum(useless) / len(useless) if useless else 0.0
    return mu, mn, mu - mn


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", required=True)
    ap.add_argument("--cache", default="/tmp/embed_cache.json")
    args = ap.parse_args()

    pairs = [p for p in json.loads(Path(args.judged).read_text()) if p.get("verdict")]
    texts = sorted({p["message"] for p in pairs} | {p["memory"] for p in pairs})
    print(f"{len(pairs)} 条已判定配对, {len(texts)} 段不重复文本\n")

    cache_path = Path(args.cache)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for model in CANDIDATES:
        missing = [t for t in texts if t not in cache.get(model, {})]
        if missing:
            print(f"  {model}: embedding {len(missing)} 段…")
            cache.setdefault(model, {}).update(await _embed_all(model, missing))
            cache_path.write_text(json.dumps(cache))

    for model in CANDIDATES:
        vectors = cache[model]
        for pair in pairs:
            pair[model] = _cosine(vectors[pair["message"]], vectors[pair["memory"]])

    by_message: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        by_message[pair["message"]].append(pair)
    by_message = {m: v for m, v in by_message.items() if len(v) >= 4}
    scoped = [p for items in by_message.values() for p in items]
    total_useful = sum(1 for p in scoped if p["verdict"] == "有用")

    def _random_at(k: int) -> float:
        expected = sum(
            sum(1 for p in v if p["verdict"] == "有用") * min(k, len(v)) / len(v)
            for v in by_message.values()
        )
        return expected / total_useful if total_useful else 0.0

    print(f"\n排序能力 (有用记忆进 top-k 的比例, {len(by_message)} 条消息 / "
          f"{total_useful} 条有用)")
    print(f"  {'模型':<22}" + "".join(f"top{k:<6}" for k in (1, 2, 3, 5)))
    rows = [("bge-m3 (现网)", "sim")] + [(m, m) for m in CANDIDATES]
    for label, key in rows:
        profile = _capture_profile(by_message, key, total_useful)
        print(f"  {label:<22}" + "".join(f"{profile[k-1]:>6.0%}    " for k in (1, 2, 3, 5)))
    print(f"  {'随机排序':<22}" + "".join(f"{_random_at(k):>6.0%}    " for k in (1, 2, 3, 5)))

    print("\n可分性 (有用组与无用组的平均相似度)")
    print(f"  {'模型':<22}{'有用':>8}{'无用':>8}{'间距':>8}")
    for label, key in rows:
        mu, mn, gap = _separation(scoped, key)
        print(f"  {label:<22}{mu:>8.3f}{mn:>8.3f}{gap:>8.3f}")

    print("\n间距越大, 越有希望把阈值提回 spec 的 0.7 而不漏召;")
    print("top3 越高, 收紧注入条数才越安全.")


if __name__ == "__main__":
    asyncio.run(main())
