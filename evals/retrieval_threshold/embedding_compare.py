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

已测过并落选的本地模型 (2026-07, 免得重复劳动):

    模型                       评审1 top3/5   评审2 top3/5   维度
    qwen3-embedding:0.6b       50% / 73%     50% / 69%     1024   ← 本地最佳
    bge-m3 (现网)               44% / 64%     42% / 61%     1024
    embeddinggemma             40% / 59%     44% / 61%      768   落选
    snowflake-arctic-embed2    37% / 57%     42% / 63%     1024   落选
    granite-embedding          Ollama 拒绝提供 embedding 接口 (HTTP 500)

后两个在两套标签下都不如 bge-m3. embeddinggemma 还是 768 维, 换它除了重算向量
还要改 pgvector 列定义.

尚未测的一个: ritrieve_zh_v1 (0.3B, 中文专用) C-MTEB 检索 76.97, 高于
qwen3-embedding-0.6B 的 71.03 且体积减半, 但 Ollama 无 manifest, 要自己转 GGUF.
换完之后若检索质量仍不够, 这是下一个值得试的.

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
# 云端 API 候选. 本地 Ollama 模型的向量由 scripts/embed_with_ollama.py 在服务器
# 上产出后并入同一份缓存, 通过 --local 声明参与比较 —— 本机连不到那个 Ollama.
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
    ap.add_argument("--local", nargs="*", default=[],
                    help="缓存里已有向量的本地模型名, 不重新调 API")
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

    for model in list(CANDIDATES) + args.local:
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
    rows = ([("bge-m3 (现网/本地)", "sim")]
            + [(f"{m} (本地)", m) for m in args.local]
            + [(f"{m} (API)", m) for m in CANDIDATES])
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
