"""用服务器本地 Ollama 的某个模型给一批文本产出向量, 供 embedding 对比使用.

对比脚本 (evals/retrieval_threshold/embedding_compare.py) 跑在本机, 但本地
Ollama 只在生产服务器上. 这个脚本在服务器上跑, 把向量导出成同样的
{model: {text: vector}} 结构, 拷回去合并进缓存即可参与同一张表的比较.

顺带量一下单条延迟 —— 换 embedding 是热路径改动, 质量再好也要看得起延迟.

用法 (在服务器容器内):
    python embed_with_ollama.py qwen3-embedding:0.6b texts.json out.json
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time

import httpx

OLLAMA_URL = "http://ollama:11434/api/embeddings"


async def main() -> None:
    model, texts_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    texts = json.loads(open(texts_path).read())
    vectors: dict[str, list[float]] = {}
    latencies: list[float] = []

    async with httpx.AsyncClient(timeout=120) as client:
        # 预热: 首次调用含模型加载, 计进延迟会严重高估稳态表现
        await client.post(OLLAMA_URL, json={"model": model, "prompt": "warmup"})
        for i, text in enumerate(texts):
            started = time.perf_counter()
            response = await client.post(
                OLLAMA_URL, json={"model": model, "prompt": text}
            )
            response.raise_for_status()
            latencies.append((time.perf_counter() - started) * 1000)
            vectors[text] = response.json()["embedding"]
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(texts)}", flush=True)

    open(out_path, "w").write(json.dumps({model: vectors}))
    latencies.sort()
    print(f"{model}  n={len(latencies)}  dim={len(next(iter(vectors.values())))}  "
          f"p50={statistics.median(latencies):.0f}ms  "
          f"p95={latencies[int(len(latencies) * 0.95) - 1]:.0f}ms")


if __name__ == "__main__":
    asyncio.run(main())
