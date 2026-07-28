"""用 sentence-transformers 给一批文本产出向量, 供 embedding 对比使用.

存在的理由: 有些候选模型 Ollama 没有 manifest (例如 ritrieve_zh_v1), 但评测
只需要向量, 不需要它先能在生产里跑起来. 先用官方实现测出值不值得, 再决定要不要
为它折腾部署 —— 反过来做会为一个可能落选的模型白花力气.

⚠️ 用官方实现测出的分数是该模型的**上限**. 若最终要走 llama.cpp/Ollama, 带
sentence-transformers Dense 投影层的模型 (ritrieve_zh_v1 就是, 1024→1792) 在
转 GGUF 时那一层可能被丢掉, 届时实际效果会低于这里测到的值, 必须重测.

依赖不进项目 venv —— torch 近 2GB, 而这是一次性评测:
    python3 -m venv /tmp/st_venv && /tmp/st_venv/bin/pip install sentence-transformers
    /tmp/st_venv/bin/python embed_with_sentence_transformers.py <model> texts.json out.json
"""

from __future__ import annotations

import json
import statistics
import sys
import time

from sentence_transformers import SentenceTransformer


def main() -> None:
    model_id, texts_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    texts = json.loads(open(texts_path).read())

    model = SentenceTransformer(model_id, device="cpu")
    model.encode(["预热"])  # 首次前向含惰性初始化, 计进延迟会高估

    latencies: list[float] = []
    vectors: dict[str, list[float]] = {}
    for i, text in enumerate(texts):
        started = time.perf_counter()
        vector = model.encode([text], show_progress_bar=False)[0]
        latencies.append((time.perf_counter() - started) * 1000)
        vectors[text] = [float(x) for x in vector]
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(texts)}", flush=True)

    open(out_path, "w").write(json.dumps({model_id: vectors}))
    latencies.sort()
    print(f"{model_id}  n={len(latencies)}  dim={len(next(iter(vectors.values())))}  "
          f"p50={statistics.median(latencies):.0f}ms  "
          f"p95={latencies[int(len(latencies) * 0.95) - 1]:.0f}ms")


if __name__ == "__main__":
    main()
