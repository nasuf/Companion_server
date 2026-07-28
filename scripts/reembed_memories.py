"""用新 embedding 模型回填 memory_embeddings.embedding_next.

换模型的三步里的第二步:

    1. 迁移 20260728030000 加 embedding_next 列 (可空, 线上无感)
    2. **本脚本**把 8181 条记忆用新模型算一遍写进新列 —— 老列继续服务检索,
       这期间线上完全正常
    3. 迁移 20260728040000 在部署停机窗口内原子换列

为什么不直接覆盖老列: 两个模型的向量不在同一空间. 边覆盖边服务意味着查询向量
和库里向量分属两个空间, 相似度是纯噪声 —— 那比暂时查不到更糟, AI 会拿着无关
记忆一本正经地说错话.

可重入: 只处理 embedding_next IS NULL 的行, 中断后重跑即可. 部署前应再跑一次
把期间新写入的记忆补上 —— 换列迁移会断言无残留 NULL, 补不齐就让部署失败, 而
不是带着半套向量上线.

并发默认压得很低: Ollama 跟线上服务共用 4 核, 打满会拖慢正在聊天的用户.

用法 (在生产容器内):
    python reembed_memories.py                 # 回填
    python reembed_memories.py --check         # 只看剩余量
"""

from __future__ import annotations

import argparse
import asyncio
import time

import httpx

from app.config import settings
from app.db import db

OLLAMA_URL = "http://ollama:11434/api/embeddings"
BATCH = 100


async def _remaining() -> int:
    rows = await db.query_raw(
        "SELECT count(*) AS c FROM memory_embeddings WHERE embedding_next IS NULL"
    )
    return int(rows[0]["c"])


async def _memory_text(memory_id: str) -> str | None:
    rows = await db.query_raw(
        "SELECT content FROM memories_ai WHERE id = $1 "
        "UNION ALL SELECT content FROM memories_user WHERE id = $1",
        memory_id,
    )
    return rows[0]["content"] if rows else None


async def _embed(client: httpx.AsyncClient, model: str, text: str) -> list[float]:
    response = await client.post(OLLAMA_URL, json={"model": model, "prompt": text})
    response.raise_for_status()
    return response.json()["embedding"]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--concurrency", type=int, default=2,
                    help="Ollama 与线上服务共用 CPU, 调高会拖慢在线聊天")
    ap.add_argument("--model", default=None, help="默认取 settings.embedding_model")
    args = ap.parse_args()

    await db.connect()
    model = args.model or settings.embedding_model
    todo = await _remaining()
    print(f"model={model}  待回填 {todo} 行")
    if args.check or todo == 0:
        await db.disconnect()
        return

    started = time.time()
    done = failed = 0
    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(timeout=120) as client:
        while True:
            rows = await db.query_raw(
                "SELECT memory_id FROM memory_embeddings "
                "WHERE embedding_next IS NULL LIMIT $1", BATCH
            )
            if not rows:
                break

            async def one(memory_id: str) -> bool:
                text = await _memory_text(memory_id)
                if not text:
                    # 记忆行没了, 向量是孤儿. 不写就会卡住换列迁移, 直接删掉.
                    await db.execute_raw(
                        "DELETE FROM memory_embeddings WHERE memory_id = $1", memory_id
                    )
                    return True
                async with semaphore:
                    try:
                        vector = await _embed(client, model, text)
                    except Exception as exc:
                        print(f"  FAIL {memory_id}: {str(exc)[:80]}")
                        return False
                literal = "[" + ",".join(str(x) for x in vector) + "]"
                await db.execute_raw(
                    "UPDATE memory_embeddings SET embedding_next = $1::extensions.vector "
                    "WHERE memory_id = $2", literal, memory_id,
                )
                return True

            results = await asyncio.gather(*(one(r["memory_id"]) for r in rows))
            done += sum(1 for ok in results if ok)
            failed += sum(1 for ok in results if not ok)
            if not any(results):
                print("整批全失败, 停止以免空转")
                break
            elapsed = time.time() - started
            left = await _remaining()
            rate = done / elapsed if elapsed else 0
            eta = left / rate / 60 if rate else 0
            print(f"  已完成 {done}  失败 {failed}  剩余 {left}  "
                  f"{rate:.1f} 行/秒  预计还需 {eta:.1f} 分钟", flush=True)

    left = await _remaining()
    print(f"\n结束: 完成 {done}, 失败 {failed}, 仍为 NULL {left}")
    if left:
        print("仍有残留 —— 重跑本脚本. 换列迁移会因残留而失败, 这是有意的.")
    await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
