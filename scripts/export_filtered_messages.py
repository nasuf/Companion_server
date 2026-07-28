"""导出被启发式过滤器拒掉的真实用户消息, 连同它的对话上下文.

背景: 619 条真实用户消息里, 记忆管线第一级 (filter.should_extract_memory) 就拒掉
348 条 (56%). 这一级是纯规则加权打分, 要求两个以上信号才放行, 而且**只看单条消息
的字面**.

问题在于对话里的意义常常是上下文给的:

    AI: 你今天还好吗
    用户: 不好          ← 单看两个字什么都不是, 放回上下文是明确的情绪信号

所以要判断这 56% 拒得对不对, 必须带上前一句 AI 的话一起看 —— 只给模型看被拒的
那句, 等于用跟过滤器一样残缺的视角去评价过滤器.

用法 (在生产容器内):
    python export_filtered_messages.py /tmp/filtered.json
"""

from __future__ import annotations

import asyncio
import json
import sys

from app.db import db
from app.services.memory.recording.filter import should_extract_memory


async def main() -> None:
    out_path = sys.argv[1]
    await db.connect()
    rows = await db.query_raw(
        """
        SELECT id, conversation_id, role, content, created_at
        FROM messages
        WHERE content IS NOT NULL AND content <> ''
        ORDER BY conversation_id, created_at
        """
    )
    await db.disconnect()

    # 按会话串起来才能取到前一句
    previous_by_conversation: dict[str, str] = {}
    rejected: list[dict] = []
    accepted = 0
    for row in rows:
        conversation = row["conversation_id"]
        content = row["content"]
        if row["role"] != "user":
            previous_by_conversation[conversation] = content
            continue
        if should_extract_memory(content):
            accepted += 1
        else:
            rejected.append({
                "message": content,
                "prev_ai": previous_by_conversation.get(conversation, ""),
            })
        previous_by_conversation[conversation] = content

    open(out_path, "w").write(json.dumps(rejected, ensure_ascii=False))
    total = accepted + len(rejected)
    print(f"用户消息 {total} 条: 放行 {accepted}, 拒绝 {len(rejected)} "
          f"({100 * len(rejected) / total:.0f}%)")
    print(f"→ {out_path}")
    with_context = sum(1 for r in rejected if r["prev_ai"])
    print(f"其中 {with_context} 条有前文 AI 发言可供判断")


if __name__ == "__main__":
    asyncio.run(main())
