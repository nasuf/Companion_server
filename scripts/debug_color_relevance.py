"""调试脚本: 直接调 memory.relevance 分类器, 看它对 '你喜欢什么颜色啊' 输出什么.

如果输出 '弱', 就证实了 relevance 分类是误杀根因 (而不是向量召回).
"""

from __future__ import annotations

import asyncio

from app.services.memory.retrieval.relevance import classify_memory_relevance


CASES = [
    # ── 应 强 (问 AI 身份/偏好/经历) ──
    ("你喜欢什么颜色啊", ""),
    ("你叫什么名字", ""),
    ("你最喜欢哪首歌", ""),
    # 表面不像问询但本质是问 AI 信息
    ("你这人怎么样啊", ""),
    ("你这两天都在干什么", ""),
    ("你这阵子心态咋样", ""),
    ("你以前学过什么", ""),
    # 用户问自己的事
    ("我刚才说了什么", ""),
    ("我之前说过我喜欢吃什么来着", ""),
    # 追问/纠正
    ("你还记得我们上次聊过什么吗", ""),
    ("你不是说过你喜欢小狗吗", ""),
    # ── 应 中 (有具体话题可以呼应) ──
    ("最近又去爬山了", ""),  # 若 AI 有"喜欢爬山"记忆, 可呼应
    ("我今天看了那部电影", ""),  # 模糊, 但话题是具体的
    # ── 应 弱 (纯情绪/招呼/抽象闲聊) ──
    ("你好", ""),
    ("嗯嗯", ""),
    ("哈哈", ""),
    ("今天天气真好", ""),
    ("我有点烦", ""),
    ("人生好难啊", ""),
    ("是吗", ""),
    # ── 边界 / 易混 ──
    ("我刚才一直在想你说的话", ""),  # 暗指之前对话内容, 中/强
    ("你猜我今天遇到了什么", ""),  # 用户引出新事, 弱-中
]


async def main() -> None:
    print(f"{'message':<40} {'context':<30} -> level")
    print("-" * 90)
    for msg, ctx in CASES:
        try:
            level = await classify_memory_relevance(msg, context=ctx)
            marker = "🟢" if level in ("strong", "medium") else "🔴"
            print(f"{msg!r:<40} {(ctx[:25] + '...' if len(ctx) > 25 else ctx)!r:<30} -> {marker} {level}")
        except Exception as e:
            print(f"{msg!r:<40} ERROR: {e}")


if __name__ == "__main__":
    asyncio.run(main())
