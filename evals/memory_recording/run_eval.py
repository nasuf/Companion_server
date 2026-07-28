"""记忆管线第一级过滤器漏掉了多少该记的东西.

`filter.should_extract_memory` 是纯规则加权打分, 拒掉的消息**不会进入后面任何
一级** —— 小模型预筛和大模型抽取都看不到它们. 所以它的假阴性是整条管线里代价最
高的一种错误: 后面再准也补不回来.

真实流量上它拒掉 56% (619 条里 348 条). 这个数字本身说明不了对错, 得看拒掉的
到底是不是废话.

关键的方法取舍: **判定时必须带上前一句 AI 的话**. 对话里的意义常常是上下文给的 ——

    AI: 你今天还好吗
    用户: 不好          ← 单看两个字什么都不是, 放回上下文是明确的情绪信号

而过滤器只看单条消息字面. 如果只把被拒的那句给评审看, 就是用跟过滤器一样残缺的
视角去评价过滤器, 只能得出"它拒得对"的同义反复.

用法:
    python -m evals.memory_recording.run_eval --filtered /tmp/filtered_msgs.json \\
        --judge deepseek:deepseek-v4-pro --sample 120
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from collections import Counter
from pathlib import Path

from evals.utility_model.run_eval import build_model

JUDGE_PROMPT = """你在检查一个 AI 伴侣的记忆系统有没有漏记东西。

这是一段对话片段：

AI：{prev_ai}
用户：{message}

系统判定用户这句话「不值得记进长期记忆」，被直接丢弃了。判断这个决定对不对。

值得记 = 这句话（结合上文）透露了关于用户的某个可以在以后用上的信息：
  · 事实：身份、经历、正在做的事、生活状况
  · 偏好：喜欢/讨厌什么、习惯
  · 情绪与状态：当下的心情、压力、身体状况 —— 哪怕只有两三个字
  · 关系信号：对 AI 的态度变化、约定、称呼
注意上下文：「不好」单独看没有信息，但如果 AI 刚问「你今天还好吗」，它就是一条
明确的情绪记录。

不值得记 = 纯招呼、纯语气词、对 AI 的提问、对上一句的机械应答、
以及虽然有内容但只关于 AI 自己而不涉及用户的话。

只输出一个词：该记 / 不该记"""


def parse(raw: str) -> str | None:
    text = (raw or "").strip()
    # 顺序要紧: "不该记" 里含 "该记"
    for label in ("不该记", "该记"):
        if label in text:
            return label
    return None


async def _judge(model, item: dict, sem: asyncio.Semaphore) -> str | None:
    prompt = JUDGE_PROMPT.format(
        prev_ai=item.get("prev_ai") or "(无)", message=item["message"]
    )
    async with sem:
        for _ in range(2):
            try:
                response = await asyncio.wait_for(model.ainvoke(prompt), timeout=90)
            except Exception:
                continue
            verdict = parse(getattr(response, "content", "") or str(response))
            if verdict:
                return verdict
    return None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filtered", required=True)
    ap.add_argument("--judge", default="deepseek:deepseek-v4-pro")
    ap.add_argument("--sample", type=int, default=120)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json")
    args = ap.parse_args()

    items = json.loads(Path(args.filtered).read_text())
    if len(items) > args.sample:
        items = random.Random(args.seed).sample(items, args.sample)
    print(f"抽查 {len(items)} 条被过滤器拒掉的真实消息，评审 = {args.judge}\n")

    model = build_model(args.judge)
    sem = asyncio.Semaphore(args.concurrency)
    verdicts = await asyncio.gather(*(_judge(model, i, sem) for i in items))
    for item, verdict in zip(items, verdicts):
        item["verdict"] = verdict

    graded = [i for i in items if i["verdict"]]
    counts = Counter(i["verdict"] for i in graded)
    missed = counts["该记"]
    rate = missed / len(graded) if graded else 0.0
    print(f"判定成功 {len(graded)}/{len(items)}")
    print(f"  该记却被丢掉 (假阴性)  {missed:>4}  {rate:.0%}")
    print(f"  确实不该记            {counts['不该记']:>4}")

    print("\n被误丢的样例:")
    for item in [i for i in graded if i["verdict"] == "该记"][:12]:
        prev = (item.get("prev_ai") or "")[:26]
        print(f"  AI「{prev}」→ 用户「{item['message'][:30]}」")

    print("\n这一级的假阴性没有任何后续环节能补 —— 被拒的消息不会进预筛, 也不会")
    print("进抽取。整条管线的记忆上限就是这里放行的那部分。")

    if args.json:
        Path(args.json).write_text(json.dumps(graded, ensure_ascii=False, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    asyncio.run(main())
