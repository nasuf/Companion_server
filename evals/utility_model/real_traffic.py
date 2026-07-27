"""把相关度闸门跑在真实用户消息上, 比较改前/改后的判定分布.

`tasks.py` 的标注集是手写的, ground truth 取自提示词自己声明的规则 —— 它回答的
是"模型听不听得懂指令", 不是"这条策略在真实流量上对不对". 这个脚本补另一半:
不需要标注, 直接看同一批真实消息在两版提示词下的判定差异, 以及差异集中在哪些
形态上. 判弱 = 整轮不查记忆, 所以判弱率的变化就是检索覆盖面的变化.

用法:
    python -m evals.utility_model.real_traffic --messages /tmp/real_msgs.json \
        --model dashscope:qwen3.5-flash --json /tmp/real_gate.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from collections import Counter
from pathlib import Path

from evals.utility_model.run_eval import build_model
from evals.utility_model.tasks import TASK_MEMORY_RELEVANCE as _TASK


def _prompt_from_git(revision: str) -> str:
    """Pull the prompt default out of a git revision so 'before' is the real
    previous text rather than a reconstruction."""
    src = subprocess.run(
        ["git", "show", f"{revision}:app/services/prompting/defaults.py"],
        capture_output=True, text=True, check=True,
    ).stdout
    ns: dict = {}
    marker = "MEMORY_RELEVANCE_PROMPT = "
    start = src.index(marker)
    end = src.index('"""', src.index('"""', start) + 3) + 3
    exec(src[start:end], ns)
    return ns["MEMORY_RELEVANCE_PROMPT"]


async def _classify(model, prompt_text: str, message: str) -> str | None:
    rendered = prompt_text.format(message=message, context="")
    try:
        response = await model.ainvoke(rendered)
    except Exception:
        return None
    raw = getattr(response, "content", None) or str(response)
    return _TASK.parse(raw)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--messages", required=True)
    ap.add_argument("--model", default="dashscope:qwen3.5-flash")
    ap.add_argument("--before-rev", default="HEAD~2")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--json")
    args = ap.parse_args()

    messages = json.loads(Path(args.messages).read_text())
    before_prompt = _prompt_from_git(args.before_rev)
    from app.services.prompting import defaults
    after_prompt = defaults.MEMORY_RELEVANCE_PROMPT
    assert before_prompt != after_prompt, "两版提示词相同, 对比无意义"

    model = build_model(args.model)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(msg: str) -> dict:
        async with sem:
            before, after = await asyncio.gather(
                _classify(model, before_prompt, msg),
                _classify(model, after_prompt, msg),
            )
        return {"message": msg, "before": before, "after": after}

    rows = await asyncio.gather(*(one(m) for m in messages))

    b = Counter(r["before"] for r in rows)
    a = Counter(r["after"] for r in rows)
    n = len(rows)
    print(f"\n真实用户消息 {n} 条\n")
    print(f"{'判定':<6}{'改前':>10}{'改后':>10}")
    for lvl in ("弱", "中", "强", None):
        print(f"{str(lvl):<6}{b.get(lvl,0):>10}{a.get(lvl,0):>10}")
    print(f"\n判弱率  {b.get('弱',0)/n:.1%} → {a.get('弱',0)/n:.1%}")

    flipped_up = [r for r in rows if r["before"] == "弱" and r["after"] in ("中", "强")]
    flipped_down = [r for r in rows if r["before"] in ("中", "强") and r["after"] == "弱"]
    print(f"\n弱 → 中/强 (新增检索) {len(flipped_up)} 条")
    for r in flipped_up[:25]:
        print(f"    {r['after']}  {r['message'][:50]}")
    print(f"\n中/强 → 弱 (不再检索) {len(flipped_down)} 条")
    for r in flipped_down[:25]:
        print(f"    {r['before']}→弱  {r['message'][:50]}")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    asyncio.run(main())
