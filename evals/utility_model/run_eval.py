"""小模型候选基准 runner.

用注册表里的**线上提示词**跑真实判定任务, 对比候选模型的准确率 / 延迟 / 成本.
提示词从 store 取而不是从 defaults 取 —— admin 在后台改过的版本才是生产实际
输入, 拿代码默认值测等于测了一个线上不存在的提示词.

Usage (容器内, /app):
    python -m evals.utility_model.run_eval \
        --models deepseek:deepseek-v4-flash ark:doubao-seed-2-0-mini-260428 \
        --repeats 3 --json /tmp/utility.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.db import db
from app.services.prompting.store import get_prompt_text
from app.services.prompting.utils import SafeDict
from app.services.runtime_config import ensure_loaded

from evals.utility_model.tasks import ALL_TASKS, Case, Task

# 元/百万 token. 官方价目, 缓存命中价单列 —— 判定类调用 prompt 前缀高度重复,
# 命中率高的模型实际成本会明显低于名义单价.
PRICING: dict[str, dict[str, float]] = {
    "deepseek-v4-flash": {"input": 1.0, "output": 2.0, "cached": 0.02},
    # 方舟 ≤32K 档官方价 (火山引擎定价页). 超过 32K 输入会翻倍, 但判定类 prompt
    # 都在 2k 以内, 用不到更高档.
    "doubao-seed-2-0-mini-260428": {"input": 0.2, "output": 2.0, "cached": 0.04},
    "doubao-seed-2-0-lite-260428": {"input": 0.6, "output": 3.6, "cached": 0.12},
    "doubao-seed-2-1-turbo-260628": {"input": 3.0, "output": 15.0, "cached": 0.6},
    "doubao-seed-character-260628": {"input": 0.8, "output": 2.0, "cached": 0.16},
    "qwen3.5-flash": {"input": 0.2, "output": 2.0, "cached": 0.04},
}


def build_model(spec: str):
    from langchain_openai import ChatOpenAI

    from app.config import settings

    provider, _, model = spec.partition(":")
    creds = {
        "deepseek": (settings.deepseek_api_key, settings.deepseek_base_url),
        "ark": (settings.ark_api_key, settings.ark_base_url),
        "dashscope": (settings.dashscope_api_key,
                      "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    }
    if provider not in creds:
        raise SystemExit(f"unknown provider {provider!r}; use {sorted(creds)}")
    api_key, base_url = creds[provider]
    if not api_key:
        raise SystemExit(f"no API key for {provider}")
    extra: dict[str, Any] = {}
    if provider == "ark":
        # 判定任务不需要思考链, 开着只会拖慢并烧 output token.
        extra["thinking"] = {"type": "disabled"}
    elif provider == "dashscope":
        extra["enable_thinking"] = False
    return ChatOpenAI(
        model=model, api_key=api_key, base_url=base_url, timeout=60,
        temperature=0, extra_body=extra or None,
    )


# --prompts 指向一份导出的线上提示词; 不给就实时从 store 读 (需要 DB/Redis).
# 导出模式的意义: 跑一轮要十几分钟, 而每次部署都会替换容器并杀掉容器内的进程
# —— 实测被这样打断两次. 导出后在任意机器上直连各家 API 跑, 不受部署影响.
_PROMPT_OVERRIDES: dict[str, str] = {}


async def render(task: Task, case: Case) -> str:
    template = _PROMPT_OVERRIDES.get(task.key)
    if template is None:
        template = str(await get_prompt_text(task.key))
    return template.format_map(SafeDict(task.params(case)))


async def run_one(model, task: Task, case: Case, prompt: str) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = await model.ainvoke(prompt)
    except Exception as e:  # noqa: BLE001 — 单点失败记录下来, 不终止整轮
        return {"error": str(e)[:120], "latency_ms": int((time.perf_counter() - started) * 1000)}
    latency_ms = int((time.perf_counter() - started) * 1000)
    raw = str(getattr(response, "content", "") or "")
    usage = getattr(response, "usage_metadata", None) or {}
    details = usage.get("input_token_details") or {}
    return {
        "raw": raw[:200],
        "got": task.parse(raw),
        "expected": case.expected,
        "latency_ms": latency_ms,
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "cached_tokens": int(details.get("cache_read", 0) or 0),
    }


def cost_cny(model_name: str, rows: list[dict]) -> float | None:
    price = PRICING.get(model_name)
    if not price:
        return None
    total = 0.0
    for r in rows:
        cached = min(r.get("cached_tokens", 0), r.get("input_tokens", 0))
        miss = r.get("input_tokens", 0) - cached
        total += (
            miss * price["input"]
            + cached * price["cached"]
            + r.get("output_tokens", 0) * price["output"]
        ) / 1_000_000
    return total


async def evaluate(spec: str, tasks: tuple[Task, ...], repeats: int,
                   concurrency: int) -> dict[str, Any]:
    model_name = spec.partition(":")[2]
    model = build_model(spec)
    sem = asyncio.Semaphore(concurrency)
    rows: list[dict[str, Any]] = []

    async def one(task: Task, case: Case, prompt: str):
        async with sem:
            r = await run_one(model, task, case, prompt)
            r.update({"task": task.name, "message": case.message})
            rows.append(r)

    jobs = []
    for task in tasks:
        for case in task.cases:
            prompt = await render(task, case)
            jobs += [one(task, case, prompt) for _ in range(repeats)]
    await asyncio.gather(*jobs)

    by_task: dict[str, dict[str, Any]] = {}
    for task in tasks:
        sub = [r for r in rows if r["task"] == task.name]
        ok = [r for r in sub if "error" not in r]
        correct = sum(1 for r in ok if r["got"] == r["expected"])
        unparsed = sum(1 for r in ok if r["got"] is None)
        by_task[task.name] = {
            "n": len(sub),
            "errors": len(sub) - len(ok),
            "unparsed": unparsed,
            "accuracy": round(correct / len(ok), 4) if ok else 0.0,
            "p50_ms": sorted(r["latency_ms"] for r in ok)[len(ok) // 2] if ok else 0,
        }

    ok_rows = [r for r in rows if "error" not in r]
    total_correct = sum(1 for r in ok_rows if r["got"] == r["expected"])
    return {
        "spec": spec,
        "model": model_name,
        "by_task": by_task,
        "accuracy": round(total_correct / len(ok_rows), 4) if ok_rows else 0.0,
        "errors": len(rows) - len(ok_rows),
        "p50_ms": sorted(r["latency_ms"] for r in ok_rows)[len(ok_rows) // 2] if ok_rows else 0,
        "p95_ms": sorted(r["latency_ms"] for r in ok_rows)[int(len(ok_rows) * 0.95)]
        if ok_rows else 0,
        "avg_input_tokens": round(sum(r["input_tokens"] for r in ok_rows) / len(ok_rows), 1)
        if ok_rows else 0,
        "avg_output_tokens": round(sum(r["output_tokens"] for r in ok_rows) / len(ok_rows), 1)
        if ok_rows else 0,
        "cost_per_1k_calls": (
            round(c / len(ok_rows) * 1000, 4)
            if ok_rows and (c := cost_cny(model_name, ok_rows)) is not None else None
        ),
        "rows": rows,
    }


def _print_statistics(results: list[dict[str, Any]]) -> None:
    """点估计之外: 配对比较 + 区间 + 稳定性 + 可疑标注."""
    from evals.utility_model.analyze import (
        paired_compare, score_by_case, stability, suspect_labels,
    )

    by_model = {r["model"]: score_by_case(r["rows"]) for r in results}
    suspects = suspect_labels(by_model)

    print("\n════════ 答案稳定性 (同一输入重复调用是否给同一答案) ════════")
    for name, scores in sorted(by_model.items(), key=lambda kv: -stability(kv[1])):
        print(f"  {name:<32} {stability(scores):>6.1%} 的用例答案稳定")

    if suspects:
        print("\n════════ 可疑标注 (所有模型全错 → 先怀疑标注) ════════")
        for task, message in suspects:
            print(f"  [{task}] «{message[:34]}»")
        print("  ↑ 这些用例已从下面的配对统计中剔除")

    ranked = sorted(results, key=lambda x: -x["accuracy"])
    if len(ranked) < 2:
        return
    best = ranked[0]["model"]
    print(f"\n════════ 配对比较: {best} vs 其他 (按用例聚类 bootstrap) ════════")
    for other in ranked[1:]:
        cmp = paired_compare(
            by_model[best], by_model[other["model"]], exclude=set(suspects),
        )
        lo, hi = cmp["ci95"]
        verdict = "显著更好" if lo > 0 else ("显著更差" if hi < 0 else "无法判定")
        print(f"\n  vs {other['model']}  (n={cmp['n_cases']} 条用例)")
        print(f"    准确率差值 {cmp['mean_diff']:+.1%}   95%CI [{lo:+.1%}, {hi:+.1%}]   {verdict}")
        print(f"    赢 {cmp['wins']} 条 / 平 {cmp['ties']} 条 / 输 {cmp['losses']} 条"
              f"   p(不优于)={cmp['p_not_better']:.3f}")
        if cmp["loss_cases"]:
            print("    输在:")
            for task, msg in cmp["loss_cases"][:5]:
                print(f"      [{task}] «{msg[:30]}»")


def print_report(results: list[dict[str, Any]], tasks: tuple[Task, ...]) -> None:
    print("\n════════ 汇总 ════════")
    head = f"{'模型':<32}{'总准确率':>9}{'p50':>7}{'p95':>7}{'元/千次':>10}{'失败':>6}"
    print(head)
    for r in sorted(results, key=lambda x: -x["accuracy"]):
        cost = f"{r['cost_per_1k_calls']:.4f}" if r["cost_per_1k_calls"] is not None else "—"
        print(f"{r['model']:<32}{r['accuracy']:>8.1%}{r['p50_ms']:>7}{r['p95_ms']:>7}"
              f"{cost:>10}{r['errors']:>6}")

    print("\n════════ 分任务准确率 ════════")
    names = [t.name for t in tasks]
    print(f"{'模型':<32}" + "".join(f"{n:>12}" for n in names))
    for r in sorted(results, key=lambda x: -x["accuracy"]):
        cells = "".join(f"{r['by_task'][n]['accuracy']:>11.0%} " for n in names)
        print(f"{r['model']:<32}{cells}")

    _print_statistics(results)

    print("\n════════ 错判明细 ════════")
    for r in results:
        bad = [
            x for x in r["rows"]
            if "error" not in x and x["got"] != x["expected"]
        ]
        if not bad:
            print(f"\n{r['model']}: 无错判")
            continue
        seen: dict[tuple, int] = defaultdict(int)
        for x in bad:
            seen[(x["task"], x["message"], x["expected"], x["got"])] += 1
        print(f"\n{r['model']}: {len(bad)} 次错判")
        for (task, msg, exp, got), n in sorted(seen.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  ×{n} [{task}] «{msg[:26]}» 期望={exp} 实际={got}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True,
                        help="provider:model, e.g. ark:doubao-seed-2-0-mini-260428")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--task", help="只跑某个任务名")
    parser.add_argument("--json", dest="json_out")
    parser.add_argument(
        "--prompts",
        help="导出的线上提示词 JSON ({key: text}); 给了就不连 DB",
    )
    args = parser.parse_args()

    tasks = tuple(t for t in ALL_TASKS if not args.task or t.name == args.task)
    n_cases = sum(len(t.cases) for t in tasks)

    if args.prompts:
        _PROMPT_OVERRIDES.update(json.loads(Path(args.prompts).read_text(encoding="utf-8")))
        missing = [t.key for t in tasks if t.key not in _PROMPT_OVERRIDES]
        if missing:
            raise SystemExit(f"导出的提示词缺少: {missing}")
    else:
        await db.connect()
        await ensure_loaded()
    try:
        print(f"提示词来源: {'导出文件 ' + args.prompts if args.prompts else '线上 store'}")
        print(f"任务 {len(tasks)} 个 / 用例 {n_cases} 条 / 每条 {args.repeats} 次")
        print(f"= 每个模型 {n_cases * args.repeats} 次调用\n")
        results = []
        for spec in args.models:
            print(f"→ {spec}")
            results.append(await evaluate(spec, tasks, args.repeats, args.concurrency))
        print_report(results, tasks)
        if args.json_out:
            Path(args.json_out).write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            print(f"\nwrote {args.json_out}")
    finally:
        if not args.prompts:
            await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
