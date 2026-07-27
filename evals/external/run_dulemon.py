"""用 DuLeMon 的第三方人工标注检验记忆相关模块.

**先读这段, 它决定哪些数字能信.**

总的结论: DuLeMon 的**正类可信, 负类不可信**. 它是召回下限的回归护栏, 不是
准确率基准. 每段对话只预设 5-6 条 persona, 用户说了别的可记事实 (「我买了辆
宝马」) 同样不会被标 —— 负类里混着大量真该记/真该查的内容, 所以特异度低不能
当错误读, 报一个总准确率更是会误导.

`--task judgement` (记忆预筛) — 正类干净.
    DuLeMon 在 `Usr:` 句上标 persona 编号, 含义是"这句在陈述一条值得长期保存的
    事实". 标注对象就是被判的那句话本身, 没有中间环节, 所以**被标了却没记 = 真
    漏记**. 召回是这个任务能给出的可信结论 (实测 92.6%).

`--task relevance` (相关度闸门) — 连正类都有系统性混淆, **不要用它下结论**.
    这里的标签只能间接取自"机器人的回复用没用 persona", 而这有两重错位:
      · DuLeMon 写手被要求把 persona 织进回复, 用不用很大程度是写作风格选择,
        跟用户那句话有没有可匹配内容无关. 实测「嗨，小度，在吗？」被标成该查,
        只因为机器人问候时叫了用户名字.
      · 反过来「现在我在东方卫视工作」这类明显该查的自我披露, 会因为写手当时
        没引用已存 persona 而落进不该查那侧.
    加 U-persona 过滤 (丢掉机器人只用自身人设的回合) 能缓解但消不掉 —— 过滤后
    闸门在"该查"侧的触发率仍低于"未用"侧. 保留这个模式是为了留住这个结论本身,
    以及给后续想换映射方式的人一个起点.

用法:
    python -m evals.external.run_dulemon --data /tmp/dulemon/DuLeMon/both/test.json \
        --task judgement --model dashscope:qwen3.5-flash --limit 400
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from pathlib import Path

from evals.external.dulemon import (
    build_judgement_cases, build_relevance_cases, load_dialogues,
)
from evals.utility_model.run_eval import build_model
from evals.utility_model.tasks import TASK_MEMORY_JUDGEMENT, TASK_MEMORY_RELEVANCE


def _prompt_from_git(revision: str, marker: str) -> str:
    src = subprocess.run(
        ["git", "show", f"{revision}:app/services/prompting/defaults.py"],
        capture_output=True, text=True, check=True,
    ).stdout
    start = src.index(marker)
    end = src.index('"""', src.index('"""', start) + 3) + 3
    ns: dict = {}
    exec(src[start:end], ns)
    return ns[marker.split(" =")[0].strip()]


async def _classify(model, prompt_text: str, case, sem, task) -> str | None:
    rendered = prompt_text.format(
        message=case.message, context=getattr(case, "context", "")
    )
    async with sem:
        try:
            response = await asyncio.wait_for(model.ainvoke(rendered), timeout=60)
        except Exception:
            return None
    return task.parse(getattr(response, "content", "") or str(response))


def _report_relevance(name: str, cases, verdicts) -> dict:
    should = [v for c, v in zip(cases, verdicts) if c.expected == "中" and v]
    neednt = [v for c, v in zip(cases, verdicts) if c.expected == "弱" and v]
    hot = ("中", "强")
    recall = sum(1 for v in should if v in hot) / len(should) if should else 0.0
    fire = sum(1 for v in neednt if v in hot) / len(neednt) if neednt else 0.0
    print(f"\n[{name}]")
    print(f"  回复确实回忆了用户信息  n={len(should):<5} 闸门判要查 = 召回 {recall:.1%}")
    print(f"  回复未回忆 (仅误报上界)  n={len(neednt):<5} 闸门判要查 {fire:.1%}")
    print(f"  提升度 {recall / fire:.2f}  (>1 才说明闸门与标注同向)")
    return {"recall": recall, "fire_on_unused": fire,
            "n_should": len(should), "n_neednt": len(neednt)}


def _report_judgement(name: str, cases, verdicts) -> dict:
    graded = [(c, v) for c, v in zip(cases, verdicts) if v]
    correct = sum(1 for c, v in graded if v == c.expected)
    pos = [(c, v) for c, v in graded if c.expected == "记"]
    neg = [(c, v) for c, v in graded if c.expected == "不记"]
    recall = sum(1 for _, v in pos if v == "记") / len(pos) if pos else 0.0
    spec = sum(1 for _, v in neg if v == "不记") / len(neg) if neg else 0.0
    print(f"\n[{name}]  n={len(graded)}")
    print(f"  准确率 {correct / len(graded):.1%}")
    print(f"  该记的记住了 (召回)  {recall:.1%}   n={len(pos)}")
    print(f"  不该记的没记 (特异)  {spec:.1%}   n={len(neg)}")
    return {"accuracy": correct / len(graded) if graded else 0.0,
            "recall": recall, "specificity": spec, "n": len(graded)}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--task", choices=("judgement", "relevance"), default="judgement")
    ap.add_argument("--model", default="dashscope:qwen3.5-flash")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--compare-rev", help="同时跑这个 git 版本的提示词做对照")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--json")
    args = ap.parse_args()

    dialogues = load_dialogues(Path(args.data))
    if args.task == "judgement":
        task = TASK_MEMORY_JUDGEMENT
        marker = "MEMORY_JUDGEMENT_USER_PROMPT = "
        report = _report_judgement
        cases = build_judgement_cases(dialogues, limit=args.limit)
    else:
        task = TASK_MEMORY_RELEVANCE
        marker = "MEMORY_RELEVANCE_PROMPT = "
        report = _report_relevance
        cases = build_relevance_cases(dialogues, limit=args.limit)
    print(f"DuLeMon {Path(args.data).parent.name}/{Path(args.data).stem}: "
          f"{len(dialogues)} 段对话 → 抽 {len(cases)} 条用例")

    from app.services.prompting import defaults
    current = getattr(defaults, marker.split(" =")[0].strip())
    model = build_model(args.model)
    sem = asyncio.Semaphore(args.concurrency)

    async def run(prompt_text: str) -> list[str | None]:
        return list(await asyncio.gather(
            *(_classify(model, prompt_text, c, sem, task) for c in cases)
        ))

    results = {}
    if args.compare_rev:
        before = await run(_prompt_from_git(args.compare_rev, marker))
        results["before"] = report(f"改前 ({args.compare_rev})", cases, before)
    after = await run(current)
    results["after"] = report("当前", cases, after)

    if args.compare_rev:
        for key in ("recall", "accuracy"):
            if key in results["after"]:
                delta = results["after"][key] - results["before"][key]
                print(f"\n{key} {delta:+.1%}")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"summary": results,
             "rows": [{"message": c.message, "expected": c.expected,
                       "after": a} for c, a in zip(cases, after)]},
            ensure_ascii=False, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    asyncio.run(main())
