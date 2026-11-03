"""CHAT01.10.2 给回放结果打分: 与 10.1 生产指标同一口径 (app/services/effect/repetition.py)。

每个采样: 本轮气泡对用例里固定的 "之前 3 轮 AI 回复" 算相似度, ≥0.60 的进评判, 一轮里有一条
判为原样照抄或换说法复述就算重复。另记回放器里那次 "是否回应了当前这句" 的评判、气泡数、字数、
主回复耗时与费用。评判结果按提示词指纹缓存在输出目录旁, 重跑只补没评过的。

    python -m evals.reply_repetition.score --cases <cases.json> --runs <output-dir> \\
        --arms current,nofix --output <scores.json>
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from app.services.effect.repetition import (
    EMBED_BATCH,
    MIN_KEY_CHARS,
    REPEAT_LABELS,
    Turn,
    TurnScore,
    Unjudgeable,
    find_candidates,
    judge_prompt,
    text_key,
    turn_label,
)
from evals.reply_repetition.run_replay import call_cost


def turn_score(case: dict, bubbles: list[str]) -> TurnScore:
    previous = [Turn("assistant", [{"content": b} for b in turn]) for turn in case["previous_ai_turns"] if turn]
    reply = Turn("assistant", [{"content": b} for b in bubbles])
    return TurnScore(reply, case["situation"], case["judge_user_text"], previous,
                     [b for b in bubbles if len(text_key(b)) >= MIN_KEY_CHARS])


def is_main_call(call: dict) -> bool:
    """主回复调用 (带 system 段的对话); 回放器里其余辅助调用只有一条 user 消息。"""
    messages = call.get("messages") or []
    return bool(messages) and messages[0].get("role") == "system"


def samples(runs_dir: Path, arm: str, case_ids: set[str]) -> list[dict]:
    """一组的全部采样, 只读跑完的证据文件; 没跑完或失败的用例在 compare 里按缺失计。"""
    out = []
    for path in sorted((runs_dir / arm).glob("rr-*.json")):
        detail = json.loads(path.read_text())
        if "total" not in detail:
            continue
        for run in detail.get("runs", []):
            if run["case"] not in case_ids:
                continue
            if "bubbles" not in run:  # 老版本回放器: 气泡边界丢了, 不能按气泡判, 也不能混进来比
                raise ValueError(f"{path.name}: replay predates bubble capture; rerun this arm")
            main = [c for c in run.get("calls", []) if is_main_call(c)]
            # 费用算本轮所有走主模型的调用 (含回放器里的辅助调用), 耗时只算主回复
            costs = [call_cost(c.get("usage")) for c in run.get("calls", []) if "messages" in c]
            out.append({
                "arm": arm, "case": run["case"], "sample": run["sample"],
                "bubbles": run["bubbles"],
                "responded": run["judge"].get("pass"),
                "main_calls": len(main),
                "seconds": round(sum(c.get("duration_seconds") or 0 for c in main), 3),
                "cost_cny": None if any(c is None for c in costs) else round(sum(costs), 6),
            })
    return out


async def score(cases: list[dict], rows: list[dict], cache_path: Path, embedder, judge) -> list[dict]:
    by_id = {c["id"]: c for c in cases}
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    scored = [(row, turn_score(by_id[row["case"]], row["bubbles"])) for row in rows]
    texts = sorted({b for _, s in scored for b in s.bubbles}
                   | {b for _, s in scored for t in s.previous for b in t.bubbles if len(text_key(b)) >= MIN_KEY_CHARS})
    vectors: dict[str, list[float]] = {}
    gate = asyncio.Semaphore(8)

    async def embed(batch: list[str]) -> None:
        async with gate:
            try:
                vectors.update(zip(batch, await embedder(batch)))
            except Unjudgeable:
                for text in batch:
                    try:
                        vectors[text] = (await embedder([text]))[0]
                    except Unjudgeable:
                        pass

    await asyncio.gather(*[embed(texts[i:i + EMBED_BATCH]) for i in range(0, len(texts), EMBED_BATCH)])

    async def judge_one(s: TurnScore, candidate) -> None:
        prompt = judge_prompt(s, candidate)
        key = hashlib.sha256(prompt.encode()).hexdigest()
        if key not in cache:
            async with gate:
                try:
                    cache[key] = await judge(prompt)
                except Unjudgeable:
                    cache[key] = "unjudgeable"
                except Exception as exc:  # 不缓存失败, 下次重跑再评
                    s.labels.append("failed")
                    print(json.dumps({"judge_failed": type(exc).__name__}))
                    return
        s.labels.append(cache[key])

    for _, s in scored:
        find_candidates(s, vectors)
    await asyncio.gather(*[judge_one(s, c) for _, s in scored for c in s.candidates])
    cache_path.write_text(json.dumps(cache, ensure_ascii=False))
    cache_path.chmod(0o600)
    results = []
    for row, s in scored:
        label = turn_label(s)
        results.append({**{k: v for k, v in row.items() if k != "bubbles"},
                        "bubble_count": len(row["bubbles"]), "chars": sum(len(b) for b in row["bubbles"]),
                        "label": label, "repeated": label in REPEAT_LABELS if label not in ("failed", "unjudgeable") else None,
                        "verbatim": label == "verbatim" if label not in ("failed", "unjudgeable") else None,
                        "max_similarity": s.max_similarity})
    return results


async def main(args) -> dict:
    from app.services.effect.repetition import dashscope_clients

    cases = json.loads(args.cases.read_text())
    ids = {c["id"] for c in cases}
    rows = [r for arm in args.arms.split(",") for r in samples(args.runs, arm, ids)]
    embedder, judge, close = dashscope_clients()
    try:
        results = await score(cases, rows, args.output.with_suffix(".judge-cache.json"), embedder, judge)
    finally:
        await close()
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    args.output.chmod(0o600)
    return {"scored": len(results), "unjudged": sum(r["repeated"] is None for r in results)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--arms", required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(asyncio.run(main(parser.parse_args())), ensure_ascii=False))
