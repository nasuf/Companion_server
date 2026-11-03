"""CHAT01.10.2 把生产轮次交给真实编排器重新回一遍 (每例多次采样)。

一组 (arm) = 一个源码目录。两组只差被评的改动: 例如 "现状" 是 main, "去掉快修" 是 main 上
回退 10.0 的三个提交。每组在自己的源码目录里运行本模块 (源码目录排在 sys.path 最前),
用同一份用例、同一份生产提示词快照、同样的模型与采样数:

    cd <arm 源码目录> && ARK_API_KEY=... DASHSCOPE_API_KEY=... \\
    python -m evals.reply_repetition.run_replay --arm current --cases <cases.json> \\
        --prompt-snapshot <prompts.json> --output-dir <dir> --shard 0/8 --live

主回复路径强制走主 prompt (生产里这些轮次都是 main_llm); 意图、检索、记忆用合成桩, 历史、
时间与本轮消息 ID 用生产原样 (见 cases.py)。每例一个证据文件, 已完成的跳过, 不覆盖。
打分在 score.py 里做, 两组共用。
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

MODEL = "doubao-seed-character-260628"  # 生产主回复模型 (get_chat_model)
JUDGE_MODEL = "qwen3.5-flash"
# Ark 价格 (元 / 百万 token): 输入 / 命中缓存的输入 / 输出, 与 context_embedding 一致
ARK_PRICE = (0.8, 0.16, 2.0)
CRITERION = ("AI 这一轮的回复（可能多条气泡）是在回应用户最后发来的这句话（message），不是答非所问。"
             "只要接上了这句就通过；长短、措辞、语气都不影响；不要评价它和之前说过的话是否重复。")
JUDGING_INSTRUCTION = ("审核重放的生产对话，不续写。只评判case.criterion里的明确条件，不得添加额外条件。"
                       "输出JSON {\"pass\":true或false,\"reason\":\"简短理由\"}。\n")


def incident(case: dict) -> dict:
    """用例 → 事故回放器 (run_incident_eval) 的输入。"""
    return {
        "id": case["id"], "now": case["now"], "gap": 0, "history": [],
        "history_rows": case["history_rows"], "message": case["message"],
        "criterion": CRITERION, "memory": False, "route": "ordinary",
        "user_message_id": case["current_ids"][-1],
        "reply_context": {"received_at": case["received_at"], "turn_message_ids": case["current_ids"]},
    }


def call_cost(usage: dict | None) -> float | None:
    """一次主模型调用的费用; 供应商没给用量时返回 None (不当作 0)。"""
    if not usage:
        return None
    prompt = int(usage.get("prompt_tokens") or 0)
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    output = int(usage.get("completion_tokens") or 0)
    return ((prompt - cached) * ARK_PRICE[0] + cached * ARK_PRICE[1] + output * ARK_PRICE[2]) / 1_000_000


def detail_cost(detail: dict) -> float:
    return sum(call_cost(c.get("usage")) or 0 for r in detail.get("runs", []) for c in r.get("calls", [])
               if "messages" in c)


def arm_spent(out: Path) -> float:
    """同组已花的主回复费用。别的分片可能正在写或改名某个文件: 读不到、读到半截的跳过,
    所以上限可能超出约每个分片一例的费用 (一例 5 次采样约 ¥0.01)。"""
    total = 0.0
    for path in out.glob("rr-*.json*"):  # 含中断后改名留下的半截文件: 那些调用也花了钱
        try:
            total += detail_cost(json.loads(path.read_text()))
        except (FileNotFoundError, ValueError):
            continue
    return total


def shard(cases: list[dict], spec: str) -> list[dict]:
    index, total = (int(x) for x in spec.split("/"))
    if not 0 <= index < total:
        raise ValueError("Shard must be i/n with 0 <= i < n")
    return [c for i, c in enumerate(cases) if i % total == index]


def force_main(snapshot: dict, original_context):
    """与 context_embedding 相同: 按快照还原提示词启用状态, 用真实的意图上下文, 主回复强制走主 prompt。"""
    def hook(patches, chat_module, io, current_case):
        from app.services.prompting import store
        patches.setattr(store.db.prompttemplate, "find_unique", AsyncMock(
            side_effect=lambda **kw: SimpleNamespace(isEnabled=snapshot[kw["where"]["key"]]["is_enabled"])
            if kw["where"]["key"] in snapshot else None))
        patches.setattr(chat_module, "_fetch_intent_context", original_context)
        generate = chat_module._generate_reply

        async def force_route(**kwargs):
            kwargs["force_main_prompt"] = True
            return await generate(**kwargs)

        patches.setattr(chat_module, "_generate_reply", force_route)
    return hook


def completed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return "total" in json.loads(path.read_text())
    except ValueError:
        return False


async def run(args) -> dict:
    from app.services.chat import orchestrator as chat
    from evals.temporal.awareness import run_incident_eval

    # 两组必须用同一种回放方式: 老版本回放器不带本轮消息 ID、不留气泡边界, 混用会让两组不可比
    if getattr(run_incident_eval, "REPLAY_PROTOCOL", 0) < 2:
        raise RuntimeError("This checkout's replay harness predates production-delivery replay")
    run_incident = run_incident_eval.run

    original_context = chat._fetch_intent_context  # 取在回放器打桩之前

    source_root = Path(__file__).resolve().parents[2]
    if (source_root / ".env").exists():
        raise ValueError("Run each arm from an isolated checkout without .env")
    cases = shard(json.loads(args.cases.read_text()), args.shard)
    snapshot = {p["key"]: p for p in json.loads(args.prompt_snapshot.read_text())}
    out = args.output_dir / args.arm
    out.mkdir(parents=True, exist_ok=True)
    summary = {"arm": args.arm, "source_root": str(source_root), "shard": args.shard, "done": 0, "failed": []}
    for case in cases:
        detail = out / f"{case['id']}.json"
        if completed(detail):
            summary["done"] += 1
            continue
        if detail.exists():  # 上次中断留下的半截: 留作证据, 换个名字, 重跑这一例
            n = len(list(out.glob(f"{case['id']}.json.partial*")))
            detail.rename(out / f"{case['id']}.json.partial{n}")
        spent = arm_spent(out)
        if spent >= args.max_cost_cny:
            summary["stopped"] = f"cost cap {args.max_cost_cny} reached ({spent:.2f})"
            break
        runner = SimpleNamespace(
            source_root=source_root, prompt_snapshot=args.prompt_snapshot, model=MODEL,
            judge_model=JUDGE_MODEL, judging_instruction=JUDGING_INSTRUCTION, guard_model="qwen3.5-flash",
            samples=args.samples, output=detail, fixture_hook=force_main(snapshot, original_context),
        )
        try:
            await run_incident(runner, [incident(case)])
            summary["done"] += 1
        except Exception as exc:  # 一例失败不拖累其余; 失败原因留在证据文件和汇总里
            summary["failed"].append({"case": case["id"], "error": type(exc).__name__})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--prompt-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5, choices=range(1, 11))
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--max-cost-cny", type=float, default=5.0,
                        help="本组 (arm) 主回复费用上限: 同组各分片共用一个输出目录, 按目录里已花的合计判断")
    parser.add_argument("--live", action="store_true", required=True)
    print(json.dumps(asyncio.run(run(parser.parse_args())), ensure_ascii=False))
