#!/usr/bin/env python
"""跑边界回复评测.

    PYTHONPATH=. python evals/boundary_reply/run_eval.py

走生产的 prompt (boundary.* registry key) 和生产的模型, 只是把多轮按脚本推进而不
依赖真实 Redis 耐心状态 —— 我们要测的是"给定这个状态该怎么回", 不是状态机本身
(那部分有 tests/test_boundary.py 覆盖)。

四条判据全部确定性, 依据见 judge.py。真正的效果指标 (再犯率) 需要真实用户, 离线
测不了。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.boundary_reply.cases import CASES, BoundaryCase, Turn  # noqa: E402
from evals.boundary_reply.judge import judge_turn  # noqa: E402

# 每一轮该用哪个 prompt。normal 档的攻击按 K3 走 (生产实录里"煞笔"判的就是 K3)。
_PROMPT_BY_ZONE = {
    "normal": "boundary.severe_attack_reply",
    "medium": "boundary.medium_patience_reply",
    "low": "boundary.low_patience_reply",
}


async def _gen(turn: Turn, history: list[str], agent_desc: str) -> str:
    from app.services.llm.models import get_chat_model
    from app.services.prompting.store import get_prompt_text

    key = _PROMPT_BY_ZONE[turn.zone]
    tpl = await get_prompt_text(key)
    ctx = "\n".join(history[-6:]) or "(无)"
    prompt = tpl.format(
        message=turn.user,
        context=ctx,
        personality_brief=agent_desc,
        user_portrait="(无)",
    )
    from langchain_core.messages import HumanMessage

    resp = await get_chat_model().ainvoke([HumanMessage(content=prompt)])
    return (getattr(resp, "content", "") or "").strip()


async def run_case(case: BoundaryCase, agent_desc: str) -> list[tuple[Turn, str, object]]:
    history: list[str] = []
    prev_replies: list[str] = []
    out = []
    for turn in case.turns:
        reply = await _gen(turn, history, agent_desc)
        verdict = judge_turn(reply, prev_replies)
        out.append((turn, reply, verdict))
        history.append(f"用户: {turn.user}")
        history.append(f"我: {reply}")
        prev_replies.append(reply)
    return out


def _score(results: list[tuple[Turn, str, object]]) -> dict[str, int]:
    """把每条判据折成"违规轮数", 越小越好."""
    bad = {"无引导": 0, "索要解释/道歉": 0, "自贬": 0, "道歉后仍抱怨": 0, "跨轮重复": 0}
    for turn, _reply, v in results:
        # 引导只对**冲突当轮**要求 —— 余波轮 (低耐心) 的产品设定是话少即情绪,
        # 对着"嗯"要求带话题引导是自相矛盾。见 cases.Turn.expects_redirect。
        if turn.expects_redirect and not v.has_redirect:
            bad["无引导"] += 1
        if v.demands_explanation:
            bad["索要解释/道歉"] += 1
        if v.self_deprecates:
            bad["自贬"] += 1
        if turn.expects_closure and v.restates_grievance:
            bad["道歉后仍抱怨"] += 1
        if v.repeats_previous:
            bad["跨轮重复"] += 1
    return bad


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", help="只跑某一个用例 id")
    ap.add_argument("--agent-desc", default="性格温和但有自己的边界, 说话口语化")
    args = ap.parse_args()

    cases = [c for c in CASES if not args.case or c.id == args.case]
    total = {"无引导": 0, "索要解释/道歉": 0, "自贬": 0, "道歉后仍抱怨": 0, "跨轮重复": 0}
    turns = 0

    for case in cases:
        print(f"\n{'=' * 72}\n{case.id}  —— {case.note.splitlines()[0]}")
        results = await run_case(case, args.agent_desc)
        for turn, reply, v in results:
            flags = []
            if turn.expects_redirect and not v.has_redirect:
                flags.append("无引导")
            if v.demands_explanation:
                flags.append("索要解释")
            if v.self_deprecates:
                flags.append("自贬")
            if turn.expects_closure and v.restates_grievance:
                flags.append("仍抱怨")
            if v.repeats_previous:
                flags.append("重复")
            tag = ("  ⚠ " + " ".join(flags)) if flags else "  ✓"
            print(f"  [{turn.zone}] 用户: {turn.user}")
            print(f"        回复: {reply}{tag}")
        s = _score(results)
        for k in total:
            total[k] += s[k]
        turns += len(results)

    print(f"\n{'=' * 72}\n合计 {turns} 轮:")
    for k, v in total.items():
        print(f"  {k:<14} {v} 轮 ({v / max(turns, 1) * 100:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
