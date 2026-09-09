"""持续性状态(durative state)评测 runner (P2)。

跑真实回复路径 (build_system_prompt + build_chat_messages + 线上豆包), 控制时间旋钮:
把一句带时长的状态陈述放在历史里 stated_hours_ago 之前, 当前中性消息落在 now, 看
AI 能否从"说于 D 前 + 持续 N 天"自推出状态还在窗内 (进行中) 还是窗外 (已结束),
并据此正确处理。判定 = 确定性红线 + LLM judge (给定窗内外 ground truth)。

必须在能连生产 prompt store 的环境跑 (chat.* 提示词只存在 DB/Redis)。

用法 (容器内 /app):
    python -m evals.durative_state.run_eval
    python -m evals.durative_state.run_eval --kind expired --judge dashscope:qwen-plus
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.db import db
from app.services.chat.prompt_builder import build_chat_messages, build_system_prompt
from app.services.chat.reply_formatting import split_and_validate_replies
from app.services.llm.models import convert_messages, get_chat_model, get_utility_model, invoke_text
from app.services.runtime_config import ensure_loaded
from app.services.schedule_domain.schedule import _BASE_SCHEDULE_TEMPLATE, get_current_status

from evals.durative_state import judge as J
from evals.durative_state import standard as S
from evals.durative_state.cases import CASES, KINDS, DurativeCase

_TZ8 = timezone(timedelta(hours=8))
_WEEKDAY_CN = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
_REF_DATE = (2026, 9, 9)  # 固定参照日 (星期几非被测维度)


def _now_local(hour: int) -> datetime:
    return datetime(*_REF_DATE, hour, 0, tzinfo=_TZ8)


def _time_context(now_local: datetime) -> str:
    return (f"当前时间：{now_local.strftime('%Y年%m月%d日')} "
            f"{now_local.hour}时 {_WEEKDAY_CN[now_local.weekday()]}")


def _history_rows(case: DurativeCase, now_local: datetime) -> list[dict]:
    """状态陈述落在 now-stated_hours_ago, AI ack 3 分钟后, 当前消息在 now。"""
    now_utc = now_local.astimezone(timezone.utc)
    stated_at = now_utc - timedelta(hours=case.stated_hours_ago)
    return [
        {"role": "user", "content": case.state_line, "createdAt": stated_at},
        {"role": "assistant", "content": case.state_ack,
         "createdAt": stated_at + timedelta(minutes=3)},
        {"role": "user", "content": case.message, "createdAt": now_utc},
    ]


async def _generate(agent, case: DurativeCase, chat_model) -> str:
    now_local = _now_local(case.now_hour)
    ai_status = get_current_status(_BASE_SCHEDULE_TEMPLATE, now_local)
    # gap = 状态陈述距今 (通常大间隔 → 重逢段照常注入, 这是真实条件)
    gap_seconds = case.stated_hours_ago * 3600
    system_prompt = await build_system_prompt(
        agent,
        time_context=_time_context(now_local),
        reengagement_gap_seconds=gap_seconds,
        ai_status=ai_status,
    )
    chat_messages = build_chat_messages(system_prompt, _history_rows(case, now_local))
    raw = await invoke_text(chat_model, convert_messages(chat_messages))
    return "".join(split_and_validate_replies(raw))


def _det_violations(case: DurativeCase, reply: str) -> list[str]:
    return [w for w in case.must_not_contain if w in reply]


async def _judge(judge_model, case: DurativeCase, reply: str, sem: asyncio.Semaphore) -> dict | None:
    prompt = J.build_prompt(
        state_line=case.state_line, duration_hint=case.duration_hint,
        kind=case.kind, message=case.message, reply=reply,
    )
    async with sem:
        for _ in range(2):
            try:
                resp = await asyncio.wait_for(judge_model.ainvoke(prompt), timeout=90)
            except Exception:
                continue
            v = J.parse_verdict(getattr(resp, "content", "") or str(resp))
            if v:
                return v
    return None


def build_model(spec: str | None):
    if not spec:
        return None
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
        raise SystemExit(f"unknown provider {provider!r}")
    key, base = creds[provider]
    extra = {}
    if provider == "ark":
        extra["thinking"] = {"type": "disabled"}
    elif provider == "dashscope":
        extra["enable_thinking"] = False
    return ChatOpenAI(model=model, api_key=key, base_url=base, timeout=90,
                      temperature=0.7, extra_body=extra or None)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent-id")
    ap.add_argument("--kind", choices=KINDS)
    ap.add_argument("--judge", help="provider:model 覆盖评审模型 (默认生产小模型)")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--json")
    args = ap.parse_args()

    await db.connect()
    try:
        try:
            await ensure_loaded()
        except Exception as e:
            print(f"[warn] runtime config 未装载 ({e}); 用 env 默认")
        agent = (await db.aiagent.find_many(take=1, order={"createdAt": "desc"}))[0] \
            if not args.agent_id else await db.aiagent.find_unique(where={"id": args.agent_id})
        print(f"agent: {agent.name} ({agent.id[:8]})")

        chat_model = get_chat_model()
        judge_model = build_model(args.judge) or get_utility_model()
        cases = tuple(c for c in CASES if not args.kind or c.kind == args.kind)
        sem = asyncio.Semaphore(args.concurrency)
        print(f"跑 {len(cases)} 用例 × {args.samples} 样本 "
              f"(chat=prod get_chat_model, judge={args.judge or 'prod-small'})\n")

        results: list[dict] = []
        for case in cases:
            for _ in range(args.samples):
                try:
                    reply = await _generate(agent, case, chat_model)
                except Exception as e:
                    print(f"  ✗ [{case.id}] 生成失败: {type(e).__name__} {str(e)[:60]}")
                    continue
                det = _det_violations(case, reply)
                v = await _judge(judge_model, case, reply, sem)
                results.append({"case": case.id, "kind": case.kind, "reply": reply,
                                "det_violations": det, "verdict": v})
                bad = det or (v and not v["state_ok"])
                flag = "🔴" if det else ("⚠️" if bad else "✓")
                vs = "" if not v else f" state_ok={v['state_ok']} mentioned={v['mentioned']}"
                print(f"  {flag} [{case.kind}][{case.id}]{vs}")
                if det:
                    print(f"       红线命中: {det}")
                print(f"       「{reply[:70]}」")
                if v and v['reason']:
                    print(f"       judge: {v['reason']}")

        # ── 汇总 ──
        graded = [r for r in results if r["verdict"]]
        n = len(results); ng = len(graded)
        det_viol = sum(1 for r in results if r["det_violations"])
        state_ok = (sum(1 for r in graded if r["verdict"]["state_ok"]) / ng) if graded else 0.0
        # 分 kind 看: expired 通常更难 (要主动意识到"早结束了")
        by_kind = defaultdict(lambda: [0, 0])  # kind → [graded, ok]
        for r in graded:
            by_kind[r["kind"]][0] += 1
            by_kind[r["kind"]][1] += int(r["verdict"]["state_ok"])
        print("\n" + "=" * 56)
        print(f"样本 {n} (评审成功 {ng})")
        print(f"  确定性红线违反率 : {det_viol}/{n} = {det_viol/(n or 1)*100:.0f}%   (目标 {S.DETERMINISTIC_MAX_VIOLATION_RATE*100:.0f}%)")
        print(f"  状态窗处理正确   : {state_ok*100:.0f}%   (目标 ≥{S.MIN_STATE_OK*100:.0f}%)")
        for k in KINDS:
            g, ok = by_kind[k]
            if g:
                print(f"     {k:<9} {ok}/{g} = {ok/g*100:.0f}%")

        if args.json:
            Path(args.json).write_text(json.dumps({
                "n": n, "graded": ng, "det_violation_rate": det_viol / (n or 1),
                "state_ok": state_ok,
                "by_kind": {k: {"graded": g, "ok": ok} for k, (g, ok) in by_kind.items()},
                "results": results,
            }, ensure_ascii=False, indent=2))
            print(f"\n写入 {args.json}")
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
