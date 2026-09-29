"""主动交流·自然度评测 runner.

跑真实主动消息生成路径 (build_proactive_context → trending_pick.pick_trending →
_generate_message, 线上模型), 但**mock 掉 tavily/brave 搜索和用户爱好记忆**: 每个
case 用它自己声明的 trending_candidates, 爱好取自 case.user_portrait。生产上热榜抓
什么无法控制, mock 让 eval 可复现, 也能干净地测 "假设内容源改好了, LLM 表现会不会好"。

用法 (容器内 /app, redis 起来):
    python -m evals.proactive_naturalness.run_eval
    python -m evals.proactive_naturalness.run_eval --samples 3 --judge dashscope:qwen-plus
    python -m evals.proactive_naturalness.run_eval --source-kind socially_hot

生成路径 = 《主动交流提示词（新增）》4-1 随机热点 / 4-2 用户爱好匹配 (筛选 + 100 字
摘要) → 4-3 热点闲聊生成; 卡片就是筛中的那条。历史 V0 / V3 基线见 standard.py。
注意: 新路径没有「AI 人设匹配」档, ai_persona_match 类 case 走的是随机热点。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from app.db import db
from app.services.runtime_config import ensure_loaded

from evals.proactive_naturalness import judge as J
from evals.proactive_naturalness.cases import CASES, SOURCE_KINDS, ProactiveCase


async def _generate_for_case(case: ProactiveCase) -> tuple[str, dict | None]:
    """Return (message, card_dict_or_None)."""
    # Late import: 模块顶导入会 pull 一堆运行时依赖, eval smoke test 想 offline 跑
    from unittest.mock import AsyncMock, patch

    from app.services.proactive import trending_pick
    from app.services.proactive.context import build_proactive_context
    from app.services.proactive.sender import _generate_message

    # 找一个活的 agent + workspace 作为运行时载体 (agent name/persona 由 case
    # 声明, 但生产的 sender._generate_message 需要真实 agent 才能读 mbti 生成
    # personality_brief). 复用最近的 agent 即可.
    agent = (await db.aiagent.find_many(take=1, order={"createdAt": "desc"}))[0]
    wss = await db.chatworkspace.find_many(
        where={"agentId": agent.id, "status": "active"}, take=1,
    )
    if not wss:
        raise RuntimeError("no active workspace for eval; need one agent+ws in DB")
    ws = wss[0]

    ctx = await build_proactive_context(
        workspace_id=ws.id, user_id=ws.userId, agent_id=agent.id,
        trigger_type=case.trigger_type, stage=case.stage,
        source="greeting", topic_theme="分享有趣见闻",
        conversation_id=None,
    )
    ctx["source"] = "greeting"
    ctx["is_decay_final"] = False
    # 覆盖 user_portrait, 让 judge 输入跟 case 声明的一致 (agent 是共享的, portrait
    # 来自 DB 里那个 workspace 的 user, 可能跟 case 声称的兴趣不匹配 → 让 eval
    # 结果不受 DB 状态影响)
    ctx["user_portrait"] = case.user_portrait

    pick = None
    if case.trending_candidates:
        hobbies = [case.user_portrait] if "喜欢" in case.user_portrait or "爱好" in case.user_portrait else []
        with patch.object(trending_pick, "_load_user_hobbies", AsyncMock(return_value=hobbies)):
            pick = await trending_pick.pick_trending(
                [asdict(c) for c in case.trending_candidates],
                user_id=ws.userId, workspace_id=ws.id,
            )
    ctx["trending_pick"] = pick

    msg = await _generate_message(ctx)
    if not msg:
        return "", None
    # 卡片就是筛中的那条 → 天然跟消息的话题源对齐
    card = None
    if pick and case.should_emit_card:
        card = {"title": pick.item.get("title", ""),
                "platform": pick.item.get("platform", ""),
                "url": pick.item.get("url", "")}
    return msg, card


async def _judge_one(judge_model, case, msg, card, sem):
    prompt = J.build_prompt(
        agent_persona=case.agent_persona_brief,
        user_portrait=case.user_portrait,
        trigger_type=case.trigger_type,
        source_kind=case.source_kind,
        trending_candidates=[asdict(c) for c in case.trending_candidates],
        message=msg,
        card=card,
    )
    async with sem:
        for _ in range(2):
            try:
                r = await asyncio.wait_for(judge_model.ainvoke(prompt), timeout=90)
            except Exception:
                continue
            v = J.parse_verdict(getattr(r, "content", "") or str(r))
            if v:
                return v
    return None


def _build_judge(spec):
    if not spec:
        return None
    from langchain_openai import ChatOpenAI

    from app.config import settings
    provider, _, model = spec.partition(":")
    creds = {"deepseek": (settings.deepseek_api_key, settings.deepseek_base_url),
             "ark": (settings.ark_api_key, settings.ark_base_url),
             "dashscope": (settings.dashscope_api_key,
                           "https://dashscope.aliyuncs.com/compatible-mode/v1")}
    if provider not in creds:
        raise SystemExit(f"unknown provider {provider!r}")
    key, base = creds[provider]
    extra = {"enable_thinking": False} if provider == "dashscope" else (
        {"thinking": {"type": "disabled"}} if provider == "ark" else {})
    return ChatOpenAI(model=model, api_key=key, base_url=base, timeout=90,
                      temperature=0, extra_body=extra or None)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-kind", choices=SOURCE_KINDS,
                    help="只跑一档 (调试用)")
    ap.add_argument("--samples", type=int, default=1,
                    help="每 case 生成多少次 (≥3 才有统计意义, samples=1 是噪音)")
    ap.add_argument("--judge", help="provider:model 评审模型 (建议非 ark, e.g. dashscope:qwen-plus)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--json", help="结果写这个路径")
    args = ap.parse_args()

    await db.connect()
    try:
        try:
            await ensure_loaded()
        except Exception as e:
            print(f"[warn] runtime config 未装载: {e}")

        from app.services.llm.models import get_utility_model
        judge_model = _build_judge(args.judge) or get_utility_model()

        cases = tuple(c for c in CASES
                      if not args.source_kind or c.source_kind == args.source_kind)
        sem = asyncio.Semaphore(args.concurrency)
        print(f"跑 {len(cases)} case × {args.samples} 样本 "
              f"(judge={args.judge or 'prod-small'})\n")

        results: list[dict] = []
        for case in cases:
            for i in range(args.samples):
                try:
                    msg, card = await _generate_for_case(case)
                except Exception as e:
                    print(f"  ✗ [{case.id}] 生成失败: {type(e).__name__} {str(e)[:80]}")
                    continue
                if not msg:
                    print(f"  ✗ [{case.id}] 空消息")
                    continue
                v = await _judge_one(judge_model, case, msg, card, sem)
                rec = {"case": case.id, "sample": i, "source_kind": case.source_kind,
                       "message": msg, "card": card, "verdict": v}
                results.append(rec)
                # 输出行
                if v:
                    bad_axes = [k for k in ("naturalness", "source_fit", "persona_match")
                                if not v.get(k)]
                    bad_neg = [k for k in ("advertorial_feel", "used_junk_content")
                               if v.get(k)]
                    flag = "🔴" if bad_neg else ("⚠️" if bad_axes else "✓")
                    print(f"  {flag} [{case.source_kind}/{case.id}]"
                          f" nat={v['naturalness']} fit={v['source_fit']}"
                          f" per={v['persona_match']} card={v['mentions_card']}"
                          f" ad={v['advertorial_feel']} junk={v['used_junk_content']}")
                    print(f"       「{msg[:70]}」" + (f"  [+card: {card['title'][:35]}]" if card else ""))
                    if v['reason']:
                        print(f"       judge: {v['reason']}")
                else:
                    print(f"  ? [{case.id}] judge 无 verdict")

        # ── 汇总 ──
        graded = [r for r in results if r["verdict"]]
        n = len(results); ng = len(graded)
        if not ng:
            print("\n[warn] 无成功 verdict, 无法汇总")
            return

        def _rate(key, want_true=True):
            return sum(1 for r in graded if r["verdict"][key] == want_true) / ng

        print("\n" + "=" * 60)
        print(f"样本 {n} (评审成功 {ng})")
        print(f"  naturalness       ✓ {_rate('naturalness')*100:.0f}%")
        print(f"  source_fit        ✓ {_rate('source_fit')*100:.0f}%")
        print(f"  persona_match     ✓ {_rate('persona_match')*100:.0f}%")
        print(f"  mentions_card     ✓ {_rate('mentions_card')*100:.0f}%   (有卡场景)")
        print(f"  advertorial_feel  ✗ {_rate('advertorial_feel')*100:.0f}%   (want 0%)")
        print(f"  used_junk_content ✗ {_rate('used_junk_content')*100:.0f}%   (want 0%)")

        # 分档
        print("\n  分档 (naturalness × source_fit both ✓ 的比例):")
        by_src = defaultdict(lambda: [0, 0])
        for r in graded:
            by_src[r["source_kind"]][0] += 1
            if r["verdict"]["naturalness"] and r["verdict"]["source_fit"]:
                by_src[r["source_kind"]][1] += 1
        for k in SOURCE_KINDS:
            t, ok = by_src[k]
            if t:
                print(f"     {k:<22} {ok}/{t} = {ok/t*100:.0f}%")

        if args.json:
            Path(args.json).write_text(json.dumps({
                "n": n, "graded": ng,
                "results": results,
            }, ensure_ascii=False, indent=2))
            print(f"\n写入 {args.json}")

    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
