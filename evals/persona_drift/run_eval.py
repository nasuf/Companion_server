"""人设一致性(persona drift)评测 runner。

在**真实累积的长对话**里, 在递增深度处独立问身份/风格探针, 看长上下文是否稀释
掉 system prompt 硬锚的人设。跟 reply_register/temporal_awareness 一样是 server
模式 (连生产 prompt store + 线上模型), 但这里是多轮累积 (逐轮真回复喂回历史)。

流程:
  1. 用 filler 中性话题跟 agent 真实来回 N 轮, 攒出一条长对话历史 (只这一段耗
     N 次生成)。
  2. 在深度 0 / 中 / 深 三个点, 各取历史前缀, 把每个探针当作下一轮独立问一次
     (探针之间不累积, 免得"刚问过名字所以记得"污染后面的探针)。
  3. 判定: 身份探针答案含锚值 (从 agent 字段动态取) + 任何深度不得人设泄漏;
     风格探针深层 vs 开场交给 LLM judge 看语气漂移。

用法 (容器内 /app):
    python -m evals.persona_drift.run_eval
    python -m evals.persona_drift.run_eval --judge dashscope:qwen-plus --turns 24
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.db import db
from app.services.chat.prompt_builder import build_chat_messages, build_system_prompt
from app.services.chat.reply_formatting import split_and_validate_replies
from app.services.llm.models import convert_messages, get_chat_model, get_utility_model, invoke_text
from app.services.runtime_config import ensure_loaded

from evals.persona_drift import judge as J
from evals.persona_drift import standard as S
from evals.persona_drift.cases import FILLER_USER_MSGS, PERSONA_LEAK_TERMS, PROBES, Probe

_UTC = timezone.utc


def _anchor_expected(agent, field: str) -> list[str]:
    """从 agent 字段推出探针答案该出现的锚 token(可接受的几种写法)。"""
    if field == "name":
        return [str(agent.name or "").strip()]
    if field == "age":
        return [str(agent.age)] if agent.age else []
    if field == "occupation":
        occ = str(agent.occupation or "").strip()
        # 职业域名词: 前 2 字通常是领域 ("皮具制作师"→"皮具"), 全称也接受
        return [x for x in {occ, occ[:2]} if len(x) >= 2]
    if field == "city":
        # "江苏省苏州市姑苏区…" → 取"市"前的市名 "苏州"; 全串也接受。
        # 先去省前缀, 否则贪婪 {2,3} 会把"苏省苏州"这种跨省市名连起来。
        raw = str(agent.city or "").strip()
        tail = raw.split("省", 1)[-1]
        m = re.search(r"([一-鿿]{2,3})市", tail)
        out = [raw] if raw else []
        if m:
            out.append(m.group(1))
        return [x for x in out if x]
    return []


def _history_rows(pairs: list[tuple[str, str]], now: datetime) -> list[dict]:
    """(role, content) 列表 → 带 createdAt 的行 (每轮间隔 2 分钟, 结束于 now-2min)."""
    n = len(pairs)
    rows = []
    for i, (role, content) in enumerate(pairs):
        rows.append({"role": role, "content": content,
                     "createdAt": now - timedelta(minutes=2 * (n - i))})
    return rows


async def _reply(agent, history_pairs: list[tuple[str, str]], user_msg: str, chat_model) -> str:
    now = datetime.now(_UTC)
    rows = _history_rows(history_pairs, now)
    rows.append({"role": "user", "content": user_msg, "createdAt": now})
    sp = await build_system_prompt(agent)
    msgs = build_chat_messages(sp, rows)
    raw = await invoke_text(chat_model, convert_messages(msgs))
    return "".join(split_and_validate_replies(raw))


def _leak(reply: str) -> list[str]:
    return [t for t in PERSONA_LEAK_TERMS if t in reply]


def _anchored(reply: str, expected: list[str]) -> bool:
    return any(e and e in reply for e in expected)


async def _judge_voice(judge_model, persona, probe: Probe, early, late, sem):
    prompt = J.build_prompt(persona=persona, question=probe.question, early=early, late=late)
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
        raise SystemExit(f"unknown judge provider {provider!r} (选 deepseek/ark/dashscope)")
    key, base = creds[provider]
    extra = {"enable_thinking": False} if provider == "dashscope" else (
        {"thinking": {"type": "disabled"}} if provider == "ark" else {})
    return ChatOpenAI(model=model, api_key=key, base_url=base, timeout=90,
                      temperature=0, extra_body=extra or None)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent-id")
    ap.add_argument("--judge", help="provider:model 评审模型 (建议非 ark)")
    ap.add_argument("--turns", type=int, default=24, help="攒多长的 filler 对话")
    ap.add_argument("--voice-samples", type=int, default=3,
                    help="深层风格探针独立重采样次数 (voice 一致率, n=1 是噪声)")
    ap.add_argument("--json")
    args = ap.parse_args()

    await db.connect()
    try:
        try:
            await ensure_loaded()
        except Exception as e:
            print(f"[warn] runtime config 未装载 ({e})")
        agent = (await db.aiagent.find_many(take=1, order={"createdAt": "desc"}))[0] \
            if not args.agent_id else await db.aiagent.find_unique(where={"id": args.agent_id})
        mbti = agent.mbti or {}
        persona = (mbti.get("summary") if isinstance(mbti, dict) else "") or f"{agent.name}, {agent.occupation}"
        chat_model = get_chat_model()
        judge_model = _build_judge(args.judge) or get_utility_model()
        sem = asyncio.Semaphore(3)
        print(f"agent: {agent.name} | 职业={agent.occupation} 城市={agent.city} 年龄={agent.age}")
        print(f"filler {args.turns} 轮 → 深度 0/中/深 各探 {len(PROBES)} 针\n")

        # ── 1. 攒真实长对话 ──
        n = min(args.turns, len(FILLER_USER_MSGS))
        history: list[tuple[str, str]] = []
        for i in range(n):
            um = FILLER_USER_MSGS[i]
            reply = await _reply(agent, history, um, chat_model)
            history.append(("user", um))
            history.append(("assistant", reply))
        print(f"已攒 {len(history)} 条历史\n")

        # 三个深度: 空 / 半 / 满 (小 turns 时可能重合, 去重防标签互相覆盖)
        depths = sorted({0, n // 2, n})
        depth_labels = {0: "深度0(开场)", n // 2: f"深度{n//2}(中)", n: f"深度{n}(深)"}

        # ── 2. 各深度独立探针 ──
        results = []
        early_style = {}  # 深度0 的风格探针回复, 供 judge 对比
        for d in depths:
            hist = history[: 2 * d]
            for p in PROBES:
                reply = await _reply(agent, hist, p.question, chat_model)
                leak = _leak(reply)
                rec = {"depth": d, "probe": p.id, "anchor": p.anchor,
                       "reply": reply, "leak": leak}
                if p.anchor:
                    exp = _anchor_expected(agent, p.anchor)
                    rec["expected"] = exp
                    rec["anchored"] = _anchored(reply, exp)
                if p.id == "style" and d == 0:
                    early_style[p.id] = reply
                results.append(rec)
                flag = "🔴" if leak else ("✗" if p.anchor and not rec["anchored"] else "✓")
                print(f"  {flag} [{depth_labels[d]}][{p.id}] 「{reply[:52]}」"
                      + (f"  锚={rec.get('anchored')}" if p.anchor else "")
                      + (f"  泄漏={leak}" if leak else ""))

        # ── 3. 语气漂移 judge: 深层 vs 开场, **matched-pair 重采样** ──
        # 关键: 早/晚各采 voice_samples 个, 逐对判 (early[i] vs late[i])。只采 1 个
        # 早基准会让"恰好很鲜活的开场"把正常的回复间波动误记成漂移 —— 对称采样才让
        # "voice 一致率"是可信的量, 而非跟单个幸运样本比。
        style_probe = next(p for p in PROBES if p.id == "style")
        deep_hist = history[: 2 * depths[-1]]
        k = max(1, args.voice_samples)
        # depth-0 与 depth-24 的首个 style 回复已在 §2 生成, 复用后各补 k-1 个
        early_replies = [early_style.get("style", "")]
        deep_replies = [next(r["reply"] for r in results
                             if r["probe"] == "style" and r["depth"] == depths[-1])]
        for _ in range(k - 1):
            early_replies.append(await _reply(agent, [], style_probe.question, chat_model))
            deep_replies.append(await _reply(agent, deep_hist, style_probe.question, chat_model))
        voice_verdicts = await asyncio.gather(*[
            _judge_voice(judge_model, persona, style_probe, er, dr, sem)
            for er, dr in zip(early_replies, deep_replies)
        ])
        voice = [{"early": er, "reply": dr, "verdict": v}
                 for er, dr, v in zip(early_replies, deep_replies, voice_verdicts)]
        graded_v = [x for x in voice if x["verdict"]]
        n_consistent = sum(1 for x in graded_v if x["verdict"]["voice_consistent"])
        n_persona = sum(1 for x in graded_v if x["verdict"]["persona_match"])
        voice_consistent_rate = (n_consistent / len(graded_v)) if graded_v else None
        persona_match_rate = (n_persona / len(graded_v)) if graded_v else None
        print(f"\n  语气漂移 judge (深度{depths[-1]} vs 开场, {len(graded_v)} 对):")
        for x in voice:
            v = x["verdict"]
            print(f"    开场「{x['early'][:40]}」")
            tag = "?" if not v else ("✓" if v["voice_consistent"] else "漂移")
            print(f"    [{tag}] 「{x['reply'][:56]}」"
                  + (f"  {v['reason'][:50]}" if v else ""))
        if voice_consistent_rate is not None:
            print(f"    → voice 一致率 {n_consistent}/{len(graded_v)}={voice_consistent_rate:.0%}"
                  f"   人设贴合率 {n_persona}/{len(graded_v)}={persona_match_rate:.0%}")

        # ── 汇总 ──
        print("\n" + "=" * 60)
        for d in depths:
            ids = [r for r in results if r["depth"] == d and r["anchor"]]
            anc = sum(1 for r in ids if r.get("anchored"))
            leaks = sum(1 for r in results if r["depth"] == d and r["leak"])
            print(f"  {depth_labels[d]:<14} 身份锚 {anc}/{len(ids)}"
                  f"   人设泄漏 {leaks}/{len([r for r in results if r['depth']==d])}")
        total_leak = sum(1 for r in results if r["leak"])
        total_id = [r for r in results if r["anchor"]]
        id_rate = sum(1 for r in total_id if r.get("anchored")) / (len(total_id) or 1)
        leak_rate = total_leak / (len(results) or 1)
        print(f"\n  合计: 身份锚保持 {sum(1 for r in total_id if r.get('anchored'))}/{len(total_id)}"
              f"={id_rate:.0%}   人设泄漏 {total_leak}/{len(results)}={leak_rate:.0%}"
              + (f"   voice 一致 {voice_consistent_rate:.0%}" if voice_consistent_rate is not None else ""))
        print(f"  判定线: 身份锚≥{S.MIN_IDENTITY_ANCHORED:.0%} 泄漏≤{S.MAX_PERSONA_LEAK:.0%} "
              f"voice≥{S.MIN_VOICE_CONSISTENT:.0%}")

        if args.json:
            Path(args.json).write_text(json.dumps({
                "agent": agent.name,
                "identity_anchored_rate": round(id_rate, 4),
                "persona_leak_rate": round(leak_rate, 4),
                "voice_consistent_rate": voice_consistent_rate,
                "persona_match_rate": persona_match_rate,
                "results": results, "voice": voice,
            }, ensure_ascii=False, indent=2))
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
