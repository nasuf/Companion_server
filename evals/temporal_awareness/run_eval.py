"""时间感知回复评测 runner (P4)。

跑真实回复路径 —— build_system_prompt + build_chat_messages + 线上 chat 模型 ——
但**控制三个时间旋钮**(reply_register 刻意冻结的正是这些): 当前消息距上一条的
间隔 gap、当前本地时刻 now_hour、间隔前的历史。判定 = 确定性红线 + LLM 维度评审。

必须在能连生产 prompt store 的环境跑 (admin 在后台改过的 chat.* 提示词只存在于
DB/Redis, 评代码默认值 = 评一个线上不存在的提示词)。

用法 (容器内 /app):
    python -m evals.temporal_awareness.run_eval
    python -m evals.temporal_awareness.run_eval --group reunion --json out.json
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

from evals.temporal_awareness import judge as J
from evals.temporal_awareness.cases import CASES, GROUPS, TemporalCase
from evals.temporal_awareness import standard as S

_TZ8 = timezone(timedelta(hours=8))
_WEEKDAY_CN = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
# 固定参照日 (可复现): 2026-09-09 是星期三。星期几本身不是被测维度, 固定即可。
_REF_DATE = (2026, 9, 9)


def _now_local(hour: int) -> datetime:
    return datetime(*_REF_DATE, hour, 0, tzinfo=_TZ8)


def _time_context(now_local: datetime) -> str:
    # 复刻 time_service.build_time_context 的格式 (小时精度 + 星期), 但用受控时刻。
    return (f"当前时间：{now_local.strftime('%Y年%m月%d日')} "
            f"{now_local.hour}时 {_WEEKDAY_CN[now_local.weekday()]}")


def _history_rows(case: TemporalCase, now_local: datetime) -> list[dict]:
    """历史消息按 gap 布局: 间隔前的对话结束于 now-gap, 当前消息在 now。"""
    now_utc = now_local.astimezone(timezone.utc)
    last_hist = now_utc - timedelta(seconds=case.gap_seconds)
    rows: list[dict] = []
    n = len(case.history)
    for i, (role, content) in enumerate(case.history):
        # 历史内部间隔 3 分钟, 最后一条落在 last_hist
        rows.append({
            "role": role, "content": content,
            "createdAt": last_hist - timedelta(minutes=3 * (n - 1 - i)),
        })
    rows.append({"role": "user", "content": case.message, "createdAt": now_utc})
    return rows


def _history_text(case: TemporalCase) -> str:
    spk = {"user": "用户", "assistant": "AI"}
    return "\n".join(f"  {spk[r]}: {c}" for r, c in case.history)


async def _generate(agent, case: TemporalCase, chat_model) -> str:
    now_local = _now_local(case.now_hour)
    ai_status = get_current_status(_BASE_SCHEDULE_TEMPLATE, now_local)
    system_prompt = await build_system_prompt(
        agent,
        time_context=_time_context(now_local),
        reengagement_gap_seconds=case.gap_seconds,
        ai_status=ai_status,
    )
    chat_messages = build_chat_messages(system_prompt, _history_rows(case, now_local))
    raw = await invoke_text(chat_model, convert_messages(chat_messages))
    return "".join(split_and_validate_replies(raw))  # 用户实际看到的


def _det_violations(case: TemporalCase, reply: str) -> list[str]:
    return [w for w in case.must_not_contain if w in reply]


async def _judge(judge_model, case: TemporalCase, reply: str, sem: asyncio.Semaphore) -> dict | None:
    prompt = J.build_prompt(
        gap_seconds=case.gap_seconds, hour=case.now_hour,
        history=_history_text(case), message=case.message, reply=reply,
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
    """provider:model → 独立评审模型 (走 .ainvoke, 不经 invoke_text); 缺省用生产小模型。"""
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
    ap.add_argument("--group", choices=GROUPS)
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

        # 生成用生产 get_chat_model(): invoke_text 的 resilience 层期望 app 工厂
        # 造的模型 (带 provider 元数据做 fallback), 裸 ChatOpenAI 传进去不工作。
        # 连生产 DB + ensure_loaded() 后, 它解析到线上真实配置 (2026-09: chat=
        # ark doubao-seed-character, small=dashscope qwen3.5-flash), 评的就是
        # 线上实际在用的模型。评审建议 --judge 换到非 ark 厂商避免自评偏好。
        chat_model = get_chat_model()
        judge_model = build_model(args.judge) or get_utility_model()
        cases = tuple(c for c in CASES if not args.group or c.group == args.group)
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
                results.append({"case": case.id, "group": case.group, "reply": reply,
                                "det_violations": det, "verdict": v})
                flag = "🔴" if det else ("⚠️" if v and (v["hallucination"] or not v["gap_ok"] or not v["tod_ok"] or v["stale_topic"]) else "✓")
                vs = "" if not v else f" gap_ok={v['gap_ok']} hall={v['hallucination']} tod={v['tod_ok']} stale={v['stale_topic']}"
                print(f"  {flag} [{case.id}]{vs}")
                if det: print(f"       红线命中: {det}")
                print(f"       「{reply[:70]}」")
                if v and v['reason']: print(f"       judge: {v['reason']}")

        # ── 汇总 ──
        graded = [r for r in results if r["verdict"]]
        n = len(results); ng = len(graded)
        det_viol = sum(1 for r in results if r["det_violations"])
        def rate(k, want_true=True):
            if not graded: return 0.0
            return sum(1 for r in graded if r["verdict"][k] == want_true) / ng
        gap_ok = rate("gap_ok"); hall = rate("hallucination"); tod = rate("tod_ok"); stale = rate("stale_topic")
        print("\n" + "=" * 56)
        print(f"样本 {n} (评审成功 {ng})")
        print(f"  确定性红线违反率 : {det_viol}/{n} = {det_viol/(n or 1)*100:.0f}%   (目标 {S.DETERMINISTIC_MAX_VIOLATION_RATE*100:.0f}%)")
        print(f"  间隔处理得体     : {gap_ok*100:.0f}%   (目标 ≥{S.MIN_GAP_OK*100:.0f}%)")
        print(f"  时间幻觉         : {hall*100:.0f}%   (目标 ≤{S.MAX_HALLUCINATION*100:.0f}%)")
        print(f"  时刻贴合         : {tod*100:.0f}%   (目标 ≥{S.MIN_TOD_OK*100:.0f}%)")
        print(f"  无端复活旧话题   : {stale*100:.0f}%   (目标 ≤{S.MAX_STALE_TOPIC*100:.0f}%)")
        print("  分组红线:")
        by_g = defaultdict(lambda: [0, 0])
        for r in results:
            by_g[r["group"]][0] += 1
            by_g[r["group"]][1] += int(bool(r["det_violations"]))
        for g in GROUPS:
            t, x = by_g[g]
            if t: print(f"     {g:<16} {x}/{t} 红线违反")

        if args.json:
            Path(args.json).write_text(json.dumps({
                "n": n, "graded": ng, "det_violation_rate": det_viol/(n or 1),
                "gap_ok": gap_ok, "hallucination": hall, "tod_ok": tod, "stale_topic": stale,
                "results": results,
            }, ensure_ascii=False, indent=2))
            print(f"\n写入 {args.json}")
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
