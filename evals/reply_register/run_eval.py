"""Reply-register eval runner.

Runs the REAL reply path — `build_system_prompt` + `build_chat_messages` +
the configured chat model — over the case bank, N samples per case, then
grades with `judge.py` against the thresholds in `standard.py`.

Must run where the production prompt store lives: the admin has edited
`chat.response_instruction` on the web console, and that edit exists only in
the server's DB/Redis. Grading the local code default would score a prompt
that is not in production.

Usage (inside the server container, from /app):
    python -m evals.reply_register.run_eval
    python -m evals.reply_register.run_eval --group fact --samples 3
    python -m evals.reply_register.run_eval --json out.json --baseline old.json

What is deliberately held constant: no memories, no portrait, no emotion or
schedule context. Those are content, not register instructions, and letting
them vary per run would make results irreproducible. The eval therefore scores
the always-on prompt stack, which is exactly the layer under test.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.db import db
from app.services.chat.prompt_builder import build_chat_messages, build_system_prompt
from app.services.chat.reply_formatting import split_and_validate_replies
from app.services.llm.models import convert_messages, get_chat_model, get_utility_model, invoke_text
from app.services.runtime_config import ensure_loaded

from evals.reply_register import judge as J
from evals.reply_register.cases import ALL_CASES, GROUPS, RegisterCase
from evals.reply_register.standard import (
    CHITCHAT_MAX_OFF_TOPIC,
    CHITCHAT_MIN_LENGTH_OK,
    CHITCHAT_MIN_NATURAL,
    EMOTION_MAX_ADVICE_FIRST,
    EMOTION_MIN_ACKNOWLEDGE_FIRST,
    FACT_MAX_ENCYCLOPEDIC,
    FACT_MIN_COMPANION,
    FORMAT_PASS_RATE,
    HUMAN_IM_CHARS_PER_LINE,
    OUTOFWINDOW_MAX_BLAMES_USER,
    OUTOFWINDOW_MAX_FLAT_DENIAL,
    OUTOFWINDOW_MAX_PLAYS_ALONG,
    OUTOFWINDOW_MIN_HONEST,
    FALSEPREMISE_MAX_EVASIVE,
    FALSEPREMISE_MAX_PLAYS_ALONG,
    FALSEPREMISE_MIN_PUSHBACK,
    HUMAN_IM_LINES_PER_TURN,
    SAMPLES_PER_CASE,
    SHORT_INPUT_CHARS,
    SHORT_INPUT_MAX_REPLY_CHARS,
)


async def _pick_agent(agent_id: str | None):
    if agent_id:
        agent = await db.aiagent.find_unique(where={"id": agent_id})
        if agent is None:
            raise SystemExit(f"agent {agent_id} not found")
        return agent
    agents = await db.aiagent.find_many(take=1, order={"createdAt": "desc"})
    if not agents:
        raise SystemExit("no agent in database")
    return agents[0]


def _history_rows(case: RegisterCase) -> list[dict[str, Any]]:
    """Case history → DB-shaped rows so build_chat_messages behaves as in prod.

    Timestamps are spaced a few minutes apart ending just before now, which is
    what the `[MM-DD HH:MM]` prefix and the re-engagement gap logic expect.
    """
    now = datetime.now(timezone.utc)
    total = len(case.history) + 1
    rows = [
        {
            "role": role,
            "content": content,
            "createdAt": now - timedelta(minutes=3 * (total - i)),
        }
        for i, (role, content) in enumerate(case.history)
    ]
    rows.append({"role": "user", "content": case.message, "createdAt": now})
    return rows


def _history_text(case: RegisterCase) -> str:
    speaker = {"user": "用户", "assistant": "AI"}
    return "\n".join(f"{speaker[r]}: {c}" for r, c in case.history)


async def _generate(agent, case: RegisterCase) -> tuple[str, str]:
    """Return (raw model output, what the user would actually see).

    Grading the raw output would measure a string no user ever receives: the
    production path runs `split_and_validate_replies`, which merges overflow
    bubbles, strips system markers and repairs the timestamp imitation. The
    first baseline run graded raw text and its format numbers were partly an
    artifact of that gap.
    """
    system_prompt = await build_system_prompt(agent)
    chat_messages = build_chat_messages(system_prompt, _history_rows(case))
    raw = await invoke_text(get_chat_model(), convert_messages(chat_messages))
    return raw, "||".join(split_and_validate_replies(raw))


def build_judge_model(spec: str | None):
    """`provider:model` → 独立评审模型; 缺省用生产小模型.

    可指定的意义有两层: 生产小模型不可用时评测不至于停摆 (2026-07-25 就撞上
    deepseek 余额耗尽), 以及避免让主模型自己评自己 —— LLM-as-judge 的自偏好
    是已知问题, 评审器最好来自另一个厂商.
    """
    if not spec:
        return get_utility_model()
    from langchain_openai import ChatOpenAI

    from app.config import settings

    provider, _, model = spec.partition(":")
    creds = {
        "dashscope": (settings.dashscope_api_key,
                      "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "ark": (settings.ark_api_key, settings.ark_base_url),
        "deepseek": (settings.deepseek_api_key, settings.deepseek_base_url),
    }
    if provider not in creds:
        raise SystemExit(f"unknown judge provider {provider!r}; use {sorted(creds)}")
    api_key, base_url = creds[provider]
    if not api_key:
        raise SystemExit(f"no API key configured for {provider}")
    return ChatOpenAI(
        model=model, api_key=api_key, base_url=base_url, timeout=90,
        # qwen 的 thinking 模式对一个三选一的判定毫无帮助, 只会让每次评审多花
        # 十几秒 —— 关掉它是这里能跑通的关键 (非 qwen 后端会忽略该参数).
        extra_body={"enable_thinking": False},
    )


async def _ask_judge(judge_model, prompt: str) -> str:
    """直接调模型, 不走 invoke_text 的生产 resilience.

    那一层的 12s 硬超时是按聊天延迟体感调的, 而评审 prompt (rubric + 历史 +
    回复) 长得多, 并发下必然超时 —— 实测 12s 超时 + 熔断 + 一个不存在的 ollama
    兜底, 让整轮校准只过 3/9. 评测不是热路径, 慢一点无所谓, 假失败才要命.
    """
    response = await judge_model.ainvoke(prompt)
    return str(getattr(response, "content", "") or "")


async def _judge(judge_model, group: str, case: RegisterCase, reply: str) -> str | None:
    prompt = J.build_judge_prompt(group, _history_text(case), case.message, reply)
    raw = await _ask_judge(judge_model, prompt)
    return J.parse_verdict(group, raw)


async def run_calibration(judge_model, concurrency: int) -> bool:
    """Judge must separate the undebatable cases before its verdicts count."""
    sem = asyncio.Semaphore(concurrency)

    async def one(group: str, message: str, reply: str, expected: str):
        async with sem:
            prompt = J.build_judge_prompt(group, "", message, reply)
            try:
                raw = await _ask_judge(judge_model, prompt)
            except Exception as e:  # noqa: BLE001
                return group, message, expected, f"ERROR:{e}"
            return group, message, expected, J.parse_verdict(group, raw)

    results = await asyncio.gather(*(one(*c) for c in J.CALIBRATION))
    bad = 0
    print("── 评审器校准 ──")
    for group, message, expected, got in results:
        ok = got == expected
        bad += 0 if ok else 1
        print(f"  {'OK  ' if ok else 'MISS'} [{group}] want={expected} got={got}  «{message}»")
    print(f"  {len(results) - bad}/{len(results)} 通过\n")
    return bad == 0


async def run_cases(
    agent, judge_model, cases: tuple[RegisterCase, ...], samples: int,
    concurrency: int,
) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)
    rows: list[dict[str, Any]] = []

    async def one(case: RegisterCase, index: int):
        async with sem:
            try:
                raw, reply = await _generate(agent, case)
            except Exception as e:  # noqa: BLE001 — 单点失败不该终止整轮
                rows.append({"case": case.id, "group": case.group, "sample": index,
                             "error": f"generate: {e}"})
                return
            fmt = J.analyse_format(reply)
            try:
                verdict = await _judge(judge_model, case.group, case, reply)
            except Exception as e:  # noqa: BLE001
                verdict = None
                print(f"  [judge-error] {case.id}#{index}: {e}")
            rows.append({
                "case": case.id,
                "group": case.group,
                "sample": index,
                "message": case.message,
                "reply": reply,
                "raw": raw,
                "verdict": verdict,
                "bubbles": fmt.bubbles,
                "max_bubble_chars": fmt.max_bubble_chars,
                "total_chars": fmt.total_chars,
                "emoji_count": fmt.emoji_count,
                "format_ok": fmt.format_ok,
            })

    tasks = [one(c, i) for c in cases for i in range(samples)]
    done = 0
    for chunk_start in range(0, len(tasks), concurrency * 4):
        chunk = tasks[chunk_start: chunk_start + concurrency * 4]
        await asyncio.gather(*chunk)
        done += len(chunk)
        print(f"  … {done}/{len(tasks)}")
    return rows


def _rate(counter: Counter, key: str, total: int) -> float:
    return counter[key] / total if total else 0.0


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    graded = [r for r in rows if "error" not in r]
    out: dict[str, Any] = {
        "n_samples": len(rows),
        "n_generate_errors": len(rows) - len(graded),
        "n_judge_failures": sum(1 for r in graded if r["verdict"] is None),
        "groups": {},
        "checks": [],
    }

    for group in GROUPS:
        g = [r for r in graded if r["group"] == group]
        if not g:
            continue
        judged = [r for r in g if r["verdict"]]
        verdicts = Counter(r["verdict"] for r in judged)
        n = len(judged)
        fmt_ok = sum(1 for r in g if r["format_ok"]) / len(g)
        stats: dict[str, Any] = {
            "n": len(g),
            "n_judged": n,
            "verdicts": dict(verdicts),
            "format_ok_rate": round(fmt_ok, 3),
            "avg_bubbles": round(sum(r["bubbles"] for r in g) / len(g), 2),
            "avg_chars_per_bubble": round(
                sum(r["total_chars"] for r in g) / max(1, sum(r["bubbles"] for r in g)), 1,
            ),
            "avg_total_chars": round(sum(r["total_chars"] for r in g) / len(g), 1),
        }

        checks: list[tuple[str, float, float, bool]] = [
            (f"{group}.format_ok", fmt_ok, FORMAT_PASS_RATE, fmt_ok >= FORMAT_PASS_RATE),
        ]
        if group == "fact":
            enc = _rate(verdicts, "encyclopedic", n)
            comp = _rate(verdicts, "companion", n)
            checks += [
                ("fact.encyclopedic(≤)", enc, FACT_MAX_ENCYCLOPEDIC, enc <= FACT_MAX_ENCYCLOPEDIC),
                ("fact.companion(≥)", comp, FACT_MIN_COMPANION, comp >= FACT_MIN_COMPANION),
            ]
        elif group == "chitchat":
            nat = _rate(verdicts, "natural", n)
            off = _rate(verdicts, "off_topic", n)
            short = [r for r in g if len(r["message"]) <= SHORT_INPUT_CHARS]
            len_ok = (
                sum(1 for r in short if r["total_chars"] <= SHORT_INPUT_MAX_REPLY_CHARS)
                / len(short)
            ) if short else 1.0
            stats["short_input_length_ok_rate"] = round(len_ok, 3)
            checks += [
                ("chitchat.natural(≥)", nat, CHITCHAT_MIN_NATURAL, nat >= CHITCHAT_MIN_NATURAL),
                ("chitchat.off_topic(≤)", off, CHITCHAT_MAX_OFF_TOPIC, off <= CHITCHAT_MAX_OFF_TOPIC),
                ("chitchat.short_len_ok(≥)", len_ok, CHITCHAT_MIN_LENGTH_OK,
                 len_ok >= CHITCHAT_MIN_LENGTH_OK),
            ]
        elif group == "outofwindow":
            honest = _rate(verdicts, "honest_uncertainty", n)
            denial = _rate(verdicts, "flat_denial", n)
            blames = _rate(verdicts, "blames_user", n)
            plays = _rate(verdicts, "plays_along", n)
            checks += [
                ("oow.honest(≥)", honest, OUTOFWINDOW_MIN_HONEST,
                 honest >= OUTOFWINDOW_MIN_HONEST),
                ("oow.flat_denial(≤)", denial, OUTOFWINDOW_MAX_FLAT_DENIAL,
                 denial <= OUTOFWINDOW_MAX_FLAT_DENIAL),
                ("oow.blames_user(≤)", blames, OUTOFWINDOW_MAX_BLAMES_USER,
                 blames <= OUTOFWINDOW_MAX_BLAMES_USER),
                ("oow.plays_along(≤)", plays, OUTOFWINDOW_MAX_PLAYS_ALONG,
                 plays <= OUTOFWINDOW_MAX_PLAYS_ALONG),
            ]
        elif group == "falsepremise":
            push = _rate(verdicts, "correct_pushback", n)
            plays = _rate(verdicts, "plays_along", n)
            evasive = _rate(verdicts, "evasive", n)
            checks += [
                ("fp.pushback(≥)", push, FALSEPREMISE_MIN_PUSHBACK,
                 push >= FALSEPREMISE_MIN_PUSHBACK),
                ("fp.plays_along(≤)", plays, FALSEPREMISE_MAX_PLAYS_ALONG,
                 plays <= FALSEPREMISE_MAX_PLAYS_ALONG),
                ("fp.evasive(≤)", evasive, FALSEPREMISE_MAX_EVASIVE,
                 evasive <= FALSEPREMISE_MAX_EVASIVE),
            ]
        elif group == "emotion":
            openings = Counter(
                J.classify_emotion_opening(r["verdict"], r["reply"]) for r in judged
            )
            stats["openings"] = dict(openings)
            ack = _rate(openings, "acknowledge_first", n)
            adv = _rate(openings, "advice_first", n)
            checks += [
                ("emotion.acknowledge_first(≥)", ack, EMOTION_MIN_ACKNOWLEDGE_FIRST,
                 ack >= EMOTION_MIN_ACKNOWLEDGE_FIRST),
                ("emotion.advice_first(≤)", adv, EMOTION_MAX_ADVICE_FIRST,
                 adv <= EMOTION_MAX_ADVICE_FIRST),
            ]

        out["groups"][group] = stats
        out["checks"] += [
            {"name": name, "value": round(val, 3), "threshold": thr, "passed": ok}
            for name, val, thr, ok in checks
        ]

    # 没有任何检查项时不能算通过 —— all([]) 为真会让"全部生成失败"报成 PASS.
    out["passed"] = bool(out["checks"]) and all(c["passed"] for c in out["checks"])
    return out


def print_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    print("\n════════ 结果 ════════")
    for group, s in summary["groups"].items():
        print(f"\n[{group}] n={s['n']} (judged {s['n_judged']})")
        print(f"  判定分布      {s['verdicts']}")
        if "openings" in s:
            print(f"  首句策略      {s['openings']}")
        print(f"  格式合规      {s['format_ok_rate']:.1%}")
        print(f"  平均气泡数    {s['avg_bubbles']}   (真人 IM ≈ {HUMAN_IM_LINES_PER_TURN})")
        print(f"  平均每泡字数  {s['avg_chars_per_bubble']}   (真人 IM ≈ {HUMAN_IM_CHARS_PER_LINE})")
        print(f"  平均总字数    {s['avg_total_chars']}")
        if "short_input_length_ok_rate" in s:
            print(f"  短输入长度达标 {s['short_input_length_ok_rate']:.1%}")

    print("\n── 阈值检查 ──")
    for c in summary["checks"]:
        mark = "PASS" if c["passed"] else "FAIL"
        print(f"  {mark}  {c['name']:<32} {c['value']:.1%}  (线 {c['threshold']:.0%})")

    failing = [c["name"].split(".")[0] for c in summary["checks"] if not c["passed"]]
    if failing:
        print("\n── 不达标项的样本 ──")
        for group in dict.fromkeys(failing):
            bad = [
                r for r in rows
                if r.get("group") == group
                and r.get("verdict") in {"encyclopedic", "over_elaborate", "off_topic"}
                or (group == "emotion" and r.get("verdict") in
                    {"providing_suggestions", "information"})
            ]
            for r in bad[:6]:
                print(f"  [{r['verdict']}] «{r['message']}»")
                print(f"      → {r['reply'][:150]}")

    print(f"\n总判定: {'PASS' if summary['passed'] else 'FAIL'}")
    if summary["n_judge_failures"]:
        print(f"(评审解析失败 {summary['n_judge_failures']} 条, 未计入分母)")


def diff_baseline(summary: dict[str, Any], baseline_path: Path) -> None:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    old = {c["name"]: c["value"] for c in baseline.get("summary", baseline).get("checks", [])}
    print("\n── 与 baseline 对比 ──")
    for c in summary["checks"]:
        before = old.get(c["name"])
        if before is None:
            print(f"  {c['name']:<32}      —  → {c['value']:.1%}  (新增)")
            continue
        delta = c["value"] - before
        arrow = "↑" if delta > 0.001 else ("↓" if delta < -0.001 else "=")
        print(f"  {c['name']:<32} {before:.1%} → {c['value']:.1%}  {arrow}{abs(delta):.1%}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-id")
    parser.add_argument("--group", choices=GROUPS)
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_CASE)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--json", dest="json_out")
    parser.add_argument("--baseline")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument(
        "--judge-model",
        help="provider:model for the judge, e.g. dashscope:qwen3.5-flash. "
             "Defaults to the production utility model.",
    )
    args = parser.parse_args()

    await db.connect()
    try:
        # 必须显式装载 runtime config: 不装载时 get_chat_model() 会静默回落到
        # env 默认值 (REMOTE_CHAT_MODEL), 而生产的模型是存在 SystemConfig 里的
        # admin 覆盖值 —— 2026-07-25 第一版基线就是这样测错了模型, 整整一轮
        # 335 次生成打在了 deepseek-v4-pro 上而不是线上的 doubao-seed-character.
        await ensure_loaded()
        agent = await _pick_agent(args.agent_id)
        judge_model = build_judge_model(args.judge_model)
        cases = tuple(c for c in ALL_CASES if not args.group or c.group == args.group)
        chat_model = get_chat_model()
        print(f"agent : {agent.name} ({agent.id[:8]})")
        print(f"model : {getattr(chat_model, 'model_name', getattr(chat_model, 'model', '?'))}")
        print(f"cases: {len(cases)} × {args.samples} 次 = {len(cases) * args.samples} 次生成\n")

        print(f"judge : {args.judge_model or '(production utility model)'}")
        if not args.skip_calibration and not await run_calibration(
            judge_model, args.concurrency,
        ):
            raise SystemExit(
                "评审器没通过校准 — 它连无争议的样本都分不开, 主评测结果不可信. "
                "先修 rubric 或换评审模型, 或 --skip-calibration 强跑 (结果仅供参考)."
            )

        rows = await run_cases(agent, judge_model, cases, args.samples, args.concurrency)
        summary = summarise(rows)
        print_report(summary, rows)

        # 先落盘再做对比: 对比是可选的锦上添花, 让它有机会吃掉一轮几百次 LLM
        # 调用的结果 (缺 baseline 文件就崩) 是不可接受的.
        if args.json_out:
            Path(args.json_out).write_text(
                json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"\nwrote {args.json_out}")
        if args.baseline:
            try:
                diff_baseline(summary, Path(args.baseline))
            except (OSError, json.JSONDecodeError) as e:
                print(f"\n(baseline 对比跳过: {e})")
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
