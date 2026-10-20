"""Opt-in live model qualification through the actual reply executor.

Only synthetic domain IO, agent, messages and memories; no database connection,
real message delivery or background task. Production prompt *snapshot* supplied
explicitly. Run old/candidate checkouts separately with identical case inputs.
Provider keys come from the environment and are never written to reports.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
import time
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

from incident_cases import CASES


async def run(args, cases=None):
    if (args.source_root / ".env").exists():
        raise ValueError("Use an isolated source checkout without .env")
    sys.path.insert(0, str(args.source_root))
    os.environ.update(DATABASE_URL="postgresql://test:test@127.0.0.1:1/time_eval",
                      REDIS_URL="redis://127.0.0.1:1/15", TRACING_BACKEND="none")
    from openai import AsyncOpenAI
    from pytest import MonkeyPatch
    from app.config import settings
    from app.services.chat import orchestrator as chat, intent_handlers, intent_replies, reply_generate
    from app.services.chat.data_fetch_phase import FetchedContext
    from app.services.chat.intent_dispatcher import IntentResult, IntentType
    from app.services.chat.prompt_builder import compute_reengagement_gap_seconds
    from app.services.memory.retrieval.context_selector import select_context
    from app.services.memory.retrieval import timeline as timeline_module
    from app.services.prompting import store
    from tests.g02_harness_support import configure_pair

    rows = json.loads(args.prompt_snapshot.read_text())
    snapshot = {r["key"]: (r["content"], r["is_enabled"]) for r in rows}
    original_tiers = {kind: getattr(chat, f"_memory_{kind}_reply") for kind in ("weak", "medium", "strong", "l3")}
    client = AsyncOpenAI(api_key=os.environ["ARK_API_KEY"], base_url=os.environ["ARK_BASE_URL"],
                         timeout=60, max_retries=1)
    judge = AsyncOpenAI(api_key=os.environ["DASHSCOPE_API_KEY"], base_url=os.environ["DASHSCOPE_BASE_URL"],
                        timeout=60, max_retries=1)
    report = {"source_root": str(args.source_root), "model": args.model,
              "prompt_snapshot_sha256": hashlib.sha256(args.prompt_snapshot.read_bytes()).hexdigest(),
              "samples": args.samples, "runs": [],
              "limitations": "Synthetic IO and forced routing; actual prompt rendering, model and reply normalization. No production delivery."}
    try:
        for case in (CASES if cases is None else cases):
            for sample in range(args.samples):
                captured = []
                async def generate(messages):
                    started = time.monotonic()
                    result = await client.chat.completions.create(model=args.model, messages=messages,
                        temperature=.7, max_tokens=700)
                    text = result.choices[0].message.content or ""
                    captured.append({"messages": messages, "raw": text, "duration_seconds": time.monotonic()-started,
                                     "usage": result.usage.model_dump() if result.usage else None})
                    return text

                async def live_text(_model, prompt):
                    return await generate([{"role": "user", "content": str(prompt)}])

                async def live_main(messages, **_):
                    return await generate(messages), False

                with MonkeyPatch.context() as patch:
                    io = configure_pair(patch)
                    io.now = datetime.fromisoformat(case["now"].replace("Z", "+00:00"))
                    patch.setattr(settings, "chat_executor", "langgraph")
                    patch.setattr(chat, "compute_reengagement_gap_seconds", compute_reengagement_gap_seconds)
                    patch.setattr(chat, "build_time_context", lambda: io.now.astimezone(
                        timezone(timedelta(hours=8))).isoformat())
                    patch.setattr(reply_generate, "_run_main_llm", live_main)
                    patch.setattr(intent_replies, "invoke_text", live_text)
                    try:
                        from app.services.chat import temporal_guard
                    except ImportError:
                        temporal_guard = None  # baseline predates this module
                    if temporal_guard is not None:
                        async def live_check(_model, prompt):
                            started = time.monotonic()
                            result = await judge.chat.completions.create(
                                model=getattr(args, "guard_model", "qwen3.5-flash"), temperature=0, max_tokens=250,
                                extra_body={"enable_thinking": False},
                                response_format={"type": "json_object"},
                                messages=[{"role": "user", "content": str(prompt)}])
                            verdict = json.loads(result.choices[0].message.content)
                            captured.append({"kind": "temporal_check", "verdict": verdict, "duration_seconds": time.monotonic()-started, "usage": result.usage.model_dump() if result.usage else None})
                            return verdict
                        patch.setattr(temporal_guard, "get_utility_model", lambda: None)
                        patch.setattr(temporal_guard, "invoke_json", live_check)

                    patch.setattr(intent_replies, "get_chat_model", lambda: None)
                    for kind, fn in original_tiers.items():
                        patch.setattr(chat, f"_memory_{kind}_reply", fn)
                    from app.services import persona_grounding
                    patch.setattr(persona_grounding, "verify_generated_locations", AsyncMock(return_value=[]))
                    history = [SimpleNamespace(id=f"history-{i}", role=role, content=text, metadata={},
                        createdAt=io.now-timedelta(seconds=case["gap"] + (len(case["history"])-i-1)*30))
                        for i, (role, text) in enumerate(case["history"])]
                    if "history_rows" in case:
                        history = [SimpleNamespace(id=f"history-{i}", role=row["role"], content=row["content"],
                            metadata=row.get("metadata", {}), createdAt=datetime.fromisoformat(row["createdAt"]))
                            for i, row in enumerate(case["history_rows"])]
                    if case.get("received_at"):
                        history.append(SimpleNamespace(id="u-new", role="user", content=case["message"], metadata={},
                            createdAt=datetime.fromisoformat(case["received_at"].replace("Z", "+00:00"))))
                    io.db.message.find_many.side_effect = lambda **_: list(reversed(history))
                    memories = select_context([dict(id="synthetic-meal", source="ai", content="9月8日午饭吃了三明治",
                        main_category="生活", sub_category="日常", importance=.7, similarity=.9, rank_score=.8,
                        occur_time="2026-09-08T04:00:00Z", statement_time="2026-09-08T04:05:00Z",
                        created_at="2026-09-08T04:05:00Z", last_accessed_at=io.now.isoformat())]) if case["memory"] else []
                    if "memories" in case:
                        memories = select_context(case["memories"])
                    timeline_rows = case.get("timeline_rows", [])
                    timeline_text = timeline_module.format_timeline(timeline_module.build_timeline(timeline_rows)) or None
                    chat.fetch_parallel_context.return_value = FetchedContext(
                        memory_relevance="strong" if memories else "weak", classified_memories=memories,
                        user_emotion={"emotion": "平静", "intensity": 20}, timeline=timeline_text)
                    intent = IntentType.CURRENT_STATE if case["route"] == "current_state" else IntentType.NONE
                    chat.detect_intent_unified.return_value = IntentResult(intent=intent, confidence=1)
                    intent_handlers.resolve_implicit_time.return_value = (io.now, "在家整理工作台")
                    token = store._prompt_snapshot.set(snapshot)
                    try:
                        events = [e async for e in chat.stream_chat_response(
                            "c-1", case["message"], io.agent, "u-1", forced_intent=intent,
                            reply_context={"received_at": case.get("received_at", io.now.isoformat())} )]
                    finally:
                        store.reset_prompt_snapshot(token)
                    texts = []
                    for event in events:
                        data = json.loads(event["data"])
                        if event["event"] == "reply" and data.get("text"):
                            texts.append(data["text"])
                    answer = " ".join(texts)
                    if not captured or not answer or not any(e["event"] == "done" for e in events):
                        raise RuntimeError(f"Incomplete actual reply flow for {case['id']}")
                    try:
                        verdict = await judge.chat.completions.create(model=args.judge_model, temperature=0,
                            max_tokens=1200, extra_body={"enable_thinking": False},
                            response_format={"type": "json_object"}, messages=[{"role": "user", "content":
                                getattr(args, "judging_instruction", "审核合成对话测试，不续写。只评判case.criterion里的明确条件，其他措辞不影响通过。不得添加额外条件：禁止追问某件旧事不等于禁止所有提问；“可以”不是“必须”。输出JSON "
                                "{\"pass\":true或false,\"reason\":\"简短理由\"}。\n")
                            +json.dumps({"case": case, "reply": answer}, ensure_ascii=False)}])
                        judge_raw = verdict.choices[0].message.content
                        try:
                            assessment = json.loads(judge_raw)
                            if not isinstance(assessment, dict) or type(assessment.get("pass")) is not bool:
                                raise ValueError("Invalid assessment")
                        except (ValueError, TypeError):
                            # Retain the actual reply for review; missing judge evidence
                            # is never a pass and must not erase the completed sample.
                            assessment = {"pass": None, "reason": "invalid_model_judgement", "raw": judge_raw}
                    except Exception as exc:
                        assessment = {"pass": None, "reason": "judge_unavailable", "error_type": type(exc).__name__}
                    report["runs"].append({"case": case["id"], "sample": sample, "reply": answer,
                                           "judge": assessment, "calls": captured})
                    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
                    args.output.chmod(0o600)
                    print(json.dumps({"case": case["id"], "sample": sample, "pass": assessment["pass"]}), flush=True)
    finally:
        await client.close()
        await judge.close()
    report["passed"] = sum(r["judge"]["pass"] is True for r in report["runs"])
    report["total"] = len(report["runs"])
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"passed": report["passed"], "total": report["total"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--prompt-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--judge-model", default="qwen3.5-flash")
    parser.add_argument("--samples", type=int, default=3, choices=range(1,6))
    parser.add_argument("--live", action="store_true", required=True)
    asyncio.run(run(parser.parse_args()))
