"""One-use isolated renderer; stdin/stdout are bounded parent RPC only."""

import asyncio
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

_PROTOCOL = sys.stdout
_INPUT = sys.stdin


def call(
    provider,
    model,
    messages,
    max_tokens=900,
    json_mode=False,
    operation="chat",
    location=None,
):
    _PROTOCOL.write(
        json.dumps(
            dict(
                type="call",
                provider=provider,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                json_mode=json_mode,
                operation=operation,
                location=location,
            ),
            ensure_ascii=False,
        )
        + "\n"
    )
    _PROTOCOL.flush()
    line = _INPUT.readline(2_000_001)
    if len(line) > 2_000_000 or not line:
        raise RuntimeError("replay_parent_unavailable")
    return json.loads(line)


def prohibit_io(event, args):
    if event in {
        "socket.connect",
        "socket.connect_ex",
        "socket.bind",
        "subprocess.Popen",
        "os.system",
        "os.posix_spawn",
    }:
        raise PermissionError("evaluation_external_io_forbidden")


class RpcClient:
    def __init__(self, api_key, **kwargs):
        self.provider = "ark" if api_key == "isolated-ark" else "dashscope"
        self.chat = SimpleNamespace(completions=self)

    async def create(self, model, messages, **kwargs):
        value = call(
            self.provider,
            model,
            messages,
            kwargs.get("max_tokens", 900),
            bool(kwargs.get("response_format")),
        )
        usage = dict(
            prompt_tokens=value["usage"]["input"],
            completion_tokens=value["usage"]["output"],
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=value["text"]))],
            usage=SimpleNamespace(model_dump=lambda: usage),
        )

    async def close(self):
        pass


async def run(payload):
    import openai

    openai.AsyncOpenAI = RpcClient
    # Fixed placeholders cannot authorise any external request. The audit hook
    # also rejects any missed DB/Redis/HTTP call before the socket connects.
    os.environ.update(
        ARK_API_KEY="isolated-ark",
        DASHSCOPE_API_KEY="isolated-dashscope",
        ARK_BASE_URL="https://unused.invalid",
        DASHSCOPE_BASE_URL="https://unused.invalid",
    )
    sys.addaudithook(prohibit_io)
    from app.services import runtime_config

    # This process has no live runtime_config loader. Pin the administrative
    # window policy too, rather than silently reverting edited windows to code defaults.
    runtime_config.chat_context_policy_overrides = (
        lambda: payload.get("context_policy") or {}
    )
    if payload["case"]["route"] == "boundary":
        return await boundary(payload)
    if payload["case"]["route"] == "memory_extraction":
        return await memory_extraction(payload)
    if payload["case"]["route"] == "persona_grounding":
        return await grounding(payload)
    from evals.temporal_awareness.run_incident_eval import run as incident_run
    from app.services.chat import orchestrator as chat

    original_context = chat._fetch_intent_context
    case = payload["case"]
    route = payload["route"]

    async def provider_call(messages, **options):
        from app.services.chat.prompt_builder import fold_trailing_system

        searching = case["route"] == "web_search"
        if searching != bool(options.get("needs_web_search")):
            raise ValueError("search_route_not_exercised")
        v = call(
            *route,
            fold_trailing_system(messages) if searching else messages,
            operation="web_search" if searching else "chat",
            location=options.get("web_search_location") if searching else None,
        )
        return dict(text=v["text"], usage=v["usage"])

    with tempfile.TemporaryDirectory(prefix="quality-replay-") as folder:
        prompts = Path(folder) / "prompts.json"
        prompts.write_text(json.dumps(payload["prompts"]))
        prompts.chmod(0o600)
        output = Path(folder) / "result.json"
        args = SimpleNamespace(
            source_root=Path.cwd(),
            prompt_snapshot=prompts,
            output=output,
            model=route[1],
            judge_model="qwen3.5-flash",
            guard_model="qwen3.5-flash",
            samples=1,
            provider_call=provider_call,
        )

        def fixture_hook(patches, chat_module, domain, current_case):
            from unittest.mock import AsyncMock
            from app.services.prompting import store

            snapshot = {p["key"]: p for p in payload["prompts"]}
            patches.setattr(
                store.db.prompttemplate,
                "find_unique",
                AsyncMock(
                    side_effect=lambda **kw: SimpleNamespace(
                        isEnabled=snapshot[kw["where"]["key"]]["is_enabled"]
                    )
                    if kw["where"]["key"] in snapshot
                    else None
                ),
            )
            patches.setattr(chat_module, "_fetch_intent_context", original_context)
            if case["route"] == "web_search":
                chat_module.fetch_parallel_context.return_value.needs_web_search = True
            generate = chat_module._generate_reply

            async def force_route(**kwargs):
                if case["route"] in {"main", "web_search"}:
                    kwargs["force_main_prompt"] = True
                return await generate(**kwargs)

            patches.setattr(chat_module, "_generate_reply", force_route)

        args.fixture_hook = fixture_hook
        with redirect_stdout(io.StringIO()):
            await incident_run(args, [case])
        result = json.loads(output.read_text())["runs"][0]
        if case.get("persona_probe"):
            from evals.persona_drift.cases import PERSONA_LEAK_TERMS

            passed = not any(t in result["reply"] for t in PERSONA_LEAK_TERMS)
            if case.get("expected_anchor"):
                passed = passed and case["expected_anchor"] in result["reply"]
            if case["persona_probe"] == "style":
                from evals.persona_drift.judge import build_prompt, parse_verdict

                early = next(t for r, t in case["history"] if r == "assistant")
                prompt = build_prompt(
                    persona="24岁，杭州陶艺师小岚，自然的朋友口吻",
                    question=case["message"],
                    early=early,
                    late=result["reply"],
                )
                value = call(
                    "dashscope",
                    "qwen3.5-flash",
                    [{"role": "user", "content": prompt}],
                    json_mode=True,
                )
                verdict = parse_verdict(value["text"])
                passed = (
                    None
                    if verdict is None
                    else passed
                    and verdict["voice_consistent"]
                    and verdict["persona_match"]
                )
            result["judge"] = {
                "pass": passed,
                "method": "existing_persona_anchor_and_voice_rules",
            }
        return dict(
            reply=result["reply"],
            judge=result["judge"],
            location_queries=result["location_queries"],
        )


def prompt_snapshot(patch, rows):
    """Read only the pinned versions, including explicit disabled templates."""
    from app.services.prompting import store
    from unittest.mock import AsyncMock

    snapshot = {r["key"]: (r["content"], r["is_enabled"]) for r in rows}
    patch.setattr(
        store,
        "db",
        SimpleNamespace(
            prompttemplate=SimpleNamespace(
                find_unique=AsyncMock(
                    side_effect=lambda **kw: SimpleNamespace(
                        isEnabled=snapshot[kw["where"]["key"]][1]
                    )
                    if kw["where"]["key"] in snapshot
                    else None
                )
            )
        ),
    )
    return store._prompt_snapshot.set(snapshot)


async def memory_extraction(payload):
    from pytest import MonkeyPatch
    from app.services.prompting import store
    from app.services.memory.recording import extraction, context_window

    case = payload["case"]

    async def invoke(model, prompt, **kwargs):
        value = call(
            *payload["route"],
            [{"role": "user", "content": str(prompt)}],
            max_tokens=1800,
            json_mode=True,
        )
        kwargs["invocation_metadata"]["actual_model"] = "/".join(payload["route"])
        return json.loads(value["text"])

    with MonkeyPatch.context() as patch:
        patch.setattr(
            extraction,
            "get_chat_model",
            lambda: SimpleNamespace(model=payload["route"][1]),
        )
        patch.setattr(extraction, "invoke_json", invoke)
        token = prompt_snapshot(patch, payload["prompts"])
        try:
            rows = case["history_rows"]
            history = context_window.format_history(context_window.history_rows(rows))
            result = await extraction.extract_memories(
                "user: " + case["message"],
                context_conversation=history,
                side="user",
                source_messages="1. user: " + case["message"],
            )
        finally:
            store.reset_prompt_snapshot(token)
    if result.get("_extraction_error"):
        raise ValueError("memory_extraction_incomplete")
    content = "\n".join(m["content"] for m in result["memories"])
    passed = all(s in content for s in case["required"]) and not any(
        s in content for s in case["forbidden"]
    )
    return dict(reply=content, judge={"pass": passed}, location_queries=0)


async def grounding(payload):
    from pytest import MonkeyPatch
    from app.services.prompting import store

    case = payload["case"]["grounding"]
    items = case.get(
        "items", [{"text": case.get("text"), "allowed": case.get("allowed")}]
    )
    with MonkeyPatch.context() as patch:
        token = prompt_snapshot(patch, payload["prompts"])
        try:
            template = await store.get_prompt_text("persona.grounding_check")
        finally:
            store.reset_prompt_snapshot(token)
    prompt = template.format(
        facts=json.dumps(
            {"name": "小伴", "city": case["city"], "occupation": "客服员", "age": 22},
            ensure_ascii=False,
        ),
        kind=case.get("kind", "reply"),
        question=json.dumps(case.get("question", ""), ensure_ascii=False),
        items=json.dumps(
            [dict(index=i, text=v["text"]) for i, v in enumerate(items)],
            ensure_ascii=False,
        ),
    )
    value = call(
        "dashscope",
        "qwen3.5-plus",
        [{"role": "user", "content": str(prompt)}],
        json_mode=True,
    )
    verdict = json.loads(value["text"])["verdicts"]
    if (
        len(verdict) != len(items)
        or sorted(v["index"] for v in verdict) != list(range(len(items)))
        or any(type(v["allowed"]) is not bool for v in verdict)
    ):
        raise ValueError("invalid_grounding_judgement")
    passed = [v["allowed"] for v in sorted(verdict, key=lambda v: v["index"])] == [
        v["allowed"] for v in items
    ]
    return dict(reply="", judge={"pass": passed}, location_queries=0)


async def boundary(payload):
    from pytest import MonkeyPatch
    from app.services.prompting import store
    from app.services.llm import models
    from evals.boundary_reply.run_eval import _gen
    from evals.boundary_reply.cases import Turn
    from evals.boundary_reply.judge import judge_turn
    from langchain_core.messages import AIMessage

    case = payload["case"]
    turn = Turn(**case["boundary_turn"])

    class Model:
        async def ainvoke(self, messages):
            v = call(
                *payload["route"],
                [dict(role="user", content=m.content) for m in messages],
            )
            return AIMessage(content=v["text"])

    with MonkeyPatch.context() as patch:
        patch.setattr(models, "get_chat_model", lambda: Model())
        snapshot = {
            r["key"]: (r["content"], r["is_enabled"]) for r in payload["prompts"]
        }
        from unittest.mock import AsyncMock

        patch.setattr(
            store,
            "db",
            SimpleNamespace(
                prompttemplate=SimpleNamespace(
                    find_unique=AsyncMock(
                        side_effect=lambda **kw: SimpleNamespace(
                            isEnabled=snapshot[kw["where"]["key"]][1]
                        )
                        if kw["where"]["key"] in snapshot
                        else None
                    )
                )
            ),
        )
        token = store._prompt_snapshot.set(snapshot)
        try:
            reply = await _gen(
                turn,
                [
                    ("用户" if r == "user" else "我") + ": " + t
                    for r, t in case["history"]
                ],
                "温和但有边界的陶艺师小岚",
            )
        finally:
            store.reset_prompt_snapshot(token)
    verdict = judge_turn(reply, [t for r, t in case["history"] if r == "assistant"])
    passed = not (
        turn.expects_redirect
        and not verdict.has_redirect
        or verdict.demands_explanation
        or verdict.self_deprecates
        or turn.expects_closure
        and verdict.restates_grievance
        or verdict.repeats_previous
    )
    return dict(reply=reply, judge={"pass": passed}, location_queries=0)


def main():
    line = _INPUT.readline(2_000_001)
    if not line or len(line) > 2_000_000:
        raise ValueError("invalid_replay_input")
    payload = json.loads(line)
    # Event loop's local socketpair is created before the IO audit hook.
    try:
        result = asyncio.run(run(payload))
    except Exception as exc:
        import traceback

        _PROTOCOL.write(
            json.dumps(
                {
                    "type": "error",
                    "error_type": type(exc).__name__,
                    "frames": [
                        {
                            "file": Path(f.filename).name,
                            "line": f.lineno,
                            "function": f.name,
                        }
                        for f in traceback.extract_tb(exc.__traceback__)[-6:]
                    ],
                }
            )
            + "\n"
        )
        _PROTOCOL.flush()
        return
    _PROTOCOL.write(
        json.dumps(dict(type="result", result=result), ensure_ascii=False) + "\n"
    )
    _PROTOCOL.flush()


if __name__ == "__main__":
    main()
