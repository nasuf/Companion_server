"""Run a real renderer in a network-denied child, fake only parent transport."""

import pytest
from app.services.quality.replay import replay_turn
from app.services.quality import transport
from app.services.prompting.registry import PROMPT_DEFINITIONS


@pytest.fixture(autouse=True)
def measure_real_child_when_requested(monkeypatch):
    """Qualification measures the child too; production never loads coverage."""
    import os
    from app.services.quality import replay

    directory = os.environ.get("QUALITY_REPLAY_COVERAGE_DIR")
    if not directory:
        return
    from pathlib import Path

    target = Path(directory).resolve()
    target.mkdir(parents=True, exist_ok=True)
    spawn = replay.asyncio.create_subprocess_exec

    async def measured(*args, **kwargs):
        if args[1:] == ("-m", "evals.quality.child"):
            args = (
                args[0],
                "-m",
                "coverage",
                "run",
                "--branch",
                "--parallel-mode",
                "--source=evals.quality",
                "--data-file=" + str(target / ".coverage"),
                *args[1:],
            )
        return await spawn(*args, **kwargs)

    monkeypatch.setattr(replay.asyncio, "create_subprocess_exec", measured)


@pytest.mark.parametrize(
    "route",
    [
        "main",
        "tier",
        "current_state",
        "boundary",
        "memory_extraction",
        "persona_grounding",
        "web_search",
    ],
)
async def test_renderer_subprocess_does_not_need_business_database(monkeypatch, route):
    calls = []

    async def fake(job, provider, model, messages, **kwargs):
        calls.append((provider, model))
        text = (
            '{"pass":true,"reason":"合成验证"}'
            if kwargs.get("json_mode")
            else "我在家整理陶艺工作台呢。"
        )
        # Temporal checks, if triggered, still use the same budgeted seam.
        if "temporal" in str(messages).lower():
            text = (
                '{"pass":true,"reason":"合成验证"}' if kwargs.get("json_mode") else text
            )
        if route == "memory_extraction":
            text = '{"memories":[{"summary":"用户喜欢乌龙茶","main_category":"偏好边界","sub_category":"饮食喜好","importance":0.7}],"entities":[],"topics":[],"preferences":[]}'
        if route == "persona_grounding":
            text = '{"verdicts":[{"index":0,"allowed":false}]}'
        return text, {"input": 10, "output": 10, "cached_input": 0, "cost_cny": 0.0001}

    monkeypatch.setattr(transport, "complete", fake)
    prompts = [
        {"key": p.key, "content": p.default_text, "is_enabled": p.default_enabled}
        for p in PROMPT_DEFINITIONS
    ]
    case = dict(
        id="isolation-smoke",
        now="2026-10-29T04:00:00Z",
        gap=120,
        history=[],
        memory=False,
        route=route,
        message="你在做什么",
        criterion="回答当前活动，不杜撰用户经历。",
    )
    if route == "boundary":
        from dataclasses import asdict
        from evals.boundary_reply.cases import CASES

        case["boundary_turn"] = asdict(CASES[0].turns[0])
    elif route == "memory_extraction":
        from evals.quality.corpus import load

        case = next(c for c in load() if c["route"] == route)
    elif route == "persona_grounding":
        case["grounding"] = {"city": "杭州", "text": "我在镇江", "allowed": False}
    result = await replay_turn(
        {}, case, prompts, ["ark", "doubao-seed-2-0-lite-260228"]
    )
    assert calls and type(result["judge"]["pass"]) is bool


@pytest.mark.parametrize(
    "frame",
    [
        b'{"type":"call","provider":"untrusted","model":"x"}\n',
        b'{"type":"error","error_type":"render_failed","frames":[]}\n',
        b'{"type":"call","provider":"ark","model":"approved","operation":"web_search"}\n',
        b'{"type":"unsupported"}\n',
        b"not-json\n",
        b"",
    ],
)
async def test_invalid_child_frames_never_spend_and_always_reap(monkeypatch, frame):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from app.services.quality import replay

    class Child:
        returncode = None

        def __init__(self):
            self.stdin = SimpleNamespace(write=lambda _: None, drain=AsyncMock())
            self.stdout = SimpleNamespace(readline=AsyncMock(return_value=frame))

        def terminate(self):
            self.returncode = -15

        async def wait(self):
            return self.returncode

    child = Child()

    async def spawn(*args, **kwargs):
        assert not any("KEY" in k or "TOKEN" in k for k in kwargs["env"])
        assert "quality_no_database" in kwargs["env"]["DATABASE_URL"]
        return child

    monkeypatch.setattr(replay.asyncio, "create_subprocess_exec", spawn)
    paid = AsyncMock()
    monkeypatch.setattr(transport, "complete", paid)
    with pytest.raises((ValueError, RuntimeError)):
        await replay.replay_turn({}, {"route": "main"}, [], ["ark", "approved"])
    paid.assert_not_awaited()
    assert child.returncode == -15


async def test_oversized_initial_frame_never_calls_provider_and_reaps_child(
    monkeypatch,
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from app.services.quality import replay

    child = SimpleNamespace(
        returncode=None,
        stdin=SimpleNamespace(write=lambda _: None, drain=AsyncMock()),
        wait=AsyncMock(),
    )

    def terminate():
        child.returncode = -15

    child.terminate = terminate
    monkeypatch.setattr(
        replay.asyncio, "create_subprocess_exec", AsyncMock(return_value=child)
    )
    paid = AsyncMock()
    monkeypatch.setattr(transport, "complete", paid)
    with pytest.raises(ValueError, match="frame_too_large"):
        await replay.replay_turn(
            {},
            {"route": "main", "message": "x" * replay.MAX_FRAME},
            [],
            ["ark", "approved"],
        )
    paid.assert_not_awaited()
    assert child.returncode == -15


async def test_cancelling_replay_reaps_child(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from app.services.quality import replay

    waiting = asyncio.Event()

    async def read():
        waiting.set()
        await asyncio.Event().wait()

    child = SimpleNamespace(
        returncode=None,
        stdin=SimpleNamespace(write=lambda _: None, drain=AsyncMock()),
        stdout=SimpleNamespace(readline=read),
        wait=AsyncMock(return_value=-15),
    )

    def terminate():
        child.returncode = -15

    child.terminate = terminate
    monkeypatch.setattr(
        replay.asyncio, "create_subprocess_exec", AsyncMock(return_value=child)
    )
    task = asyncio.create_task(
        replay.replay_turn({}, {"route": "main"}, [], ["ark", "approved"])
    )
    await waiting.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert child.returncode == -15


async def test_real_renderer_uses_pinned_context_window(monkeypatch):
    rendered = []

    async def fake(job, provider, model, messages, **kwargs):
        if provider == "ark":
            rendered.append(str(messages))
        text = (
            '{"pass":true,"reason":"合成验证"}'
            if kwargs.get("json_mode")
            else "那就慢慢整理吧。"
        )
        return text, dict(input=10, output=10, cached_input=0, cost_cny=0.0001)

    monkeypatch.setattr(transport, "complete", fake)
    prompts = [
        dict(key=p.key, content=p.default_text, is_enabled=p.default_enabled)
        for p in PROMPT_DEFINITIONS
    ]
    history = []
    for i in range(8):
        history += [
            ["user", f"今天我准备整理第{i}号工具箱。"],
            ["assistant", f"第{i}号整理完就歇会儿吧。"],
        ]
    case = dict(
        id="pinned-window",
        now="2026-10-29T04:00:00Z",
        gap=120,
        history=history,
        memory=False,
        route="tier",
        message="现在我打算继续收拾啦。",
        criterion="回应最新消息。",
    )
    for rounds in (1, 8):
        await replay_turn(
            dict(snapshot=dict(context_policy=dict(rounds=rounds))),
            case,
            prompts,
            ["ark", "doubao-seed-character-260628"],
        )
    assert len(rendered) == 2
    assert "第0号" not in rendered[0] and "第0号" in rendered[1]
    assert "第7号" in rendered[0] and "第7号" in rendered[1]
