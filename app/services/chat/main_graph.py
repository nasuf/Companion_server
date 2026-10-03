"""The chat state graph. State contains JSON; runtime contains disposable IO.

No checkpointer, interrupt, automatic retry or restart is supported here.
A failed node may have performed domain effects and must not be rerun.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Literal, TypedDict

from langgraph.config import get_config, get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from app.services.chat import graph_phases
from app.services.chat.graph_contracts import validate_fields

GRAPH_VERSION = "chat-g01-v1"
DURABLE_EXECUTION_READY = False
PHASES = (
    "load_turn",
    "guard",
    "pending",
    "prepare_reads",
    "identify",
    "prepare_routes",
    "early_route",
    "prepare_context",
    "special_route",
    "prepare_reply",
    "generate_reply",
    "normalize_reply",
    "persist_reply",
)


class ChatState(TypedDict, total=False):
    version: Literal[1]
    phase: str
    next_node: str
    fragment_index: int
    intent: dict
    diagnostics: dict
    completed: bool


@dataclass(frozen=True)
class PhaseOutcome:
    phase: str


def _phase(name):
    async def execute(state: ChatState, runtime: Runtime):
        context = runtime.context
        frame = context.frame
        if (
            state.get("version") != 1
            or state.get("fragment_index") != context.fragment_index
        ):
            raise ValueError("Chat graph invocation and state differ")
        validate_fields(frame, name)
        completed = False
        events = getattr(graph_phases, name)(frame)
        from app.services.chat.local_tracer import graph_node

        callbacks = get_config().get("callbacks")
        with graph_node(getattr(callbacks, "parent_run_id", None)):
            try:
                async for event in events:
                    if isinstance(event, PhaseOutcome):
                        if event.phase != name or completed:
                            raise ValueError("Invalid chat node completion")
                        completed = True
                    else:
                        if (
                            completed
                            or not isinstance(event, dict)
                            or event.get("event") not in {"reply", "delay"}
                        ):
                            raise ValueError("Invalid chat node event")
                        context.note_event(event)
                        get_stream_writer()(event)
            finally:
                await events.aclose()
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError("Chat node cancelled")
        if completed:
            validate_fields(frame, name, output=True)
        following = (
            PHASES[PHASES.index(name) + 1] if name != PHASES[-1] else "fragments"
        )
        # Early returns are intentional guard/handler short circuits.
        result = {
            "phase": name,
            "next_node": following if completed else "fragments",
            "intent": frame.intent_snapshot(),
            "diagnostics": dict(frame.response_diagnostics or {}),
        }
        json.dumps(result, allow_nan=False)
        return result

    return execute


async def _fragments(state: ChatState, runtime: Runtime):
    context = runtime.context
    more = await context.advance_fragment()
    return {
        "phase": "fragments",
        "next_node": "load_turn" if more else "finish_turn",
        "fragment_index": context.fragment_index,
        "intent": {},
        "diagnostics": {},
    }


async def _finish(state: ChatState, runtime: Runtime):
    event = await runtime.context.finish_turn()
    get_stream_writer()(event)
    return {"phase": "finish_turn", "next_node": "stop", "completed": True}


def _build_graph():
    from app.services.chat.graph_runtime import ChatGraphContext

    graph = StateGraph(ChatState, context_schema=ChatGraphContext)
    for name in PHASES:
        graph.add_node(name, _phase(name))
    graph.add_node("fragments", _fragments)
    graph.add_node("finish_turn", _finish)
    graph.add_edge(START, PHASES[0])
    for index, name in enumerate(PHASES):
        following = PHASES[index + 1] if index + 1 < len(PHASES) else "fragments"
        graph.add_conditional_edges(
            name,
            lambda state: state["next_node"],
            {following: following, "fragments": "fragments"},
        )
    graph.add_conditional_edges(
        "fragments",
        lambda state: state["next_node"],
        {"load_turn": "load_turn", "finish_turn": "finish_turn"},
    )
    graph.add_edge("finish_turn", END)
    return graph.compile(name="main_chat", checkpointer=None)


# Built lazily so legacy turns neither import nor compile LangGraph.
_GRAPH = None


async def stream_main(context):
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    final_state = None
    stream = _GRAPH.astream(
        {"version": 1, "fragment_index": 0, "completed": False},
        context=context,
        config={
            "recursion_limit": (len(PHASES) + 1) * 12 + 5,
            "run_name": "main_chat",
            "metadata": {
                "graph": "main_chat",
                "graph_version": GRAPH_VERSION,
                "executor": "langgraph",
                "checkpoint_enabled": False,
            },
        },
        stream_mode=["custom", "values"],
    )
    try:
        async for kind, part in stream:
            if kind == "custom":
                yield part
            elif kind == "values":
                final_state = part
    finally:
        await stream.aclose()
    # LangGraph may terminate after a node cancellation without propagating it.
    if not final_state or not final_state.get("completed"):
        raise asyncio.CancelledError("Chat graph ended before turn completion")
