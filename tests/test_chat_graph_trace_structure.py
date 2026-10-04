"""Exercise actual LangGraph callbacks, without database/model/network effects."""
import pytest
from langchain_core.callbacks import BaseCallbackHandler

from app.services.chat import main_graph


@pytest.mark.asyncio
async def test_named_routes_keep_the_same_graph_path_and_one_terminal_event(monkeypatch):
    from tests.graph_harness_support import configure_chat
    from app.services.chat import orchestrator as chat

    io = configure_chat(monkeypatch)
    names = []
    routes = []

    class Capture(BaseCallbackHandler):
        def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
            names.append(kwargs.get("name"))
            if kwargs.get("name") == "route_next":
                routes.append((inputs.get("phase"), inputs.get("next_node")))

    # Observe the compiled graph used by the real stream_chat_response harness.
    compiled = main_graph._build_graph()

    class ObservedGraph:
        def astream(self, *args, **kwargs):
            kwargs["config"] = {**kwargs["config"], "callbacks": [Capture()]}
            return compiled.astream(*args, **kwargs)

    monkeypatch.setattr(main_graph, "_GRAPH", ObservedGraph())
    events = [event async for event in chat.stream_chat_response("c-1", "今晚想聊天", io.agent, "u-1")]
    assert [event["event"] for event in events] == ["reply", "done"]
    expected = list(main_graph.PHASES) + ["fragments", "finish_turn"]
    assert [name for name in names if name in expected] == expected
    assert "Unnamed" not in names
    assert names.count("route_next") == 14
    assert routes == list(zip(expected[:-1], expected[1:]))
    io.db.message.create.assert_awaited_once()
    chat._save_replies.assert_awaited_once()
    chat.finish_assistant_turn.assert_awaited_once()
    chat._background_post_process.assert_called_once()
    assert compiled.checkpointer is None
