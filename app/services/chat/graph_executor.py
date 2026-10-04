"""Select an executor once per turn, and preserve a permanent legacy fallback.

Failures after graph execution starts are surfaced, never rerun in legacy.
"""

from __future__ import annotations


def select_executor(conversation_id: str) -> str:
    from app.config import settings

    if settings.chat_executor != "langgraph":
        return "legacy"
    if settings.chat_graph_all_conversations and conversation_id:
        return "langgraph"
    allowed = {
        item.strip()
        for item in settings.chat_graph_conversation_allowlist.split(",")
        if item.strip()
    }
    return "langgraph" if conversation_id in allowed else "legacy"


async def stream_graph_response(
    conversation_id,
    user_message,
    agent,
    user_id,
    reply_context=None,
    *,
    save_user_message=True,
    user_message_id=None,
    delivered_from_queue=False,
    sub_intent_mode=False,
    forced_intent=None,
    reply_index_offset=0,
    parent_patience=None,
    parent_trace_id=None,
    achievement_turn_final=True,
):
    from app.services.chat import orchestrator as services
    from app.services.chat.graph_runtime import (
        ChatFrame,
        ChatGraphContext,
        ParentTraceView,
    )
    from app.services.chat.main_graph import GRAPH_VERSION, stream_main
    from app.services.llm import usage_tracker
    from app.services.memory.retrieval.trace import (
        reset_retrieval_trace,
        start_retrieval_trace,
    )
    from app.services.prompting.trace_components import (
        reset_prompt_render_trace,
        start_prompt_render_trace,
    )
    from app.services.runtime_config import bind_agent_context, reset_current_agent

    # Existing recursive legacy child calls stay on the legacy executor. Graph
    # children are explicit fragment transitions and never enter this driver.
    if sub_intent_mode:
        raise ValueError("Graph executor cannot start a recursive child turn")
    agent_id = getattr(agent, "id", None)
    conversation = await services.db.conversation.find_unique(
        where={"id": conversation_id}
    )
    workspace_id = getattr(conversation, "workspaceId", None)
    context = None
    tracer = None
    agent_token = usage_token = retrieval_token = prompt_token = None
    try:
        if save_user_message:
            saved = await services.db.message.create(
                data={
                    "conversation": {"connect": {"id": conversation_id}},
                    "role": "user",
                    "content": user_message,
                }
            )
            user_message_id = saved.id
            from app.services.interaction_streak import (
                record_user_message_day_for_conversation,
            )

            services._fire_background(
                record_user_message_day_for_conversation(
                    conversation_id,
                    workspace_id=workspace_id,
                    user_id=getattr(conversation, "userId", None),
                )
            )
        agent_token = await bind_agent_context(agent_id)
        usage_token = usage_tracker.start_session()
        tracer = services.create_tracer(
            user_message, conversation_id, executor="langgraph",
            graph_version=GRAPH_VERSION,
        ).enter()
        retrieval_token = start_retrieval_trace()
        prompt_token = start_prompt_render_trace()
        frame = ChatFrame(
            services=services,
            owner=None,
            conversation_id=conversation_id,
            user_message=user_message,
            agent=agent,
            user_id=user_id,
            reply_context=reply_context,
            user_message_id=user_message_id,
            delivered_from_queue=delivered_from_queue,
            sub_intent_mode=False,
            forced_intent=forced_intent,
            reply_index_offset=reply_index_offset,
            parent_patience=parent_patience,
            achievement_turn_final=False,
            agent_id=agent_id,
            workspace_id=workspace_id,
            tracer=ParentTraceView(tracer),
        )
        context = ChatGraphContext(
            frame=frame,
            original_message=user_message,
            achievement_turn_final=achievement_turn_final,
            next_reply_index=reply_index_offset,
        )
        frame.owner = context
        events = stream_main(context)
        try:
            async for event in events:
                yield event
        finally:
            await events.aclose()
    except BaseException as error:
        from app.services.chat.local_tracer import LocalTracer

        if isinstance(tracer, LocalTracer) and not (context and context.finished):
            tracer.mark_failed(error)
        raise
    finally:
        try:
            if context is not None:
                await context.frame.cleanup()
        finally:
            if usage_token is not None:
                from app.services.llm.usage_repo import write_usage_row
                from app.services.chat.local_tracer import LocalTracer

                if isinstance(tracer, LocalTracer):
                    tracer.note_usage_expected(usage_tracker.session_has_usage_signal())
                summary = usage_tracker.flush_session(usage_token)
                if summary:
                    try:
                        await write_usage_row(
                            summary=summary,
                            conversation_id=conversation_id,
                            agent_id=agent_id,
                            user_id=user_id,
                            trace_id=getattr(tracer, "trace_id", None),
                        )
                    except Exception:
                        services.logger.warning(
                            "[llm-usage] graph usage write failed", exc_info=True
                        )
            if retrieval_token is not None:
                reset_retrieval_trace(retrieval_token)
            if prompt_token is not None:
                reset_prompt_render_trace(prompt_token)
            reset_current_agent(agent_token)
            if tracer is not None:
                tracer.close()
