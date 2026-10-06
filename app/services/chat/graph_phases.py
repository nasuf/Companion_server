"""Adapters for the production chat phases (baseline 51a1501).

Domain decisions and prompt calls stay in the existing services. This module
only adapts their results and events to the invocation-owned graph context.
It deliberately has no checkpoint or replay implementation.
"""

from __future__ import annotations


async def load_turn(ctx):
    services = ctx.services
    ctx.recent_messages = await services.db.message.find_many(
        where={"conversationId": ctx.conversation_id},
        order={"createdAt": "desc"},
        take=services._HISTORY_FETCH_LIMIT,
    )
    ctx.recent_messages.reverse()
    ctx.messages_dicts = [
        {
            "id": getattr(m, "id", None),
            "role": m.role,
            "content": services.render_message_content_for_prompt(
                m.content, m.metadata if isinstance(m.metadata, dict) else None
            ),
            "createdAt": m.createdAt.isoformat()
            if getattr(m, "createdAt", None)
            else None,
        }
        for m in ctx.recent_messages
        if not services.is_offering_received_notice(
            m.metadata if isinstance(m.metadata, dict) else None
        )
    ]
    ctx.messages_dicts = services._ensure_current_user_message(
        ctx.messages_dicts,
        user_message=ctx.user_message,
        user_message_id=ctx.user_message_id,
        reply_context=ctx.reply_context,
    )
    ctx.current_turn_ids = services._current_turn_message_ids(
        ctx.reply_context, ctx.user_message_id
    )
    if ctx.current_turn_ids:
        ctx.reply_context = dict(ctx.reply_context or {})
        ctx.reply_context["turn_message_ids"] = sorted(ctx.current_turn_ids)
    ctx.aggregated_turn_text = ctx.user_message
    ctx.current_achievement_turn_id = services._achievement_turn_id(
        ctx.current_turn_ids
    )
    ctx.covered_until_user_ts = services._max_user_created_at(ctx.messages_dicts)
    ctx.reengagement_gap_seconds = services.compute_reengagement_gap_seconds(
        ctx.messages_dicts, exclude_ids=ctx.current_turn_ids
    )
    ctx.session_recap_task: services.asyncio.Task | None = None
    ctx.previous_assistant = None
    if not ctx.sub_intent_mode:
        ctx.previous_assistant = services._previous_assistant_message(
            ctx.recent_messages, ctx.user_message_id
        )
        if ctx.previous_assistant is not None:
            services._fire_background(
                services._record_memory_retrieval_feedback(
                    assistant_message_id=ctx.previous_assistant.id,
                    assistant_reply=ctx.previous_assistant.content,
                    user_message=ctx.user_message,
                    user_message_id=ctx.user_message_id,
                )
            )
    yield ctx.completed("load_turn")


async def guard(ctx):
    services = ctx.services
    ctx.crisis_decision = await services.run_crisis_guard(
        conversation_id=ctx.conversation_id,
        user_id=ctx.user_id,
        workspace_id=ctx.workspace_id,
        agent_id=ctx.agent_id,
        user_message=ctx.user_message,
        sub_intent_mode=ctx.sub_intent_mode,
        messages_dicts=ctx.messages_dicts,
        user_message_id=ctx.user_message_id,
    )
    ctx.crisis_force_intent = ctx.crisis_decision.crisis_force_intent
    ctx.crisis_followup_active = ctx.crisis_decision.crisis_followup_active
    ctx.crisis_care_turn = ctx.crisis_decision.crisis_care_turn
    ctx.recent_crisis_context = ctx.crisis_decision.recent_crisis_context
    ctx.crisis_followup_check_mode = ctx.crisis_decision.crisis_followup_check_mode
    ctx.cached_patience = (
        ctx.crisis_decision.cached_patience
        if ctx.crisis_decision.cached_patience is not None
        else ctx.parent_patience
        if ctx.parent_patience is not None
        else services.PATIENCE_MAX
    )
    ctx.recent_context_text = services.format_recent_context(
        ctx.messages_dicts,
        exclude_message_id=ctx.user_message_id,
        exclude_message_ids=ctx.current_turn_ids,
    )
    if not ctx.crisis_decision.skip_boundary:
        ctx.boundary_ctx = services.BoundaryPhaseCtx(
            conversation_id=ctx.conversation_id,
            agent_id=ctx.agent_id,
            user_id=ctx.user_id,
            agent=ctx.agent,
            user_message=ctx.user_message,
            sub_intent_mode=ctx.sub_intent_mode,
            parent_patience=ctx.parent_patience,
            tracer=ctx.tracer,
            short_circuit_fn=ctx.boundary_reply,
            fire_background_fn=services._fire_background,
            bg_memory_pipeline_fn=services._bg_memory_pipeline,
            recent_context=ctx.recent_context_text,
        )
        async for evt in services.run_boundary(ctx.boundary_ctx):
            yield evt
        if ctx.boundary_ctx.stopped:
            return
        ctx.cached_patience = ctx.boundary_ctx.cached_patience
    yield ctx.completed("guard")


async def pending(ctx):
    services = ctx.services
    if ctx.crisis_care_turn and (not ctx.sub_intent_mode):
        await services.discard_pending_states_for_crisis(ctx.conversation_id)
    if not ctx.sub_intent_mode and (not ctx.crisis_care_turn):
        ctx.preflight_ctx = services.PreflightCtx(
            conversation_id=ctx.conversation_id,
            agent_id=ctx.agent_id,
            user_id=ctx.user_id,
            agent=ctx.agent,
            tracer=ctx.tracer,
            short_circuit_fn=services.functools.partial(
                ctx.system_reply,
                turn_user_message_ids=sorted(ctx.current_turn_ids),
                workspace_id=ctx.workspace_id,
            ),
        )
        async for evt in services.resolve_recent_undo(
            ctx.user_message, ctx.preflight_ctx
        ):
            yield evt
        if ctx.preflight_ctx.stopped:
            return
        async for evt in services.resolve_pending_contradiction(
            ctx.user_message, ctx.preflight_ctx
        ):
            yield evt
        if ctx.preflight_ctx.stopped:
            return
        async for evt in services.resolve_pending_deletion(
            ctx.user_message, ctx.preflight_ctx
        ):
            yield evt
        if ctx.preflight_ctx.stopped:
            return
        async for evt in services.resolve_retrieval_feedback_correction(
            user_message=ctx.user_message,
            previous_assistant=ctx.previous_assistant,
            ctx=ctx.preflight_ctx,
            workspace_id=ctx.workspace_id,
        ):
            yield evt
        if ctx.preflight_ctx.stopped:
            return
        filler_emoji = services.build_filler_emoji_reply(
            ctx.user_message,
            previous_assistant_text=getattr(ctx.previous_assistant, "content", None),
        )
        if filler_emoji is not None:
            services.logger.info(
                f"[FILLER-EMOJI] reply={filler_emoji}",
                extra={
                    "event": services.EVT_FILLER_EMOJI,
                    "filler_preview": ctx.user_message[:10],
                    "emoji": filler_emoji,
                },
            )
            ctx.preflight_ctx.last_short_circuit_reply = filler_emoji
            for evt in await ctx.short_reply(
                filler_emoji,
                ctx.conversation_id,
                ctx.agent_id,
                ctx.user_id,
                trace_id=ctx.tracer.safe_trace_id,
                turn_user_message_ids=sorted(ctx.current_turn_ids),
                workspace_id=ctx.workspace_id,
            ):
                yield evt
            ctx.tracer.close()
            return
    yield ctx.completed("pending")


async def prepare_reads(ctx):
    services = ctx.services
    ctx.current_state_fast_path = (
        ctx.forced_intent is None
        and (not ctx.sub_intent_mode)
        and (not ctx.crisis_care_turn)
        and services.detect_current_state_fast_path(ctx.user_message)
    )
    ctx.response_diagnostics: dict[str, services.Any] = {
        "version": 1,
        "reply_path": None,
        "memory_relevance": None,
        "main_prompt_built": False,
        "main_prompt_build_ms": None,
        "memory_retrieval_skipped_reason": None,
        "empty_prompt_sections_removed_count": None,
        "intent_fast_path": "current_state_phrase"
        if ctx.current_state_fast_path
        else None,
        "crisis_guard_status": ctx.crisis_decision.status,
        "crisis_guard_reason": ctx.crisis_decision.reason,
        "crisis_semantic_checked": ctx.crisis_decision.semantic_checked,
        "crisis_semantic_detected": ctx.crisis_decision.semantic_detected,
        "crisis_boundary_attack_present": ctx.crisis_decision.boundary_attack_present
        if ctx.crisis_decision.crisis_care_turn
        else None,
    }
    ctx.fetch_task: services.asyncio.Task | None = None
    ctx.crisis_memory_task: services.asyncio.Task | None = None
    ctx.crisis_portrait_task: services.asyncio.Task | None = None
    early_parsed_times: list = []
    if ctx.crisis_force_intent or ctx.crisis_followup_active:
        from app.services.memory.retrieval.safety import (
            retrieve_crisis_followup_memories,
            retrieve_crisis_memories,
        )

        ctx.retrieve_crisis_followup_memories = retrieve_crisis_followup_memories
        ctx.retrieve_crisis_memories = retrieve_crisis_memories
        from app.services.portrait import get_latest_portrait

        ctx.get_latest_portrait = get_latest_portrait
        if ctx.crisis_followup_active:
            ctx.crisis_memory_task = ctx.spawn(
                ctx.retrieve_crisis_followup_memories(
                    ctx.user_message,
                    ctx.user_id,
                    recent_context=ctx.recent_crisis_context,
                    workspace_id=ctx.workspace_id,
                )
            )
        else:
            ctx.crisis_memory_task = ctx.spawn(
                ctx.retrieve_crisis_memories(
                    ctx.user_message, ctx.user_id, workspace_id=ctx.workspace_id
                )
            )
        if ctx.agent_id:
            ctx.crisis_portrait_task = ctx.spawn(
                ctx.get_latest_portrait(ctx.user_id, ctx.agent_id)
            )
    elif ctx.forced_intent is None and (not ctx.current_state_fast_path):
        early_parsed_times = (
            services.parse_time_expressions(ctx.user_message)
            if not ctx.skip_time_memory_lookup
            and services.has_explicit_time(ctx.user_message)
            else []
        )
        ctx.fetch_task = ctx.spawn(
            services.fetch_parallel_context(
                user_id=ctx.user_id,
                agent_id=ctx.agent_id,
                workspace_id=ctx.workspace_id,
                user_message=ctx.user_message,
                messages_dicts=ctx.messages_dicts,
                parsed_times=early_parsed_times,
            )
        )
        ctx.session_recap_task = ctx.spawn(
            services.get_or_build_session_recap(
                ctx.conversation_id,
                ctx.messages_dicts,
                gap_seconds=ctx.reengagement_gap_seconds,
                exclude_ids=ctx.current_turn_ids,
            )
        )
        if ctx.previous_assistant is not None and (not ctx.crisis_care_turn):
            ctx.continuation_task = ctx.spawn(
                services.build_topic_continuation(
                    conversation_id=ctx.conversation_id,
                    previous_assistant=ctx.previous_assistant,
                    replied_at=services._turn_started_at(
                        ctx.messages_dicts, ctx.current_turn_ids
                    ),
                    user_message=ctx.user_message,
                    history=ctx.recent_messages,
                    current_turn_ids=ctx.current_turn_ids,
                    agent=ctx.agent,
                    offering_turn=bool(ctx.offering_context),
                    patience_low=ctx.cached_patience < services.PATIENCE_NORMAL_MIN,
                )
            )
    yield ctx.completed("prepare_reads")


async def identify(ctx):
    services = ctx.services
    if ctx.forced_intent is not None:
        ctx.detected_intent = services.IntentResult(
            intent=ctx.forced_intent, confidence=1.0
        )
    elif ctx.crisis_force_intent:
        ctx.detected_intent = services.IntentResult(
            intent=services.IntentType.CRISIS, confidence=1.0
        )
    elif ctx.crisis_followup_active:
        ctx.detected_intent = services.IntentResult(
            intent=services.IntentType.CRISIS,
            confidence=1.0,
            metadata=ctx.crisis_decision.intent_metadata,
        )
    elif ctx.current_state_fast_path:
        ctx.detected_intent = services.IntentResult(
            intent=services.IntentType.CURRENT_STATE,
            confidence=1.0,
            metadata={"fast_path": "current_state_phrase"},
        )
    else:
        context_text = await services._fetch_intent_context(
            ctx.conversation_id,
            exclude_id=ctx.user_message_id,
            exclude_ids=ctx.current_turn_ids,
            exclude_content=ctx.user_message if not ctx.user_message_id else None,
        )
        ctx.detected_intent = await services.detect_intent_unified(
            ctx.user_message, context=context_text
        )
        if ctx.detected_intent.intent != services.IntentType.NONE:
            services.logger.info(
                f"[INTENT-LLM] '{ctx.user_message[:30]}' → {ctx.detected_intent.intent.value} (labels={ctx.detected_intent.metadata.get('llm_labels')})"
            )
        ctx.detected_intent = services._downgrade_non_explicit_current_schedule_query(
            ctx.detected_intent, ctx.user_message, ctx.response_diagnostics
        )
    if ctx.detected_intent.intent != services.IntentType.CRISIS:
        ctx.detected_intent = services._downgrade_non_explicit_schedule_adjust(
            ctx.detected_intent,
            ctx.user_message,
            ctx.response_diagnostics,
            previous_assistant_text=getattr(ctx.previous_assistant, "content", None),
        )
    yield ctx.completed("identify")


async def prepare_routes(ctx):
    services = ctx.services

    async def _cancel_fetch_task() -> None:
        """短路 / 异常时调用: cancel + 等待 propagate, 避免 orphan task warning.

        session_recap_task 一并取消 (review 发现): 意图短路路径不消费摘要,
        让它跑完是浪费一次 LLM 调用. 模块内部全链路吞异常, cancel 安全.
        """
        for task in (ctx.fetch_task, ctx.session_recap_task, ctx.continuation_task):
            if task is None or task.done():
                continue
            task.cancel()
            try:
                await task
            except (services.asyncio.CancelledError, Exception):
                pass

    ctx._cancel_fetch_task = _cancel_fetch_task
    if ctx.forced_intent is None and (not ctx.crisis_care_turn):
        fragments = (
            ctx.detected_intent.metadata.get("fragments")
            if ctx.detected_intent.metadata
            else None
        )
        if fragments and len(fragments) > 1:
            fragments = services._filter_non_explicit_sub_fragments(
                fragments, ctx.response_diagnostics
            )
            if len(fragments) <= 1:
                fragments = None
        if fragments and len(fragments) > 1:
            primary_label = next(
                (
                    lb
                    for lb, it in services.LABEL_TO_INTENT.items()
                    if it == ctx.detected_intent.intent and lb in fragments
                ),
                None,
            )
            if primary_label and fragments.get(primary_label):
                ctx.user_message = (
                    str(fragments[primary_label]).strip() or ctx.user_message
                )
            ctx.pending_sub_fragments = {
                lb: str(txt).strip()
                for lb, txt in fragments.items()
                if lb != primary_label and str(txt).strip()
            }
            if ctx.pending_sub_fragments:
                services.logger.info(
                    f"[INTENT-MULTI] primary={ctx.detected_intent.intent.value} sub={list(ctx.pending_sub_fragments.keys())}",
                    extra={
                        "event": services.EVT_INTENT_SPLIT,
                        "intent_primary": ctx.detected_intent.intent.name,
                        "sub_intents": list(ctx.pending_sub_fragments.keys()),
                        "n_sub": len(ctx.pending_sub_fragments),
                    },
                )
    ctx.sc_ctx = services.ShortCircuitCtx(
        conversation_id=ctx.conversation_id,
        agent_id=ctx.agent_id,
        user_id=ctx.user_id,
        workspace_id=ctx.workspace_id,
        agent=ctx.agent,
        reply_context=ctx.reply_context,
        tracer=ctx.tracer,
        save_replies_fn=ctx.save_replies,
        pending_sub_fragments=ctx.pending_sub_fragments,
        sub_intent_mode=ctx.sub_intent_mode,
        reply_index_offset=ctx.reply_index_offset,
        cached_patience=ctx.cached_patience,
        recent_context=ctx.recent_context_text,
        response_diagnostics=ctx.response_diagnostics,
        covered_until_user_ts=ctx.covered_until_user_ts,
        achievement_turn_final=ctx.achievement_turn_final,
        defer_turn_finalization=True,
    )
    ctx.response_diagnostics["crisis_followup_check_mode"] = (
        ctx.crisis_followup_check_mode if ctx.crisis_followup_active else None
    )
    if ctx.agent_id:
        try:
            from app.services.achievements.service import handle_intent_event

            intent_metadata = dict(ctx.detected_intent.metadata or {})
            intent_metadata["confidence"] = ctx.detected_intent.confidence
            intent_metadata["source"] = "chat_intent"
            services._fire_background(
                handle_intent_event(
                    intent=ctx.detected_intent.intent.value,
                    user_id=ctx.user_id,
                    agent_id=ctx.agent_id,
                    workspace_id=ctx.workspace_id,
                    conversation_id=ctx.conversation_id,
                    message_id=ctx.user_message_id,
                    metadata=intent_metadata,
                )
            )
        except Exception as achievement_err:
            services.logger.debug(f"[ACH] intent event hook skipped: {achievement_err}")
    yield ctx.completed("prepare_routes")


async def early_route(ctx):
    services = ctx.services
    if ctx.detected_intent.intent == services.IntentType.CONVERSATION_END:
        await ctx._cancel_fetch_task()
        async for evt in services.handle_conversation_end(
            ctx.user_message, ctx.sc_ctx, services._intent_llm_reply
        ):
            yield evt
        return
    if ctx.detected_intent.intent == services.IntentType.APOLOGY_PROMISE:
        handled, events = await services.handle_apology_promise(
            ctx.user_message, ctx.sc_ctx
        )
        if handled and events is not None:
            await ctx._cancel_fetch_task()
            async for evt in events:
                yield evt
            return
    elif ctx.detected_intent.intent == services.IntentType.DELETION:
        handled, events = await services.handle_deletion(ctx.user_message, ctx.sc_ctx)
        if handled and events is not None:
            await ctx._cancel_fetch_task()
            async for evt in events:
                yield evt
            return
    if ctx.detected_intent.intent == services.IntentType.CRISIS:
        await ctx._cancel_fetch_task()
        if ctx.crisis_memory_task is None:
            from app.services.memory.retrieval.safety import (
                retrieve_crisis_followup_memories,
                retrieve_crisis_memories,
            )

            ctx.retrieve_crisis_followup_memories = retrieve_crisis_followup_memories
            ctx.retrieve_crisis_memories = retrieve_crisis_memories
            if ctx.detected_intent.metadata.get("followup"):
                ctx.crisis_memory_task = ctx.spawn(
                    ctx.retrieve_crisis_followup_memories(
                        ctx.user_message,
                        ctx.user_id,
                        recent_context=ctx.recent_crisis_context,
                        workspace_id=ctx.workspace_id,
                    )
                )
            else:
                ctx.crisis_memory_task = ctx.spawn(
                    ctx.retrieve_crisis_memories(
                        ctx.user_message, ctx.user_id, workspace_id=ctx.workspace_id
                    )
                )
        if ctx.crisis_portrait_task is None and ctx.agent_id:
            from app.services.portrait import get_latest_portrait

            ctx.get_latest_portrait = get_latest_portrait
            ctx.crisis_portrait_task = ctx.spawn(
                ctx.get_latest_portrait(ctx.user_id, ctx.agent_id)
            )
        crisis_classified: list = []
        crisis_portrait: services.Any = None
        if ctx.crisis_memory_task is not None:
            try:
                retrieval_result = await ctx.crisis_memory_task
                if isinstance(retrieval_result, dict):
                    crisis_classified = retrieval_result.get("memories") or []
                elif isinstance(retrieval_result, list):
                    crisis_classified = retrieval_result
            except Exception as e:
                services.logger.warning(f"Crisis memory fetch failed: {e}")
        if ctx.crisis_portrait_task is not None:
            try:
                crisis_portrait = await ctx.crisis_portrait_task
            except Exception as e:
                services.logger.warning(f"Crisis portrait fetch failed: {e}")
        crisis_accessed_ids = [
            getattr(m, "id", "") for m in crisis_classified if getattr(m, "id", "")
        ]
        if crisis_accessed_ids:
            services._fire_background(
                services.log_memory_access(
                    ctx.user_id, crisis_accessed_ids, workspace_id=ctx.workspace_id
                )
            )
            services._fire_background(
                services.record_memory_usage(contributed_ids=crisis_accessed_ids)
            )
        if ctx.detected_intent.metadata.get("followup"):
            async for evt in services.handle_crisis_followup(
                ctx.user_message,
                ctx.sc_ctx,
                classified_memories=crisis_classified,
                portrait=crisis_portrait,
                safety_check_mode=ctx.detected_intent.metadata.get(
                    "safety_check_mode", "none"
                ),
            ):
                yield evt
        else:
            async for evt in services.handle_crisis(
                ctx.user_message,
                ctx.sc_ctx,
                classified_memories=crisis_classified,
                portrait=crisis_portrait,
            ):
                yield evt
        return
    yield ctx.completed("early_route")


async def prepare_context(ctx):
    services = ctx.services
    topic_info = await services.push_topic(
        ctx.conversation_id, ctx.user_message, gap_seconds=ctx.reengagement_gap_seconds
    )
    ctx.topic_context = topic_info or None
    ctx.mbti = services.get_mbti(ctx.agent)
    ctx.l3_task: services.asyncio.Task[tuple[list[str], str]] | None = None
    if ctx.fetch_task is not None:
        ctx.fetched = await ctx.fetch_task
        ctx.l3_task = ctx.spawn(
            services.maybe_awaken_l3(
                ctx.user_message,
                ctx.user_id,
                ctx.workspace_id,
                ctx.detected_intent,
                ctx.fetched.memory_relevance,
                services._l3_trigger_analyze,
                enhanced_query=ctx.fetched.enhanced_query,
                l1_l2_count=len(ctx.fetched.classified_memories or []),
                recent_context=services.format_recent_context(
                    ctx.messages_dicts, exclude_message_ids=ctx.current_turn_ids
                ),
            )
        )
    elif ctx.current_state_fast_path:
        ctx.schedule = (
            await services.get_cached_schedule(ctx.agent_id) if ctx.agent_id else None
        )
        ctx.ai_status = (
            services.get_current_status(ctx.schedule) if ctx.schedule else None
        )
        ctx.schedule_context = (
            services.format_schedule_context(ctx.ai_status) if ctx.ai_status else None
        )
        ctx.fetched = services.FetchedContext(
            memory_relevance="weak",
            classified_memories=[],
            memory_strings=[],
            schedule=ctx.schedule,
            ai_status=ctx.ai_status,
            schedule_context=ctx.schedule_context,
        )
        ctx.response_diagnostics["memory_retrieval_skipped_reason"] = (
            "current_state_fast_path"
        )
    else:
        parsed_times = (
            services.parse_time_expressions(ctx.user_message)
            if not ctx.skip_time_memory_lookup
            and services.has_explicit_time(ctx.user_message)
            else []
        )
        ctx.fetched = await services.fetch_parallel_context(
            user_id=ctx.user_id,
            agent_id=ctx.agent_id,
            workspace_id=ctx.workspace_id,
            user_message=ctx.user_message,
            messages_dicts=ctx.messages_dicts,
            parsed_times=parsed_times,
            detected_intent=ctx.detected_intent,
            l3_trigger_classify_fn=services._l3_trigger_analyze,
        )
    ctx.memory_relevance = ctx.fetched.memory_relevance
    ctx.classified_memories = ctx.fetched.classified_memories
    ctx.prompt_user_emotion = ctx.fetched.user_emotion
    ctx.portrait = ctx.fetched.portrait
    ctx.schedule = ctx.fetched.schedule
    ctx.topic_intimacy = ctx.fetched.topic_intimacy
    ctx.time_memories = ctx.fetched.time_memories
    ctx.l3_memories = ctx.fetched.l3_memories
    ctx.ai_status = ctx.fetched.ai_status
    ctx.schedule_context = ctx.fetched.schedule_context
    ctx.needs_web_search = ctx.fetched.needs_web_search
    ctx.response_diagnostics.update(
        {
            "memory_relevance": ctx.memory_relevance,
            "memory_retrieval_skipped_reason": ctx.response_diagnostics.get(
                "memory_retrieval_skipped_reason"
            )
            or ("weak_relevance" if ctx.memory_relevance == "weak" else None),
        }
    )

    async def _cancel_l3_task() -> None:
        """短路 / 异常时调用: cancel L3 task + 等待 propagate, 防 orphan task warning."""
        if ctx.l3_task is None or ctx.l3_task.done():
            return
        ctx.l3_task.cancel()
        try:
            await ctx.l3_task
        except (services.asyncio.CancelledError, Exception):
            pass

    ctx._cancel_l3_task = _cancel_l3_task
    ctx.delay_context = None
    if ctx.reply_context:
        received_status = ctx.reply_context.get("received_status") or {}
        received_activity = (
            str(received_status.get("activity", "")).strip() or "处理自己的事"
        )
        received_status_label = str(received_status.get("status", "idle"))
        received_at = str(ctx.reply_context.get("received_at", ""))
        elapsed = services.actual_delay_seconds(ctx.reply_context)
        if elapsed is not None and elapsed < 60:
            rounded_delay = max(1, round(elapsed))
            delay_reason_text = await services.explain_delay_reason(
                str(ctx.reply_context.get("delay_reason", "")),
                activity=received_activity,
                status=received_status_label,
            )
            ctx.delay_context = {
                "received_at": received_at,
                "activity": received_activity,
                "status": received_status_label,
                "delay_seconds": rounded_delay,
                "delay_reason": delay_reason_text,
            }
    ctx.relational_context = services.detect_relational_context(
        ctx.user_message, ctx.prompt_user_emotion
    )
    ctx.time_context = services.build_time_context()
    ctx.intimacy_stage = services.get_relationship_stage(ctx.topic_intimacy)
    yield ctx.completed("prepare_context")


async def special_route(ctx):
    services = ctx.services
    if ctx.detected_intent.intent == services.IntentType.SCHEDULE_ADJUST:
        handled, events = await services.handle_schedule_adjust(
            ctx.user_message,
            ctx.sc_ctx,
            schedule=ctx.schedule,
            ai_status=ctx.ai_status,
            portrait=ctx.portrait,
            user_emotion=ctx.prompt_user_emotion,
            topic_intimacy=ctx.topic_intimacy,
            mbti=ctx.mbti,
        )
        if handled and events is not None:
            async for evt in events:
                yield evt
            await ctx._cancel_l3_task()
            return
    if ctx.detected_intent.intent == services.IntentType.RECORD_REQUEST:
        handled, events = await services.handle_record_request(
            ctx.user_message, ctx.sc_ctx
        )
        if handled and events is not None:
            async for evt in events:
                yield evt
            await ctx._cancel_l3_task()
            return
    ctx.detected_intent = services._route_current_schedule_query_to_current_state(
        ctx.detected_intent, ctx.user_message, ctx.response_diagnostics
    )
    if ctx.detected_intent.intent == services.IntentType.SCHEDULE_QUERY:
        query_type = ctx.detected_intent.metadata.get("query_type", "current")
        handled, events, schedule_ctx_for_prompt = await services.handle_schedule_query(
            ctx.user_message,
            ctx.sc_ctx,
            schedule=ctx.schedule,
            ai_status=ctx.ai_status,
            portrait=ctx.portrait,
            user_emotion=ctx.prompt_user_emotion,
            query_type=query_type,
        )
        if schedule_ctx_for_prompt is not None:
            ctx.schedule_context = schedule_ctx_for_prompt
        if handled and events is not None:
            async for evt in events:
                yield evt
            await ctx._cancel_l3_task()
            return
    ctx.detected_intent = services._downgrade_non_explicit_current_state(
        ctx.detected_intent, ctx.user_message, ctx.response_diagnostics
    )
    if ctx.detected_intent.intent == services.IntentType.CURRENT_STATE:
        handled, events = await services.handle_current_state(
            ctx.user_message,
            ctx.sc_ctx,
            ai_status=ctx.ai_status,
            schedule_context=ctx.schedule_context,
            portrait=ctx.portrait,
            user_emotion=ctx.prompt_user_emotion,
        )
        if handled and events is not None:
            async for evt in events:
                yield evt
            await ctx._cancel_l3_task()
            return
    yield ctx.completed("special_route")


async def prepare_reply(ctx):
    services = ctx.services
    patience_instruction = await services.get_patience_prompt_instruction(
        ctx.cached_patience
    )
    ctx.contradiction_inquiry: str | None = None
    if ctx.memory_relevance in ("strong", "medium"):
        try:
            conflict = await services.detect_l1_contradiction(
                ctx.user_message, ctx.user_id, workspace_id=ctx.workspace_id
            )
            if conflict:
                inquiry = await services.generate_contradiction_inquiry(
                    conflict, agent_name=ctx.agent.name if ctx.agent else "AI"
                )
                ctx.contradiction_inquiry = inquiry
                await services.save_pending_contradiction(ctx.conversation_id, conflict)
                services.logger.info(
                    f"L1 contradiction detected: {conflict.get('conflict_description', '')}",
                    extra={
                        "event": services.EVT_MEMORY_CONTRADICTION,
                        "conflict_summary": (
                            conflict.get("conflict_description") or ""
                        )[:80],
                    },
                )
        except Exception as e:
            services.logger.warning(f"Contradiction detection failed: {e}")
    ctx.last_reply_count = (
        await services.load_last_reply_count(ctx.conversation_id)
        if ctx.conversation_id
        else None
    )
    if ctx.relational_context:
        ctx.reply_count = 1
        ctx.allow_count_variation = False
    elif ctx.contradiction_inquiry:
        ctx.reply_count = 1
        ctx.allow_count_variation = False
    else:
        ctx.reply_count = services.pick_reply_count_target(
            ctx.last_reply_count, services.MAX_REPLY_COUNT
        )
        ctx.allow_count_variation = True
    ctx.max_reply_count = services.MAX_REPLY_COUNT
    ctx.max_total = services.MAX_TOTAL_CHARS
    if ctx.l3_task is not None:
        try:
            ctx.l3_memories, l3_trigger_label = await ctx.l3_task
            ctx.fetched.l3_memories = ctx.l3_memories
            ctx.fetched.l3_trigger_label = l3_trigger_label
        except (services.asyncio.CancelledError, Exception) as e:
            services.logger.warning(f"L3 awakening failed: {e}")
    ctx.response_diagnostics.update(
        {
            "memory_relevance": ctx.memory_relevance,
            "memory_retrieval_skipped_reason": ctx.response_diagnostics.get(
                "memory_retrieval_skipped_reason"
            )
            or ("weak_relevance" if ctx.memory_relevance == "weak" else None),
            "empty_prompt_sections_removed_count": None,
        }
    )

    async def _build_main_chat_messages() -> list[dict]:
        started = services.perf_counter()
        prompt_diagnostics: dict[str, services.Any] = {}
        music_context = None
        try:
            from app.services import music as music_service

            active_music = await music_service.get_active_co_listening(
                conversation_id=ctx.conversation_id
            )
            if active_music and active_music.track:
                tpl = await services.get_prompt_text("music.co_listening_context")
                music_context = services.safe_format(
                    tpl,
                    {
                        "current_song": active_music.track.title,
                        "current_artist": active_music.track.artist,
                    },
                )
        except Exception as music_context_err:
            services.logger.debug(
                f"[MUSIC] co-listening context skipped: {music_context_err}"
            )
        try:
            expression_habits = await services.sample_expression_habits(
                ctx.agent_id, ctx.user_id
            )
        except Exception as expr_err:
            services.logger.debug(f"[EXPR] habits sample skipped: {expr_err}")
            expression_habits = []
        session_recap = None
        if ctx.session_recap_task is not None:
            try:
                session_recap = await ctx.session_recap_task
            except Exception as recap_err:
                services.logger.debug(f"[RECAP] session recap skipped: {recap_err}")
        try:
            relation_meta_line = services.format_relation_meta_line(
                await services.get_relation_meta(ctx.conversation_id)
            )
        except Exception as meta_err:
            services.logger.debug(f"[RELMETA] skipped: {meta_err}")
            relation_meta_line = ""
        ai_mood_text = services.format_ai_mood_text(
            await services.load_ai_mood(ctx.conversation_id)
        )
        offline_activity = None
        try:
            from app.services.offline.module_settings import is_activity_enabled

            if await is_activity_enabled():
                from app.services.offline import repository as _offline_repo

                offline_activity = await _offline_repo.get_active_activity_brief(
                    ctx.user_id, ctx.workspace_id
                )
        except Exception as _off_err:
            services.logger.debug(f"[offline-ctx] skipped: {_off_err}")
            offline_activity = None
        system_prompt = await services.build_system_prompt(
            agent=ctx.agent,
            memories=ctx.classified_memories,
            delay_context=ctx.delay_context,
            portrait=ctx.portrait,
            topic_context=ctx.topic_context,
            music_context=music_context,
            user_emotion=ctx.prompt_user_emotion,
            ai_status=ctx.ai_status,
            patience_instruction=patience_instruction,
            reply_count=ctx.reply_count,
            reply_total=ctx.max_total,
            intimacy_stage=ctx.intimacy_stage,
            time_context=ctx.time_context,
            time_memories=ctx.time_memories or None,
            timeline=getattr(ctx.fetched, "timeline", None),
            l3_memories=ctx.l3_memories or None,
            memory_relevance=ctx.memory_relevance,
            reengagement_gap_seconds=ctx.reengagement_gap_seconds,
            topic_continuation=topic_continuation,
            session_recap=session_recap,
            relation_meta_line=relation_meta_line,
            ai_mood_text=ai_mood_text,
            expression_habits=expression_habits or None,
            red_packet_context=ctx.red_packet_context,
            gift_context=ctx.gift_context,
            offline_activity=offline_activity,
            last_reply_count=ctx.last_reply_count,
            needs_web_search=ctx.needs_web_search,
            discussed_titles=services.extract_discussed_titles(
                ctx.messages_dicts, current_message=ctx.user_message
            )
            if ctx.needs_web_search
            else None,
            diagnostics=prompt_diagnostics,
        )
        ctx.response_diagnostics["main_prompt_built"] = True
        ctx.response_diagnostics["main_prompt_build_ms"] = round(
            (services.perf_counter() - started) * 1000, 3
        )
        ctx.response_diagnostics.update(prompt_diagnostics)
        reply_messages = services.collapse_turn_fragments(
            ctx.messages_dicts,
            turn_message_ids=ctx.current_turn_ids,
            combined_text=ctx.aggregated_turn_text,
            combined_id=ctx.user_message_id,
        )
        drop_older = (
            services.RECAP_GAP_SECONDS
            if ctx.reengagement_gap_seconds is not None
            and ctx.reengagement_gap_seconds >= services.RECAP_GAP_SECONDS
            else None
        )
        return services.build_chat_messages(
            system_prompt, reply_messages, drop_older_than_seconds=drop_older
        )

    ctx._build_main_chat_messages = _build_main_chat_messages
    injected_ids: list[str] = []
    if ctx.classified_memories:
        injected_ids.extend(
            (
                getattr(m, "id", "")
                for m in ctx.classified_memories
                if getattr(m, "id", "")
            )
        )
    candidate_only = [
        mid
        for mid in getattr(ctx.fetched, "candidate_ids", []) or []
        if mid and mid not in set(injected_ids)
    ]
    if injected_ids:
        services._fire_background(
            services.log_memory_access(
                ctx.user_id, injected_ids, workspace_id=ctx.workspace_id
            )
        )
    if injected_ids or candidate_only:
        services._fire_background(
            services.record_memory_usage(
                contributed_ids=injected_ids, accessed_ids=candidate_only
            )
        )
    from app.services.interaction.chat_management import (
        clamp_reply_delay_seconds,
    )
    from app.services.interaction.chat_management import (
        reply_delay_enabled as _reply_delay_enabled,
    )

    if _reply_delay_enabled():
        reply_delay = services.calculate_reply_delay(
            len(ctx.user_message), mbti=ctx.mbti
        )
        queued_delay = clamp_reply_delay_seconds(
            float((ctx.reply_context or {}).get("delay_seconds", 0.0) or 0.0)
        )
        conceptual_delay = clamp_reply_delay_seconds(max(reply_delay, queued_delay))
        if ctx.delivered_from_queue:
            actual_sleep = min(reply_delay, 1.5)
        else:
            actual_sleep = min(conceptual_delay, 2.0)
            if conceptual_delay > 5.0:
                yield {
                    "event": "delay",
                    "data": services.json.dumps({"duration": conceptual_delay}),
                }
        if actual_sleep > 0:
            await services.asyncio.sleep(actual_sleep)
    if ctx.relational_context:
        topic_continuation = None
    else:
        topic_continuation = await services.await_topic_continuation(
            ctx.continuation_task, ctx.response_diagnostics
        )
    ctx.continuation_lines = topic_continuation.lines if topic_continuation else []
    if ctx.continuation_lines:
        ctx.max_reply_count = max(
            1, services.MAX_REPLY_COUNT - len(ctx.continuation_lines)
        )
        ctx.reply_count = min(ctx.reply_count, ctx.max_reply_count)
    yield ctx.completed("prepare_reply")


async def generate_reply(ctx):
    services = ctx.services
    (
        ctx.replies,
        _raw_response,
        ctx.reply_is_fallback,
        ctx.reply_emotion_pre,
    ) = await services._generate_reply(
        user_id=ctx.user_id,
        workspace_id=ctx.workspace_id,
        contradiction_inquiry=ctx.contradiction_inquiry,
        detected_intent=ctx.detected_intent,
        memory_relevance=ctx.memory_relevance,
        relational_context=ctx.relational_context,
        schedule_context=ctx.schedule_context,
        delay_context=ctx.delay_context,
        l3_memories=ctx.l3_memories,
        classified_memories=ctx.classified_memories or [],
        messages_dicts=ctx.messages_dicts,
        portrait=ctx.portrait,
        prompt_user_emotion=ctx.prompt_user_emotion,
        user_message=ctx.user_message,
        agent=ctx.agent,
        chat_messages_factory=ctx._build_main_chat_messages,
        needs_web_search=ctx.needs_web_search,
        reply_count=ctx.reply_count,
        max_reply_count=ctx.max_reply_count,
        max_total=ctx.max_total,
        last_reply_count=ctx.last_reply_count,
        allow_count_variation=ctx.allow_count_variation,
        tier_fns={
            "weak": services._memory_weak_reply,
            "medium": services._memory_medium_reply,
            "strong": services._memory_strong_reply,
            "l3": services._memory_l3_reply,
        },
        truncate_fn=lambda text, max_len: services.truncate_at_sentence(
            services._clean_reply_part(text), max_len
        ),
        pipe_fallback_fn=services.split_and_validate_replies,
        reply_emotion_fn=services._ai_reply_emotion,
        reengagement_gap_seconds=ctx.reengagement_gap_seconds,
        force_main_prompt=bool(ctx.offering_context) or bool(ctx.continuation_lines),
        diagnostics=ctx.response_diagnostics,
    )
    yield ctx.completed("generate_reply")


async def normalize_reply(ctx):
    services = ctx.services
    if ctx.continuation_lines and (not ctx.reply_is_fallback):
        ctx.replies = [*ctx.continuation_lines, *ctx.replies]
    ctx.full_response = " ".join(ctx.replies)
    if ctx.reply_emotion_pre is not None:
        reply_emotion = ctx.reply_emotion_pre
    else:
        reply_emotion = await services._ai_reply_emotion(ctx.full_response)
    if reply_emotion.get("emotion"):
        services.logger.info(
            f"[REPLY-EMO] emotion={reply_emotion['emotion']} intensity={reply_emotion.get('intensity', 0)}",
            extra={
                "event": services.EVT_REPLY_EMOTION,
                "ai_emotion": reply_emotion["emotion"],
                "intensity": reply_emotion.get("intensity", 0),
                "reply_text_len": len(ctx.full_response),
            },
        )
    ctx.emitted_replies: list[dict] = []
    ctx.prepared_voice = None
    from app.services.speech_output.policy import VoiceContext, should_generate_voice

    client_supports_voice = bool((ctx.reply_context or {}).get("client_supports_voice"))
    elapsed_for_voice = services.actual_delay_seconds(ctx.reply_context)
    if elapsed_for_voice is not None and elapsed_for_voice >= 60:
        voice_context = VoiceContext.SYSTEM
    else:
        voice_context = VoiceContext.NORMAL_CHAT
    if not ctx.reply_is_fallback and await should_generate_voice(
        context=voice_context, client_supports_voice=client_supports_voice
    ):
        try:
            from app.services.speech_output.delivery import prepare_voice_output

            ctx.prepared_voice = await prepare_voice_output(
                text=ctx.full_response,
                user_id=ctx.user_id,
                agent=ctx.agent,
                conversation_id=ctx.conversation_id,
                source="chat",
                emotion=reply_emotion.get("emotion"),
                intensity=reply_emotion.get("intensity"),
            )
        except Exception as voice_error:
            services.logger.warning(
                "[TTS] chat synthesis failed; falling back to text: %s",
                type(voice_error).__name__,
            )
    if ctx.prepared_voice is not None:
        ctx.voice_data: dict = {
            "text": ctx.prepared_voice.transcript,
            "index": ctx.reply_index_offset,
            "display_mode": "voice",
            "attachments": [ctx.prepared_voice.metadata],
        }
        if reply_emotion.get("emotion"):
            ctx.voice_data["ai_emotion"] = reply_emotion["emotion"]
            ctx.voice_data["emotion_intensity"] = int(
                reply_emotion.get("intensity", 0) or 0
            )
        ctx.emitted_replies.append(ctx.voice_data)
    else:
        async for evt in services._emit_replies(
            ctx.replies,
            reply_context=ctx.reply_context,
            reply_index_offset=ctx.reply_index_offset,
            sub_intent_mode=ctx.sub_intent_mode,
            agent=ctx.agent,
            user_message=ctx.user_message,
            delay_reply_fn=services._delay_explanation_reply,
            fallback_fn=services._intent_llm_reply,
            emitted_replies=ctx.emitted_replies,
            reply_emotion=reply_emotion,
            reply_is_fallback=ctx.reply_is_fallback,
            conversation_id=ctx.conversation_id,
        ):
            yield evt
    yield ctx.completed("normalize_reply")


async def persist_reply(ctx):
    services = ctx.services
    if ctx.emitted_replies and ctx.covered_until_user_ts is not None:
        first = ctx.emitted_replies[0]
        if isinstance(first, dict):
            first.setdefault(
                "covered_until_user_ts", ctx.covered_until_user_ts.isoformat()
            )
    if ctx.emitted_replies:
        first = ctx.emitted_replies[0]
        if isinstance(first, dict):
            first.setdefault("response_diagnostics", ctx.response_diagnostics)
        else:
            ctx.emitted_replies[0] = {
                "text": str(first),
                "response_diagnostics": ctx.response_diagnostics,
            }
        prompt_render_traces = services.snapshot_prompt_render_traces()
        if prompt_render_traces:
            first = ctx.emitted_replies[0]
            if isinstance(first, dict):
                first.setdefault("prompt_render_traces", prompt_render_traces)
            else:
                ctx.emitted_replies[0] = {
                    "text": str(first),
                    "prompt_render_traces": prompt_render_traces,
                }
        from app.services.memory.retrieval.trace import (
            build_retrieval_quality_analysis,
            snapshot_retrieval_traces,
        )

        retrieval_traces = snapshot_retrieval_traces()
        if retrieval_traces:
            retrieval_analysis = build_retrieval_quality_analysis(
                retrieval_traces,
                assistant_reply=ctx.full_response,
                user_message=ctx.user_message,
            )
            first = ctx.emitted_replies[0]
            if isinstance(first, dict):
                first.setdefault("memory_retrievals", retrieval_traces)
                if retrieval_analysis:
                    first.setdefault("memory_retrieval_analysis", retrieval_analysis)
            else:
                ctx.emitted_replies[0] = {
                    "text": str(first),
                    "memory_retrievals": retrieval_traces,
                }
                if retrieval_analysis:
                    ctx.emitted_replies[0]["memory_retrieval_analysis"] = (
                        retrieval_analysis
                    )
    ctx.first_assistant_message_id = await ctx.save_replies(
        ctx.conversation_id,
        ctx.emitted_replies,
        trace_id=ctx.tracer.trace_id if ctx.tracer.is_active else None,
        turn_user_message_ids=list(ctx.current_turn_ids),
        achievement_turn_id=ctx.current_achievement_turn_id,
        achievement_turn_final=ctx.achievement_turn_final
        and (not ctx.pending_sub_fragments),
    )
    if ctx.offering_context and (not ctx.sub_intent_mode):
        try:
            from app.services import offerings as offerings_svc

            await offerings_svc.mark_offering_received(
                offering_id=str(ctx.offering_context["offering_id"]),
                user_id=ctx.user_id,
                conversation_id=ctx.conversation_id,
            )
        except Exception:
            services.logger.exception("offering receive mark failed")
    if ctx.prepared_voice is not None:
        if ctx.first_assistant_message_id:
            try:
                from app.services.speech_output.delivery import (
                    bind_prepared_voice_output,
                )

                await bind_prepared_voice_output(
                    ctx.prepared_voice, message_id=ctx.first_assistant_message_id
                )
                ctx.voice_attachment_bound = True
                ctx.voice_data = ctx.emitted_replies[0]
                ctx.voice_data["assistant_message_id"] = ctx.first_assistant_message_id
                public_voice_data = {
                    key: ctx.voice_data[key]
                    for key in (
                        "text",
                        "index",
                        "display_mode",
                        "attachments",
                        "assistant_message_id",
                        "ai_emotion",
                        "emotion_intensity",
                    )
                    if key in ctx.voice_data
                }
                yield {"event": "reply", "data": services.json.dumps(public_voice_data)}
            except Exception as voice_bind_error:
                services.logger.warning(
                    "[TTS] chat attachment bind failed; falling back to text: %s",
                    type(voice_bind_error).__name__,
                )
                from app.services.speech_output.delivery import (
                    discard_prepared_voice_output,
                )

                await discard_prepared_voice_output(ctx.prepared_voice)
                await services.db.execute_raw(
                    "\n                        UPDATE messages\n                        SET metadata = COALESCE(metadata, '{}'::jsonb)\n                            - 'attachments' - 'display_mode'\n                        WHERE id = $1\n                        ",
                    ctx.first_assistant_message_id,
                )
                yield {
                    "event": "reply",
                    "data": services.json.dumps(
                        {
                            "text": ctx.prepared_voice.transcript,
                            "index": ctx.reply_index_offset,
                            "assistant_message_id": ctx.first_assistant_message_id,
                        }
                    ),
                }
        else:
            from app.services.speech_output.delivery import (
                discard_prepared_voice_output,
            )

            await discard_prepared_voice_output(ctx.prepared_voice)
            yield {
                "event": "reply",
                "data": services.json.dumps(
                    {
                        "text": ctx.prepared_voice.transcript,
                        "index": ctx.reply_index_offset,
                    }
                ),
            }
    services.logger.info(
        f"[REPLY-EMIT] n={len(ctx.emitted_replies)} sub_intent_mode={ctx.sub_intent_mode}",
        extra={
            "event": services.EVT_REPLY_EMITTED,
            "n_replies": len(ctx.emitted_replies),
            "sub_intent_mode": ctx.sub_intent_mode,
            "is_fallback": ctx.reply_is_fallback,
        },
    )
    sentence_bubble_count = sum(
        (1 for r in ctx.emitted_replies if not r.get("delay_explanation"))
    )
    if sentence_bubble_count > 0:
        services._fire_background(
            services.save_last_reply_count(
                ctx.conversation_id, ctx.reply_index_offset + sentence_bubble_count
            )
        )
    yield ctx.completed("persist_reply")
