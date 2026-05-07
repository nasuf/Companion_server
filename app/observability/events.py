"""结构化 event 名常量.

用法 (调用站点):
    from app.observability.events import EVT_INTENT_DETECTED
    logger.info(
        f"[INTENT-LLM] labels={labels}",
        extra={"event": EVT_INTENT_DETECTED, "intent_primary": primary.name},
    )

约定:
- event 名 = "<domain>.<verb>" 短句, 全小写
- domain ∈ {chat, intent, reply, memory, boundary, proactive, reminder, llm, ws, http, scheduler}
- 加事件: 只改这一处. 调用方 import 常量, 不写裸字符串.
- 字段命名:
  * `*_id` / `*_name`: ID / 显示名 (uuid 类全量记录)
  * `*_preview`: 截断后预览 (e.g. user_message_preview, max 40)
  * `*_len`: 长度但不存内容 (e.g. reply_text_len)
  * 数值: latency_ms / count / score (snake_case)
- 红线: 绝不在 extra 塞 user_message 全文 / API key / 完整 LLM response body
"""

from __future__ import annotations

# Chat / Intent
EVT_INTENT_DETECTED = "intent.detected"
EVT_INTENT_SPLIT = "intent.split"
EVT_INTENT_SHORT_CIRCUIT = "intent.short_circuit"
EVT_CHAT_CRISIS_DETECTED = "chat.crisis_detected"  # 自伤/极端念头关键字命中, 走主回复 + crisis hint

# Reply
EVT_REPLY_TIER = "reply.tier_path"
EVT_REPLY_LLM = "reply.llm_main"
EVT_REPLY_SPLIT = "reply.split"
EVT_REPLY_EMOTION = "reply.emotion_detected"
EVT_REPLY_EMITTED = "reply.emitted"

# Memory
EVT_MEMORY_RELEVANCE = "memory.relevance_classified"
EVT_MEMORY_RETRIEVED = "memory.retrieved"
EVT_MEMORY_L3_AWAKEN = "memory.l3_awaken"
EVT_MEMORY_STORED = "memory.stored"
EVT_MEMORY_DEDUP_HIT = "memory.dedup_hit"
EVT_MEMORY_CONTRADICTION = "memory.contradiction_detected"
EVT_MEMORY_CONTRADICTION_STEP = "memory.contradiction_step"  # 5 步状态机每步
EVT_MEMORY_DELETED = "memory.deleted"
EVT_MEMORY_DELETION_PENDING = "memory.deletion_pending"
EVT_MEMORY_EXTRACTED = "memory.extracted"  # LLM 抽出 N 条记忆
EVT_MEMORY_L2_ADJUSTED = "memory.l2_adjusted"  # cron 完成统计
# Phase 0.4: embedding 链路可观测性 (区分 transient retry vs 终极失败 vs 孤儿 row)
EVT_EMBEDDING_RETRY = "memory.embedding_retry"  # 单次重试 (transient Ollama/PG hiccup)
EVT_EMBEDDING_FAIL = "memory.embedding_fail"  # 重试用尽, 用户记忆丢失
EVT_MEMORY_ORPHAN = "memory.orphan_row"  # memory 已写但 embedding 失败 + rollback 也失败

# Boundary
EVT_BOUNDARY_PATIENCE = "boundary.patience_delta"
EVT_BOUNDARY_BLOCKED = "boundary.blocked"
EVT_BOUNDARY_APOLOGY = "boundary.apology_handled"
EVT_BOUNDARY_BANNED_KW = "boundary.banned_keyword_hit"
EVT_BOUNDARY_ATTACK = "boundary.attack_recorded"
EVT_BOUNDARY_ZONE = "boundary.zone_classified"
EVT_BOUNDARY_VIOLATION_FAIL = "boundary.violation_process_failed"

# Proactive
EVT_PROACTIVE_WINDOW = "proactive.window_evaluated"
EVT_PROACTIVE_SENT = "proactive.sent"
EVT_PROACTIVE_SKIPPED = "proactive.skipped"
EVT_PROACTIVE_DECAY = "proactive.decay_advanced"
EVT_PROACTIVE_FIRST_GREETING = "proactive.first_greeting"
EVT_SPECIAL_DATE_TRIGGERED = "proactive.special_date_triggered"

# Reminder (Part 5)
EVT_REMINDER_HANDLED = "reminder.handled"
EVT_REMINDER_PRECHECK = "reminder.precheck"
EVT_REMINDER_RENEWED = "reminder.renewed"
EVT_REMINDER_RESCHEDULED = "reminder.rescheduled"
EVT_REMINDER_DLQ = "reminder.dead_letter"

# Infra
EVT_LLM_CALL = "llm.call"
EVT_LLM_FAIL = "llm.fail"
EVT_LLM_FALLBACK = "llm.fallback_to_ollama"
EVT_DB_LOOKUP_FAIL = "db.lookup_failed"  # 通用 infra: 用于 enrichment 路径 DB lookup 失败
EVT_WS_CONNECT = "ws.connect"
EVT_WS_DISCONNECT = "ws.disconnect"
EVT_WS_MESSAGE_RECV = "ws.message_received"
EVT_HTTP_REQUEST = "http.request"
EVT_SCHEDULER_JOB = "scheduler.job_executed"

# Aggregation (碎片聚合)
EVT_AGG_PUSHED = "aggregation.fragment_pushed"
EVT_AGG_FLUSHED = "aggregation.flushed"
EVT_AGG_SCAN = "aggregation.scan_expired"

# Background tasks (post_process 5 件)
EVT_BG_DONE = "bg.task_done"  # 5 个 _bg_* 通用 — kind 区分

# Preflight (跨消息 pending state 解析)
EVT_PREFLIGHT_RESOLVED = "preflight.resolved"  # contradiction / deletion / set_reminder
EVT_PREFLIGHT_FAILED = "preflight.failed"

# Multi-intent
EVT_INTENT_SUB_RECURSED = "intent.sub_recursed"  # sub fragment 递归入口

# Reply post-process
EVT_REPLY_DECORATION = "reply.decoration"  # emoji / sticker / none

# Proactive (扩展)
EVT_PROACTIVE_DEFERRED = "proactive.deferred"  # mutex defer (recent_user / topic_fatigue / etc)
EVT_PROACTIVE_FALLBACK = "proactive.trigger_fallback"  # scene→greeting / source→greeting

# State machine transitions
EVT_PROACTIVE_STATE_TRANSITION = "proactive.state_transition"  # session start, escalate n→n+1, cycle restart, user_replied reset
