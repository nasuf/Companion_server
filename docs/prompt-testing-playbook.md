# 提示词测试手册（运维版）

> 目标：验证任意一个提示词在后台管理中的修改（编辑 / 停用 / canary）确实生效。
> 全库 144 个 prompt key 的守卫测试（`test_trace_prompt_key_coverage_matches_runtime_surfaces`）
> 保证每个 key 必属于以下三类之一：独立 LLM 步骤（100）/ 组合组件段（43）/ 非运行时保留（1）。

## 一、三条测试通道

| 通道 | 适用范围 | 入口 |
|---|---|---|
| **① Trace 面板** | 聊天热路径 + 后台管线（post 步骤）+ 主动消息 | 聊天消息 / 主动消息上的 Trace 按钮 → 步骤详情内可编辑该步 prompt 并重放 |
| **② Replay 端点** | **全部 144 个 key**（含 trace 覆盖不到的） | `POST /admin-api/prompts/replay`，传 prompt_key + 参数即试跑；支持复放原消息的完整 prompt 栈 |
| **③ Axiom 日志** | 验证行为开关类（非文案）改动 | app.axiom.co，按 event 字段过滤 |

**核心结论：不是所有 prompt 都出现在 trace 面板（见第四节缺口清单），但所有 prompt 都可通过 ② 测试。**
Trace 面板是"真实场景端到端验证"，Replay 是"单点文案验证"——建议改动先用 ② 快验，再用 ① 场景回归。

## 二、Trace 面板可测的 key 及触发方法

### 2.1 每条消息必然出现（发任意一条聊天消息即可）
- 主 prompt 组件（trace 内主回复步骤展开可编辑）：`chat.system_base` / `chat.anti_hallucination_hard_rule` / `chat.personality_section`（含 style_base_rule / style_closing_rule）/ `chat.consistency_rules` / `chat.response_instruction` / `chat.time_context_section` / 记忆段（`chat.memory_section_body` + 6 个 memory_label_*，需检索命中；无命中时 `chat.memory_empty_anchor`）
- 独立步骤：`intent.unified`（消息 >4 字）、`memory.relevance`
- post 步骤（**延迟出现**，后台任务完成后刷新 trace）：`memory.extraction_user` / `memory.extraction_ai` / `memory.judgement_user` / `memory.judgement_ai`（预筛）/ `emotion.user_label`
- `reply.emotion_detection`：仅当主 LLM 的 [EMO] 标记缺失/失效时回退出现（正常时 trace 里看不到 = 省了这次调用，属预期）

### 2.2 条件段 / 条件步骤（按场景触发）

| key | 触发方法 |
|---|---|
| `chat.reengagement_short/long/day` | 距上条消息 30min-3h / 3h-24h / >24h 后发消息。**测试环境加速**：SQL 把该会话最后一条消息 `created_at` 改老即可 |
| `chat.session_recap` + `chat.session_recap_section` | 同上，gap ≥3h（同一次重逢只调一次生成，第二轮起走缓存段） |
| `chat.ai_mood_section` | 先聊出明显情绪（如报喜讯），30min 内再发下一条 |
| `chat.expression_habits_section` / `expression.learn_style` | 同会话累计 20 条用户消息触发学习（post 步骤）；之后每轮注入习惯段 |
| `chat.relationship_stage_section`（含关系时长行） | 每轮出现（有 intimacy_stage 时） |
| `chat.topic_context_section` | 同话题连聊 2 轮以上 |
| `chat.delay_context_section` / `reply.delay_reason_*` | 需 `REPLY_DELAY_ENABLED=true`（当前产品决策默认关，测试环境可开） |
| `reply.delay_explanation` | 延迟 ≥1min 且概率命中（35%-85% 按时长）；测试环境可重发几次 |
| `chat.l3_memory_section` / `memory.l3_trigger` / `memory.l3_reply` | "你还记得很久以前我说过…" 句式 |
| `memory.weak/medium/strong_reply`（tier） | 纯闲聊 + 无情绪/延迟上下文 + gap <3h；trace 的 reply 步骤显示 tier 路径 |
| `intent.split` | 一句话带两个意图："别提醒我了，对了你明天有空吗" |
| `intent.current_state_reply` | "你在干嘛呢" |
| `intent.schedule_query_reply` / `intent.schedule_missing_context` | "你明天有空吗" |
| `intent.schedule_adjust_reply` | AI 问"要我再陪你会吗"后回"好" |
| `intent.end_reply` / `intent.conversation_end_fallback_instruction` | "先聊到这吧" |
| `intent.record_confirm_reply` / `intent.record_ask_time` | "提醒我喝水"（无时间→追问时间） |
| `memory.deletion_intent` / `intent.deletion_confirm` / `intent.deletion_reply` | "忘掉我说过的 X" → 确认 |
| `memory.contradiction_*`（4 个） | 先说"我住在苏州"，等记忆入库（trace 看 post 完成），再说"我住在北京" |
| `boundary.*`（13 个） | 攻击性话术分级触发（K1 轻讽/K2 辱骂/K3 severe）；`blacklist_reply` 需耐心打到 0；`apology*` 拉黑后道歉；`positive_interaction` 发感谢语 |
| `intent.crisis_*`（4 个） | 危机话术（测试环境专用账号操作） |
| `boundary.patience_instruction_*` | 耐心 <100 后正常聊天，主 prompt 内出现该段 |
| `chat.special_instruction_appendix` | 道别/延迟解释兜底路径（终结意图即可见） |
| `music.co_listening_context` | web 端发起共听后聊天 |
| `music.accept_invite` / `busy_reject` / `sleep_reject` / `switch_track` | web 端分享歌曲发起共听（AI 空闲/忙碌/睡眠状态下分别触发），回复消息带 Trace 按钮 |
| `music.busy_exit` / `agent_join_after_busy` / `agent_late_missed` 等共听事件 | 共听中暂停超时/AI 作息切换等事件，回复消息带 Trace 按钮 |
| `offline.gift_sent_message` / `gift_delivered_message` / `gift_thanks_reply` / `gift_first_address_request` / `activity_invite_message` | 礼物寄出/送达/感谢/首次要地址/活动邀请消息，均带 Trace 按钮（2026-07-04 已接 tracer） |

### 2.3 主动消息（主动消息本体带 Trace 按钮）
`proactive.silence_*`（4）/ `proactive.memory_ai|user` / `proactive.scheduled_scene` / `proactive.decay_final` / `proactive.first_greeting`（新会话首条）/ `proactive.special_*`（4，特殊日期）/ `proactive.reminder_message` + `proactive.reminder_pre_check`（设一个"2 分钟后提醒我X"最快）/ `proactive.memory_topic_rerank`。
触发加速：提醒类设短时提醒即到点；沉默唤醒类可等窗口或调 proactive_states 表。

## 三、只能用 Replay 端点测试的 key（trace 面板无载体）

| 组 | keys | 原因 |
|---|---|---|
| 初始化 | `character.generation` / `character.repair_missing_fields` / `agent.personality_scoring` | 建 agent 流程，无聊天消息 |
| 作息 cron | `schedule.daily_schedule(_with_memory)` / `schedule.daily_summary(_memories)` / `schedule.life_overview` | 每日定时任务，无消息载体 |
| 画像 | `portrait.generation` / `portrait.update` / `portrait.tags` | 后台画像任务 |
| L1 扫描 | `memory.pairwise_contradiction` / `memory.reconciliation`* | cron / 深层管线（*reconciliation 在聊天 post 链上理论可见，但依赖去重命中，replay 更可控） |
| 线下互动·非 LLM | `offline.gift_candidate_pick` / `gift_selection` / `activity_card`（后台挑选/生成类，无消息载体） | 后台任务，replay 验证 |
| 兜底文案 | `reply.delay_explanation_fallback_instruction` 等结构性兜底 | 仅主链路失败时出现，replay 验证文案即可 |

## 四、已知缺口与建议（工程侧后续项）

1. ~~offline 礼物/活动消息无 Trace 按钮~~（✅ 2026-07-04 已修：5 条 LLM 链包 offline_trace，消息带 trace_id）
2. ~~music WS 事件回复无 tracer~~（✅ 同日已修：共听邀请响应 + 事件回复漏斗均包 traced_usage_session）
3. 仍无消息载体的组（初始化/作息 cron/画像/L1 扫描/礼物挑选后台步骤）继续用 replay 端点。
4. 概率性行为（延迟解释概率、语气词仅表情、错别字）不是 prompt 文案问题，用 Axiom event 验证：`reply.filler_emoji` / `chat.session_recap_built` / `expression.learned` / `reply.decoration`。

## 五、测试环境快捷操作备忘

- 重逢/摘要场景加速：`UPDATE messages SET created_at = created_at - interval '5 hours' WHERE conversation_id = '...';`
- 停用验证：后台停用某 section key → 发消息 → trace 主 prompt 中该段整段消失（diagnostics 的 empty_prompt_sections_removed 可核对）
- prompt 修改的 Redis 生效延迟 ≤10s（进程内缓存）
- 测试后清理：表达学习 `DEL expression:{agent}:{user}`、心情 `DEL ai_mood:{conv}`、摘要 `DEL recap:{conv}:*`
