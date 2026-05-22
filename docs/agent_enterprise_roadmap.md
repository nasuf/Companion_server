# AI Companion Backend Enterprise Roadmap

本文档基于 `Companion_server` 当前源码梳理后续任务，不以产品手册或架构说明为唯一依据。目标是把现有“复杂可运行的长期陪伴 agent”推进到“可评测、可运维、可扩展、可审计”的企业级后端。

## 源码结论

当前后端已经具备完整 agent runtime 雏形：

- 聊天编排：`app/services/chat/orchestrator.py`
- 记忆录入：`app/services/memory/recording/pipeline.py`
- 记忆检索：`app/services/memory/retrieval/hybrid.py`
- 记忆写入与调和：`app/services/memory/storage/persistence.py`
- 记忆治理：`app/services/memory/lifecycle/hygiene.py`
- 后台善后：`app/services/chat/post_process.py`
- 主动交流与提醒：`jobs/scheduler.py`, `app/services/proactive/`, `app/services/reminder/`
- 观测与 trace：`app/services/memory/retrieval/trace.py`, `app/services/chat/tracing.py`, `app/services/chat/trace_mirror.py`
- 管理反馈：`app/api/admin/bug_reports.py`
- 部署：`.github/workflows/deploy.yml`, `Dockerfile`, `docker-compose.deploy.yml`

主要短板不是“缺模块”，而是质量门禁、跨实例可靠性、安全隐私基线、长期记忆治理和运营闭环还不够硬。

## 状态标记

- `[完成-第一版]`：核心能力已经落地、已有基础测试或可运行验证，但后续仍可能需要生产化增强。
- `[未完成]`：仍缺关键闭环，不能视为企业级完成。
- `[下一阶段]`：建议作为下一轮架构优化优先推进。

## P0. Agent Eval 质量门禁 `[完成-第一版]`

**执行状态**：已完成第一版落地。新增 `evals/` 本地 runner、versioned cases、deterministic graders、CI smoke workflow 和 `tests/test_agent_evals.py`。离线校验与测试已通过；真实 server mode 后续可继续扩充到多轮端到端回归。

### 源码依据

- `tests/` 单测数量很多，但主要是模块级和 mock 边界测试。
- `tests/test_life_story_pipeline.py` 明确跳过 `generate_l1_coverage` 端到端测试，因为需要真实 LLM + Redis。
- `scripts/agent_dialogue_test_runner.py` 有长对话脚本和红旗检查，但不是版本化 eval，也没有接入 CI。
- `.github/workflows/deploy.yml` 目前偏部署流程，没有先执行 eval gate。
- 记忆错误曾经真实发生过，尤其是 unsupported memory hallucination，应被固化为回归样例。

### 要做

1. 建立 `evals/` 目录，包含 versioned cases、grader、local runner、README。
2. 固化高风险 case：
   - unsupported memory hallucination
   - memory recall with evidence
   - L1 contradiction and correction
   - deletion / undo
   - reminder create / reschedule / cancel
   - crisis and aftercare
   - boundary and apology
   - proactive appropriateness
   - human-like tone consistency
3. 支持两种运行方式：
   - `--validate-only`: 离线校验 eval case schema 与 grader 逻辑，适合 CI。
   - server mode: 连接本地后端，发送真实对话，基于回复文本和 metadata 打分。
4. 把现有 `reports/agent_dialogue_test_*.jsonl` 与 admin bug report 转化为 regression cases。
5. 在 CI 中先跑 smoke eval，再允许部署。

### 验收标准

- 没有模型和数据库时，CI 仍可运行 eval schema/grade smoke。
- 本地后端启动后，可以跑真实对话 eval 并输出 JSON 报告。
- 编造记忆、错用记忆、危机回复失守、提醒不触发、人格口吻泄漏会被自动标红。

## P0. 后台任务企业化 `[完成-第一版]`

**执行状态**：已完成第一版跨实例防重保护 + Redis runtime job queue。生产环境下 scheduler 高风险 cron 统一走单实例执行，memory pipeline 在生产环境按 conversation 增加跨实例锁；新增 runtime job queue（queued/running/succeeded/dead-letter 状态、retry、stale running recovery、DLQ），并将 agent 初始化从裸 `asyncio.create_task` 迁入队列。更完整的后台任务产品化（可视化任务状态、更多 job 类型迁移、人工重放 DLQ）仍未完成，进入 P3。

### 源码依据

- `app/main.py` 在进程启动时直接 `setup_scheduler()`。
- `jobs/scheduler.py` 使用进程内 APScheduler。
- `app/services/chat/post_process.py` 用进程内 `dict[str, asyncio.Lock]` 做 conversation 级 memory pipeline 锁。
- `app/api/public/agents.py` 在创建 agent 后用 `asyncio.create_task` 跑长生命周期任务。
- `docker-compose.deploy.yml` 当前只有一个 `server` 服务；如果未来横向扩容，scheduler 和 in-memory lock 会重复或失效。

### 要做

1. 引入 durable job queue 或至少抽象任务层。
2. 将以下路径迁出进程内 fire-and-forget：
   - agent 初始化
   - memory pipeline
   - reminder trigger emit
   - proactive scan/send
   - daily schedule / weekly portrait / memory hygiene
3. 加跨实例锁：
   - `conversation_id + memory side`
   - `agent_id + generation`
   - `trigger_id`
   - scheduler job id
4. 所有任务具备 retry、dead-letter、状态查询和幂等键。

### 验收标准

- 两个 server 实例同时运行时，不重复发提醒、不重复抽记忆、不重复主动消息。
- 任一 worker crash 后，未完成任务可恢复或进入 DLQ。

## P0. 安全与隐私基线 `[完成-第一版]`

**执行状态**：已完成第一版 production fail-fast + auth abuse control + memory privacy API。新增 `APP_ENV`、`CORS_ALLOWED_ORIGINS`、生产强校验、部署前必需项检查、登录/注册 rate limit 与 auth audit、记忆 export/edit/bulk delete/workspace wipe，以及对应安全回归测试。注意：生产部署前必须在 GitHub variables 配置 `CORS_ALLOWED_ORIGINS`，否则会按预期阻断部署。旧 BasicAuth env 已确认不再挂载使用并从部署文档/脚本环境读取中删除。

### 源码依据

- `app/main.py` 当前 CORS 是 `allow_origins=["*"]` 且 `allow_credentials=True`。
- `app/config.py` 的 `jwt_secret` 默认是空字符串。
- `app/services/auth.py` 直接用配置值签发 JWT。
- `app/api/public/memories.py` 已有 list/search/get/hygiene，但缺完整导出、编辑、批量删除、隐私审计。
- `app/api/public/traces.py` 有 trace resolve，但 trace URL/detail 的访问边界需要持续守住。

### 要做

1. 生产启动时强校验：
   - `JWT_SECRET` 不可为空或默认弱值
   - CORS 必须使用 allowlist
2. 给 `/auth/login`、`/auth/register` 加 rate limit、失败计数和审计日志。
3. 用户记忆管理补齐：
   - export
   - edit
   - bulk delete
   - workspace wipe
   - deletion audit
4. 给 trace、memory、admin endpoints 增加安全回归测试。

### 验收标准

- 生产环境不能用空 secret 启动。
- 用户可以完整查看、导出、删除自己的长期记忆。
- 跨用户 trace/memory 访问有自动测试覆盖。

## P1. 记忆治理升级 `[完成-第一版 / P3继续]`

**执行状态**：已完成第一版 schema-stable 质量信号层。新增记忆 evidence changelog、`include_quality=true` 派生视图（confidence / evidence_message_ids / last_verified_at / contradiction_state / user_corrected_count / access_count），并把质量因子接入 L2 动态分数。尚未做数据库字段迁移、长期 consolidation 物化表或人工修复队列 UI，因此“记忆可信度闭环”仍未完成，是 P3 的最高优先级。

### 源码依据

- `app/services/memory/recording/pipeline.py` 已有 filter、pre-filter、extraction、time parse、reminder validation。
- `app/services/memory/storage/persistence.py` 已有 taxonomy、singleton、reconciliation、embedding rollback。
- `app/services/memory/lifecycle/l2_dynamics.py` 只有 L2 动态升降级。
- `app/services/memory/lifecycle/hygiene.py` 只有 bounded duplicate cleanup 与 fact evolution，还不是完整长期 consolidation。

### 要做

1. 给记忆增加或派生：
   - confidence
   - evidence message ids
   - last_verified_at
   - contradiction_state
   - user_corrected_count
2. 把“用户纠正 AI 记错”从 trace signal 推进到待修复队列。
3. 做 memory consolidation：
   - 多条碎片事实合成稳定画像
   - 保留来源证据
   - 合并后更新 embedding
4. L2/L3 调整加入质量因素：
   - 用户纠正降权
   - 长期未验证降权
   - 被稳定使用且未纠错升权

### 验收标准

- 1000+ 条记忆下检索仍准确。
- AI 引用个人事实时能追溯到来源消息或明确表达不确定。

## P1. CI/CD 质量流水线 `[完成-第一版 / P3继续]`

**执行状态**：已完成第一版 CI gate。`ci.yml` 现在执行依赖安装、Prisma validate/generate、Python compileall、eval validate 和 backend quality pytest；`deploy.yml` 改为 CI 成功后的 `workflow_run` 触发，并保留 `workflow_dispatch` 手动部署入口。后续仍可扩展 lint/type check、部署后 smoke chat。

### 源码依据

- `.github/workflows/deploy.yml` 当前 push/PR 都进入部署 job。
- `Dockerfile` 只构建运行镜像，不执行质量检查。
- 未看到 PR 阶段的 pytest、Prisma validate、lint 或 eval gate。

### 要做

1. 拆分 `ci.yml` 与 `deploy.yml`。
2. PR 执行：
   - install
   - prisma validate/generate
   - targeted pytest
   - eval validate
   - optional lint/type check
3. main 分支：
   - CI 通过后部署
   - migrate deploy
   - health check
   - smoke chat

### 验收标准

- PR 不通过测试/eval 不允许部署。
- 部署后自动验证 `/health` 和最小聊天链路。

## P1. 观测与运营指标产品化 `[完成-第一版 / P3继续]`

**执行状态**：已完成第一版运营健康 endpoint + bug report → eval case 闭环，并完成 P1.5 LLM runtime metrics 落库。新增 `/admin-api/stats/operations`，聚合 `memory_changelogs`、`llm_usage`、`proactive_event_logs`、`proactive_states`、`time_triggers`、`bug_reports`、`memory_visible_use_events`、`crisis_events` 与 Redis DLQ/queue 计数，覆盖记忆写入/召回、可见使用率、危机事件、LLM 用量、LLM latency/fallback/circuit、主动交流、提醒、runtime jobs、人工 bug report 的基础健康视图。新增 `/admin-api/bug-reports/{report_id}/eval-case`，可从人工标注的问题回复生成 validated JSONL eval draft，显式 `append_to_cases=true` 时才写入 `evals/cases.jsonl`。

### 源码依据

- `app/observability/events.py` 已有结构化 event 常量。
- `app/services/memory/retrieval/trace.py` 已能分析 retrieval quality。
- `app/services/chat/trace_mirror.py` 能本地镜像 trace。
- `app/api/admin/bug_reports.py` 有人工 bug report。

### 要做

1. Dashboard 指标：
   - LLM latency
   - fallback rate
   - circuit breaker open count
   - memory extraction success（第一版：`memory_changelogs.operation='insert'`）
   - retrieval selected count（第一版：`memory_changelogs.operation='access'`）
   - visible use rate（P3 已落库：`memory_visible_use_events`）
   - reminder DLQ（第一版：Redis `reminder:dlq` 计数）
   - proactive sent/skipped（第一版：`proactive_event_logs` 聚合）
   - crisis count（P3 已落库：`crisis_events`）
2. P0 事件告警：
   - memory pipeline failed
   - embedding orphan
   - reminder DLQ
   - scheduler repeated failure
   - LLM circuit open
3. bug report 一键生成 eval case。（第一版：生成 deterministic JSONL draft，可显式 append）

### 验收标准

- 线上问题可在 5 分钟内定位到模型、检索、后台任务、DB/Redis 或 prompt。

## P2. 长期陪伴体验策略 `[完成-第一版]`

**执行状态**：已完成第一版长期陪伴 eval 基线、主动交流 fatigue score、用户级节奏学习和 30 天模拟评测 harness。`evals/run_local.py` 现在按 turn 顺序发送并等待每轮 assistant 回复，支持 `grade_target=last_reply`，可验证多轮承接而不是只看最终批量回复。`evals/cases.jsonl` 新增 P2 专项 case：关系/长期目标追踪、低落陪伴非机械安慰、睡前降速、高频确认不重复。主动发送前除固定日上限外，会基于近 24/72h 主动消息、reply timeout、跳过/延迟事件计算用户级疲劳分，高于阈值则跳过并写入 `send_skipped(reason=fatigue_score)`。正常概率窗口会根据近 30 天同本地小时的发送、回复、reply timeout、疲劳跳过事件得到 `rate_multiplier`，保守调整 `should_hit_window()` 的 final_rate。`evals/long_companion_sim.py` 可对 30 天 transcript 做 deterministic 检查，覆盖人格泄漏、机械安慰、目标连续性和主动过度。

### 源码依据

- `app/services/proactive/` 有主动交流状态机。
- `app/services/interaction/reply_context.py` 有异步回复节奏。
- `app/services/interaction/boundary.py` 有耐心值和边界恢复。
- `app/services/relationship/` 有亲密度、情绪与一致性相关逻辑。

### 要做

1. 主动交流引入用户级节奏学习。
2. 增加 relationship consistency eval。
3. 对睡前、低落、高频确认、长期目标追踪建立专项 case。
4. 主动消息使用 fatigue score，而不只靠固定次数。

### 验收标准

- 30 天模拟对话中不突然换人格、不机械安慰、不失忆、不过度主动。

## P2. Prompt 运营闭环 `[完成-第一版]`

**执行状态**：已完成第一版 Prompt 运营闭环。`/admin-api/stats/operations` 现在除基础 bug report 状态外，还返回 `by_error_type`、`by_eval_category`，并固定给出最近 24 小时 `high_risk_traces` 列表；风险条件覆盖 trace 总耗时 ≥20s、LLM step ≥8、trace share 失败和未解决人工标注。Web 端运营健康面板同步展示“问题分类”和“高风险 Trace”。Prompt 保存、重置、回退、代码同步会在 `prompt_template_versions.eval_result` 绑定离线 eval 校验快照；`prompt_templates.canary_config` 支持按 agent allowlist 或稳定百分比流量启用 canary prompt，Web 提示词编辑器可查看/保存 canary 配置与 eval 结果。

### 源码依据

- `app/services/prompting/store.py` 支持 prompt template/version/cache。
- `app/api/admin/prompts.py` 支持管理 prompt。
- trace enrich 能映射 prompt 组件。
- `/admin-api/stats/operations` 已聚合人工 bug 分类与最近 24h 高风险 trace。
- `prompt_template_versions.eval_result` 与 `prompt_templates.canary_config` 已承载变更校验与 canary 配置。

### 要做

1. 每次 prompt 改动绑定 eval run 结果。（第一版：离线 validate-only 快照）
2. 支持 prompt canary：按 agent 或小流量启用。（第一版：agent allowlist / 稳定百分比）
3. admin bug report 分类聚合。（第一版已接入 operations stats）
4. 后台提供最近 24h 高风险 trace 列表。（第一版已接入 operations stats + Web 面板）

### 验收标准

- prompt 不靠感觉上线，每次变更有 before/after 质量报告。

## 当前完成情况

| 阶段 | 状态 | 说明 |
|------|------|------|
| P0 Agent Eval 质量门禁 | `[完成-第一版]` | 已有 versioned cases、deterministic grader、CI validate；真实 server mode 多轮回归仍可增强。 |
| P0 安全与隐私基线 | `[完成-第一版]` | 已有生产 fail-fast、CORS allowlist、auth abuse control、memory privacy API；生产环境变量仍需部署时严格配置。 |
| P0 后台任务企业化 | `[完成-第一版 / P3继续]` | 已有 runtime queue、retry、DLQ、跨实例防重、任务状态可视化和 DLQ 人工重放；更多 job 类型迁移仍可继续。 |
| P1 记忆治理升级 | `[完成-第一版 / P3继续]` | 已有质量信号派生视图和 L2 质量因子；长期 consolidation、物化字段、人工修复队列未完成。 |
| P1 CI/CD 质量流水线 | `[完成-第一版 / P3继续]` | 已有 CI gate 与 deploy workflow_run；lint/type check、部署后 smoke chat、真实 server eval 未完成。 |
| P1 观测与运营指标产品化 | `[完成-第一版 / P3继续]` | 已有 operations stats、LLM runtime metrics、bug report eval draft、visible use rate 和 crisis count 结构化落库；告警闭环仍未完成。 |
| P2 长期陪伴体验策略 | `[完成-第一版]` | 已有 fatigue score、节奏学习、长期陪伴 eval、30 天模拟 harness；后续可继续扩大真实长周期样本。 |
| P2 Prompt 运营闭环 | `[完成-第一版]` | 已有 eval snapshot、canary config、Web prompt 管理和高风险 trace 面板；before/after 真实质量报告仍可增强。 |

## P3. 生产化闭环与长期记忆可信度 `[下一阶段 / 未完成]`

P3 的目标是把 P0-P2 的“第一版能力”推进成可长期运营的闭环：线上能发现问题、定位问题、修复问题，并且让长期记忆在数千条规模下仍可追溯、可合并、可纠错。

### P3-1. 记忆 consolidation + 人工修复队列 `[完成-运营闭环第一版]`

**为什么优先**：长期陪伴型 agent 的核心资产是记忆。当前系统能派生质量信号，但还没有把“低置信 / 被纠错 / 互相冲突 / 碎片重复”的记忆推进到稳定画像或人工修复流。

**执行状态**：已完成 repair queue MVP 闭环、人工修复动作、质量状态物化和自动 consolidation 审计。后端新增 `memory_repair_items`、`memory_quality_states`、`memory_consolidation_runs` 旁路表；接入 `bug_report_memory_safety`、`retrieval_feedback_unresolved`、`contradiction_*` 写入来源。Web admin 已新增“记忆修复”入口，可按状态/来源查看候选、查看证据 JSON，并执行标记已解决/忽略/重新打开；证据面板已支持归档、降级、编辑、插入替代记忆、标记已验证、合并记忆。涉及内容变化的动作会重建 embedding，所有动作写入 `memory_changelogs` 并自动关闭 repair item；repair/consolidation/changelog 会刷新 `memory_quality_states`，consolidation run 保留合并/归档 evidence。当前仍未完成独立细粒度操作审计表，第一版先复用 changelog + consolidation run。

**要做**：

1. 设计记忆质量物化字段或旁路表，承载 confidence、evidence、verified、contradiction、correction 等长期状态。`[x]`
2. 建立 memory repair queue：
   - 用户纠错过的记忆。
   - 被 trace 标记为 unsupported / wrong recall 的记忆。
   - evidence 缺失或冲突的高重要度记忆。
3. 做 consolidation job：`[x]`
   - 将多条碎片事实合并成稳定画像。
   - 保留来源 memory/message/changelog evidence。
   - 更新 embedding 与 changelog。
4. Web admin 增加修复入口：
   - [x] 查看候选。
   - [x] 查看证据与上下文。
   - [x] 标记已解决 / 忽略 / 重新打开。
   - [x] 合并 / 降级 / archive / 标记已验证。
   - [x] 编辑记忆 / 插入替代记忆。
   - [x] 通过 `memory_changelogs` 记录 repair action、admin、repair item、原因和 before/after。
   - [ ] 独立细粒度操作审计表。

**验收标准**：

- AI 引用 L1/L2 个人事实时能追溯 evidence。
- 用户纠错后，相关记忆会进入 repair queue 或自动降权。
- 1000+ 条记忆下，重复碎片不会持续挤占检索上下文。

### P3-2. visible use rate + crisis count 结构化落库 `[完成-第一版]`

**为什么优先**：现在能看到“记忆被注入/访问”，但还不能稳定衡量“最终回复有没有真的使用这些记忆”；危机事件也还没有独立结构化统计。

**执行状态**：已新增 `memory_visible_use_events` 与 `crisis_events`。聊天回复保存首条 assistant message 时，会从现有 `memory_retrieval_analysis` / `response_diagnostics` metadata 中 best-effort 落库，不增加额外 LLM 调用。Operations 面板已展示平均可见使用率、可见使用条数、未支撑引用数与危机事件数量/严重度。

**要做**：

1. 在回复生成后记录 injected memory 与 replied evidence 的匹配结果。`[x]`
2. 新增 visible use rate 聚合：`[x]`
   - injected_count。
   - visibly_used_count。
   - unsupported_reference_count。
   - by prompt / agent / user / time window。
3. 危机事件结构化落库：`[x]`
   - crisis detected。
   - crisis category / severity。
   - handler path。
   - aftercare 是否触发。
   - 人工标注结果。
4. Web operations / trace 面板展示：`[x]`
   - 记忆可见使用率。
   - 危机事件趋势。
   - 高风险 agent / conversation。

**验收标准**：

- 可以回答“召回的记忆有多少真正进入用户可见回复”。
- 可以按天/agent/user 统计危机事件，并追溯到 trace 与回复。

### P3-3. Runtime job admin 面板 + DLQ 重放 `[完成-第一版]`

**为什么优先**：后台任务已经有 queue / retry / DLQ，但生产运营需要可视化和人工恢复能力，否则问题还是只能查日志和 Redis。

**执行状态**：已新增 `/admin-api/runtime-jobs` 查询、inspect、retry、resolve API，并在 Web admin 增加“任务队列”面板，可查看 queued/delayed/running/dead-letter 任务、payload、错误原因，并执行重放或标记解决。第一版仍以 Redis job queue 为 source of truth，后续如需要跨重启长周期审计，再增加 DB-backed runtime_jobs 表。

**要做**：

1. 后端增加 runtime job 查询 API：`[x]`
   - queued / running / succeeded / failed / dead-letter。
   - by job type。
2. 增加 DLQ 操作：`[x]`
   - inspect。
   - retry once。
   - retry batch。
   - mark resolved。
3. Web admin 增加任务面板：`[x]`
   - job 状态列表。
   - 失败原因。
   - retry 按钮。
   - stale running 提示。
4. 将更多后台路径迁入 runtime queue：`[部分完成]`
   - memory pipeline。
   - reminder trigger emit。
   - proactive scan/send。
   - daily schedule / portrait / hygiene。

**验收标准**：

- worker crash 后能在后台看到失败任务并重放。
- 多实例运行时不重复发提醒、不重复主动消息、不重复初始化。

### P3-4. 部署后 smoke + server mode eval `[未完成]`

**要做**：

1. 部署后自动检查 `/health`。
2. 执行最小真实 chat smoke：
   - 创建/复用测试 agent。
   - 发送一轮消息。
   - 验证 assistant reply、message persistence、trace metadata。
3. 扩展 `evals/run_local.py` server mode 到 CI 手动触发或 nightly。
4. 失败时阻断自动推广或标记部署风险。

**验收标准**：

- 不是“部署成功”就算成功，而是最小对话链路真的可用。

### P3-5. Prompt before/after 质量报告 + canary 效果归因 `[未完成]`

**要做**：

1. prompt 保存时运行 before/after 对比 eval，而不只是 validate-only。
2. canary 分组记录实际命中情况与质量指标：
   - bug report rate。
   - high-risk trace rate。
   - latency / fallback。
   - visible use rate。
3. Web 端展示 canary 当前效果。
4. 支持一键回滚 canary。

**验收标准**：

- prompt 变更能看到质量变化，不靠感觉上线。
- canary 变差时能快速定位和回滚。

### P3-6. 国内生产部署 playbook `[未完成]`

**要做**：

1. 腾讯云 CVM 初始化脚本：
   - 数据盘挂载。
   - Docker / Nginx / certbot。
   - firewall / security group checklist。
2. Supabase → 国内 Postgres 迁移流程：
   - schema。
   - data dump / restore。
   - pgvector extension。
   - smoke check。
3. 备份恢复演练：
   - daily `pg_dump`。
   - COS 归档。
   - 恢复到临时库验证。
4. 环境变量清单：
   - required。
   - deprecated。
   - secret rotation。
5. 回滚方案：
   - app rollback。
   - migration rollback policy。
   - DB snapshot restore boundary。

**验收标准**：

- 新机器可以按 playbook 从 0 部署到可用。
- 备份不是只“生成了”，而是真的能恢复。
