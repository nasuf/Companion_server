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

## P0. Agent Eval 质量门禁

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

## P0. 后台任务企业化

**执行状态**：已完成第一版跨实例防重保护 + Redis runtime job queue。生产环境下 scheduler 高风险 cron 统一走单实例执行，memory pipeline 在生产环境按 conversation 增加跨实例锁；新增 runtime job queue（queued/running/succeeded/dead-letter 状态、retry、stale running recovery、DLQ），并将 agent 初始化从裸 `asyncio.create_task` 迁入队列。更完整的后台任务产品化（可视化任务状态、更多 job 类型迁移、人工重放 DLQ）进入 P1/P1.5。

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

## P0. 安全与隐私基线

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

## P1. 记忆治理升级

**执行状态**：已完成第一版 schema-stable 质量信号层。新增记忆 evidence changelog、`include_quality=true` 派生视图（confidence / evidence_message_ids / last_verified_at / contradiction_state / user_corrected_count / access_count），并把质量因子接入 L2 动态分数。尚未做数据库字段迁移、长期 consolidation 物化表或人工修复队列 UI。

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

## P1. CI/CD 质量流水线

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

## P1. 观测与运营指标产品化

**执行状态**：已完成第一版运营健康 endpoint + bug report → eval case 闭环，并完成 P1.5 LLM runtime metrics 落库。新增 `/admin-api/stats/operations`，聚合 `memory_changelogs`、`llm_usage`、`proactive_event_logs`、`proactive_states`、`time_triggers`、`bug_reports` 与 Redis DLQ/queue 计数，覆盖记忆写入/召回、LLM 用量、LLM latency/fallback/circuit、主动交流、提醒、runtime jobs、人工 bug report 的基础健康视图。新增 `/admin-api/bug-reports/{report_id}/eval-case`，可从人工标注的问题回复生成 validated JSONL eval draft，显式 `append_to_cases=true` 时才写入 `evals/cases.jsonl`。当前 visible use rate 与 crisis count 仍未独立结构化落库，后续需要打通 injected-vs-replied evidence 与 crisis 事件持久化。

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
   - visible use rate（第一版暂未落库；需打通 injected vs replied evidence）
   - reminder DLQ（第一版：Redis `reminder:dlq` 计数）
   - proactive sent/skipped（第一版：`proactive_event_logs` 聚合）
   - crisis count（第一版暂未落库；当前只有结构化日志事件）
2. P0 事件告警：
   - memory pipeline failed
   - embedding orphan
   - reminder DLQ
   - scheduler repeated failure
   - LLM circuit open
3. bug report 一键生成 eval case。（第一版：生成 deterministic JSONL draft，可显式 append）

### 验收标准

- 线上问题可在 5 分钟内定位到模型、检索、后台任务、DB/Redis 或 prompt。

## P2. 长期陪伴体验策略

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

## P2. Prompt 运营闭环

### 源码依据

- `app/services/prompting/store.py` 支持 prompt template/version/cache。
- `app/api/admin/prompts.py` 支持管理 prompt。
- trace enrich 能映射 prompt 组件。

### 要做

1. 每次 prompt 改动绑定 eval run 结果。
2. 支持 prompt canary：按 agent 或小流量启用。
3. admin bug report 分类聚合。
4. 后台提供最近 24h 高风险 trace 列表。

### 验收标准

- prompt 不靠感觉上线，每次变更有 before/after 质量报告。

## 当前执行顺序

1. P0 Agent Eval 质量门禁。
2. P0 安全与隐私基线。
3. P0 后台任务企业化。
4. P1 记忆治理升级。
5. P1 CI/CD 质量流水线。
6. P1 观测与运营指标产品化。
7. P2 长期陪伴体验策略。
8. P2 Prompt 运营闭环。
