# 成就系统代码审计

审计基线：`/Users/songtao/Projects/companion/成就系统.xlsx`
审计日期：2026-07-11

## 结论

- Excel 与 `definitions.py` 的 97 项定义、名称、文案、等级和分值逐项一致。
- 规则注册表固定为 82 项启用、15 项停用。停用 ID：
  `3,4,10,11,12,13,14,15,16,17,22,24,32,34,40`。
- 停用项在 `unlock_achievement()` 统一入口被硬拒绝，不会被遗留调用误解锁。
- 所有字数规则统一只统计 Unicode 文字和数字；空白、标点、符号、emoji 均不计。
- 成就作用域统一为 `user_id + agent_id`；响应配对额外限制在同一 conversation。
- 解锁由数据库唯一键 `(user_id, agent_id, achievement_id)` 保证并发幂等。
- 事件由 `(user_id, agent_id, event_type, source_id)` 部分唯一索引防止重复累计。

## 运行模式（2026-07-20，H5 上线支持）

全局模式解析顺序：`SystemConfig.achievement_mode`（web 后台「系统设置」可
动态切换，写库后 ~10s 内全部实例生效，见 `app/services/achievements/mode.py`
的进程内 TTL 缓存）→ `.env ACHIEVEMENT_MODE` 默认值（`on`）：

| 模式 | 规则评估/解锁写库 | 日终任务 | WS/APNs 通知 | GET /achievements | 时间线成就行 | 钱包积分同步 |
|------|-------------------|----------|--------------|-------------------|--------------|--------------|
| `on` | 运行 | 运行 | 发送 | 完整数据 | 合成 | 同步 |
| `silent` | 运行（H5 静默计算） | 运行 | 抑制 | 完整数据 | 跳过 | 同步 |
| `off` | 停止 | 跳过且 checkpoint 冻结 | 抑制 | 隐藏 | 跳过 | 跳过 |

- `silent` 是 H5 纯聊天上线的推荐模式（2026-07-22 口径调整）：解锁行实时落库，
  `unlocked_at` 与 `conversation_id` 即为真实达成时刻/会话；成就页与钱包积分
  照常可见可用，仅静默「达成时刻」——聊天 WS 弹窗、聊天时间线成就行、APNs
  系统推送。切回 `on` 后聊天时间线历史位置自动出现，无需任何回填任务。
- 闸门分两类（`mode.py`）：`display`（成就页 API + 钱包积分，on/silent 开）
  与 `alerts`（聊天弹窗 + 时间线成就行 + 系统推送，仅 on 开）。
- 切回 `on` 不会补发历史通知：通知只在解锁当下发送，静默期解锁的
  `notified_at` 保持 NULL 且无补扫逻辑。
- `off` 为应急开关；恢复后日终 catch-up 依据冻结的 checkpoint 重放
  （上限 366 天），但实时/累计类在停用窗口内的达成时刻无法恢复。
- 代码闸门：`engine.handle_achievement_event`（评估）、
  `repository.unlock_achievement`（写库防御 + 通知侧）、
  `jobs/scheduler._run_achievement_daily_rollup`（日终）、
  `wallet.sync_achievement_points`（积分）、
  `api/public/achievements.get_achievements` 与
  `api/public/conversations.list_messages`（用户可见面）。
- 后台切换入口：`GET/PUT /admin-api/achievement-settings`
  （`app/api/admin/achievement_settings.py`），web 端在
  `Companion_web/src/OfflineSettingsWorkspace.tsx` 的「系统设置」面板与
  线下活动/礼物开关同列，三态即选即存。
- 行为测试：`tests/test_achievement_mode.py`（26 项）。

## 当前执行链路

1. 用户消息持久化后触发 `UserMessageAchievementEvent`。
2. AI 每个持久化气泡触发 `AssistantMessageAchievementEvent`，完整回合另触发
   `AssistantTurnAchievementEvent`。
3. 已解析意图触发 `IntentAchievementEvent`。
4. 用户记忆写入触发 `MemoryChangelogAchievementEvent`。
5. 碎片窗口实际完成且含至少两个碎片时触发 `AggregationAchievementEvent`。
6. 每日 00:05（Asia/Shanghai）运行前一自然日的完整日规则。
7. 所有解锁统一进入 repository，写 DB、更新 Redis 缓存、推送通知并进入时间线。

## 本轮修复

- #5：按完整 48 小时窗口检查，消息判重使用原文完全相同，不再忽略标点。
- #44：除 AI 字数严格大于用户三倍外，要求双方都发过消息且总消息数至少 20。
- #48：同一个跨午夜睡眠时段最多累计一次叫醒，防止连续发十条消息刷成就。
- #59：只统计持久化消息 metadata 中的 sticker，不再把普通 emoji 当表情包。
- #62：改为完整自然日判定；至少 20 条 AI 消息，且每条都在 10 秒内收到回复。
- #70：连续十条消息只要求包含问号，不再错误要求问号必须位于句尾。
- #85：每次用户消息必须紧邻一条 AI 消息且字数相同，连续用户消息不能复用旧 AI。
- #86：按成功完成的多碎片聚合窗口计次，不按碎片条数计次。
- #61/#68/#78：从特殊日期消息的 `metadata.occasions` 区分节日、用户生日和二者合并，
  修复统一 `trigger_type=special_date` 后三项不可达的问题。
- #39/#67/#75/#88：读取当前实际维护的 Redis 成长亲密度，不再读取未同步的旧 DB 值。
- 实时序列查询截止到当前持久化消息时间，避免后台任务乱序读到后续消息后漏掉达成窗口。
- 累计事件、连续日 flag 和作息状态 streak 均排除已软删除 conversation 的数据。
- 日终任务显式使用 Asia/Shanghai；单个 user-agent 失败不阻塞后续 pair，
  但整日任务保持失败状态，由 Redis checkpoint 和启动补偿任务重跑。
- 成就积分同步放入数据库事务并锁定 wallet 行，防止并发重复发分和中途失败半提交。

## 已确认的产品口径

- #40 继续停用；若将来恢复，比例规则至少需要 20 条总消息。
- #44 使用相同的 20 条最低样本门槛。
- #60 只统计用户字数。
- #74/#93 要求连续周期内的每一天都覆盖 idle、busy、sleep 三种状态。
- #56 按 Excel 固定 23:00–07:00 判断，不改为动态作息。
- “累计 N 天每日 30 句”统计不同自然日，不要求这些自然日连续。

## 自动化验证

- `test_achievement_catalog.py`：Excel 快照、97 项注册完整性、82/15 状态、停用硬门。
- `test_achievement_*_matrix.py`：用户消息、AI 消息、记忆、意图、主动消息、日终、
  连续序列、里程碑的正例和反例。
- `test_achievement_strict_rules.py`：#5/#48/#62/#85/#86 的严格边界。
- `test_achievement_repository_concurrency.py`：并发解锁、agent 隔离、事件幂等。
- `test_achievement_scheduler.py`：时区、misfire、单 pair 失败隔离。
- `test_wallet_service.py`：并发只发一次积分、ledger 失败整体回滚。
- 成就专项共 268 tests；全量后端回归为 2325 passed、1 skipped。

逐项算法与 testcase 见 `docs/achievement_testcase_matrix.md`。
