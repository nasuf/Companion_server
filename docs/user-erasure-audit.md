# 用户删除审计与验收（2026-10-20）

原删除实现不完整，不能据此声称“所有数据和所有副本都已彻底删除”。本次补齐在线 PostgreSQL、已知 Redis 私有数据、服务器用户媒体的清理和失败保护；生产部署、历史备份、外部平台和无法归属的历史材料不计作已通过。

## 修复的实际问题

- 高：部分 SQL、Redis 和离线媒体删除错误被吞掉，账号可能已消失，旁路记录或文件却仍在。现在清理失败向上传递，数据库事务回滚；账号保留为 `deleting`，管理员可见“删除未完成”，允许重试。磁盘和 Redis 不支持数据库回滚，已清除的私有内容不会恢复；数据库原始记录保留为重试线索。
- 高：原实现没有清理保存完整输入/输出的 `trace_runs`，遗漏无用户外键的语音用量。现在先捕获 Trace 根及依赖，再清除整条 Trace；补齐语音记录和迟到写入约束。`message_traces` 镜像也有会话级联保护。
- 高：用户删除会扫描并清空所有人的 `cache:ret:*` / `cache:graph:*` / `cache:emb:*`。现在不再全局清缓存，新检索/图谱缓存带用户作用域；旧缓存按结构化资源归属清理，共享索引只移除目标成员。
- 高：关闭账号后，旧 JWT 原本仍只校验签名和过期时间。现在每次检查 SQL 账号状态和当前角色；删除中或已删除账号不可继续使用旧凭据。主动关闭目标账号 WebSocket，并通过现有跨进程订阅通知其他 worker。连接登记后复查账号，堵住检查与登记之间的撤销竞态；管理员查看他人会话时也记录实际操作者，撤销管理员不会关闭随后替换上来的用户连接。在线状态的迟到写入同样受账号锁保护。
- 高：在途上传可能在删除扫描后重新落盘。头像、反馈、胶囊、聊天图片/用户语音/生成语音、离线活动媒体和礼物转存图片的短写入阶段增加用户行锁；资料更新直接取排他锁，避免并发更新的锁升级死锁，其余短写入取共享锁。锁不跨模型调用或外部图片下载。
- 高：链接封面使用独立文件台账与容量额度，原清理没有释放这部分记录。现在封面落盘也受账号锁保护，取消请求会等待磁盘写入真正结束；用户文件清理后，在同一删除事务里清除该用户封面台账并准确释放占用额度。
- 高：迟到的摘要、话题、上下文、记忆确认候选和队列生产者可能重新写入私有内容。相应短写入阶段增加生命周期校验；运行中任务阻断删除，待停止后重试。SQL side table 的新写入与用户归属重绑也受数据库约束保护。
- 高：部分旧用量/诊断记录的 `user_id` 为空，但仍绑定私有会话、workspace 或 Agent；现在按这些可信资源绑定清理，同时捕获其关联 Trace。新写入必须通过资源存活校验；记忆向量的迟到插入和更新也会在数据库拒绝。
- 高：历史错误绑定可能让两个账号共用微信身份标识。清理身份缓存前先查跨账号冲突，发现冲突就停止，不删除另一账号仍需使用的认证状态。
- 高：部署原来没有持久化反馈截图，更新容器可能丢失其他用户的图片。现在增加独立持久目录，更新前停止旧容器并迁移截图；同名文件内容不同、目录异常或复制失败会阻止替换容器。文件合并不覆盖已有图片，重复迁移可安全重试；中断时的临时文件也有用户前缀，纳入删除扫描。
- 中：反馈写库失败原本在文件清理 try 范围外，可能留下未引用截图。现在写库失败也清理本次图片。
- 中：批量删除原本直接删除当前列表所有用户。现在进入勾选模式，明确确认选中名单；取消/翻页/搜索清空选择，失败项保留重试，正在删除时锁定筛选与分页。

## 数据归属与原子性

1. 禁止删除系统模板账号；接口禁止自删，最后一名活跃管理员的检查在数据库中串行化，防止两名管理员同时互删。
2. 先提交账号 `deleting` 状态和执行代数更新，使新请求/SQL scope 失效。
3. 检查错误归属的 Agent、workspace、会话、资源引用、跨账号 Run/来源/支付链；有冲突则中止，不能靠删除另一人的记录“修好”。
4. 在独立请求的事务上下文执行账号清理。失败不提交数据库删除。文件必须先清理，再删除其引用记录。
5. 删除用户后，事务内遍历实际 `information_schema` 的所有受检归属列，以及 Agent/workspace/会话/消息/记忆/实体/Trace 资源列；任何残留都会中止。新增直接用户表、只记录资源 ID 的间接表都不能静默漏过。没有结构化归属的历史材料仍受下文边界限制。
6. 清理完成才返回成功。`stats` 包含显式阶段的发现/清理数量：`*_found` 是发现数量；显式 SQL 清理按行、文件按实际移除文件、Redis 按键/索引成员操作计数。级联删除没有完整逐表计数，不能将这些字段相加当作用户总数据量。作用域仅该用户，不是全库累计指标，也不表示第三方/备份已抹除。

本次复用“用户管理”的删除进度、失败原因和账号状态展示，不新增独立后台页面。所有删除接口仍限管理员；确认名单只来自当前页实际勾选项。

## 服务器媒体

| 存储目录 | 清理内容 | 隔离策略 |
| --- | --- | --- |
| `CHAT_MEDIA_DIR` | 聊天原图、缩略图、用户语音、AI 生成音频、链接封面、无引用遗留文件 | 数据库引用 + 用户/会话存储前缀；外来用户 key 中止删除 |
| `CAPSULE_MEDIA_DIR` | 时光胶囊图片/音频与上传后未封存的孤立文件 | 验证用户前缀，引用和前缀扫描一起覆盖 |
| `USER_AVATAR_DIR` | 自设头像及小尺寸版本 | 目标用户前缀；有已知头像时目录缺失不得视为成功 |
| `FEEDBACK_MEDIA_DIR` | 意见反馈截图与遗留图片 | 目标用户前缀；有图片记录时检查挂载可用 |
| `OFFLINE_MEDIA_DIR` | 活动照片/音频、礼物图片及遗留文件 | 先文件后引用；公共 `place_` 图片保留 |

公共 Agent 头像、音乐/贴纸等静态资源、公共地点图片和全局音色库不是某账号独占文件，不因删用户而删除。全局音色库的克隆录音样本采用 `tts_enroll_` 名称与独立定时清理，不能把所有管理员共用音色当作当前用户附件。

## 在线表覆盖清单

以下为独立测试库结构的实际清单，不是生产库已验收的声明。除直接用户表外，会话消息、游戏事件、物流节点、记忆证据、SQL Run/Job/Action/Outbox、接管记录等子表通过已核对的级联关系或明确清理覆盖。没有完整级联的旁路表由显式清理与残留检查兜底。共享配置审计只移除或匿名化目标操作者归属，不删除其他用户业务数据。

受检归属表：72 张。

| 表 | 事务结束前必须无目标用户值的列 |
| --- | --- |
| `achievement_events` | `user_id` |
| `achievement_unlocks` | `user_id` |
| `agent_runs` | `actor_user_id`, `owner_user_id` |
| `agent_wallets` | `user_id` |
| `ai_agents` | `user_id` |
| `auth_identities` | `user_id` |
| `bug_reports` | `reporter_id`, `resolved_by_id` |
| `chat_link_cards` | `user_id` |
| `chat_message_attachments` | `user_id` |
| `chat_workspaces` | `user_id` |
| `conversations` | `user_id` |
| `crisis_events` | `user_id` |
| `game_point_ledger` | `user_id` |
| `game_sessions` | `user_id` |
| `gift_addresses` | `user_id` |
| `human_takeover_dismissals` | `admin_id` |
| `human_takeovers` | `owner_admin_id`, `user_id` |
| `iap_subscription_state` | `user_id` |
| `iap_transactions` | `user_id` |
| `intimacies` | `user_id` |
| `last_wills` | `user_id` |
| `llm_usage` | `user_id` |
| `memories_ai` | `user_id` |
| `memories_user` | `user_id` |
| `memory_changelogs` | `user_id` |
| `memory_consolidation_runs` | `user_id` |
| `memory_entities` | `user_id` |
| `memory_evidence_links` | `source_user_id`, `user_id` |
| `memory_input_origins` | `user_id` |
| `memory_mentions` | `user_id` |
| `memory_profile_origins` | `user_id` |
| `memory_quality_states` | `user_id` |
| `memory_repair_items` | `resolved_by_id`, `user_id` |
| `memory_visible_use_events` | `user_id` |
| `music_co_listening_sessions` | `user_id` |
| `music_favorites` | `user_id` |
| `music_playbacks` | `user_id` |
| `native_game_skill_states` | `user_id` |
| `notification_events` | `user_id` |
| `offline_activity_feedback` | `user_id` |
| `offline_activity_media` | `user_id` |
| `offline_activity_recommendations` | `user_id` |
| `patience_states` | `user_id` |
| `proactive_chat_logs` | `user_id` |
| `proactive_counters` | `user_id` |
| `proactive_event_logs` | `user_id` |
| `proactive_states` | `user_id` |
| `proactive_tuning_changes` | `admin_id` |
| `push_devices` | `user_id` |
| `real_world_gifts` | `user_id` |
| `real_world_recharge_ledger` | `user_id` |
| `real_world_trigger_states` | `user_id` |
| `speech_usage` | `user_id` |
| `time_capsules` | `user_id` |
| `time_triggers` | `user_id` |
| `tts_usage` | `user_id` |
| `user_consumable_batch` | `user_id` |
| `user_daily_activity` | `user_id` |
| `user_feedback` | `user_id` |
| `user_game_wallets` | `user_id` |
| `user_message_quota` | `user_id` |
| `user_music_quota` | `user_id` |
| `user_offerings` | `user_id` |
| `user_portraits` | `user_id` |
| `user_profile_tags` | `user_id` |
| `user_profiles` | `user_id` |
| `user_store_inventory` | `user_id` |
| `user_wallets` | `user_id` |
| `vip_activation_codes` | `created_by` |
| `vip_code_redemptions` | `user_id` |
| `wallet_ledger` | `user_id` |
| `workspace_interaction_days` | `user_id` |

`trace_runs` 没有直接用户列，按会话/Trace 元数据/usage/消息绑定捕获 Trace ID 后清理根和子节点。`iap_notifications` 没有直接用户列，按可信交易/订阅链、provider 和 environment 清理；跨账号交易链中止处理。`memory_embeddings` 按记忆 ID 清理，检查跨用户同 ID 冲突，保留其他用户向量。

`chat_link_cover_objects` 按经过校验的用户文件前缀清理，`chat_link_storage_state` 只扣除目标用户实际登记字节；共享容量台账和其他账号封面保留。跨用户 Trace 绑定也会中止删除，避免清走另一账号的调试记录。

## Redis 覆盖

清理用户、Agent、workspace、conversation、Trace 的命名空间键；清理目标后台任务及 ready/delayed/running/DLQ/succeeded 索引中的对应成员、幂等绑定、单次 WebSocket 票据及限流、碎片聚合/整轮队列、记忆处理诊断 hash/ZSET、前台在线索引。任务正在运行则拒绝假成功并允许停止后重试。共享队列、公共缓存和其他账号成员保留。查询使用分批 SCAN 并设置扫描预算，不能用 `FLUSHDB` 或全局私有缓存删除。

## 验证与明确边界

- 基线 `0e06f945` 上的本地全量门禁：520 个测试文件、9583 项收集，9580 通过，3 项按既有规则跳过（两项 Flutter 跨仓验证、一项 Linux `/proc` 资源检查）；没有失败或网络越界。整体行/分支覆盖率为 81.85% / 70.97%，本次改动为 91.53% / 85.59%，原有门槛通过。测试副本与当时待提交的 1508 个源码、测试、脚本及配置文件内容一致。
- 随后整合远端主分支至 `b28d6a77`，再次验证时间查询、活动管理、账号权限、20 项真实删除场景和连接撤销等，949 项增量测试通过。最终提交的整套门禁及生产验收状态以对应 CI/发布结果为准。
- Web 最新主分支整合后，生产构建以及用户批量删除、用户搜索、聊天弹窗、活动管理浏览器测试通过；Trace 浏览器回归亦通过。上述均为本地验证，不代表生产删除验收。
- 浏览器夹具：批量删除进入选择、未选禁用、确认名单、只发送勾选 ID、失败重试、取消/搜索/翻页重置；并回归用户搜索和聊天弹窗。
- 真 PostgreSQL + Redis，专门新建的本地隔离容器，20 项删除与并发场景：目标和对照用户、Trace 根/子节点、空用户 ID 的诊断记录、语音记录、记忆/向量、任务和媒体；对照用户行与文件按原值比较。包含正常缓存/上下文/在线状态写入、删除后的迟到写入、错误跨用户绑定及媒体目录不可用。全量 migration 也在新建空库从头执行通过。
- 反馈截图迁移：19 项归档安全、冲突、重复执行、旧挂载、部署顺序与实际 shell 失败保护测试；另用实际停止的隔离 Docker 容器复制两个合成账号的图片，重复执行得到 0 个新增、2 个一致文件，原字节保留。未读取生产截图。
- 失败路径：磁盘删除失败保留引用并能重试；新建未清理用户表或间接会话表触发残留校验和事务回滚；错误用户附件 key、跨用户封面/Trace 不触碰对方数据；上传与删除并发串行；已删除/删除中旧 JWT 返回 401；迟到语音/Trace/封面/队列写入不能复活已清理记录。
- 新 migration 在专用测试库实际执行，Prisma schema 校验通过。未对生产用户执行删除，不把本地夹具称为生产 E2E。

仍不能宣称所有副本立即归零的边界：

1. **历史备份**：现有备份引擎的保留窗口为 14 天（代码配置，未核实生产实际副本），且为整库共享备份。本次账号接口不改写备份，也不会误删包含其他用户的整个备份。当前没有经过本次验收的“恢复前按用户重放删除清单”机制；恢复旧备份可能恢复历史账号内容。若要求备份也即时不可恢复，需要单独实施选择性重写/加密密钥销毁及恢复流程，不能声称已完成。
2. **第三方**：Apple、微信、短信/推送、模型提供商和历史 LangSmith 服务的副本不由本地数据库级联删除。若曾启用相关上传或公共分享，必须使用供应商自身的删除/撤销能力并取得回执；本次未验证这些远端数据的实际存在或删除结果。
3. **无法归属的历史导出/日志/Trace/通知及旧缓存**：有确定账号、会话、Trace 或交易绑定的在线数据按上述链路清理。失去全部结构化归属的历史材料不能通过搜索姓名或聊天文本自动推断归属，避免误删其他人。旧版检索/图谱缓存键只有哈希；能通过结构化资源 ID 归属的条目会清理，但如果关联记忆此前已删除且值没有用户归属，就无法安全定位该历史条目。其余旧条目按原 TTL 到期，不能称为立即全部抹除；新版私有缓存键已带用户 ID，后续删除可准确定位全部版本。需另行核对生产旧缓存、日志/导出位置及无归属历史记录；本次没有证据证明所有历史材料都不存在。
4. **共享嵌入计算缓存**：`cache:emb` 只保存按文本哈希索引的数值向量，既无用户列也无原文，TTL 为 30 分钟；不能把清空所有人的共享缓存当成账号隔离清理。本次没有引入可追溯每次历史计算归属的倒排索引。因此不把它描述为“所有派生数据即时抹除”。

如产品要求包含上述边界的“绝对无任何留存”，当前架构仍需后续设计与运营验收。本次解决受控在线数据清理的具体缺口，但不会把这些外部/历史风险标成已关闭。

已下载到终端、浏览器或第三方 CDN 的副本也不能通过删除服务器磁盘文件主动召回；需按对应缓存/存储机制处理。
