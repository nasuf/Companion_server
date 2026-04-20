"""交互系统模块（spec 第三部分）。

按 spec 章节归并：
- §1 碎片聚合 → `aggregation`
- §2 边界系统（耐心值/违禁判定/攻击分级/道歉恢复/拉黑计时） → `boundary`
- §6 异步回复（延迟计算、延迟队列、时间戳沿用） → `reply_context` + `delayed_queue`

其他交互分支（§3 意图识别 / §4 日常交流 / §5 回复加工）暂留 `chat/` 下，
由 `chat.orchestrator` 统一编排。
"""
