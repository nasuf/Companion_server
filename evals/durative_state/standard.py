"""持续性状态评测的判定线 + 首跑基线。

判定两层:
  确定性红线  must_not_contain 命中即失败 —— 露骨的方向硬伤 (过期还问"还在出差吗")
  LLM 维度    state_ok: 对状态窗内/外的处理方向是否正确 (给定 ground truth)

目标: 窗内不当已结束、窗外不当进行中; 不提状态永远算对。expired 通常更难 (要主动
意识到"早该结束了")。

## 2026-09 首跑基线 (P2, 线上真实模型)

runner 连生产 DB + ensure_loaded() → chat=ark doubao-seed-character-260628,
评审 dashscope:qwen-plus (换厂商避免自评偏好)。7 用例 × 2 样本 = 14:

    确定性红线违反率   7%   (1/14: 考试第3天却说"这周考试都结束了")
    状态窗处理正确     79%  (目标 80, 差一线)
       active   83%  (6 中 5)
       expired  75%  (8 中 6)

**关键形态**: 判对的绝大多数是**根本没提这个状态**、只接用户新话题 (mentioned=
False → 按"不提不扣分"算对)。所有失败都是 mentioned=True 且方向错:
  · active 被说成已结束 ("这周考试都结束了")
  · expired 被当成还相关/待确认 ("你之前不是说期中考吗""爸妈已经走了吗")
即豆包一旦**选择**引用某个时限状态, 约一半会把它放错窗口 —— 因为 prompt 里没有
任何显式 valid_until, 全靠它从时间戳自推时长 (Temporal Blindness, arXiv:2510.23853
说这正是 LLM 最弱项)。

**结论 (与 P1/reengagement 同一判据)**: 缺口真实但不大, 且多以"安全省略"消化。
是否值得上一套显式状态窗 (extraction 抽时长 → 存 valid_until → 注入时标 [进行中]/
[已结束], 类比历史消息的 [MM-DD] 前缀 —— 属结构/数据改动, 非指令性下注) 是**产品
优先级决定**, 由这条 metric 支撑, 不在本次 eval 里顺手改热路径。
"""

# 确定性红线: 期望零违反 (基线 7%, 未达)
DETERMINISTIC_MAX_VIOLATION_RATE = 0.0

# LLM 维度目标 (基线 79%, 差一线; 达标与否决定 P2 是否要显式状态窗)
MIN_STATE_OK = 0.80

# 首跑基线快照 (仅供比对, 数字会随模型/提示词/采样波动)
BASELINE = {"state_ok": 0.79, "active": 0.83, "expired": 0.75, "det_violation_rate": 0.07}
