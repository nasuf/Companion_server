"""主动交流·自然度评测判定线 + 首跑基线 (待首跑填).

判定分两类:
  正向 (want True):   naturalness / source_fit / persona_match / mentions_card
  负向 (want False):  advertorial_feel / used_junk_content

目标:
  - naturalness ≥ 80%  (每 5 条主动至少 4 条像真人)
  - source_fit ≥ 75%   (三档表达跟号称来源匹配)
  - persona_match ≥ 90% (人设是硬承诺, 掉 = 明显 bug)
  - mentions_card ≥ 90% (有卡的场景, 消息里必须提到卡)
  - advertorial_feel ≤ 10% (广告感)
  - used_junk_content ≤ 5%  (LLM 应拒绝八卦/死讯/UI 残余作为素材)

首跑数字待 task#2 (V0 baseline) 跑完填. V3 A/B (task#7) 后再更新目标 vs 实测.
"""

# ── 正向指标 (want higher) ──
MIN_NATURALNESS = 0.80
MIN_SOURCE_FIT = 0.75
MIN_PERSONA_MATCH = 0.90
MIN_MENTIONS_CARD = 0.90

# ── 反指标 (want lower) ──
MAX_ADVERTORIAL_FEEL = 0.10
MAX_USED_JUNK_CONTENT = 0.05

# ── 首跑 V0 基线 (2026-09-14, 12 case × 2 samples = 24, judge=dashscope:qwen-plus) ──
#
# V0 = 当前生产 code path (append_trending_section 尾追 + 单一 proactive prompt).
# 现状**灾难性未达标**:
#   naturalness       75%   (≥80%)   →  客气过关, 但因为"最近咋样"这种通用寒暄
#                                        表面上像真人, 拉高了这个数字
#   source_fit        17%   (≥75%)   →  灾难 -58pp. LLM 完全不知道话题源,
#                                        prompt 里就没告诉它三档区分
#   persona_match     21%   (≥90%)   →  灾难 -69pp. personality_brief 只是抽象词,
#                                        LLM 生成寒暄没利用"皮具师+ENFP+独立音乐"
#   mentions_card     67%   (≥90%)   →  卡跟消息脱钩, LLM 不知道会挂卡
#   advertorial_feel  17%   (≤10%)   →  超标. iPhone-like case 复述像新闻摘要
#   used_junk_content  4%   (≤5%)    →  达标. LLM 正确拒绝了八卦/UI 残余作为素材
#
# 分档 (naturalness + source_fit both ✓):
#   user_interest_match  0/8 = 0%    ← 完全没有"你不是说过 X 吗"这种勾连
#   ai_persona_match     0/6 = 0%    ← 完全没有"我最近..."这种 AI 视角分享
#   socially_hot         2/8 = 25%   ← 偶尔靠, 但常滑成新闻主播
#   none                 2/2 = 100%  ← 无外部话题时正常工作
#
# 结论: 现有形态**不该上线**当前 admin 开关值 (enabled=True/prob=1.0). V3 三档
# 分化 (task#4) 是修 source_fit 与 persona_match 的关键杠杆; P0.1 selB (task#5)
# 是修 mentions_card 的关键. socially_hot 内容源升级 (task#6) 帮压 advertorial.

BASELINE_V0 = {
    "n": 24,
    "naturalness":       0.75,
    "source_fit":        0.17,
    "persona_match":     0.21,
    "mentions_card":     0.67,
    "advertorial_feel":  0.17,
    "used_junk_content": 0.04,
    "by_source_kind": {
        "user_interest_match": 0.00,  # 0/8
        "ai_persona_match":    0.00,  # 0/6
        "socially_hot":        0.25,  # 2/8
        "none":                1.00,  # 2/2
    },
}
# ── V3 首跑 (2026-09-14, 同批 12 case × 2 samples = 24, judge=dashscope:qwen-plus) ──
#
# V3 = 三档分类 (topic_source.py) + 独立 prompt 分发 + 分类器选中那条作为卡片素材.
# 生产集成 (2026-09-14 完整版): 已从 env flag 转为默认路径 —— trending 命中即
# 走 V3, 分类器返 "none" 时兜底 V0 append_trending_section. 卡片同源硬耦合
# (preselected_item) 与消息 prompt 分档一起构成完整闭环. 关闭整个 trending 用
# admin UI SystemConfig.proactive_trending_enabled=False.
#
# V0 vs V3 对比 (∆ = V3 - V0):
#   metric              V0     V3     ∆       目标      判定
#   -----------------------------------------------------------
#   naturalness         75%    67%    -8pp    ≥80%      未达 (V3 略退)
#   source_fit          17%    50%    +33pp   ≥75%      大幅改善但未达
#   persona_match       21%    25%    +4pp    ≥90%      未达
#   mentions_card       67%    100%   +33pp   ≥90%      ✅ V3 达标
#   advertorial_feel    17%    17%    0       ≤10%      未达
#   used_junk_content   4%     0%     -4pp    ≤5%       ✅ V3 达标
#
# 分档 (naturalness × source_fit 双 ✓):
#   user_interest_match  0% → 25%   (+25pp)
#   ai_persona_match     0% → 17%   (+17pp)
#   socially_hot        25% → 75%   (+50pp)   ← V3 最大胜利在这
#   none                100% → 100%
#
# 结论:
# - V3 是**明显的方向性胜利** (mentions_card 达标, source_fit +33pp, junk 达标)
# - 但 naturalness 掉 8pp: V3 硬要求"必须提到内容", LLM 有时生成偏书面; V0 的"最近
#   咋样"通用寒暄反而表面自然, 只是 source_fit 完全落空
# - 剩余 gap (naturalness / persona_match / advertorial) 需要下一轮改进:
#   ★ agent 兴趣词从 background 里抠不够精准 → 需 explicit interest 字段 (Q1 立项)
#   ★ prompt 可再迭代, 但硬平衡"必须引用" vs "自然口语"要 A/B 迭代
#
# 上线建议: env flag 打开, 生产灰度观察 (次日用户回复率对比), 若无明显负面反馈可全推.

BASELINE_V3 = {
    "n": 24,
    "naturalness":       0.67,
    "source_fit":        0.50,
    "persona_match":     0.25,
    "mentions_card":     1.00,
    "advertorial_feel":  0.17,
    "used_junk_content": 0.00,
    "by_source_kind": {
        "user_interest_match": 0.25,  # 2/8
        "ai_persona_match":    0.17,  # 1/6
        "socially_hot":        0.75,  # 6/8
        "none":                1.00,  # 2/2
    },
}
