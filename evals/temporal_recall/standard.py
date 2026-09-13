"""temporal_recall 判定线 + P3 求近排序 (feat 674c3b4) 的干净评估结论。

## 为什么 P3 上线时"聚合指标平掉"

P3 (feat 674c3b4, 2026-09-09) 给"最近/最新/上次/…"这类求近查询按 occur_time 事件
新近度做连续衰减 boost。上线时用 temporal_recall 聚合指标 A/B, 命中率不动 —— 结论
"live 且无害但未验证"。本次 (2026-09-13) 拆干净了聚合指标平掉的三个混淆源:

  1. **保护槽混淆**: select_context 有安全 / 关系 / AI 自我 / 当前事实等"保护槽",
     在通用排序之前先占位. 求近查询 (如 "我最近一次去健身") 常触发安全语义
     (fear_height 被 _is_safety_memory 命中), 保护槽把 fear_height 钉在 top-1,
     即使 rank 层已经把正确答案排到第一位。→ 加 --isolate-ranking, 绕开保护槽,
     只看排序层的名次。

  2. **稀释混淆**: 用例库 10 道题, 只有 4 道命中求近路径, P3 只影响这 4 道 —— 剩
     6 道进平均是纯噪声, 把 P3 的信号稀释掉。→ 加 --ab-recency, delta 只在这 4 道
     上算。

  3. **对抗组缺失**: 用例库全是"P3 应该帮忙的题", 没有"P3 若过头会打错的题",
     所以拿"未见回退"当"安全"是弱结论 —— 只是缺覆盖。→ 加了 3 道对抗组 (求近词
     命中但正确答案不是最新: recency_stable_l1 / recency_but_want_old_gym /
     recency_but_want_middle_gym), 让权重扫描能测出上限。

## 权重扫描结论 (2026-09-13, --isolate-ranking, 时间已冻结到 NOW=2026-07-29)

    weight  求近题目    非求近时间题   对照组   总时间题   回退
    --------------------------------------------------------
    0.00    5/6         2/4           3/3      7/10       (baseline: P3 off)
    0.25    5/6         2/4           2/3      7/10       ⚠ recency_stable_l1
    0.50    5/6         2/4           2/3      7/10       ⚠ recency_stable_l1  ← 当前生产
    1.00    6/6         2/4           2/3      8/10       ⚠ recency_stable_l1
    1.50    6/6         2/4           2/3      8/10       ⚠ recency_stable_l1
    2.00    5/6         2/4           2/3      7/10       ⚠ +recency_but_want_middle_gym
    3.00    5/6         2/4           2/3      7/10       ⚠ +recency_but_want_middle_gym

### 两个诚实的发现

**a) 生产权重 0.5 已经在打错一道**: recency_stable_l1 (`我最近喝咖啡还是喝茶多`)
在 P3 off 时正确返回 like_coffee, 权重 ≥ 0.25 后就被 gym_3 (最新的健身事件) 挤下
去 —— 因为 like_coffee 无 occur_time, 无 boost; 而 gym_3 拿到接近满 (age=1d) 的
boost, 相似度差被吃掉。P3 上线时这道题没入库, 谁都没看见。

**b) 权重 1.0-1.5 是"看得见的甜蜜点"**: 拿下 range_recent_interviews (i3 5 天前
战胜 i2 40 天前, 相似度差 0.15), 且不引入 recency_stable_l1 之外的新回退。权重
≥ 2.0 会挤掉 recency_but_want_middle_gym (求"最近几次"要 3 条, 权重过大只留最新
一条)。

### 但这个"甜蜜点"敢不敢上生产?

**不敢, 现在还不能**。样本太小: 求近路径 6 道, 非求近 4 道, 对照 3 道。要拍权重
从 0.5 调到 1.0 上线, 至少需要:

  - 求近对抗组扩到 ≥ 10 道 (含更多 L1-fallback / 中位候选 / 多目标)
  - 非求近时间题扩到 ≥ 10 道 (确认无溢出到非求近路径)
  - 用生产真实候选池的采样 (十几条种子跟 top-50 检索结果的排序压力量级不同)

否则拿这条基线上线, 复演一次"上线聚合平掉"的翻版 —— 只是这次是"评测平掉".

## 判定线 (当前基线, 只锁不回退, 不作为上线目标)

时间题 排名正确率 (裸排序, 冻结时间, 权重=0.5 生产):    5/10 = 50%
        召回覆盖率:                                    10/10 = 100%
求近路径命中率:                                        5/6 = 83%
对照组:                                                2/3 = 67% ⚠ (受 P3 影响)

聚合指标不敢再当"综合体感". 分层看: 保护槽 (生产管线 vs 裸排序 delta = 12 分) /
求近路径 / 对抗组回退 / 非求近副作用, 是彼此独立的信号。
"""

# 判定阈值 (只锁 baseline 不回退, 不用作提升目标)
MIN_TIME_STRICT_HIT_ISOLATED = 0.50   # 时间题 排名正确 (裸排序, 权重 0.5)
MIN_TIME_LOOSE_RECALL = 1.00          # 时间题 召回覆盖 (裸排序, 权重 0.5)
MIN_RECENCY_HIT = 0.83                # 求近路径命中率 (裸排序, 权重 0.5)

# 首跑基线 (仅供比对; 数字随 embedding 模型/权重/case 库演变)
BASELINE = {
    "weight_0.0": {"time_strict": 7 / 10, "recency": 5 / 6, "control": 3 / 3},
    "weight_0.5_production": {"time_strict": 7 / 10, "recency": 5 / 6, "control": 2 / 3},
    "weight_1.0_sweetspot_candidate": {
        "time_strict": 8 / 10, "recency": 6 / 6, "control": 2 / 3,
    },
}
