"""时间感知评测的判定线 + 首跑基线记录。

判定分两层:
  确定性红线  must_not_contain 命中即失败 —— 露骨的时段/间隔硬伤, 该 100% 不犯
  LLM 维度    gap 得体 / 无时间幻觉 / 时刻贴合 / 不复活旧话题

## 演化史

### v1: 2026-09 首跑基线 (P4, samples=1, 线上豆包)

runner 连生产 DB + ensure_loaded() → 解析线上真实配置:
  chat  = ark  doubao-seed-character-260628   (豆包, 主回复)
  small = dashscope qwen3.5-flash              (小模型)
评审 dashscope:qwen-plus.

    确定性红线违反率   0%     ✓
    时刻贴合          100%    ✓
    时间幻觉           0%     ✓
    间隔处理得体       64%    ✗  目标 70
    无端复活旧话题     45%    ✗  目标 20

samples=1 数字有噪音, 后来发现 v2 baseline 实际差得多.

### v2: 2026-09-13 baseline 稳定化 (samples=3, 15 用例 → 45 样本)

samples=1 显然是噪音: 45% stale 是"偶尔靠住了几道"的假象. 拆到 samples=3 后
真实 baseline 打脸得多:

    ALL GROUPS baseline (samples=3, drop=off):
      确定性红线违反率   16%  (7/45)   ← 露骨复读"成都/四五天"
      间隔处理得体       58%
      时间幻觉           0%
      时刻贴合          100%
      无端复活旧话题     51%           ← 每两道就有一道翻旧话题
      分组红线:
        stale_topic     7/15  违反 (仅这一组产生红线)

**为什么这么惨**: build_chat_messages 一直把 pre-gap 逐字历史塞进 message list,
即使加了重逢感知段 (chat.reengagement_*, 明写"别接旧话题"), 模型看到旧话题原文
就是会接. **指令**拦不住**上下文里存在**的信息. 这跟 P1/reengagement 强化两次
A/B 挂了完全一致 —— 指令性下注对"pre-gap 原文可见"这个结构性问题无效.

### v3: 2026-09-13 结构性修法 A/B (samples=3, drop_pre_gap_history=ON)

思路: 大间隔时 (gap ≥ 3h) 把 pre-gap 逐字历史从 message list 里砍掉, 让模型
物理看不见旧话题原文. 3h 阈值跟 topic 栈重置 / session_recap 触发对齐
(TOPIC_RESET_GAP_SECONDS / RECAP_GAP_SECONDS = 3*3600). session_recap 段仍在
system_prompt 里, 给"上次聊到什么"的语义抓手.

    ALL GROUPS 对比 (drop=off vs drop=on):
      metric          drop=off  drop=on   delta   target
      ------------------------------------------------------
      红线违反率      16%       0%        -16pp   0%       ✅ 达标
      间隔处理得体    58%       84%       +26pp   ≥70%     ✅ 达标
      时间幻觉        0%        0%        0       ≤15%     ✅
      时刻贴合        100%      100%      0       ≥80%     ✅
      无端复活旧话题  51%       20%       -31pp   ≤20%     ✅ 达标
      分组红线:
        stale_topic   7/15      0/15      -7      -        ✅
        (其它三组本来就 0/N, 保持 0/N)

**5 个指标里之前 3 个未达标, 全部达标. 无回退**. reunion / time_of_day /
no_hallucination 三组的红线保持 0/N (drop 只在 gap ≥ 3h 触发, 短间隔用例的
message 时间戳都在 3h 内, 不受影响).

**机制解释**:
  1. 红线 16→0: 模型没有旧话题原文可以复读, 露骨复读消失
  2. stale 51→20: 语义级"接旧话题"下降一多半 (剩下的 20% 是模型自己的先验联想,
     比如"忙疯了→歇歇看电影", 不是我们喂原文进去的)
  3. gap_ok 58→84: 少了"硬接旧话题"的干扰, 模型更倾向自然承接当前消息 + 承认
     离开, 得体率被动上升

**评测 vs 生产的 gap**:
  评测里 drop=on 时**没有** session_recap (test runner 简化, 不预生成 recap).
  生产里 drop_older_than_seconds 触发时, session_recap 段也会同时注入 system
  prompt (RECAP_GAP_SECONDS 是同一个 3h 阈值). 所以生产的 drop=on 应比评测这里
  的更好 (有语义抓手 + 无原文可复读). 若某个 recap 摘要恰好带回了老话题词, 有
  可能小幅上升 stale, 但摘要是 1-2 句的高阶总结, 不是原文级刺激.

### 上线动作

生产已改: build_chat_messages 增 drop_older_than_seconds 参数, orchestrator
在 gap ≥ RECAP_GAP_SECONDS 时传 RECAP_GAP_SECONDS 值 (2026-09-13 提交).
单元回归: tests/test_chat_history_window.py::TestDropPreGapHistory (5 case,
含"当前 turn 永不砍"兜底).
"""

# 确定性红线: 期望零违反 (drop=on 后达到)
DETERMINISTIC_MAX_VIOLATION_RATE = 0.0

# LLM 维度目标 (drop=on 后全部达标)
MIN_GAP_OK = 0.70          # 对间隔处理得体   (drop=on 实测 84%)
MAX_HALLUCINATION = 0.15   # 时间幻觉          (drop=on 实测 0%)
MIN_TOD_OK = 0.80          # 时刻贴合          (drop=on 实测 100%)
MAX_STALE_TOPIC = 0.20     # 无端复活旧话题    (drop=on 实测 20% 恰达标线)

# 首跑 v3 (drop=on) baseline 快照, 供未来对比 (数字随模型/提示词/采样波动)
BASELINE_DROP_ON = {
    "det_violation_rate": 0.00,
    "gap_ok": 0.84,
    "hallucination": 0.00,
    "tod_ok": 1.00,
    "stale_topic": 0.20,
    "n": 45,  # 15 cases × samples=3
}
