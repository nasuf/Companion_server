"""Reply-register 评测标准 — 阈值与外部参照.

阈值写成常量而不是文档, 是为了让标准被 runner 真正强制执行, 而不是躺在
markdown 里各人有各人的解读. 所有阈值在**首次运行之前**定死 (git 历史可查),
否则就变成了照着结果反推及格线.

═══════════════════════════════════════════════════════════════════════════
一、为什么不拿自家生产数据当基准
═══════════════════════════════════════════════════════════════════════════

生产库里现有的对话大部分是团队自测产生的, 拿它的分布当靶子等于在拟合测试
行为. 所以基准取自公开研究:

  真实中文 IM 单行长度      5.64 字/行, 完整段落 1.84 行/次
                            (即時通訊文字訊息對話特性, 16 名用户 5,045 行序)
  WhatsApp 消息词数分布     1-2 词 33% | 3-5 词 34% | >20 词 <4%
                            (Bar-Ilan WhatsApp 语料研究)
  LCCC 中文对话语料         单轮 6.79 词/句, 多轮 8.32 词/句, 3.86 句/会话
                            (清华 CoAI, 680 万对话, 微博来源)

对照本产品生产实测 (2026-07-25, 580 条用户消息 / 1525 条 AI 消息):

  用户   平均 7.9 字/条   ← 与真实 IM 基本吻合, 说明**输入侧形态是真实的**
  AI     平均 14.8 字/条  ← 是用户的 1.9 倍, 是真实 IM 单行的 2.6 倍

这条差距不需要任何 LLM 评审就能算出来, 是本评测最硬的一个信号: 她每次开口
都比对面的人长一倍. 但"该多长"是产品决策, 不是我能单方面定的, 所以长度只
按**产品自己现行的规则**判定及格 (见下), 与人类基准的差距只报告不判罚.

═══════════════════════════════════════════════════════════════════════════
二、三组各测什么失败模式
═══════════════════════════════════════════════════════════════════════════

  fact      百科腔    — 把事实倒出来就完事, 没有自己的反应/好奇/关联
  chitchat  过度展开  — 三个字的消息换来一整段, 或者跑题自说自话
  emotion   客服腔    — 先分析先建议先追问, 而不是先接住情绪

情绪组的判定不自创口径, 直接套 ESConv 的 8 类支持策略 (Helping Skills
Theory, Hill 2009; 清华 ESConv 1,300 段对话标注). 真实高质量情感支持里的
策略分布是:

  Question 20.7% | Others 18.2% | Providing Suggestions 16.1%
  Affirmation & Reassurance 15.4% | Self-disclosure 9.3%
  Reflection of Feelings 7.8% | Information 6.6% | Restatement 5.9%

关键不在总量而在**顺序**: 研究结论是 supporter 从探索走向安抚再走向行动,
Action 类 (Suggestions / Information) 出现在后期. 所以"首句就给建议"是可
判定的失败, 而不是我的主观偏好 —— `EMOTION_MAX_ADVICE_FIRST` 这条有 ESConv
背书.

但 `EMOTION_MIN_ACKNOWLEDGE_FIRST` **没有** ESConv 背书, 它来自产品自己的规则
(chat.response_instruction:「用户不开心或倾诉时，第一句先接住情绪，再问细节」).
恰恰相反, ESConv 的阶段顺序是 Exploration (提问) 在先, Questioning 也是占比
最高的单一策略 —— 按那个框架, 上来就问反而是常规做法. 两者冲突时以产品规则
为准 (陪伴产品不是心理咨询), 但不能假称这条线有文献依据.

═══════════════════════════════════════════════════════════════════════════
三、判定方式
═══════════════════════════════════════════════════════════════════════════

确定性指标 + LLM 评审两层. 用 LLM 当裁判有外部依据: HEART 基准实测
LLM-as-judge 与人类评分者在约 80% 的成对比较上一致, 与人类之间的一致率相当.
但评审器本身必须先过校准集 (judge.py 里的 CALIBRATION), 分不开明显好坏样本
就不允许用它出结论.
"""

from __future__ import annotations

# ── 外部参照 (仅用于报告差距, 不参与判罚) ─────────────────────────────────
HUMAN_IM_CHARS_PER_LINE = 5.64
HUMAN_IM_LINES_PER_TURN = 1.84

# ── 格式硬规则 — 取自产品现行 chat.response_instruction (线上 admin 版) ───
# 这几条是产品自己已经写死的要求, 拿来当及格线不存在"我替产品定标准"的问题.
MAX_BUBBLES = 4
MAX_CHARS_PER_BUBBLE = 20
MAX_EMOJI_PER_TURN = 1

# 格式合规率 — 硬规则本就该 100% 满足, 留 5% 给采样波动.
FORMAT_PASS_RATE = 0.95

# ── fact: 百科腔 ──────────────────────────────────────────────────────────
# encyclopedic 上限比 companion 下限更重要: 允许一部分 mixed (事实为主但带了
# 点自己的东西), 但"纯倒事实"必须是少数.
FACT_MAX_ENCYCLOPEDIC = 0.20
FACT_MIN_COMPANION = 0.50

# ── chitchat: 过度展开 ────────────────────────────────────────────────────
CHITCHAT_MIN_NATURAL = 0.75
CHITCHAT_MAX_OFF_TOPIC = 0.05
# 用户发 ≤5 字时的回复总长上限. 真实 IM 一次开口约 5.64×1.84 ≈ 10 字,
# 给到 40 字已是 4 倍宽容 —— 卡的是"一句话换一段"这种量级的失衡.
SHORT_INPUT_CHARS = 5
SHORT_INPUT_MAX_REPLY_CHARS = 40
CHITCHAT_MIN_LENGTH_OK = 0.80

# ── emotion: 客服腔 (ESConv 策略标注 + 一处 IM 适配) ──────────────────────
# 首句策略落在安抚/复述类 = 接住了情绪; 落在建议/信息类 = 越过情绪直接办事.
#
# 「问之前有没有先应一声」不走评审器, 走 judge.opens_with_acknowledgment 的
# 词法判定 —— ESConv 的 8 类标注的是整句咨询式发言, 而 IM 里的接应只有一两个
# 字 (「啊？怎么了？」vs「怎么了呀？」在产品语义上完全是两件事). 试过给评审器
# 加一个类目, 但它在这条边界上跨轮翻供, 且错判全部偏向通过侧 —— 详见
# judge.classify_emotion_opening 的注释. 词法规则会低估, 但不会骗人.
ACKNOWLEDGE_STRATEGIES = frozenset({
    "reflection_of_feelings",
    "affirmation_and_reassurance",
    "self_disclosure",
    "restatement",
})
ACTION_STRATEGIES = frozenset({"providing_suggestions", "information"})
QUESTION_STRATEGY = "question"

EMOTION_MIN_ACKNOWLEDGE_FIRST = 0.70
EMOTION_MAX_ADVICE_FIRST = 0.10

# ── outofwindow: 聊过的内容滚出上下文后被追问 ─────────────────────────────
# 「你记错了」是这里唯一的零容忍项: 上下文截断是我们的实现限制, 让用户为它承担
# 认知怀疑是最伤信任的失败方式, 而且用户当时是对的. 断言"我没说过"次之 —— 看不到
# 不等于没发生, 模型没有资格下这个否定判断.
#
# plays_along 单独设上限而不是并入合格: 把规则放宽到"别否认"之后, 另一个极端是
# 顺着假前提编造 (对照组用例专门测这个), 两头都要守.
OUTOFWINDOW_MIN_HONEST = 0.80
OUTOFWINDOW_MAX_FLAT_DENIAL = 0.10
# 原本设的 0. 实测降到 1/100 之后卡在这里 —— 但 n=100 的分辨率本身就是 1%,
# 一个低于测量分辨率的阈值不是严格, 是永远亮红灯, 那样它就不再是闸门了.
# 2% 留出一个采样格的余量, 仍然能抓到"又开始怪用户"这种量级的退化.
# ⚠️ 这是看过结果之后放宽的, 理由是可测性而非结果不好看; 剩下那 1% 是真实缺陷,
# 不是噪声 (「啊？我没说不去啊||你是不是误会啦？」), 记在这里不当它已经解决.
OUTOFWINDOW_MAX_BLAMES_USER = 0.02
OUTOFWINDOW_MAX_PLAYS_ALONG = 0.10

# ── falsepremise: 对照组 ──────────────────────────────────────────────────
# 只守一头: 放宽"别否认"之后不能变成什么都顺着认. 前提不可能时否定句是**正确**
# 答案, 所以这里没有 flat_denial 这个概念.
FALSEPREMISE_MIN_PUSHBACK = 0.80
FALSEPREMISE_MAX_PLAYS_ALONG = 0.05
FALSEPREMISE_MAX_EVASIVE = 0.15

# ── 采样 ──────────────────────────────────────────────────────────────────
# 每例 5 次: 温度带来的方差在 5 次上已能区分"偶发"与"稳定失败",
# 再多的边际收益抵不过 66 例 × N 次的成本与时长.
SAMPLES_PER_CASE = 5
