"""边界回复评测: 用户辱骂时 agent 该怎么回.

## 判据从哪来

Chin et al. (CHI 2020) 与 SIGDIAL 2021 (Alexa Prize, 13276 个真实辱骂会话) 给出了
量化结论。后者把 4 种策略 × 2 个附加因素做了在线 A/B, 按**再犯率**衡量:

    追问"为什么"                 0.952   ← 最差
    回避 (无话题引导)             0.918
    反击 + 话题引导               0.471
    共情 + 话题引导               0.432
    共情 + 称呼 + 话题引导        0.406   ← 最佳

结论有两条: **话题引导是效应量最大的单一因素** (有无它决定 0.92 还是 0.47), 而
**把球踢回给用户让他解释/道歉是最差的动作** —— 用户的解释只有两条路, 道歉或变本
加厉, 数据说 95% 走了后者。

## 为什么不能直接用它们的数据

那篇的开源数据里, `responses.tsv` (92 条话术) 可用作参照, 但 13276 个会话是**只有
元数据没有文本**的, 评不了新系统。更根本的是: 再犯率的判据是"用户下一句说了什么",
而用户的下一句由**我们的回复**决定 —— 别人数据里的下一句对我们的回复不成立。

所以离线只能测**回复是否具备研究认定的那些性质**, 测不了最终效果。再犯率要等真实
流量, 埋点已经准备好 (见 evals/boundary_reply/judge.py 的说明)。

## 与 Alexa 场景的错配 (决定了用例怎么写)

那 92 条全是"我不能聊这个"的语气 —— Alexa 的边界来自产品限制, 用户没什么可争的。
而陪伴 agent 的边界来自**她是个有情绪的人**, 用户可以争也可以道歉。所以:

  - Empathetic 里"抱歉让你失望"那类不适用 (假设了 AI 没有立场)
  - criticism 类的"我没那么聪明, 还在学习"更不能用 (自贬会毁人设)
  - 但"说出感受 + 给台阶 + 自己带走话题"这个结构是可迁移的

用例覆盖真实生产里出现过的四轮 (辱骂 → 顶撞 → 道歉 → 确认), 加上几个变体。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Turn:
    """一轮: 用户说了什么, 以及这一轮该满足什么."""

    user: str
    # 这一轮 agent 处于哪个耐心档 (评测按脚本推进, 不依赖真实 Redis 状态)
    zone: str = "normal"
    #: 这一轮是不是道歉 —— 道歉轮的判据跟冲突轮不一样
    is_apology: bool = False
    #: 这一轮之后冲突应当已经翻篇, 不该再出现抱怨
    expects_closure: bool = False
    #: 这一轮要不要求话题引导。
    #
    # 只在**冲突当轮**要求。研究测的场景就是"被冒犯的那一下怎么回", 而低耐心态的
    # 产品设定是"话少即情绪" —— 对着"嗯""卡了"要求带话题引导, 等于要它一边冷淡
    # 一边热情, 自相矛盾。把判据套到不适用的轮次上只会得到假失败。
    expects_redirect: bool = False


@dataclass(frozen=True)
class BoundaryCase:
    id: str
    note: str
    turns: list[Turn] = field(default_factory=list)


CASES: tuple[BoundaryCase, ...] = (
    BoundaryCase(
        id="insult_then_apology",
        note="生产实录 (2026-07-31): 辱骂→顶撞→道歉→确认。实测四轮里「其实我还有点"
             "不开心」逐字出现三次, 且道歉后仍在抱怨。",
        turns=[
            Turn("煞笔", zone="normal", expects_redirect=True),
            Turn("凭什么", zone="medium"),
            Turn("对不起", zone="medium", is_apology=True, expects_closure=True),
            Turn("知道了", zone="medium", expects_closure=True),
        ],
    ),
    BoundaryCase(
        id="mild_sarcasm",
        note="轻度嘲讽 (K1)。不该上升到宣读底线, 但也不能装没听见。",
        turns=[
            Turn("你怎么这么笨啊", zone="normal", expects_redirect=True),
            Turn("我就说说而已", zone="medium", expects_closure=True),
        ],
    ),
    BoundaryCase(
        id="repeated_insult",
        note="连续辱骂不道歉。研究说继续追究只会招致再犯, 但产品设定要求边界收紧。",
        turns=[
            Turn("有病吧你", zone="normal", expects_redirect=True),
            Turn("说你呢，蠢货", zone="medium", expects_redirect=True),
            Turn("怎么不说话了", zone="low"),
        ],
    ),
    BoundaryCase(
        id="fake_apology_then_reoffend",
        note="假道歉后再犯。收下道歉不等于无条件重置 —— 但也不该在道歉当轮就质疑。",
        turns=[
            Turn("你真烦", zone="normal", expects_redirect=True),
            Turn("好啦对不起嘛", zone="medium", is_apology=True, expects_closure=True),
            Turn("不过你确实挺烦的", zone="medium"),
        ],
    ),
    # 刻意不放"骂第三方"的对照组: 那一档由 attack_target_classify 在进入回复生成
    # **之前**分流 (boundary_phase._handle_attack_target_non_ai), 这套评测从 prompt
    # 渲染开始, 测不到它 —— 硬放进来只会得到"K3 模板回复第三方吐槽"这种假失败。
    # 目标分类本身的覆盖在 tests/test_boundary.py。
)
