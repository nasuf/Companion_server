"""相关度判定 prompt 的性质守卫.

这一步判"弱"就**完全不检索**, 那一轮 AI 手里没有任何关于用户的记忆. 生产基准
(evals/utility_model, 45 条标注用例 × 5 次) 显示原措辞让三个模型都系统性偏弱:

    qwen3.5-flash   84.4%   doubao-2.0-lite 93%   deepseek-v4-flash 81%
    失败全是同一方向 —— 「我今天买了双跑鞋」「你觉得养狗怎么样」判成弱

根因不是规则没写, 而是"弱"的判断特征写成了"任何角色都能立刻回" —— 这个测试
对绝大多数消息都成立, 模型遵循了总则而忽略了下面枚举的场景.

改法是把"弱"变成窄的封闭集合, 并写明代价不对称. 改后:

    qwen 97.8%   doubao-lite 100%   deepseek 94.7%
    deepseek 剩余错判全是中→强 (多查), 属安全方向
"""

from __future__ import annotations

from app.services.prompting.registry import PROMPT_DEFINITION_MAP

KEY = "memory.relevance"


def _text() -> str:
    return PROMPT_DEFINITION_MAP[KEY].default_text


def test_weak_is_defined_as_a_narrow_closed_set():
    """"弱"必须是"只有这几种", 不能是一个宽泛判据."""
    text = _text()
    assert "只有这几种算弱" in text
    assert "很窄的类别" in text


def test_the_misleading_criterion_is_gone():
    """「任何角色都能立刻回」这个判据对几乎所有消息都成立, 是漏检的根源."""
    assert "任何角色都能立刻回" not in _text()


def test_cost_asymmetry_is_stated():
    """漏查一轮 = 整轮失忆; 多查一次 = 几十毫秒. 不写明这一点, 模型会
    对称地权衡两类错误."""
    text = _text()
    assert "拿不准时判中" in text
    assert "代价不对等" in text


def test_the_shapes_that_used_to_fail_are_named():
    """三类曾经稳定判错的形态, 现在必须在 prompt 里被点名为"不是弱"."""
    text = _text()
    section = text.split("反过来", 1)[-1].split("【拿不准", 1)[0]
    for shape in ("买了什么", "你觉得", "情绪表达"):
        assert shape in section, shape


def test_opinion_questions_are_carved_out_of_abstract_chat():
    """「你觉得早起有用吗」问的是 AI 的看法, 不是世间道理 —— 这条边界必须写明,
    否则会被"抽象闲聊"那一档吞掉."""
    assert "世间道理" in _text()


def test_weak_examples_stay_contentless():
    """列进"弱"的清单项本身不能带话题内容, 否则等于给模型反向示范.

    只检查清单项: 这一段里还有"…那就不是弱"这类边界注解, 它们本来就要引用
    带内容的说法作反例.
    """
    text = _text()
    section = text.split("只有这几种算弱", 1)[-1].split("反过来", 1)[0]
    bullets = [
        line for line in section.splitlines()
        if line.strip().startswith("-") and "不是弱" not in line
    ]
    assert bullets, "解析不到弱的清单项, 测试本身失效了"
    for line in bullets:
        for token in ("买", "看过", "去过"):
            assert token not in line, f"弱的清单项不该出现「{token}」: {line.strip()}"
