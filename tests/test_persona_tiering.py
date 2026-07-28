"""建号人设按事实种类分层, 不再整份进 L1.

原来 life_story 给人设档案的每个字段硬编码 ≥0.85, 于是整份角色档案都是 L1 ——
永不衰减、永不被淘汰。拿 423 条真实检索配对逐条判"这条记忆对回复有没有用"
(仅 AI 侧, 已排除来源混淆):

    L1 · 建号人设   n=279   有用率 11% / 20%
    L2 · 聊天学到   n= 87   有用率 29% / 28%
    差 18 个百分点, 置换检验 p=0.0001

最派不上用场的一类, 恰好是唯一永不衰减的。importance ≥0.85 的有用率 (12%) 也低
于 0.70-0.85 档 (37%) —— 分数在高端与有用性反相关。

**这个改动本身不会立刻改善检索。** 拿同一批判定模拟重新分层: 检索池 387 条不变,
有用率 16%/22% 也不变 —— 因为 L1 和 L2 是一起被检索的 (levels=[1,2]), 降级不等于
移出池子。真正的收益要等 L2 动态分级把长期用不到的降到 L3, 那需要约一年。

所以它的定位是**改善的前提**, 不是改善本身。这个文件把这一点也钉住, 免得后来人
以为分层没见效就是白改了。
"""

from __future__ import annotations

import pytest

from app.services.life_story import _tiered_importance
from app.services.memory.config import level_for_importance
from app.services.memory.taxonomy import L1_SINGLETON_SUBS


@pytest.mark.parametrize("main,sub", sorted(L1_SINGLETON_SUBS))
def test_core_identity_stays_permanent(main, sub):
    """会被直接问到、必须永远答得上的事实留在 L1。

    哪些算核心不由这里另立标准 —— 直接用代码库已有的 L1_SINGLETON_SUBS
    (每个子类只能有一条的那些)。
    """
    assert _tiered_importance(main, sub, 0.90) == 0.90
    assert level_for_importance(_tiered_importance(main, sub, 0.90)) == 1


@pytest.mark.parametrize("main,sub,base", [
    ("偏好", "禁忌/雷区", 0.93),
    ("偏好", "生活习惯", 0.88),
    ("思维", "人生观", 0.92),
    ("生活", "工作", 0.88),
    ("身份", "亲属关系", 0.90),
    ("身份", "外貌特征", 0.86),
])
def test_non_core_persona_becomes_decayable(main, sub, base):
    """人设不是错的, 只是可能用不上 —— 放 L2 让它随使用情况慢慢淡出。"""
    scored = _tiered_importance(main, sub, base)
    assert scored < base
    assert level_for_importance(scored) == 2


def test_relative_ordering_is_preserved():
    """原来那些 literal 里编着人工判断 (禁忌 0.93 > 审美 0.86), 降档不该抹平。"""
    taboo = _tiered_importance("偏好", "禁忌/雷区", 0.93)
    looks = _tiered_importance("偏好", "审美厌恶", 0.86)
    assert taboo > looks


def test_nothing_falls_out_of_the_retrieval_pool():
    """降档不能把人设直接踢出常规检索 —— L3 只在唤醒时参与召回。

    实测同一批判定里 0 条"有用"的记忆掉进 L3; 这里守住产生该结果的前提。
    """
    for base in (0.85, 0.86, 0.88, 0.90, 0.92, 0.93, 0.95):
        scored = _tiered_importance("偏好", "生活习惯", base)
        assert level_for_importance(scored) == 2, f"{base} 降过头掉出检索池"


def test_occupation_can_drop_because_the_prompt_anchors_it():
    """职业是最常被问的事实之一, 降到 L2 看着危险, 但它已经硬锚在每条 system
    prompt 里 (prompt_builder 从 agent 行直接拼, 不走记忆)。所以记忆侧降档不会
    让 AI 答不上自己的职业。"""
    scored = _tiered_importance("身份", "职业/与经济", 0.95)
    assert level_for_importance(scored) == 2

    from pathlib import Path
    builder = (Path(__file__).resolve().parent.parent
               / "app/services/chat/prompt_builder.py").read_text()
    assert "你的职业是" in builder, "职业不再进 prompt 的话, 这条降档就不安全了"


class TestLevelDerivationIsShared:
    """层级换算原本散在三处各写各的字面量, 调整规则时漏掉一处, 同一条记忆在不同
    写入路径下会落到不同层。"""

    def test_bands_match_spec(self):
        assert level_for_importance(0.85) == 1
        assert level_for_importance(0.84) == 2
        assert level_for_importance(0.50) == 2
        assert level_for_importance(0.49) == 3

    def test_pipeline_uses_the_shared_helper(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent
               / "app/services/memory/recording/pipeline.py").read_text()
        assert "level_for_importance" in src
        assert "importance >= 0.85" not in src, "又内联了一份换算"

    def test_life_story_uses_the_shared_helper(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent
               / "app/services/life_story.py").read_text()
        assert "level_for_importance" in src
        assert '"level": 1,' not in src, "又把建号记忆硬写成 L1 了"


def test_llm_gap_fill_scores_are_no_longer_clamped():
    """补齐人设缺口时 LLM 给的低分曾被 max(0.85, ...) 抬上去 —— 模型判断这条只值
    0.4 也照样变成永不衰减的核心记忆, 那份判断被白白丢掉。"""
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent
           / "app/services/life_story.py").read_text()
    assert "max(0.85, float(mem.get" not in src
