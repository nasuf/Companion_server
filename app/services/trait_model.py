"""七维人格模型转换服务。

七维度（0-100）：活泼度、理性度、感性度、计划度、随性度、脑洞度、幽默度
Big Five（0-1）：extraversion、openness、conscientiousness、agreeableness、neuroticism

提供双向映射和统一访问接口。
"""

from __future__ import annotations

from typing import Any

# 七维度名称
SEVEN_DIMS = ["活泼度", "理性度", "感性度", "计划度", "随性度", "脑洞度", "幽默度"]

# 默认七维值（中等偏上，对应 Big Five 全0.5）
_DEFAULT_SEVEN_DIM = {
    "活泼度": 50, "理性度": 50, "感性度": 50,
    "计划度": 50, "随性度": 50, "脑洞度": 50, "幽默度": 50,
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def seven_dim_to_big_five(seven: dict) -> dict:
    """七维(0-100) → Big Five(0-1)。"""
    lively = seven.get("活泼度", 50) / 100
    rational = seven.get("理性度", 50) / 100
    emotional = seven.get("感性度", 50) / 100
    planned = seven.get("计划度", 50) / 100
    spontaneous = seven.get("随性度", 50) / 100
    creative = seven.get("脑洞度", 50) / 100
    # humor not directly used in Big Five

    return {
        "extraversion": _clamp(lively, 0, 1),
        "openness": _clamp(creative, 0, 1),
        "conscientiousness": _clamp(planned, 0, 1),
        "agreeableness": _clamp((emotional + 1 - rational) / 2, 0, 1),
        "neuroticism": _clamp((emotional + spontaneous - planned + 1) / 3, 0, 1),
    }


def big_five_to_seven_dim(big5: dict) -> dict:
    """Big Five(0-1) → 七维(0-100)。"""
    e = big5.get("extraversion", 0.5)
    o = big5.get("openness", 0.5)
    c = big5.get("conscientiousness", 0.5)
    a = big5.get("agreeableness", 0.5)
    n = big5.get("neuroticism", 0.5)

    return {
        "活泼度": round(_clamp(e * 100, 0, 100)),
        "理性度": round(_clamp((1 - a + (1 - n)) * 50, 0, 100)),
        "感性度": round(_clamp((a + n) * 50, 0, 100)),
        "计划度": round(_clamp(c * 100, 0, 100)),
        "随性度": round(_clamp((1 - c) * 100, 0, 100)),
        "脑洞度": round(_clamp(o * 100, 0, 100)),
        "幽默度": round(_clamp((e + o) * 50, 0, 100)),
    }


def get_seven_dim(agent: Any) -> dict:
    """统一获取七维人格。优先 sevenDimTraits/currentTraits，fallback 从 personality 推导。"""
    # 优先 currentTraits（已调整的版本）
    current = getattr(agent, "currentTraits", None)
    if isinstance(current, dict) and current:
        return current

    # 其次 sevenDimTraits（初始设定）
    seven = getattr(agent, "sevenDimTraits", None)
    if isinstance(seven, dict) and seven:
        return seven

    # Fallback: 从 Big Five 推导
    personality = getattr(agent, "personality", None)
    if isinstance(personality, dict) and personality:
        return big_five_to_seven_dim(personality)

    return _DEFAULT_SEVEN_DIM.copy()


def get_dim(traits: dict, dim_name: str) -> float:
    """获取指定维度的值，归一化到0-1。"""
    value = traits.get(dim_name, 50)
    return _clamp(value / 100, 0, 1)
