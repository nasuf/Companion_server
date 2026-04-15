"""Personality compatibility shim.

Per spec《产品手册·背景信息》§1.2 MBTI 是 canonical personality 表达。
本模块只剩两个用途：

  1. 接收用户在 agent 创建表单里填的 7 维 (活泼度/理性度/...) 输入，
     `seven_dim_to_signals()` 把它转成 0-1 通用信号，再喂给
     `compute_mbti()` 这个 LLM 调用。
  2. `MBTI_DIMS` 常量 + `is_mbti_dim()` helper 给 trait_adjustment 用。

历史的 7 维↔Big Five 双向转换、`get_seven_dim`、`big_five_to_seven_dim`
都已删除。所有下游模块直接读 `mbti.get_mbti(agent)` 或 `mbti.signal()`。
"""

from __future__ import annotations

# 7 维度名称（仅作为 user input form 字段集合，不再常驻数据库）
SEVEN_DIMS = ["活泼度", "理性度", "感性度", "计划度", "随性度", "脑洞度", "幽默度"]

# MBTI 4 个维度键（trait_adjustment 调整这些）
MBTI_DIMS: tuple[str, ...] = ("EI", "NS", "TF", "JP")


def is_mbti_dim(name: str) -> bool:
    return name in MBTI_DIMS
