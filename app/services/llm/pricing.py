"""Qwen Dashscope 模型计价 (元/1M token).

定价来自阿里云百炼帮助中心 (2026 年中国区, 0-128K tier).
我们的 prompt 都在 4-5K 量级, 不进 >128K 的高档阶梯, 所以单档表足够.

调用方按 model_name 查表, 走 estimate_cost_cny() 算钱.
未知 model_name → 返回 0 (不会挂, 但统计会少计 — 加新模型时要更新这里).
"""

from __future__ import annotations

QWEN_PRICING_CNY_PER_1M: dict[str, dict[str, float]] = {
    "qwen3.5-flash": {"input": 0.2, "output": 2.0},
    "qwen3.5-plus": {"input": 0.8, "output": 4.8},
}


def estimate_cost_cny(model: str, input_tokens: int, output_tokens: int) -> float:
    p = QWEN_PRICING_CNY_PER_1M.get(model)
    if not p:
        return 0.0
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


def is_known_model(model: str) -> bool:
    return model in QWEN_PRICING_CNY_PER_1M
