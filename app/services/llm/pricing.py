"""LLM 计价 (元/1M token) — 价格来源 model_registry DB 表.

调用方走 estimate_cost_cny(model, input_tokens, output_tokens) 算钱.
未知 model_name (registry 没有该 identifier 或 DB 整体不可达) → 返回 0,
不挂主流程, 但 admin 加新模型后填价格才会被正确计入统计.

价格读 runtime_config._PRICING_CACHE (sync, 由 load_caches 装载, admin
PUT model_registry 后 invalidate_caches 触发重 load — 改完立即生效).
"""

from __future__ import annotations


def estimate_cost_cny(model: str, input_tokens: int, output_tokens: int) -> float:
    from app.services.runtime_config import get_pricing
    p = get_pricing(model)
    if not p:
        return 0.0
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


def is_known_model(model: str) -> bool:
    from app.services.runtime_config import get_pricing
    return get_pricing(model) is not None
