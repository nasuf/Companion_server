"""LLM 计价 (元/1M token) — 价格来源 model_registry DB 表.

调用方走 estimate_cost_cny(model, input_tokens, output_tokens) 算钱. 新 usage
行使用 provider-qualified key, 例如 `deepseek/deepseek-v4-pro`.
未知 model_name (registry 没有该 key 或 DB 整体不可达) → 返回 0,
不挂主流程, 但 admin 加新模型后填价格才会被正确计入统计.

价格读 runtime_config._PRICING_CACHE (sync, 由 load_caches 装载, admin
PUT model_registry 后 invalidate_caches 触发重 load — 改完立即生效).
"""

from __future__ import annotations


def estimate_cost_cny(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float:
    """按 (未命中 input + 命中 input + output) 三段计价.

    cached_input_tokens 是 input_tokens 中命中 prefix cache 的部分 (DeepSeek
    命中价 ≈ 未命中 1/40-1/120). registry 未配缓存价时 get_pricing 已回退为
    未命中价 — 计费保守不低估. 传 0 时与旧两段计价完全等价.
    """
    from app.services.runtime_config import get_pricing
    p = get_pricing(model)
    if not p:
        return 0.0
    cached = min(max(int(cached_input_tokens or 0), 0), int(input_tokens or 0))
    miss = int(input_tokens or 0) - cached
    cached_price = p.get("cached_input", p["input"])
    return (
        miss * p["input"] + cached * cached_price + output_tokens * p["output"]
    ) / 1_000_000


def is_known_model(model: str) -> bool:
    from app.services.runtime_config import get_pricing
    return get_pricing(model) is not None
