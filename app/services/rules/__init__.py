"""Deterministic rule catalog for agent control flow.

Rules in this package are pure data/helpers. They must not import chat,
memory, DB, Redis, or LLM modules, so business flows can depend on them
without creating hidden side effects or circular imports.
"""
