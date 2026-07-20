"""Memory recall regression eval (Phase 2).

Measures the retrieval layers where tuning actually happens — similarity
threshold, deterministic ranking boosts, protected-slot selection — against a
fixed case bank, so any parameter change can be quantified in seconds instead
of discovered in production.

No DB and no LLM: seed memories + queries are embedded (real bge-m3 via Ollama
for the full run; injectable fake embedder for CI smoke), then the real
`rank_memory_candidate` + `select_context` code paths run in-process.
"""
