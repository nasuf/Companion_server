"""3-Layer Summarizer.

Runs 3 concurrent small-model calls before chat response:
  Layer 1: Conversation review (30 msgs -> 200-300 word summary)
  Layer 2: Memory distillation (memories + current -> 150-200 word key points)
  Layer 3: Current state (5 recent + current -> 100-150 word emotion/topic/intent)

Graceful degradation: if any layer fails, returns empty string for that layer.
Redis caching: results cached for 5 minutes keyed by conversation hash.
"""

import asyncio
import hashlib
import logging

from app.services.cache import cache_summarizer, cache_set_summarizer
from app.services.llm.models import get_summarizer_model, invoke_text
from app.services.prompt_store import get_prompt_text

logger = logging.getLogger(__name__)


async def _run_layer(prompt: str, label: str) -> str:
    """Run a single summarizer layer with error handling."""
    try:
        model = get_summarizer_model()
        result = await invoke_text(model, prompt)
        return result.strip()
    except Exception as e:
        logger.warning(f"Summarizer {label} failed: {e}")
        return ""


def _conv_hash(messages: list[dict], current_message: str) -> str:
    """Create a hash of conversation for cache key."""
    text = current_message + "|" + "|".join(
        m.get("content", "")[-50:] for m in messages[-5:]
    )
    return hashlib.md5(text.encode()).hexdigest()[:16]


async def summarize(
    messages: list[dict],
    current_message: str,
    memories: list[str] | None = None,
) -> dict:
    """Run 3-layer summarizer concurrently.

    Returns dict with keys: review, distillation, state.
    Results are cached in Redis for 5 minutes.
    """
    # Check cache
    ch = _conv_hash(messages, current_message)
    cached = await cache_summarizer(ch)
    if cached:
        logger.debug("Summarizer cache hit")
        return cached

    # Prepare inputs
    # Layer 1: Last 30 messages
    conv_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in messages[-30:]
    )

    # Layer 2: Memories + current
    mem_text = "\n".join(f"- {m}" for m in (memories or [])) or "No stored memories yet."

    # Layer 3: Last 5 messages + current
    recent_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in messages[-5:]
    )

    # Build prompts
    layer1_template, layer2_template, layer3_template = await asyncio.gather(
        get_prompt_text("summarizer.layer1_review"),
        get_prompt_text("summarizer.layer2_distillation"),
        get_prompt_text("summarizer.layer3_state"),
    )
    layer1_prompt = layer1_template.format(conversation=conv_text)
    layer2_prompt = layer2_template.format(
        memories=mem_text, current_message=current_message
    )
    layer3_prompt = layer3_template.format(
        recent=recent_text, current_message=current_message
    )

    # Run all 3 concurrently
    results = await asyncio.gather(
        _run_layer(layer1_prompt, "Layer 1 (review)"),
        _run_layer(layer2_prompt, "Layer 2 (distillation)"),
        _run_layer(layer3_prompt, "Layer 3 (state)"),
        return_exceptions=True,
    )

    output = {
        "review": results[0] if isinstance(results[0], str) else "",
        "distillation": results[1] if isinstance(results[1], str) else "",
        "state": results[2] if isinstance(results[2], str) else "",
    }

    # Cache results
    try:
        await cache_set_summarizer(ch, output)
    except Exception:
        pass

    return output
