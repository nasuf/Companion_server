import logging
import math
from typing import Any
from app.services.memory.taxonomy import TAXONOMY, resolve_taxonomy
from app.services.memory.embedding import generate_embedding

logger = logging.getLogger(__name__)

# Cache for subcategory embeddings to avoid repeated API calls
_SUB_EMBEDDING_CACHE: dict[tuple[str, str], list[float]] = {}

SIMILARITY_THRESHOLD = 0.65

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    if len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

async def normalize_memory_category(
    main_category: str | None,
    sub_category: str | None,
    legacy_type: str | None = None
) -> Any:
    """Normalize category using taxonomy first, then semantic similarity fallback."""
    # 1. Standard resolution (Direct match + Alias map)
    res = resolve_taxonomy(
        main_category=main_category,
        sub_category=sub_category,
        legacy_type=legacy_type
    )
    
    # 2. If it's still '其他' but we have an input sub_category, try semantic fallback
    input_sub = (sub_category or "").strip()
    if res.sub_category == "其他" and input_sub and input_sub != "其他":
        try:
            allowed_subs = [s for s in TAXONOMY.get(res.main_category, ()) if s != "其他"]
            if not allowed_subs:
                return res

            # Get embedding for the input
            input_vec = await generate_embedding(input_sub)
            
            best_sub = None
            max_sim = -1.0
            
            for allowed in allowed_subs:
                # Get or compute embedding for the allowed category name
                cache_key = (res.main_category, allowed)
                if cache_key not in _SUB_EMBEDDING_CACHE:
                    _SUB_EMBEDDING_CACHE[cache_key] = await generate_embedding(allowed)
                
                allowed_vec = _SUB_EMBEDDING_CACHE[cache_key]
                sim = cosine_similarity(input_vec, allowed_vec)
                
                if sim > max_sim:
                    max_sim = sim
                    best_sub = allowed
            
            if max_sim >= SIMILARITY_THRESHOLD and best_sub:
                logger.info(f"Semantic normalization: '{input_sub}' -> '{best_sub}' (sim: {max_sim:.3f})")
                return resolve_taxonomy(
                    main_category=res.main_category,
                    sub_category=best_sub,
                    legacy_type=legacy_type
                )
                
        except Exception as e:
            logger.warning(f"Semantic normalization failed for '{input_sub}': {e}")
            
    return res
