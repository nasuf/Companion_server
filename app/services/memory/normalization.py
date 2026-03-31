import logging
import math
from typing import Any
from app.services.memory.taxonomy import TAXONOMY, resolve_taxonomy
from app.services.memory.embedding import generate_embedding

logger = logging.getLogger(__name__)

# Cache for subcategory embeddings to avoid repeated API calls
_SUB_EMBEDDING_CACHE: dict[tuple[str, str], list[float]] = {}

SIMILARITY_THRESHOLD = 0.55

# ── Keyword-based category hints ──
# High-confidence keywords in memory summaries that indicate a specific category.
# Used as a deterministic fallback when taxonomy resolution returns "其他".
# Format: (keyword, main_category, sub_category).
# Sorted by length descending at module load time so longer matches win.
_KEYWORD_HINTS_RAW: list[tuple[str, str, str]] = [
    # 宠物
    ("猫咪", "身份", "宠物"),
    ("猫猫", "身份", "宠物"),
    ("小猫", "身份", "宠物"),
    ("养猫", "身份", "宠物"),
    ("养狗", "身份", "宠物"),
    ("狗狗", "身份", "宠物"),
    ("狗子", "身份", "宠物"),
    ("小狗", "身份", "宠物"),
    ("仓鼠", "身份", "宠物"),
    ("兔子", "身份", "宠物"),
    ("鹦鹉", "身份", "宠物"),
    ("乌龟", "身份", "宠物"),
    ("金鱼", "身份", "宠物"),
    ("遛狗", "生活", "宠物"),
    ("喂猫", "生活", "宠物"),
    ("喂鱼", "生活", "宠物"),
    ("铲屎", "生活", "宠物"),
    ("猫", "身份", "宠物"),
    ("狗", "身份", "宠物"),
    # 亲属关系
    ("爸爸", "身份", "亲属关系"),
    ("妈妈", "身份", "亲属关系"),
    ("父亲", "身份", "亲属关系"),
    ("母亲", "身份", "亲属关系"),
    ("爷爷", "身份", "亲属关系"),
    ("奶奶", "身份", "亲属关系"),
    ("外公", "身份", "亲属关系"),
    ("外婆", "身份", "亲属关系"),
    ("哥哥", "身份", "亲属关系"),
    ("姐姐", "身份", "亲属关系"),
    ("弟弟", "身份", "亲属关系"),
    ("妹妹", "身份", "亲属关系"),
    ("老公", "身份", "亲属关系"),
    ("老婆", "身份", "亲属关系"),
    ("儿子", "身份", "亲属关系"),
    ("女儿", "身份", "亲属关系"),
    # 社会关系
    ("男朋友", "身份", "社会关系"),
    ("女朋友", "身份", "社会关系"),
    ("闺蜜", "身份", "社会关系"),
    ("室友", "身份", "社会关系"),
    # 健康
    ("住院", "生活", "健康"),
    ("体检", "生活", "健康"),
    ("感冒", "生活", "健康"),
    ("发烧", "生活", "健康"),
    ("生病", "生活", "健康"),
    # 居住
    ("搬家", "生活", "居住"),
    ("租房", "生活", "居住"),
    ("买房", "生活", "居住"),
]

_KEYWORD_HINTS: list[tuple[str, str, str]] = sorted(
    _KEYWORD_HINTS_RAW, key=lambda t: len(t[0]), reverse=True
)


def _keyword_hint(summary: str) -> tuple[str, str] | None:
    """Return (main_category, sub_category) if summary contains a high-confidence keyword."""
    if not summary:
        return None
    for keyword, main_cat, sub_cat in _KEYWORD_HINTS:
        if keyword in summary:
            return (main_cat, sub_cat)
    return None

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
    legacy_type: str | None = None,
    summary: str | None = None,
) -> Any:
    """Normalize category using taxonomy first, then keyword hint, then semantic similarity fallback."""
    # 1. Standard resolution (Direct match + Alias map + Contains match)
    res = resolve_taxonomy(
        main_category=main_category,
        sub_category=sub_category,
        legacy_type=legacy_type
    )

    # 2. Keyword hint from summary (deterministic, no LLM/embedding)
    if res.sub_category == "其他" and summary:
        hint = _keyword_hint(summary)
        if hint:
            hint_main, hint_sub = hint
            # Use the hint's sub_category. Prefer the LLM's main_category if the
            # hinted sub exists there; otherwise use the hint's main_category.
            if hint_sub in TAXONOMY.get(res.main_category, ()):
                logger.info(f"Keyword hint: '{summary[:30]}' -> {res.main_category}/{hint_sub}")
                return resolve_taxonomy(
                    main_category=res.main_category,
                    sub_category=hint_sub,
                    legacy_type=legacy_type,
                )
            elif hint_sub in TAXONOMY.get(hint_main, ()):
                logger.info(f"Keyword hint (cross-category): '{summary[:30]}' -> {hint_main}/{hint_sub}")
                return resolve_taxonomy(
                    main_category=hint_main,
                    sub_category=hint_sub,
                    legacy_type=legacy_type,
                )

    # 3. If it's still '其他' but we have an input sub_category, try semantic fallback
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
