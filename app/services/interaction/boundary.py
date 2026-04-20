"""边界系统 — AI自我保护与耐心值管理。

耐心值(0-100)控制AI对攻击性消息的反应：
- 70-100: 正常（温和提醒）
- 30-69: 中等（冷淡回应）
- 1-29: 低（警告）
- ≤0: 拉黑（固定模板回复）

5B增强：500+违禁词库、拼音变体检测、正面互动恢复、24h自动解除拉黑。
Redis缓存 + DB持久化，热路径无LLM调用。
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from app.db import db
from app.redis_client import get_redis
from app.services.llm.models import get_chat_model, get_utility_model, invoke_json, invoke_text
from app.services.prompting.store import get_prompt_text
from app.services.prompting.utils import SafeDict, pad_params

logger = logging.getLogger(__name__)

# --- 耐心值状态区间 ---

PATIENCE_MAX = 100
PATIENCE_NORMAL_MIN = 70
PATIENCE_MEDIUM_MIN = 30
PATIENCE_LOW_MIN = 1
PATIENCE_HOURLY_RECOVERY = 10  # spec §2.5: 每小时 +10


def get_patience_zone(patience: int) -> str:
    """返回耐心值所在区间: normal/medium/low/blocked。"""
    if patience >= PATIENCE_NORMAL_MIN:
        return "normal"
    if patience >= PATIENCE_MEDIUM_MIN:
        return "medium"
    if patience >= PATIENCE_LOW_MIN:
        return "low"
    return "blocked"


# --- Redis 耐心值 CRUD ---

def _patience_key(agent_id: str, user_id: str) -> str:
    return f"patience:{agent_id}:{user_id}"


async def get_patience(agent_id: str, user_id: str) -> int:
    """获取当前耐心值。Redis → DB → 默认100。"""
    redis = await get_redis()
    val = await redis.get(_patience_key(agent_id, user_id))
    if val is not None:
        return int(val)

    # Redis miss → 查 DB
    try:
        record = await db.patiencestate.find_unique(
            where={"agentId_userId": {"agentId": agent_id, "userId": user_id}},
        )
        if record:
            # 回填 Redis 缓存
            await redis.set(_patience_key(agent_id, user_id), str(record.value))
            return record.value
    except Exception as e:
        logger.warning(f"DB patience lookup failed: {e}")

    return PATIENCE_MAX


async def set_patience(agent_id: str, user_id: str, value: int) -> int:
    """设置耐心值（clamp到0-100），同步写 Redis + DB。"""
    value = max(0, min(PATIENCE_MAX, value))
    redis = await get_redis()
    await redis.set(_patience_key(agent_id, user_id), str(value))

    # 同步持久化到 DB
    try:
        await db.patiencestate.upsert(
            where={"agentId_userId": {"agentId": agent_id, "userId": user_id}},
            data={
                "create": {
                    "agent": {"connect": {"id": agent_id}},
                    "user": {"connect": {"id": user_id}},
                    "value": value,
                },
                "update": {"value": value},
            },
        )
    except Exception as e:
        logger.warning(f"DB patience persist failed: {e}")

    return value


async def init_patience(agent_id: str, user_id: str) -> int:
    """创建时显式初始化耐心值为100（Redis + DB）。"""
    return await set_patience(agent_id, user_id, PATIENCE_MAX)


async def adjust_patience(agent_id: str, user_id: str, delta: int) -> int:
    """调整耐心值（正数恢复，负数扣除）。返回新值。"""
    current = await get_patience(agent_id, user_id)
    new_val = await set_patience(agent_id, user_id, current + delta)

    # 5B.3: 耐心值归零时启动24h拉黑计时器
    if new_val <= 0 and current > 0:
        redis = await get_redis()
        blacklist_key = f"blacklist_timer:{agent_id}:{user_id}"
        await redis.set(blacklist_key, "1", ex=86400)  # 24h TTL
        logger.info(f"Blacklist timer started: agent={agent_id} user={user_id}")

    return new_val


async def recover_patience_hourly(agent_id: str, user_id: str) -> int:
    """每小时自然恢复耐心值。满值或拉黑时跳过。"""
    redis = await get_redis()
    val = await redis.get(_patience_key(agent_id, user_id))
    if val is None:
        return PATIENCE_MAX  # 无记录=满值，跳过写入
    current = int(val)
    if current <= 0 or current >= PATIENCE_MAX:
        return current
    return await set_patience(agent_id, user_id, current + PATIENCE_HOURLY_RECOVERY)


# --- 5B.1 违禁词库（从JSON加载，500+词） ---

_BANNED_WORDS_PATH = Path(__file__).parent.parent.parent / "data" / "banned_words.json"
_BANNED_CACHE: dict | None = None
_BANNED_KEYWORDS_CACHE: list[str] | None = None
_PINYIN_VARIANTS_CACHE: dict[str, list[str]] | None = None


def _load_banned_words() -> dict:
    """加载违禁词库（带缓存）。"""
    global _BANNED_CACHE
    if _BANNED_CACHE is not None:
        return _BANNED_CACHE

    try:
        with open(_BANNED_WORDS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        _BANNED_CACHE = data
        return data
    except Exception as e:
        logger.warning(f"Failed to load banned words: {e}")
        _BANNED_CACHE = {}
        return {}


def _get_all_banned_keywords() -> list[str]:
    """获取所有违禁关键词（展平，带缓存）。"""
    global _BANNED_KEYWORDS_CACHE
    if _BANNED_KEYWORDS_CACHE is not None:
        return _BANNED_KEYWORDS_CACHE

    data = _load_banned_words()
    keywords: list[str] = []
    for key, value in data.items():
        if key == "pinyin_variants":
            continue
        if isinstance(value, list):
            keywords.extend(value)
    _BANNED_KEYWORDS_CACHE = keywords
    return keywords


def _get_pinyin_variants() -> dict[str, list[str]]:
    """获取拼音变体映射（带缓存）。"""
    global _PINYIN_VARIANTS_CACHE
    if _PINYIN_VARIANTS_CACHE is not None:
        return _PINYIN_VARIANTS_CACHE
    data = _load_banned_words()
    _PINYIN_VARIANTS_CACHE = data.get("pinyin_variants", {})
    return _PINYIN_VARIANTS_CACHE


def check_banned_keywords(message: str) -> list[str]:
    """检查消息中的违禁词，返回命中列表。

    5B.1: 从JSON加载500+词库 + 拼音变体匹配。
    纯计算，用于热路径。
    """
    hits: list[str] = []
    msg_lower = message.lower()

    # 关键词匹配
    for kw in _get_all_banned_keywords():
        if kw in message:
            hits.append(kw)

    # 拼音变体匹配
    for pinyin, meanings in _get_pinyin_variants().items():
        if pinyin in msg_lower:
            hits.append(f"{pinyin}({meanings[0]})")

    return hits


# --- 攻击意图分类（LLM，后台异步） ---


async def classify_attack_intent(message: str) -> dict:
    """LLM分类攻击意图。后台异步调用。"""
    prompt = (await get_prompt_text("boundary.attack_intent")).format(message=message)
    try:
        result = await invoke_json(get_utility_model(), prompt)
        return {
            "intent": result.get("intent", "none"),
            "confidence": float(result.get("confidence", 0.0)),
        }
    except Exception as e:
        logger.warning(f"Attack intent classification failed: {e}")
        return {"intent": "none", "confidence": 0.0}


# --- 攻击严重度分级 ---


async def assess_severity(message: str, intent: str) -> dict:
    """LLM评估攻击严重度。后台异步调用。"""
    prompt = (await get_prompt_text("boundary.severity")).format(message=message, intent=intent)
    try:
        result = await invoke_json(get_utility_model(), prompt)
        level = result.get("level", "L0")
        # 确保扣分在合理范围
        deduction_ranges = {"L0": (5, 10), "L1": (15, 25), "L2": (50, 100)}
        lo, hi = deduction_ranges.get(level, (5, 10))
        deduction = max(lo, min(hi, int(result.get("deduction", lo))))
        return {"level": level, "deduction": deduction, "reason": result.get("reason", "")}
    except Exception as e:
        logger.warning(f"Severity assessment failed: {e}")
        return {"level": "L0", "deduction": 5, "reason": ""}


# --- 24小时重复攻击加重 ---

def _attack_history_key(agent_id: str, user_id: str, level: str | None = None) -> str:
    if level:
        return f"attack_history:{agent_id}:{user_id}:{level}"
    return f"attack_history:{agent_id}:{user_id}"


async def record_attack(agent_id: str, user_id: str, level: str | None = None) -> int:
    """记录攻击事件 24h 过期。若提供 level，同时累计该级别次数。返回同级别当日累计次数。"""
    redis = await get_redis()
    pipe = redis.pipeline()
    pipe.incr(_attack_history_key(agent_id, user_id))
    pipe.expire(_attack_history_key(agent_id, user_id), 86400)
    if level:
        pipe.incr(_attack_history_key(agent_id, user_id, level))
        pipe.expire(_attack_history_key(agent_id, user_id, level), 86400)
    results = await pipe.execute()
    return int(results[-2]) if level else 0


# spec §2.4 基础扣分与上限
# K1=L0: 5 首次, 10 上限；K2=L1: 15 首次, 25 上限；K3=L2: 40 首次, 50 上限
_LEVEL_BASE = {"L0": 5, "L1": 15, "L2": 40}
_LEVEL_CAP = {"L0": 10, "L1": 25, "L2": 50}


def compute_repeat_deduction(level: str, count: int) -> int:
    """spec §2.4: 实际扣除 = ⌈base × (1 + 0.5 × (n-1))⌉，不超过上限。"""
    import math
    base = _LEVEL_BASE.get(level, 5)
    cap = _LEVEL_CAP.get(level, 10)
    n = max(1, count)
    return min(cap, math.ceil(base * (1 + 0.5 * (n - 1))))


# --- 边界反应生成（纯计算，热路径使用） ---

_BOUNDARY_RESPONSES = {
    "normal": [
        "嗯...你这么说我有点难过。",
        "这样说话不太好吧...",
        "我们好好聊天吧～",
    ],
    "medium": [
        "你一直这样我会不想聊天的。",
        "能不能好好说话？",
        "我不太想理你了。",
    ],
    "low": [
        "你再这样我就不回你了。",
        "我已经很不开心了。",
        "最后一次警告。",
    ],
    "blocked": [
        "...",
        "我不想和你说话了。",
    ],
}


def generate_boundary_response(zone: str) -> str:
    """兜底的固定模板回复。仅在 LLM 调用失败时使用。"""
    responses = _BOUNDARY_RESPONSES.get(zone, _BOUNDARY_RESPONSES["normal"])
    return random.choice(responses)


# --- spec §2.6 分级 LLM 回复 ---

_ATTACK_LEVEL_TO_PROMPT = {
    "K1": "boundary.light_attack_reply",
    "K2": "boundary.medium_attack_reply",
    "K3": "boundary.severe_attack_reply",
}

_ZONE_TO_PATIENCE_PROMPT = {
    "medium": "boundary.medium_patience_reply",
    "low": "boundary.low_patience_reply",
    "blocked": "boundary.blacklist_reply",
}


async def generate_boundary_reply_llm(
    *,
    zone: str,
    message: str,
    context: str = "",
    user_emotion: dict | None = None,
    personality_brief: str = "",
    user_portrait: str = "",
    attack_level: str | None = None,
) -> str | None:
    """spec §2.6 用大模型生成分级边界回复。失败时返回 None 让调用方用兜底模板。

    - attack_level 给定（K1/K2/K3）→ 攻击分级回复
    - 否则按 zone（medium/low/blocked）→ 耐心分级回复
    """
    key = _ATTACK_LEVEL_TO_PROMPT.get(attack_level) if attack_level else None
    key = key or _ZONE_TO_PATIENCE_PROMPT.get(zone)
    if not key:
        return None

    params = {
        "message": message,
        "context": context or "(无)",
        "personality_brief": personality_brief or "真诚朋友",
        "user_portrait": user_portrait or "(未知)",
        **pad_params(user_emotion),
    }
    try:
        template = await get_prompt_text(key)
        # 各 prompt 的占位符子集不同，format_map 容错
        prompt = template.format_map(SafeDict(params))
        return (await invoke_text(get_chat_model(), prompt)).strip().split("||")[0][:120]
    except Exception as e:
        logger.warning(f"Boundary LLM reply failed ({key}): {e}")
        return None


# --- 道歉/承诺识别（LLM，后台异步） ---

APOLOGY_KEYWORDS = [
    "对不起", "抱歉", "sorry", "我错了", "不应该", "原谅",
    "道歉", "是我不好", "我不该", "请原谅", "别生气",
    # 隐式道歉表达（流程图要求覆盖更多道歉形式）
    "是我不对", "怪我", "我反省", "我知道错了", "不要生气",
    "消消气", "别气了", "我太过分了", "对不住",
]


def has_apology_keyword(message: str) -> bool:
    """检查消息是否包含道歉关键词。"""
    return any(kw in message for kw in APOLOGY_KEYWORDS)


async def detect_apology(message: str) -> dict:
    """检测道歉意图。后台异步调用。"""
    prompt = (await get_prompt_text("boundary.apology")).format(message=message)
    try:
        result = await invoke_json(get_utility_model(), prompt)
        return {
            "is_apology": bool(result.get("is_apology", False)),
            "sincerity": float(result.get("sincerity", 0.0)),
        }
    except Exception as e:
        logger.warning(f"Apology detection failed: {e}")
        return {"is_apology": False, "sincerity": 0.0}


async def _restore_patience(agent_id: str, user_id: str, delta: int, blocked_floor: int) -> int:
    """恢复耐心值的共享逻辑：非拉黑+delta（上限100），拉黑恢复到blocked_floor。"""
    current = await get_patience(agent_id, user_id)
    if current <= 0:
        new_val = await set_patience(agent_id, user_id, blocked_floor)
        logger.info(
            f"[PATIENCE-DELTA] agent={agent_id} user={user_id} "
            f"reason=apology_unblock current=0 new={new_val}"
        )
        return new_val
    new_val = await set_patience(agent_id, user_id, min(PATIENCE_MAX, current + delta))
    logger.info(
        f"[PATIENCE-DELTA] agent={agent_id} user={user_id} "
        f"reason=restore delta=+{delta} current={current} new={new_val}"
    )
    return new_val


async def handle_apology(agent_id: str, user_id: str) -> int:
    """spec §2.5: 道歉/承诺恢复 +70（上限100），拉黑时直接恢复至70并解除拉黑。

    spec §3.4.4 把"道歉"和"承诺"合并为一个意图，使用同一恢复规则。
    """
    return await _restore_patience(agent_id, user_id, delta=70, blocked_floor=PATIENCE_NORMAL_MIN)


# --- 正面互动恢复耐心 ---


async def check_positive_recovery(
    agent_id: str,
    user_id: str,
) -> int | None:
    """spec §2.5: 正向互动恢复 +20（上限100），耐心值≤0 时不生效。"""
    current = await get_patience(agent_id, user_id)
    if current <= 0:
        return None
    if current >= PATIENCE_MAX:
        return current

    new_val = await set_patience(agent_id, user_id, current + 20)
    logger.info(f"Positive recovery: agent={agent_id} user={user_id} +20 → {new_val}")
    return new_val


# --- 5B.3 拉黑24h自动解除 ---

async def check_blacklist_expiry(agent_id: str, user_id: str) -> bool:
    """检查拉黑计时器是否已过期。

    由 scheduler 定期调用。如果 blacklist_timer key 已过期且耐心值≤0，
    恢复耐心值到30。
    返回是否执行了恢复。
    """
    redis = await get_redis()
    blacklist_key = f"blacklist_timer:{agent_id}:{user_id}"

    # 如果计时器还在，未过期
    if await redis.exists(blacklist_key):
        return False

    # 计时器已过期，检查是否仍然拉黑
    current = await get_patience(agent_id, user_id)
    if current > 0:
        return False  # 已经不是拉黑状态了

    # 恢复到30点（low区间，作为缓冲）
    await set_patience(agent_id, user_id, 30)
    logger.info(f"Blacklist auto-lifted: agent={agent_id} user={user_id} → 30")
    return True


async def scan_blacklist_expiry() -> int:
    """扫描所有拉黑计时器，解除已过期的。

    由 scheduler 每5分钟调用。
    返回解除的数量。
    """
    redis = await get_redis()
    count = 0

    # 扫描所有 patience key 值为 0 的
    cursor = 0
    while True:
        cursor, keys = await redis.scan(cursor, match="patience:*", count=100)
        for key in keys:
            val = await redis.get(key)
            if val is not None and int(val) <= 0:
                # 解析 agent_id 和 user_id
                parts = key if isinstance(key, str) else key.decode()
                segments = parts.split(":")
                if len(segments) == 3:
                    _, agent_id, user_id = segments
                    if await check_blacklist_expiry(agent_id, user_id):
                        count += 1
        if cursor == 0:
            break

    return count


# --- 5B.4 耐心区间描述（供Prompt注入） ---

def get_patience_prompt_instruction(patience: int) -> str | None:
    """5B.4 根据耐心区间返回语气描述，注入系统提示。

    - normal (70-100): 无额外描述
    - medium (30-69): "你对用户有些不满"
    - low (1-29): "你非常不开心"
    - blocked (≤0): 用户正在道歉（只有道歉消息才能到达此处）
    """
    zone = get_patience_zone(patience)
    if zone == "blocked":
        return (
            "你之前因为用户的冒犯非常生气，已经不想理用户了。"
            "现在用户在向你道歉。你可以表现得还在生气、委屈，"
            "但不要完全无视。根据你的性格决定是否接受道歉，回复要简短。"
        )
    if zone == "medium":
        return "你对用户有些不满，因为用户之前说了一些不好听的话。你的回复会带点冷淡，不那么热情。"
    elif zone == "low":
        return "你非常不开心，用户之前对你很不友善。你的回复非常简短冷漠，明确表达你的不满。"
    return None


# --- 热路径入口：边界检查 ---

async def check_boundary(
    agent_id: str, user_id: str, message: str,
) -> tuple[dict | None, int]:
    """热路径边界检查（关键词匹配 + Redis 读取，无 LLM）。

    返回 (signal, patience)。signal 为 None 表示通过；否则包含：
      - blocked: 是否拉黑（没耐心≤0 且非道歉）
      - zone: normal / medium / low / blocked
      - hits: 命中的违禁关键词列表
      - fallback: 兜底文案（LLM 生成失败时使用）

    spec §2.6：
    - 拉黑状态：拦截非道歉消息；道歉消息放行进入正常流程（§2.2）
    - 正常/中/低：仅在含违禁词时拦截（违禁词作为步骤 3 的硬关键词兜底）
    """
    patience = await get_patience(agent_id, user_id)
    if patience <= 0:
        # 道歉消息放行，进入正常聊天流程（后台任务恢复耐心）
        if has_apology_keyword(message):
            return None, patience
        return {
            "blocked": True,
            "zone": "blocked",
            "hits": [],
            "fallback": generate_boundary_response("blocked"),
        }, patience

    hits = check_banned_keywords(message)
    if not hits:
        return None, patience

    zone = get_patience_zone(patience)
    return {
        "blocked": False,
        "zone": zone,
        "hits": hits,
        "fallback": generate_boundary_response(zone),
    }, patience


# --- 后台异步处理：扣分 + 攻击记录 ---

async def process_boundary_violation(
    agent_id: str, user_id: str, message: str,
) -> None:
    """后台处理边界违规：分类→分级→扣分→记录。"""
    # 已拉黑用户跳过LLM调用
    current = await get_patience(agent_id, user_id)
    if current <= 0:
        return

    intent_result = await classify_attack_intent(message)
    intent = intent_result.get("intent", "none")
    confidence = intent_result.get("confidence", 0.0)

    if intent == "none" or confidence < 0.8:
        return

    if intent == "attack_third":
        # 攻击第三方不扣耐心值
        return

    severity = await assess_severity(message, intent)
    level = severity.get("level", "L0")

    # spec §2.4: 先记录累计次数，按 (1 + 0.5×(n-1)) 公式并按级别上限封顶
    count = await record_attack(agent_id, user_id, level=level)
    deduction = compute_repeat_deduction(level, count if count > 0 else 1)

    new_val = await adjust_patience(agent_id, user_id, -deduction)

    logger.info(
        f"[PATIENCE-DELTA] agent={agent_id} user={user_id} "
        f"reason=violation intent={intent} level={level} count={count} "
        f"delta=-{deduction} current={current} new={new_val}"
    )
