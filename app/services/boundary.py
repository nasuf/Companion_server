"""边界系统 — AI自我保护与耐心值管理。

耐心值(0-100)控制AI对攻击性消息的反应：
- 70-100: 正常（温和提醒）
- 30-69: 中等（冷淡回应）
- 1-29: 低（警告）
- ≤0: 拉黑（固定模板回复）

5B增强：500+违禁词库、拼音变体检测、正面互动恢复、24h自动解除拉黑。
纯计算 + Redis，热路径无LLM调用。
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from app.redis_client import get_redis
from app.services.llm.models import get_utility_model, invoke_json

logger = logging.getLogger(__name__)

# --- 耐心值状态区间 ---

PATIENCE_MAX = 100
PATIENCE_NORMAL_MIN = 70
PATIENCE_MEDIUM_MIN = 30
PATIENCE_LOW_MIN = 1
PATIENCE_HOURLY_RECOVERY = 5


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
    """获取当前耐心值，默认100。"""
    redis = await get_redis()
    val = await redis.get(_patience_key(agent_id, user_id))
    if val is None:
        return PATIENCE_MAX
    return int(val)


async def set_patience(agent_id: str, user_id: str, value: int) -> int:
    """设置耐心值（clamp到0-100）。"""
    value = max(0, min(PATIENCE_MAX, value))
    redis = await get_redis()
    await redis.set(_patience_key(agent_id, user_id), str(value))
    return value


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

_BANNED_WORDS_PATH = Path(__file__).parent.parent / "data" / "banned_words.json"
_BANNED_CACHE: dict | None = None


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
    """获取所有违禁关键词（展平）。"""
    data = _load_banned_words()
    keywords: list[str] = []
    for key, value in data.items():
        if key == "pinyin_variants":
            continue
        if isinstance(value, list):
            keywords.extend(value)
    return keywords


def _get_pinyin_variants() -> dict[str, list[str]]:
    """获取拼音变体映射。"""
    data = _load_banned_words()
    return data.get("pinyin_variants", {})


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

_ATTACK_INTENT_PROMPT = """分析以下消息的攻击意图。

消息："{message}"

分类为以下之一：
1. attack_ai — 直接攻击/侮辱AI
2. attack_third — 攻击第三方（不针对AI）
3. profanity_no_target — 无目标脏话/发泄
4. none — 无负面意图

返回JSON：
{{"intent": "attack_ai/attack_third/profanity_no_target/none", "confidence": 0.0-1.0}}"""


async def classify_attack_intent(message: str) -> dict:
    """LLM分类攻击意图。后台异步调用。"""
    prompt = _ATTACK_INTENT_PROMPT.format(message=message)
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

_SEVERITY_PROMPT = """评估以下攻击性消息的严重程度。

消息："{message}"
攻击意图：{intent}

分级：
- L0: 轻微（不耐烦/轻微不满） → 扣5-10点
- L1: 中等（明确侮辱/攻击） → 扣15-25点
- L2: 严重（极端侮辱/威胁/人身攻击） → 扣50点或归零

返回JSON：
{{"level": "L0/L1/L2", "deduction": 5-100, "reason": "简短原因"}}"""


async def assess_severity(message: str, intent: str) -> dict:
    """LLM评估攻击严重度。后台异步调用。"""
    prompt = _SEVERITY_PROMPT.format(message=message, intent=intent)
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

def _attack_history_key(agent_id: str, user_id: str) -> str:
    return f"attack_history:{agent_id}:{user_id}"


async def check_repeat_attack(agent_id: str, user_id: str) -> bool:
    """检查24h内是否有攻击记录。"""
    redis = await get_redis()
    count = await redis.get(_attack_history_key(agent_id, user_id))
    return int(count or 0) > 0


async def record_attack(agent_id: str, user_id: str) -> None:
    """记录攻击事件，24h过期。"""
    redis = await get_redis()
    key = _attack_history_key(agent_id, user_id)
    await redis.incr(key)
    await redis.expire(key, 86400)


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
    """根据耐心区间生成边界回复。纯计算。"""
    responses = _BOUNDARY_RESPONSES.get(zone, _BOUNDARY_RESPONSES["normal"])
    return random.choice(responses)


# --- 道歉/承诺识别（LLM，后台异步） ---

APOLOGY_KEYWORDS = [
    "对不起", "抱歉", "sorry", "我错了", "不应该", "原谅",
    "道歉", "是我不好", "我不该", "请原谅", "别生气",
]

_APOLOGY_PROMPT = """分析以下消息是否包含道歉或承诺改正。

消息："{message}"

返回JSON：
{{"is_apology": true/false, "sincerity": 0.0-1.0}}"""


async def detect_apology(message: str) -> dict:
    """检测道歉意图。后台异步调用。"""
    prompt = _APOLOGY_PROMPT.format(message=message)
    try:
        result = await invoke_json(get_utility_model(), prompt)
        return {
            "is_apology": bool(result.get("is_apology", False)),
            "sincerity": float(result.get("sincerity", 0.0)),
        }
    except Exception as e:
        logger.warning(f"Apology detection failed: {e}")
        return {"is_apology": False, "sincerity": 0.0}


async def handle_apology(agent_id: str, user_id: str) -> int:
    """处理道歉：非拉黑恢复到60点，拉黑恢复到70点。"""
    current = await get_patience(agent_id, user_id)
    if current <= 0:
        return await set_patience(agent_id, user_id, PATIENCE_NORMAL_MIN)
    return await set_patience(agent_id, user_id, max(current, 60))


# --- 5B.2 正面互动恢复耐心 ---

def _positive_streak_key(agent_id: str, user_id: str) -> str:
    return f"positive_streak:{agent_id}:{user_id}"


async def check_positive_recovery(
    agent_id: str,
    user_id: str,
    is_positive: bool,
) -> int | None:
    """5B.2 正面互动恢复耐心值。

    连续3次正面消息（无违禁词且情绪正面）→ 恢复+5~10点。
    耐心值≤0（拉黑）时不生效。
    返回新耐心值，或None（未触发恢复）。
    """
    redis = await get_redis()
    key = _positive_streak_key(agent_id, user_id)

    if not is_positive:
        await redis.set(key, "0", ex=86400)
        return None

    # 递增正面计数
    streak = int(await redis.get(key) or 0) + 1
    await redis.set(key, str(streak), ex=86400)

    if streak < 3:
        return None

    # 连续3次正面 → 恢复
    current = await get_patience(agent_id, user_id)
    if current <= 0:
        return None  # 拉黑时不生效

    if current >= PATIENCE_MAX:
        return current  # 已满值

    recovery = random.randint(5, 10)
    new_val = await set_patience(agent_id, user_id, current + recovery)

    # 重置计数
    await redis.set(key, "0", ex=86400)

    logger.info(
        f"Positive recovery: agent={agent_id} user={user_id} "
        f"+{recovery} → {new_val}"
    )
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
    - blocked (≤0): 不会到这里（直接短路模板回复）
    """
    zone = get_patience_zone(patience)
    if zone == "medium":
        return "你对用户有些不满，因为用户之前说了一些不好听的话。你的回复会带点冷淡，不那么热情。"
    elif zone == "low":
        return "你非常不开心，用户之前对你很不友善。你的回复非常简短冷漠，明确表达你的不满。"
    return None


# --- 热路径入口：边界检查 ---

async def check_boundary(
    agent_id: str, user_id: str, message: str,
) -> dict | None:
    """热路径边界检查。

    返回 {"blocked": True, "response": str, "zone": str} 或 None（通过）。
    纯关键词匹配 + Redis读取，无LLM调用。
    """
    hits = check_banned_keywords(message)
    if not hits:
        return None

    patience = await get_patience(agent_id, user_id)
    zone = get_patience_zone(patience)

    if zone == "blocked":
        return {
            "blocked": True,
            "response": generate_boundary_response("blocked"),
            "zone": zone,
        }

    return {
        "blocked": False,
        "response": generate_boundary_response(zone),
        "zone": zone,
        "hits": hits,
    }


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

    if intent == "none" or confidence < 0.6:
        return

    if intent == "attack_third":
        # 攻击第三方不扣耐心值
        return

    severity = await assess_severity(message, intent)
    deduction = severity.get("deduction", 5)

    # 24h重复攻击加重50%
    if await check_repeat_attack(agent_id, user_id):
        deduction = int(deduction * 1.5)

    await adjust_patience(agent_id, user_id, -deduction)
    await record_attack(agent_id, user_id)

    logger.info(
        f"Boundary violation: agent={agent_id} intent={intent} "
        f"severity={severity.get('level')} deduction={deduction}"
    )
