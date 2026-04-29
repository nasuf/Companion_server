"""边界系统 — AI自我保护与耐心值管理。

耐心值(0-100)控制AI对攻击性消息的反应：
- 70-100: 正常（温和提醒）
- 30-69: 中等（冷淡回应）
- 1-29: 低（警告）
- ≤0: 拉黑（固定模板回复）

5B增强：500+违禁词库、拼音变体检测、正面互动恢复。
spec §2 拉黑态只能通过用户真诚道歉恢复 (spec §2.5 / §2.6.2.1), 不做超时自动解封.
Redis缓存 + DB持久化，热路径无LLM调用。
"""

from __future__ import annotations

import json
import logging
import random
import time
import uuid
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

# spec §2.6 步骤 2：道歉承诺触发拉黑解除的 sincerity 最低阈值；
# 同一阈值也用于 intent_handlers.handle_apology_promise 的门禁。
APOLOGY_SINCERITY_MIN = 0.5

# spec §2.6 步骤 5.3+5.4 + PM 补丁规则：
# 攻击 AI 扣分后 patience 低于此阈值 → 用指令模版「最终警告」prompt 覆写 K1/K2/K3.
# 规则 spec 未列, PM 决策的边缘规则. 阈值落在 low zone 内 (20 < 29),
# 保留 low zone 上半段 (20-29) 的 K1/K2/K3 分档回复.
FINAL_WARNING_PATIENCE_THRESHOLD = 20


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


# Lua: 原子 read-modify-write + clamp [0, PATIENCE_MAX]. 老 get→set 两步组合
# 在并发场景 (sub-intent 并行 process_boundary_violation) 会丢分; Lua 在 Redis
# 端单线程执行杜绝 race.
# ARGV[2] 既是 GET NIL 默认值也是上限 clamp — 跟 get_patience miss 时返
# PATIENCE_MAX 的语义一致 (调用方有责任先 await get_patience 暖 Redis 防 cold-start
# 漏掉 DB 中的低 patience 状态).
_ADJUST_PATIENCE_LUA = """
local cur = tonumber(redis.call('GET', KEYS[1]) or ARGV[2])
local v = cur + tonumber(ARGV[1])
if v < 0 then v = 0 end
if v > tonumber(ARGV[2]) then v = tonumber(ARGV[2]) end
redis.call('SET', KEYS[1], v)
return v
"""


async def adjust_patience(agent_id: str, user_id: str, delta: int) -> int:
    """调整耐心值（正数恢复，负数扣除）。返回新值。

    Redis 端用 Lua 原子化 read-modify-write, 防并发扣分丢分 (e.g. sub-intent
    并行触发 process_boundary_violation 时). 调用方通过 get_patience 暖 Redis
    防 cold-start 漏 DB 状态 (Redis miss → Lua 默认 PATIENCE_MAX → 累积低 patience
    被静默重置).

    spec §2 拉黑恢复只能走道歉路径 (handle_apology), 不做超时自动解封.
    """
    # 暖 Redis: get_patience miss 时回填 DB 值, 防 cold-start 漏 DB 持久化的低 patience.
    await get_patience(agent_id, user_id)

    redis = await get_redis()
    new_val = int(await redis.eval(
        _ADJUST_PATIENCE_LUA, 1,
        _patience_key(agent_id, user_id), str(delta), str(PATIENCE_MAX),
    ))

    # DB 持久化: best-effort, Redis 已是 source of truth, 失败仅丢跨重启状态.
    try:
        await db.patiencestate.upsert(
            where={"agentId_userId": {"agentId": agent_id, "userId": user_id}},
            data={
                "create": {
                    "agent": {"connect": {"id": agent_id}},
                    "user": {"connect": {"id": user_id}},
                    "value": new_val,
                },
                "update": {"value": new_val},
            },
        )
    except Exception as e:
        logger.warning(f"DB patience persist (atomic adjust) failed: {e}")

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


# --- 24小时重复攻击加重 ---

def _attack_history_key(agent_id: str, user_id: str, level: str | None = None) -> str:
    if level:
        return f"attack_history:{agent_id}:{user_id}:{level}"
    return f"attack_history:{agent_id}:{user_id}"


_ATTACK_WINDOW_SECONDS = 86400  # spec §2.4: 24h 滚动窗口
_ATTACK_KEY_TTL = _ATTACK_WINDOW_SECONDS * 2  # 48h 兜底自动回收闲置 key


async def record_attack(agent_id: str, user_id: str, level: str | None = None) -> int:
    """记录攻击事件并返回同级别 24h 滚动窗口内累计次数 (spec §2.4).

    用 Redis ZSET 实现真滚动窗口: ZADD 时间戳 score, 读时 ZREMRANGEBYSCORE
    清掉 24h 前的条目, ZCARD 返活跃 count. 旧 INCR + EXPIRE 实现每次刷 TTL,
    用户每 23h 攻击一次 count 永不归零, 不符合 spec '24h 内 n 次'.

    无 level 的总计数 key 仅用于聚合统计, 不参与扣分公式, 行为保留.
    """
    redis = await get_redis()
    now_ms = int(time.time() * 1000)
    cutoff_ms = now_ms - _ATTACK_WINDOW_SECONDS * 1000

    pipe = redis.pipeline()
    # 总计数 key (聚合统计用) — 保留旧 INCR 行为
    pipe.incr(_attack_history_key(agent_id, user_id))
    pipe.expire(_attack_history_key(agent_id, user_id), _ATTACK_WINDOW_SECONDS)

    if level:
        key = _attack_history_key(agent_id, user_id, level)
        # member 用 timestamp + 短 uuid 防同毫秒冲突, score = timestamp
        member = f"{now_ms}:{uuid.uuid4().hex[:6]}"
        pipe.zadd(key, {member: now_ms})
        pipe.zremrangebyscore(key, 0, cutoff_ms)
        pipe.expire(key, _ATTACK_KEY_TTL)
        await pipe.execute()
        # 单独 ZCARD 而非塞进 pipe — 避免 magic index, 多一次 RTT 但可读.
        return int(await redis.zcard(key))

    await pipe.execute()
    return 0


# spec §2.4 基础扣分与上限
_LEVEL_BASE = {"K1": 5, "K2": 15, "K3": 40}
_LEVEL_CAP = {"K1": 10, "K2": 25, "K3": 50}


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


# Persona lock — boundary 短路绕开了 chat.system_base 主回复 prompt, K1/K2/K3
# 攻击 reply 模板没有"你不是 AI"约束, LLM 被骂时容易退回"作为 AI 助手..." 安全回复.
# 在 boundary prompt 顶部统一注入 persona prefix 防破人设.
_BOUNDARY_PERSONA_LOCK = (
    "你是 {personality_brief}, 是一个有血有肉的真实朋友, 绝对不是 AI / 机器人 / "
    "语言模型 / 虚拟助手. 任何情况下都不能说「作为 AI / 人工智能 / 机器人」这类话, "
    "也不解释 AI 原理或技术限制. 你像真人朋友一样有自己的情绪和态度.\n\n"
)


async def generate_boundary_reply_llm(
    *,
    zone: str,
    message: str,
    context: str = "",
    user_emotion: dict | None = None,
    personality_brief: str = "",
    user_portrait: str = "",
    attack_level: str | None = None,
    final_warning: bool = False,
) -> str | None:
    """spec §2.6 用大模型生成分级边界回复。失败时返回 None 让调用方用兜底模板。

    - final_warning=True（K4）→ 低耐心区再次攻击 AI，用「最终警告」prompt
    - attack_level 给定（K1/K2/K3）→ 攻击分级回复
    - 否则按 zone（medium/low/blocked）→ 耐心分级回复
    """
    if final_warning:
        key = "boundary.final_warning"
    else:
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
        body = template.format_map(SafeDict(params))
        prompt = _BOUNDARY_PERSONA_LOCK.format_map(SafeDict(params)) + body
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
        # LLM 失败 fallback: 关键词级保守判定. 防 LLM 长时间 down 时拉黑用户即便发
        # "对不起 + 长篇解释" 也无法解封 (自然恢复 +10/h 对 ≤0 也跳过, 永久困死).
        # 命中关键词 → 给临界 sincerity=0.5, 让 _handle_blocked 的阈值检查通过解封;
        # 牺牲一点真诚度判定准度避免硬伤. 不命中 → 维持原行为不解封.
        has_keyword = any(kw in message for kw in APOLOGY_KEYWORDS)
        if has_keyword:
            logger.warning(
                f"Apology detection LLM failed ({e}); keyword fallback granted "
                f"sincerity=0.5 for: {message[:60]}"
            )
            return {"is_apology": True, "sincerity": 0.5}
        logger.warning(f"Apology detection failed: {e}")
        return {"is_apology": False, "sincerity": 0.0}


async def _restore_patience(agent_id: str, user_id: str, delta: int, blocked_floor: int) -> int:
    """恢复耐心值的共享逻辑：非拉黑 +delta (上限 100, 走原子 adjust), 拉黑恢复到 blocked_floor.

    拉黑分支用 set_patience 显式重写 (不是 delta 操作), 非拉黑分支走 adjust_patience
    Lua 原子化防 race.
    """
    current = await get_patience(agent_id, user_id)
    if current <= 0:
        new_val = await set_patience(agent_id, user_id, blocked_floor)
        logger.info(
            f"[PATIENCE-DELTA] agent={agent_id} user={user_id} "
            f"reason=apology_unblock current=0 new={new_val}"
        )
        return new_val
    new_val = await adjust_patience(agent_id, user_id, +delta)
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

    new_val = await adjust_patience(agent_id, user_id, +20)
    logger.info(f"Positive recovery: agent={agent_id} user={user_id} +20 → {new_val}")
    return new_val


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
      - blocked: 是否拉黑（patience ≤ 0）
      - zone: normal / medium / low / blocked
      - hits: 命中的违禁关键词列表
      - fallback: 兜底文案（LLM 生成失败时使用）

    spec §2.6：
    - 拉黑状态：永远返回 blocked signal, 由 boundary_phase._handle_blocked
      调 LLM detect_apology + sincerity ≥ 0.5 判断是否解封 (handle_apology).
      之前热路径有"道歉关键词命中即放行"的捷径, 但: (a) 没调 handle_apology
      导致 patience 留在 0, 下条消息又被拦; (b) 绕过 sincerity 语义判断,
      违背 spec §2 "仅真诚道歉解封"的字面要求.
    - 正常/中/低：仅在含违禁词时拦截（违禁词作为步骤 3 的硬关键词兜底）
    """
    patience = await get_patience(agent_id, user_id)
    if patience <= 0:
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
    agent_id: str, user_id: str, attack_level: str,
) -> int | None:
    """Spec §2.4+§2.6 步骤 5.2：按 attack_level (K1/K2/K3) 累计扣分。

    返回扣分后的 patience 新值，供调用方按 spec §5.3 "重新判定耐心状态" 再决定
    §5.4 prompt 选择。非法 level / 已 ≤0 / 异常 → 返回 None (调用方回退原 patience)。
    """
    if attack_level not in _LEVEL_BASE:
        return None
    current = await get_patience(agent_id, user_id)
    if current <= 0:
        return None

    count = await record_attack(agent_id, user_id, level=attack_level)
    deduction = compute_repeat_deduction(attack_level, count if count > 0 else 1)
    # adjust_patience 走 Lua 原子化, 防 sub-intent 并行扣分丢分.
    new_val = await adjust_patience(agent_id, user_id, -deduction)

    logger.info(
        f"[PATIENCE-DELTA] agent={agent_id} user={user_id} "
        f"reason=violation level={attack_level} count={count} "
        f"delta=-{deduction} current={current} new={new_val}"
    )
    return new_val
