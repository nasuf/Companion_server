"""Keep generated scenes and self memories subordinate to the Agent profile.

The memory table's owner identifies whose experience it is, not the subject of
every fact in that experience. In particular, remembering a user's location
does not put the Agent there. Classification labels are not a trust boundary.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Awaitable, Callable

from app.services.llm.models import get_grounding_model, invoke_json
from app.services.prompting.store import PromptDisabledError, get_prompt_text

logger = logging.getLogger(__name__)

# These are eligibility gates for semantic verification, not a city dictionary
# or a semantic verdict. The verifier also receives the complete candidate.
_LOCATION = re.compile(
    r"住|居住|定居|搬家|现居|老家|家在|回家|到家|所在地|城市|街|路|巷|区|镇|"
    r"旅行|旅游|出差|来这|在这|这里|那边|附近|出门|散步|走到|漫步|醒来|园|天地|参观|游览|游玩|途经|前往|抵达|去往|拜访|"
    r"\b(?:live|living|home|moved|street|road|city|travel|arrived)\b", re.I,
)
_SELF = re.compile(
    r"我|自己|^\s*(?:清晨|早上|早晨|上午|中午|午后|下午|傍晚|晚上|夜间|凌晨|今天|昨日|昨天)|"
    r"^\s*(?:刚刚|刚|正|已经|还)?(?:走到|走在|路过|在|到|去|来|回到|漫步|醒来|住在)|"
    r"\b(?:I|my|me)\b", re.I,
)
_PHYSICAL = re.compile(
    r"(?:我(?:们)?|自己)(?:现在|刚刚|刚|正|目前|已经|一直|就|也)?(?:在|来到|到了|去了|去|来|到|待在|呆在|身处)|"
    r"^\s*(?:刚刚|刚|正|已经)?(?:在|来到|到了|去了|待在|呆在|身处)|"
    r"\bI\s+(?:am\s+)?(?:in|at|visiting)\b", re.I,
)
_GENERIC_PHYSICAL = re.compile(
    r"^(?:我|自己)?(?:现在|刚刚|刚|正|目前|已经|一直|就|也)?(?:在|回到|待在|呆在)"
    r"(?:吃|喝|想|处理|做|看|写|睡|忙|整理|学习|休息|家|公司|办公室|工作室|目前|现有|能|这些|记忆|对话|记录)", re.I,
)
_STABLE_SELF = re.compile(
    r"(?:我|^自己)(?:现在|目前|本来|一直|其实|就|也|才|今年|已经|还|刚|刚刚|真的|确实|不|没|是|在|的|家)*"
    r"(?:住(?:在)?(?!的|得|了|着|过)|居住|定居|现居|搬到|搬来|家在|叫(?:做)?(?!了|外卖|车|你|他|她)|名叫|姓|职业是|工作是|是\d{1,3}岁|\d{1,3}岁)|"
    r"(?:我|自己)的?家(?:现在|本来|一直|就|也)?在|"
    r"^(?:现居地|居住地|职业|姓名|年龄)[：:是]|"
    r"\bI\s+(?:live|reside|am\s+\d{1,3}\s+years\s+old)\b|\bmy\s+(?:home|name)\s+is\b", re.I,
)


# The hot-path budget: a reply guard runs while the user waits.
VERIFY_TIMEOUT_S = 5
# Paths that do not pass the user's location (memories, schedules, proactive
# messages) keep the single-sided check; the prompt skips the user rule.
USER_LOCATION_UNCHECKED = "（本次不校验对方位置）"
_ABOUT = frozenset({"self", "user"})


class GroundingUnavailable(RuntimeError):
    """A risky generated claim has not passed verification; retry is possible."""


class Rejected(list):
    """Rejected indices — a plain list for existing callers — plus whose
    location each one got wrong ("self" = the Agent, "user" = the user)."""

    def __init__(self, subjects: dict[int, str] | None = None):
        super().__init__(subjects or {})
        self.subjects = dict(subjects or {})


def canonical_facts(agent: Any) -> dict[str, Any]:
    facts = {}
    # nickname: 聊天热路径不再常驻注入 L1 记忆 (纯检索), 姓名记忆里的「你可以叫我…」
    # 未必每轮都被召回; 放进这份每轮都注入的档案, 用户叫昵称时 AI 才认得出自己。
    for name in ("name", "nickname", "city", "occupation"):
        value = getattr(agent, name, None)
        if isinstance(value, str) and value.strip():
            facts[name] = value.strip()
    age = getattr(agent, "age", None)
    if isinstance(age, int) and not isinstance(age, bool) and age > 0:
        facts["age"] = age
    return facts


def generated_stable_self_claim(text: str) -> bool:
    """Stable facts must come from profile provisioning, irrespective of label.

    User-subject interaction memories remain valid. Splitting into clauses
    prevents a user mention in one clause from hiding an Agent claim in another.
    """
    for clause in re.split(r"[，,。！？!?；;\n]", text):
        if re.search(r"^(?:用户|对方|你)(?:向我|告诉我|说|分享|现在|住|家|的)", clause.strip()):
            continue
        if _STABLE_SELF.search(clause.strip()):
            return True
    return False


def needs_location_verification(text: str, *, implicit_self: bool = False) -> bool:
    if (re.match(r"\s*用户(?:向我分享了当前位置|当前在|告诉我|说自己|住在)", text)
            and not re.search(r"[，,。；;]\s*我|我(?:也|刚|正|在|住)", text)):
        return False
    # Conversational presence is not a physical-location assertion. Strip only
    # this clause, so an accompanying real city claim still gets verified.
    location_text = re.sub(r"我(?:会|一直|就)?在(?:这儿|这里|你身边)(?:陪着你|陪你|听你说)", "", text)
    location_text = re.sub(r"我(?:这里|这儿)(?=没(?:有)?(?:看到|找到|记录|记得))", "", location_text)
    location_text = re.sub(r"我在(?=[，。！？!?；;]|$)", "", location_text)
    physical = _PHYSICAL.search(location_text) and not _GENERIC_PHYSICAL.search(location_text)
    return bool((_LOCATION.search(location_text) or physical) and (implicit_self or _SELF.search(text)))


async def workspace_agent(user_id: str, workspace_id: str | None) -> Any:
    """Resolve by the memory's workspace, never by the user's latest Agent."""
    from app.db import db
    if not workspace_id:
        raise GroundingUnavailable("A generated location needs an explicit workspace")
    workspace = await db.chatworkspace.find_first(
        where={"id": workspace_id, "userId": user_id, "status": "active"},
        include={"agent": True},
    )
    if workspace is None:
        raise GroundingUnavailable("The memory workspace is no longer active")
    agent = workspace.agent
    if agent is None or agent.userId != user_id or agent.id != workspace.agentId:
        raise GroundingUnavailable("The workspace Agent does not match its owner")
    return agent


async def grounding_context(
    agent: Any, *, template: str | None = None,
    prompt_reader: Callable[[str], Awaitable[str]] | None = None,
) -> str:
    facts = canonical_facts(agent)
    if not facts:
        return ""
    try:
        tpl = template if template is not None else await (prompt_reader or get_prompt_text)("persona.grounding_context")
    except PromptDisabledError:
        return ""
    return tpl.format(facts=json.dumps(facts, ensure_ascii=False, sort_keys=True))


async def verify_generated_locations(
    agent: Any, texts: list[str], *, kind: str, always: bool = False, question: str = "",
    policy_template: str | None = None, timeout: float = VERIFY_TIMEOUT_S,
    user_location: dict[str, Any] | None = None,
) -> list[int]:
    """Return rejected indices; only fully typed, complete verdicts are accepted.

    No hot-path call for ordinary non-location replies. Schedules are checked
    once when generated. This is never allowed to modify canonical facts or to
    use a previous model claim as proof of travel/moving home.

    `timeout` defaults to the hot-path budget; callers off the hot path may
    allow more, since an expired check counts as unverified, never as allowed.

    `user_location` (chat/location_context verifier facts) makes the check
    two-sided: claims about where the user is, or that both are in one city,
    must match it. Without it the user rule is skipped. The returned list
    also says whose location each rejection was about (`Rejected.subjects`).
    """
    from app.services.chat.location_context import claims_user_location

    facts = canonical_facts(agent)
    if not texts:
        return Rejected()
    implicit_self = kind in {"schedule", "memory", "daily_summary"}
    indices = [i for i, t in enumerate(texts)
               if always or needs_location_verification(t, implicit_self=implicit_self)
               or (user_location is not None and claims_user_location(t))]
    if not indices:
        return Rejected()
    if not facts.get("city"):
        if any(needs_location_verification(texts[i], implicit_self=implicit_self) for i in indices):
            raise GroundingUnavailable("No authoritative city for a generated location")
        return Rejected()
    if len(texts) > 50 or any(len(t) > 4000 for t in texts):
        raise GroundingUnavailable("Grounding input exceeds its bounded batch")
    items = [{"index": i, "text": texts[i]} for i in indices]
    try:
        tpl = policy_template if policy_template is not None else await get_prompt_text("persona.grounding_check")
        prompt = tpl.format(
            facts=json.dumps(facts, ensure_ascii=False, sort_keys=True),
            kind=kind, items=json.dumps(items, ensure_ascii=False),
            question=json.dumps(question, ensure_ascii=False),
            user_location=(json.dumps(user_location, ensure_ascii=False, sort_keys=True)
                           if user_location is not None else USER_LOCATION_UNCHECKED),
        )
        async with asyncio.timeout(timeout):
            result = await invoke_json(get_grounding_model(), prompt, temperature=0)
        verdicts = result["verdicts"]
        if not isinstance(verdicts, list) or len(verdicts) != len(indices):
            raise ValueError("Incomplete verdict set")
        seen, rejected = set(), {}
        for item in verdicts:
            i, allowed = item["index"], item["allowed"]
            if type(i) is not int or i not in indices or i in seen or type(allowed) is not bool:
                raise ValueError("Invalid grounding verdict")
            seen.add(i)
            if not allowed:
                # An unlabeled rejection keeps the original, self-sided meaning.
                about = item.get("about")
                rejected[i] = about if isinstance(about, str) and about in _ABOUT else "self"
        if seen != set(indices):
            raise ValueError("Missing grounding verdict")
        if rejected:
            logger.warning("[persona-grounding] rejected kind=%s count=%d", kind, len(rejected))
        return Rejected(rejected)
    except Exception as exc:
        raise GroundingUnavailable("Location claim could not be verified") from exc


async def location_correction(agent: Any) -> str:
    # A deterministic correction cannot invent a journey to explain the drift.
    city = canonical_facts(agent).get("city")
    if city:
        return f"抱歉，刚才地点说错了。我住在{city}。"
    return "刚才把地点说乱了，抱歉，我不该把你的所在地说成我的。"


def user_location_correction(agent: Any, user_location: Any) -> str:
    """说错了对方在哪 / 是否同城时的确定性更正, 只用已知事实, 不补故事。"""
    from app.services.chat.location_context import agent_city, city_core

    mine = city_core(agent_city(agent))
    theirs = city_core(getattr(user_location, "user_city", None))
    relation = getattr(user_location, "relation", "unknown")
    if relation == "different" and mine and theirs:
        return f"刚才说岔了，我在{mine}，你在{theirs}，我们不在一个城市哈"
    if relation == "same" and theirs:
        return f"刚才说岔了，我们都在{theirs}"
    if mine:
        return f"刚才说岔了，我在{mine}，还不知道你在哪儿呢"
    return "刚才说岔了，还不知道你在哪儿呢"


async def guard_reply(
    agent: Any, text: str, *, question: str = "", user_location: Any = None,
) -> tuple[str, bool]:
    """`user_location` (chat/location_context.LocationContext) enables the
    two-sided check on the main reply path; other callers stay single-sided."""
    from app.services.chat.location_context import (
        affirms_same_place_wrongly,
        asks_about_user_location,
    )

    if affirms_same_place_wrongly(user_location, question, text):
        logger.warning("[persona-grounding] rejected kind=reply same-place affirmation")
        return user_location_correction(agent, user_location), True
    # Ordinary activity questions must not add a geography-model round trip.
    # Match each clause independently: an activity question cannot hide a
    # second question asking whether the Agent is in a particular city.
    location_question = bool(re.search(
        r"你.{0,16}(?:住|(?<!现)在(?!\s*(?:干嘛|干什么|做什么|忙|吃|喝|想|看|听|写|工作|学习|睡|休息|玩|等|陪))"
        r".{0,8}(?:哪|吗|？|\?)|哪(?:里|儿|个城市)|搬家|回家)", question,
    ))
    user_facts = user_location.verifier_facts() if user_location is not None else None
    user_question = user_facts is not None and asks_about_user_location(question)
    try:
        rejected = await verify_generated_locations(
            agent, [text], kind="reply", always=location_question or user_question,
            question=question, user_location=user_facts,
        )
        about = getattr(rejected, "subjects", {}).get(0, "self") if rejected else None
    except GroundingUnavailable:
        # Fail closed on whichever side made the reply eligible.
        self_risk = location_question or needs_location_verification(text)
        about = "self" if self_risk or user_facts is None else "user"
    if about == "user":
        return user_location_correction(agent, user_location), True
    if about:
        return await location_correction(agent), True
    return text, False
