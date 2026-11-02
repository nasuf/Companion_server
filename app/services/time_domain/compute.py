"""次数、先后、间隔: 由代码按"对方跟你提到的日子"算好再交给回复 (TIME08)。

"这个月我去了几次健身房" "最近一次面试是什么时候" "我多久没去健身了" —— 以前把
检索到的若干条记忆交给模型自己数、自己排、自己算日期差。检索是 top-k, 同一件事说
三遍会被数成三次, 日期差也常算错。

记忆本身也不是完整的事件账本: 同一件事反复说会在录入时合并成一条 (只加提及次数),
约一半事件记忆的"发生时间"其实就是说这句话的时间。所以这里不冒充"现实中一共几次",
只算"对方跟你提到过的日子": 按事件词扫描用户自己的生活/情绪记忆 (有上限), 合并过的
记忆用它各次来源消息的时间补回提及日, 按本地日去重。次数永远是下限 ("至少"),
最近一次说成"最近一次跟你提到"。

拿不准就不算: 问的是 AI 或别人、问"我们"、几件事连在一起、问句带时间范围却没解析
出唯一范围、否定/意愿/计划/提醒类说法 —— 都不计入或整体放弃。不写库, 不调模型。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from app.services.time_domain.operations import OTHER_SUBJECT
from app.services.time_domain.timestamps import exclusive_end, parse_time

logger = logging.getLogger(__name__)

SCAN_LIMIT = 200
_OPERATIONS = {"count", "order", "duration"}

# 问句里不属于"事件"的部分: 时间词、疑问词、虚词。剩下的连续中文就是事件词。
_STRIP = re.compile(
    r"几次|多少次|多少回|几回|几场|多少场|几趟|次数|一共|总共|加起来|"
    r"最近一次|上一次|上上次|上次|最后一次|倒数第[一二三四五六七八九十\d]+次|第[一二三四五六七八九十\d]+次|"
    r"多久|多长时间|多少天|几天|几个月|隔了|间隔|距离|离现在|到现在|以来|"
    r"什么时候|啥时候|哪天|几号|是不是|有没有|"
    r"这个月|上个月|下个月|今年|去年|前年|本周|这周|上周|这礼拜|上礼拜|最近|今天|昨天|前天|"
    r"[0-9一二三四五六七八九十]+月份?|[0-9]+年|"
    # 指代词: "上次那个" 说不清是哪件事 (排在 "这个月" 之后)
    r"那个|这个|那件事|这件事|那事|这事|"
    r"我|了|过|的|去|是|在|呢|吗|啊|呀|来着|没有|没|还|都|又|再|跟|和|一下|过去|"
    r"[，。！？?!,\s]"
)
_TERM = re.compile(r"[一-鿿A-Za-z]{2,8}")
# 问的不是用户本人做的事: AI ("你去过几次" 且没有"我")、"我们"、家人朋友。
_OTHER_SUBJECT = OTHER_SUBJECT
# 动宾事件词的动词: 记忆里常写成"吃了火锅""看了场电影", 只按宾语匹配。
_LEADING_VERB = re.compile(r"^[吃喝看打玩做上逛买听踢唱跑]")
# 地点后缀: 问"健身房", 记忆里多半是"去健身了"。
_PLACE_SUFFIX = re.compile(r"(?<=..)[房馆院店场厅]$")
# 这些说法提到了事件词, 但不是"做了这件事"。按事件词附近的窗口判断, 不按整句:
# "好久没健身了今天终于去了" 是去了; "今天没健身" 是没去。
_NEGATION_BEFORE = r"(?:(?:没有?|不|没能|忘了|不想|懒得|去不了)[^，。！？,]{0,2}|取消了?[^，。！？,]{0,6})"
_NEGATION_AFTER = r"[^，。！？,]{0,3}(?:不了|没成|取消|泡汤|改天)"
_PLAN_BEFORE = (r"(?:打算|计划着|准备|想(?![到起法通])|要去|将要|下次|明天|下周|下个月|改天|"
                r"约了|预约了?|约好|报了|报名了?)[^，。！？,]{0,5}")
# "该洗车了" "应该去趟医院": 还没做, 带 "了" 也不算做了
_SHOULD_BEFORE = r"(?:应该(?!是|已经|也)|(?<!应)该(?!死)|(?<![记觉懂舍])得去)[^，。！？,]{0,3}"
# 计划之后真的做了: 事件词后面跟着 "了/完" ("打算去健身, 结果练完了")
_DONE_AFTER = r"[^，。！？,]{0,4}(?:了|完)"
# 提到了事件词, 说的却是票、名额之类 ("抢到了演唱会门票")
_NOT_EVENT_AFTER = r"(?:门票|票|名额)"
# 做这件事的是别人 ("他同事去了咖啡店" "给儿子报了游泳班"); "和同事打球" 是一起做的
_ACTORS = (r"(?:他|她|同事|男朋友|女朋友|朋友|闺蜜|室友|我妈|妈妈|我爸|爸爸|哥哥|姐姐|弟弟|妹妹|"
           r"孩子|儿子|女儿|老公|老婆|对象)")
_OTHER_ACTOR = re.compile(_ACTORS)
# "和男朋友" "带着孩子" "陪她": 一起做的, 先去掉
_WITH_ACTOR = re.compile(r"(?:和|跟|陪|与|同|带)着?" + _ACTORS)
_FINALLY_DONE = re.compile(r"终于|总算|又去了|还是去了")
_DURATION_SINCE = re.compile(r"多久没|多长时间没|多少天没|几天没|隔了多久|间隔多久|距离(?:我)?上次|距(?:我)?上次|离(?:我)?上次|过去几天")
# 问了时间范围 ("这个月" "上周" "九月份")。"最近" 不算范围 (模糊词, 解析不出区间), 按不限范围算。
_TIME_HINT = re.compile(r"这个月|上个月|今年|去年|前年|这周|本周|上周|这礼拜|上礼拜|月份?|号|今天|昨天|前天")
_ORDINAL = re.compile(r"(倒数)?第([一二三四五六七八九十\d]{1,3})次")
_ORDINAL_WORDS = re.compile(r"最近一次|上一次|上上次|上次|最后一次|(?:倒数)?第[一二三四五六七八九十\d]{1,3}次")


@dataclass(frozen=True)
class EventRow:
    id: str
    text: str
    occur_time: datetime | None
    statement_time: datetime | None
    mention_count: int = 0
    source_times: tuple[datetime, ...] = ()


@dataclass
class TemporalAnswerPlan:
    operation: str
    status: str  # computed | no_events | too_few | topic_unclear | unavailable
    topic: str = ""
    reasons: list[str] = field(default_factory=list)
    mention_days: list[date] = field(default_factory=list)
    undated: int = 0
    skipped: int = 0
    scanned: int = 0
    truncated: bool = False
    period_label: str = ""
    value: dict[str, Any] = field(default_factory=dict)

    def diagnostics(self) -> dict:
        return {
            "version": 1, "operation": self.operation, "status": self.status,
            # 记忆只记得"提到过", 不是完整事件账本; 次数永远只是下限。
            "lower_bound": True, "reasons": list(self.reasons),
            "mention_days": len(self.mention_days), "undated": self.undated,
            "skipped": self.skipped, "scanned": self.scanned, "scan_limit": SCAN_LIMIT,
            "truncated": self.truncated, "value": dict(self.value), "model_assist": "not_used",
        }


def topic_term(question: str) -> str:
    """问句里的事件词: "这个月我去了几次健身房" → "健身"。拿不准返回空串。"""
    question = question or ""
    if _OTHER_SUBJECT.search(question) or ("你" in question and "我" not in question):
        return ""
    rest = _STRIP.sub(" ", question.replace("你", " "))
    terms = _TERM.findall(rest)
    if not terms:
        # "加了几次班": 动宾被次数词隔开, 剩下的单字拼回去
        joined = "".join(re.findall(r"[一-鿿]", rest))
        return joined if 2 <= len(joined) <= 4 else ""
    if len(terms) != 1 or len(terms[0]) > 4:
        return ""  # 几件事连在一起 ("吃饭看电影") 拿不准是哪件
    return _PLACE_SUFFIX.sub("", terms[0])


def _ordinal(question: str) -> tuple[str, int] | None:
    """("first"|"last", n): 第一次 → (first,1); 倒数第二次 → (last,2); 最近一次/上次 → (last,1)。"""
    from app.services.time_domain.parser import _parse_cn_number

    match = _ORDINAL.search(question)
    if match:
        n = _parse_cn_number(match.group(2))
        if not n or n < 1:
            return None
        return ("last" if match.group(1) else "first", n)
    if "上上次" in question:
        return ("last", 2)
    if re.search(r"最近一次|上一次|上次|最后一次", question):
        return ("last", 1)
    return None


def search_term(term: str) -> str:
    """扫描用的词: 动宾事件去掉动词 ("吃火锅" → "火锅"), 其余原样。"""
    return term[1:] if len(term) >= 3 and _LEADING_VERB.match(term) else term


# 同一件事的常见说法。问"健身", 记忆里可能写的是"撸铁"。每族两组: 同义说法互通;
# 更具体的说法只在问泛称时算进来 (问"健身"算"练腿", 问"练腿"不算别的健身)。
# 只放含义明确、不会把别的事算进来的词; 族内按天去重。
_SYNONYM_FAMILIES = (
    (("健身", "撸铁", "举铁", "力量训练"), ("练腿", "练背", "练胸", "练肩", "私教课")),
    (("跑步",), ("夜跑", "晨跑", "慢跑", "长跑")),
    (("游泳", "游了泳"), ()),
    # 问"去医院"算上看病; 问"看病"不算 "在医院值班" 这类只是人在医院的说法
    (("医院",), ("看病", "看医生", "就诊", "门诊", "挂号", "体检")),
    (("看病", "看医生", "就诊"), ("门诊", "挂号")),
    (("面试",), ("复试", "终面")),
    (("吵架", "吵了一架"), ("闹别扭",)),
    (("喝酒", "喝了点酒", "小酌"), ()),
    (("火锅", "涮锅"), ()),
    (("电影", "影院"), ()),
    (("约会",), ("见网友",)),
    (("看牙", "牙医", "牙科"), ("补牙", "洗牙", "拔牙", "正畸")),
    (("看展", "展览", "看了个展"), ("画展", "摄影展", "美术馆")),
)


def search_terms(term: str) -> tuple[str, ...]:
    """扫描用的一组词: 问的词所在的同义说法; 问的是泛称时再加上更具体的说法。"""
    base = search_term(term)
    for same, narrower in _SYNONYM_FAMILIES:
        if base in same:
            return tuple(dict.fromkeys((base, *same, *narrower)))
    return (base,)


def _not_occurred(text: str, term: str) -> bool:
    """这条说法里, 事件词附近是否是否定 / 计划 (没做成或还没做)。"""
    t = "(?:" + "|".join(re.escape(x) for x in search_terms(term)) + ")"
    for clause in re.split(r"[，。！？,!?；;]", text):
        if not re.search(t, clause):
            continue
        if _FINALLY_DONE.search(clause):
            continue  # "好久没健身了今天终于去了": 最后是做了
        if re.search(_NEGATION_BEFORE + t, clause) or re.search(t + _NEGATION_AFTER, clause):
            return True
        if re.search(t + _NOT_EVENT_AFTER, clause) or re.search(_SHOULD_BEFORE + t, clause):
            return True
        if re.search(_PLAN_BEFORE + t, clause) and not re.search(t + _DONE_AFTER, clause):
            return True
        head = _WITH_ACTOR.sub("", re.split(t, clause, maxsplit=1)[0])
        actor = None
        for actor in _OTHER_ACTOR.finditer(head):
            pass
        # 别人是主语, 且之后没再出现用户 ("她陪用户去医院" "同事叫上用户" 是一起做的)
        if actor and "用户" not in head[actor.end():]:
            return True
        if re.search(r"给[^，。！？,]{0,3}(?:儿子|女儿|孩子|老公|老婆)", head):
            return True
    return False


def _local_day(value: datetime, zone: ZoneInfo) -> date:
    return value.astimezone(zone).date()


def compute_plan(operation: str, question: str, rows: list[EventRow], *, now: datetime,
                 zone: ZoneInfo, period=None, truncated: bool = False) -> TemporalAnswerPlan:
    """纯计算: 给定扫描结果算出"提到过的日子"; 不读库。period 为问句里的时间范围。"""
    plan = TemporalAnswerPlan(operation=operation, status="computed", topic=topic_term(question),
                              scanned=len(rows), truncated=truncated)
    start = parse_time(period.start) if period is not None else None
    end = exclusive_end(period.end, period.end_boundary) if period is not None else None
    if start is not None and end is not None:
        plan.period_label = f"{_local_day(start, zone).isoformat()} 到 {_local_day(end, zone).isoformat()} 之前"
    today = _local_day(now, zone)
    days: set[date] = set()
    terms = search_terms(plan.topic) if plan.topic else ()
    for row in rows:
        if terms and not any(t in row.text for t in terms):
            continue  # 扫描是子串匹配, 这里再确认一遍说的就是这件事
        if _not_occurred(row.text, plan.topic or question):
            plan.skipped += 1
            continue
        # 一条记忆只按一个日子算: 说了发生日就用发生日, 否则用说这件事的那天。合并过的
        # 重复说法, 后来各次的提及日各算一天 (近似发生日)。
        mentions = sorted(row.source_times)
        primary = row.occur_time or row.statement_time or (mentions[0] if mentions else None)
        if primary is None:
            plan.undated += 1
            continue
        if _local_day(primary, zone) > today:
            plan.skipped += 1  # 还没到的日子: 计划, 整条不算
            continue
        stamps = [primary] + (mentions[1:] if row.mention_count > 0 else [])
        for when in stamps:
            if _local_day(when, zone) > today:
                continue
            if start is not None and (when < start or (end is not None and when >= end)):
                continue
            days.add(_local_day(when, zone))
        if row.mention_count > 0 and len(mentions) <= row.mention_count:
            # 合并过的重复说法, 来源时间没全: 可能还有更多、更近的提及
            if "merged_mentions" not in plan.reasons:
                plan.reasons.append("merged_mentions")
    if truncated:
        plan.reasons.append("scan_limit_reached")
    if plan.undated:
        plan.reasons.append("undated_mentions")
    plan.mention_days = sorted(days)
    if not plan.mention_days:
        plan.status = "no_events"
        return plan
    dates = plan.mention_days
    if operation == "count":
        plan.value = {"at_least_days": len(dates)}
    elif operation == "order":
        side, n = _ordinal(question) or ("last", 1)
        if side == "first" and truncated:
            plan.status = "topic_unclear"  # 最早的那些可能没扫到
            plan.reasons.append("first_beyond_scan")
            return plan
        if n > len(dates):
            plan.status = "too_few"
            plan.value = {"which": side, "n": n, "mentioned": len(dates)}
            return plan
        pick = dates[n - 1] if side == "first" else dates[-n]
        plan.value = {"which": side, "n": n, "date": pick.isoformat(), "days_ago": (today - pick).days}
    elif operation == "duration":
        last = dates[-1]
        plan.value = {"since_last_days": (today - last).days, "last_date": last.isoformat()}
        if len(dates) >= 2:
            plan.value["gap_between_last_two_days"] = (dates[-1] - dates[-2]).days
    return plan


def render_plan(plan: TemporalAnswerPlan) -> str:
    """给主回复的事实句: 算好的结果和它的边界。"""
    if plan.status == "too_few":
        v = plan.value
        return (f"关于「{plan.topic}」，对方跟你提到过的只有 {v['mentioned']} 次（按天算），"
                f"对不上问的第 {v['n']} 次。可以这么说，不要编出那一次的日期。")
    if plan.status == "no_events":
        return (f"关于「{plan.topic}」，没找到对方跟你提到过、能对上日子的这类经历。"
                "不能说没发生过，只能说不确定。")
    if plan.status != "computed":
        return ""
    scope = f"（范围：{plan.period_label}）" if plan.period_label else ""
    lines = [f"事件：{plan.topic}{scope}。只算对方跟你说过的：说了哪天发生就按哪天，没说就按说的那天；"
             "同一天算一次，对方没跟你说过的不在里面。"]
    v = plan.value
    if plan.operation == "count":
        n = v["at_least_days"]
        lines.append(f"次数：对方跟你提到过的至少 {n} 次。用你自己的话带出\"这是你跟我说过的\"这层意思，"
                     f"不能说成一共正好{n}次")
    elif plan.operation == "order":
        if v["which"] == "last":
            name = "最近一次" if v["n"] == 1 else f"倒数第{v['n']}次"
        else:
            name = "第一次" if v["n"] == 1 else f"第{v['n']}次"
        lines.append(f"对方{name}跟你提到：{v['date']}，距今 {v['days_ago']} 天")
    elif plan.operation == "duration":
        lines.append(f"对方最近一次跟你提到：{v['last_date']}，到今天 {v['since_last_days']} 天")
        if "gap_between_last_two_days" in v:
            lines.append(f"最近两次提到相隔 {v['gap_between_last_two_days']} 天")
        lines.append("用你自己的话说成是对方上次跟你提起的时间，不要断定对方这段时间一次都没去")
    if "merged_mentions" in plan.reasons or "undated_mentions" in plan.reasons:
        earlier = plan.operation == "order" and v.get("which") == "first"
        lines.append(f"还有说法没记清日子，实际可能更多、{'更早' if earlier else '更近'}")
    lines.append("这些数字和日期由程序算好，照着说，不要自己重新数或算日期差。")
    return "\n".join(lines)


async def load_event_rows(*, user_id: str, workspace_id: str, term: str | tuple[str, ...],
                          database=None) -> tuple[list[EventRow], bool]:
    """按事件词 (及同义说法) 扫描用户自己的生活/情绪记忆 (不含提醒; 上限 SCAN_LIMIT)。"""
    from app.db import db as default_db

    client = database or default_db
    terms = (term,) if isinstance(term, str) else tuple(term)
    patterns = ["%" + t.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%" for t in terms]
    rows = await client.query_raw(
        """SELECT m.id, m.content, m.occur_time, m.statement_time, m.mention_count,
                  COALESCE((SELECT array_agg(DISTINCT l.source_statement_time)
                            FROM memory_evidence_links l
                            WHERE l.user_memory_id = m.id AND l.source_statement_time IS NOT NULL),
                           '{}') AS source_times
           FROM memories_user m
           WHERE m.user_id = $1 AND m.workspace_id = $2 AND m.is_archived = false
             AND m.main_category IN ('生活', '情绪') AND m.sub_category IS DISTINCT FROM '提醒'
             AND m.content ILIKE ANY($3::text[])
           ORDER BY COALESCE(m.occur_time, m.statement_time, m.created_at) DESC, m.id
           LIMIT $4""",
        user_id, workspace_id, patterns, SCAN_LIMIT + 1,
    )
    truncated = len(rows) > SCAN_LIMIT

    def utc(value):
        parsed = parse_time(value)
        if parsed is not None and parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    out = []
    for r in rows[:SCAN_LIMIT]:
        sources = tuple(t for t in (utc(v) for v in (r.get("source_times") or [])) if t is not None)
        out.append(EventRow(str(r["id"]), str(r.get("content") or ""), utc(r.get("occur_time")),
                            utc(r.get("statement_time")), int(r.get("mention_count") or 0), sources))
    return out, truncated


async def plan_for_question(question: str, temporal_query, *, user_id: str | None,
                            workspace_id: str | None, now: datetime) -> TemporalAnswerPlan | None:
    """count/order/duration 问句的计算结果; 其他问句返回 None。失败返回 unavailable, 不抛。"""
    operation = getattr(getattr(temporal_query, "operation", None), "type", None)
    if operation not in _OPERATIONS:
        return None
    if operation == "duration" and not _DURATION_SINCE.search(question or ""):
        return None  # "我减肥多久了" 问的是持续时长, 不是距上次多久
    if operation == "order" and _ordinal(question or "") is None:
        return None
    term = topic_term(question)
    if not term or not user_id or not workspace_id:
        return TemporalAnswerPlan(operation=operation, status="topic_unclear", topic=term)
    expressions = getattr(temporal_query, "expressions", ()) or ()
    if _TIME_HINT.search(_ORDINAL_WORDS.sub("", question)) and len(expressions) != 1:
        plan = TemporalAnswerPlan(operation=operation, status="topic_unclear", topic=term)
        plan.reasons.append("range_unclear")  # 问了范围却没解析出唯一范围, 不拿全部历史顶替
        return plan
    zone = ZoneInfo(temporal_query.timezone)
    period = expressions[0] if len(expressions) == 1 else None
    try:
        rows, truncated = await load_event_rows(user_id=user_id, workspace_id=workspace_id,
                                                term=search_terms(term))
    except Exception as exc:
        logger.warning("[TEMPORAL-COMPUTE] scan unavailable: %s", type(exc).__name__)
        return TemporalAnswerPlan(operation=operation, status="unavailable", topic=term)
    return compute_plan(operation, question, rows, now=now, zone=zone, period=period, truncated=truncated)


_REPLY_COUNT = re.compile(r"(?<![第再下这那每])([0-9]{1,3}|[一二两三四五六七八九十]{1,3})\s*(?:次|回|趟)")
_HEDGE = re.compile(r"至少|起码|最少|跟我(?:说|提)|你说过|你提过|你说了|说过的|提过的")


def check_reply(plan_diag: dict | None, reply: str) -> str | None:
    """生成后核对回复里的次数是否和算好的一致 (TIME09)。只用于 Trace, 不改写回复。

    consistent: 说的数等于下限 / 没说数; overclaimed: 说了更大的数, 或同一个数却没带
    "至少"意思且也没说"跟我说过"; mismatch: 说了更小的数。
    """
    from app.services.time_domain.parser import _parse_cn_number

    if not plan_diag or plan_diag.get("status") != "computed":
        return None
    if plan_diag.get("operation") in ("order", "duration"):
        return _check_dates(plan_diag.get("value") or {}, reply or "")
    if plan_diag.get("operation") != "count":
        return None
    expected = (plan_diag.get("value") or {}).get("at_least_days")
    if not isinstance(expected, int):
        return None
    reply = reply or ""
    said = []
    for m in _REPLY_COUNT.finditer(reply):
        n = _parse_cn_number(m.group(1))
        if n is not None and not re.search(r"[第再下这那每]", reply[max(0, m.start() - 2):m.start()]):
            # 限定语只看这个数字前面几个字, 不让整段回复共用一个"至少"
            said.append((bool(_HEDGE.search(reply[max(0, m.start() - 8):m.start()])), n))
    if not said:
        return "consistent"
    for hedged, n in said:
        if n > expected or (n == expected and not hedged):
            return "overclaimed"
        if n < expected:
            return "mismatch"
    return "consistent"


_REPLY_DATE = re.compile(r"(\d{1,2}|[一二三四五六七八九十]{1,3})\s*月\s*(\d{1,2}|[一二三四五六七八九十]{1,3})\s*[日号]")
# "10月3日前后" 的 "3日前" 不是天数; "隔了5天" 是两次之间的间隔, 不是距今多久
_REPLY_DAYS = re.compile(r"(?<![月隔间])(?<!隔了)(\d{1,3}|[一二两三四五六七八九十]{1,3})\s*天(?:前|了|没|之前)")


def _check_dates(value: dict, reply: str) -> str:
    """先后/间隔: 回复里说的"X月Y日""N天前/N天了"要和算好的一致 (TIME09)。

    mismatch: 说了别的日子或天数; consistent: 一致或没说具体日子。只用于 Trace。
    """
    from datetime import date as date_cls
    from app.services.time_domain.parser import _parse_cn_number

    target = value.get("date") or value.get("last_date")
    days = value.get("days_ago") if "days_ago" in value else value.get("since_last_days")
    try:
        expected_date = date_cls.fromisoformat(target) if target else None
    except ValueError:
        expected_date = None
    # 回复里可能顺带提到别的日子 ("国庆10月1日之后…"), 说到的日子里有算好的那个就算一致
    said_dates = {(_parse_cn_number(m.group(1)), _parse_cn_number(m.group(2)))
                  for m in _REPLY_DATE.finditer(reply)}
    if expected_date and said_dates and (expected_date.month, expected_date.day) not in said_dates:
        return "mismatch"
    said_days = {_parse_cn_number(m.group(1)) for m in _REPLY_DAYS.finditer(reply)} - {None}
    if isinstance(days, int) and said_days and days not in said_days:
        return "mismatch"
    return "consistent"
