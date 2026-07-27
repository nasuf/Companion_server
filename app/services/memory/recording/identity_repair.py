"""把 LLM 漏分的身份事实拉回正确子类, 并保证它们进 L1.

为什么需要这一层: 抽取时的类目由 LLM 判定, 而它对同一句式并不稳定. 生产实测
(2026-07-27, 82 条用户记忆):

    用户叫陈默   → 身份/姓名 0.90 → L1        ✓
    用户叫李杰   → 身份/其他 0.80 → L2        ✗
    用户叫Kiki   → 身份/其他 0.80 → L2        ✗
    用户是广东的 → 身份/其他 0.80 → L2        ✗

同样的句式两种结果. 落到「其他」有两重伤害:

1. importance 一起被压到 0.80, 低于 L1 门槛 0.85, 于是姓名这种最该永驻的事实
   进了会衰减的 L2;
2. 绕过 L1_SINGLETON 保护 —— 「姓名」限一条, 「其他」不限, 同一个人可以攒下
   好几个名字, 后续检索谁也不知道该信哪个.

做法是规则兜底而不是继续调 prompt: 姓名/性别/籍贯这类事实的表达方式高度固定,
正则能以接近 100% 的精度覆盖主流说法; LLM 分对时这层不介入, 分错时才纠正.
跟 pipeline 里既有的「提醒 importance clamp」是同一种思路 —— 关键字段不能只
依赖模型的当次发挥.

**只在高置信度模式上动手**: 宁可漏掉一些身份事实 (维持现状, 不更差), 也不能
把非身份的句子错标成身份 —— 后者会污染 L1 和 singleton 槽位, 比漏判严重得多.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 身份类的最低 importance. 高于 L1 门槛 (0.85), 让这些事实必然永驻;
# 不取 1.0 是给「用户明确强调」这类更高确信度留出空间.
IDENTITY_IMPORTANCE_FLOOR = 0.90


@dataclass(frozen=True)
class _Rule:
    sub: str
    pattern: re.Pattern[str]
    # 命名组 `name` / `place` 存在时, 还要过对应的校验才算命中.
    validate_name: bool = False
    validate_place: bool = False
    validate_birthday: bool = False
    validate_first_person: bool = False


def _p(expr: str) -> re.Pattern[str]:
    return re.compile(expr)


# 「叫」在中文里同时是"名叫"和"叫某人做某事". 「用户叫我明天提醒他」「用户叫外卖」
# 都能套进「用户叫X」的壳子, 误判成姓名后会强占 L1 和 singleton 槽位 —— 比漏判
# 严重得多. 用一个否定词表卡住: 真名字里不会出现人称代词和动词.
_NOT_A_NAME_TOKENS: tuple[str, ...] = (
    "我", "你", "他", "她", "它", "咱", "大家", "别人", "对方",
    "了", "着", "过", "别", "不", "要", "去", "来", "做", "帮",
    # 「的」放这里而不是只写进字符类: 三条姓名规则都要挡, 「老板的助理」不是名字.
    "的",
    "提醒", "外卖", "车", "起床", "上", "下", "回", "出",
)


def _looks_like_a_name(candidate: str) -> bool:
    """粗判一个片段像不像人名/昵称.

    宁可判否 —— 漏掉一个名字只是维持现状, 认错一个会污染 L1.
    """
    text = candidate.strip()
    if not (1 <= len(text) <= 12):
        return False
    return not any(token in text for token in _NOT_A_NAME_TOKENS)


# 「用户是X的/X人」的 X 位置什么都能塞: 籍贯 (广东)、性格 (内向的)、状态
# (已婚的)、年龄段 (年轻)。停用词表治不了这个 —— 形容词是开放集合, 列不完。
# 所以反过来做: 只认**带地名标志**的 X。有界、精确、零误判, 代价是没收录的
# 地名 (如「潮汕」) 会漏判, 那只是回到修复前的状态, 加一行数据即可。
_PLACE_SUFFIXES: tuple[str, ...] = ("省", "市", "县", "区", "州", "镇", "村", "岛", "旗")

_PROVINCE_LEVEL: frozenset[str] = frozenset({
    "北京", "天津", "上海", "重庆",
    "河北", "山西", "辽宁", "吉林", "黑龙江", "江苏", "浙江", "安徽", "福建",
    "江西", "山东", "河南", "湖北", "湖南", "广东", "海南", "四川", "贵州",
    "云南", "陕西", "甘肃", "青海", "台湾",
    "内蒙古", "广西", "西藏", "宁夏", "新疆", "香港", "澳门",
    # 高频地级市/都市圈简称 —— 口语里同样常见「我是苏州的」
    "深圳", "广州", "杭州", "南京", "苏州", "成都", "武汉", "西安", "长沙",
    "青岛", "厦门", "东北", "潮汕",
})


# 生日必须是一个**固定日期**. 「用户的生日是明天，记得提醒」既是相对时间也是
# 提醒请求 —— 把它钉成永不衰减的 L1「生日是明天」, 明天就成了假记忆。
_RELATIVE_OR_REMINDER_TOKENS: tuple[str, ...] = (
    "今天", "明天", "后天", "昨天", "下周", "下个月", "这周", "本月",
    "提醒", "别忘", "记得", "快到", "要到",
)


# 主语和谓语之间插了这些词, 说的就不是用户自己了.
_THIRD_PARTY_TOKENS: tuple[str, ...] = (
    "对方", "他", "她", "同事", "朋友", "别人", "家人", "同学", "老板", "妈", "爸",
)


def _is_about_the_user(gap: str) -> bool:
    return not any(token in gap for token in _THIRD_PARTY_TOKENS)


def _looks_like_a_birthday(candidate: str) -> bool:
    text = candidate.strip()
    if any(token in text for token in _RELATIVE_OR_REMINDER_TOKENS):
        return False
    # 至少要出现一个数字, 否则不是具体日期.
    return any(ch.isdigit() for ch in text)


# 带行政后缀不等于是个具体地点: 「本市」「大城市」「外省」都以 市/省 结尾, 但它们
# 是泛指, 写进 singleton 槽位会把真正的籍贯/现居地挡在门外.
_GENERIC_PLACE_TOKENS: tuple[str, ...] = (
    "本市", "本省", "本地", "外省", "外地", "老家", "城市", "农村", "县城",
    "一线", "二线", "三线", "某", "这个", "那个", "其他", "别的", "同一",
)


def _looks_like_a_place(candidate: str) -> bool:
    text = candidate.strip().rstrip("的")
    if not text:
        return False
    if any(token in text for token in _GENERIC_PLACE_TOKENS):
        return False
    # 白名单优先: 省级/高频城市简称没有行政后缀 (「广东」「苏州」).
    return text in _PROVINCE_LEVEL or text.endswith(_PLACE_SUFFIXES)


# 顺序即优先级: 先匹配到的先用. 姓名放最前, 因为「用户叫X」是最常见也最明确的.
#
# 所有模式都要求出现在 summary 开头附近的主语之后, 避免命中「用户说他同事叫李杰」
# 这类第三人的信息 —— 抽取 prompt 要求 summary 以「用户」开头, 因此锚定 ^用户
# 是安全且必要的.
# 每条规则都必须**校验载荷**, 不能只认前缀.
#
# 这是踩出来的教训: 第一版全是前缀匹配, 于是「用户是内向的人」→出生地、
# 「用户叫我明天提醒他」→姓名、「用户现在住在心里」→现居地. 误判的代价不是
# 记错一条, 而是占掉 singleton 槽位 —— 之后真正的籍贯/姓名会被当成重复拒写,
# 且这条错的还进了永不衰减的 L1.
#
# 所以每条要么带 `place`/`name` 命名组交给校验器, 要么用 `$` 锚死整句。
_RULES: tuple[_Rule, ...] = (
    _Rule(
        "姓名",
        # 交替分支按长度倒序: 正则从左到右取第一个能匹配的, 「叫」放前面会让
        # 「用户叫做阿山」只吃掉「叫」, 把「做阿山」当成名字然后被停用词拒掉.
        _p(r"^用户(?:的名字)?(?:名字是|姓名是|叫做|名叫|叫)\s*(?P<name>[^\s，,。的]{1,12})$"),
        validate_name=True,
    ),
    _Rule(
        "姓名",
        _p(r"^用户(?:希望|要求|让)(?:他人|别人|对方|AI)?称呼(?:自己|他|她)?为"
           r"\s*(?P<name>[^\s，,。]{1,12})$"),
        validate_name=True,
    ),
    _Rule(
        "姓名",
        _p(r"^用户(?:的)?小名(?:叫|是)\s*(?P<name>[^\s，,。]{1,12})$"),
        validate_name=True,
    ),
    # 句尾锚死 + 中间段限长: 生产原句是「用户称和AI说过自己是男生」, 主语和「是」
    # 之间有插入语, 所以前缀要留活口; 但不锚句尾的话「用户是女生缘很好」也会命中.
    # 中间段再过一遍第三人称检查, 挡住「用户觉得对方是女生」.
    _Rule(
        "性别",
        _p(r"^用户(?P<gap>.{0,12}?)是(?:一个)?(?:男生|女生|男性|女性|男孩|女孩|男的|女的)$"),
        validate_first_person=True,
    ),
    # 同样锚死: 「用户29岁的时候去了北京」是往事, 不是当前年龄.
    _Rule("年龄", _p(r"^用户(?:今年|现在)?\s*\d{1,3}\s*岁了?$")),
    _Rule(
        "生日",
        _p(r"^用户(?:的)?生日(?:是|在)\s*(?P<date>[^\s，,。]{2,20})$"),
        validate_birthday=True,
    ),
    _Rule(
        "现居地",
        _p(r"^用户(?:现在|目前)?(?:住在|定居在?|生活在|常住)\s*(?P<place>[^\s，,。]{1,12})$"),
        validate_place=True,
    ),
    _Rule(
        "出生地",
        _p(r"^用户(?:是|来自)\s*(?P<place>[^\s，,。]{1,10}?)(?:人|的)$"),
        validate_place=True,
    ),
    _Rule(
        "出生地",
        _p(r"^用户来自\s*(?P<place>[^\s，,。]{1,10})$"),
        validate_place=True,
    ),
    _Rule(
        "出生地",
        _p(r"^用户(?:的)?(?:老家|籍贯)(?:在|是)\s*(?P<place>[^\s，,。]{1,12})$"),
        validate_place=True,
    ),
)

# 只有这些子类会被纠正过来. 全部是 L1_SINGLETON_SUBS 的成员 —— 兜底的价值正在于
# 恢复 singleton 保护, 对非 singleton 子类做同样的事既没必要也更容易误伤.
REPAIRABLE_SUBS: frozenset[str] = frozenset(r.sub for r in _RULES)

# 只在这些「模型没想清楚」的落点上介入. 模型给了具体子类 (哪怕是别的身份子类)
# 就说明它做了判断, 不该被正则推翻.
_VAGUE_SUBS: frozenset[str] = frozenset({"其他", "", "未知"})


def detect_identity_sub(summary: str) -> str | None:
    """summary 命中高置信度身份模式时返回应有的子类, 否则 None."""
    text = (summary or "").strip()
    if not text:
        return None
    for rule in _RULES:
        m = rule.pattern.search(text)
        if not m:
            continue
        if rule.validate_name and not _looks_like_a_name(m.group("name")):
            continue
        if rule.validate_place and not _looks_like_a_place(m.group("place")):
            continue
        if rule.validate_birthday and not _looks_like_a_birthday(m.group("date")):
            continue
        if rule.validate_first_person and not _is_about_the_user(m.group("gap")):
            continue
        return rule.sub
    return None


def repair_identity_classification(
    *,
    summary: str,
    main_category: str | None,
    sub_category: str | None,
    importance: float,
) -> tuple[str, str, float, str | None]:
    """返回 (main, sub, importance, 纠正原因); 不需要纠正时原样返回, 原因为 None.

    两种介入:
    - 子类是「其他」/空 且 summary 命中身份模式 → 补上正确子类
    - 子类已是 (或刚被补成) 可修复的身份 singleton 子类 → 抬到 L1 下限

    第二条独立于第一条: LLM 有时分对了子类却仍给 0.80 (生产可见), 那种情况同样
    要抬, 否则姓名照样进不了 L1.
    """
    main = main_category or ""
    sub = sub_category or ""
    reason: str | None = None

    if sub in _VAGUE_SUBS:
        detected = detect_identity_sub(summary)
        if detected is not None:
            # main 一并纠正: 模型把身份事实放进「生活/其他」时也要拉回来.
            reason = f"sub {main or '∅'}/{sub or '∅'} → 身份/{detected}"
            main, sub = "身份", detected

    if main == "身份" and sub in REPAIRABLE_SUBS and importance < IDENTITY_IMPORTANCE_FLOOR:
        bumped = f"importance {importance:.2f} → {IDENTITY_IMPORTANCE_FLOOR:.2f}"
        reason = f"{reason}; {bumped}" if reason else bumped
        importance = IDENTITY_IMPORTANCE_FLOOR

    return main, sub, importance, reason
