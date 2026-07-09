"""Import a structured txt agent template document into CharacterProfile schema.

The product document used for template creation is already written as a
"five-dimension memory profile". This module keeps the conversion deterministic:
decode uploaded text, split known sections, then map them into the same profile
shape consumed by life_story.convert_profile_to_memories.

Parsing quality rules (2026-07 rework, driven by the 小伴.txt production audit):
- Third-person narration ("林昕/她...") is normalized to first person outside
  quoted speech, so stored AI self-memories read naturally as "我...".
- Enumeration splitting is sentence-first, bracket/quote aware, and refuses to
  split clauses — no more mid-parenthesis fragments like "我喜欢吃大芫荽）".
- Sentences carrying negation ("不喜欢高强度、对抗性的运动") are never fed into
  a "likes" list, which previously inverted their meaning after prefixing.
- Labeled-event splitting no longer treats speech verbs ("王老师说：") or
  clause text as item labels, so one event stays one memory.
- 人生观/价值观 no longer truncate trailing sentences; goal/relationship/social
  view items keep their labels ("短期目标（1-3年）：...").
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class ImportedAgentProfile:
    profile: dict[str, Any]
    career_template: dict[str, Any]
    name: str | None
    gender: str | None
    background: str
    text: str


_IDENTITY_SECTIONS = (
    "AI自我姓名", "AI自我年龄", "AI自我性别", "AI自我生日", "AI自我星座",
    "AI自我生肖", "AI自我血型", "AI自我民族", "AI自我出生地", "AI自我成长地",
    "AI自我现居地", "AI自我外貌特征", "AI自我教育背景", "AI自我职业与经济",
    "AI自我亲属关系", "AI自我社会关系", "AI自我宠物",
)
_PREFERENCE_SECTIONS = (
    "饮食喜好", "饮食厌恶", "审美爱好", "审美厌恶", "人际喜好", "人际厌恶",
    "生活习惯", "绝对禁忌/雷区",
)
_LIFE_SECTION_MAP = {
    "交互事件": "life",
    "重要教育事件": "education",
    "重要工作事件": "work",
    "重要旅行事件": "travel",
    "重要居住事件": "living",
    "重要健康事件": "health",
    "重要宠物事件": "pet",
    "重要人际事件": "relationships",
    "重要技能学习事件": "skill_learning",
    "重要生活事件": "life",
    "重要其他特殊事件": "special",
}
_EMOTION_SECTION_MAP = {
    "重要高兴的记忆": "happy",
    "重要悲伤的记忆": "sad",
    "重要愤怒的记忆": "angry",
    "重要恐惧的记忆": "fear",
    "重要厌恶的记忆": "disgust",
    "重要焦虑的记忆": "anxiety",
    "重要失望的记忆": "disappointment",
    "重要自豪的记忆": "pride",
    "重要感动的记忆": "moved",
    "重要尴尬的记忆": "embarrassed",
    "重要遗憾的记忆": "regret",
    "重要孤独的记忆": "lonely",
    "重要惊讶的记忆": "surprised",
    "重要感激的记忆": "grateful",
    "重要释怀的记忆": "relieved",
}
_THOUGHT_SECTIONS = (
    "人生观", "价值观", "世界观", "理想与目标", "人际关系观", "社会观点",
    "自我认知", "信仰/精神寄托",
)
_KNOWN_SECTIONS = {
    *_IDENTITY_SECTIONS,
    *_PREFERENCE_SECTIONS,
    *_LIFE_SECTION_MAP.keys(),
    *_EMOTION_SECTION_MAP.keys(),
    *_THOUGHT_SECTIONS,
}


def parse_agent_profile_document(data: bytes, filename: str | None = None) -> ImportedAgentProfile:
    """Parse uploaded bytes into a profile suitable for template provisioning."""
    text = extract_document_text(data, filename=filename)
    sections = _parse_sections(text)
    if len(sections) < 5:
        raise ValueError("文档结构无法识别：请上传包含五维记忆档案标题的 txt 文件")

    # Normalize third-person narration to first person before field parsing.
    # The 姓名 section is excluded so the real name survives into name_detail.
    name_hint = _extract_primary_name(sections.get("AI自我姓名", ""))
    gender_hint = _first_line(sections.get("AI自我性别", ""))
    sections = {
        key: (
            value
            if key == "AI自我姓名"
            else _normalize_third_person(
                value, name_hint, gender_hint,
                aggressive=key not in _NARRATIVE_SECTIONS,
            )
        )
        for key, value in sections.items()
    }

    profile, career_template = _build_profile(sections)
    identity = profile.setdefault("identity", {})
    name = _clean_scalar(identity.get("name"))
    gender = _clean_scalar(identity.get("gender"))
    if not name:
        raise ValueError("文档中未识别到 AI 自我姓名")

    background = _build_background(profile, career_template)
    return ImportedAgentProfile(
        profile=profile,
        career_template=career_template,
        name=name,
        gender=gender,
        background=background,
        text=text,
    )


def extract_document_text(data: bytes, filename: str | None = None) -> str:
    if not data:
        raise ValueError("上传文件为空")
    return _normalize_text(_decode_text(data))


def _decode_text(data: bytes) -> str:
    sample = data[:1000].decode("ascii", errors="ignore")
    match = re.search(r"charset\s*=\s*['\"]?([a-zA-Z0-9_-]+)", sample, flags=re.I)
    encodings = []
    if match:
        encodings.append(match.group(1))
    encodings.extend(["utf-8", "gb18030", "utf-16", "latin-1"])
    seen: set[str] = set()
    for enc in encodings:
        key = enc.lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
        except LookupError:
            continue
    return data.decode("utf-8", errors="replace")


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ").replace("\u3000", " ")
    text = re.sub(r"[\u2028\u2029]+", "\n", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def _parse_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("【") or line.endswith("记忆档案"):
            continue
        heading = _section_heading(line)
        if heading:
            current = heading
            sections.setdefault(current, [])
            remainder = _heading_remainder(line, heading)
            if remainder:
                sections[current].append(remainder)
            continue
        if current:
            sections[current].append(line)
    return {key: "\n".join(value).strip() for key, value in sections.items() if value}


def _section_heading(line: str) -> str | None:
    candidate = line
    match = re.match(r"^\d+[.、]\s*(.+)$", candidate)
    if match:
        candidate = match.group(1)
    candidate = _canonical_heading(candidate)
    return candidate if candidate in _KNOWN_SECTIONS else None


def _canonical_heading(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[：:]\s*$", "", value)
    value = re.sub(r"（.*?）$", "", value)
    return value.strip()


def _heading_remainder(line: str, heading: str) -> str:
    line_no_number = re.sub(r"^\d+[.、]\s*", "", line).strip()
    if _canonical_heading(line_no_number) == heading:
        return ""
    prefix = re.escape(heading)
    return re.sub(rf"^\d+[.、]\s*{prefix}(?:（.*?）)?\s*", "", line).strip()


# ── Third-person → first-person normalization ─────────────────────────────

_QUOTE_SPAN_RE = re.compile(r"[“「『][^”」』]*[”」』]")
# Vocative/honorific uses must keep the original name ("林昕同学" in a quote
# would already be protected, this guards unquoted occurrences too).
_NAME_SUFFIX_GUARD = r"(?!同学|老师|小姐|女士|先生|哥|姐)"

# Event/narrative sections describe interactions with *other* people, so a
# gender-matched pronoun mid-sentence is often the object (e.g. "背她去医务室"
# = a friend). There we only rewrite the sentence subject. Self-descriptive
# sections (identity/appearance/preferences/values/…) are exclusively about the
# persona, so every gender-matched pronoun can be converted.
_NARRATIVE_SECTIONS = frozenset(_LIFE_SECTION_MAP) | frozenset(_EMOTION_SECTION_MAP)


def _extract_primary_name(name_text: str) -> str | None:
    """The persona's formal name (大名) from the 姓名 section."""
    return _first_match(name_text, r"大名[:：]\s*([^\s（(]+)") or _first_line(name_text)


def _normalize_third_person(
    text: str, name: str | None, gender: str | None, *, aggressive: bool = False,
) -> str:
    """Rewrite third-person self-references to first person outside quotes.

    - Persona name → 我 (skipping honorific compounds like 林昕同学). Always safe:
      the name is unique to the persona and honorifics/quotes are guarded.
    - Gender pronoun → 我. The pronoun is gender-matched so references to other
      people of the opposite sex (e.g. "他60多岁" about a male user in a female
      persona's document) are always left untouched. Coverage depends on mode:
      * aggressive (self-descriptive sections): every gender-matched pronoun,
        since the subject is exclusively the persona.
      * default (narrative/event sections): only the sentence subject
        (initial position), so object-position references to other same-sex
        people ("背她去医务室") are preserved.
    Quoted speech is preserved verbatim — quotes are other people's words.
    """
    if not text:
        return text
    pronoun = None
    g = (gender or "").strip().lower()
    if g in ("女", "female"):
        pronoun = "她"
    elif g in ("男", "male"):
        pronoun = "他"

    name_re = (
        re.compile(re.escape(name) + _NAME_SUFFIX_GUARD) if name and len(name) >= 2 else None
    )

    def _fix(chunk: str) -> str:
        if name_re is not None:
            chunk = name_re.sub("我", chunk)
        if pronoun:
            if aggressive:
                chunk = re.sub(rf"{pronoun}(?!们)", "我", chunk)
            else:
                chunk = re.sub(
                    rf"(^|(?<=[。！？!?；;\n])){pronoun}(?!们)", "我", chunk,
                )
            chunk = chunk.replace(f"对{pronoun}而言", "对我而言")
            chunk = chunk.replace(f"在{pronoun}看来", "在我看来")
        return chunk

    if name_re is None and pronoun is None:
        return text

    parts: list[str] = []
    last = 0
    for span in _QUOTE_SPAN_RE.finditer(text):
        parts.append(_fix(text[last:span.start()]))
        parts.append(span.group(0))
        last = span.end()
    parts.append(_fix(text[last:]))
    return "".join(parts)


def _build_profile(sections: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = _parse_identity(sections)
    appearance = _parse_appearance(sections.get("AI自我外貌特征", ""))
    education = _parse_education(sections.get("AI自我教育背景", ""))
    career = _parse_career(sections.get("AI自我职业与经济", ""))

    likes, dislikes = _parse_preferences(sections)
    interpersonal = _parse_interpersonal(sections)
    lifestyle = _parse_lifestyle(sections.get("生活习惯", ""))
    taboo = {"items": _split_labeled_items(sections.get("绝对禁忌/雷区", ""))}
    values, abilities = _parse_thoughts(sections)

    profile = {
        "identity": identity,
        "appearance": appearance,
        "education_knowledge": education,
        "career": career,
        "likes": likes,
        "dislikes": dislikes,
        "interpersonal": interpersonal,
        "lifestyle": lifestyle,
        "taboo": taboo,
        "values": values,
        "abilities": abilities,
        "life_events": _parse_life_events(sections),
        "emotion_events": _parse_emotion_events(sections),
    }
    return profile, career


def _parse_identity(sections: dict[str, str]) -> dict[str, Any]:
    name_text = sections.get("AI自我姓名", "")
    name_line = _first_line(name_text)
    name = _first_match(name_text, r"大名[:：]\s*([^\s（(]+)") or name_line
    # Keep the full naming line (大名/小名/称呼) so the 姓名 memory can carry
    # aliases even when the admin form overrides identity.name later.
    name_detail = name_line if name_line and name_line != name else None
    gender = _first_line(sections.get("AI自我性别", ""))
    family = _labeled_values(
        sections.get("AI自我亲属关系", ""),
        ("父母职业", "与父母的关系模式", "兄弟姐妹"),
    )
    social = _labeled_values(
        sections.get("AI自我社会关系", ""),
        ("朋友数量质量", "同事关系", "社交圈层特点"),
    )
    pet = _labeled_values(sections.get("AI自我宠物", ""), ("种类与名字", "由来"))
    raw_location = _first_line(sections.get("AI自我现居地", ""))
    location = _strip_parenthetical(raw_location)
    location_note = _parenthetical_note(raw_location)
    return {
        "name": name,
        "name_detail": name_detail,
        "age": _first_int(sections.get("AI自我年龄", "")),
        "gender": gender,
        "birthday": _first_line(sections.get("AI自我生日", "")),
        "constellation": _first_line(sections.get("AI自我星座", "")),
        "zodiac": _first_line(sections.get("AI自我生肖", "")),
        "blood_type": _first_line(sections.get("AI自我血型", "")),
        "ethnicity": _first_line(sections.get("AI自我民族", "")),
        "birthplace": _first_line(sections.get("AI自我出生地", "")),
        "growing_up_location": _first_line(sections.get("AI自我成长地", "")),
        "location": location,
        "location_note": location_note,
        "family": [v for v in family.values() if v],
        "social_relations": [v for v in social.values() if v],
        "pet_profile": [v for v in pet.values() if v],
    }


def _parse_appearance(text: str) -> dict[str, Any]:
    fields = _labeled_values(text, ("身高", "体型", "五官特征", "穿搭风格", "声音特点"))
    return {
        "height": fields.get("身高"),
        "weight": fields.get("体型"),
        "features": _split_paragraph_items(fields.get("五官特征", "")),
        "style": _split_paragraph_items(fields.get("穿搭风格", "")),
        "voice": _split_paragraph_items(fields.get("声音特点", "")),
    }


def _parse_education(text: str) -> dict[str, Any]:
    fields = _labeled_values(text, ("学历", "知识擅长范围", "自学过的特殊技能"))
    return {
        "degree": _as_list(fields.get("学历")),
        "strengths": _split_numbered_items(fields.get("知识擅长范围", "")),
        "self_taught": _split_numbered_items(fields.get("自学过的特殊技能", "")),
    }


def _parse_career(text: str) -> dict[str, Any]:
    fields = _labeled_values(
        text,
        ("职业", "工作内容", "主要产出物", "社会价值", "服务对象", "经济状况"),
    )
    duties = "；".join(_split_numbered_items(fields.get("工作内容", "")))
    outputs = _clean_scalar(fields.get("主要产出物"))
    if outputs:
        duties = f"{duties}；主要产出物：{outputs}" if duties else f"主要产出物：{outputs}"
    return {
        "title": _clean_scalar(fields.get("职业")) or "",
        "duties": duties,
        "social_value": _clean_scalar(fields.get("社会价值")) or "",
        "clients": _as_list(fields.get("服务对象")),
        "income": _clean_scalar(fields.get("经济状况")) or "",
    }


def _parse_preferences(sections: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    diet = _labeled_values(sections.get("饮食喜好", ""), ("食物", "水果", "菜系"))
    aesthetic = _labeled_values(
        sections.get("审美爱好", ""),
        ("颜色", "季节", "天气", "植物", "动物", "音乐类型", "歌曲", "声音",
         "气味", "书籍类型", "电影", "运动", "小癖好"),
    )
    aesthetic_dislikes = _labeled_values(sections.get("审美厌恶", ""), ("噪音", "气味", "习惯"))
    likes = {
        "foods": _split_inline_list(diet.get("食物", "")) + _split_inline_list(diet.get("菜系", "")),
        "fruits": _split_inline_list(diet.get("水果", "")),
        "colors": _split_inline_list(aesthetic.get("颜色", "")),
        "season": _split_inline_list(aesthetic.get("季节", "")),
        "weather": _split_inline_list(aesthetic.get("天气", "")),
        "plants": _split_inline_list(aesthetic.get("植物", "")),
        "animals": _split_inline_list(aesthetic.get("动物", "")),
        "music": _split_inline_list(aesthetic.get("音乐类型", "")),
        "songs": _split_inline_list(aesthetic.get("歌曲", "")),
        "sounds": _split_inline_list(aesthetic.get("声音", "")),
        "scents": _split_inline_list(aesthetic.get("气味", "")),
        "books": _split_inline_list(aesthetic.get("书籍类型", "")),
        "movies": _split_inline_list(aesthetic.get("电影", "")),
        "sports": _split_inline_list(aesthetic.get("运动", "")),
        "quirks": _split_paragraph_items(aesthetic.get("小癖好", "")),
    }
    dislikes = {
        "foods": _split_labeled_items(sections.get("饮食厌恶", "")),
        "sounds": _split_inline_list(aesthetic_dislikes.get("噪音", "")),
        "smells": _split_inline_list(aesthetic_dislikes.get("气味", "")),
        "habits": _split_inline_list(aesthetic_dislikes.get("习惯", "")),
    }
    return likes, dislikes


_LIKE_VERB_STRIP_RE = re.compile(
    r"^(?:我)?(?:也|还|最|尤其|特别|非常|比较|更|很)?"
    r"(?:喜欢|喜爱|偏爱|钟爱|热爱|欣赏|珍视|享受|爱)(?:吃|喝|听|看|玩|穿)?(?!的)"
)


def _parse_interpersonal(sections: dict[str, str]) -> dict[str, Any]:
    # Strip leading like-verbs so the conversion prefix ("我欣赏") never doubles
    # up ("我欣赏她欣赏真诚..." in the old pipeline).
    liked = [
        _LIKE_VERB_STRIP_RE.sub("", s) or s
        for s in _split_sentences(sections.get("人际喜好", ""))
    ]
    return {
        "liked_traits": [v for v in (_clean_scalar(s) for s in liked) if v],
        "disliked_traits": _split_labeled_items(sections.get("人际厌恶", "")),
    }


def _parse_lifestyle(text: str) -> dict[str, Any]:
    fields = _labeled_values(text, ("作息规律", "卫生习惯", "休闲方式"))
    return {
        "routine": _as_list(fields.get("作息规律")),
        "hygiene": _as_list(fields.get("卫生习惯")),
        "leisure": _as_list(fields.get("休闲方式")),
    }


def _parse_life_events(sections: dict[str, str]) -> dict[str, list[str]]:
    events: dict[str, list[str]] = {}
    for section, key in _LIFE_SECTION_MAP.items():
        items = _split_labeled_items(sections.get(section, ""))
        if not items:
            continue
        events.setdefault(key, []).extend(items)
    return events


def _parse_emotion_events(sections: dict[str, str]) -> dict[str, list[str]]:
    events: dict[str, list[str]] = {}
    for section, key in _EMOTION_SECTION_MAP.items():
        items = _split_labeled_items(sections.get(section, ""))
        if items:
            events[key] = items
    return events


_BELIEF_VERB_STRIP_RE = re.compile(r"^(?:我|她|他)?(?:一直|始终|都)?(?:相信|坚信|认为|觉得)[，,]?\s*")
_OPPOSE_VERB_STRIP_RE = re.compile(r"^(?:我|她|他)?(?:一贯|一直)?反对[，,]?\s*")


def _parse_thoughts(sections: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    goals = _labeled_values(sections.get("理想与目标", ""), ("短期目标（1-3年）", "长期目标（5-10年）"))
    relationships = _labeled_values(sections.get("人际关系观", ""), ("亲情", "友情", "爱情"))
    social = _labeled_values(sections.get("社会观点", ""), ("关于“数字陪伴”", "关于“内卷与躺平”"))
    self_view = _labeled_values(sections.get("自我认知", ""), ("擅长的事情", "绝对不会做的事情", "能力上限"))

    # 价值观: route 反对-sentences into opposes and the rest into believes so
    # nothing is truncated and nothing lands in both lists. The conversion
    # layer re-adds 我相信/我反对 prefixes, so strip the document's own verbs.
    value_sentences = _split_sentences(sections.get("价值观", ""), max_items=12)
    believes = [
        _clean_scalar(_BELIEF_VERB_STRIP_RE.sub("", s)) or s
        for s in value_sentences
        if "反对" not in s
    ]
    opposes = [
        _clean_scalar(_OPPOSE_VERB_STRIP_RE.sub("", s)) or s
        for s in value_sentences
        if "反对" in s
    ]

    values = {
        "motto": _split_sentences(sections.get("人生观", ""), max_items=8),
        "believes": [v for v in believes if v],
        "opposes": [v for v in opposes if v],
        "worldview": _split_sentences(sections.get("世界观", ""), max_items=8),
        # Keep labels: "短期目标（1-3年）：..." — without them the stored memory
        # loses its temporal/topical framing.
        "goal": [f"{label}：{v}" for label, v in goals.items() if v],
        "interpersonal_view": [f"{label}：{v}" for label, v in relationships.items() if v],
        "social_view": [f"{label}：{v}" for label, v in social.items() if v],
        "faith": _split_sentences(sections.get("信仰/精神寄托", ""), max_items=8),
    }
    abilities = {
        "good_at": _split_numbered_items(self_view.get("擅长的事情", "")),
        "never_do": _split_numbered_items(self_view.get("绝对不会做的事情", "")),
        "limits": _split_numbered_items(self_view.get("能力上限", "")),
    }
    return values, abilities


def _labeled_values(text: str, labels: tuple[str, ...]) -> dict[str, str]:
    if not text:
        return {}
    escaped = "|".join(re.escape(label) for label in labels)
    pattern = re.compile(rf"(?P<label>{escaped})\s*[:：]")
    matches = list(pattern.finditer(text))
    values: dict[str, str] = {}
    for idx, match in enumerate(matches):
        label = match.group("label")
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        values[label] = _clean_scalar(text[start:end]) or ""
    return values


def _split_numbered_items(text: str) -> list[str]:
    text = _clean_scalar(text) or ""
    if not text:
        return []
    normalized = re.sub(r"\s+(?=\d+[.、]\s*)", "\n", text)
    if not re.search(r"(?:^|\n)\s*\d+[.、]\s*", normalized):
        return _split_paragraph_items(normalized)
    pieces = re.split(r"(?:^|\n)\s*\d+[.、]\s*", normalized)
    return [_clean_scalar(piece) for piece in pieces if _clean_scalar(piece)]


# Item labels are short event/topic titles. Excluding sentence punctuation
# (。！？) prevents a label from swallowing preceding narrative text (the
# "直到现在，她连…。害怕的氛围——…" bug), and the boundary tolerates a
# closing quote between the period and the next title.
_LABELED_ITEM_RE = re.compile(
    r"(?:^|[。！？\n；;][”」』]?\s*)(?P<label>[^：:\n。！？；;]{2,42})[:：]"
)
# Colon intros that end with a speech/quote verb ("王老师说：", "上面写着：")
# open quoted dialogue inside an event, not a new item.
_LABEL_REJECT_END_RE = re.compile(
    r"(?:说|道|着|问|答|喊|念|回复|附言|留言|备注|配文|评语|写的是|说的是|她|他|我|你)$"
)


def _is_valid_item_label(label: str) -> bool:
    """Distinguish real item titles from narrative clauses ending in a colon.

    Titles are compact noun phrases: at most one comma, no perfective 了
    (aspect markers signal narration like "结果AI回复了她一句诗："), and they
    never end with a speech verb or a bare pronoun.
    """
    if label.count("，") + label.count(",") > 1:
        return False
    if "了" in label:
        return False
    return not _LABEL_REJECT_END_RE.search(label)


def _split_labeled_items(text: str) -> list[str]:
    text = _clean_scalar(text) or ""
    if not text:
        return []
    matches = [
        m for m in _LABELED_ITEM_RE.finditer(text)
        if _is_valid_item_label(m.group("label"))
    ]
    if not matches:
        return _split_paragraph_items(text)
    items: list[str] = []
    lead = _clean_scalar(text[: matches[0].start()])
    if lead:
        # Untitled text before the first label would otherwise be dropped.
        items.extend(_split_paragraph_items(lead))
    for idx, match in enumerate(matches):
        label = _clean_scalar(match.group("label"))
        start = match.end()
        end = matches[idx + 1].start("label") if idx + 1 < len(matches) else len(text)
        body = _clean_scalar(text[start:end].strip("。；; "))
        if label and body:
            items.append(f"{label}：{body}")
        elif label:
            items.append(label)
    return items


# Sentences that state what the persona does NOT like must never enter a
# "likes" list — prefixing them with 我喜欢 inverts their meaning.
_NEGATION_RE = re.compile(r"不喜欢|不太喜欢|不爱|讨厌|反感|受不了|无法接受|不能接受|不能容忍|害怕")
# Anaphoric commentary about the preceding items ("这些颜色让我感到平静") is
# meaningless as a standalone list item.
_COMMENTARY_START_RE = re.compile(r"^(?:这些|这种|这|那些|那种|它们|它)")
_LEADING_CONNECTOR_RE = re.compile(r"^(?:此外|另外|除此之外|同时|还有|以及|毫无疑问是|毫无疑问|当然)[，,、]?\s*")
# "喜爱的歌手包括陈绮贞…" → keep only the enumeration part.
_LIKE_NOUN_INTRO_RE = re.compile(r"^我?喜[欢爱]的[^，。：:]{0,8}?(?:包括|有|是)")

_BRACKET_OPEN = "（(《【“「『"
_BRACKET_CLOSE = "）)》】”」』"


def _split_enum_segments(sentence: str) -> list[str]:
    """Split an enumeration sentence on 、 outside brackets/quotes.

    Refuses to split (returns the whole sentence) when the result does not
    look like a clean enumeration: any segment carrying a clause comma or any
    segment shorter than 4 chars means 、 was joining clauses or tight
    modifiers, not list items.
    """
    segments: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in sentence:
        if ch in _BRACKET_OPEN:
            depth += 1
        elif ch in _BRACKET_CLOSE and depth > 0:
            depth -= 1
        if ch == "、" and depth == 0:
            segments.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    segments.append("".join(buf))
    segments = [s.strip() for s in segments if s.strip()]
    if len(segments) < 2:
        return [sentence]

    def _outside_bracket_text(seg: str) -> str:
        out: list[str] = []
        d = 0
        for ch in seg:
            if ch in _BRACKET_OPEN:
                d += 1
                continue
            if ch in _BRACKET_CLOSE and d > 0:
                d -= 1
                continue
            if d == 0:
                out.append(ch)
        return "".join(out)

    for seg in segments:
        if len(seg) < 5:
            return [sentence]
        if re.search(r"[，,]", _outside_bracket_text(seg)):
            return [sentence]
        # A segment ending with a modifier particle means 、 was chaining
        # adjectives ("洗过的、透明的灰蓝色"), not enumerating items.
        if seg.endswith(("的", "地", "得")):
            return [sentence]
    return segments


def _split_inline_list(text: str) -> list[str]:
    text = _clean_scalar(text) or ""
    if not text:
        return []
    if "：" in text or ":" in text:
        return _split_labeled_items(text)

    items: list[str] = []
    for sentence in re.split(r"(?<=[。！？!?])(?![”」』’])\s*", text):
        sentence = _clean_scalar(sentence.strip("。！？!? ")) or ""
        if not sentence:
            continue
        if _NEGATION_RE.search(sentence):
            continue
        sentence = _LEADING_CONNECTOR_RE.sub("", sentence)
        if _COMMENTARY_START_RE.match(sentence):
            continue
        sentence = _LIKE_NOUN_INTRO_RE.sub("", sentence)
        stripped = _LIKE_VERB_STRIP_RE.sub("", sentence)
        if stripped != sentence:
            stripped = stripped.strip("，, ")
            # After removing the like-verb the remainder must be a noun-ish
            # phrase; grammar particles or anaphora mean the sentence was
            # self-narration that cannot take a "我喜欢X" prefix.
            if not stripped or re.match(r"^(?:把|被|让|对|给|在)", stripped):
                continue
            if _COMMENTARY_START_RE.match(stripped):
                continue
            sentence = stripped
        elif re.match(r"^我", sentence):
            # Self-narration without a like-verb ("我很少...") cannot be
            # safely prefixed either.
            continue
        for segment in _split_enum_segments(sentence):
            segment = _clean_scalar(segment)
            if segment:
                items.append(segment)
    return items


def _split_paragraph_items(text: str) -> list[str]:
    text = _clean_scalar(text) or ""
    if not text:
        return []
    lines = [_clean_scalar(line) for line in text.split("\n")]
    lines = [line for line in lines if line]
    if len(lines) > 1:
        return lines
    return _split_sentences(text, max_items=8) or [text]


def _split_sentences(text: str, max_items: int = 8) -> list[str]:
    text = _clean_scalar(text) or ""
    if not text:
        return []
    # Do not split between a period and its closing quote — "……。”她相信"
    # belongs to one motto/claim, not two.
    parts = re.split(r"(?<=[。！？!?])(?![”」』’])\s*", text)
    items = [_clean_scalar(part) for part in parts if _clean_scalar(part)]
    return items[:max_items] if items else [text]


def _as_list(value: str | None) -> list[str]:
    value = _clean_scalar(value)
    return [value] if value else []


def _first_line(text: str) -> str | None:
    for line in text.splitlines():
        value = _clean_scalar(line)
        if value:
            return value
    return None


def _first_int(text: str) -> int | None:
    match = re.search(r"\d+", text or "")
    return int(match.group(0)) if match else None


def _first_match(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text or "")
    return _clean_scalar(match.group(1)) if match else None


def _clean_scalar(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text if text else None


def _strip_parenthetical(value: str | None) -> str | None:
    value = _clean_scalar(value)
    if not value:
        return None
    return re.sub(r"[（(].*$", "", value).strip() or value


def _parenthetical_note(value: str | None) -> str | None:
    """The trailing （...） explanation dropped by _strip_parenthetical."""
    value = _clean_scalar(value)
    if not value:
        return None
    match = re.search(r"[（(](.+?)[）)]?\s*$", value)
    return _clean_scalar(match.group(1)) if match else None


def _build_background(profile: dict[str, Any], career: dict[str, Any]) -> str:
    identity = profile.get("identity", {}) if isinstance(profile.get("identity"), dict) else {}
    bits = [
        _clean_scalar(identity.get("name")),
        f"{identity.get('age')}岁" if identity.get("age") else None,
        _clean_scalar(identity.get("gender")),
        _clean_scalar(career.get("title")),
        _clean_scalar(identity.get("location")),
    ]
    headline = "，".join(bit for bit in bits if bit)
    values = profile.get("values", {}) if isinstance(profile.get("values"), dict) else {}
    motto = ""
    if isinstance(values.get("motto"), list) and values["motto"]:
        motto = str(values["motto"][0])
    return f"文档导入角色档案：{headline}。{motto}".strip()
