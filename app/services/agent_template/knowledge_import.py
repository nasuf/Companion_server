"""Parse a knowledge-supplement document into memory-ready items.

Knowledge documents extend an EXISTING template's memory bank with world facts
the persona should know (company / product / event information), as opposed to
the five-dimension persona profile that CREATES a template (document_import.py).
Expected format — short section headings + "标签：内容" lines:

    公司介绍
    公司名称：伴生
    公司定位：陪伴科技公司
    合作项目介绍
    项目名称：2026年恒洁杯第二十届佛山“西甲”足球联赛
    赛事时间：2026年7月10日至8月23日

Parsing is deterministic (no LLM). Every labeled line becomes ONE memory item;
the stored summary is made self-contained by prefixing the section subject
(the section's 「XX名称」 value when present, else the section title) so vector
retrieval hits standalone questions ("西甲什么时候开始") whose tokens only
appear in the section subject, not in the line itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.agent_template.document_import import (
    _section_heading,  # five-dimension heading detector (same package)
    extract_document_text,
)

# Hard cap per upload — protects the template memory bank (and the downstream
# sync fan-out to every cloned agent) from an accidentally huge document.
MAX_KNOWLEDGE_ITEMS = 200

# Lines without a colon are treated as section headings only when short;
# longer colon-less lines are free-form content that joins the current section.
_MAX_HEADING_LEN = 24

# A label like 公司名称/项目名称 names the section subject that prefixes the
# other lines of the same section.
_NAME_LABEL_SUFFIX = "名称"

# Suffixes stripped from a section title used as fallback subject
# ("合作项目介绍" → "合作项目").
_TITLE_NOISE_SUFFIXES = ("介绍", "简介", "信息", "说明")

# "1. 公司介绍" / "2、赛事时间：…" enumeration prefixes.
_ENUM_PREFIX_RE = re.compile(r"^\d+[.、]\s*")

# Five-dimension section headings this many times = the wrong document type.
_PROFILE_HEADING_REJECT_THRESHOLD = 3


@dataclass(frozen=True)
class KnowledgeItem:
    """One parsed knowledge fact.

    ``summary`` is the final self-contained text stored as the memory row
    (content == summary); section/label are kept for admin UI preview.
    """

    section: str
    label: str
    content: str
    summary: str


def parse_knowledge_document(data: bytes, filename: str | None = None) -> list[KnowledgeItem]:
    """Parse uploaded bytes into knowledge items ready for memory storage."""
    text = extract_document_text(data, filename=filename)
    _reject_profile_document(text)
    sections = _split_sections(text)
    items = _build_items(sections)
    if not items:
        raise ValueError("未识别到任何知识条目：请使用「小节标题 + 标签：内容」的行式格式")
    if len(items) > MAX_KNOWLEDGE_ITEMS:
        raise ValueError(f"知识条目过多（{len(items)} 条），单次上传最多 {MAX_KNOWLEDGE_ITEMS} 条")
    return items


def _reject_profile_document(text: str) -> None:
    """A five-dimension persona profile must go through template CREATION.

    Appending it as knowledge would bypass the singleton / coverage rules of
    the L1 provisioning pipeline (e.g. a second 姓名 row), so refuse early and
    point the admin at the right flow.
    """
    hits = 0
    for raw in text.splitlines():
        if _section_heading(raw.strip()):
            hits += 1
            if hits >= _PROFILE_HEADING_REJECT_THRESHOLD:
                raise ValueError(
                    "检测到五维人格档案标题：五维档案请使用「从 TXT 创建模板」；"
                    "记忆补充请上传「标签：内容」格式的知识文档"
                )


def _split_colon(line: str) -> tuple[str, str] | None:
    """Split at the earliest full/half-width colon; None when no colon."""
    indices = [i for i in (line.find("："), line.find(":")) if i >= 0]
    if not indices:
        return None
    idx = min(indices)
    return line[:idx].strip(), line[idx + 1 :].strip()


def _split_sections(text: str) -> list[tuple[str, list[tuple[str, str]]]]:
    """Group lines into ``[(section_title, [(label, content), ...])]``.

    Rules (deterministic, order-preserving):
    - leading ``#`` markers and ``1.``/``1、`` enumeration prefixes are noise
    - a short colon-less line (or a "标题：" line with empty value) opens a
      new section
    - "标签：内容" becomes an item of the current section; when the "label"
      is over-long the colon sits inside a sentence, so the whole line is
      kept as a label-less content item instead
    - long colon-less lines are label-less content items too
    """
    sections: list[tuple[str, list[tuple[str, str]]]] = []

    def _open_section(title: str) -> None:
        sections.append((title, []))

    def _current_entries() -> list[tuple[str, str]]:
        if not sections:
            _open_section("")
        return sections[-1][1]

    for raw in text.splitlines():
        line = raw.strip().lstrip("#").strip()
        line = _ENUM_PREFIX_RE.sub("", line)
        if not line:
            continue
        parts = _split_colon(line)
        if parts is None:
            if len(line) <= _MAX_HEADING_LEN:
                _open_section(line)
            else:
                _current_entries().append(("", line))
            continue
        label, content = parts
        if not content:
            # "公司介绍：" — a heading written with a trailing colon.
            if label:
                _open_section(label)
            continue
        if not label or len(label) > _MAX_HEADING_LEN:
            _current_entries().append(("", line))
            continue
        _current_entries().append((label, content))
    return sections


def _section_subject(title: str, entries: list[tuple[str, str]]) -> str:
    """The subject used to make each line self-contained.

    Prefer the section's 「XX名称」 value ("伴生App"); fall back to the section
    title with descriptive suffixes stripped ("合作项目介绍" → "合作项目").
    """
    for label, content in entries:
        if label.endswith(_NAME_LABEL_SUFFIX) and content:
            return content
    subject = title.strip()
    for suffix in _TITLE_NOISE_SUFFIXES:
        if subject.endswith(suffix) and len(subject) > len(suffix):
            subject = subject[: -len(suffix)]
            break
    return subject.strip()


def _build_items(sections: list[tuple[str, list[tuple[str, str]]]]) -> list[KnowledgeItem]:
    items: list[KnowledgeItem] = []
    seen: set[str] = set()
    for title, entries in sections:
        subject = _section_subject(title, entries)
        for label, content in entries:
            if not content:
                continue
            base = f"{label}：{content}" if label else content
            if subject and subject not in base:
                # Prefix the subject so the line survives standalone retrieval
                # ("赛事时间：7月10日" alone would never match a 西甲 query).
                summary = f"{subject}的{label}：{content}" if label else f"{subject}：{content}"
            else:
                summary = base
            if summary in seen:
                continue
            seen.add(summary)
            items.append(
                KnowledgeItem(section=title, label=label, content=content, summary=summary)
            )
    return items
