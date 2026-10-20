"""Bounded, conservative syntax binding for read queries, never event truth.

The legacy calendar parser also serves scheduling. Keep its behaviour intact:
reconstruct repeated occurrences here, and map normalised aliases back to the
original message. Trace contains offsets only, never a second copy of text.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.schedule_domain.time_parser import ParsedTime


@dataclass(frozen=True)
class SourceText:
    text: str
    offsets: tuple[tuple[int, int], ...]

    @classmethod
    def original(cls, text: str) -> SourceText:
        return cls(text, tuple((i, i + 1) for i in range(len(text))))

    def substitute(self, pattern: re.Pattern, replacement) -> SourceText:
        pieces, offsets, cursor = [], [], 0
        for match in pattern.finditer(self.text):
            pieces.append(self.text[cursor:match.start()])
            offsets.extend(self.offsets[cursor:match.start()])
            value = replacement(match)
            pieces.append(value)
            span = (self.offsets[match.start()][0], self.offsets[match.end() - 1][1])
            offsets.extend([span] * len(value))
            cursor = match.end()
        pieces.append(self.text[cursor:])
        offsets.extend(self.offsets[cursor:])
        return SourceText("".join(pieces), tuple(offsets))


@dataclass(frozen=True)
class DatePart:
    parsed: ParsedTime
    start: int
    end: int


def locate_parts(source: SourceText, parsed: list[ParsedTime], limit: int) -> tuple[list[DatePart], str | None]:
    """Longest matches win, as in the calendar parser; repeated text stays distinct."""
    by_text = {p.original_text: p for p in parsed if p.original_text}
    candidates = []
    for text, item in by_text.items():
        for match in re.finditer(re.escape(text), source.text):
            candidates.append(DatePart(item, match.start(), match.end()))
    # Input is capped by the caller. Stop at limit+1 rather than accumulating
    # unbounded output or silently dropping occurrences.
    selected = []
    for part in sorted(candidates, key=lambda p: (p.start, -(p.end - p.start))):
        if selected and part.start < selected[-1].end:
            continue
        selected.append(part)
        if len(selected) > limit:
            return [], "expression_limit"
    if not selected or len(by_text) != len({p.parsed.original_text for p in selected}):
        return [], "source_span_unavailable"
    return selected, None


_NONFACTUAL = re.compile(
    r"不是|并非|并不|没有|没去|没在|未曾|不曾|否认|如果|假如|假设|假使|倘若|要是|"
    r"本来|原本|本想|原定|改期|取消|改成|更正|说错|记错|梦见|梦到"
)
_QUOTES = re.compile(r"[“”‘’「」『』\"']")


def binding_uncertainty(text: str) -> str | None:
    """Conservative gates, not a claim to resolve negation or quoted speakers."""
    if _QUOTES.search(text):
        return "quoted_expression"
    if _NONFACTUAL.search(text):
        return "nonfactual_or_corrected"
    return None


def binding_diagnostics(source: SourceText, parts: list[DatePart], decision: str) -> dict:
    return {
        "version": 1, "decision": decision, "offset_unit": "unicode_codepoint",
        "spans": [{"start": source.offsets[p.start][0], "end": source.offsets[p.end - 1][1]}
                  for p in parts],
    }
