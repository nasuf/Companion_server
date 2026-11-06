"""Rolling "relevant memories" note sent with each reply of a call.

Each user turn's retrieval lands here; the bridge sends the rendered block
as a system message just before the reply and deletes it once the reply is
done. DashScope reads instructions changed mid-call only for later sessions,
so a per-turn item is the only way to reach the very next reply (verified
2026-11-06). Entries outlive their turn for ``ttl_turns`` turns, because
follow-ups usually stay on the topic just retrieved.

The base instructions are fixed for the call; they are kept here only to
skip memories already present in the core block.
"""

from __future__ import annotations

from dataclasses import dataclass

_HEADER = (
    "【此刻相关的记忆】\n"
    "下面是和刚才聊的内容有关的记忆，按需自然带出，用你自己的话说；"
    "和当前话题无关就忽略，不要照读，也不要提“记忆”两个字。"
)
# Raw-text prefix that marks a memory as already present in the core block.
_DEDUP_PREFIX = 24
# Lines the agent has said twice; the newest few are named so it stops.
_MAX_AVOIDED = 2


@dataclass
class _Entry:
    key: str
    line: str
    turn: int


class MemorySlot:
    """The rolling memory block and the lines the agent should not repeat.

    Entries are keyed by memory id (or text). A memory hit again is
    refreshed; one not hit for ``ttl_turns`` user turns drops out. The block
    keeps at most ``max_lines`` entries within ``char_budget``, most recently
    hit first.
    """

    def __init__(
        self,
        base_instructions: str,
        *,
        max_lines: int = 6,
        char_budget: int = 700,
        ttl_turns: int = 3,
    ) -> None:
        self._base = base_instructions
        self._max_lines = max_lines
        self._char_budget = char_budget
        self._ttl_turns = ttl_turns
        self._entries: list[_Entry] = []
        self._rendered = ""
        self._avoided: list[str] = []

    @property
    def lines(self) -> list[str]:
        return [entry.line for entry in self._entries]

    def note(self) -> str:
        """Per-reply context: the memory block and any lines not to repeat."""
        parts = [self._rendered]
        if self._avoided:
            quoted = "、".join(f"「{line}」" for line in self._avoided)
            parts.append(f"【别再重复】你这通电话里已经说过{quoted}，接下来换个说法或换个话题，不要再说这句。")
        return "\n\n".join(part for part in parts if part)

    def avoid(self, line: str) -> bool:
        """Name a line the agent keeps repeating. True when it is new."""
        line = line.strip()
        if not line or line in self._avoided:
            return False
        self._avoided = [*self._avoided, line][-_MAX_AVOIDED:]
        return True

    def merge(self, memories: list[dict[str, str]], turn: int) -> bool:
        """Add one retrieval's memories. True when the block text changed."""
        fresh: list[_Entry] = []
        for memory in memories:
            line = str(memory.get("line") or "").strip()
            raw = str(memory.get("raw") or "").strip()
            if not line or self._in_base(raw):
                continue
            key = str(memory.get("id") or "") or raw or line
            if any(entry.key == key for entry in fresh):
                continue
            fresh.append(_Entry(key=key, line=line, turn=turn))
        keys = {entry.key for entry in fresh}
        self._entries = fresh + [e for e in self._entries if e.key not in keys]
        return self._render(turn)

    def expire(self, turn: int) -> bool:
        """Drop entries not hit recently. True when the block text changed."""
        return self._render(turn)

    def _in_base(self, raw: str) -> bool:
        return bool(raw) and raw[:_DEDUP_PREFIX] in self._base

    def _render(self, turn: int) -> bool:
        alive = [e for e in self._entries if turn - e.turn < self._ttl_turns]
        kept: list[_Entry] = []
        used = 0
        for entry in alive:
            if len(kept) >= self._max_lines:
                break
            if kept and used + len(entry.line) > self._char_budget:
                break
            kept.append(entry)
            used += len(entry.line)
        self._entries = kept
        rendered = (
            _HEADER + "\n" + "\n".join(f"- {e.line}" for e in kept) if kept else ""
        )
        changed = rendered != self._rendered
        self._rendered = rendered
        return changed
