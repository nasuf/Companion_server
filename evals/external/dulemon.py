"""把 DuLeMon 转成我们两个闸门的外部标注集.

DuLeMon (Xu et al., ACL 2022, 百度) 是中文长期记忆对话数据集, 27.5k 对话 /
449k 话语, 逐句人工标注了"这句话用到了哪条 persona". 我们自己的标注集是手写的,
ground truth 取自提示词自己声明的规则 —— 它测的是指令遵循度, 测不出策略本身
错在哪. DuLeMon 的标签由第三方人工给出, 与我们的提示词无关, 正好补这一面.

标注格式: 每句 `Usr:`/`Bot:` 后可跟制表符 + persona 编号 (U1/B3...).

    "Bot: 孩子 多 大 了 ？\tU3"   ← 这句回复用了用户 persona U3
    "Usr: 我 可是 个 壮年 男子\tU6" ← 用户在陈述一条 persona 事实

两个标签由此导出:

  相关度闸门 — 看用户第 t 句之后机器人的回复挂没挂标签. 挂了说明人工判定"这轮
    回复确实用上了长期记忆", 那第 t 句就该判中/强; 没挂就判弱.
  记忆预筛 — 看用户句自身挂没挂标签. 挂了说明这句在陈述一条值得长期保存的
    persona 事实, 该判"记".

已知偏差, 用的时候必须记住: DuLeMon 是众包写出来的"长期记忆对话", 写手被要求
围绕 persona 展开, persona 密度远高于真实闲聊. 所以绝对准确率会偏乐观, 它适合
做**相对比较** (新旧提示词谁强 / 换模型退不退化), 不适合当绝对达标线 —— 绝对
水位仍要靠生产采样.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

_TAG = re.compile(r"^(Usr|Bot):\s*(.*?)(?:\t(.+))?$")


@dataclass(frozen=True)
class Turn:
    speaker: str
    text: str
    personas: tuple[str, ...]


_CJK = r"\u4e00-\u9fff"
_PUNCT = r"，。！？、；：“”‘’（）《》~…—"


def _detokenize(text: str) -> str:
    """DuLeMon 的文本是分词后用空格连接的.

    中文字符之间和标点两侧的空格都要去掉, 但拉丁字母/数字之间的必须留着
    (`IT 部门`, `15 年`) —— 送进闸门的文本得读起来像真人打的字, 否则测的是
    模型对分词残留的鲁棒性, 不是判定能力.
    """
    out = re.sub(rf"(?<=[{_CJK}])\s+(?=[{_CJK}])", "", text)
    out = re.sub(rf"\s+(?=[{_PUNCT}])", "", out)
    out = re.sub(rf"(?<=[{_PUNCT}])\s+", "", out)
    return re.sub(r"\s{2,}", " ", out).strip()


def _parse_turn(raw: str) -> Turn | None:
    match = _TAG.match(raw.strip())
    if not match:
        return None
    speaker, text, tags = match.groups()
    personas = tuple(t.strip() for t in (tags or "").split() if t.strip())
    return Turn(speaker, _detokenize(text), personas)


def load_dialogues(path: Path) -> list[list[Turn]]:
    dialogues = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        turns = [t for t in (_parse_turn(r) for r in record["conversation"]) if t]
        if turns:
            dialogues.append(turns)
    return dialogues


@dataclass(frozen=True)
class RelevanceCase:
    message: str
    expected: str          # 中 (该查) / 弱 (不必查)
    context: str           # 前两轮, 给省略式追问一点上下文
    grounded_on: tuple[str, ...]


def _user_side(personas: tuple[str, ...]) -> bool:
    """标签只保留机器人回忆**用户**信息的那类 (U*), 丢掉它调用自身人设的 (B*).

    这一刀是让 DuLeMon 能用的关键. 不加的话, "嗨，小度，在吗？" 会被标成该查
    —— 因为机器人的问候回复引用了自己的 B-persona. 但在我们的架构里 AI 自我
    人设由 L1 常驻注入, 不经过相关度闸门, 这类句子判弱本来就是对的. 混进来只会
    让闸门在一批纯招呼上"漏检", 得出反向结论.
    """
    return any(p.startswith("U") for tag in personas for p in tag.split(","))


def build_relevance_cases(
    dialogues: list[list[Turn]], *, limit: int | None = None, seed: int = 0,
    min_chars: int = 3, user_side_only: bool = True,
) -> list[RelevanceCase]:
    """用户第 t 句 → 机器人 t+1 句是否回忆了用户信息, 决定该不该查记忆."""
    cases: list[RelevanceCase] = []
    for turns in dialogues:
        for i, turn in enumerate(turns[:-1]):
            if turn.speaker != "Usr":
                continue
            reply = turns[i + 1]
            if reply.speaker != "Bot":
                continue
            if len(turn.text) < min_chars:
                continue
            grounded = (
                _user_side(reply.personas) if user_side_only else bool(reply.personas)
            )
            # B-persona only: 机器人只用了自身人设, 对我们的闸门既不算该查也不算
            # 不该查 —— 丢掉, 不要污染任一侧.
            if user_side_only and reply.personas and not grounded:
                continue
            context = "\n".join(
                f"{'用户' if t.speaker == 'Usr' else 'AI'}: {t.text}"
                for t in turns[max(0, i - 2):i]
            )
            cases.append(RelevanceCase(
                message=turn.text,
                expected="中" if grounded else "弱",
                context=context,
                grounded_on=reply.personas,
            ))
    if limit is not None and len(cases) > limit:
        cases = random.Random(seed).sample(cases, limit)
    return cases


@dataclass(frozen=True)
class JudgementCase:
    message: str
    expected: str          # 记 / 不记
    personas: tuple[str, ...]


def build_judgement_cases(
    dialogues: list[list[Turn]], *, limit: int | None = None, seed: int = 0,
    min_chars: int = 3,
) -> list[JudgementCase]:
    """用户句自身挂了 persona 标签 = 在陈述一条值得长期保存的事实."""
    cases = [
        JudgementCase(t.text, "记" if t.personas else "不记", t.personas)
        for turns in dialogues for t in turns
        if t.speaker == "Usr" and len(t.text) >= min_chars
    ]
    if limit is not None and len(cases) > limit:
        cases = random.Random(seed).sample(cases, limit)
    return cases
