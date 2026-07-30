"""把 LongMemEval 的时间推理子集转成我们检索层的外部标注集.

LongMemEval (Wu et al., ICLR 2025, arXiv 2410.10813) 把长期记忆拆成五种能力,
其中 **temporal reasoning** 是我们目前唯一没有外部集验证的一维 —— 已接入的
DuLeMon 在论文的能力对比表里那一列是 ✗, 它测 persona 使用, 测不到时间。

为什么必须用外部集: 我们自建的 `evals/temporal_recall` ground truth 是自己定的,
而生产数据也用不了 —— 测试用户平均每人 45 条消息、跨度几天, **根本没有可追问的
过去**。时间类提问的前提是有历史, 拿几天的对话去测时间能力, 测到的是"没有历史"
而不是"没有能力"。LongMemEval 正是为此构造 30-40 个 session / 115k token 的历史。

## 接法

论文把记忆设计拆成 indexing / retrieval / reading 三段, 我们这里只测前两段 ——
`has_answer` 标在**轮次**上, 正好对应"该被注入 prompt 的那一轮有没有被召回"。
reading 段 (LLM 读完给答案) 不测: 那考的是主模型, 跟我们的记忆设计无关。

value 粒度取 **round** (一问一答), 与论文 Table 4 的 "Value = Round" 对齐, 数字
可以直接比。论文该列 baseline 是 Recall@5 = 0.421 (K=V), 加事实扩展后 0.489。

刻意**不**跑我们的抽取管线: 133 道题 × 30-40 session × 12 轮 ≈ 5 万次 LLM 调用,
成本不成比例; 而且抽取质量会跟检索质量混在一起, 测不出是哪一层的问题。这里只测
embedding + 排序 + 时间过滤这三件我们能直接改的事。

## 已知偏差

数据是**英文**的。它能测架构 (时间戳索引、排序、召回), 测不到 `time_parser.py`
那些中文正则 —— 中文那半仍要靠自建集。另外题型以 order (哪个先) 和 duration
(间隔几天) 为主, 跟中文口语里常见的"这个月/上次"分布不同, 用的时候要记住这点。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# LongMemEval 的日期形如 "2023/04/10 (Mon) 17:50"
_DATE_FMT = "%Y/%m/%d (%a) %H:%M"

# 单轮截断长度。数据里 20% 的轮次超过 2000 字符 (最长 11604), 整轮直接嵌入会把本地
# embedder 打挂 —— 但更重要的是**不截才是失真的**: 我们的系统绝不会把一段 3000 字
# 的对话整条存成记忆, 它会抽成若干条短事实, 单条上限 180 token (英文约 720 字符)。
# 按这个量级截, 测出来的才是我们实际会检索的东西。
MAX_TURN_CHARS = 800


def parse_lme_date(raw: str) -> datetime | None:
    try:
        return datetime.strptime(raw.strip(), _DATE_FMT)
    except (ValueError, AttributeError):
        return None


@dataclass
class Round:
    """一问一答, 检索的最小单位 (论文的 Value = Round)."""

    id: str
    text: str
    at: datetime | None
    session_index: int
    is_evidence: bool


@dataclass
class TemporalCase:
    question_id: str
    question: str
    answer: str
    asked_at: datetime | None
    rounds: list[Round] = field(default_factory=list)

    @property
    def evidence_ids(self) -> set[str]:
        return {r.id for r in self.rounds if r.is_evidence}


def _chunks_of_turn(content: str, size: int = 240) -> list[str]:
    """把一条长发言按句子切成小块.

    为什么需要: 数据里一轮对话经常是"顺带提一句 A + 大段讲 B"。实测某条证据 1310
    字符, 绝大部分在讲电视挂架, 而答案 (Nordstrom 特卖会) 只是第 242 字符处的一句
    —— 整轮的 embedding 被电视挂架主导, 问 Nordstrom 根本匹配不上 (相似度 0.307,
    排 13 名)。

    这不是模型不行, 是**检索单位太粗**。我们的生产管线会把那句话抽成一条独立的短
    记忆, 所以用整轮做单位测出来的分数不代表我们的架构。切块是不调 LLM 的近似。
    """
    import re

    sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+", content) if s.strip()]
    chunks: list[str] = []
    buf = ""
    for s in sentences:
        if buf and len(buf) + len(s) > size:
            chunks.append(buf)
            buf = s
        else:
            buf = f"{buf} {s}".strip()
    if buf:
        chunks.append(buf)
    return chunks or [content]


def _chunk_units_of_session(
    session: list[dict], session_index: int, at: datetime | None, qid: str
) -> list[Round]:
    """按"句子块"建索引单位, 近似我们抽取后的记忆粒度."""
    units: list[Round] = []
    for ti, turn in enumerate(session):
        role = "User" if turn.get("role") == "user" else "Assistant"
        for ci, chunk in enumerate(_chunks_of_turn(turn.get("content") or "")):
            units.append(
                Round(
                    id=f"{qid}:s{session_index}:t{ti}:c{ci}",
                    text=f"{role}: {chunk}",
                    at=at,
                    session_index=session_index,
                    # 证据标在轮次上, 该轮切出的每一块都算候选证据 —— 只要任一块
                    # 被召回, 那句事实就进了 prompt。
                    is_evidence=bool(turn.get("has_answer")),
                )
            )
    return units


def _rounds_of_session(
    session: list[dict], session_index: int, at: datetime | None, qid: str
) -> list[Round]:
    """把一个 session 切成 round.

    一个 round = 一条 user 消息 + 紧随其后的 assistant 回复。带 `has_answer` 的
    轮次标成 evidence —— 注意标签可能落在 user 侧也可能落在 assistant 侧, 两者
    任一命中都算这个 round 是证据 (论文的 turn-level recall 就是这么算的)。
    """
    rounds: list[Round] = []
    buf: list[dict] = []

    def flush() -> None:
        if not buf:
            return
        text = "\n".join(
            f"{'User' if t.get('role') == 'user' else 'Assistant'}: "
            f"{(t.get('content') or '')[:MAX_TURN_CHARS]}"
            for t in buf
        )
        rounds.append(
            Round(
                id=f"{qid}:s{session_index}:r{len(rounds)}",
                text=text,
                at=at,
                session_index=session_index,
                is_evidence=any(t.get("has_answer") for t in buf),
            )
        )
        buf.clear()

    for turn in session:
        if turn.get("role") == "user" and buf:
            flush()
        buf.append(turn)
    flush()
    return rounds


def load_temporal_cases(
    path: Path, limit: int | None = None, granularity: str = "round"
) -> list[TemporalCase]:
    """读 longmemeval_{oracle,s_cleaned}.json, 只取 temporal-reasoning 题.

    granularity 决定检索单位:
        round  一问一答整轮 —— 论文 Table 4 的 "Value = Round" 基线
        chunk  句子块 —— 近似我们生产管线抽取后的记忆粒度

    这个选项不是调参, 是**测的对象不一样**。实测同一条证据在 round 粒度相似度
    0.307 排 13 名, 切成 chunk 后 0.727 排第 1 —— 整轮的向量被同轮里占比更大的
    另一个话题稀释了。我们的系统存的是抽取后的原子事实, 用 round 测等于测了一个
    我们并不采用的设计。
    """
    data = json.loads(Path(path).read_text())
    cases: list[TemporalCase] = []
    for item in data:
        if item.get("question_type") != "temporal-reasoning":
            continue
        qid = item["question_id"]
        dates = item.get("haystack_dates") or []
        rounds: list[Round] = []
        split = _chunk_units_of_session if granularity == "chunk" else _rounds_of_session
        for i, session in enumerate(item.get("haystack_sessions") or []):
            at = parse_lme_date(dates[i]) if i < len(dates) else None
            rounds.extend(split(session, i, at, qid))
        if not any(r.is_evidence for r in rounds):
            # 没有证据标注的题测不了召回 —— 跳过而不是当成失败, 否则分数被
            # 数据缺陷拉低, 看不出系统真实水平。
            continue
        cases.append(
            TemporalCase(
                question_id=qid,
                question=item["question"],
                answer=str(item.get("answer", "")),
                asked_at=parse_lme_date(item.get("question_date", "")),
                rounds=rounds,
            )
        )
        if limit and len(cases) >= limit:
            break
    return cases


def parent_turn(unit_id: str) -> str:
    """把 chunk id 收敛到它所属的轮次.

    chunk 粒度下一轮会切出好几块, 全标成证据的话"证据条数"就跟 round 粒度不是一个
    口径, 两个模式的分数不能比。按所属轮次去重后再计分, 衡量的都是"该注入的那一轮
    内容有没有被召回", 口径一致。

    已知宽松之处: 召回同一轮里**不含答案**的那一块也会被记成命中。要精确到块需要
    知道答案落在哪一块, 而数据只标到轮 —— 这个偏差对两种粒度不对称 (只影响 chunk),
    所以 chunk 的分数要当成上界看。
    """
    return unit_id.rsplit(":c", 1)[0] if ":c" in unit_id else unit_id


def _to_turns(ids: list[str]) -> list[str]:
    """保序去重到轮次级."""
    seen: set[str] = set()
    out: list[str] = []
    for i in ids:
        t = parent_turn(i)
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def recall_at_k(ranked_ids: list[str], evidence: set[str], k: int) -> float:
    """top-k 里覆盖了多少比例的证据轮次.

    用覆盖率而不是"命中任意一条": order 题 ("X 和 Y 哪个先") 需要**两条都召回**
    主模型才比得了, 只召回一条等于答不出。
    """
    if not evidence:
        return 0.0
    ev = {parent_turn(e) for e in evidence}
    return len(set(_to_turns(ranked_ids)[:k]) & ev) / len(ev)


def all_evidence_at_k(ranked_ids: list[str], evidence: set[str], k: int) -> bool:
    """证据是否**全部**进了 top-k —— 多跳题的真实达标线."""
    if not evidence:
        return False
    ev = {parent_turn(e) for e in evidence}
    return ev <= set(_to_turns(ranked_ids)[:k])
