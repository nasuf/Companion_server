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


def load_temporal_cases(path: Path, limit: int | None = None) -> list[TemporalCase]:
    """读 longmemeval_{oracle,s_cleaned}.json, 只取 temporal-reasoning 题."""
    data = json.loads(Path(path).read_text())
    cases: list[TemporalCase] = []
    for item in data:
        if item.get("question_type") != "temporal-reasoning":
            continue
        qid = item["question_id"]
        dates = item.get("haystack_dates") or []
        rounds: list[Round] = []
        for i, session in enumerate(item.get("haystack_sessions") or []):
            at = parse_lme_date(dates[i]) if i < len(dates) else None
            rounds.extend(_rounds_of_session(session, i, at, qid))
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


def recall_at_k(ranked_ids: list[str], evidence: set[str], k: int) -> float:
    """top-k 里覆盖了多少比例的证据轮次.

    用覆盖率而不是"命中任意一条": order 题 ("X 和 Y 哪个先") 需要**两条都召回**
    主模型才比得了, 只召回一条等于答不出。
    """
    if not evidence:
        return 0.0
    return len(set(ranked_ids[:k]) & evidence) / len(evidence)


def all_evidence_at_k(ranked_ids: list[str], evidence: set[str], k: int) -> bool:
    """证据是否**全部**进了 top-k —— 多跳题的真实达标线."""
    return bool(evidence) and evidence <= set(ranked_ids[:k])
