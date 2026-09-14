"""V3 主动交流·三档话题源分类 (2026-09-14).

## 背景

V0 现状 (feat ba6dd42): 一个 append_trending_section 把 tavily 抓来的所有内容
一股脑塞进 prompt 尾, 消息 LLM 拿到"如合适可自然带一句"的软指令 —— eval 实测
source_fit 17%, persona_match 21%, 消息 90% 无视 trending 生成通用寒暄.

## V3 思路

真人朋友主动分享话题至少有三种合法来源, 每种表达完全不同:

  user_interest_match  "你不是说过X吗, 刚看到Y"        (勾用户)
  ai_persona_match     "我最近在X" / "我在X, 你听过吗"  (AI 视角分享)
  socially_hot         "刷到个X, 挺有意思的"            (公共谈资)

不该用单一 prompt 让 LLM 猜, 而是**先分类, 再走对应 prompt**. 分类基于:
  - 内容候选 (trending fetch 的结果) 跟哪个语义空间重叠最多
  - 用户 portrait 兴趣 vs AI 人设兴趣 vs 都不搭

## 分类算法 (纯规则, 无 LLM 调用)

  1. 提取 user_portrait 里的兴趣词 (简单分词, 抠出"喜欢/爱好/常/迷"后面的名词短语)
  2. 提取 agent 兴趣词 (background + occupation + lifeOverview 里同法抠)
  3. 对每个 trending candidate, 打分 (user_hits, ai_hits, is_hot_quality)
  4. 汇总: 哪一档得分最高 → 选那一档; 都很弱 → "none"

规则化的理由: 决策链清晰可测, 不再叠 LLM 调用 (proactive 主流程已经调 chat model
生成消息了, 分类再叠一次会增加 1-2s 延迟 + 一次评估失败点); 且规则化让 eval 里
V3 vs V0 的对比不受"分类 LLM"这层随机性污染. 未来可以升级到 embedding 相似度
甚至 LLM classifier, 但先证明"分档 + 独立 prompt"这条路走得通再优化.

## 与生产 sender 的接入

- feat/proactive_trending_v3_dispatch flag (默认 False) 控制启用
- 启用后: 分类 → 塞 ctx["topic_source_kind"] → sender._generate_message
  按 source_kind 挑对应 prompt (proactive.trending_user_interest / _ai_persona /
  _socially_hot); source_kind="none" 走老的 silence_plain
- V0 code path 不动, 打开 flag 才走 V3
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

TopicSourceKind = Literal[
    "user_interest_match", "ai_persona_match", "socially_hot", "none",
]

# 从自由文本里抠兴趣词的正则: "喜欢/爱好/常/迷/关注" 后面到句读为止的短语.
# 例: "喜欢摄影和露营" → 摄影, 露营.
_INTEREST_HOOK = re.compile(
    r"(?:喜欢|爱好|常|经常|迷上?|关注|热爱|痴迷|沉迷)"
    r"([一-龥A-Za-z0-9/、和与及,，\s]{1,40})"
)
# 分隔符, 兴趣词组切开
_INTEREST_SPLIT = re.compile(r"[、和与及,，/\s]+")


def _extract_interests(text: str, max_terms: int = 6) -> list[str]:
    """从 portrait / background / lifeOverview 里抠兴趣关键词."""
    if not text:
        return []
    terms: list[str] = []
    for m in _INTEREST_HOOK.finditer(text):
        chunk = m.group(1).strip()
        for term in _INTEREST_SPLIT.split(chunk):
            t = term.strip()
            if 2 <= len(t) <= 12 and t not in terms:  # 2-12 字, 去重
                terms.append(t)
                if len(terms) >= max_terms:
                    return terms
    return terms


def _count_hits(candidate: dict, terms: list[str]) -> int:
    """一条候选内容跟兴趣词有多少个命中. 简单子串, 不做分词."""
    if not terms:
        return 0
    haystack = (str(candidate.get("title") or "")
                + " " + str(candidate.get("snippet") or ""))
    return sum(1 for t in terms if t in haystack)


# 已知负面/敏感内容黑名单 (socially_hot 档必过). 命名"blocked"是因为我们
# 明确拒绝: 名人死讯/事故/凶杀/自杀/政治敏感/性/歧视都不是朋友主动谈资.
_HOT_CONTENT_BLOCKLIST: tuple[str, ...] = (
    "去世", "身故", "遇难", "自杀", "死亡", "殉职", "溺亡",
    "车祸", "事故", "凶杀", "命案", "被杀", "谋杀",
    "性侵", "强奸", "猥亵",
    "抗议", "游行", "起义", "冲突", "占领", "封控", "抓捕",
    "歧视", "辱骂",
    # UI 残余 / API tracker
    "keyword_pinyin", "热榜聚合", "更多作品推荐",
)


def _is_hot_quality(candidate: dict) -> bool:
    """socially_hot 档的质量门: 有 title/snippet, 不命中黑名单."""
    title = str(candidate.get("title") or "").strip()
    snippet = str(candidate.get("snippet") or "").strip()
    if not title:
        return False
    full = title + " " + snippet
    for banned in _HOT_CONTENT_BLOCKLIST:
        if banned in full:
            return False
    return True


@dataclass(frozen=True)
class SourceClassification:
    """一次分类的完整结果."""
    kind: TopicSourceKind
    selected_candidate: dict | None  # LLM 应该用的那条 (可能是 None = 空档)
    reason: str  # 一句话说明为何选这档 (log/debug 用)


def classify_topic_source(
    *,
    trending_candidates: list[dict],
    user_portrait: str = "",
    agent: Any = None,
    exclude_titles: set[str] | frozenset[str] = frozenset(),
) -> SourceClassification:
    """V3 分类主入口. 决定用哪档 + 挑哪条候选内容.

    优先级:
      1. user_interest_match: 有候选命中用户兴趣词
      2. ai_persona_match:    有候选命中 AI 人设兴趣词
      3. socially_hot:        有质量过关的候选
      4. none:                都不满足, 走通用寒暄

    优先级 (而非按概率抽) 的理由: 分类是"给现有内容找最合适的表达"; 概率抽签让"内容
    明明能勾用户 A 却硬套 socially_hot 表达"的错配自动发生. 概率控制留给上游"要不要
    trending" (proactive_trending_probability), 下游选源用确定性优先级更 clean.

    ## exclude_titles (2026-09-14): 排除该 user × agent 最近已 featured 的候选

    根因: 分类器纯确定性 (相同输入总选第 0 条), 加上 DailyHot 上游 top-N 短时间
    内不动, 用户短时间内连测多次会得到"同一件事情反复推". featured_topics.py
    追踪 workspace 最近 N 条 featured title (6h TTL), 传进来的集合会**在打分前**
    从 candidates 里 pop 掉 —— 于是自动选下一条最热的.

    过滤在打分前做, 不是在打分后 penalty, 因为:
      - 一旦排除, 优先级判定该走 next-tier (user_interest 全排除 → 该走 ai_persona
        或 socially_hot) —— penalty 会让"命中用户兴趣但已 featured"的候选仍然
        排在 socially_hot 未 featured 之前, 语义错.
      - 排除只对 title 完全匹配, 不做子串/相似度. 保守: 换个说法的相邻热点仍算
        新话题 (e.g. "赵雷当爸爸" vs "赵雷官宣二胎" 视为不同, 都可推).
    """
    if not trending_candidates:
        return SourceClassification(
            kind="none", selected_candidate=None,
            reason="no trending candidates → 走 none",
        )

    # 过滤已 featured. 空集合 = 无过滤 (向后兼容 caller 未传参).
    if exclude_titles:
        pre_count = len(trending_candidates)
        trending_candidates = [
            c for c in trending_candidates
            if str(c.get("title") or "").strip() not in exclude_titles
        ]
        dropped = pre_count - len(trending_candidates)
        if not trending_candidates:
            return SourceClassification(
                kind="none", selected_candidate=None,
                reason=f"全部 {pre_count} 条候选都在 exclude_titles 里 → 走 none "
                       f"(user × agent 短时间内已看过, 等 featured_topics TTL 过期)",
            )
        filter_note = f" [排除 {dropped} 条已 featured]" if dropped else ""
    else:
        filter_note = ""

    user_terms = _extract_interests(user_portrait or "")

    # AI 兴趣词从 agent.background / occupation / lifeOverview 抠
    ai_text = ""
    if agent is not None:
        parts = [
            getattr(agent, "background", "") or "",
            getattr(agent, "lifeOverview", "") or "",
            getattr(agent, "occupation", "") or "",
        ]
        ai_text = "\n".join(str(p) for p in parts if p)
    ai_terms = _extract_interests(ai_text)

    # 每条候选打三档分
    scored: list[tuple[int, int, bool, dict]] = []  # (user_hits, ai_hits, hot_ok, cand)
    for cand in trending_candidates:
        u = _count_hits(cand, user_terms)
        a = _count_hits(cand, ai_terms)
        h = _is_hot_quality(cand)
        scored.append((u, a, h, cand))

    # 1. user_interest_match: 有条命中用户兴趣词 (u ≥ 1) → 选 u 分最高的
    user_wins = sorted(
        (s for s in scored if s[0] >= 1),
        key=lambda s: (-s[0], -s[1]),  # user 分最高, 平局比 AI 分
    )
    if user_wins:
        _, _, _, cand = user_wins[0]
        return SourceClassification(
            kind="user_interest_match", selected_candidate=cand,
            reason=f"命中用户兴趣: {user_terms}{filter_note}",
        )

    # 2. ai_persona_match: 命中 AI 兴趣
    ai_wins = sorted(
        (s for s in scored if s[1] >= 1),
        key=lambda s: (-s[1],),
    )
    if ai_wins:
        _, _, _, cand = ai_wins[0]
        return SourceClassification(
            kind="ai_persona_match", selected_candidate=cand,
            reason=f"命中 AI 人设兴趣: {ai_terms}{filter_note}",
        )

    # 3. socially_hot: 有质量过关的候选
    hot_ok = [s for s in scored if s[2]]
    if hot_ok:
        _, _, _, cand = hot_ok[0]  # 已经按 trending_candidates 顺序 (通常热度已排)
        return SourceClassification(
            kind="socially_hot", selected_candidate=cand,
            reason=f"有质量过关的社交谈资候选{filter_note}",
        )

    # 4. 都拒绝 → none
    return SourceClassification(
        kind="none", selected_candidate=None,
        reason=f"{len(trending_candidates)} 条候选全部被质量门拒绝{filter_note}",
    )
