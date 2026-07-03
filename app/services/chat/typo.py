"""错别字生成器（Phase E1，借鉴 MaiBot typo_generator 的拟人化思路）。

真人打字会偶尔出同音错字，且约半数会随手补一条 "*正确字" 纠正——完美无误
的输出反而是"AI 感"的来源之一。与 MaiBot 的 pypinyin+jieba+字频加权实现
不同，这里用人工整理的**高频混淆对**（的得/在再/做作/他她…）：零新依赖，
且比随机同音替换更像中文用户真实打错的样子（真人错的就是这些字）。

只收录读者能"脑内自动纠正"的混淆对；意义翻转类（买/卖、带/戴）刻意排除，
避免错字改变语义造成误解。

默认关闭（settings.typo_enabled=False），由运营灰度开启。
"""

from __future__ import annotations

import random

# 高频同音/近音混淆对（单向替换：正确字 → 常见错字）。
# 双向收录（的↔得）让任一方向都可能出错。
_CONFUSION_PAIRS: dict[str, str] = {
    "的": "得",
    "得": "的",
    "在": "再",
    "再": "在",
    "做": "作",
    "作": "做",
    "他": "她",
    "她": "他",
    "那": "哪",
    "哪": "那",
    "像": "象",
    "以": "已",
    "已": "以",
    "应": "因",
    "座": "坐",
    "坐": "座",
}

# 自我纠正概率：真人不总是发现自己打错（MaiBot 用 50%）。
_CORRECTION_PROBABILITY = 0.5


def maybe_typo(
    text: str,
    *,
    rate: float,
    rng: random.Random | None = None,
) -> tuple[str, str | None]:
    """以 rate 概率给 text 注入一个同音错字。

    返回 (可能带错字的文本, 纠正字或 None)。纠正字非 None 时，调用方应在
    该条消息之后追加一条 "*正确字" 风格的纠正气泡（微信惯例）。

    - 每条消息最多 1 个错字（多了像键盘坏了，不像手滑）。
    - 文本里没有可混淆字时原样返回。
    """
    r = rng or random
    if rate <= 0 or not text or r.random() >= rate:
        return text, None
    positions = [i for i, ch in enumerate(text) if ch in _CONFUSION_PAIRS]
    if not positions:
        return text, None
    i = r.choice(positions)
    correct = text[i]
    wrong = _CONFUSION_PAIRS[correct]
    typo_text = f"{text[:i]}{wrong}{text[i + 1:]}"
    correction = correct if r.random() < _CORRECTION_PROBABILITY else None
    return typo_text, correction
