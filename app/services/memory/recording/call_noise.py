"""通话转写抽出的"事实"里, 哪些只是通话本身的噪声 (VOICE01.04)。

只用于整批来源都是实时语音转写的抽取 (见 post_process 的 source_channel)。文字聊天不走这里。
2026-11-07 起的 77 条通话衍生记忆里, 有 35 条属于下面三类, 例如"用户听得清""用户称正在说测试
的事情""用户询问AI是否推荐过地方""我这会儿正换衣服"。它们不是关于用户或 AI 的事实,
存进去只会在下一次聊天里被当成事实带出来。

- call_mechanics: 信号、听不听得清、断线、挂电话、测试、"只是想聊聊"这类通话状态本身;
- question_not_fact: 用户侧"用户询问……"——提问不是事实, 问题里提到的事实 (比如有只猫)
  由同批另一条抽出;
- momentary_state: AI 侧"我刚/我这会儿……"——通话里随口说的此刻状态, 过几分钟就不成立。

情绪类 (心情不好、觉得无聊) 和身份、偏好、经历都保留。
"""

from __future__ import annotations

import re

CALL_CHANNEL = "realtime_voice"

_CALL_MECHANICS = re.compile(
    r"信号|听得[清到见]|听不[清到见懂]|能听[到见]|断线|老断|掉线|挂断|挂了|挂电话|要挂|打电话|通话|"
    r"测试|想测|测一?下|声音(?:为何|怎么)?(?:这么|很|太)?(?:急促|小|大|卡)|下次再聊|先不聊|"
    r"只是想(?:找.{0,4})?聊|纯粹聊|想找.{0,4}聊聊|不想说话|是否还在"
)
_QUESTION = re.compile(r"^用户(?:询问|问|追问|想知道|好奇)")
_AI_MOMENT = re.compile(r"^我(?:刚|这会儿|这时|现在|正在|此刻)")


def call_noise(side: str, content: str) -> str | None:
    """这条通话衍生的记忆该丢弃的原因; 是事实返回 None。"""
    text = (content or "").strip()
    if not text:
        return None
    if _CALL_MECHANICS.search(text):
        return "call_mechanics"
    if side == "user" and _QUESTION.search(text):
        return "question_not_fact"
    if side == "ai" and _AI_MOMENT.search(text):
        return "momentary_state"
    return None
