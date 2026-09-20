"""此行小预言（spec §4.7）。

预言文案来自预言池（不走大模型，spec §7 说明）。当前为内置常量池；后续可改为
prompting registry / module_settings 配置，`pick_prophecy` 的调用方无需变更。
"""

from __future__ import annotations

import random

PROPHECY_POOL: tuple[str, ...] = (
    "你会在一个不经意的角落，停得比预计更久。",
    "路上会有一阵刚好合适的风，记得抬头看一眼。",
    "今天遇见的小事，晚上回想时会突然觉得温柔。",
    "有人会在你路过时，轻轻对你笑一下。",
    "你会想把此刻的光线，悄悄收进口袋里。",
    "不必赶着到终点，半路上也值得多待一会儿。",
    "你会听见一句刚好想听的话，哪怕只是路过。",
    "此行有一块安静的地方，在等你坐下来。",
    "离开时你会比来时轻一点，像放下了什么。",
    "这一趟不必圆满，只要有一点点被记住就好。",
)


def pick_prophecy() -> str:
    return random.choice(PROPHECY_POOL)
