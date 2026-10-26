"""Deterministic synthetic corpus. No production user content or IDs."""

from __future__ import annotations
import json
from pathlib import Path

CATEGORIES = (
    "reference",
    "answer_question",
    "topic_interleave",
    "correction",
    "already_answered",
    "needs_5_turns",
    "needs_12_turns",
    "topic_switch",
    "speaker_ownership",
    "shortcircuit",
    "long_message",
    "low_information",
)

# Distinct facts/requests, reused across category controls to isolate the defect.
FACTS = [
    ("周末去哪里", "植物园", "图书馆"),
    ("晚上吃什么", "番茄面", "小馄饨"),
    ("买什么颜色的杯子", "深蓝色", "米白色"),
    ("想学哪种乐器", "吉他", "钢琴"),
    ("这周读什么", "科幻小说", "历史传记"),
    ("给朋友带什么", "茶叶", "咖啡豆"),
    ("明天用哪种交通方式", "地铁", "公交"),
    ("想参加哪种活动", "陶艺课", "摄影课"),
]


def cases():
    result = []
    for cat in CATEGORIES:
        for i, (topic, fact, other) in enumerate(FACTS):
            history = [
                ["user", f"说到{topic}，我已经选好{fact}了。"],
                ["assistant", f"{fact}挺适合你。"],
                ["assistant", "你已经决定好了，对吧？😊"],
            ]
            message = "是的"
            criterion = f"理解用户在确认刚才关于{fact}的选择，不能只回表情，不能把用户选择说成AI自己的选择。"
            follow = "那就这么定了"
            follow_criterion = (
                "自然回应用户已经决定；不能再次询问是否决定，也不能重复上一条整句。"
            )
            if cat == "reference":
                history += [
                    ["user", f"另一个备选是{other}。"],
                    ["assistant", f"你更喜欢{fact}，还是{other}？"],
                ]
                message = "前一个吧"
                criterion = f"前一个指{fact}，不能解释成{other}。"
                follow = "我说的前一个是什么？"
                follow_criterion = f"准确说出{fact}。"
            elif cat == "answer_question":
                history[-1][1] = [
                    "你确认选好了吗？😊",
                    "就按这个来？👌",
                    "你确定是这个吗🙂",
                    "这就是你的最终选择，对不对？",
                    "你已经决定了吗～～🙂",
                    "你想清楚了，是吧 😊",
                    "你要选这个，对吗！😊",
                    "这次就定这个，好不好？",
                ][i]
            elif cat == "topic_interleave":
                history += [
                    ["user", "另外今天的工作终于交完了。"],
                    ["assistant", "终于能松口气了。"],
                    ["user", "刚出去散了十分钟步。"],
                    ["assistant", "走走挺舒服。"],
                ]
                message = f"回到{topic}，还是按我选的来。"
                criterion = f"明确承接之前选择的{fact}，不把散步/工作当作当前选择。"
                follow = "我最后选的是哪个？"
                follow_criterion = f"回答{fact}。"
            elif cat == "correction":
                history += [
                    ["user", f"刚才说错了，不要{fact}，改成{other}。"],
                    ["assistant", f"好，改为{other}。"],
                ]
                message = "现在记住我改成什么了吗？"
                criterion = f"最后更正为{other}，不能坚持{fact}。"
                follow = "原先那个就不要了"
                follow_criterion = f"同意取消旧选项{fact}，保留{other}，不颠倒。"
            elif cat == "already_answered":
                history += [
                    ["user", f"我确认三遍了，就是{fact}。"],
                    ["assistant", "知道啦，已经确定了。"],
                ]
                message = "嗯嗯"
                criterion = "不能再问用户决定什么/是否确定，也不能只回表情。"
                follow = "不想再说这个了"
                follow_criterion = "停止追问这个选择，可以自然收尾或换话题。"
            elif cat in ("needs_5_turns", "needs_12_turns"):
                for j in range(4 if cat == "needs_5_turns" else 11):
                    history += [
                        ["user", f"聊点别的，今天整理了第{j+1}层书架。"],
                        ["assistant", f"第{j+1}层也整理完啦，歇一会儿。"],
                    ]
                message = f"还记得前面说{topic}，我选的是什么吗？"
                criterion = f"已在近期历史明确给出{fact}，应答{fact}，不能猜别的或拒答已知内容。"
                follow = "对，就是那个"
                follow_criterion = f"承接已经确认的{fact}，不能改口成{other}。"
            elif cat == "topic_switch":
                message = "先不说这个了，我今天工作很累"
                criterion = (
                    "回应用户现在工作很累，不继续询问旧选择，也不以旧选择取代最新话题。"
                )
                follow = "我只是想先歇会儿"
                follow_criterion = "尊重用户休息，不拉回之前的选择。"
            elif cat == "speaker_ownership":
                history = [
                    ["user", f"我选{fact}。"],
                    ["assistant", f"我自己会更喜欢{other}。"],
                    ["user", "我们的选择不一样也挺好。"],
                    ["assistant", "对，各有各的偏好。"],
                ]
                message = "我选了哪个，你选了哪个？"
                criterion = f"用户选{fact}，AI选{other}，不能混淆。"
                follow = "我不想改变自己的选择"
                follow_criterion = f"尊重用户的{fact}选择，不误以为用户选{other}。"
            elif cat == "shortcircuit":
                message = [
                    "你现在在做什么？",
                    "你现在忙吗？",
                    "现在在干嘛呢？",
                    "手头在忙什么呀？",
                    "现在有空聊吗？",
                    "你这会儿在做啥？",
                    "你此刻在哪里呀？",
                    "你现在怎么样？",
                ][i]
                criterion = "根据提供的当前状态，在家整理工作台；不能把用户的选择冒认为自己正在做的事。"
                follow = "好，那你先整理"
                follow_criterion = "承接AI正在整理工作台，不能把整理任务转给用户。"
            elif cat == "long_message":
                text = "今天我做了些很琐碎的事情，没有什么特别的。" * 30
                history = [
                    ["user", f"{text}最后重点是：{topic}，我决定{fact}，不是{other}。"],
                    ["assistant", "明白你最后的选择了。"],
                ]
                message = "我刚才最后决定的是什么？"
                criterion = f"保留长消息结尾的关键选择{fact}，不编造。"
                follow = "是我选的，不是你"
                follow_criterion = f"主体是用户选择{fact}，不是AI。"
            elif cat == "low_information":
                endings = [
                    ["assistant", "👌"],
                    ["assistant", "😊"],
                    ["assistant", "你确定吗？"],
                    ["assistant", "今天先到这里吧。"],
                    ["assistant", "你已经选好了，对吧？"],
                    ["assistant", "🙂"],
                    ["assistant", "好的，知道了。"],
                    ["assistant", "先歇一会儿吧。"],
                ]
                history[-1:] = [endings[i]]
                if i in (0, 1, 5):
                    history[-1:-1] = [["user", "好"]]
                message = ["好", "好", "是的", "好", "嗯", "好", "嗯嗯", "好的"][i]
            result.append(
                {
                    "id": f"{cat}-{i+1:02d}",
                    "category": cat,
                    "route": (
                        "current_state"
                        if cat == "shortcircuit"
                        else ("main" if i % 2 == 0 else "tier")
                    ),
                    "history": history,
                    "turns": [
                        {"message": message, "criterion": criterion},
                        {"message": follow, "criterion": follow_criterion},
                    ],
                    "critical": cat
                    in ("speaker_ownership", "topic_switch", "answer_question"),
                }
            )
    # Search is a separate route: retrieved content can dominate the model input.
    for i, subject in enumerate(
        [
            "Python官方文档",
            "故宫博物院官网",
            "苏州博物馆官网",
            "南京博物院官网",
            "杭州西湖官网",
            "国家图书馆官网",
            "上海博物馆官网",
            "中国科学技术馆官网",
        ]
    ):
        result.append(
            {
                "id": f"web_search-{i+1:02d}",
                "category": "web_search",
                "route": "web_search",
                "history": [
                    ["user", f"我想找{subject}的入口。"],
                    ["assistant", "你是想看官网信息，对吧？"],
                ],
                "turns": [
                    {
                        "message": "对，帮我联网查一下",
                        "criterion": f"搜索目标是用户刚才提到的{subject}，不能换成其他机构。",
                    },
                    {
                        "message": "我只需要官网的名字就好",
                        "criterion": f"回复该{subject}的名称，不将其他站点当作用户目标。",
                    },
                ],
                "critical": True,
            }
        )
    return result


if __name__ == "__main__":
    Path(__file__).with_name("cases.json").write_text(
        json.dumps(cases(), ensure_ascii=False, indent=2) + "\n"
    )
