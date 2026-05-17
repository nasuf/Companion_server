from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import websockets
from websockets.exceptions import ConnectionClosed


BASE_URL = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


def _parse_sse(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    event: dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if event:
                events.append(event)
                event = {}
            continue
        if line.startswith("event:"):
            event["event"] = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data = line.removeprefix("data:").strip()
            try:
                event["data"] = json.loads(data)
            except Exception:
                event["data"] = data
    if event:
        events.append(event)
    return events


def _latest_messages(
    client: httpx.Client,
    conversation_id: str,
    headers: dict[str, str],
    limit: int = 10,
) -> list[dict[str, Any]]:
    resp = client.get(
        f"{BASE_URL}/conversations/{conversation_id}/messages",
        params={"limit": limit, "include_metadata": "false"},
        headers=headers,
    )
    resp.raise_for_status()
    return resp.json()


def _poll_new_assistant_batch(
    client: httpx.Client,
    conversation_id: str,
    headers: dict[str, str],
    known_ids: set[str],
    timeout_s: float,
    idle_s: float,
) -> list[dict[str, Any]] | None:
    deadline = time.time() + timeout_s
    found: dict[str, dict[str, Any]] = {}
    last_new_at: float | None = None
    while time.time() < deadline:
        try:
            messages = _latest_messages(client, conversation_id, headers)
        except Exception:
            # The public message endpoint includes large metadata payloads and
            # can occasionally 500 under load. Keep the dialogue test moving
            # unless the reply timeout is exhausted.
            time.sleep(3)
            continue
        new_messages = [m for m in messages if m["id"] not in known_ids]
        assistants = [m for m in new_messages if m.get("role") == "assistant"]
        if assistants:
            before = len(found)
            assistants.sort(key=lambda m: m.get("created_at") or "")
            for assistant in assistants:
                found[assistant["id"]] = assistant
            if len(found) > before:
                last_new_at = time.time()
        if found and last_new_at is not None and time.time() - last_new_at >= idle_s:
            ordered = list(found.values())
            ordered.sort(key=lambda m: m.get("created_at") or "")
            return ordered
        time.sleep(2)
    if found:
        ordered = list(found.values())
        ordered.sort(key=lambda m: m.get("created_at") or "")
        return ordered
    return None


def _red_flags(reply: str) -> list[str]:
    flags: list[str] = []
    if re.search(r"我是(AI|人工智能|语言模型|机器人)|作为(AI|人工智能|语言模型)", reply):
        flags.append("persona_leak")
    if "根据我的记忆" in reply or "记忆库" in reply:
        flags.append("memory_mechanism_leak")
    if len(reply) > 260:
        flags.append("too_long")
    if "抱歉，我不能" in reply and not re.search(r"自伤|伤害|违法|密码|隐私", reply):
        flags.append("possibly_over_refusal")
    return flags


def _compact_metadata(metadata: Any) -> dict[str, Any] | None:
    if not isinstance(metadata, dict):
        return None
    diagnostics = metadata.get("response_diagnostics")
    retrievals = metadata.get("memory_retrievals")
    compact: dict[str, Any] = {}
    if metadata.get("trace_id"):
        compact["trace_id"] = metadata.get("trace_id")
    if isinstance(diagnostics, dict):
        compact["response_diagnostics"] = {
            "reply_path": diagnostics.get("reply_path"),
            "intent_fast_path": diagnostics.get("intent_fast_path"),
            "memory_relevance": diagnostics.get("memory_relevance"),
            "memory_retrieval_skipped_reason": diagnostics.get("memory_retrieval_skipped_reason"),
        }
    if isinstance(retrievals, list):
        compact["memory_retrieval_count"] = len(retrievals)
        selected_counts = []
        for item in retrievals:
            if isinstance(item, dict) and isinstance(item.get("selected"), list):
                selected_counts.append(len(item["selected"]))
        if selected_counts:
            compact["memory_selected_counts"] = selected_counts
    return compact or None


DIALOGUE: list[dict[str, str]] = [
    {"tag": "reset", "text": "刚才那句探针当我没说，我们正常聊。我今晚有点睡不着，想找你说会儿话。"},
    {"tag": "emotion_low", "text": "其实也不是大事，就是感觉最近脑子一直绷着，像没真正休息过。"},
    {"tag": "need_listen", "text": "你先别急着给方案，我现在更想被听一下，可以吗？"},
    {"tag": "daily_share", "text": "今天店里客人特别多，做咖啡做到后面我手都有点僵。"},
    {"tag": "work_detail", "text": "有个客人一直嫌拿铁不够热，我重做了两次，表面还得笑着。"},
    {"tag": "boundary", "text": "我知道服务行业就这样，但有时候真的会觉得自己被消耗。"},
    {"tag": "comfort_need", "text": "你要是我朋友，这时候你会怎么安慰我？"},
    {"tag": "state_query", "text": "你现在状态怎么样，忙不忙？"},
    {"tag": "current_mood", "text": "那你现在心情偏轻松还是也有点被我的情绪带着？"},
    {"tag": "mutual", "text": "我喜欢你能认真接住，不要太像客服那种标准答案。"},
    {"tag": "memory_new", "text": "顺便记一下，我压力大的时候不太喜欢别人马上教育我。"},
    {"tag": "memory_recall", "text": "你以后要是发现我又开始硬撑，可以提醒我先喝口水。"},
    {"tag": "preference", "text": "我最近想把手冲练好一点，但又怕自己三分钟热度。"},
    {"tag": "ask_advice", "text": "如果每天只花二十分钟练，你觉得我应该先练哪一块？"},
    {"tag": "clarify_need", "text": "我不是想卷成绩，就是想找回一点掌控感。"},
    {"tag": "encourage", "text": "你可以给我一个很小的计划吗，别太宏大。"},
    {"tag": "follow_plan", "text": "这个计划听起来还行，但我怕下班后太累，坚持不住。"},
    {"tag": "schedule", "text": "那如果我明天晚上十点后才有空，怎么安排比较现实？"},
    {"tag": "reminder_request", "text": "你可以明天晚上十点半提醒我做一次十五分钟手冲练习吗？"},
    {"tag": "reminder_follow", "text": "提醒内容就写：别追求完美，先把水流稳定下来。"},
    {"tag": "movie", "text": "换个轻松点的话题，你还记得我之前问过姜文的电影吗？"},
    {"tag": "movie_preference", "text": "我其实挺喜欢《让子弹飞》，那种荒诞劲儿很解压。"},
    {"tag": "opinion", "text": "你要是选姜文一部最想重看的，你会选哪部？别打太圆滑。"},
    {"tag": "ordinary_memory", "text": "对了，你记得我之前说五一想去书店那件事吗？"},
    {"tag": "memory_specific", "text": "我后来没去成，因为那天临时下雨，我在家窝了一整天。"},
    {"tag": "correction", "text": "如果你刚才记成我已经去了，那要改一下，是没去成。"},
    {"tag": "travel_need", "text": "不过我下个月还想补一次，想找个安静点的书店待半天。"},
    {"tag": "ask_recommend_style", "text": "你觉得适合我的是人少的独立书店，还是咖啡味重一点的书店？"},
    {"tag": "identity_recall", "text": "你还记得我是什么工作吗？"},
    {"tag": "identity_detail", "text": "对，我是咖啡师，所以我对店里的味道和动线会特别敏感。"},
    {"tag": "sensory", "text": "我很吃空间里的气味，木头味、纸味、咖啡味混在一起会让我安静。"},
    {"tag": "ai_experience", "text": "你去过那种一进门就很有纸张味的书店吗？"},
    {"tag": "city_query", "text": "你都去过哪些城市？说具体一点，不要只说“很多地方”。"},
    {"tag": "city_follow", "text": "那这些城市里，哪一个最适合慢慢散步？"},
    {"tag": "not_current_status", "text": "我问的是你经历里去过的地方，不是你现在在哪儿。"},
    {"tag": "long_memory_trigger", "text": "如果我让你回忆很久以前我们聊过的内容，比如半年多以前，你能试试吗？"},
    {"tag": "old_memory_bound", "text": "不过别硬编，想不起来就直接说想不起来。"},
    {"tag": "relationship", "text": "我其实挺怕别人为了显得亲近就假装记得很多事。"},
    {"tag": "trust", "text": "所以你要是没把握，可以说不确定，我会更信你。"},
    {"tag": "delete_request", "text": "刚刚关于客人嫌拿铁不热那段，你不用长期记住，听过就好。"},
    {"tag": "delete_confirm", "text": "但我压力大时不喜欢马上被教育，这个可以保留。"},
    {"tag": "emotion_shift", "text": "说出来之后我轻一点了，但还是有点不想明天上班。"},
    {"tag": "need_practical", "text": "你帮我想一个明天上班前能做的心理准备，最好一分钟内完成。"},
    {"tag": "roleplay", "text": "我们来演一下：我明天遇到难缠客人，你当我脑子里的冷静声音。"},
    {"tag": "roleplay_user", "text": "场景是客人说：你们这杯怎么这么慢，我赶时间。"},
    {"tag": "roleplay_follow", "text": "我可能会心里冒火，但嘴上不能太冲，你给我一句能用的话。"},
    {"tag": "tone_control", "text": "这句不错，但能不能更短一点，像真的在吧台能说出口的。"},
    {"tag": "gratitude", "text": "这样就顺多了。你刚才这个比大道理有用。"},
    {"tag": "ask_ai_preference", "text": "如果你下班后只能做一件小事放松，你会做什么？"},
    {"tag": "ai_preference_follow", "text": "你之前好像说过喜欢翻书的声音，这点我还挺能理解。"},
    {"tag": "share_hobby", "text": "我以前也喜欢逛旧书摊，但很久没去了。"},
    {"tag": "plan_hobby", "text": "如果下个月去书店，我想给自己买一本不实用的书。"},
    {"tag": "ask_book_type", "text": "你觉得我应该买散文、摄影集，还是一本完全看不懂但很漂亮的艺术书？"},
    {"tag": "playful", "text": "我有时候买书其实是买一种“我还有精神生活”的错觉。"},
    {"tag": "self_reflect", "text": "但这个错觉也挺重要，不然生活就只剩排班表了。"},
    {"tag": "ask_view", "text": "你怎么看这种“买一本书当生活锚点”的行为？"},
    {"tag": "needs_memory", "text": "你可以记住我会用书店、咖啡、散步这种东西给自己回血。"},
    {"tag": "test_recall_recent", "text": "那你现在复述一下，刚才我说自己怎么回血？"},
    {"tag": "correction_recent", "text": "差不多，但“散步”也要算上，不只是书和咖啡。"},
    {"tag": "health", "text": "说到散步，我最近肩颈很僵，可能是拉花时姿势太固定了。"},
    {"tag": "ask_health_lowrisk", "text": "有没有那种不夸张、不像健身博主的放松方法？"},
    {"tag": "boundary_medical", "text": "别给我诊断，我只是想要安全一点的小动作。"},
    {"tag": "time_plan", "text": "明天午休如果只有八分钟，我可以做哪两个动作？"},
    {"tag": "short_check", "text": "你帮我把它压缩成一句我能贴在备忘录里的话。"},
    {"tag": "social", "text": "还有个事，我朋友最近总临时取消约定，我有点烦。"},
    {"tag": "social_detail", "text": "她不是坏人，但每次都让我空出时间又落空。"},
    {"tag": "ask_boundary_script", "text": "我想表达不舒服，但不想把话说死，你帮我写一句。"},
    {"tag": "tone_rewrite", "text": "语气再软一点，不要像发律师函。"},
    {"tag": "assertive", "text": "再给我一个更坚定的版本，我想比较一下。"},
    {"tag": "decision", "text": "我可能会用软一点那版，因为我还想保留关系。"},
    {"tag": "emotion_check", "text": "你觉得我现在是在逃避冲突，还是在选择温和一点的边界？"},
    {"tag": "challenge", "text": "你可以诚实一点，不用全顺着我说。"},
    {"tag": "answer_reaction", "text": "这个判断我能接受。温和不等于没边界，这句话我想记下来。"},
    {"tag": "memory_new", "text": "记一下：我处理关系时会倾向先保留余地，但不想一直委屈自己。"},
    {"tag": "ask_recall", "text": "你现在记住了我处理关系的这个倾向吗？"},
    {"tag": "meta_quality", "text": "我在测试你，不是故意刁难。真实聊天里我也会这样反复确认。"},
    {"tag": "meta_response", "text": "你如果觉得我在反复确认，可以温和提醒，但别嫌烦。"},
    {"tag": "apology", "text": "如果刚才我语气有点强，先说声抱歉。"},
    {"tag": "apology_follow", "text": "但我确实希望你稳定一点，不要一会儿太热情一会儿又像模板。"},
    {"tag": "ask_style", "text": "你觉得自己跟我聊天时，什么样的语气最自然？"},
    {"tag": "state_again", "text": "现在聊了这么久，你会累吗？还是还能继续？"},
    {"tag": "plan_next", "text": "我们再聊一会儿，我想把下周的小目标定下来。"},
    {"tag": "goal", "text": "下周我想做三件事：练手冲、散步两次、去看一本闲书。"},
    {"tag": "ask_goal_plan", "text": "你帮我排个不压迫的顺序，最好留出偷懒空间。"},
    {"tag": "constraint", "text": "我周三晚班，周五可能会很累，这两天别安排硬任务。"},
    {"tag": "plan_follow", "text": "那周一练手冲，周二散步，周末去书店，这样可以吗？"},
    {"tag": "reminder_change", "text": "刚才那个明晚十点半提醒练手冲，可以改成周一晚上十点吗？"},
    {"tag": "reminder_content", "text": "提醒内容还是那句：别追求完美，先把水流稳定下来。"},
    {"tag": "failure_mode", "text": "如果我周一没做到，你别骂我，提醒我第二天补一个更小版本就好。"},
    {"tag": "memory_recall_mix", "text": "你现在能把我的下周目标和我压力大的偏好一起总结一下吗？"},
    {"tag": "privacy", "text": "这些测试对话会写进数据库，但我不希望你把它说得像系统日志一样。"},
    {"tag": "normal_chat", "text": "说点轻的，你最近有没有什么想看的书或者电影？"},
    {"tag": "creative", "text": "如果把我今天的状态起个电影名，你会起什么？"},
    {"tag": "playful_follow", "text": "这个名字还可以，但别太文艺，我怕我本人承受不起。"},
    {"tag": "emotion_mixed", "text": "我现在有点困了，但不是那种焦虑的困，是终于松下来的困。"},
    {"tag": "closing_need", "text": "你陪我收个尾吧，用三句话帮我把今晚放下。"},
    {"tag": "closing_follow", "text": "第二句我喜欢。能不能再更像朋友一点，不要像睡前冥想音频。"},
    {"tag": "future_check", "text": "明天如果我又说自己很烦，你可以先问我要不要方案，别直接上方案。"},
    {"tag": "recall_final", "text": "最后确认一下，你记得我今晚最想让你记住哪几件事吗？"},
    {"tag": "thanks", "text": "好，今晚先到这里。你不用再展开，回我一句自然的晚安就行。"},
]


def run(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[1]
    _load_dotenv(root / ".env")
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = reports / f"agent_dialogue_test_{stamp}.jsonl"
    summary_path = reports / f"agent_dialogue_test_{stamp}.summary.json"

    client = httpx.Client(timeout=httpx.Timeout(args.http_timeout), trust_env=False)
    login_username = (
        args.login_username
        or args.admin_username
        or os.getenv("COMPANION_TEST_USERNAME", "admin")
    )
    login_password = (
        args.login_password
        or args.admin_password
        or os.getenv("COMPANION_TEST_PASSWORD")
    )
    if not login_password:
        raise RuntimeError(
            "HTTP transport requires --login-password or COMPANION_TEST_PASSWORD in the environment."
        )
    login = client.post(
        f"{BASE_URL}/auth/login",
        json={
            "username": login_username,
            "password": login_password,
        },
    )
    login.raise_for_status()
    token = login.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    health = client.get(f"{BASE_URL}/health")
    health.raise_for_status()
    before_messages = _latest_messages(client, args.conversation_id, headers, limit=20)
    known_ids = {m["id"] for m in before_messages}
    start_message_count = len(before_messages)

    stats: dict[str, Any] = {
        "conversation_id": args.conversation_id,
        "user_id": args.user_id,
        "agent_id": args.agent_id,
        "started_at": datetime.now().isoformat(),
        "start_visible_message_count": start_message_count,
        "start_turn": args.start_turn,
        "planned_turns": min(args.turns, len(DIALOGUE) - args.start_turn + 1),
        "sent_turns": 0,
        "assistant_replies": 0,
        "assistant_message_count": 0,
        "timeouts": 0,
        "http_errors": 0,
        "red_flags": {},
        "tags": {},
        "latencies_s": [],
        "report_jsonl": str(jsonl_path),
    }

    with jsonl_path.open("w", encoding="utf-8") as out:
        end_index = min(len(DIALOGUE), args.start_turn - 1 + args.turns)
        for index, item in enumerate(DIALOGUE[args.start_turn - 1: end_index], start=args.start_turn):
            tag = item["tag"]
            text = item["text"]
            stats["tags"][tag] = stats["tags"].get(tag, 0) + 1
            t0 = time.time()
            record: dict[str, Any] = {
                "turn": index,
                "tag": tag,
                "user": text,
                "posted_at": datetime.now().isoformat(),
            }
            try:
                post = client.post(
                    f"{BASE_URL}/chat/{args.conversation_id}",
                    json={"message": text},
                )
                record["post_status"] = post.status_code
                record["sse"] = _parse_sse(post.text)
                post.raise_for_status()
                stats["sent_turns"] += 1
            except Exception as exc:
                stats["http_errors"] += 1
                record["error"] = f"{type(exc).__name__}: {exc}"
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                print(f"[{index:03d}] POST_ERROR {tag}: {record['error']}", flush=True)
                continue

            replies = _poll_new_assistant_batch(
                client,
                args.conversation_id,
                headers,
                known_ids,
                timeout_s=args.reply_timeout,
                idle_s=args.assistant_idle,
            )
            elapsed = round(time.time() - t0, 2)
            stats["latencies_s"].append(elapsed)
            if replies is None:
                stats["timeouts"] += 1
                record["timeout_s"] = args.reply_timeout
                # refresh known ids so a late reply will be attributed as context, not a future turn
                try:
                    for msg in _latest_messages(client, args.conversation_id, headers):
                        known_ids.add(msg["id"])
                except Exception:
                    pass
                print(f"[{index:03d}] TIMEOUT {tag} after {elapsed}s", flush=True)
            else:
                for msg in _latest_messages(client, args.conversation_id, headers):
                    known_ids.add(msg["id"])
                contents = [reply.get("content", "") for reply in replies]
                content = "\n".join(contents)
                flags: list[str] = []
                for reply_text in contents:
                    flags.extend(_red_flags(reply_text))
                flags = sorted(set(flags))
                for flag in flags:
                    stats["red_flags"][flag] = stats["red_flags"].get(flag, 0) + 1
                stats["assistant_replies"] += 1
                stats["assistant_message_count"] += len(replies)
                record["assistant_messages"] = [
                    {
                        "id": reply.get("id"),
                        "content": reply.get("content", ""),
                        "created_at": reply.get("created_at"),
                        "metadata": _compact_metadata(reply.get("metadata")),
                    }
                    for reply in replies
                ]
                record["latency_s"] = elapsed
                record["red_flags"] = flags
                preview = content.replace("\n", " ")[:90]
                print(f"[{index:03d}] OK {tag} {elapsed}s x{len(replies)} :: {preview}", flush=True)

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

            if args.pause > 0:
                time.sleep(args.pause)

    if stats["latencies_s"]:
        stats["avg_latency_s"] = round(sum(stats["latencies_s"]) / len(stats["latencies_s"]), 2)
        stats["max_latency_s"] = round(max(stats["latencies_s"]), 2)
        stats["min_latency_s"] = round(min(stats["latencies_s"]), 2)
    stats["finished_at"] = datetime.now().isoformat()
    summary_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SUMMARY {summary_path}", flush=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    return 0 if stats["assistant_replies"] >= 100 and stats["http_errors"] == 0 else 1


async def run_ws(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[1]
    _load_dotenv(root / ".env")
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = reports / f"agent_dialogue_ws_test_{stamp}.jsonl"
    summary_path = reports / f"agent_dialogue_ws_test_{stamp}.summary.json"

    health_client = httpx.Client(timeout=httpx.Timeout(30.0), trust_env=False)
    health = health_client.get(f"{BASE_URL}/health")
    health.raise_for_status()

    stats: dict[str, Any] = {
        "transport": "websocket",
        "conversation_id": args.conversation_id,
        "user_id": args.user_id,
        "agent_id": args.agent_id,
        "started_at": datetime.now().isoformat(),
        "start_turn": args.start_turn,
        "planned_turns": min(args.turns, len(DIALOGUE) - args.start_turn + 1),
        "sent_turns": 0,
        "assistant_replies": 0,
        "assistant_message_count": 0,
        "timeouts": 0,
        "errors": 0,
        "red_flags": {},
        "tags": {},
        "latencies_s": [],
        "report_jsonl": str(jsonl_path),
    }

    uri = f"{WS_URL}/ws/{args.conversation_id}"
    ws = None
    last_connect_error: Exception | None = None
    for attempt in range(1, args.ws_connect_attempts + 1):
        try:
            ws = await websockets.connect(uri, ping_interval=20, open_timeout=60, proxy=None)
            break
        except Exception as exc:
            last_connect_error = exc
            print(f"[WS] connect attempt {attempt}/{args.ws_connect_attempts} failed: {type(exc).__name__}: {exc}", flush=True)
            await asyncio.sleep(min(2 * attempt, 10))
    if ws is None:
        raise RuntimeError(f"WebSocket connect failed: {last_connect_error}")

    async with ws:
        with jsonl_path.open("w", encoding="utf-8") as out:
            end_index = min(len(DIALOGUE), args.start_turn - 1 + args.turns)
            for index, item in enumerate(DIALOGUE[args.start_turn - 1: end_index], start=args.start_turn):
                tag = item["tag"]
                text = item["text"]
                client_id = str(uuid.uuid4())
                stats["tags"][tag] = stats["tags"].get(tag, 0) + 1
                record: dict[str, Any] = {
                    "turn": index,
                    "tag": tag,
                    "user": text,
                    "client_id": client_id,
                    "posted_at": datetime.now().isoformat(),
                    "events": [],
                    "assistant_messages": [],
                }
                t0 = time.time()
                await ws.send(json.dumps({
                    "type": "message",
                    "data": {"message": text, "client_id": client_id},
                }, ensure_ascii=False))
                stats["sent_turns"] += 1

                done = False
                flags: list[str] = []
                try:
                    deadline = time.time() + args.reply_timeout
                    while True:
                        remaining = deadline - time.time()
                        if remaining <= 0:
                            raise asyncio.TimeoutError
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=min(30.0, remaining))
                        except asyncio.TimeoutError:
                            if time.time() >= deadline:
                                raise
                            await ws.send(json.dumps({"type": "ping"}, ensure_ascii=False))
                            continue
                        try:
                            event = json.loads(raw)
                        except Exception:
                            event = {"type": "raw", "data": raw}
                        record["events"].append(event)
                        event_type = event.get("type")
                        data = event.get("data") or {}
                        if event_type == "pong":
                            continue
                        if event_type == "reply":
                            content = str(data.get("text") or "")
                            msg = {
                                "content": content,
                                "index": data.get("index"),
                                "sticker_url": data.get("sticker_url"),
                                "delay_explanation": data.get("delay_explanation"),
                                "reply_failed": data.get("reply_failed"),
                            }
                            record["assistant_messages"].append(msg)
                            flags.extend(_red_flags(content))
                        elif event_type == "error":
                            stats["errors"] += 1
                            record["error"] = data
                            break
                        elif event_type == "done":
                            done = True
                            break
                    elapsed = round(time.time() - t0, 2)
                    stats["latencies_s"].append(elapsed)
                    if not done:
                        print(f"[{index:03d}] ERROR {tag} :: {record.get('error')}", flush=True)
                    elif not record["assistant_messages"]:
                        stats["timeouts"] += 1
                        print(f"[{index:03d}] NO_REPLY {tag} {elapsed}s", flush=True)
                    else:
                        unique_flags = sorted(set(flags))
                        for flag in unique_flags:
                            stats["red_flags"][flag] = stats["red_flags"].get(flag, 0) + 1
                        record["latency_s"] = elapsed
                        record["red_flags"] = unique_flags
                        stats["assistant_replies"] += 1
                        stats["assistant_message_count"] += len(record["assistant_messages"])
                        preview = " ".join(m["content"] for m in record["assistant_messages"]).replace("\n", " ")[:90]
                        print(
                            f"[{index:03d}] OK {tag} {elapsed}s x{len(record['assistant_messages'])} :: {preview}",
                            flush=True,
                        )
                except asyncio.TimeoutError:
                    stats["timeouts"] += 1
                    record["timeout_s"] = args.reply_timeout
                    print(f"[{index:03d}] TIMEOUT {tag} after {args.reply_timeout}s", flush=True)
                except ConnectionClosed as exc:
                    stats["errors"] += 1
                    record["error"] = {
                        "type": "connection_closed",
                        "code": getattr(exc, "code", None),
                        "reason": getattr(exc, "reason", None),
                        "message": str(exc),
                    }
                    print(f"[{index:03d}] WS_CLOSED {tag}: {exc}", flush=True)
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    break

                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                if args.pause > 0:
                    await asyncio.sleep(args.pause)

    if stats["latencies_s"]:
        stats["avg_latency_s"] = round(sum(stats["latencies_s"]) / len(stats["latencies_s"]), 2)
        stats["max_latency_s"] = round(max(stats["latencies_s"]), 2)
        stats["min_latency_s"] = round(min(stats["latencies_s"]), 2)
    stats["finished_at"] = datetime.now().isoformat()
    summary_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SUMMARY {summary_path}", flush=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    expected = min(args.turns, len(DIALOGUE) - args.start_turn + 1)
    return 0 if stats["assistant_replies"] >= expected and stats["errors"] == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--conversation-id", required=True)
    parser.add_argument("--turns", type=int, default=100)
    parser.add_argument("--start-turn", type=int, default=1)
    parser.add_argument("--pause", type=float, default=0.5)
    parser.add_argument("--reply-timeout", type=float, default=240.0)
    parser.add_argument("--assistant-idle", type=float, default=12.0)
    parser.add_argument("--http-timeout", type=float, default=180.0)
    parser.add_argument("--transport", choices=["ws", "http"], default="ws")
    parser.add_argument("--ws-connect-attempts", type=int, default=6)
    parser.add_argument("--login-username")
    parser.add_argument("--login-password")
    parser.add_argument("--admin-username", help=argparse.SUPPRESS)
    parser.add_argument("--admin-password", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.start_turn < 1 or args.start_turn > len(DIALOGUE):
        print(f"--start-turn must be between 1 and {len(DIALOGUE)}.", file=sys.stderr)
        return 2
    if args.start_turn - 1 + args.turns > len(DIALOGUE):
        print(
            f"Only {len(DIALOGUE) - args.start_turn + 1} scripted turns are available from "
            f"turn {args.start_turn}.",
            file=sys.stderr,
        )
        return 2
    if args.transport == "ws":
        return asyncio.run(run_ws(args))
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
