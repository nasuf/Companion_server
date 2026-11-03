"""CHAT01.10.2 用例脱敏: 规则替换 → 模型改写 → 人工逐条复核 (decided_policies.eval_data_policy)。

只有人工复核通过的用例才能进仓库 (evals/reply_repetition/cases.json); 本脚本的输出一律标
pending_human_review, 和原始窗口一起放在受控证据目录。

- 规则: 手机号、证件号、邮箱、链接、门牌/小区/楼栋等城市以下的地址; 所有时间整体平移
  SHIFT_DAYS 天 (整周, 星期与相对间隔不变), 正文里的 "X月X日/号" 同步平移
- 模型: 人名 (用户/AI/第三方)、单位、学校、具体地点等规则抓不到的身份信息换成占位, 其余逐字
  保留 —— 复述的形态 (原样照抄还是换说法) 就是评测对象, 不能被改写掉
- 映射表、账号、会话、消息 ID 不出现在输出里
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path

SHIFT_DAYS = -35  # 整五周
MODEL = "qwen3.5-plus"
# 地名用字: 不从 "住/在/去/的/我…" 这类虚词、代词开始, 免得把前面的话一起吞掉
_PLACE = r"(?:(?![住在去到从回的了我你他她是和跟就也都还要想说家])[一-龥])"
# 号码与网址只用 ASCII 字符类: Python 的 \w、\S 会连中文一起吞掉
_RULES = [
    (re.compile(r"https?://[!-~]+"), "[链接]"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"), "[邮箱]"),
    (re.compile(r"(?<![\dA-Za-z])\d{17}[\dXx](?![\dA-Za-z])"), "[证件号]"),
    (re.compile(r"(?<!\d)(?:\d{4}[ -]?){3}\d{4}(?:[ -]?\d{1,3})?(?!\d)"), "[卡号]"),
    (re.compile(r"(?<!\d)(?:\+?86[ -]?)?1[3-9]\d(?:[ -]?\d{4}){2}(?!\d)"), "[手机号]"),
    (re.compile(r"(?<!\d)0\d{2,3}-?\d{7,8}(?!\d)"), "[电话]"),
    (re.compile(r"(?i)(微信号?|vx|wx|qq号?)[:：\s]*[A-Za-z0-9_-]{5,20}"), r"\1[账号]"),
    (re.compile(r"[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼][A-Z][A-Z0-9]{5,6}"), "[车牌]"),
    (re.compile(_PLACE + r"{1,12}(?:路|街|大道|巷|弄)\d+号(?:\d+(?:栋|幢|号楼|单元|室))*"), "[地址]"),
    (re.compile(_PLACE + r"{2,10}(?:小区|公寓|花园|新村|家园)(?:\d+(?:栋|幢|号楼|单元|室))*"), "[小区]"),
    (re.compile(r"\d+(?:栋|幢|号楼)\d*(?:单元)?\d*(?:室)?"), "[楼栋]"),
]
_TEXT_DATE = re.compile(r"(\d{1,2})月(\d{1,2})([日号])")
_ISO_DATE = re.compile(r"(?<!\d)(20\d{2})([-/.])(\d{1,2})\2(\d{1,2})(?!\d)")

REWRITE_PROMPT = """下面是一段聊天记录的若干行（JSON 数组，每项 {{"i": 序号, "text": 原文}}），用于内部评测。请只做脱敏：
- 人名（包括双方的名字、昵称、第三方的人名）换成「[人名A]」「[人名B]」这样的占位，同一个人全段用同一个占位
- 单位、公司、学校、医院、店名，以及比城市更细的地点（区、街道、商场、景点等）换成「[单位]」「[学校]」「[地点]」
- 城市名和省份保留；其余文字一字不改，标点、表情、语气词、错别字都保留，不要润色、不要合并或拆分行
- 已经是方括号占位的保持不动
输出 JSON：{{"rows": [{{"i": 序号, "text": 脱敏后的文字}}, ...]}}，行数和序号与输入一致。

{rows}"""


def shift_timestamp(value: str, days: int = SHIFT_DAYS) -> str:
    moment = datetime.fromisoformat(value)
    return (moment + timedelta(days=days)).isoformat()


def _moved(year: int, month: str, day: str, days: int) -> date | None:
    try:
        return date(year, int(month), int(day)) + timedelta(days=days)
    except ValueError:
        return None  # 不是合法日期 (如 2月30号、比分 3/12…): 原样保留


def shift_text_dates(text: str, year: int, days: int = SHIFT_DAYS) -> str:
    """正文里的日期跟时间戳一起平移: X月X日/号、2026-10-30 (或 / .)。"""
    def text_date(m: re.Match) -> str:
        moved = _moved(year, m[1], m[2], days)
        return f"{moved.month}月{moved.day}{m[3]}" if moved else m[0]

    def iso_date(m: re.Match) -> str:
        moved = _moved(int(m[1]), m[3], m[4], days)
        return f"{moved.year}{m[2]}{moved.month:02d}{m[2]}{moved.day:02d}" if moved else m[0]

    # "10/30" 这类不改: 和比分、分数分不开 (3/12), 改错比漏改更糟; 留给人工复核
    return _TEXT_DATE.sub(text_date, _ISO_DATE.sub(iso_date, text))


def apply_rules(text: str, year: int) -> str:
    for pattern, placeholder in _RULES:
        text = pattern.sub(placeholder, text)
    return shift_text_dates(text, year)


def texts_of(case: dict) -> list[str]:
    """一个用例里所有要脱敏的文字, 顺序固定: 历史各行、当前消息、评判用的用户原话、之前各轮气泡、生产回复。"""
    return ([r["content"] for r in case["history_rows"]] + [case["message"], case["judge_user_text"]]
            + [b for turn in case["previous_ai_turns"] for b in turn] + list(case["production_reply"]))


def with_texts(case: dict, texts: list[str]) -> dict:
    it = iter(texts)
    rows = [{**r, "content": next(it), "createdAt": shift_timestamp(r["createdAt"])} for r in case["history_rows"]]
    message = next(it)
    judge_user_text = next(it)
    previous = [[next(it) for _ in turn] for turn in case["previous_ai_turns"]]
    reply = [next(it) for _ in case["production_reply"]]
    return {**case, "history_rows": rows, "message": message, "judge_user_text": judge_user_text,
            "previous_ai_turns": previous,
            "production_reply": reply, "now": shift_timestamp(case["now"]),
            "received_at": shift_timestamp(case["received_at"]),
            "sanitization": {"rules": "v1", "model": MODEL, "shift_days": SHIFT_DAYS,
                             "review_status": "pending_human_review", "reviewer": None, "reviewed_at": None}}


def rule_pass(case: dict) -> list[str]:
    year = datetime.fromisoformat(case["now"]).year
    return [apply_rules(t, year) for t in texts_of(case)]


async def model_pass(texts: list[str], complete) -> list[str]:
    """模型改写; 行数或序号对不上就整例报错 (不能静默错位)。"""
    raw = await complete(REWRITE_PROMPT.format(
        rows=json.dumps([{"i": i, "text": t} for i, t in enumerate(texts)], ensure_ascii=False)))
    rows = json.loads(raw)["rows"]
    if [r["i"] for r in rows] != list(range(len(texts))):
        raise ValueError("rewrite changed row count or order")
    return [str(r["text"]) for r in rows]


async def main(args) -> dict:
    from openai import AsyncOpenAI

    from app.config import settings

    cases = json.loads(args.cases.read_text())
    client = AsyncOpenAI(api_key=settings.dashscope_api_key, base_url=settings.dashscope_base_url,
                         timeout=120, max_retries=2)
    gate = asyncio.Semaphore(6)

    async def complete(prompt: str) -> str:
        result = await client.chat.completions.create(
            model=MODEL, temperature=0, messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}, extra_body={"enable_thinking": False})
        return result.choices[0].message.content or "{}"

    out, failed = [], []

    async def one(case: dict) -> None:
        async with gate:
            try:
                out.append(with_texts(case, await model_pass(rule_pass(case), complete)))
            except Exception as exc:
                failed.append({"case": case["id"], "error": type(exc).__name__})

    try:
        await asyncio.gather(*[one(c) for c in cases])
    finally:
        await client.close()
    out.sort(key=lambda c: c["id"])
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    args.output.chmod(0o600)
    return {"sanitized": len(out), "failed": failed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(asyncio.run(main(parser.parse_args())), ensure_ascii=False))
