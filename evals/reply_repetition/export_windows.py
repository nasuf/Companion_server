"""CHAT01.10.2 生产只读导出: 抽样的主回复轮次 + 之前 24 小时的原始消息。

在生产容器里经 stdin 运行, 结果只写到受控证据目录 (不进仓库, 90 天删除):

    ssh ... 'sudo docker exec -i -e PYTHONPATH=/app companion-server python - --seed 20261103' \\
        < evals/reply_repetition/export_windows.py > $EVIDENCE/windows-raw.json

抽样口径 (10.1 的打分表 reply_repetition_scores, 当前打分版本):
- 正例: 判为原样照抄 / 换说法复述的轮次; 附和与交叠优先, 其余随机补足
- 负例: 判为合理重复的优先, 再从 "有候选但判为不重复" 的高相似轮次 (难负例) 里随机补足
- 每个用户最多占 PER_USER_CAP 条, 免得一个高频用户主导整个评测集 (优先的那些也受这个上限)
- 窗口取该轮之前 24 小时内最近 WINDOW_ROWS 行; 用例只用最后 7 轮

只读; 不写库; 元数据只导出复现所需的字段。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys

POSITIVE_TARGET = 130
NEGATIVE_TARGET = 130
PER_USER_CAP = 20
WINDOW_ROWS = 60
META_KEYS = ("proactive", "trigger_type", "achievement_turn_id", "kind", "component_card",
             "offering_received", "trace_id")
DIAG_KEYS = ("reply_path", "planned_delay_ms", "delay_reason", "turn_received_at", "generation_started_at")


def slim_metadata(meta) -> dict:
    if isinstance(meta, str):
        meta = json.loads(meta)
    if not isinstance(meta, dict):
        return {}
    out = {k: meta[k] for k in META_KEYS if k in meta}
    diag = meta.get("response_diagnostics")
    if isinstance(diag, dict):
        out["response_diagnostics"] = {k: diag[k] for k in DIAG_KEYS if k in diag}
        topic = diag.get("topic_continuation")
        if isinstance(topic, dict):
            out["response_diagnostics"]["topic_continuation"] = {"status": topic.get("status")}
    return out


def pick(rows: list[dict], target: int, rng: random.Random, taken_per_user: dict[str, int],
         must: list[dict] | None = None) -> list[dict]:
    """先放 must, 再随机补足; 两者都受每用户上限约束。"""
    chosen = []
    pool = list(must or []) + [r for r in rng.sample(rows, len(rows)) if r not in (must or [])]
    for row in pool:
        if len(chosen) >= target:
            break
        if taken_per_user.get(row["user_id"], 0) >= PER_USER_CAP:
            continue
        taken_per_user[row["user_id"]] = taken_per_user.get(row["user_id"], 0) + 1
        chosen.append(row)
    return chosen


async def main(seed: int) -> dict:
    from app.db import db
    from app.services.effect.repetition import SCORER_VERSION

    await db.connect()
    try:
        scores = await db.query_raw(
            """
            SELECT turn_message_id, local_date::text AS local_date, user_id, conversation_id, occurred_at,
                   situation, label, repeated, max_similarity
            FROM reply_repetition_scores WHERE scorer_version = $1 ORDER BY turn_message_id
            """,
            SCORER_VERSION,
        )
        rng = random.Random(seed)
        per_user: dict[str, int] = {}
        positives = [s for s in scores if s["repeated"]]
        must_pos = [s for s in positives if s["situation"] in ("ack", "overlap")]
        chosen_pos = pick(positives, POSITIVE_TARGET, rng, per_user, must_pos)
        legit = [s for s in scores if s["label"] == "legit"]
        hard = [s for s in scores if s["label"] == "not_repeat" and (s["max_similarity"] or 0) >= 0.6]
        chosen_neg = pick(hard + legit, NEGATIVE_TARGET, rng, per_user, legit)
        windows = []
        for score in chosen_pos + chosen_neg:
            rows = await db.query_raw(
                """
                SELECT m.id, m.role, m.content, m.created_at, m.metadata
                FROM messages m
                WHERE m.conversation_id = $1 AND m.role IN ('user', 'assistant')
                  AND m.created_at >= $2::timestamp - interval '24 hours'
                  AND m.created_at <= $2::timestamp + interval '2 minutes'
                ORDER BY m.created_at DESC, m.id DESC
                LIMIT $3
                """,
                score["conversation_id"], score["occurred_at"], WINDOW_ROWS,
            )
            windows.append({
                "score": {k: score[k] for k in ("turn_message_id", "local_date", "user_id", "conversation_id",
                                                 "situation", "label", "max_similarity")},
                "rows": [{"id": r["id"], "role": r["role"], "content": r["content"],
                          "created_at": r["created_at"], "metadata": slim_metadata(r["metadata"])}
                         for r in reversed(rows)],
            })
        return {"seed": seed, "scorer_version": SCORER_VERSION, "positives": len(chosen_pos),
                "negatives": len(chosen_neg), "available": {"positives": len(positives), "legit": len(legit),
                                                            "hard_negatives": len(hard)},
                "windows": windows}
    finally:
        await db.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args(sys.argv[1:])
    print(json.dumps(asyncio.run(main(args.seed)), ensure_ascii=False, default=str))
