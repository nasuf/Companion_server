"""检查提示词有没有在后台被人工改过 —— 改代码默认值之前必跑.

## 为什么需要这个

`sync_prompt_definitions` 的逻辑是: 代码默认值一变, 就用它覆盖线上内容。

    default_changed = existing.defaultContent != definition.default_text
    if default_changed:
        content = definition.default_text   # 不管它是不是人工改过的

所以改代码默认值等于**无条件覆盖**后台的编辑。项目历史上 `chat.response_instruction`
就是这么被覆盖过一次的, 事后只能从版本表里把内容捞回来重新提交。

规矩因此是: 没被人工改过的, 直接改代码默认值; 改过的, 走 update_prompt_text 提交
新版本 (那条路径会同时更新 content 和 defaultContent, 不会在下次部署被回滚)。

## 判据

    content != default_content  →  被人工改过

不要看版本表的 `source` 字段。bootstrap 播种时它写的是 'db', 一眼看去像是人工
操作 —— 我就这么误判过一次, 幸好两个模板恰好都没被改过。

用法 (生产容器内):
    python check_prompt_edits.py portrait.generation portrait.update
    python check_prompt_edits.py --all      # 列出全部被改过的
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from app.db import db


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*", help="要检查的 prompt key")
    ap.add_argument("--all", action="store_true", help="列出全部被人工改过的")
    args = ap.parse_args()
    if not args.keys and not args.all:
        raise SystemExit("给出 key, 或用 --all")

    await db.connect()
    if args.all:
        rows = await db.query_raw(
            """
            SELECT key, LENGTH(content)::int AS content_len,
                   LENGTH(default_content)::int AS default_len
            FROM prompt_templates
            WHERE content IS DISTINCT FROM default_content
            ORDER BY key
            """
        )
        print(f"被人工改过的提示词: {len(rows)} 个")
        for r in rows:
            print(f"  {r['key']:<40} 生效 {r['content_len']:>5} 字 / "
                  f"默认 {r['default_len']:>5} 字")
        if rows:
            print("\n改这些的代码默认值会覆盖后台编辑 —— 走 update_prompt_text 提交新版本")
        await db.disconnect()
        return

    edited = []
    for key in args.keys:
        rows = await db.query_raw(
            """
            SELECT LENGTH(content)::int AS content_len,
                   LENGTH(default_content)::int AS default_len,
                   content IS NOT DISTINCT FROM default_content AS same
            FROM prompt_templates WHERE key = $1
            """,
            key,
        )
        if not rows:
            print(f"{key}: 不在 DB 里 (新增的 key, 部署后才会播种)")
            continue
        row = rows[0]
        if row["same"]:
            print(f"{key}: 未被人工编辑 —— 可以直接改代码默认值")
        else:
            edited.append(key)
            print(f"{key}: ⚠ 已被人工编辑 "
                  f"(生效 {row['content_len']} 字 / 默认 {row['default_len']} 字)")
            print("    改代码默认值会覆盖它。走 update_prompt_text 提交新版本。")

    await db.disconnect()
    sys.exit(1 if edited else 0)


if __name__ == "__main__":
    asyncio.run(main())
