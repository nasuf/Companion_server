"""聊天回复约束 + token 预算常量。

历史: 这里曾从 defaults.py 重导出 SYSTEM_BASE / CONSISTENCY_RULES 等 prompt
别名, 但全仓没消费方——chat 路径已通过 get_prompt_text("chat.system_base")
等 registry key 取用. 别名已删, 文件保留是为了集中放回复 / token 预算常量.
"""

# 回复约束
MAX_PER_REPLY = 60           # 单条回复最大字数
# spec §5.5 原值 3 (1-3 均匀); 2026-07-20 图灵测试版 response_instruction (web
# 定制) 要求 1-4 条随机 — 上限升 4, 否则 split 的 parts[:max_count] 会把第 4 条
# 静默丢弃 (内容丢失). 条数由 LLM 决定 + chat.reply_count_variation 段约束
# "≠上一轮", 这里只是硬上限兜底.
MAX_REPLY_COUNT = 4
MAX_TOTAL_CHARS = 150        # 总字数上限

# Token预算常量
MEMORY_TOKEN_BUDGET = 800
SUMMARIZER_TOKEN_BUDGET = 600
GRAPH_CONTEXT_TOKEN_BUDGET = 200
MAX_SYSTEM_PROMPT_TOKENS = 2000
CHAT_HISTORY_TOKEN_BUDGET = 4000  # 聊天记录 token 预算，从最新消息往前填充
