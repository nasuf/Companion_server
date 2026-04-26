"""聊天回复约束 + token 预算常量。

历史: 这里曾从 defaults.py 重导出 SYSTEM_BASE / CONSISTENCY_RULES 等 prompt
别名, 但全仓没消费方——chat 路径已通过 get_prompt_text("chat.system_base")
等 registry key 取用. 别名已删, 文件保留是为了集中放回复 / token 预算常量.
"""

# 回复约束 (spec §5.5: n = random.randint(1, 3) 均匀分布)
MAX_PER_REPLY = 60           # 单条回复最大字数
MAX_REPLY_COUNT = 3          # spec §5.5: 严格 1-3 均匀随机
MAX_TOTAL_CHARS = 150        # 总字数上限

# Token预算常量
MEMORY_TOKEN_BUDGET = 800
SUMMARIZER_TOKEN_BUDGET = 600
GRAPH_CONTEXT_TOKEN_BUDGET = 200
MAX_SYSTEM_PROMPT_TOKENS = 2000
CHAT_HISTORY_TOKEN_BUDGET = 4000  # 聊天记录 token 预算，从最新消息往前填充
