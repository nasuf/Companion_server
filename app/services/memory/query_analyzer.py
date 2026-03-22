"""查询分析器 — 用于混合检索的意图分析。"""

import logging

from app.services.llm.models import get_utility_model, invoke_json

logger = logging.getLogger(__name__)

ANALYZER_PROMPT = """分析用户消息，判断需要什么类型的检索。

用户消息：{message}

返回JSON：
{{
  "intent": "ask_preference|ask_event|ask_person|casual|greeting|question|other",
  "entities": ["提到的命名实体列表"],
  "retrieve_memory": true或false,
  "retrieve_graph": true或false,
  "retrieve_structured": true或false
}}

规则：
- retrieve_memory：消息涉及过去的对话或需要上下文时为true
- retrieve_graph：消息涉及人际关系、偏好或实体时为true
- retrieve_structured：消息询问具体存储数据（姓名、生日等）时为true
- 日常问候时，全部可以为false
"""


async def analyze_query(message: str) -> dict:
    """Analyze user message to determine retrieval strategy."""
    model = get_utility_model()
    prompt = ANALYZER_PROMPT.format(message=message)

    try:
        result = await invoke_json(model, prompt)
        result.setdefault("intent", "other")
        result.setdefault("entities", [])
        result.setdefault("retrieve_memory", True)
        result.setdefault("retrieve_graph", False)
        result.setdefault("retrieve_structured", False)
        return result
    except Exception as e:
        logger.warning(f"Query analysis failed: {e}")
        return {
            "intent": "other",
            "entities": [],
            "retrieve_memory": True,
            "retrieve_graph": False,
            "retrieve_structured": False,
        }
