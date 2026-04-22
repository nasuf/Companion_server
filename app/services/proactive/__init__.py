"""主动聊天领域服务。"""

from app.services.proactive.history import (
    can_send_proactive,
    get_proactive_history,
    increment_proactive_count,
)
