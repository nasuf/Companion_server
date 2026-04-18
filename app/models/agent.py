from pydantic import BaseModel


class PersonalityInput(BaseModel):
    """用户在前端拖动 7 个直觉滑块得到的整数百分比 (0-100)。

    spec §1.1: 活泼度/理性度/感性度/计划度/随性度/脑洞度/幽默度。
    后端收到后确定性映射为 MBTI 4 维, 不存储 7 维本身。
    """
    lively: int       # 活泼度 → E/I
    rational: int     # 理性度 → T/F (正)
    emotional: int    # 感性度 → T/F (反)
    planned: int      # 计划度 → J/P (正)
    spontaneous: int  # 随性度 → J/P (反)
    creative: int     # 脑洞度 → N/S
    humor: int        # 幽默度 → 不直接映射 MBTI, 作为 summary seed


class MbtiInput(BaseModel):
    """MBTI 4 维百分比 (admin 直编 / regenerate-mbti 用)。

    每个值 0-100；> 50 偏向首字母 (E/N/T/J)，<= 50 偏向后字母 (I/S/F/P)。
    """
    EI: int
    NS: int
    TF: int
    JP: int


class AgentCreate(BaseModel):
    name: str
    user_id: str
    # spec §1.1: 用户填 7 维性格, 后端转 MBTI
    personality: PersonalityInput
    background: str | None = None
    values: dict | None = None
    gender: str | None = None


class RegenerateMbtiRequest(BaseModel):
    """POST /agents/{id}/regenerate-mbti — admin 直编 4 维 MBTI。"""
    mbti: MbtiInput


class AgentUpdate(BaseModel):
    name: str | None = None
    background: str | None = None
    values: dict | None = None


class AgentResponse(BaseModel):
    id: str
    name: str
    user_id: str
    workspace_id: str | None = None
    mbti: dict | None = None
    background: str | None = None
    values: dict | None = None
    gender: str | None = None
    life_overview: str | None = None
    created_at: str | None = None
