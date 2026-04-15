from pydantic import BaseModel


class MbtiInput(BaseModel):
    """用户在前端拖动 4 个 MBTI 滑块得到的整数百分比。

    每个值 0-100；> 50 偏向首字母 (E/N/T/J)，<= 50 偏向后字母 (I/S/F/P)。
    """
    EI: int
    NS: int
    TF: int
    JP: int


class AgentCreate(BaseModel):
    name: str
    user_id: str
    # spec §1.2: MBTI 是 canonical，用户直接填 4 维，不再有 7 维中转。
    mbti: MbtiInput
    background: str | None = None
    values: dict | None = None
    gender: str | None = None


class RegenerateMbtiRequest(BaseModel):
    """POST /agents/{id}/regenerate-mbti 的 body。

    用户重抖性格 → 拖动 4 个 MBTI 滑块得到新的 EI/NS/TF/JP 百分比。
    后端按这 4 个数字重写 mbti + currentMbti，并让 LLM 重新生成 summary。
    """
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
    # MBTI 是 canonical 性格表达 (spec §1.2)
    mbti: dict | None = None
    background: str | None = None
    values: dict | None = None
    gender: str | None = None
    life_overview: str | None = None
    created_at: str | None = None
