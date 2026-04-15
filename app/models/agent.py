from pydantic import BaseModel


class AgentCreate(BaseModel):
    name: str
    user_id: str
    # 用户在前端填的 7 维 / Big Five 性格表单，仅作为创建时计算 MBTI 的
    # transient 输入，不会持久化。spec §1.2: MBTI 才是 canonical。
    personality: dict | None = None
    background: str | None = None
    values: dict | None = None
    gender: str | None = None


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
