from pydantic import BaseModel


class AgentCreate(BaseModel):
    name: str
    user_id: str
    # 用户在前端填的性格表单（7 维或 Big Five，自动检测）。仅作为创建时
    # 计算 MBTI 的 transient 输入，不会持久化。spec §1.2: MBTI 是 canonical。
    seven_dim_input: dict | None = None
    background: str | None = None
    values: dict | None = None
    gender: str | None = None


class RegenerateMbtiRequest(BaseModel):
    """POST /agents/{id}/regenerate-mbti 的可选 body。

    不传 → LLM 基于现有 mbti 重抖一次（带轻微随机 seed 不一定能保证完全
    复现；适合"我不喜欢这个性格，重新生成一个"场景）。
    传 seven_dim_input → 用户重新填了 7 维，覆盖原 MBTI。
    """
    seven_dim_input: dict | None = None


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
