from pydantic import BaseModel


class UserUpdate(BaseModel):
    email: str | None = None


class UserResponse(BaseModel):
    id: str
    username: str
    email: str | None = None
    created_at: str | None = None
