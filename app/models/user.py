from pydantic import BaseModel


class UserCreate(BaseModel):
    name: str
    email: str | None = None


class UserUpdate(BaseModel):
    name: str | None = None
    email: str | None = None


class UserResponse(BaseModel):
    id: str
    name: str
    email: str | None = None
    created_at: str | None = None
