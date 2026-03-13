from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    postgres: bool
    redis: bool
    neo4j: bool
