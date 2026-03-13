from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/companion"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j_password"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"

    # LLM models
    chat_model: str = "qwen2.5:14b"
    summarizer_model: str = "qwen2.5:7b"
    ollama_model: str = "qwen2.5:7b"

    # Embedding model
    embedding_model: str = "nomic-embed-text"

    # LLM provider: "ollama" or "claude"
    llm_provider: str = "ollama"

    # Anthropic (when llm_provider == "claude")
    anthropic_api_key: str = ""

    # LangSmith tracing
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
