from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/companion"
    direct_database_url: str = ""

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"

    # DashScope / Alibaba Cloud Bailian (OpenAI-compatible)
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_enable_thinking: bool = False

    # Simple model switch
    online_model: bool = False
    local_chat_model: str = "qwen2.5:14b"
    local_small_model: str = "qwen2.5:7b"
    remote_chat_model: str = "qwen3.5-plus"
    remote_small_model: str = "qwen3.5-flash"

    # Embedding (always via Ollama, set EMBEDDING_MODEL in .env)
    embedding_model: str = "bge-m3"
    embedding_dimensions: int = 1024

    # Advanced / legacy overrides
    chat_model: str = ""
    utility_model: str = ""
    prefilter_model: str = ""  # Override for pre-filter model (default: utility_model)
    enable_memory_prefilter: bool = True  # Spec §2.1.2: small model "记/不记" before big model extraction
    ollama_model: str = ""
    llm_provider: str = ""
    chat_provider: str = ""
    utility_provider: str = ""
    embedding_provider: str = ""

    # Anthropic (when llm_provider == "claude")
    anthropic_api_key: str = ""

    # Schedule
    schedule_timezone: str = "Asia/Shanghai"

    # LangSmith tracing
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_org_id: str = ""
    langsmith_project_id: str = ""

    # JWT authentication
    jwt_secret: str = ""
    jwt_expiry_hours: int = 168  # 7 days

    # Admin prompt management
    admin_username: str = ""
    admin_password: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
