from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Runtime environment
    app_env: str = "development"

    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/companion"
    direct_database_url: str = ""
    migration_database_url: str = ""
    db_connection_limit: int = 3
    db_connection_limit_max: int = 5
    db_max_concurrent_queries: int = 3
    db_query_max_retries: int = 4

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"

    # DashScope / Alibaba Cloud Bailian (OpenAI-compatible)
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_enable_thinking: bool = False

    # DeepSeek direct API (OpenAI-compatible)
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"

    # Simple model switch
    online_model: bool = False
    remote_provider: str = "dashscope"
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

    # 回复延迟开关 (spec §6 异步回复机制).
    # False (默认): compute_delay_profile 直接返 0, ws 走同步快路径 (跳过 delayed
    #   queue + scheduler 调度), 用户看不到 "已排队" 提示, 测试时反馈即时.
    # True (生产): 走 spec §6 完整流程 — 按用户情绪标签/作息计算 delay_seconds,
    #   入 delayed queue, scheduler 每秒扫到期推送, 模拟真人间隔回复.
    reply_delay_enabled: bool = False

    # LangSmith tracing
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_org_id: str = ""
    langsmith_project_id: str = ""

    # Axiom 结构化日志 (https://app.axiom.co). 字段在此声明仅为通过 Settings 校验
    # (避免 extra_forbidden); 实际运行 app/observability/axiom_setup.py 走 os.getenv.
    axiom_token: str = ""
    axiom_dataset: str = ""
    axiom_org_id: str = ""
    axiom_log_level: str = "INFO"

    # JWT authentication
    jwt_secret: str = ""
    jwt_expiry_hours: int = 168  # 7 days
    # Optional separate key for last-will content/contact encryption. If unset,
    # production falls back to the strong JWT secret enforced below.
    last_will_encryption_key: str = ""

    # WeChat Open Platform mobile app login. AppSecret must stay server-side.
    wechat_login_enabled: bool = False
    wechat_mobile_app_id: str = ""
    wechat_mobile_app_secret: str = ""
    wechat_oauth_timeout_s: float = 6.0

    # SUD / SudGIP mini-game integration. The Flutter client uses app_id/app_key
    # to initialize SudGIP, while app_secret stays server-side for short-lived
    # code / ss_token signing and callback validation.
    sud_app_id: str = ""
    sud_app_key: str = ""
    sud_app_secret: str = ""
    sud_default_mg_id: str = ""
    sud_bundle_id: str = ""
    sud_is_test_env: bool = True
    sud_callback_public_base_url: str = ""

    # CORS. Comma-separated list, e.g. "https://app.example.com,https://admin.example.com".
    # Development defaults to "*" for local convenience; production must configure
    # an explicit allowlist.
    cors_allowed_origins: str = ""

    # LLM resilience layer (app/services/llm/resilience.py)
    # 紧急 kill switch: 设为 False 时, 所有 LLM 调用只保留 per-profile timeout,
    # 跳过 circuit breaker + retry + Ollama fallback (回到原始行为)
    llm_resilience_enabled: bool = True
    # Circuit breaker: 滑动窗口内连续失败次数达到 threshold 则 open
    # threshold=10 容许一次 chat (data_fetch_phase 7 并发 utility 调用) 的 burst
    # 失败而不立刻开 CB — provider 真挂时 10 次失败仍能短时间内累计.
    llm_cb_failure_threshold: int = 10
    llm_cb_window_sec: float = 10.0
    # open 状态持续 cooldown_sec 后进入 half_open, 放 1 个 probe
    llm_cb_cooldown_sec: float = 30.0
    # Per-profile timeout (秒). 小模型快分类 / 大模型抽取 / 大模型流式
    # utility=12s: dev 跨地域到 dashscope 网络抖动时 8s 容易超, 给点 buffer.
    llm_utility_timeout_s: float = 12.0
    llm_chat_extract_timeout_s: float = 45.0
    llm_chat_stream_timeout_s: float = 90.0
    # 流式首 chunk 超时 (连不上 / 模型未加载时触发 fallback, 防用户长时间无响应)
    llm_chat_stream_first_chunk_timeout_s: float = 30.0

    # admin 批量生成 character profile 单请求最大数量 + 内部并发上限.
    # 单请求 N 个 profile 时, 后端用 Semaphore 控制实际并发 LLM 数, 防止 100
    # 个同时打 DashScope 触发 429. 总耗时 ≈ ceil(N / concurrency) × 单次 LLM 时间.
    # 默认 10 是 DashScope 默认 tier 安全值; 升级 tier 后可调高 env 覆盖。
    character_profile_batch_max: int = 100
    character_profile_batch_concurrency: int = 10

    # Redis client resilience (app/redis_client.py)
    # socket_timeout 防止 Redis 卡顿永久阻塞 asyncio event loop; 超时触发后抛
    # redis.TimeoutError, 继承自 RedisError, 下游 try/except 可捕获走降级.
    # 5s 是所有 Lua 脚本/单 op 的合理上限; 连接阶段给 2s 足够 (内网 < 100ms).
    redis_socket_timeout_s: float = 5.0
    redis_connect_timeout_s: float = 2.0
    redis_max_connections: int = 50

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    def is_production(self) -> bool:
        return self.app_env.strip().lower() in {"prod", "production"}

    def cors_origins(self) -> list[str]:
        raw = self.cors_allowed_origins.strip()
        if not raw:
            return [] if self.is_production() else ["*"]
        return [origin.strip() for origin in raw.split(",") if origin.strip()]

    def validate_security_config(self) -> None:
        """Fail fast on unsafe production security settings.

        Development stays permissive so local tests and prototypes do not need a
        full deployment secret set. Production must be explicit.
        """
        if not self.is_production():
            return

        weak_jwt_values = {"", "change_me", "changeme", "secret", "dev", "development"}
        jwt_secret = self.jwt_secret.strip()
        if jwt_secret.lower() in weak_jwt_values or len(jwt_secret) < 32:
            raise RuntimeError(
                "Unsafe production config: JWT_SECRET must be set to a strong value "
                "with at least 32 characters."
            )

        origins = self.cors_origins()
        if not origins or "*" in origins:
            raise RuntimeError(
                "Unsafe production config: CORS_ALLOWED_ORIGINS must be an explicit "
                "comma-separated allowlist."
            )

        if self.wechat_login_enabled and (
            not self.wechat_mobile_app_id.strip()
            or not self.wechat_mobile_app_secret.strip()
        ):
            raise RuntimeError(
                "Unsafe production config: WECHAT_LOGIN_ENABLED=true requires "
                "WECHAT_MOBILE_APP_ID and WECHAT_MOBILE_APP_SECRET."
            )


settings = Settings()
