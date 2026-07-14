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

    # Volcengine Ark / Doubao vision (OpenAI-compatible chat completions)
    ark_api_key: str = ""
    ark_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    doubao_vision_model: str = "doubao-1-5-vision-pro-32k-250115"

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
    # When a message hits a high-precision emotion keyword, skip the utility LLM
    # emotion call (~300-400 tok + 200-600ms saved). Ambiguous messages (no
    # keyword) still use the LLM. Emotion is a soft signal (emoji/tone), so the
    # coarser keyword estimate is an acceptable trade for the latency/cost win.
    emotion_keyword_fast_path: bool = True
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

    # Phase E1 (拟人度): 错别字生成器 — 以 typo_rate 概率给回复注入一个高频
    # 同音错字, ~50% 概率追加 "*正确字" 纠正气泡 (typo.py).
    # 2026-07-03 产品决策: 默认开启 (rate=0.05, 每 20 条约 1 条错字).
    # 用户负反馈时可 .env TYPO_ENABLED=false 一键关闭.
    typo_enabled: bool = True
    typo_rate: float = 0.05

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
    # WeChat Mini Program login (jscode2session). Uses the Mini Program's own
    # AppID/AppSecret. UnionID is only returned when the Mini Program is bound to
    # the same WeChat Open Platform account as the mobile app, which is required
    # for cross-platform account/conversation continuity.
    wechat_mini_app_id: str = ""
    wechat_mini_app_secret: str = ""
    # WeChat Official Account (服务号) credentials for the H5 web-page OAuth
    # login (公众号网页授权). Same sns/oauth2 flow as the mobile app but with the
    # OA's appid/secret; unionid continuity likewise requires the OA to be bound
    # to the same WeChat Open Platform account.
    wechat_h5_app_id: str = ""
    wechat_h5_app_secret: str = ""
    # 微信 H5 JS-SDK 页面签名只允许这些 origin。逗号分隔，必须写协议，
    # 例如 https://banshengcomp.com。与「网页授权域名」/CORS 是独立配置。
    wechat_jssdk_allowed_origins: str = ""

    # 霸王餐服务员页访问口令: 登录 staff.html 后获得短时 JWT，再调用微信扫一扫。
    # 请配全大写英文字母；空值仅允许本地开发，生产会拒绝服务员登录。
    meal_staff_key: str = ""
    meal_staff_jwt_expiry_hours: int = 12

    # 霸王餐券服务员校验后的有效期 (天): activatedAt 起算，超时未核销即过期，
    # 无法再找商家兑换. 有效期口径按 UTC+8 自然时间 (activatedAt + N 天).
    meal_validity_days: int = 7
    # 每日霸王餐核销总量上限 (先到先得): 单个 UTC+8 自然日内核销总数达到该值后,
    # 后续核销请求被拒并留痕, 提示用户次日 (仍需在有效期内) 再来.
    meal_daily_redeem_cap: int = 1000
    # 用户券动态二维码：Redis 一次性凭证有效期。页面会在过期前自动刷新。
    meal_qr_ttl_seconds: int = 60
    # 商家自助登录 JWT 使用独立短有效期，不复用普通用户默认 7 天。
    meal_merchant_jwt_expiry_hours: int = 12

    # Optional: a fully-provisioned "template" agent id. When set, a brand-new
    # user (e.g. first WeChat Mini Program login) is given an instant clone of
    # this agent (persona + L1 memory + embeddings copied, no LLM), so they can
    # chat immediately. Each clone is fully isolated per user. Empty = disabled
    # (new users stay agent-less until they build one via the normal flow).
    default_template_agent_id: str = ""

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

    # iOS remote notifications via APNs. Disabled unless APNS_ENABLED=true and
    # token-auth credentials are configured.
    apns_enabled: bool = False
    apns_team_id: str = ""
    apns_key_id: str = ""
    apns_auth_key: str = ""
    apns_auth_key_path: str = ""
    apns_topic: str = ""
    apns_use_sandbox: bool = True
    notification_max_attempts: int = 3
    notification_dispatch_batch_size: int = 50

    # Jamendo music integration. The client id is configured on the server;
    # Flutter receives normalized metadata plus Jamendo file endpoint URLs.
    jamendo_client_id: str = ""
    jamendo_base_url: str = "https://api.jamendo.com/v3.0"
    jamendo_default_libraries: str = "focus,ambient,sleep"

    # Link-card proactive recommendations. The search endpoint is optional: if
    # configured it should return JSON with `results: [{url: "..."}]` for a
    # query/platform payload. Candidate URLs are a deterministic fallback pool.
    proactive_link_recommendation_enabled: bool = True
    proactive_link_recommendation_probability: float = 0.03
    proactive_link_candidate_urls: str = ""
    chat_link_search_provider: str = "custom"
    chat_link_search_endpoint: str = ""
    chat_link_search_api_key: str = ""
    chat_link_search_timeout_s: float = 8.0
    tavily_api_key: str = ""
    tavily_search_endpoint: str = "https://api.tavily.com/search"
    brave_search_api_key: str = ""
    brave_search_endpoint: str = "https://api.search.brave.com/res/v1/web/search"

    # Real-world gift commerce/logistics. Keep the default mock provider for
    # local/dev. In production, point custom_http at a buyer-side purchasing
    # service that can legally search, order, pay, and track gifts.
    gift_commerce_provider: str = "mock"
    gift_commerce_base_url: str = ""
    gift_commerce_api_key: str = ""
    gift_commerce_timeout_s: float = 12.0
    gift_logistics_provider: str = "mock"
    gift_logistics_base_url: str = ""
    gift_logistics_api_key: str = ""
    gift_logistics_timeout_s: float = 10.0

    # 1688 开放平台采购接入（gift_commerce_provider / gift_logistics_provider = "ali1688"）。
    # app_key/app_secret 在 open.1688.com 创建应用获得；access_token 走 OAuth 授权采购账号后获得，
    # 会过期，需用 refresh_token 定期续期（建议另起 cron 刷新后写回此处）。
    ali1688_app_key: str = ""
    ali1688_app_secret: str = ""
    ali1688_access_token: str = ""
    ali1688_refresh_token: str = ""  # OAuth 授权得到，cron 用它定期换新 access_token
    ali1688_search_recall: int = 40  # 关键词召回条数（再做硬过滤+粗排）
    ali1688_require_one_piece: bool = True  # 是否强制只要支持一件代发/起订量<=1 的商品

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

    # Proactive: a state claimed into `processing` that never reaches a terminal
    # transition (instance OOM/killed mid-send) would stall that workspace's
    # proactive messaging forever. The scan reclaims `processing` states whose
    # last claim is older than this many seconds back to a re-eligible status.
    proactive_processing_timeout_s: int = 300

    # fire_background concurrency ceiling. Each user turn fans out several
    # background tasks (memory extraction, PAD, trait, achievements, ...); under
    # a storm this bounds concurrent execution (tasks queue, never drop) so the
    # event loop / memory can't be overwhelmed. Set high enough that normal
    # traffic never queues; crossing the high-water mark logs a warning.
    background_task_max_concurrency: int = 256

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

    def wechat_jssdk_origins(self) -> list[str]:
        raw = self.wechat_jssdk_allowed_origins.strip()
        if not raw:
            return []
        return [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]

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

        if self.wechat_jssdk_origins() and (
            not self.wechat_h5_app_id.strip()
            or not self.wechat_h5_app_secret.strip()
        ):
            raise RuntimeError(
                "Unsafe production config: WECHAT_JSSDK_ALLOWED_ORIGINS requires "
                "WECHAT_H5_APP_ID and WECHAT_H5_APP_SECRET."
            )


settings = Settings()
