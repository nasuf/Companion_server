from pydantic import model_validator
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

    # 回复延迟开关 (spec §6 异步回复机制).
    # False (默认): compute_delay_profile 直接返 0, ws 走同步快路径 (跳过 delayed
    #   queue + scheduler 调度), 用户看不到 "已排队" 提示, 测试时反馈即时.
    # True (生产): 走 spec §6 完整流程 — 按 PAD/作息计算 delay_seconds,
    #   入 delayed queue, scheduler 每秒扫到期推送, 模拟真人间隔回复.
    reply_delay_enabled: bool = False

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

    # LLM resilience layer (app/services/llm/resilience.py)
    # 紧急 kill switch: 设为 False 时, 所有 LLM 调用只保留 per-profile timeout,
    # 跳过 circuit breaker + retry + Ollama fallback (回到原始行为)
    llm_resilience_enabled: bool = True
    # Circuit breaker: 滑动窗口内连续失败次数达到 threshold 则 open
    llm_cb_failure_threshold: int = 5
    llm_cb_window_sec: float = 10.0
    # open 状态持续 cooldown_sec 后进入 half_open, 放 1 个 probe
    llm_cb_cooldown_sec: float = 30.0
    # Per-profile timeout (秒). 小模型快分类 / 大模型抽取 / 大模型流式
    llm_utility_timeout_s: float = 8.0
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

    # 防御性: GitHub Actions vars.X 未设时, deploy.yml heredoc 把 .env 写成
    # X=, pydantic 不能把空串 parse 成 bool/int/float, 直接 ValidationError
    # 让进程起不来. 这里在 model 解析前把所有 "" 值从输入里剔除, pydantic 找
    # 不到字段值就走 field default — 等价于 env 没设. 不影响显式 = 0 / = false.
    # str 字段默认值多数本来就是 "" (api_key 等), 剔除后用 default 仍是 "" 一致.
    @model_validator(mode="before")
    @classmethod
    def _strip_empty_envs(cls, data):
        if isinstance(data, dict):
            return {
                k: v for k, v in data.items()
                if not (isinstance(v, str) and v.strip() == "")
            }
        return data

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
