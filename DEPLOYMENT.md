# Companion Server Deployment

This repo deploys to the Tencent Cloud CVM through GitHub Actions.

## What the workflow deploys

- `companion-server` bound to `127.0.0.1:8000`
- Self-hosted Postgres + pgvector on the CVM data disk
- Self-hosted Redis on the CVM data disk
- Self-hosted Ollama for `bge-m3` embeddings on the CVM data disk
- Prompt template migration applied during deploy

Server path on the VPS:

- `/app/companion-server`

## GitHub configuration

Use both repository `Secrets` and repository `Variables`.

### VPS access

- `VPS_SSH_KEY`

### Repository Secrets

- `VPS_HOST`
- `POSTGRES_PASSWORD`
- `JWT_SECRET`
- `WECHAT_MOBILE_APP_SECRET`
- `DASHSCOPE_API_KEY`
- `DEEPSEEK_API_KEY`
- `LANGSMITH_API_KEY`
- `LANGSMITH_ORG_ID`
- `LANGSMITH_PROJECT_ID`

Optional:

- `ANTHROPIC_API_KEY`

### Repository Variables

- `VPS_PORT`
- `VPS_USERNAME`
- `POSTGRES_USER`
- `POSTGRES_DB`
- `ONLINE_MODEL`
- `REMOTE_PROVIDER`
- `OLLAMA_BASE_URL`
- `DASHSCOPE_BASE_URL`
- `DASHSCOPE_ENABLE_THINKING`
- `DEEPSEEK_BASE_URL`
- `LOCAL_CHAT_MODEL`
- `LOCAL_SMALL_MODEL`
- `REMOTE_CHAT_MODEL`
- `REMOTE_SMALL_MODEL`
- `LANGSMITH_TRACING`
- `CORS_ALLOWED_ORIGINS`
- `WECHAT_LOGIN_ENABLED`
- `WECHAT_MOBILE_APP_ID`

### Database and cache

## Recommended current values

### Database

```env
POSTGRES_USER=companion
POSTGRES_DB=companion
POSTGRES_PASSWORD=<strong-url-safe-password>
DATABASE_URL=postgresql://companion:<password>@postgres:5432/companion?connection_limit=3&pool_timeout=30&connect_timeout=30
DIRECT_DATABASE_URL=postgresql://companion:<password>@postgres:5432/companion?connection_limit=1&pool_timeout=60&connect_timeout=30
MIGRATION_DATABASE_URL=postgresql://companion:<password>@postgres:5432/companion?connection_limit=1&pool_timeout=60&connect_timeout=30
DB_CONNECTION_LIMIT=3
DB_CONNECTION_LIMIT_MAX=5
DB_MAX_CONCURRENT_QUERIES=3
DB_QUERY_MAX_RETRIES=4
```

### Redis

```env
REDIS_URL=redis://redis:6379/4
```

### Model switch

For Alibaba Cloud Bailian / DashScope:

```env
VPS_PORT=22
VPS_USERNAME=ubuntu
ONLINE_MODEL=true
REMOTE_PROVIDER=dashscope
OLLAMA_BASE_URL=http://ollama:11434
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_ENABLE_THINKING=false
LOCAL_CHAT_MODEL=qwen2.5:14b
LOCAL_SMALL_MODEL=qwen2.5:7b
REMOTE_CHAT_MODEL=qwen3.5-plus
REMOTE_SMALL_MODEL=qwen3.5-flash
LANGSMITH_TRACING=true
CORS_ALLOWED_ORIGINS=https://your-web-domain.example.com
WECHAT_LOGIN_ENABLED=false
WECHAT_MOBILE_APP_ID=
WECHAT_MOBILE_APP_SECRET=
```

For DeepSeek direct API:

```env
ONLINE_MODEL=true
REMOTE_PROVIDER=deepseek
DEEPSEEK_BASE_URL=https://api.deepseek.com
LOCAL_CHAT_MODEL=qwen2.5:14b
LOCAL_SMALL_MODEL=qwen2.5:7b
REMOTE_CHAT_MODEL=deepseek-v4-pro
REMOTE_SMALL_MODEL=deepseek-v4-flash
```

For local Ollama:

```env
ONLINE_MODEL=false
REMOTE_PROVIDER=dashscope
OLLAMA_BASE_URL=http://ollama:11434
DASHSCOPE_API_KEY=
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_ENABLE_THINKING=false
DEEPSEEK_BASE_URL=https://api.deepseek.com
LOCAL_CHAT_MODEL=qwen2.5:14b
LOCAL_SMALL_MODEL=qwen2.5:7b
REMOTE_CHAT_MODEL=qwen3.5-plus
REMOTE_SMALL_MODEL=qwen3.5-flash
```

## Notes

- Postgres runtime guidance:
  - `DATABASE_URL` is for the long-running server process and should keep `connection_limit` small on the 4C/8G CVM.
  - `DB_CONNECTION_LIMIT_MAX` is a hard runtime cap for accidental oversized values. Increase it only after the database connection budget has been increased.
  - Keep `DB_MAX_CONCURRENT_QUERIES` less than or equal to `DB_CONNECTION_LIMIT` so request bursts queue in the app instead of exhausting database sessions.
  - `MIGRATION_DATABASE_URL` is for Prisma CLI / migrations only and should use `connection_limit=1`.
  - Do not run Prisma migrations through a transaction pooler because Prisma Migrate needs a stable database connection.
- The memory system still uses embeddings internally, but you do not need to configure an embedding model separately anymore.
- `ONLINE_MODEL=true` means chat / utility calls use `REMOTE_PROVIDER` (`dashscope`, `deepseek`, or `claude`) plus `REMOTE_*` model ids.
- `ONLINE_MODEL=false` means chat / utility calls use local Ollama defaults.
- Embeddings stay on the embedding provider path and are not switched to DeepSeek by `REMOTE_PROVIDER`.
- The backend API is not exposed directly to the public internet in this deploy shape; Nginx on the web repo proxies requests to `127.0.0.1:8000`.
- Admin APIs, including the prompt console endpoints, are protected by JWT admin role checks.
- Keep local and deployed environments on different Redis DBs. Recommended:
  - local: `redis://localhost:6380/0`
  - production container network: `redis://redis:6379/4`
