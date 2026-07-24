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
- `APNS_AUTH_KEY`
- `APNS_KEY_ID`
- `CHAT_LINK_SEARCH_API_KEY` (custom provider)
- `TAVILY_API_KEY` (Tavily provider)
- `BRAVE_SEARCH_API_KEY` (Brave provider)

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
- `TRACE_BACKEND` (default `local`; self-hosted trace collection, no LangSmith quota)
- `TRACE_RETENTION_DAYS` (default `30`)
- `LANGSMITH_TRACING` (keep `false` unless `TRACE_BACKEND=langsmith`)
- `CORS_ALLOWED_ORIGINS`
- `WECHAT_LOGIN_ENABLED`
- `WECHAT_MOBILE_APP_ID`
- `APNS_ENABLED`
- `APNS_TEAM_ID`
- `APNS_TOPIC`
- `APNS_USE_SANDBOX`
- `NOTIFICATION_MAX_ATTEMPTS`
- `NOTIFICATION_DISPATCH_BATCH_SIZE`
- `PROACTIVE_LINK_RECOMMENDATION_ENABLED`
- `PROACTIVE_LINK_RECOMMENDATION_PROBABILITY`
- `PROACTIVE_LINK_CANDIDATE_URLS`
- `CHAT_LINK_SEARCH_PROVIDER`
- `CHAT_LINK_SEARCH_ENDPOINT`
- `CHAT_LINK_SEARCH_TIMEOUT_S`
- `TAVILY_SEARCH_ENDPOINT`
- `BRAVE_SEARCH_ENDPOINT`

### Database and cache

## Recommended current values

### Database

```env
POSTGRES_USER=companion
POSTGRES_DB=companion
POSTGRES_PASSWORD=<strong-url-safe-password>
DATABASE_URL=postgresql://companion:<password>@postgres:5432/companion?connection_limit=12&pool_timeout=30&connect_timeout=30
DIRECT_DATABASE_URL=postgresql://companion:<password>@postgres:5432/companion?connection_limit=1&pool_timeout=60&connect_timeout=30
MIGRATION_DATABASE_URL=postgresql://companion:<password>@postgres:5432/companion?connection_limit=1&pool_timeout=60&connect_timeout=30
DB_CONNECTION_LIMIT=12
DB_CONNECTION_LIMIT_MAX=15
DB_MAX_CONCURRENT_QUERIES=10
DB_QUERY_MAX_RETRIES=4
```

### Redis

```env
REDIS_URL=redis://redis:6379/0
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
TRACE_BACKEND=local
LANGSMITH_TRACING=false
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

### Music provider

Jamendo is configured on the server during GitHub Actions deploy. Store the
client id as a GitHub secret and keep non-secret provider defaults as GitHub
variables:

```env
# GitHub Secret
JAMENDO_CLIENT_ID=2721f58a

# GitHub Variables
JAMENDO_BASE_URL=https://api.jamendo.com/v3.0
JAMENDO_DEFAULT_LIBRARIES=focus,ambient,sleep
```

`deploy.yml` writes these values into the server `.env` on the VPS. `JAMENDO_CLIENT_ID`
is required for production deploy; the base URL and default libraries have deploy
fallbacks matching the values above.

### Chat link cards and proactive link search

Link cards are created when users paste/share links from Xiaohongshu, Weibo,
Toutiao, Douyin, and Zhihu. Proactive link recommendations can also search
those platforms and send an `external_link` card from the agent. The agent
never invents URLs: search results are filtered to supported platform domains
before the backend fetches metadata and creates a card.

Choose one provider:

```env
# Disable live search and use only PROACTIVE_LINK_CANDIDATE_URLS fallback.
CHAT_LINK_SEARCH_PROVIDER=custom
CHAT_LINK_SEARCH_ENDPOINT=

# Custom internal provider. It must accept:
#   POST <endpoint> {"query": "...", "platforms": [...], "limit": 5}
# and return either {"results": [{"url": "..."}]} or a list of URLs/items.
CHAT_LINK_SEARCH_PROVIDER=custom
CHAT_LINK_SEARCH_ENDPOINT=https://your-search-service.example.com/chat-links
CHAT_LINK_SEARCH_API_KEY=<secret>

# Tavily Search API.
CHAT_LINK_SEARCH_PROVIDER=tavily
TAVILY_API_KEY=<secret>
TAVILY_SEARCH_ENDPOINT=https://api.tavily.com/search

# Brave Web Search API.
CHAT_LINK_SEARCH_PROVIDER=brave
BRAVE_SEARCH_API_KEY=<secret>
BRAVE_SEARCH_ENDPOINT=https://api.search.brave.com/res/v1/web/search
```

Recommended defaults:

```env
PROACTIVE_LINK_RECOMMENDATION_ENABLED=true
PROACTIVE_LINK_RECOMMENDATION_PROBABILITY=0.03
PROACTIVE_LINK_CANDIDATE_URLS=
CHAT_LINK_SEARCH_TIMEOUT_S=8
```

After the deploy writes `.env`, run a live smoke test on the server:

```bash
cd /app/companion-server
uv run python scripts/smoke_chat_link_provider.py --query "周末咖啡 分享" --require-live
```

Expected success output includes `"status": "ok"`, at least one supported
platform URL in `results`, and a `first_card` with `platform`, `title`,
`final_url`, and `has_content`.

### iOS remote notifications

Remote notifications use APNs token-based authentication. Store the `.p8`
private key file on the VPS at
`/app/companion-secrets/apns/AuthKey_SG87KSNWZH.p8` and set the GitHub secret
`APNS_AUTH_KEY` to that exact server-side file path. The deploy workflow writes
that path into `APNS_AUTH_KEY_PATH` in the generated `.env`.
`docker-compose.deploy.yml` mounts `/app/companion-secrets` read-only into the
server container, so the same path is readable both on the host and inside the
container.

```env
# GitHub Secrets
APNS_KEY_ID=SG87KSNWZH
APNS_AUTH_KEY=/app/companion-secrets/apns/AuthKey_SG87KSNWZH.p8

# GitHub Variables
APNS_ENABLED=true
APNS_TEAM_ID=F3FB94L862
APNS_TOPIC=com.bansheng.dev
APNS_USE_SANDBOX=false
NOTIFICATION_MAX_ATTEMPTS=3
NOTIFICATION_DISPATCH_BATCH_SIZE=50
```

Use `APNS_USE_SANDBOX=true` only for a development server that sends to Debug
builds installed directly from Xcode/Flutter. TestFlight and App Store builds
use the production APNs environment, so the production deploy should keep
`APNS_USE_SANDBOX=false`.

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
- This deployment uses Redis DB 0 because the Tencent CVM Redis instance is dedicated to Companion.
  - local tunnel: `redis://localhost:6380/0`
  - production container network: `redis://redis:6379/0`
