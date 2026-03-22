# Companion Server Deployment

This repo deploys to RackNerd through GitHub Actions.

## What the workflow deploys

- `companion-server` bound to `127.0.0.1:8000`
- `neo4j:5-community` via Docker Compose
- External Redis via `REDIS_URL`
- External Postgres / Supabase via `DATABASE_URL`
- Prompt template migration applied during deploy

Server path on the VPS:

- `/app/companion-server`

## GitHub configuration

Use both repository `Secrets` and repository `Variables`.

### VPS access

- `VPS_PASSWORD`

### Repository Secrets

- `VPS_HOST`
- `DATABASE_URL`
- `REDIS_URL`
- `NEO4J_PASSWORD`
- `DASHSCOPE_API_KEY`
- `LANGSMITH_API_KEY`
- `LANGSMITH_ORG_ID`
- `LANGSMITH_PROJECT_ID`
- `ADMIN_USERNAME`
- `ADMIN_PASSWORD`

Optional:

- `ANTHROPIC_API_KEY`

### Repository Variables

- `VPS_PORT`
- `VPS_USERNAME`
- `ONLINE_MODEL`
- `OLLAMA_BASE_URL`
- `DASHSCOPE_BASE_URL`
- `DASHSCOPE_ENABLE_THINKING`
- `LOCAL_CHAT_MODEL`
- `LOCAL_SMALL_MODEL`
- `REMOTE_CHAT_MODEL`
- `REMOTE_SMALL_MODEL`
- `LANGSMITH_TRACING`

### Database and cache

## Recommended current values

### Database

```env
DATABASE_URL=postgresql://postgres.lzmeacugbmdzqhasitog:Hellocompanion713%21@aws-1-ap-south-1.pooler.supabase.com:5432/postgres?sslmode=require
```

### Redis

```env
REDIS_URL=redis://:nasuf713%21@192.210.235.115:6380/4
```

### Neo4j

Choose your own password:

```env
NEO4J_PASSWORD=change_this_to_a_strong_password
```

### Model switch

For Alibaba Cloud Bailian / DashScope:

```env
VPS_PORT=22
VPS_USERNAME=root
ONLINE_MODEL=true
OLLAMA_BASE_URL=http://127.0.0.1:11434
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_ENABLE_THINKING=false
LOCAL_CHAT_MODEL=qwen2.5:14b
LOCAL_SMALL_MODEL=qwen2.5:7b
REMOTE_CHAT_MODEL=qwen3.5-plus
REMOTE_SMALL_MODEL=qwen3.5-flash
LANGSMITH_TRACING=true
```

### Admin prompt console

```env
ADMIN_USERNAME=your_admin_username
ADMIN_PASSWORD=your_admin_password
```

For local Ollama:

```env
ONLINE_MODEL=false
OLLAMA_BASE_URL=http://127.0.0.1:11434
DASHSCOPE_API_KEY=
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_ENABLE_THINKING=false
LOCAL_CHAT_MODEL=qwen2.5:14b
LOCAL_SMALL_MODEL=qwen2.5:7b
REMOTE_CHAT_MODEL=qwen3.5-plus
REMOTE_SMALL_MODEL=qwen3.5-flash
```

## Notes

- The memory system still uses embeddings internally, but you do not need to configure an embedding model separately anymore.
- `ONLINE_MODEL=true` means chat / summarizer / utility / embedding all use DashScope defaults.
- `ONLINE_MODEL=false` means the same roles all use local Ollama defaults.
- The backend API is not exposed directly to the public internet in this deploy shape; Nginx on the web repo proxies requests to `127.0.0.1:8000`.
- The prompt admin API is protected by HTTP Basic auth using `ADMIN_USERNAME` and `ADMIN_PASSWORD`.
- Keep local and deployed environments on different Redis DBs. Recommended:
  - local: `redis://localhost:6380/0`
  - dev server: `redis://:***@host:6380/4`
