FROM python:3.13-slim

ARG DEBIAN_MIRROR=""
ARG DEBIAN_SECURITY_MIRROR=""
ARG DEBIAN_FALLBACK_MIRROR=""
ARG DEBIAN_SECURITY_FALLBACK_MIRROR=""
ARG PIP_INDEX_URL=""
ARG NPM_REGISTRY=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# The configured mirror sometimes serves broken files (persistent SSL
# "unexpected eof" on specific .debs), so retry per file, and if the install
# still fails, switch to the fallback mirror (deb.debian.org when unset —
# too slow from a China VPS, so deploys pass a domestic fallback).
RUN set -eu; \
    cp /etc/apt/sources.list.d/debian.sources /tmp/debian.sources.orig; \
    if [ -n "$DEBIAN_MIRROR" ]; then \
        sed -i "s|http://deb.debian.org/debian|$DEBIAN_MIRROR|g" /etc/apt/sources.list.d/debian.sources; \
    fi; \
    if [ -n "$DEBIAN_SECURITY_MIRROR" ]; then \
        sed -i "s|http://deb.debian.org/debian-security|$DEBIAN_SECURITY_MIRROR|g" /etc/apt/sources.list.d/debian.sources; \
    fi; \
    printf 'Acquire::Retries "5";\nAcquire::http::Timeout "30";\nAcquire::https::Timeout "30";\n' \
        > /etc/apt/apt.conf.d/80-retries; \
    install_deps() { \
        apt-get update && apt-get install -y --no-install-recommends \
            build-essential \
            ca-certificates \
            curl \
            ffmpeg \
            postgresql-client; \
    }; \
    if ! install_deps; then \
        echo "primary mirror failed; switching to fallback sources" >&2; \
        cp /tmp/debian.sources.orig /etc/apt/sources.list.d/debian.sources; \
        if [ -n "$DEBIAN_FALLBACK_MIRROR" ]; then \
            sed -i "s|http://deb.debian.org/debian|$DEBIAN_FALLBACK_MIRROR|g" /etc/apt/sources.list.d/debian.sources; \
        fi; \
        if [ -n "$DEBIAN_SECURITY_FALLBACK_MIRROR" ]; then \
            sed -i "s|http://deb.debian.org/debian-security|$DEBIAN_SECURITY_FALLBACK_MIRROR|g" /etc/apt/sources.list.d/debian.sources; \
        fi; \
        install_deps; \
    fi; \
    rm -rf /var/lib/apt/lists/* /tmp/debian.sources.orig

COPY pyproject.toml runtime-constraints.txt ./
COPY app ./app
COPY jobs ./jobs
COPY prisma ./prisma
COPY scripts ./scripts
# Prompt version snapshots use these deterministic validators at runtime.
# Keep generated results and optional evaluation datasets out of the image.
COPY evals/__init__.py evals/graders.py evals/run_local.py evals/long_companion_sim.py evals/cases.jsonl ./evals/

RUN if [ -n "$PIP_INDEX_URL" ]; then \
        pip install --upgrade pip -i "$PIP_INDEX_URL" \
        && pip install -c runtime-constraints.txt . -i "$PIP_INDEX_URL"; \
    else \
        pip install --upgrade pip \
        && pip install -c runtime-constraints.txt .; \
    fi \
    && npm_config_registry="${NPM_REGISTRY:-https://registry.npmjs.org}" \
       npm_config_fetch_timeout=120000 npm_config_fetch_retries=2 \
       prisma generate --schema prisma/schema.prisma

EXPOSE 8000

# --workers 取自 WEB_CONCURRENCY，与 settings.web_concurrency 保持同值。
# app.api_server 只补充父进程的 worker 失败诊断，生命周期仍委托给固定版本
# Uvicorn；init:true 已在 compose 里配好，信号转发不受影响。
#
# 多 worker 的前提在部署前逐项扫过 (2026-07-29): 29 个 cron 全部带分布式锁、
# WS 走 Redis pub/sub 跨进程投递、记忆管线 fail-closed 锁、启动 seeding 加锁串行
# 化。详见 CLAUDE.md §11。
# Cold Prisma/Pydantic imports may hold the GIL beyond Uvicorn's 5s default.
# Keep two workers; allow bounded startup/heartbeat time under deployment load.
CMD ["sh", "-c", "exec python -m app.api_server app.main:app --host 0.0.0.0 --port 8000"]
