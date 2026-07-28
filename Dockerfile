FROM python:3.13-slim

ARG DEBIAN_MIRROR=""
ARG DEBIAN_SECURITY_MIRROR=""
ARG DEBIAN_FALLBACK_MIRROR=""
ARG DEBIAN_SECURITY_FALLBACK_MIRROR=""
ARG PIP_INDEX_URL=""

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

COPY pyproject.toml ./
COPY app ./app
COPY jobs ./jobs
COPY prisma ./prisma
COPY scripts ./scripts

RUN if [ -n "$PIP_INDEX_URL" ]; then \
        pip install --upgrade pip -i "$PIP_INDEX_URL" \
        && pip install . -i "$PIP_INDEX_URL"; \
    else \
        pip install --upgrade pip \
        && pip install .; \
    fi \
    && prisma generate --schema prisma/schema.prisma

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
