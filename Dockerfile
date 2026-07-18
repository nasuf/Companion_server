FROM python:3.13-slim

ARG DEBIAN_MIRROR=""
ARG DEBIAN_SECURITY_MIRROR=""
ARG PIP_INDEX_URL=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Mirror fetches occasionally drop mid-download (SSL "unexpected eof"), so
# let apt retry each file and give the whole install one --fix-missing pass
# before failing the build.
RUN if [ -n "$DEBIAN_MIRROR" ]; then \
        sed -i "s|http://deb.debian.org/debian|$DEBIAN_MIRROR|g" /etc/apt/sources.list.d/debian.sources; \
    fi \
    && if [ -n "$DEBIAN_SECURITY_MIRROR" ]; then \
        sed -i "s|http://deb.debian.org/debian-security|$DEBIAN_SECURITY_MIRROR|g" /etc/apt/sources.list.d/debian.sources; \
    fi \
    && printf 'Acquire::Retries "5";\nAcquire::http::Timeout "30";\nAcquire::https::Timeout "30";\n' \
        > /etc/apt/apt.conf.d/80-retries \
    && apt-get update \
    && { apt-get install -y --no-install-recommends \
            build-essential \
            ca-certificates \
            curl \
            ffmpeg \
        || { sleep 5 && apt-get update && apt-get install -y --no-install-recommends --fix-missing \
            build-essential \
            ca-certificates \
            curl \
            ffmpeg; }; } \
    && rm -rf /var/lib/apt/lists/*

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
