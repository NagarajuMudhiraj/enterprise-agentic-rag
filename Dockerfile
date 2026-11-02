# syntax=docker/dockerfile:1

# ---------------------------------------------------------------- base
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# ---------------------------------------------------------------- deps
# Kept in its own stage so the (slow) wheel install is cached independently of
# source changes.
FROM base AS deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------------------- runtime
FROM base AS runtime
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

COPY src ./src
COPY main.py ./

# Production defaults: JSON logs, auth required, index in a mounted volume.
ENV APP_ENV=production \
    LOG_JSON=true \
    STORAGE_DIR=/data
RUN mkdir -p /data && chown -R appuser:appuser /app /data
VOLUME ["/data"]

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

# One worker: the vector/BM25/graph indexes and the state lock are per-process.
# Scale out only after moving them to a shared store (see ALLOW_MULTI_WORKER).
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--no-server-header"]
