FROM python:3.13.7-trixie AS builder

RUN pip install --user --no-cache-dir poetry==2.1

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy dependency files
COPY weather_models/pyproject.toml weather_models/poetry.lock ./weather_models/
COPY openmeteo_client/pyproject.toml openmeteo_client/poetry.lock ./openmeteo_client/
COPY maintenance_service/pyproject.toml maintenance_service/poetry.lock ./maintenance_service/

# Copy source code
COPY weather_models/ ./weather_models/
COPY openmeteo_client/ ./openmeteo_client/
COPY maintenance_service/ ./maintenance_service/

# Change to the service directory for poetry operations
WORKDIR /app/maintenance_service

RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --only=main --no-root

FROM python:3.13.7-slim-trixie

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
 && rm -rf /var/lib/apt/lists/* \
 && apt-get clean

RUN groupadd -r appuser --gid=1000 && useradd -r -g appuser --uid=1000 appuser

WORKDIR /app/

ENV VIRTUAL_ENV=/app/maintenance_service/.venv \
    PATH="/app/maintenance_service/.venv/bin:$PATH" \
    PYTHONPATH="/app"

COPY config/ ./config/

# Copy installed packages from builder
COPY --from=builder --chown=appuser:appuser /app/maintenance_service/.venv ./maintenance_service/.venv

# Copy application files
COPY --chown=appuser:appuser maintenance_service/ ./maintenance_service/
COPY --chown=appuser:appuser weather_models/ ./weather_models/
COPY --chown=appuser:appuser openmeteo_client/ ./openmeteo_client/

# Create cache directory and set ownership
RUN mkdir -p /app/.cache && chown -R appuser:appuser /app/.cache

USER appuser

ENTRYPOINT ["python", "-m", "maintenance_service.maintenance"]