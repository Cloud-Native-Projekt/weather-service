FROM weather-build-service:1.0.0 AS wheels
FROM python:3.13.7-trixie AS builder

WORKDIR /app

# Copy pre-built wheels
COPY --from=wheels /wheels /wheels

RUN pip install --user --no-cache-dir /wheels/weather_models-0.3.1-py3-none-any.whl
RUN pip install --user --no-cache-dir /wheels/openmeteo_client-0.1.2-py3-none-any.whl


FROM python:3.13.7-slim-trixie AS weather-maintenance-service

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
 && rm -rf /var/lib/apt/lists/* \
 && apt-get clean

RUN groupadd -r appuser --gid=1000 && useradd -r -g appuser --uid=1000 appuser

WORKDIR /app/

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application files
COPY maintenance_service .

# Copy config files
COPY config/ ./config/

RUN chown -R appuser:appuser /app

USER appuser

# Make sure scripts are in PATH
ENV PATH=/home/appuser/.local/bin:$PATH

CMD ["python", "maintenance.py"]