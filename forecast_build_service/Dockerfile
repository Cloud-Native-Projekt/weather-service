ARG BUILD_SERVICE_IMAGE=weather-build-service:1.0.0
FROM ${BUILD_SERVICE_IMAGE} AS wheels
FROM python:3.13.7-trixie AS builder

WORKDIR /app

# Copy pre-built wheels
COPY --from=wheels /wheels /wheels

RUN pip install --user --no-cache-dir --find-links /wheels weather_models

FROM python:3.13.7-slim-trixie

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
COPY forecast_build_service .

# Create models directory and set ownership
RUN mkdir -p /app/models && \
    chown -R appuser:appuser /app

USER appuser

# Make sure scripts are in PATH
ENV PATH=/home/appuser/.local/bin:$PATH

CMD ["python", "build_models.py"]