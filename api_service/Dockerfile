ARG BUILD_SERVICE_IMAGE=ghcr.io/cloud-native-projekt/weather-service/build_service:latest
FROM ${BUILD_SERVICE_IMAGE} AS wheels
FROM python:3.13.7-trixie AS builder

WORKDIR /app

# Copy pre-built wheels
COPY --from=wheels /wheels /wheels

RUN pip install --user --no-cache-dir --find-links /wheels weather_models
RUN pip install --user --no-cache-dir --find-links /wheels frontend_api

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

COPY api_service .

# Set ownership
RUN chown -R appuser:appuser /app

# Make sure scripts are in PATH
ENV PATH=/home/appuser/.local/bin:$PATH

USER appuser

# Expose the port that FastAPI will run on
EXPOSE 8000

# CMD ["tail", "-f", "/dev/null"]

# Run the FastAPI application with uvicorn
CMD ["python", "-m", "uvicorn", "frontend_api:app", "--host", "0.0.0.0", "--port", "8000"]