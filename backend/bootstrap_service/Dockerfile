FROM python:3.13.7-trixie AS builder

WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.13.7-slim-trixie

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
 && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appuser --gid=1000 && useradd -r -g appuser --uid=1000 appuser

WORKDIR /app/

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application files
COPY models/ ./models/
COPY backend/openmeteo_client/ ./openmeteo_client/
COPY backend/bootstrap_service .

RUN chown -R appuser:appuser /app

USER appuser

# Make sure scripts are in PATH
ENV PATH=/home/appuser/.local/bin:$PATH

CMD ["python", "init.py"]