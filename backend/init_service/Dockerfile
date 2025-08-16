FROM python:3.13.6-bookworm

# Install system packages
RUN apt-get update && apt-get install -y \
    curl \
    git \
 && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app/

# Copy requirements
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy models
COPY models/ ./models/

# Copy packages
COPY backend/openmeteo_client/ ./openmeteo_client/

# Copy application code
COPY backend/init_service .

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Entry point
CMD ["python", "db_init.py"]