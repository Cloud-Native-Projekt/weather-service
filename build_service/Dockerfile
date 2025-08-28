FROM python:3.13.7-trixie AS wheel-builder


RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
 && rm -rf /var/lib/apt/lists/* \
 && apt-get clean

RUN pip install --no-cache-dir poetry==2.1 build

WORKDIR /app

# Copy source packages
COPY weather_models/ ./weather_models/
COPY openmeteo_client/ ./openmeteo_client/

RUN mkdir -p /wheels

# Build wheels
WORKDIR /app/weather_models
RUN poetry build
RUN mv dist/*.whl /wheels/

WORKDIR /app/openmeteo_client
RUN poetry build
RUN mv dist/*.whl /wheels/

FROM scratch AS weather-build-service
ARG VERSION=latest
LABEL version=1.0.0
COPY --from=wheel-builder /wheels /wheels