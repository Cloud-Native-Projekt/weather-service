# Weather Service Frontend API

A RESTful FastAPI application that provides access to weather data stored in a PostgreSQL database. This service acts as the frontend API layer for retrieving historical and forecast weather data at daily and weekly aggregation levels.

## Features

- **Health Monitoring**: Health check endpoint for monitoring database connectivity
- **Location Discovery**: Find available weather data points across geographic locations
- **Time Range Queries**: Determine data availability ranges for different time periods
- **Metrics Discovery**: Explore available weather measurements and parameters
- **Daily Weather Data**: Retrieve daily weather observations and forecasts
- **Weekly Weather Data**: Access weekly aggregated weather data
- **Smart Data Routing**: Automatic routing between historical and forecast data based on cutoff dates
- **Location Matching**: Nearest location matching using Euclidean distance calculation

## API Endpoints

### Health Check
- `GET /health` - Service and database health status

### Data Discovery
- `GET /locations/{table}` - Available geographic locations for a table
- `GET /timespan/{table}` - Available date/time ranges for a table  
- `GET /metrics/{table}` - Available weather metrics for a table

### Weather Data Retrieval
- `GET /daily/{datum}/{latitude}/{longitude}/{metrics}` - Daily weather data
- `GET /weekly/{calendar_week}/{latitude}/{longitude}/{metrics}` - Weekly weather data

## Supported Tables

- **daily_history**: Historical daily weather observations
- **daily_forecast**: Future daily weather predictions
- **weekly_history**: Historical weekly weather aggregations
- **weekly_forecast**: Future weekly weather aggregations and predictions

## Data Formats

### Date Parameters
- **Single date**: `2023-01-01`
- **Multiple dates**: `2023-01-01,2023-01-02,2023-01-03`
- **Date range**: `2023-01-01:2023-01-31`

### Calendar Week Parameters
- **Single week**: `2023-15`
- **Multiple weeks**: `2023-15,2023-16,2023-17`
- **Week range**: `2023-15:2023-20`

### Metrics Parameter
- **All metrics**: `all`
- **Specific metrics**: `temperature_2m_mean,precipitation_sum,wind_speed_10m_mean`

## Installation

### Prerequisites
- Python 3.13+
- PostgreSQL database with weather data
- Weather models package (custom dependency)

## Data Flow

The API automatically determines whether to query historical or forecast tables based on configurable cutoff dates:

- **HISTORY_CUTOFF**: `today - 2 days` (for daily data)
- **WEEKLY_HISTORY_CUTOFF**: Corresponding week for the daily cutoff

For requests spanning multiple time periods, the service combines results from both historical and forecast data sources.

## API Docs

FastAPI automatically creates Swagger and Redoc documentation. For more detailed information run the API service and navigate to the appropriate documentation endpoints.