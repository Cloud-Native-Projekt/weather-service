# OpenMeteo Client

A comprehensive Python client library for interacting with OpenMeteo APIs to retrieve weather data including historical observations and forecasts. Features rate limiting, data processing, and aggregation capabilities specifically designed for weather service infrastructure.

## Overview

The `openmeteo_client` package provides a robust interface for OpenMeteo API interactions, offering:

- **API Clients**: Specialized clients for Archive and Forecast endpoints
- **Configuration Management**: Flexible configuration from JSON files and kwargs
- **Rate Limiting**: Multi-tier rate limiting with automatic backoff
- **Data Processing**: Daily to weekly aggregation with meteorological best practices
- **Geographic Processing**: Coordinate grid generation and multi-location optimization

## Features

### API Client Architecture

- **Abstract Base Class**: Common functionality across API clients
- **Rate Limiting**: Multi-window limits (minutely/hourly/daily)
- **Automatic Retry**: Progressive backoff with exponential delays
- **Request Caching**: 24-hour HTTP cache for performance optimization

### Configuration System

- **JSON Configuration**: Centralized parameter management
- **Runtime Overrides**: Direct kwargs parameter passing
- **Parameter Validation**: Type checking and API constraint validation
- **Geographic Grid**: Automatic coordinate generation from bounding boxes

### Data Aggregation

- **Weekly Summaries**: Daily-to-weekly conversion using meteorological statistics
- **Temporal Boundaries**: Complete week detection with partial data separation
- **Statistical Methods**: Variable-specific aggregation strategies
- **ISO Week Numbering**: Standardized temporal reference system

## Installation

Install from the built wheel:

```bash
pip install openmeteo-client-x.y.z-py3-none-any.whl
```

## Quick Start

### Configuration Setup

```python
from openmeteo_client import OpenMeteoClientConfig

# From JSON configuration file
config = OpenMeteoClientConfig(create_from_file=True)

# From JSON with parameter overrides
config = OpenMeteoClientConfig(
    create_from_file=True,
    kwargs={"forecast_days": 10}
)

# From direct parameters
config = OpenMeteoClientConfig(
    create_from_file=False,
    kwargs={
        "history_start_date": "2024-01-01",
        "history_end_date": "latest",
        "bounding_box": {
            "north": 50.0, "south": 40.0, 
            "west": -10.0, "east": 5.0
        },
        "metrics": ["temperature_2m_mean", "precipitation_sum"]
    }
)
```

### Historical Data Retrieval

```python
from openmeteo_client import OpenMeteoArchiveClient

# Initialize client
config = OpenMeteoClientConfig(
    create_from_file=True,
    kwargs={
        "history_start_date": "2020-01-01",
        "history_end_date": "2023-12-31"
    }
)
archive_client = OpenMeteoArchiveClient(config)

# Retrieve historical data
historical_data = archive_client.main()
print(f"Retrieved {len(historical_data)} historical records")
```

### Forecast Data Retrieval

```python
from openmeteo_client import OpenMeteoForecastClient

# Initialize client
config = OpenMeteoClientConfig(
    create_from_file=True,
    kwargs={
        "forecast_days": 7,
        "forecast_past_days": 1
    }
)
forecast_client = OpenMeteoForecastClient(config)

# Retrieve forecast data
forecast_data = forecast_client.main()
print(f"Retrieved {len(forecast_data)} forecast records")
```

### Weekly Data Aggregation

```python
from openmeteo_client import WeeklyTableConstructor

# Aggregate daily data to weekly
constructor = WeeklyTableConstructor()
weekly_data, head_partial, tail_partial = constructor.main(daily_data)

print(f"Complete weeks: {len(weekly_data)}")
print(f"Partial head data: {len(head_partial)}")
print(f"Partial tail data: {len(tail_partial)}")
```

## API Reference

### OpenMeteoClientConfig

#### Configuration Sources

**JSON Configuration File**:
```json
{
    "history_start_date": "YYYY-MM-DD",
    "history_end_date": "latest",
    "forecast_days": 7,
    "forecast_past_days": 1,
    "bounding_box": {
        "north": 50.0,
        "south": 40.0,
        "west": -10.0,
        "east": 5.0
    },
    "metrics": ["temperature_2m_mean", "precipitation_sum"]
}
```

#### Initialization

- `create_from_file` (bool): Load base configuration from JSON file
- `config_file` (str): Path to configuration file (optional)
- `kwargs` (Dict): Parameter overrides or direct configuration

#### Configuration Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `history_start_date` | date/str | Start date for historical data (YYYY-MM-DD) |
| `history_end_date` | date/str | End date for historical data ("latest" or YYYY-MM-DD) |
| `forecast_days` | int | Number of forecast days (1-16) |
| `forecast_past_days` | int | Number of past days in forecast (1-5) |
| `bounding_box` | Dict | Geographic boundaries (north, south, east, west) |
| `metrics` | List[str] | OpenMeteo daily metrics to retrieve |
| `locations` | NDArray | Auto-generated coordinate grid |

### OpenMeteoArchiveClient

#### API Characteristics

- **Endpoint**: `https://archive-api.open-meteo.com/v1/archive`
- **Cost**: 31.3 API units per location-year
- **Data Delay**: 2+ days due to API constraints
- **Coverage**: Global historical observations

#### Methods

- `main()`: Primary interface for historical data retrieval
- `get_data(url)`: Execute year-chunked API requests with rate limiting
- `get_request_time_estimate(num_requests)`: Calculate estimated completion time

#### Request Strategy

- **Year-based Chunking**: Optimizes API costs for large date ranges
- **Geographic Iteration**: Processes all coordinate grid points
- **Rate Limiting**: Progressive limits across minutely/hourly/daily windows

### OpenMeteoForecastClient

#### API Characteristics

- **Endpoint**: `https://api.open-meteo.com/v1/forecast`
- **Cost**: 1.2 API units per location
- **Update Frequency**: Multiple times daily
- **Coverage**: Global numerical weather predictions

#### Methods

- `main()`: Primary interface for forecast data retrieval
- `get_data(url)`: Execute location-based API requests
- Automatic date range calculation from configuration

#### Forecast Configuration

- **Forecast Horizon**: 1-16 days ahead
- **Past Days**: 1-5 days for continuity
- **Date Range**: Auto-computed from current date

### WeeklyTableConstructor

#### Aggregation Strategy

**Meteorological Statistics**:
- **Temperature**: Mean, max, min preservation
- **Cloud Cover**: Statistical aggregation
- **Wind Speed**: Mean, min, max variability
- **Precipitation**: Sum totals and duration
- **Sunshine**: Mean duration averages

#### Methods

- `main(daily_data)`: Primary aggregation interface
- Returns: `(weekly_data, head_partial, tail_partial)`

#### Week Boundary Logic

- **Complete Weeks**: Monday to Sunday alignment
- **Partial Data**: Separated head and tail segments
- **ISO Numbering**: Standardized week identification (1-53)

## Rate Limiting System

### Multi-Tier Limits

| Window | Limit | Backoff |
|--------|-------|---------|
| Minutely | 600 requests | 61 seconds |
| Hourly | 5,000 requests | 1 hour |
| Daily | 10,000 requests | 24 hours |

### Implementation Features

- **Automatic Backoff**: Progressive delays when limits approached
- **Usage Tracking**: Real-time monitoring across all time windows
- **Cost Estimation**: Request time prediction for batch operations
- **Graceful Handling**: Transparent blocking during backoff periods

## Geographic Processing

### Coordinate Grid Generation

```python
# Bounding box specification
bounding_box = {
    "north": 50.0,    # Northern latitude boundary
    "south": 40.0,    # Southern latitude boundary
    "west": -10.0,    # Western longitude boundary  
    "east": 5.0       # Eastern longitude boundary
}

# Grid generation (default 0.5° spacing)
locations = config.locations  # Auto-generated coordinate array
```

### Grid Characteristics

- **Default Resolution**: 0.5° spacing (~55km at equator)
- **Output Format**: 2D NumPy array [latitude, longitude]
- **Coverage**: All points within bounding box
- **Optimization**: Regular grid for efficient API usage

## Supported Weather Metrics

### OpenMeteo Daily Parameters

| Metric | Unit | Description |
|--------|------|-------------|
| `temperature_2m_mean` | °C | Mean temperature at 2m height |
| `temperature_2m_max` | °C | Maximum temperature at 2m height |
| `temperature_2m_min` | °C | Minimum temperature at 2m height |
| `cloud_cover_mean` | % | Mean cloud cover percentage |
| `cloud_cover_max` | % | Maximum cloud cover percentage |
| `cloud_cover_min` | % | Minimum cloud cover percentage |
| `wind_speed_10m_mean` | km/h | Mean wind speed at 10m height |
| `wind_speed_10m_max` | km/h | Maximum wind speed at 10m height |
| `wind_speed_10m_min` | km/h | Minimum wind speed at 10m height |
| `sunshine_duration` | seconds | Sunshine duration |
| `precipitation_sum` | mm | Total precipitation |
| `precipitation_hours` | hours | Precipitation duration |

## Configuration Examples

### Historical Data Configuration

```json
{
    "history_start_date": "2020-01-01",
    "history_end_date": "latest",
    "bounding_box": {
        "north": 55.0,
        "south": 45.0,
        "west": -15.0,
        "east": 10.0
    },
    "metrics": [
        "temperature_2m_mean",
        "temperature_2m_max", 
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_mean"
    ]
}
```

### Forecast Configuration

```json
{
    "forecast_days": 14,
    "forecast_past_days": 2,
    "bounding_box": {
        "north": 50.0,
        "south": 40.0,
        "west": -10.0,
        "east": 5.0
    },
    "metrics": [
        "temperature_2m_mean",
        "cloud_cover_mean",
        "precipitation_sum",
        "sunshine_duration"
    ]
}
```

## Dependencies

- **openmeteo_requests**: Official OpenMeteo SDK
- **openmeteo_sdk**: Response parsing utilities
- **pandas**: Data manipulation and temporal operations
- **numpy**: Numerical computations
- **retry-requests**: Automatic retry logic
- **requests-cache**: HTTP caching optimization

## Integration

This package integrates with other weather service components:

- **[`weather_models`](../weather_models)**: ORM models and database operations
- **[`bootstrap_service`](../bootstrap_service)**: Initial data population
- **[`daily_maintenance_service`](../daily_maintenance_service)**: Daily updates
- **[`weekly_maintenance_service`](../weekly_maintenance_service)**: Weekly aggregation
- **[`forecast_service`](../forecast_service)**: Real-time forecast generation

## Error Handling

### Comprehensive Error Management

- **Parameter Validation**: Type checking and constraint verification
- **API Response Verification**: Response structure validation
- **Network Resilience**: Automatic retry with exponential backoff
- **Graceful Degradation**: Partial failure handling with detailed logging

### Common Error Scenarios

- **Invalid Configuration**: Parameter validation with descriptive messages
- **API Quota Exceeded**: Automatic backoff with progress logging
- **Network Issues**: Retry logic with exponential backoff
- **Malformed Responses**: Response structure validation and error reporting

## Performance Characteristics

### Optimization Features

- **Request Caching**: 24-hour HTTP cache for repeated requests
- **Geographic Batching**: Efficient multi-location processing
- **Year-based Chunking**: Optimized API cost for historical data

### Scalability Considerations

- **Rate Limiting**: Prevents API quota violations
- **Geographic Partitioning**: Supports parallel processing
- **Configurable Resolution**: Performance tuning via grid spacing

## Logging

### Operation Monitoring

- **Request Progress**: Real-time logging of API operations
- **Rate Limiting**: Backoff period notifications
- **Data Processing**: Aggregation and conversion progress
- **Error Conditions**: Detailed exception information