# Configuration

Centralized configuration management for the Weather Service, providing standardized parameter files for development and production environments. This directory contains JSON configuration files that define OpenMeteo API parameters, geographic boundaries, and data collection settings.

## Overview

The configuration system enables consistent parameter management across all Weather Service components while supporting environment-specific customizations. Configuration files define the geographic scope, temporal ranges, and meteorological metrics for weather data collection and processing.

## Configuration Files

### config.json (Production Configuration Example)

Production configuration for comprehensive weather data coverage across Europe and Central Asia:

```json
{
    "history_start_date": "2001-01-01",
    "history_end_date": "latest",
    "forecast_days": 16,
    "forecast_past_days": 1,
    "bounding_box": {
        "north": 71.0,
        "south": 36.0,
        "west": 9.0,
        "east": 68.0
    },
    "metrics": [
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "cloud_cover_mean",
        "cloud_cover_max",
        "cloud_cover_min",
        "wind_speed_10m_mean",
        "wind_speed_10m_min",
        "wind_speed_10m_max",
        "sunshine_duration",
        "precipitation_sum",
        "precipitation_hours"
    ]
}
```

**Coverage Characteristics:**
- **Geographic**: Europe and Central Asia (35° coverage area)
- **Temporal**: 24+ years of historical data
- **Locations**: ~4,900 grid points (0.5° spacing)
- **API Cost**: ~590,000 API units for full historical bootstrap

### dev_config.json (Development Configuration Example)

Lightweight configuration for development and testing with minimal geographic scope:

```json
{
    "history_start_date": "2010-01-01",
    "history_end_date": "latest",
    "forecast_days": 16,
    "forecast_past_days": 1,
    "bounding_box": {
        "north": 37.0,
        "south": 36.0,
        "west": 9.0,
        "east": 10.0
    },
    "metrics": [
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "cloud_cover_mean",
        "cloud_cover_max",
        "cloud_cover_min",
        "wind_speed_10m_mean",
        "wind_speed_10m_min",
        "wind_speed_10m_max",
        "sunshine_duration",
        "precipitation_sum",
        "precipitation_hours"
    ]
}
```

**Development Characteristics:**
- **Geographic**: Small Mediterranean region (1° x 1° area)
- **Temporal**: 15+ years of historical data
- **Locations**: 4 grid points (reduced for testing)
- **API Cost**: ~1,900 API units for development bootstrap

## Configuration Parameters

### Temporal Configuration

| Parameter | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| `history_start_date` | string | Start date for historical data | YYYY-MM-DD format |
| `history_end_date` | string | End date for historical data | "latest" or YYYY-MM-DD |
| `forecast_days` | integer | Number of forecast days | 1-16 (OpenMeteo API limit) |
| `forecast_past_days` | integer | Past days in forecast | 1-5 (OpenMeteo API limit) |

### Geographic Configuration

| Parameter | Type | Description | Valid Range |
|-----------|------|-------------|-------------|
| `bounding_box.north` | float | Northern latitude boundary | -90.0 to 90.0 |
| `bounding_box.south` | float | Southern latitude boundary | -90.0 to 90.0 |
| `bounding_box.east` | float | Eastern longitude boundary | -180.0 to 180.0 |
| `bounding_box.west` | float | Western longitude boundary | -180.0 to 180.0 |

### Weather Metrics

Can be any metric supported by the Open Meteo API:

| Metric | Unit | Description |
|--------|------|-------------|
| `temperature_2m_mean` | °C | Mean temperature at 2m height |
| `temperature_2m_max` | °C | Maximum temperature at 2m height |
| `temperature_2m_min` | °C | Minimum temperature at 2m height |
| `cloud_cover_mean` | % | Mean cloud cover percentage |
| `cloud_cover_max` | % | Maximum cloud cover percentage |
| `cloud_cover_min` | % | Minimum cloud cover percentage |
| `wind_speed_10m_mean` | km/h | Mean wind speed at 10m height |
| `wind_speed_10m_min` | km/h | Minimum wind speed at 10m height |
| `wind_speed_10m_max` | km/h | Maximum wind speed at 10m height |
| `sunshine_duration` | seconds | Sunshine duration |
| `precipitation_sum` | mm | Total precipitation |
| `precipitation_hours` | hours | Precipitation duration |

## Usage Patterns

### Environment Selection

Configuration file selection is controlled via the `CONFIG_FILE` environment variable:

```bash
# Use production configuration
export CONFIG_FILE=config.json

# Use development configuration
export CONFIG_FILE=dev_config.json

# Default behavior (config.json if not specified)
unset CONFIG_FILE
```

### Service Integration

All Weather Service components automatically load configuration:

```python
from openmeteo_client import OpenMeteoClientConfig

# Load from environment-specified config file
config = OpenMeteoClientConfig(create_from_file=True)

# Override specific parameters
config = OpenMeteoClientConfig(
    create_from_file=True,
    kwargs={"forecast_days": 7}
)
```

## Geographic Coverage Analysis

### Production Coverage (config.json)

**Bounding Box**: 71°N to 36°N, 9°E to 68°E
- **Countries**: Most of Europe, Russia, Central Asia, parts of Middle East
- **Grid Points**: ~4,900 locations (0.5° spacing)
- **Area**: ~35° latitude × 59° longitude = ~2,065 degree²

### Development Coverage (dev_config.json)

**Bounding Box**: 37°N to 36°N, 9°E to 10°E
- **Region**: Small area in Mediterranean (Southern Spain/Algeria border)
- **Grid Points**: 4 locations (2×2 grid)
- **Area**: 1° latitude × 1° longitude = 1 degree²

## API Cost Estimation

Cost estimation can be derived from official [Open Meteo API cost calculator](https://open-meteo.com/en/pricing)

## Configuration Validation

### Common Validation Errors

**Invalid Date Range**:
```json
{
    "history_start_date": "2025-01-01",  // Future date
    "history_end_date": "2020-01-01"    // Before start date
}
```

**Invalid Bounding Box**:
```json
{
    "bounding_box": {
        "north": 30.0,
        "south": 40.0,  // South > North
        "west": 10.0,
        "east": 5.0     // East < West
    }
}
```

**API Limit Violations**:
```json
{
    "forecast_days": 30,        // Exceeds API limit (16)
    "forecast_past_days": 10    // Exceeds API limit (5)
}
```

## Parameter Override Examples

```python
# Override forecast horizon
config = OpenMeteoClientConfig(
    create_from_file=True,
    kwargs={"forecast_days": 3}
)

# Override geographic scope
config = OpenMeteoClientConfig(
    create_from_file=True,
    kwargs={
        "bounding_box": {
            "north": 60.0, "south": 50.0,
            "west": 0.0, "east": 10.0
        }
    }
)

# Override metrics subset
config = OpenMeteoClientConfig(
    create_from_file=True,
    kwargs={
        "metrics": ["temperature_2m_mean", "precipitation_sum"]
    }
)
```

## Performance Considerations

### Grid Resolution Impact

| Spacing | Grid Points (10°×10°) | API Cost Multiplier |
|---------|----------------------|-------------------|
| 0.25° | 1,600 | 4x |
| 0.5° | 400 | 1x (baseline) |
| 1.0° | 100 | 0.25x |

### Rate Limiting Impact

## Best Practices

### Environment Management

1. **Use dev_config.json for testing**: Minimize API usage during development
2. **Validate before production**: Test configuration changes with small datasets
3. **Monitor API usage**: Track costs for large geographic areas
4. **Environment isolation**: Keep development and production configurations separate

### Configuration Design

1. **Start small**: Begin with limited geographic scope
2. **Incremental expansion**: Gradually increase coverage area
3. **Cost awareness**: Calculate API costs before large deployments
4. **Metric selection**: Include only required weather parameters

### Troubleshooting

**High API Costs**:
- Reduce bounding box size
- Increase grid spacing (reduce resolution)
- Limit historical date range
- Remove unnecessary metrics

**Rate Limiting Issues**:
- Use development configuration for testing
- Implement progressive rollouts for large areas
- Monitor request time estimates

**Data Quality Issues**:
- Verify bounding box coordinates
- Check date range validity
- Ensure metric names match OpenMeteo parameters

## Configuration Schema

### JSON Schema Definition

```json
{
    "type": "object",
    "required": [
        "history_start_date",
        "history_end_date", 
        "forecast_days",
        "forecast_past_days",
        "bounding_box",
        "metrics"
    ],
    "properties": {
        "history_start_date": {
            "type": "string",
            "pattern": "^\\d{4}-\\d{2}-\\d{2}$"
        },
        "history_end_date": {
            "oneOf": [
                {"type": "string", "enum": ["latest"]},
                {"type": "string", "pattern": "^\\d{4}-\\d{2}-\\d{2}$"}
            ]
        },
        "forecast_days": {
            "type": "integer",
            "minimum": 1,
            "maximum": 16
        },
        "forecast_past_days": {
            "type": "integer", 
            "minimum": 1,
            "maximum": 5
        },
        "bounding_box": {
            "type": "object",
            "required": ["north", "south", "east", "west"],
            "properties": {
                "north": {"type": "number", "minimum": -90, "maximum": 90},
                "south": {"type": "number", "minimum": -90, "maximum": 90},
                "east": {"type": "number", "minimum": -180, "maximum": 180},
                "west": {"type": "number", "minimum": -180, "maximum": 180}
            }
        },
        "metrics": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1
        }
    }
}
```

This configuration system provides flexible, validated parameter management for the entire Weather Service ecosystem while supporting both development and production deployment scenarios.