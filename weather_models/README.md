# Weather Models

A comprehensive Python package providing SQLAlchemy ORM models, database management, and machine learning capabilities for weather data storage and forecasting in the weather service.

## Overview

The `weather_models` package is the core data layer of the weather service, providing:

- **Database Models**: SQLAlchemy ORM models for weather data storage
- **Database Management**: High-level interface for all database operations
- **Machine Learning**: Vector Autoregression (VAR) models for weekly weather forecasting
- **Data Processing**: Utilities for DataFrame to ORM conversion and time series analysis

## Features

### Database Models

- **Standardized Schema**: Common weather measurements across all tables
- **Geographic Indexing**: Efficient spatial queries via latitude/longitude
- **Temporal Indexing**: Optimized date and time-based operations
- **Data Integrity**: Unique constraints and validation

### Database Management

- **Bootstrap Detection**: Prevents accidental data loss during initialization
- **Health Checks**: Validates data completeness and identifies gaps
- **Session Management**: Proper connection handling and cleanup
- **Rollover Operations**: Automated forecast-to-history transitions

### Machine Learning

- **Location-Specific Models**: Tailored forecasts for each geographic coordinate
- **Automatic Optimization**: BIC-based lag order selection
- **Stationarity Handling**: First-order differencing and undifferencing
- **Model Persistence**: Save and load trained models using joblib

## Installation

Install from the built wheel:

```bash
pip install weather_models-x.y.z-py3-none-any.whl
```

## Quick Start

### Database Operations

```python
from weather_models import WeatherDatabase, DailyWeatherHistory

# Initialize database connection
db = WeatherDatabase()

try:
    # Check if bootstrap is needed
    if db.bootstrap:
        db.create_tables()
        # Populate initial data via bootstrap service
    
    # Create ORM objects from DataFrame
    objects = db.create_orm_objects(weather_df, DailyWeatherHistory)
    db.write_data(objects)
    
    # Query data
    history = db.get_table(DailyWeatherHistory)
    location_data = db.get_data_by_location(
        WeeklyWeatherHistory, 
        location=(40.7, -74.0)
    )

finally:
    db.close()
```

### Machine Learning Model

```python
from weather_models import WeeklyForecastModel

# Create and train model
model = WeeklyForecastModel(location=(40.7128, -74.0060))
model.build_model(historical_data)

# Generate forecasts
forecast = model.forecast(horizon=4, data=historical_data)

# Save model
model.save('./models')

# Load model
loaded_model = WeeklyForecastModel.from_file(
    './models', 
    location=(40.7128, -74.0060)
)
```

## Database Schema

### Common Weather Measurements (WeatherBase)

All weather tables inherit these standardized measurements:

| Column | Type | Description |
|--------|------|-------------|
| `latitude` | Float | Geographic latitude coordinate |
| `longitude` | Float | Geographic longitude coordinate |
| `temperature_2m_mean` | Float | Mean temperature (°C) at 2m height |
| `temperature_2m_max` | Float | Maximum temperature (°C) at 2m height |
| `temperature_2m_min` | Float | Minimum temperature (°C) at 2m height |
| `cloud_cover_mean` | Float | Mean cloud cover percentage |
| `cloud_cover_max` | Float | Maximum cloud cover percentage |
| `cloud_cover_min` | Float | Minimum cloud cover percentage |
| `wind_speed_10m_mean` | Float | Mean wind speed (km/h) at 10m height |
| `wind_speed_10m_max` | Float | Maximum wind speed (km/h) at 10m height |
| `wind_speed_10m_min` | Float | Minimum wind speed (km/h) at 10m height |
| `sunshine_duration` | Float | Sunshine duration (seconds) |
| `precipitation_sum` | Float | Total precipitation (mm) |
| `precipitation_hours` | Float | Precipitation duration (hours) |

### Data Tables

#### DailyWeatherHistory
Historical daily weather observations with date-based temporal indexing.

#### DailyWeatherForecast
Daily weather forecast predictions with date-based validity periods.

#### WeeklyWeatherHistory
Aggregated weekly historical data using year/week temporal identification.

#### WeeklyWeatherForecast
Weekly forecast predictions with source tracking (`OpenMeteo` or `WeeklyForecastModel`).

## Configuration

### Environment Variables

The package requires PostgreSQL connection parameters:

```bash
POSTGRES_USER=myuser
POSTGRES_PASSWORD=mypassword
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=weatherdb
```

### Database Engine

The [`DatabaseEngine`](weather_models/weather_models.py) class automatically constructs connection URLs using:
- **Dialect**: `postgresql`
- **Driver**: `psycopg2`
- **Connection**: Environment variable-based configuration

## API Reference

### WeatherDatabase

#### Core Methods

- [`create_tables()`](weather_models/weather_models.py): Create all weather data tables
- [`create_orm_objects(data, table)`](weather_models/weather_models.py): Convert DataFrames to ORM objects
- [`write_data(orm_objects)`](weather_models/weather_models.py): Persist ORM objects to database
- [`get_table(table)`](weather_models/weather_models.py): Retrieve all records from a table
- [`truncate_table(table)`](weather_models/weather_models.py): Remove all records from a table

#### Query Methods

- [`get_data_by_location(table, location)`](weather_models/weather_models.py): Get records for specific coordinates
- [`get_data_by_date_range(table, start_date, end_date)`](weather_models/weather_models.py): Get records within date range
- [`get_locations(table)`](weather_models/weather_models.py): Get all unique coordinate pairs
- [`get_date_range(table)`](weather_models/weather_models.py): Get all unique dates in table

#### Maintenance Methods

- [`health_check(start_date, end_date, table)`](weather_models/weather_models.py): Validate data completeness
- [`get_missing_dates(start_date, end_date, table)`](weather_models/weather_models.py): Identify data gaps
- [`rollover_weekly_data(year, week)`](weather_models/weather_models.py): Transfer forecast to history

#### Utility Methods

- [`to_dataframe(data)`](weather_models/weather_models.py): Convert ORM objects to DataFrame
- [`close()`](weather_models/weather_models.py): Close database session

#### Properties

- [`bootstrap`](weather_models/weather_models.py): Boolean indicating if database initialization is needed

### WeeklyForecastModel

#### Initialization

- [`__init__(location)`](weather_models/weather_models.py): Initialize model for specific coordinates

#### Model Operations

- [`build_model(data)`](weather_models/weather_models.py): Train VAR model on historical data
- [`forecast(horizon, data)`](weather_models/weather_models.py): Generate future predictions
- [`save(directory, file_name)`](weather_models/weather_models.py): Persist trained model
- [`from_file(directory, location)`](weather_models/weather_models.py): Load saved model

## Machine Learning Architecture

### Vector Autoregression (VAR) Model

The [`WeeklyForecastModel`](weather_models/weather_models.py) implements a sophisticated time series forecasting approach:

1. **Data Preprocessing**:
   - Converts year/week to datetime index
   - Removes non-meteorological columns
   - Applies first-order differencing for stationarity

2. **Model Training**:
   - Computes maximum feasible lag order
   - Uses Bayesian Information Criterion (BIC) for optimization
   - Fits VAR model with ordinary least squares estimation

3. **Forecasting**:
   - Generates differenced predictions
   - Applies cumulative sum for undifferencing
   - Adds back last observed values for real-scale forecasts
   - Creates proper datetime index and metadata

### Model Constraints

- **Minimum Lags**: 1
- **Maximum Lags**: 156 or data-constrained
- **Data Requirements**: Minimum 20 observations for parameter estimation
- **Stationarity**: Achieved through first-order differencing

## Dependencies

- **SQLAlchemy**: ORM and database abstraction
- **pandas**: Data manipulation and DataFrame operations
- **psycopg2**: PostgreSQL database adapter
- **statsmodels**: VAR model implementation
- **joblib**: Model serialization and persistence

## Integration

This package integrates with other weather service components:

- **[`bootstrap_service`](bootstrap_service)**: Initial data population
- **[`daily_maintenance_service`](daily_maintenance_service)**: Daily data updates
- **[`weekly_maintenance_service`](weekly_maintenance_service)**: Weekly data aggregation
- **[`forecast_build_service`](forecast_build_service)**: Model training
- **[`forecast_service`](forecast_service)**: Forecast generation
- **[`openmeteo_client`](openmeteo_client)**: Weather data retrieval

## Error Handling

The package includes comprehensive error handling:

- **Database Errors**: Transaction rollback and session cleanup
- **Data Validation**: Type checking and constraint validation
- **Model Errors**: Training failure detection and logging
- **Connection Issues**: Automatic retry and graceful degradation

## Logging

All operations include detailed logging:

- **Database Operations**: Connection, query, and transaction logging
- **Model Training**: Progress, optimization, and performance metrics
- **Error Conditions**: Detailed exception information and context