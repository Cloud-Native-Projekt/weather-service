"""Weather Service Bootstrap Module

This module implements the initial setup and data population routines for the weather service
database. It handles the creation of database tables and populates them with historical and
forecast data when the system is first deployed or when tables are empty.

Bootstrap Operations:
- Creates all required database tables if they don't exist
- Truncates existing tables to ensure clean initialization
- Retrieves comprehensive historical weather data from OpenMeteo Archive API
- Fetches current forecast data from OpenMeteo Forecast API
- Generates weekly aggregations from daily data for both historical and forecast periods

The bootstrap service only runs when the database requires initialization, determined by
checking if tables exist and contain data. This prevents accidental data loss during
routine operations.

Data Tables Populated:
- DailyWeatherHistory: Historical daily weather observations
- DailyWeatherForecast: Current daily weather forecasts
- WeeklyWeatherHistory: Aggregated weekly historical data
- WeeklyWeatherForecast: Aggregated weekly forecast data

The service uses the OpenMeteo API clients to fetch weather data across
the configured geographic region and time periods. It automatically handles the
aggregation of daily data into weekly summaries using the WeeklyTableConstructor.

Dependencies:
- OpenMeteo API clients for historical and forecast data retrieval
- WeatherDatabase for data persistence and table management
- WeeklyTableConstructor for daily-to-weekly data aggregation
- Pandas for data manipulation and merging operations

Usage:
    This module is typically run once during initial system deployment or when
    database reinitialization is required. The service automatically detects
    whether bootstrapping is needed.

Example:
    python bootstrap.py

Note:
    The bootstrap process may take considerable time depending on the configured
    geographic area and historical data range due to API rate limiting and the
    volume of data being processed.
"""

import logging

import pandas as pd

from openmeteo_client import (
    OpenMeteoArchiveClient,
    OpenMeteoClientConfig,
    OpenMeteoForecastClient,
    WeeklyTableConstructor,
)
from weather_models import (
    DailyWeatherForecast,
    DailyWeatherHistory,
    WeatherDatabase,
    WeeklyWeatherForecast,
    WeeklyWeatherHistory,
)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(name="Bootstrap Service")

    database = WeatherDatabase()

    try:
        if database.bootstrap:
            logger.info("Starting bootstrap routine...")

            database.create_tables()
            database.truncate_table(DailyWeatherHistory)
            database.truncate_table(DailyWeatherForecast)
            database.truncate_table(WeeklyWeatherHistory)
            database.truncate_table(WeeklyWeatherForecast)

            config = OpenMeteoClientConfig(create_from_file=True)

            ArchiveClient = OpenMeteoArchiveClient(config)
            historic_data_daily = ArchiveClient.main()
            history_orm_objects_daily = database.create_orm_objects(
                data=historic_data_daily, table=DailyWeatherHistory
            )

            ForecastClient = OpenMeteoForecastClient(config)
            forecast_data_daily = ForecastClient.main()
            forecast_orm_objects_daily = database.create_orm_objects(
                data=forecast_data_daily, table=DailyWeatherForecast
            )

            historic_data_weekly, _, historic_data_daily_tail = (
                WeeklyTableConstructor().main(historic_data_daily)
            )
            history_orm_objects_weekly = database.create_orm_objects(
                data=historic_data_weekly, table=WeeklyWeatherHistory
            )

            database.write_data(history_orm_objects_daily)
            database.write_data(forecast_orm_objects_daily)
            database.write_data(history_orm_objects_weekly)

            logger.info("Bootstrap routine completed successfully!")

        else:
            logger.info("Tables already exist. Skipping bootstrap routine...")
    except Exception as e:
        logger.error(e)
    finally:
        database.close()
