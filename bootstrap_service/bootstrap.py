"""Weather Service Bootstrap Module

This module implements the initial setup and data population routines for the weather service
database. It handles the creation of database tables and populates them with historical and
forecast data when the system is first deployed or when database initialization is required.

Bootstrap Operations:
- Creates all required database tables if they don't exist
- Truncates existing tables to ensure clean initialization
- Retrieves comprehensive historical weather data from OpenMeteo Archive API
- Fetches current forecast data from OpenMeteo Forecast API
- Generates weekly aggregations from daily historical data
- Populates all weather data tables with fresh data

The bootstrap service runs conditionally based on the WeatherDatabase.bootstrap property,
which determines if the database requires initialization. This prevents accidental data
overwrites during routine operations while ensuring proper setup for new deployments.

Data Tables Populated:
- DailyWeatherHistory: Historical daily weather observations
- DailyWeatherForecast: Current daily weather forecasts
- WeeklyWeatherHistory: Aggregated weekly historical data
- WeeklyWeatherForecast: Weekly forecast data (table created but not populated)

The service uses OpenMeteo API clients to fetch weather data and automatically handles
the aggregation of daily historical data into weekly summaries using the
WeeklyTableConstructor. All tables are truncated before new data insertion to ensure
data consistency.

Dependencies:
- OpenMeteo API clients (Archive and Forecast) for weather data retrieval
- WeatherDatabase for data persistence and table management
- WeeklyTableConstructor for daily-to-weekly data aggregation
- Weather model classes for ORM object creation

Usage:
    This module is designed to run as a standalone script during system deployment
    or when database reinitialization is required. The bootstrap logic automatically
    detects whether initialization is needed.

Example:
    python bootstrap.py

Note:
    The bootstrap process may take considerable time depending on the configured
    geographic area and historical data range due to API rate limiting and the
    volume of data being processed.
"""

import logging

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
        logger.exception("An error occurred during the bootstrap routine: ")
    finally:
        database.close()
